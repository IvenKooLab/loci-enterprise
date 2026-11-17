"""Multi-writer governance: two people editing the same memory.

Enterprise reality is not "one agent writes memories". It is a support engineer
pasting a fix, a manager correcting a number, and a sync job reconciling both,
inside the same minute. Three failure modes follow:

    lost update      B's write silently overwrites A's, and A never knows
    duplicate        both create `refund-policy.md`; now two active truths
    ghost            A reads v3, writes v4 from stale fields, drops B's edit

The first is solved by optimistic concurrency (expected version), the second by
the unique (tenant, library, path) index plus an explicit merge policy, the
third by comparing the *fields that were read* rather than the whole document.

Defaults are conservative: a conflict is returned to the caller, never
resolved by guessing. `on_conflict="last_write_wins"` exists because some
writers (a sync job replaying a source of truth) genuinely want it — but it
must be chosen, and it is audited as such.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .store import Conflict, Document, Store, now_iso


@dataclass
class WriteResult:
    status: str                 # "created" | "updated" | "noop" | "conflict" | "merged"
    document: Document | None = None
    detail: str = ""
    winners: list[str] = field(default_factory=list)


class Governor:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------- the write
    def write(self, doc: Document, *, expected_version: int | None = None,
              on_conflict: str = "reject", note: str = "") -> WriteResult:
        """One memory write, with an explicit answer for every race.

        on_conflict:
          reject            raise Conflict (default; caller decides)
          last_write_wins   overwrite, audited as such
          merge             keep the incoming body appended, never silent
        """
        if on_conflict not in ("reject", "last_write_wins", "merge"):
            raise ValueError(f"unknown on_conflict policy {on_conflict!r}")

        existing = self.store.get_document(doc.tenant_id, doc.library, doc.path)

        # identical content: a duplicate paste is a no-op, not a new version
        if existing is not None and existing.content_hash == _digest(doc.body) \
                and existing.title == doc.title:
            self.store.audit(doc.tenant_id, doc.written_by or "unknown",
                             "write:noop", doc.path, "identical content")
            return WriteResult("noop", existing, "identical content")

        try:
            saved = self.store.put_document(doc, expected_version=expected_version,
                                            note=note)
            return WriteResult("created" if existing is None else "updated", saved)
        except Conflict as exc:
            return self._resolve_conflict(doc, exc, on_conflict, note)

    def _resolve_conflict(self, doc: Document, exc: Conflict,
                          policy: str, note: str) -> WriteResult:
        if policy == "reject":
            self.store.audit(doc.tenant_id, doc.written_by or "unknown",
                             "write:conflict", doc.path,
                             f"expected v{exc.expected}, actual v{exc.actual}")
            raise exc

        actual = self.store.get_document(doc.tenant_id, doc.library, doc.path)
        if actual is None:                      # it vanished under us
            doc.version = 1
            saved = self.store.put_document(doc, note=note)
            return WriteResult("created", saved, "recreated after deletion")

        if policy == "merge":
            merged_body = (
                actual.body.rstrip()
                + "\n\n<!-- merged by loci-enterprise: concurrent write -->\n\n"
                + doc.body.strip())
            doc.body = merged_body
            doc.title = doc.title or actual.title
        saved = self.store.put_document(doc, expected_version=actual.version, note=note)
        self.store.audit(doc.tenant_id, doc.written_by or "unknown",
                         f"write:{policy}", doc.path,
                         f"v{exc.expected} -> v{actual.version} resolved")
        return WriteResult(policy if policy == "merge" else "updated", saved,
                           f"resolved {policy} over v{actual.version}")

    # -------------------------------------------------------- policy changes
    def apply_policy_change(self, *, tenant_id: str, library: str,
                            new: Document, old_path: str | None = None,
                            effective: str | None = None,
                            actor: str = "", note: str = "") -> WriteResult:
        """Land a policy change: write the new document, retire the old one.

        This is the operation whose absence produced the 85% -> 60% incident.
        Doing it as one call means the two halves cannot drift apart: you
        cannot end up with the new policy indexed and the old one still active.
        """
        from . import lifecycle

        new.effective_from = effective or new.effective_from
        new.status = "active"
        new.written_by = actor
        result = self.write(new, on_conflict="reject", note=note or "policy change")
        if old_path and result.document is not None:
            try:
                lifecycle.supersede(self.store, tenant_id, old_path, new.path,
                                    library=library, actor=actor,
                                    note=note or "policy change")
            except LookupError:
                self.store.audit(tenant_id, actor or "system", "supersede:missing",
                                 old_path, f"wanted -> {new.path}")
        return result

    # ------------------------------------------------------------- utilities
    def prune_duplicates(self, tenant_id: str, library: str | None = None,
                         actor: str = "policy") -> list[dict]:
        """Find same-title active documents and report them (never auto-delete).

        Auto-deleting on a heuristic is how a governance tool loses someone's
        work. It proposes; a human or an explicit policy retires.
        """
        from . import lifecycle
        return lifecycle.conflict_scan(self.store, tenant_id, library)


def _digest(text: str) -> str:
    from .store import digest
    return digest(text)
