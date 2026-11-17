"""Lifecycle: the part that makes an answer stop being right.

The 85% -> 60% incident this module exists for has a signature: retrieval
did not get worse, the *world* changed. The index still holds last quarter's
policy, the retriever faithfully finds it, the model faithfully quotes it, and
the answer is now wrong. Adding documents makes it worse, because a second
equally-similar document with different numbers creates an ambiguity that
similarity cannot resolve.

So the corpus needs three things a vector index cannot express on its own:

    validity    effective_from / effective_to — when a document is true
    lineage     supersedes / superseded_by     — what replaced what
    freshness   stale_after_days               — when to distrust silence

This module computes lifecycle state, walks supersede chains, and proposes
retirements. It never rewrites content; it only decides what may be *trusted*
and what must be *excluded*.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from .store import ACTIVE, Document, Store, now_iso

# How much a document's status multiplies its retrieval score.
STATUS_WEIGHT = {
    "active": 1.0,
    "stale": 0.55,        # still true, but nobody has confirmed it lately
    "superseded": 0.05,   # ~excluded: useful only as a last resort / for history
    "expired": 0.0,       # excluded outright
    "retired": 0.0,
}

EXCLUDED = {"expired", "retired"}


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[: len(fmt) + 6 if "T" in fmt else 10],
                                     fmt).date()
        except ValueError:
            continue
    return None


def today() -> date:
    return datetime.now(timezone.utc).date()


@dataclass
class Verdict:
    """What lifecycle has to say about one document, for one query, now."""
    doc_id: str
    status: str
    weight: float
    reason: str
    superseded_by: str | None = None
    age_days: int | None = None


def evaluate(doc: Document, *, on: date | None = None) -> Verdict:
    """Resolve the stored status against the calendar.

    Stored status is what a human decided; the calendar can still disqualify a
    document that nobody remembered to retire. Effective dates win, because
    they are the machine-checkable half of "the policy changed on the 1st".
    """
    ref = on or today()
    start = _parse_day(doc.effective_from)
    end = _parse_day(doc.effective_to)
    age = None
    try:
        age = (ref - datetime.strptime(doc.updated_at[:10], "%Y-%m-%d").date()).days
    except (ValueError, TypeError):
        pass

    if doc.status in EXCLUDED:
        return Verdict(doc.id, doc.status, 0.0, f"excluded ({doc.status})",
                       doc.superseded_by, age)
    if start and ref < start:
        return Verdict(doc.id, "expired", 0.0,
                       f"not in force until {doc.effective_from}", doc.superseded_by, age)
    if end and ref > end:
        return Verdict(doc.id, "expired", 0.0,
                       f"expired on {doc.effective_to}", doc.superseded_by, age)
    if doc.superseded_by:
        return Verdict(doc.id, "superseded", STATUS_WEIGHT["superseded"],
                       f"superseded by {doc.superseded_by}", doc.superseded_by, age)

    limit = doc.stale_after_days
    if limit and age is not None and age > limit:
        return Verdict(doc.id, "stale", STATUS_WEIGHT["stale"],
                       f"unreviewed for {age}d (limit {limit}d)", None, age)
    if doc.status == "stale":
        return Verdict(doc.id, "stale", STATUS_WEIGHT["stale"], "marked stale", None, age)
    return Verdict(doc.id, doc.status or ACTIVE, STATUS_WEIGHT["active"],
                   "ok", None, age)


def evaluate_all(store: Store, tenant_id: str, library: str | None = None,
                 *, on: date | None = None) -> dict[str, Verdict]:
    return {d.id: evaluate(d, on=on)
            for d in store.list_documents(tenant_id, library)}


# ------------------------------------------------------------------ lineage
def supersede(store: Store, tenant_id: str, old_path: str, new_path: str,
              *, library: str, actor: str, note: str = "") -> tuple[Document, Document]:
    """Point old at new and new at old, in one place, atomically.

    Both directions are written because readers ask both questions: "is this
    still true?" (forward) and "what replaced it?" (backward). Writing only
    one is how a chain quietly breaks in the middle.
    """
    old = store.get_document(tenant_id, library, old_path)
    new = store.get_document(tenant_id, library, new_path)
    if old is None or new is None:
        raise LookupError(f"missing document: {old_path if old is None else new_path}")

    store.conn.execute(
        "UPDATE documents SET superseded_by=?, status='superseded', updated_at=? WHERE id=?",
        (new.path, now_iso(), old.id))
    store.conn.execute(
        "UPDATE documents SET supersedes=?, status='active', updated_at=? WHERE id=?",
        (old.path, now_iso(), new.id))
    store.audit(tenant_id, actor, "supersede", old.path, f"-> {new.path} {note}".strip())
    old = store.get_document(tenant_id, library, old_path)
    new = store.get_document(tenant_id, library, new_path)
    assert old is not None and new is not None
    return old, new


def chain(store: Store, tenant_id: str, library: str, path: str) -> list[dict]:
    """Oldest -> newest along supersede links, cycles included but bounded."""
    doc = store.get_document(tenant_id, library, path)
    if doc is None:
        return []
    # walk backwards to the root
    back, seen = [], set()
    cur = doc
    while cur.supersedes and cur.supersedes not in seen and len(seen) < 64:
        seen.add(cur.path)
        prev = store.get_document(tenant_id, library, cur.supersedes)
        if prev is None:
            break
        back.append(prev)
        cur = prev
    order = list(reversed(back)) + [doc]
    # then forwards
    seen = {d.path for d in order}
    cur = doc
    while cur.superseded_by and cur.superseded_by not in seen and len(seen) < 64:
        seen.add(cur.path)
        nxt = store.get_document(tenant_id, library, cur.superseded_by)
        if nxt is None:
            break
        order.append(nxt)
        cur = nxt
    out = []
    for d in order:
        v = evaluate(d)
        out.append({"path": d.path, "status": v.status, "weight": v.weight,
                    "effective_from": d.effective_from, "reason": v.reason})
    return out


def current(store: Store, tenant_id: str, library: str, path: str) -> dict | None:
    """The document that is in force for this lineage, right now."""
    items = chain(store, tenant_id, library, path)
    for item in reversed(items):
        if item["status"] == ACTIVE:
            return item
    for item in reversed(items):
        if item["status"] == "stale":
            return item
    return items[-1] if items else None


# --------------------------------------------------------------- retirement
@dataclass
class RetirementCandidate:
    path: str
    library: str
    reason: str
    severity: str          # "high" | "medium" | "low"
    suggested_action: str
    detail: str = ""


def retirement_candidates(store: Store, tenant_id: str, *, library: str | None = None,
                          min_age_days: int = 180,
                          zero_citation: set[str] | None = None,
                          corrected: set[str] | None = None,
                          on: date | None = None) -> list[RetirementCandidate]:
    """Propose removals. Never removes anything itself.

    Four signals, in descending confidence:
      superseded    — already resolved by lineage; just needs to leave the index
      expired       — outside its validity window
      corrected     — feedback says it produced wrong answers
      never-cited   — old and never used; dead weight, not necessarily wrong
    """
    out: list[RetirementCandidate] = []
    zero_citation = zero_citation or set()
    corrected = corrected or set()
    for doc in store.list_documents(tenant_id, library):
        v = evaluate(doc, on=on)
        if doc.status == "superseded":
            out.append(RetirementCandidate(
                doc.path, doc.library, "superseded by a newer document", "high",
                "move out of the active index; keep for history",
                v.reason))
            continue
        if v.status == "expired":
            out.append(RetirementCandidate(
                doc.path, doc.library, "outside its validity window", "high",
                "archive or extend the effective window", v.reason))
            continue
        if doc.path in corrected:
            out.append(RetirementCandidate(
                doc.path, doc.library, "cited in answers marked wrong", "high",
                "review and correct; it is actively misleading", ""))
            continue
        age = v.age_days or 0
        if doc.path in zero_citation and age > min_age_days:
            out.append(RetirementCandidate(
                doc.path, doc.library, "never cited and unreviewed", "low",
                "confirm still true, or retire", f"age {age}d"))
    return out


def mark_stale(store: Store, tenant_id: str, *, older_than_days: int,
               actor: str = "policy", library: str | None = None,
               dry_run: bool = False) -> list[str]:
    """Flag documents nobody has touched, so trust decays visibly."""
    flagged = []
    ref = today()
    for doc in store.list_documents(tenant_id, library):
        if doc.status != ACTIVE:
            continue
        try:
            age = (ref - datetime.strptime(doc.updated_at[:10], "%Y-%m-%d").date()).days
        except (ValueError, TypeError):
            continue
        if age > older_than_days:
            flagged.append(doc.path)
            if not dry_run:
                store.conn.execute(
                    "UPDATE documents SET status='stale', updated_at=? WHERE id=?",
                    (now_iso(), doc.id))
                store.audit(tenant_id, actor, "status:stale", doc.path, f"age {age}d")
    return flagged


def conflict_scan(store: Store, tenant_id: str, library: str | None = None) -> list[dict]:
    """Documents that claim the same facts but disagree.

    A cheap proxy for "two equally-similar documents with different numbers":
    same title, active, different body. This is the shape of the incident —
    and it is exactly what a similarity score refuses to rank.
    """
    by_title: dict[str, list[Document]] = {}
    for doc in store.list_documents(tenant_id, library):
        if evaluate(doc).status not in ("active", "stale"):
            continue
        key = (doc.title or doc.path).strip().lower()
        by_title.setdefault(key, []).append(doc)
    found = []
    for key, docs in by_title.items():
        if len(docs) > 1:
            found.append({
                "title": key,
                "documents": [d.path for d in docs],
                "reason": "same title, different bodies, both active",
                "hint": "declare supersedes/effective, or retire one",
            })
    return found
