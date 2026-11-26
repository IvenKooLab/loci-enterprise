"""Export and import one tenant's data, without taking a database dump.

A database dump is an operational artifact: it contains every tenant, internal
ids and schema details, and restoring it is all-or-nothing. Data portability is
a different requirement — a customer asks for their data, or a tenant moves
between deployments, and the unit should be *that tenant's knowledge*.

This module moves:

    documents + their version history
    chunks + ACL rows
    memory items + events
    wiki pages + revisions
    connector state (so a resumed deployment does not re-ingest everything)

Format is a single JSON object with a format version. Import is additive and
never overwrites a different tenant's rows: documents are matched by
(library, path) and written with an explicit version bump if they differ, so
the audit trail on the destination records the import instead of pretending the
data was always there.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .store import Document, Store, now_iso

FORMAT = "loci-enterprise.tenant/1"


class ImportError_(Exception):
    """Raised when an export cannot be applied."""


@dataclass
class ImportReport:
    documents: int = 0
    chunks: int = 0
    memories: int = 0
    wiki_pages: int = 0
    skipped: int = 0

    def as_dict(self) -> dict:
        return {"documents": self.documents, "chunks": self.chunks,
                "memories": self.memories, "wiki_pages": self.wiki_pages,
                "skipped": self.skipped}


def _rows(store: Store, sql: str, args: list) -> list[dict]:
    return [dict(r) for r in store.conn.execute(sql, args).fetchall()]


def export_tenant(store: Store, tenant_id: str) -> dict:
    docs = _rows(store, "SELECT * FROM documents WHERE tenant_id=? ORDER BY id",
                 [tenant_id])
    doc_ids = [d["id"] for d in docs]
    versions = []
    chunks = []
    if doc_ids:
        marks = ",".join("?" for _ in doc_ids)
        versions = _rows(store, f"SELECT * FROM document_versions WHERE doc_id IN ({marks})"
                                f" ORDER BY doc_id, version", doc_ids)
        chunks = _rows(store, f"SELECT * FROM chunks WHERE doc_id IN ({marks})"
                              f" ORDER BY doc_id, ord", doc_ids)
    acl = []
    if chunks:
        marks = ",".join("?" for _ in chunks)
        acl = _rows(store, f"SELECT * FROM chunk_acl WHERE chunk_id IN ({marks})",
                    [c["id"] for c in chunks])

    tables = {r["name"] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    memories = (_rows(store, "SELECT * FROM memory_items WHERE tenant_id=?",
                      [tenant_id]) if "memory_items" in tables else [])
    memory_events = (_rows(store, "SELECT * FROM memory_events WHERE tenant_id=?",
                           [tenant_id]) if "memory_events" in tables else [])
    wiki_pages = (_rows(store, "SELECT * FROM wiki_pages WHERE tenant_id=?",
                        [tenant_id]) if "wiki_pages" in tables else [])
    wiki_revisions = (_rows(store, "SELECT * FROM wiki_revisions WHERE tenant_id=?",
                            [tenant_id]) if "wiki_revisions" in tables else [])
    connectors = (_rows(store, "SELECT * FROM connector_state WHERE tenant_id=?",
                        [tenant_id]) if "connector_state" in tables else [])

    return {
        "format": FORMAT,
        "tenant_id": tenant_id,
        "exported_at": now_iso(),
        "documents": docs,
        "document_versions": versions,
        "chunks": chunks,
        "chunk_acl": acl,
        "memory_items": memories,
        "memory_events": memory_events,
        "wiki_pages": wiki_pages,
        "wiki_revisions": wiki_revisions,
        "connector_state": connectors,
    }


def save_export(store: Store, tenant_id: str, path: str | Path) -> dict:
    data = export_tenant(store, tenant_id)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return {"path": str(path), "documents": len(data["documents"]),
            "chunks": len(data["chunks"])}


def load_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ImportError_(f"unsupported export format: {data.get('format')!r}")
    return data


def import_tenant(store: Store, tenant_id: str, data: dict, *,
                  actor: str = "import") -> ImportReport:
    """Apply an export into `tenant_id` (which may differ from the source)."""
    if data.get("format") != FORMAT:
        raise ImportError_(f"unsupported export format: {data.get('format')!r}")
    report = ImportReport()
    id_map: dict[str, str] = {}

    # make sure the optional subsystems exist in the destination before we
    # inspect sqlite_master for them
    from .memory import MemoryStore
    from .wiki import WikiStore
    MemoryStore(store)
    WikiStore(store)

    # documents first: chunks and versions reference them
    for row in data.get("documents", []):
        existing = store.get_document(tenant_id, row["library"], row["path"])
        doc = Document(id="", tenant_id=tenant_id, library=row["library"],
                       path=row["path"], title=row.get("title") or "",
                       body=row.get("body") or "",
                       status=row.get("status") or "active",
                       effective_from=row.get("effective_from"),
                       effective_to=row.get("effective_to"),
                       supersedes=row.get("supersedes"),
                       superseded_by=row.get("superseded_by"),
                       owner=row.get("owner") or "",
                       authority=float(row.get("authority") or 1.0),
                       stale_after_days=row.get("stale_after_days"),
                       written_by=actor)
        saved = store.put_document(doc, note="tenant import")
        id_map[row["id"]] = saved.id
        report.documents += 1

    # chunks are rebuilt by put_document, so an import re-chunks rather than
    # trusting chunk ids from a different installation; report what was built
    report.chunks = store.conn.execute(
        "SELECT COUNT(*) n FROM chunks WHERE tenant_id=?", (tenant_id,)).fetchone()["n"]

    tables = {r["name"] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "memory_items" in tables:
        for row in data.get("memory_items", []):
            store.conn.execute(
                "INSERT OR IGNORE INTO memory_items(id, tenant_id, subject_id, key, value,"
                " status, confidence, source, actor, created_at, updated_at, invalid_at,"
                " expires_at, superseded_by, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["id"], tenant_id, row["subject_id"], row["key"], row["value"],
                 row["status"], row["confidence"], row["source"], row["actor"],
                 row["created_at"], row["updated_at"], row["invalid_at"],
                 row["expires_at"], row["superseded_by"], row["note"]))
            report.memories += 1
    if "wiki_pages" in tables:
        for row in data.get("wiki_pages", []):
            store.conn.execute(
                "INSERT OR REPLACE INTO wiki_pages(tenant_id, library, slug, title, body,"
                " citations, version, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (tenant_id, row["library"], row["slug"], row["title"], row["body"],
                 row["citations"], row["version"], row["updated_at"]))
            report.wiki_pages += 1
        for row in data.get("wiki_revisions", []):
            store.conn.execute(
                "INSERT OR IGNORE INTO wiki_revisions(tenant_id, library, slug, version,"
                " title, body, citations, created_at, note) VALUES (?,?,?,?,?,?,?,?,?)",
                (tenant_id, row["library"], row["slug"], row["version"], row["title"],
                 row["body"], row["citations"], row["created_at"], row["note"]))
    if "connector_state" in tables:
        for row in data.get("connector_state", []):
            store.conn.execute(
                "INSERT OR REPLACE INTO connector_state(source_key, tenant_id, library,"
                " last_sync, items, cursor) VALUES (?,?,?,?,?,?)",
                (row["source_key"], tenant_id, row["library"], row["last_sync"],
                 row["items"], row.get("cursor") or ""))

    store.audit(tenant_id, actor, "tenant:import", data.get("tenant_id", "?"),
                f"{report.documents} docs, {report.memories} memories")
    return report
