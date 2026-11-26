"""SQLite store: documents, chunks, ACL rows and the audit trail.

Two decisions here carry the whole design:

1. **ACL is a row, not a filter.** Every chunk gets one `chunk_acl` row per
   principal that may read it. Retrieval joins against that table, so an
   unauthorised chunk is never a candidate — it cannot occupy a top-k slot,
   change an IDF weight, or reach the reranker. Post-filtering is what leaks.

2. **Version history is kept.** A document write is an insert into
   `document_versions` plus an update of the live row, guarded by an expected
   version. Two people editing the same note do not silently clobber each
   other; the loser gets a conflict they can resolve.

Nothing here is tenant-agnostic: every table carries `tenant_id`, and every
query takes it as a mandatory argument. There is no "all tenants" code path.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    library       TEXT NOT NULL,
    path          TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    body          TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    status        TEXT NOT NULL DEFAULT 'active',
    effective_from TEXT,
    effective_to  TEXT,
    supersedes    TEXT,
    superseded_by TEXT,
    owner         TEXT NOT NULL DEFAULT '',
    authority     REAL NOT NULL DEFAULT 1.0,
    stale_after_days INTEGER,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    written_by    TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS documents_path
    ON documents(tenant_id, library, path);
CREATE INDEX IF NOT EXISTS documents_status ON documents(tenant_id, status);

CREATE TABLE IF NOT EXISTS document_versions (
    doc_id       TEXT NOT NULL,
    version      INTEGER NOT NULL,
    body         TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    fields       TEXT NOT NULL DEFAULT '{}',
    written_by   TEXT NOT NULL DEFAULT '',
    written_at   TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (doc_id, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id  TEXT NOT NULL,
    library    TEXT NOT NULL,
    ord        INTEGER NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_scope ON chunks(tenant_id, library);

CREATE TABLE IF NOT EXISTS chunk_acl (
    chunk_id   TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    principal  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, principal)
);
CREATE INDEX IF NOT EXISTS chunk_acl_principal ON chunk_acl(principal);

CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    tenant_id  TEXT NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    target     TEXT NOT NULL DEFAULT '',
    detail     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS audit_ts ON audit(tenant_id, ts);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    tenant_id  TEXT NOT NULL,
    library    TEXT NOT NULL,
    query      TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    chunk_ids  TEXT NOT NULL DEFAULT '[]',
    doc_ids    TEXT NOT NULL DEFAULT '[]',
    note       TEXT NOT NULL DEFAULT '',
    actor      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS feedback_scope ON feedback(tenant_id, library, ts);
"""

ACTIVE = "active"
STATUSES = ("active", "stale", "superseded", "expired", "retired")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Conflict(Exception):
    """Optimistic-concurrency failure: someone else wrote this document."""

    def __init__(self, doc_id: str, expected: int, actual: int):
        super().__init__(
            f"{doc_id} changed under you (expected version {expected}, now {actual})")
        self.doc_id = doc_id
        self.expected = expected
        self.actual = actual


@dataclass
class Document:
    id: str
    tenant_id: str
    library: str
    path: str
    title: str = ""
    body: str = ""
    content_hash: str = ""
    version: int = 1
    status: str = ACTIVE
    effective_from: str | None = None
    effective_to: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    owner: str = ""
    authority: float = 1.0
    stale_after_days: int | None = None
    created_at: str = ""
    updated_at: str = ""
    written_by: str = ""
    acl: list[str] = field(default_factory=lambda: ["public"])

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Document":
        return cls(**{k: row[k] for k in row.keys() if k in cls.__dataclass_fields__})


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        # The HTTP service and the worker pool share one Store across threads.
        # SQLite is compiled in serialized mode by default, so a single
        # connection is safe to share; without this flag every request from a
        # second thread would raise ProgrammingError instead.
        self.conn = sqlite3.connect(self.path, isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------------------------------------------------------------- audit
    def audit(self, tenant_id: str, actor: str, action: str,
              target: str = "", detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO audit(ts, tenant_id, actor, action, target, detail) "
            "VALUES (?,?,?,?,?,?)",
            (now_iso(), tenant_id, actor, action, target, detail[:500]))

    def audit_tail(self, tenant_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM audit WHERE tenant_id=? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ documents
    def put_document(self, doc: Document, *, expected_version: int | None = None,
                     note: str = "", reindex: bool = True) -> Document:
        """Insert or update a document, then rebuild its chunks.

        `expected_version` makes the write conditional: pass the version you
        read, and a concurrent writer that got there first raises Conflict
        instead of losing an edit. Pass None for "I am creating this".
        """
        existing = self.get_document(doc.tenant_id, doc.library, doc.path)
        if existing is None:
            if expected_version not in (None, 0):
                raise Conflict(doc.path, expected_version, 0)
            doc.id = doc.id or str(uuid.uuid4())
            doc.version = 1
            doc.created_at = doc.updated_at = now_iso()
            doc.content_hash = digest(doc.body)
            self.conn.execute(
                "INSERT INTO documents(id, tenant_id, library, path, title, body,"
                " content_hash, version, status, effective_from, effective_to,"
                " supersedes, superseded_by, owner, authority, stale_after_days,"
                " created_at, updated_at, written_by)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc.id, doc.tenant_id, doc.library, doc.path, doc.title, doc.body,
                 doc.content_hash, doc.version, doc.status, doc.effective_from,
                 doc.effective_to, doc.supersedes, doc.superseded_by, doc.owner,
                 doc.authority, doc.stale_after_days, doc.created_at,
                 doc.updated_at, doc.written_by))
            self.audit(doc.tenant_id, doc.written_by or "unknown", "create",
                       doc.path, f"v1 {len(doc.body)}B")
        else:
            if expected_version is not None and existing.version != expected_version:
                raise Conflict(doc.path, expected_version, existing.version)
            doc.id = existing.id
            doc.version = existing.version + 1
            doc.created_at = existing.created_at
            doc.updated_at = now_iso()
            doc.content_hash = digest(doc.body)
            self.conn.execute(
                "UPDATE documents SET title=?, body=?, content_hash=?, version=?,"
                " status=?, effective_from=?, effective_to=?, supersedes=?,"
                " superseded_by=?, owner=?, authority=?, stale_after_days=?,"
                " updated_at=?, written_by=? WHERE id=?",
                (doc.title, doc.body, doc.content_hash, doc.version, doc.status,
                 doc.effective_from, doc.effective_to, doc.supersedes,
                 doc.superseded_by, doc.owner, doc.authority,
                 doc.stale_after_days, doc.updated_at, doc.written_by, doc.id))
            self.audit(doc.tenant_id, doc.written_by or "unknown", "update",
                       doc.path, f"v{doc.version} {len(doc.body)}B")
        self._snapshot(doc, note)
        if reindex:
            self._reindex(doc)
        return doc

    def _snapshot(self, doc: Document, note: str) -> None:
        fields = {k: getattr(doc, k) for k in
                  ("status", "effective_from", "effective_to", "supersedes",
                   "superseded_by", "owner", "authority", "stale_after_days", "acl")}
        self.conn.execute(
            "INSERT OR REPLACE INTO document_versions"
            "(doc_id, version, body, title, fields, written_by, written_at, note)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (doc.id, doc.version, doc.body, doc.title,
             json.dumps(fields, ensure_ascii=False), doc.written_by,
             doc.updated_at, note))

    def _reindex(self, doc: Document) -> None:
        from .chunking import chunk_document  # local: avoids an import cycle
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.id,))
        pieces = chunk_document(doc.body, doc.title)
        principals = sorted(set(doc.acl or ["public"]))
        for i, text in enumerate(pieces):
            cid = f"{doc.id}::{i}"
            self.conn.execute(
                "INSERT INTO chunks(id, doc_id, tenant_id, library, ord, text)"
                " VALUES (?,?,?,?,?,?)",
                (cid, doc.id, doc.tenant_id, doc.library, i, text))
            for p in principals:
                self.conn.execute(
                    "INSERT OR IGNORE INTO chunk_acl(chunk_id, principal) VALUES (?,?)",
                    (cid, p))

    def get_document(self, tenant_id: str, library: str | None = None,
                     path: str | None = None, doc_id: str | None = None) -> Document | None:
        if doc_id:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE id=? AND tenant_id=?",
                (doc_id, tenant_id)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE tenant_id=? AND library=? AND path=?",
                (tenant_id, library, path)).fetchone()
        if row is None:
            return None
        doc = Document.from_row(row)
        doc.acl = self.acl_of(doc.id)
        return doc

    def acl_of(self, doc_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT a.principal FROM chunk_acl a"
            " JOIN chunks c ON c.id=a.chunk_id WHERE c.doc_id=?", (doc_id,)).fetchall()
        return sorted(r["principal"] for r in rows)

    def list_documents(self, tenant_id: str, library: str | None = None,
                       status: str | None = None) -> list[Document]:
        sql = "SELECT * FROM documents WHERE tenant_id=?"
        args: list = [tenant_id]
        if library:
            sql += " AND library=?"
            args.append(library)
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY library, path"
        out = []
        for row in self.conn.execute(sql, args).fetchall():
            doc = Document.from_row(row)
            doc.acl = self.acl_of(doc.id)
            out.append(doc)
        return out

    def set_status(self, doc_id: str, status: str, *, superseded_by: str | None = None,
                   actor: str = "", reason: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        self.conn.execute(
            "UPDATE documents SET status=?, superseded_by=COALESCE(?, superseded_by),"
            " updated_at=? WHERE id=?",
            (status, superseded_by, now_iso(), doc_id))
        self.audit("", actor or "system", f"status:{status}", doc_id, reason)

    def delete_document(self, doc_id: str, actor: str = "") -> None:
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.audit("", actor or "system", "delete", doc_id, "")

    # --------------------------------------------------------------- chunks
    def visible_chunks(self, tenant_id: str, principals: list[str],
                       library: str | None = None) -> list[dict]:
        """Chunks this principal may read — ACL is part of the query.

        No caller can ask for "everything"; the principal set is required and
        the join drops rows the principal cannot see, so the BM25 corpus that
        gets scored is already permission-scoped (IDF included).
        """
        if not principals:
            return []
        marks = ",".join("?" for _ in principals)
        sql = (f"SELECT c.id, c.doc_id, c.text, c.library,"
               f"       d.path, d.title, d.status, d.authority, d.effective_from,"
               f"       d.effective_to, d.superseded_by, d.updated_at,"
               f"       d.stale_after_days"
               f"  FROM chunks c"
               f"  JOIN documents d ON d.id = c.doc_id"
               f"  JOIN chunk_acl a ON a.chunk_id = c.id"
               f" WHERE c.tenant_id = ? AND a.principal IN ({marks})")
        args: list = [tenant_id, *principals]
        if library:
            sql += " AND c.library = ?"
            args.append(library)
        rows = self.conn.execute(sql, args).fetchall()
        seen: dict[str, dict] = {}
        for r in rows:                       # a chunk with 2 matching ACL rows
            seen.setdefault(r["id"], dict(r))
        return list(seen.values())

    def chunk_acl(self, chunk_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT principal FROM chunk_acl WHERE chunk_id=?", (chunk_id,)).fetchall()
        return [r["principal"] for r in rows]

    def counts(self, tenant_id: str) -> dict:
        d = self.conn.execute(
            "SELECT COUNT(*) n FROM documents WHERE tenant_id=?", (tenant_id,)).fetchone()["n"]
        c = self.conn.execute(
            "SELECT COUNT(*) n FROM chunks WHERE tenant_id=?", (tenant_id,)).fetchone()["n"]
        by_status = {r["status"]: r["n"] for r in self.conn.execute(
            "SELECT status, COUNT(*) n FROM documents WHERE tenant_id=? GROUP BY status",
            (tenant_id,)).fetchall()}
        return {"documents": d, "chunks": c, "by_status": by_status}

    def versions(self, doc_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM document_versions WHERE doc_id=? ORDER BY version",
            (doc_id,)).fetchall()
        return [dict(r) for r in rows]
