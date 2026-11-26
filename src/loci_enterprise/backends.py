"""Retrieval backends behind one contract.

The point of this module is not to have two implementations. It is to make the
*seam* explicit, so swapping SQLite for pgvector, Elasticsearch or a hosted
service is a bounded piece of work instead of a rewrite.

The contract has four obligations, and the first one is the one that matters:

1. **Filtering happens inside the backend.** A backend receives the principal
   set and the tenant and returns only chunks that principal may read. There is
   deliberately no "search everything, filter later" method — that shape leaks
   through result counts, scores and timing even when the text is withheld.
2. Candidates are scored with the same hybrid function everywhere, so a
   migration changes latency, not ranking semantics.
3. ``upsert`` is idempotent per chunk id and carries the ACL with the content.
4. ``stats`` reports what is stored, per tenant, without exposing content.

Two implementations ship here: an in-memory one (tests, small instances) and a
SQLite one (single file, no server). The contract test suite runs against both.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[a-z0-9_]+", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    out = [m.group(0).lower() for m in _WORD.finditer(text or "")]
    for run in _CJK.findall(text or ""):
        out.append(run)
    # CJK bigrams over each contiguous run
    for m in re.finditer(r"[\u4e00-\u9fff]+", text or ""):
        run = m.group(0)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


@dataclass
class IndexRecord:
    chunk_id: str
    doc_id: str
    tenant_id: str
    library: str
    text: str
    vector: list[float] | None = None
    acl: list[str] = field(default_factory=lambda: ["public"])
    status: str = "active"
    authority: float = 1.0


@dataclass
class Query:
    text: str
    tenant_id: str
    principals: list[str]
    vector: list[float] | None = None
    library: str | None = None
    top_k: int = 10
    allowed_docs: set[str] | None = None
    rrf_k: int = 60


@dataclass
class Scored:
    chunk_id: str
    doc_id: str
    score: float
    lexical: float = 0.0
    vector: float = 0.0


# ------------------------------------------------------------------ scoring
def _bm25(docs: list[str], query: str) -> list[float]:
    if not docs:
        return []
    toks = [tokenize(d) for d in docs]
    n = len(docs)
    avg = sum(len(t) for t in toks) / n or 1.0
    q = tokenize(query)
    df: dict[str, int] = {}
    for t in toks:
        for term in set(t):
            df[term] = df.get(term, 0) + 1
    out = []
    for t in toks:
        tf: dict[str, int] = {}
        for x in t:
            tf[x] = tf.get(x, 0) + 1
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * 2.5 / (tf[term] + 1.5 * (1 - 0.75 + 0.75 * len(t) / avg))
        out.append(score)
    return out


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    return [0.0] * len(values) if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in values]


def rank(records: list[IndexRecord], query: Query) -> list[Scored]:
    """The one ranking function every backend must delegate to."""
    if not records:
        return []
    lex = _bm25([r.text for r in records], query.text)
    lexn = _norm(lex)
    if query.vector:
        vec = _norm([_cos(query.vector, r.vector) if r.vector else 0.0 for r in records])
    else:
        vec = [0.0] * len(records)
    lex_rank = {i: r for r, i in enumerate(sorted(range(len(lex)), key=lambda i: -lex[i]))}
    vec_rank = {i: r for r, i in enumerate(sorted(range(len(vec)), key=lambda i: -vec[i]))}
    scored: list[Scored] = []
    for i, rec in enumerate(records):
        s = 0.0
        if lex[i] > 0:
            s += 1.0 / (query.rrf_k + lex_rank[i] + 1)
        if query.vector and vec[i] > 0:
            s += 1.0 / (query.rrf_k + vec_rank[i] + 1)
        s *= max(0.1, min(3.0, rec.authority))
        if s > 0:
            scored.append(Scored(rec.chunk_id, rec.doc_id, s, lexn[i], vec[i]))
    scored.sort(key=lambda x: -x.score)
    return scored[: query.top_k]


def _visible(rec: IndexRecord, q: Query) -> bool:
    if rec.tenant_id != q.tenant_id:
        return False
    if q.library and rec.library != q.library:
        return False
    if q.allowed_docs is not None and rec.doc_id not in q.allowed_docs:
        return False
    if rec.status not in ("active", "stale"):
        return False
    return bool(set(rec.acl) & set(q.principals))


# ---------------------------------------------------------------- backends
class MemoryBackend:
    name = "memory"

    def __init__(self):
        self._records: dict[str, IndexRecord] = {}

    def upsert(self, records: list[IndexRecord]) -> int:
        for r in records:
            self._records[r.chunk_id] = r
        return len(records)

    def delete_doc(self, tenant_id: str, doc_id: str) -> int:
        ids = [k for k, r in self._records.items()
               if r.tenant_id == tenant_id and r.doc_id == doc_id]
        for k in ids:
            del self._records[k]
        return len(ids)

    def candidates(self, q: Query) -> list[IndexRecord]:
        return [r for r in self._records.values() if _visible(r, q)]

    def search(self, q: Query) -> list[Scored]:
        return rank(self.candidates(q), q)

    def stats(self, tenant_id: str) -> dict:
        rows = [r for r in self._records.values() if r.tenant_id == tenant_id]
        return {"backend": self.name, "chunks": len(rows),
                "documents": len({r.doc_id for r in rows})}


SCHEMA = """
CREATE TABLE IF NOT EXISTS backend_chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    tenant_id  TEXT NOT NULL,
    library    TEXT NOT NULL,
    text       TEXT NOT NULL,
    vector     BLOB,
    acl        TEXT NOT NULL DEFAULT '["public"]',
    status     TEXT NOT NULL DEFAULT 'active',
    authority  REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS backend_scope ON backend_chunks(tenant_id, library, status);
CREATE INDEX IF NOT EXISTS backend_doc ON backend_chunks(tenant_id, doc_id);
"""


class SQLiteBackend:
    name = "sqlite"

    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @staticmethod
    def _pack(vec: list[float] | None) -> bytes | None:
        return struct.pack(f"<{len(vec)}f", *vec) if vec else None

    @staticmethod
    def _unpack(blob: bytes | None) -> list[float] | None:
        if not blob:
            return None
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))

    def upsert(self, records: list[IndexRecord]) -> int:
        for r in records:
            self.conn.execute(
                "INSERT INTO backend_chunks(chunk_id, doc_id, tenant_id, library, text,"
                " vector, acl, status, authority) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(chunk_id) DO UPDATE SET doc_id=excluded.doc_id,"
                " library=excluded.library, text=excluded.text, vector=excluded.vector,"
                " acl=excluded.acl, status=excluded.status, authority=excluded.authority",
                (r.chunk_id, r.doc_id, r.tenant_id, r.library, r.text,
                 self._pack(r.vector), json.dumps(r.acl, ensure_ascii=False),
                 r.status, r.authority))
        return len(records)

    def delete_doc(self, tenant_id: str, doc_id: str) -> int:
        cur = self.conn.execute(
            "DELETE FROM backend_chunks WHERE tenant_id=? AND doc_id=?",
            (tenant_id, doc_id))
        return cur.rowcount

    def candidates(self, q: Query) -> list[IndexRecord]:
        sql = ("SELECT * FROM backend_chunks WHERE tenant_id=? AND status IN ('active','stale')")
        args: list = [q.tenant_id]
        if q.library:
            sql += " AND library=?"
            args.append(q.library)
        if q.allowed_docs is not None:
            if not q.allowed_docs:
                return []
            marks = ",".join("?" for _ in q.allowed_docs)
            sql += f" AND doc_id IN ({marks})"
            args.extend(sorted(q.allowed_docs))
        principals = set(q.principals)
        out = []
        for row in self.conn.execute(sql, args).fetchall():
            acl = set(json.loads(row["acl"]))
            if principals & acl:                 # ACL filter before scoring
                out.append(IndexRecord(
                    chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                    tenant_id=row["tenant_id"], library=row["library"],
                    text=row["text"], vector=self._unpack(row["vector"]),
                    acl=sorted(acl), status=row["status"], authority=row["authority"]))
        return out

    def search(self, q: Query) -> list[Scored]:
        return rank(self.candidates(q), q)

    def stats(self, tenant_id: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) chunks, COUNT(DISTINCT doc_id) docs"
            " FROM backend_chunks WHERE tenant_id=?", (tenant_id,)).fetchone()
        return {"backend": self.name, "chunks": row["chunks"], "documents": row["docs"]}
