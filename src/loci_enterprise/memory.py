"""Memory items that know what replaced them and when they stop being true.

A memory store that only appends is a rumour mill. The failure mode is
specific and common: on Monday an agent is told the budget is 500, on Friday
it is told 800, and both statements now sit in the index with equal standing.
Next week's answer depends on which one the retriever likes more.

This module makes the lifecycle explicit and the default conservative:

    pending ──approve──► active ──contradicted──► superseded
                            │
                            ├──expire──► expired
                            └──archive─► archived

* A write to the same ``(subject, key)`` with a *different* value supersedes
  the old item rather than deleting it — the history is the point.
* An identical write is a no-op, so a chatty agent cannot inflate the store.
* Machine-inferred memories start ``pending`` and are invisible to recall
  until a human (or a policy) approves them.
* ``expires_at`` is a fact about the statement, not about the row's age, so a
  "current sprint" note can decay on schedule while a policy stays put.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from .store import Store, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    subject_id   TEXT NOT NULL,
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    confidence   REAL NOT NULL DEFAULT 1.0,
    source       TEXT NOT NULL DEFAULT 'agent',
    actor        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    invalid_at   REAL,
    expires_at   REAL,
    superseded_by TEXT,
    note         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS memory_subject ON memory_items(tenant_id, subject_id, status);
CREATE INDEX IF NOT EXISTS memory_key ON memory_items(tenant_id, subject_id, key, status);

CREATE TABLE IF NOT EXISTS memory_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    action    TEXT NOT NULL,
    detail    TEXT NOT NULL DEFAULT ''
);
"""

ACTIVE = "active"
PENDING = "pending"
SUPERSEDED = "superseded"
ARCHIVED = "archived"
EXPIRED = "expired"
STATUSES = (ACTIVE, PENDING, SUPERSEDED, ARCHIVED, EXPIRED)


@dataclass
class MemoryItem:
    id: str
    tenant_id: str
    subject_id: str
    key: str
    value: str
    status: str = ACTIVE
    confidence: float = 1.0
    source: str = "agent"
    actor: str = ""
    created_at: str = ""
    updated_at: str = ""
    invalid_at: float | None = None
    expires_at: float | None = None
    superseded_by: str | None = None
    note: str = ""

    @property
    def live(self) -> bool:
        if self.status != ACTIVE:
            return False
        return not (self.expires_at and self.expires_at < time.time())

    def as_dict(self) -> dict:
        return dict(self.__dict__, live=self.live)

    @classmethod
    def from_row(cls, row) -> "MemoryItem":
        return cls(**{k: row[k] for k in row.keys()
                      if k in cls.__dataclass_fields__})


@dataclass
class WriteResult:
    status: str            # created | replaced | noop | pending | rejected
    item: MemoryItem | None = None
    replaced: MemoryItem | None = None
    detail: str = ""


class MemoryStore:
    def __init__(self, store: Store, *, max_items_per_subject: int = 200):
        self.store = store
        self.store.conn.executescript(SCHEMA)
        self.max_items = max_items_per_subject

    # --------------------------------------------------------------- writes
    def remember(self, tenant_id: str, subject_id: str, key: str, value: str,
                 *, source: str = "agent", confidence: float = 1.0,
                 ttl_days: float | None = None, actor: str = "",
                 requires_review: bool = False, note: str = "") -> WriteResult:
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("memory key and value must be non-empty")
        live = self._live_for(tenant_id, subject_id, key)
        if live and live.value.strip() == value:
            self._event(tenant_id, subject_id, live.id, "noop", "identical value")
            return WriteResult("noop", live, detail="identical value")

        if not requires_review and self._capacity_full(tenant_id, subject_id):
            return WriteResult("rejected", None,
                               detail=f"subject at capacity ({self.max_items})")

        item = MemoryItem(
            id=str(uuid.uuid4()), tenant_id=tenant_id, subject_id=subject_id,
            key=key, value=value,
            status=PENDING if requires_review else ACTIVE,
            confidence=max(0.0, min(1.0, confidence)), source=source,
            actor=actor, created_at=now_iso(), updated_at=now_iso(),
            expires_at=(time.time() + ttl_days * 86400) if ttl_days else None,
            note=note)
        self._insert(item)

        replaced = None
        if live is not None and not requires_review:
            replaced = self.supersede(live.id, by=item.id, actor=actor,
                                      reason=f"contradicted by {source}")
        action = "pending" if requires_review else ("replaced" if replaced else "created")
        self._event(tenant_id, subject_id, item.id, action, f"{key}={value}")
        return WriteResult(action, item, replaced,
                           detail=f"superseded {replaced.id}" if replaced else "")

    def _insert(self, item: MemoryItem) -> None:
        self.store.conn.execute(
            "INSERT INTO memory_items(id, tenant_id, subject_id, key, value, status,"
            " confidence, source, actor, created_at, updated_at, invalid_at,"
            " expires_at, superseded_by, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item.id, item.tenant_id, item.subject_id, item.key, item.value,
             item.status, item.confidence, item.source, item.actor,
             item.created_at, item.updated_at, item.invalid_at, item.expires_at,
             item.superseded_by, item.note))

    def _live_for(self, tenant_id: str, subject_id: str, key: str) -> MemoryItem | None:
        row = self.store.conn.execute(
            "SELECT * FROM memory_items WHERE tenant_id=? AND subject_id=? AND key=?"
            " AND status=? ORDER BY created_at DESC LIMIT 1",
            (tenant_id, subject_id, key, ACTIVE)).fetchone()
        return MemoryItem.from_row(row) if row else None

    def _capacity_full(self, tenant_id: str, subject_id: str) -> bool:
        row = self.store.conn.execute(
            "SELECT COUNT(*) n FROM memory_items WHERE tenant_id=? AND subject_id=?"
            " AND status IN (?,?)", (tenant_id, subject_id, ACTIVE, PENDING)).fetchone()
        return row["n"] >= self.max_items

    # ---------------------------------------------------------- transitions
    def supersede(self, memory_id: str, *, by: str, actor: str = "",
                  reason: str = "") -> MemoryItem | None:
        row = self._get(memory_id)
        if row is None or row.status != ACTIVE:
            return None
        now = time.time()
        self.store.conn.execute(
            "UPDATE memory_items SET status=?, invalid_at=?, superseded_by=?,"
            " updated_at=? WHERE id=?",
            (SUPERSEDED, now, by, now_iso(), memory_id))
        self._event(row.tenant_id, row.subject_id, memory_id, "superseded", reason)
        return self._get(memory_id)

    def approve(self, memory_id: str, *, actor: str = "") -> MemoryItem | None:
        row = self._get(memory_id)
        if row is None or row.status != PENDING:
            return None
        live = self._live_for(row.tenant_id, row.subject_id, row.key)
        if live and live.value.strip() == row.value.strip():
            self.reject(memory_id, actor=actor, reason="duplicate of an active memory")
            return None
        self.store.conn.execute(
            "UPDATE memory_items SET status=?, updated_at=? WHERE id=?",
            (ACTIVE, now_iso(), memory_id))
        if live:
            self.supersede(live.id, by=memory_id, actor=actor,
                           reason="approved memory contradicts it")
        self._event(row.tenant_id, row.subject_id, memory_id, "approved", f"by {actor}")
        return self._get(memory_id)

    def reject(self, memory_id: str, *, actor: str = "", reason: str = "") -> bool:
        row = self._get(memory_id)
        if row is None or row.status != PENDING:
            return False
        self.store.conn.execute(
            "UPDATE memory_items SET status=?, updated_at=?, note=? WHERE id=?",
            (ARCHIVED, now_iso(), reason[:300], memory_id))
        self._event(row.tenant_id, row.subject_id, memory_id, "rejected", reason)
        return True

    def archive(self, memory_id: str, *, actor: str = "", reason: str = "") -> bool:
        row = self._get(memory_id)
        if row is None or row.status not in (ACTIVE, PENDING):
            return False
        self.store.conn.execute(
            "UPDATE memory_items SET status=?, updated_at=? WHERE id=?",
            (ARCHIVED, now_iso(), memory_id))
        self._event(row.tenant_id, row.subject_id, memory_id, "archived", reason)
        return True

    def expire_due(self, *, tenant_id: str | None = None) -> int:
        """Mark items whose statement expired. Returns how many changed."""
        sql = "SELECT id, tenant_id, subject_id FROM memory_items WHERE status=? AND expires_at IS NOT NULL AND expires_at < ?"
        args: list = [ACTIVE, time.time()]
        if tenant_id:
            sql += " AND tenant_id=?"
            args.append(tenant_id)
        rows = self.store.conn.execute(sql, args).fetchall()
        for r in rows:
            self.store.conn.execute(
                "UPDATE memory_items SET status=?, updated_at=? WHERE id=?",
                (EXPIRED, now_iso(), r["id"]))
            self._event(r["tenant_id"], r["subject_id"], r["id"], "expired", "")
        return len(rows)

    # ----------------------------------------------------------------- reads
    def _get(self, memory_id: str) -> MemoryItem | None:
        row = self.store.conn.execute(
            "SELECT * FROM memory_items WHERE id=?", (memory_id,)).fetchone()
        return MemoryItem.from_row(row) if row else None

    def recall(self, tenant_id: str, subject_id: str, key: str | None = None, *,
               include_history: bool = False) -> list[MemoryItem]:
        self.expire_due(tenant_id=tenant_id)
        sql = "SELECT * FROM memory_items WHERE tenant_id=? AND subject_id=?"
        args: list = [tenant_id, subject_id]
        if key:
            sql += " AND key=?"
            args.append(key)
        if not include_history:
            sql += " AND status=?"
            args.append(ACTIVE)
        sql += " ORDER BY rowid DESC"
        items = [MemoryItem.from_row(r) for r in
                 self.store.conn.execute(sql, args).fetchall()]
        return [i for i in items if include_history or i.live]

    def pending(self, tenant_id: str) -> list[MemoryItem]:
        return [MemoryItem.from_row(r) for r in self.store.conn.execute(
            "SELECT * FROM memory_items WHERE tenant_id=? AND status=?"
            " ORDER BY created_at", (tenant_id, PENDING)).fetchall()]

    def history(self, tenant_id: str, subject_id: str, key: str) -> list[MemoryItem]:
        # insertion order, not timestamp: two writes in the same second must
        # still be ordered the way they happened
        return [MemoryItem.from_row(r) for r in self.store.conn.execute(
            "SELECT * FROM memory_items WHERE tenant_id=? AND subject_id=? AND key=?"
            " ORDER BY rowid", (tenant_id, subject_id, key)).fetchall()]

    def capacity(self, tenant_id: str, subject_id: str) -> dict:
        rows = self.store.conn.execute(
            "SELECT status, COUNT(*) n FROM memory_items WHERE tenant_id=? AND subject_id=?"
            " GROUP BY status", (tenant_id, subject_id)).fetchall()
        by_status = {r["status"]: r["n"] for r in rows}
        used = by_status.get(ACTIVE, 0) + by_status.get(PENDING, 0)
        return {"used": used, "limit": self.max_items, "by_status": by_status}

    # ---------------------------------------------------------------- forget
    def forget(self, tenant_id: str, subject_id: str, *, key: str | None = None,
               actor: str = "") -> dict:
        """Delete a subject's memories and return a receipt (account-erasure)."""
        cond = "tenant_id=? AND subject_id=?"
        args: list = [tenant_id, subject_id]
        if key:
            cond += " AND key=?"
            args.append(key)
        n = self.store.conn.execute(
            f"SELECT COUNT(*) n FROM memory_items WHERE {cond}", args).fetchone()["n"]
        self.store.conn.execute(f"DELETE FROM memory_items WHERE {cond}", args)
        self._event(tenant_id, subject_id, "*", "forgotten", f"{n} item(s) key={key or '*'}")
        return {"deleted": n, "subject_id": subject_id, "key": key or "*",
                "at": now_iso(), "actor": actor}

    # ---------------------------------------------------------------- events
    def _event(self, tenant_id: str, subject_id: str, memory_id: str,
               action: str, detail: str = "") -> None:
        self.store.conn.execute(
            "INSERT INTO memory_events(ts, tenant_id, subject_id, memory_id, action, detail)"
            " VALUES (?,?,?,?,?,?)",
            (now_iso(), tenant_id, subject_id, memory_id, action, detail[:500]))

    def events(self, tenant_id: str, subject_id: str | None = None,
               limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM memory_events WHERE tenant_id=?"
        args: list = [tenant_id]
        if subject_id:
            sql += " AND subject_id=?"
            args.append(subject_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.store.conn.execute(sql, args).fetchall()]

    def export(self, tenant_id: str, subject_id: str) -> dict:
        """Everything held about one subject, for a data-access request."""
        return {"subject_id": subject_id,
                "items": [i.as_dict() for i in self.recall(
                    tenant_id, subject_id, include_history=True)],
                "events": self.events(tenant_id, subject_id, limit=10_000)}
