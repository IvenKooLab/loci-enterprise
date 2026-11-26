"""Schedules: the clock that turns one-off jobs into operations.

A queue executes what it is given. Something has to decide that the wiki should
be rebuilt nightly, that a connector should sync every ten minutes, and that
the expiry sweep should run hourly. That is this module — deliberately small:

* a schedule stores *what to enqueue* (kind + payload) and *how often*;
* ``tick()`` enqueues only what is due and advances ``next_run`` from *now*,
  so a process that was down for a day does not fire 144 catch-up jobs;
* a disabled or removed schedule enqueues nothing, and removal does not
  resurrect on the next tick;
* ``last_run`` and ``next_run`` are visible, because "did the nightly build
  actually run" should never require reading logs.

Schedules and jobs are separate tables. Deleting a schedule does not cancel
jobs it already created — that is intentional: work already promised to the
queue should finish or be cancelled explicitly.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from .queue import Queue
from .store import Store, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    interval_s  REAL NOT NULL,
    next_run    REAL NOT NULL,
    last_run    REAL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS schedules_due ON schedules(enabled, next_run);
CREATE UNIQUE INDEX IF NOT EXISTS schedules_name ON schedules(tenant_id, name);
"""

MIN_INTERVAL = 30.0          # refuse sub-30s schedules: they are a busy loop in disguise


@dataclass
class Schedule:
    id: str
    tenant_id: str
    name: str
    kind: str
    payload: dict = field(default_factory=dict)
    interval_s: float = 3600.0
    next_run: float = 0.0
    last_run: float | None = None
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name,
                "kind": self.kind, "payload": self.payload,
                "interval_s": self.interval_s, "next_run": self.next_run,
                "last_run": self.last_run, "enabled": self.enabled}

    @classmethod
    def from_row(cls, row) -> "Schedule":
        return cls(id=row["id"], tenant_id=row["tenant_id"], name=row["name"],
                   kind=row["kind"], payload=json.loads(row["payload"] or "{}"),
                   interval_s=row["interval_s"], next_run=row["next_run"],
                   last_run=row["last_run"], enabled=bool(row["enabled"]),
                   created_at=row["created_at"], updated_at=row["updated_at"])


class Schedules:
    def __init__(self, store: Store):
        self.store = store
        self.store.conn.executescript(SCHEMA)

    # --------------------------------------------------------------- manage
    def add(self, tenant_id: str, name: str, kind: str, *,
            interval_seconds: float, payload: dict | None = None,
            first_run_in: float | None = None) -> Schedule:
        if interval_seconds < MIN_INTERVAL:
            raise ValueError(f"interval must be >= {MIN_INTERVAL:.0f}s")
        if not name.strip():
            raise ValueError("schedule name must be non-empty")
        existing = self.get(tenant_id, name)
        if existing:
            raise ValueError(f"schedule {name!r} already exists; remove it first")
        start = time.time() + (first_run_in if first_run_in is not None
                               else interval_seconds)
        sched = Schedule(id=str(uuid.uuid4()), tenant_id=tenant_id, name=name,
                         kind=kind, payload=payload or {},
                         interval_s=float(interval_seconds), next_run=start,
                         created_at=now_iso(), updated_at=now_iso())
        self.store.conn.execute(
            "INSERT INTO schedules(id, tenant_id, name, kind, payload, interval_s,"
            " next_run, enabled, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sched.id, tenant_id, name, kind,
             json.dumps(sched.payload, ensure_ascii=False), sched.interval_s,
             sched.next_run, 1, sched.created_at, sched.updated_at))
        self.store.audit(tenant_id, "scheduler", "schedule:add", name,
                         f"{kind} every {interval_seconds:.0f}s")
        return sched

    def get(self, tenant_id: str, name: str) -> Schedule | None:
        row = self.store.conn.execute(
            "SELECT * FROM schedules WHERE tenant_id=? AND name=?",
            (tenant_id, name)).fetchone()
        return Schedule.from_row(row) if row else None

    def list(self, tenant_id: str) -> list[Schedule]:
        return [Schedule.from_row(r) for r in self.store.conn.execute(
            "SELECT * FROM schedules WHERE tenant_id=? ORDER BY name",
            (tenant_id,)).fetchall()]

    def remove(self, tenant_id: str, name: str) -> bool:
        cur = self.store.conn.execute(
            "DELETE FROM schedules WHERE tenant_id=? AND name=?", (tenant_id, name))
        if cur.rowcount:
            self.store.audit(tenant_id, "scheduler", "schedule:remove", name, "")
        return bool(cur.rowcount)

    def set_enabled(self, tenant_id: str, name: str, enabled: bool) -> bool:
        cur = self.store.conn.execute(
            "UPDATE schedules SET enabled=?, updated_at=? WHERE tenant_id=? AND name=?",
            (1 if enabled else 0, now_iso(), tenant_id, name))
        return bool(cur.rowcount)

    # ---------------------------------------------------------------- clock
    def due(self, now: float | None = None) -> list[Schedule]:
        now = now if now is not None else time.time()
        return [Schedule.from_row(r) for r in self.store.conn.execute(
            "SELECT * FROM schedules WHERE enabled=1 AND next_run<=? ORDER BY next_run",
            (now,)).fetchall()]

    def tick(self, queue: Queue, *, now: float | None = None,
             limit: int = 100) -> list[dict]:
        """Enqueue every due schedule once and advance its clock."""
        now = now if now is not None else time.time()
        fired: list[dict] = []
        for sched in self.due(now)[:limit]:
            job = queue.enqueue(sched.tenant_id, sched.kind, sched.payload,
                                priority=-1)          # background work yields to requests
            # advance from now, not from the old value: a long outage must not
            # produce a burst of catch-up runs
            next_run = now + sched.interval_s
            self.store.conn.execute(
                "UPDATE schedules SET last_run=?, next_run=?, updated_at=? WHERE id=?",
                (now, next_run, now_iso(), sched.id))
            self.store.audit(sched.tenant_id, "scheduler", "schedule:fire",
                             sched.name, f"job {job.id}")
            fired.append({"schedule": sched.name, "job_id": job.id,
                          "kind": sched.kind, "next_run": next_run})
        return fired
