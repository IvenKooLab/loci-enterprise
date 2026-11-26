"""A job queue that survives the process that started it.

Ingestion, wiki rebuilds and connector syncs are long enough to outlive a
request and important enough that losing one is not acceptable. The queue is
therefore a table, not a list:

    queued -> running -> succeeded
                     -> failed -> queued (retry, with backoff)
                                -> dead   (attempts exhausted)
    any    -> cancelled

Two properties matter more than throughput:

* **A crashed worker must not swallow a job.** Rows carry ``heartbeat_at``;
  ``reclaim()`` puts anything that has been "running" past the timeout back in
  the queue, so a killed process delays work instead of losing it.
* **Retries are bounded and visible.** Every attempt increments ``attempts``
  and records the error; after ``max_attempts`` the job goes ``dead`` and stays
  in the table for inspection instead of disappearing into a log line.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .store import Store, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'queued',
    priority    INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    run_at      REAL NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    heartbeat_at REAL,
    finished_at TEXT,
    error       TEXT NOT NULL DEFAULT '',
    result      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status, run_at, priority);
CREATE INDEX IF NOT EXISTS jobs_tenant ON jobs(tenant_id, status);
"""

QUEUED, RUNNING, SUCCEEDED, FAILED, DEAD, CANCELLED = (
    "queued", "running", "succeeded", "failed", "dead", "cancelled")
TERMINAL = (SUCCEEDED, DEAD, CANCELLED)


@dataclass
class Job:
    id: str
    tenant_id: str
    kind: str
    payload: dict = field(default_factory=dict)
    status: str = QUEUED
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    run_at: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    heartbeat_at: float | None = None
    finished_at: str | None = None
    error: str = ""
    result: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_row(cls, row) -> "Job":
        return cls(id=row["id"], tenant_id=row["tenant_id"], kind=row["kind"],
                   payload=json.loads(row["payload"] or "{}"),
                   status=row["status"], priority=row["priority"],
                   attempts=row["attempts"], max_attempts=row["max_attempts"],
                   run_at=row["run_at"], created_at=row["created_at"],
                   updated_at=row["updated_at"], heartbeat_at=row["heartbeat_at"],
                   finished_at=row["finished_at"], error=row["error"],
                   result=row["result"])


class Queue:
    def __init__(self, store: Store):
        self.store = store
        self.store.conn.executescript(SCHEMA)

    # -------------------------------------------------------------- produce
    def enqueue(self, tenant_id: str, kind: str, payload: dict | None = None, *,
                priority: int = 0, delay: float = 0.0,
                max_attempts: int = 3) -> Job:
        job = Job(id=str(uuid.uuid4()), tenant_id=tenant_id, kind=kind,
                  payload=payload or {}, priority=priority,
                  max_attempts=max(1, max_attempts), run_at=time.time() + delay,
                  created_at=now_iso(), updated_at=now_iso())
        self.store.conn.execute(
            "INSERT INTO jobs(id, tenant_id, kind, payload, status, priority, attempts,"
            " max_attempts, run_at, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (job.id, job.tenant_id, job.kind, json.dumps(job.payload, ensure_ascii=False),
             job.status, job.priority, 0, job.max_attempts, job.run_at,
             job.created_at, job.updated_at))
        return job

    # --------------------------------------------------------------- consume
    def claim(self, *, tenant_id: str | None = None) -> Job | None:
        """Atomically take the highest-priority ready job."""
        now = time.time()
        while True:
            where = "status=? AND run_at<=?"
            args: list = [QUEUED, now]
            if tenant_id:
                where += " AND tenant_id=?"
                args.append(tenant_id)
            row = self.store.conn.execute(
                f"SELECT id FROM jobs WHERE {where}"
                f" ORDER BY priority DESC, run_at ASC LIMIT 1", args).fetchone()
            if row is None:
                return None
            # conditional update is the actual claim: if another worker won the
            # race, rowcount is 0 and we look again
            cur = self.store.conn.execute(
                "UPDATE jobs SET status=?, attempts=attempts+1, heartbeat_at=?,"
                " updated_at=? WHERE id=? AND status=?",
                (RUNNING, now, now_iso(), row["id"], QUEUED))
            if cur.rowcount:
                return self.get(row["id"])
            if time.time() - now > 5:        # pathological contention: give up
                return None

    def get(self, job_id: str) -> Job | None:
        row = self.store.conn.execute("SELECT * FROM jobs WHERE id=?",
                                      (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def heartbeat(self, job_id: str) -> bool:
        cur = self.store.conn.execute(
            "UPDATE jobs SET heartbeat_at=?, updated_at=? WHERE id=? AND status=?",
            (time.time(), now_iso(), job_id, RUNNING))
        return bool(cur.rowcount)

    def complete(self, job_id: str, result: str = "") -> bool:
        cur = self.store.conn.execute(
            "UPDATE jobs SET status=?, result=?, finished_at=?, updated_at=?"
            " WHERE id=? AND status=?",
            (SUCCEEDED, result[:2000], now_iso(), now_iso(), job_id, RUNNING))
        return bool(cur.rowcount)

    def fail(self, job_id: str, error: str, *, backoff: float = 2.0,
             retry: bool = True) -> Job | None:
        """Record a failed attempt; requeue with exponential backoff or go dead."""
        job = self.get(job_id)
        if job is None:
            return None
        terminal = (not retry) or job.attempts >= job.max_attempts
        if terminal:
            self.store.conn.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=?, updated_at=?"
                " WHERE id=?",
                (DEAD, error[:2000], now_iso(), now_iso(), job_id))
        else:
            delay = backoff ** job.attempts
            self.store.conn.execute(
                "UPDATE jobs SET status=?, error=?, run_at=?, heartbeat_at=NULL,"
                " updated_at=? WHERE id=?",
                (QUEUED, error[:2000], time.time() + delay, now_iso(), job_id))
        return self.get(job_id)

    def cancel(self, tenant_id: str, job_id: str) -> bool:
        cur = self.store.conn.execute(
            "UPDATE jobs SET status=?, finished_at=?, updated_at=?"
            " WHERE id=? AND tenant_id=? AND status IN (?,?)",
            (CANCELLED, now_iso(), now_iso(), job_id, tenant_id, QUEUED, RUNNING))
        return bool(cur.rowcount)

    def reclaim(self, *, older_than_seconds: float = 300) -> int:
        """Return jobs whose worker stopped heart-beating to the queue."""
        cutoff = time.time() - older_than_seconds
        rows = self.store.conn.execute(
            "SELECT id, attempts, max_attempts FROM jobs WHERE status=? AND"
            " COALESCE(heartbeat_at, run_at) < ?", (RUNNING, cutoff)).fetchall()
        for r in rows:
            if r["attempts"] >= r["max_attempts"]:
                self.store.conn.execute(
                    "UPDATE jobs SET status=?, error=?, finished_at=?, updated_at=?"
                    " WHERE id=?", (DEAD, "worker heartbeat lost",
                                    now_iso(), now_iso(), r["id"]))
            else:
                self.store.conn.execute(
                    "UPDATE jobs SET status=?, heartbeat_at=NULL, updated_at=?"
                    " WHERE id=?", (QUEUED, now_iso(), r["id"]))
        return len(rows)

    # --------------------------------------------------------------- worker
    def run_once(self, handlers: dict[str, Callable[[Job], str]], *,
                 tenant_id: str | None = None) -> Job | None:
        """Claim and execute one job. Unknown kind fails without consuming retries."""
        job = self.claim(tenant_id=tenant_id)
        if job is None:
            return None
        handler = handlers.get(job.kind)
        if handler is None:
            self.fail(job.id, f"no handler for kind {job.kind!r}", retry=False)
            return self.get(job.id)
        try:
            result = handler(job)
        except Exception as exc:                     # noqa: BLE001 - recorded, not raised
            self.fail(job.id, f"{type(exc).__name__}: {exc}")
        else:
            self.complete(job.id, str(result or ""))
        return self.get(job.id)

    def drain(self, handlers: dict[str, Callable[[Job], str]], *,
              tenant_id: str | None = None, limit: int = 100) -> list[Job]:
        out = []
        for _ in range(limit):
            job = self.run_once(handlers, tenant_id=tenant_id)
            if job is None:
                break
            out.append(job)
        return out

    # ------------------------------------------------------------ reporting
    def stats(self, tenant_id: str | None = None) -> dict:
        where, args = "", []
        if tenant_id:
            where, args = " WHERE tenant_id=?", [tenant_id]
        rows = self.store.conn.execute(
            f"SELECT status, COUNT(*) n FROM jobs{where} GROUP BY status", args).fetchall()
        by_status = {r["status"]: r["n"] for r in rows}
        return {"by_status": by_status,
                "total": sum(by_status.values()),
                "pending": by_status.get(QUEUED, 0),
                "running": by_status.get(RUNNING, 0),
                "dead": by_status.get(DEAD, 0)}

    def list(self, tenant_id: str, *, status: str | None = None,
             limit: int = 100) -> list[Job]:
        sql = "SELECT * FROM jobs WHERE tenant_id=?"
        args: list = [tenant_id]
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        args.append(limit)
        return [Job.from_row(r) for r in self.store.conn.execute(sql, args).fetchall()]
