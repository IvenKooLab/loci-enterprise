"""Doctor: the checks an operator runs before believing a deployment.

Every check here exists because the corresponding failure is silent:

* a deleted document whose chunks survived still answers searches — orphan
  chunks are invisible in the UI and wrong in the results;
* a document with no ACL rows is visible to nobody, which looks like "the
  connector didn't work";
* jobs stuck in ``running`` after a crash look like a busy system;
* a connector that has not synced for a week looks healthy until someone
  notices the answers are stale.

The report is a list of findings with a severity, and the exit code is the only
thing a cron job needs: 0 clean, 1 warnings, 2 errors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .store import Store

ERROR, WARN, INFO = "error", "warn", "info"


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"check": self.check, "severity": self.severity,
                "message": self.message, "detail": self.detail}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, check, severity, message, **detail):
        self.findings.append(Finding(check, severity, message, detail))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    def exit_code(self) -> int:
        return 2 if self.errors else (1 if self.warnings else 0)

    def as_dict(self) -> dict:
        return {"stats": self.stats, "ok": not self.findings,
                "errors": len(self.errors), "warnings": len(self.warnings),
                "findings": [f.as_dict() for f in self.findings]}


def _tables(store: Store) -> set[str]:
    return {r["name"] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


REQUIRED = {"documents", "document_versions", "chunks", "chunk_acl", "audit"}


def run(store: Store, *, tenant_id: str | None = None,
        stale_connector_days: float = 7.0,
        stuck_job_minutes: float = 30.0) -> Report:
    report = Report()

    # 1. database integrity
    try:
        row = store.conn.execute("PRAGMA integrity_check").fetchone()
        ok = row and (row[0] == "ok" or row["integrity_check"] == "ok")
    except Exception as exc:                    # noqa: BLE001 - reported
        report.add("db_integrity", ERROR, f"integrity check failed: {exc}")
    else:
        if not ok:
            report.add("db_integrity", ERROR, "database integrity check failed",
                       result=str(tuple(row)))
        else:
            report.add("db_integrity", INFO, "database integrity ok")

    # 2. schema
    missing = sorted(REQUIRED - _tables(store))
    if missing:
        report.add("schema", ERROR, f"missing tables: {', '.join(missing)}")
        # dependent checks would raise; report the root cause and stop
        report.stats.update({"documents": None, "chunks": None})
        return report

    tenant_clause = " WHERE tenant_id=?" if tenant_id else ""
    args = [tenant_id] if tenant_id else []

    # 3. counts
    docs = store.conn.execute(f"SELECT COUNT(*) n FROM documents{tenant_clause}",
                              args).fetchone()["n"]
    chunks = store.conn.execute(f"SELECT COUNT(*) n FROM chunks{tenant_clause}",
                                args).fetchone()["n"]
    report.stats.update({"documents": docs, "chunks": chunks})
    if docs and not chunks:
        report.add("empty_index", WARN, "documents exist but no chunks were built")

    # 4. orphan chunks (document disappeared but chunks stayed)
    orphan_q = ("SELECT COUNT(*) n FROM chunks c LEFT JOIN documents d"
                " ON d.id = c.doc_id WHERE d.id IS NULL")
    if tenant_id:
        orphan_q += " AND c.tenant_id=?"
    orphans = store.conn.execute(orphan_q, args).fetchone()["n"]
    if orphans:
        report.add("orphan_chunks", ERROR,
                   f"{orphans} chunk(s) reference a document that no longer exists",
                   count=orphans)

    # 5. documents with no ACL rows -> readable by nobody
    acl_q = ("SELECT COUNT(*) n FROM documents d WHERE NOT EXISTS"
             " (SELECT 1 FROM chunks c JOIN chunk_acl a ON a.chunk_id=c.id"
             "  WHERE c.doc_id=d.id)")
    if tenant_id:
        acl_q += " AND d.tenant_id=?"
    no_acl = store.conn.execute(acl_q, args).fetchone()["n"]
    if no_acl:
        report.add("missing_acl", ERROR,
                   f"{no_acl} document(s) have no ACL rows and are visible to nobody",
                   count=no_acl)

    tables = _tables(store)

    # 6. jobs stuck in running
    cutoff = time.time() - stuck_job_minutes * 60
    stuck_q = ("SELECT COUNT(*) n FROM jobs WHERE status='running'"
               " AND COALESCE(heartbeat_at, run_at) < ?")
    stuck_args = [cutoff]
    if tenant_id:
        stuck_q += " AND tenant_id=?"
        stuck_args.append(tenant_id)
    stuck = (store.conn.execute(stuck_q, stuck_args).fetchone()["n"]
             if "jobs" in tables else 0)
    if stuck:
        report.add("stuck_jobs", WARN,
                   f"{stuck} job(s) running with no heartbeat for over "
                   f"{stuck_job_minutes:.0f} min", count=stuck)

    # 7. dead jobs
    dead_q = "SELECT COUNT(*) n FROM jobs WHERE status='dead'"
    if tenant_id:
        dead_q += " AND tenant_id=?"
    dead = (store.conn.execute(dead_q, args).fetchone()["n"]
            if "jobs" in tables else 0)
    if dead:
        report.add("dead_jobs", WARN, f"{dead} job(s) exhausted their retries",
                   count=dead)

    # 8. expired memories that were never swept
    exp_q = ("SELECT COUNT(*) n FROM memory_items WHERE status='active'"
             " AND expires_at IS NOT NULL AND expires_at < ?")
    exp_args = [time.time()]
    if tenant_id:
        exp_q += " AND tenant_id=?"
        exp_args.append(tenant_id)
    expired = (store.conn.execute(exp_q, exp_args).fetchone()["n"]
               if "memory_items" in tables else 0)
    if expired:
        report.add("unswept_expiry", WARN,
                   f"{expired} memory item(s) are past expiry but still active",
                   count=expired)

    # 9. connectors that stopped syncing
    if "connector_state" not in tables:
        return report
    stale_q = "SELECT source_key, last_sync FROM connector_state WHERE 1=1"
    stale_args: list = []
    if tenant_id:
        stale_q += " AND tenant_id=?"
        stale_args.append(tenant_id)
    for row in store.conn.execute(stale_q, stale_args).fetchall():
        try:
            ts = time.mktime(time.strptime(row["last_sync"], "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, TypeError):
            continue
        age_days = (time.time() - ts) / 86400
        if age_days > stale_connector_days:
            report.add("stale_connector", WARN,
                       f"{row['source_key']} has not synced for {age_days:.1f} days",
                       source=row["source_key"], age_days=round(age_days, 1))

    return report


def render(report: Report) -> str:
    lines = [f"documents: {report.stats.get('documents', 0)}   "
             f"chunks: {report.stats.get('chunks', 0)}"]
    if not report.findings:
        lines.append("no findings")
        return "\n".join(lines)
    for f in report.findings:
        mark = {"error": "✗", "warn": "!", "info": "·"}.get(f.severity, "·")
        lines.append(f"{mark} [{f.check}] {f.message}")
    lines.append(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return "\n".join(lines)
