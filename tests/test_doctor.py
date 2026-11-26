"""Doctor findings: each check fires on the failure it describes."""
from __future__ import annotations

import time

from loci_enterprise import doctor
from loci_enterprise.memory import MemoryStore
from loci_enterprise.queue import Queue
from loci_enterprise.store import Document, Store


def make_store():
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="t1", library="lib", path="a.md",
                            title="A", body="# A\n\n内容", acl=["public"],
                            written_by="t"))
    return s


def codes(report):
    return report.exit_code()


def test_clean_store_has_no_warnings():
    report = doctor.run(make_store(), tenant_id="t1")
    assert codes(report) == 0
    assert not report.errors and not report.warnings


def test_integrity_is_reported_as_info():
    report = doctor.run(make_store())
    assert any(f.check == "db_integrity" and f.severity == doctor.INFO
               for f in report.findings)


def test_missing_schema_is_an_error():
    s = Store(":memory:")
    s.conn.execute("DROP TABLE chunk_acl")
    report = doctor.run(s)
    assert any(f.check == "schema" and f.severity == doctor.ERROR
               for f in report.findings)


def test_orphan_chunks_are_an_error():
    s = make_store()
    # disable FK enforcement so the chunks survive the document delete: this is
    # exactly the corruption an interrupted migration can leave behind
    s.conn.execute("PRAGMA foreign_keys=OFF")
    s.conn.execute("DELETE FROM documents")
    report = doctor.run(s)
    assert any(f.check == "orphan_chunks" for f in report.errors)


def test_document_without_acl_is_an_error():
    s = make_store()
    s.conn.execute("DELETE FROM chunk_acl")
    report = doctor.run(s)
    assert any(f.check == "missing_acl" for f in report.errors)


def test_stuck_job_is_a_warning():
    s = make_store()
    q = Queue(s)
    job = q.enqueue("t1", "x")
    q.claim()
    s.conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?",
                   (time.time() - 3600, job.id))
    report = doctor.run(s, tenant_id="t1")
    assert any(f.check == "stuck_jobs" for f in report.warnings)


def test_dead_job_is_a_warning():
    s = make_store()
    q = Queue(s)
    job = q.enqueue("t1", "x", max_attempts=1)
    q.claim()
    q.fail(job.id, "boom", retry=False)
    report = doctor.run(s, tenant_id="t1")
    assert any(f.check == "dead_jobs" for f in report.warnings)


def test_unswept_expired_memory_is_a_warning():
    s = make_store()
    mem = MemoryStore(s)
    res = mem.remember("t1", "u1", "k", "v", ttl_days=1)
    s.conn.execute("UPDATE memory_items SET expires_at=? WHERE id=?",
                   (time.time() - 10, res.item.id))
    report = doctor.run(s, tenant_id="t1")
    assert any(f.check == "unswept_expiry" for f in report.warnings)


def test_stale_connector_is_a_warning():
    s = make_store()
    s.conn.executescript(
        "CREATE TABLE IF NOT EXISTS connector_state (source_key TEXT PRIMARY KEY,"
        " tenant_id TEXT, library TEXT, last_sync TEXT, items INTEGER, cursor TEXT)")
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30 * 86400))
    s.conn.execute("INSERT INTO connector_state VALUES ('local:/x','t1','lib',?,1,'')",
                   (old,))
    report = doctor.run(s, tenant_id="t1", stale_connector_days=7)
    assert any(f.check == "stale_connector" for f in report.warnings)


def test_exit_codes_escalate():
    clean = doctor.Report()
    assert clean.exit_code() == 0
    warn = doctor.Report(); warn.add("x", doctor.WARN, "w")
    assert warn.exit_code() == 1
    err = doctor.Report(); err.add("x", doctor.ERROR, "e")
    assert err.exit_code() == 2


def test_render_is_readable():
    report = doctor.run(make_store(), tenant_id="t1")
    text = doctor.render(report)
    assert "documents:" in text and "error(s)" in text
