"""Job queue: ordering, retries, crash recovery, cancellation."""
from __future__ import annotations

import time

from loci_enterprise.queue import (CANCELLED, DEAD, QUEUED, RUNNING, SUCCEEDED,
                                   Queue)
from loci_enterprise.store import Store


def q() -> Queue:
    return Queue(Store(":memory:"))


def test_enqueue_and_claim_sets_running_and_counts_attempt():
    queue = q()
    job = queue.enqueue("t1", "ingest", {"path": "a"})
    claimed = queue.claim()
    assert claimed.id == job.id and claimed.status == RUNNING and claimed.attempts == 1


def test_complete_records_result():
    queue = q()
    job = queue.enqueue("t1", "ingest")
    queue.claim()
    assert queue.complete(job.id, "3 chunks")
    done = queue.get(job.id)
    assert done.status == SUCCEEDED and done.result == "3 chunks" and done.finished_at


def test_priority_then_fifo_ordering():
    queue = q()
    low = queue.enqueue("t1", "x", priority=0)
    high = queue.enqueue("t1", "x", priority=9)
    later = queue.enqueue("t1", "x", priority=0)
    assert queue.claim().id == high.id
    assert queue.claim().id == low.id
    assert queue.claim().id == later.id


def test_delayed_job_is_not_ready_yet():
    queue = q()
    queue.enqueue("t1", "x", delay=60)
    assert queue.claim() is None


def test_failure_requeues_with_backoff():
    queue = q()
    job = queue.enqueue("t1", "x", max_attempts=3)
    queue.claim()
    after = queue.fail(job.id, "boom", backoff=0.01)
    assert after.status == QUEUED and after.attempts == 1 and "boom" in after.error
    assert after.run_at > time.time()


def test_retries_stop_at_max_attempts_and_mark_dead():
    queue = q()
    job = queue.enqueue("t1", "x", max_attempts=2)
    queue.claim()
    queue.fail(job.id, "first", backoff=0)
    queue.claim()
    final = queue.fail(job.id, "second", backoff=0)
    assert final.status == DEAD and final.attempts == 2


def test_non_retryable_failure_goes_straight_to_dead():
    queue = q()
    job = queue.enqueue("t1", "x", max_attempts=5)
    queue.claim()
    assert queue.fail(job.id, "fatal", retry=False).status == DEAD


def test_cancel_only_affects_own_tenant_and_pending_jobs():
    queue = q()
    mine = queue.enqueue("t1", "x")
    other = queue.enqueue("t2", "x")
    assert not queue.cancel("t2", mine.id)          # not yours
    assert queue.cancel("t1", mine.id)
    assert queue.get(mine.id).status == CANCELLED
    assert queue.get(other.id).status == QUEUED


def test_reclaim_returns_a_silent_worker_job_to_the_queue():
    queue = q()
    job = queue.enqueue("t1", "x")
    queue.claim()
    # simulate a worker that died three minutes ago
    queue.store.conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?",
                             (time.time() - 600, job.id))
    assert queue.reclaim(older_than_seconds=300) == 1
    assert queue.get(job.id).status == QUEUED


def test_reclaim_buries_a_job_that_used_its_attempts():
    queue = q()
    job = queue.enqueue("t1", "x", max_attempts=1)
    queue.claim()
    queue.store.conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?",
                             (time.time() - 600, job.id))
    queue.reclaim(older_than_seconds=300)
    assert queue.get(job.id).status == DEAD


def test_run_once_executes_the_handler():
    queue = q()
    job = queue.enqueue("t1", "sum", {"a": 1})
    result = queue.run_once({"sum": lambda j: f"ok {j.payload['a']}"})
    assert result.status == SUCCEEDED and result.result == "ok 1"


def test_handler_exception_is_recorded_not_raised():
    queue = q()
    job = queue.enqueue("t1", "boom", max_attempts=1)
    done = queue.run_once({"boom": lambda j: (_ for _ in ()).throw(ValueError("bad"))})
    assert done.status == DEAD and "ValueError" in done.error


def test_unknown_kind_fails_without_burning_retries():
    queue = q()
    job = queue.enqueue("t1", "mystery", max_attempts=5)
    done = queue.run_once({})
    assert done.status == DEAD and "no handler" in done.error


def test_drain_processes_everything_ready():
    queue = q()
    for i in range(5):
        queue.enqueue("t1", "noop", {"i": i})
    done = queue.drain({"noop": lambda j: "done"})
    assert len(done) == 5
    assert queue.stats("t1")["by_status"][SUCCEEDED] == 5


def test_stats_and_list():
    queue = q()
    queue.enqueue("t1", "a")
    queue.enqueue("t1", "b")
    queue.claim()
    stats = queue.stats("t1")
    assert stats["total"] == 2 and stats["running"] == 1 and stats["pending"] == 1
    assert len(queue.list("t1")) == 2
    assert len(queue.list("t1", status=QUEUED)) == 1


def test_tenants_are_isolated_when_claiming():
    queue = q()
    queue.enqueue("t1", "a")
    queue.enqueue("t2", "b")
    assert queue.claim(tenant_id="t2").tenant_id == "t2"
