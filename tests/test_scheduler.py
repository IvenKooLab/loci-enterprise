"""Schedules: due calculation, burst prevention, enable/remove."""
from __future__ import annotations

import time

import pytest

from loci_enterprise.queue import Queue
from loci_enterprise.scheduler import Schedules
from loci_enterprise.store import Store


@pytest.fixture
def sched():
    store = Store(":memory:")
    return Schedules(store), Queue(store)


def test_add_and_list(sched):
    s, _ = sched
    s.add("t1", "nightly-wiki", "wiki.build", interval_seconds=3600,
          payload={"library": "hr"})
    rows = s.list("t1")
    assert len(rows) == 1 and rows[0].name == "nightly-wiki"
    assert rows[0].payload == {"library": "hr"}


def test_interval_below_the_floor_is_refused(sched):
    s, _ = sched
    with pytest.raises(ValueError):
        s.add("t1", "too-fast", "x", interval_seconds=5)


def test_duplicate_names_are_refused(sched):
    s, _ = sched
    s.add("t1", "a", "x", interval_seconds=60)
    with pytest.raises(ValueError):
        s.add("t1", "a", "x", interval_seconds=60)


def test_not_due_before_its_time(sched):
    s, q = sched
    s.add("t1", "a", "x", interval_seconds=60, first_run_in=100)
    assert s.tick(q, now=50) == []
    assert q.stats("t1")["total"] == 0


def test_tick_enqueues_and_advances(sched):
    s, q = sched
    s.add("t1", "a", "sync.local", interval_seconds=60, first_run_in=0)
    now = time.time() + 1
    fired = s.tick(q, now=now)
    assert len(fired) == 1 and fired[0]["kind"] == "sync.local"
    job = q.claim(tenant_id="t1")
    assert job is not None and job.kind == "sync.local"
    assert s.get("t1", "a").next_run == now + 60


def test_long_outage_does_not_burst(sched):
    """One tick after a week down fires once, not once per missed interval."""
    s, q = sched
    s.add("t1", "a", "x", interval_seconds=60, first_run_in=0)
    now = time.time() + 7 * 86400
    assert len(s.tick(q, now=now)) == 1
    assert s.tick(q, now=now) == []          # same instant: already advanced
    assert s.get("t1", "a").next_run == now + 60


def test_disabled_schedule_does_not_fire(sched):
    s, q = sched
    s.add("t1", "a", "x", interval_seconds=60, first_run_in=0)
    s.set_enabled("t1", "a", False)
    assert s.tick(q, now=time.time() + 10_000) == []
    assert q.stats("t1")["total"] == 0


def test_removed_schedule_stays_removed(sched):
    s, q = sched
    s.add("t1", "a", "x", interval_seconds=60, first_run_in=0)
    assert s.remove("t1", "a")
    assert s.tick(q, now=time.time() + 10_000) == []
    assert s.get("t1", "a") is None


def test_schedules_are_tenant_scoped(sched):
    s, q = sched
    s.add("t1", "a", "x", interval_seconds=60, first_run_in=0)
    s.add("t2", "b", "x", interval_seconds=60, first_run_in=0)
    fired = s.tick(q, now=time.time() + 1)
    assert {f["schedule"] for f in fired} == {"a", "b"}
    assert q.stats("t1")["total"] == 1 and q.stats("t2")["total"] == 1


def test_background_jobs_yield_to_interactive_ones(sched):
    s, q = sched
    s.add("t1", "bg", "x", interval_seconds=60, first_run_in=0)
    q.enqueue("t1", "interactive", priority=5)
    s.tick(q, now=time.time() + 1)
    first = q.claim(tenant_id="t1")
    assert first.kind == "interactive"


def test_firing_is_audited(sched):
    s, q = sched
    s.add("t1", "a", "x", interval_seconds=60, first_run_in=0)
    s.tick(q, now=time.time() + 1)
    actions = [a["action"] for a in s.store.audit_tail("t1", 10)]
    assert "schedule:fire" in actions
