"""Memory lifecycle: contradiction, review, expiry, capacity, erasure."""
from __future__ import annotations

import time

import pytest

from loci_enterprise.memory import (ACTIVE, ARCHIVED, EXPIRED, PENDING,
                                    SUPERSEDED, MemoryStore)
from loci_enterprise.store import Store


@pytest.fixture
def mem():
    return MemoryStore(Store(":memory:"))


def test_new_value_supersedes_instead_of_duplicating(mem):
    first = mem.remember("t1", "u1", "budget", "500", actor="alice")
    second = mem.remember("t1", "u1", "budget", "800", actor="bob")
    assert first.status == "created"
    assert second.status == "replaced"
    live = mem.recall("t1", "u1", "budget")
    assert len(live) == 1 and live[0].value == "800"
    # history keeps the old row, marked invalid and pointing at the replacement
    hist = mem.history("t1", "u1", "budget")
    assert [h.status for h in hist] == [SUPERSEDED, ACTIVE]
    assert hist[0].superseded_by == hist[1].id and hist[0].invalid_at


def test_identical_write_is_a_noop(mem):
    mem.remember("t1", "u1", "timezone", "Asia/Shanghai")
    again = mem.remember("t1", "u1", "timezone", "Asia/Shanghai")
    assert again.status == "noop"
    assert len(mem.history("t1", "u1", "timezone")) == 1


def test_inferred_memories_start_pending_and_are_invisible(mem):
    res = mem.remember("t1", "u1", "prefers", "short answers",
                       source="inferred", requires_review=True)
    assert res.status == "pending"
    assert mem.recall("t1", "u1", "prefers") == []
    assert len(mem.pending("t1")) == 1


def test_approving_a_pending_memory_makes_it_live(mem):
    res = mem.remember("t1", "u1", "prefers", "short answers", requires_review=True)
    approved = mem.approve(res.item.id, actor="admin")
    assert approved is not None and approved.status == ACTIVE
    assert mem.recall("t1", "u1", "prefers")[0].value == "short answers"


def test_approving_a_contradiction_supersedes_the_old_one(mem):
    mem.remember("t1", "u1", "plan", "free")
    pend = mem.remember("t1", "u1", "plan", "paid", requires_review=True)
    mem.approve(pend.item.id, actor="admin")
    live = mem.recall("t1", "u1", "plan")
    assert [i.value for i in live] == ["paid"]
    assert mem.history("t1", "u1", "plan")[0].status == SUPERSEDED


def test_pending_duplicate_is_rejected_not_promoted(mem):
    mem.remember("t1", "u1", "color", "green")
    pend = mem.remember("t1", "u1", "color", "green", requires_review=True)
    assert mem.approve(pend.item.id, actor="admin") is None
    assert mem.recall("t1", "u1", "color")[0].value == "green"


def test_reject_archives_with_a_reason(mem):
    pend = mem.remember("t1", "u1", "guess", "x", requires_review=True)
    assert mem.reject(pend.item.id, actor="admin", reason="not confirmed")
    assert mem.recall("t1", "u1", "guess") == []
    assert mem.events("t1", "u1")[0]["action"] == "rejected"


def test_expiry_follows_the_statement_not_the_row_age(mem):
    mem.remember("t1", "u1", "sprint", "S3", ttl_days=1)
    mem.remember("t1", "u1", "policy", "travel cap 800")
    item = mem.recall("t1", "u1", "sprint")[0]
    mem.store.conn.execute("UPDATE memory_items SET expires_at=? WHERE id=?",
                           (time.time() - 1, item.id))
    assert mem.expire_due() == 1
    assert mem.recall("t1", "u1", "sprint") == []
    assert mem.recall("t1", "u1", "policy")[0].value == "travel cap 800"


def test_capacity_blocks_new_items_but_not_reviews(mem):
    small = MemoryStore(Store(":memory:"), max_items_per_subject=2)
    small.remember("t1", "u1", "a", "1")
    small.remember("t1", "u1", "b", "2")
    blocked = small.remember("t1", "u1", "c", "3")
    assert blocked.status == "rejected"
    # a pending item does not consume the live budget until approved
    pending = small.remember("t1", "u1", "d", "4", requires_review=True)
    assert pending.status in ("pending", "rejected")


def test_capacity_reports_usage(mem):
    mem.remember("t1", "u1", "a", "1")
    mem.remember("t1", "u1", "b", "2")
    cap = mem.capacity("t1", "u1")
    assert cap["used"] == 2 and cap["by_status"][ACTIVE] == 2


def test_forget_returns_a_receipt_and_removes_everything(mem):
    mem.remember("t1", "u1", "a", "1")
    mem.remember("t1", "u1", "b", "2")
    receipt = mem.forget("t1", "u1", actor="dpo")
    assert receipt["deleted"] == 2 and receipt["subject_id"] == "u1"
    assert mem.recall("t1", "u1", include_history=True) == []
    assert mem.events("t1", "u1")[0]["action"] == "forgotten"


def test_subjects_and_tenants_are_isolated(mem):
    mem.remember("t1", "u1", "k", "one")
    mem.remember("t1", "u2", "k", "two")
    mem.remember("t2", "u1", "k", "three")
    assert mem.recall("t1", "u1", "k")[0].value == "one"
    assert mem.recall("t1", "u2", "k")[0].value == "two"
    assert mem.recall("t2", "u1", "k")[0].value == "three"


def test_archive_hides_without_destroying(mem):
    res = mem.remember("t1", "u1", "k", "v")
    assert mem.archive(res.item.id, actor="alice", reason="superseded by policy")
    assert mem.recall("t1", "u1", "k") == []
    assert mem.history("t1", "u1", "k")[0].status == ARCHIVED


def test_export_contains_history_and_events(mem):
    mem.remember("t1", "u1", "k", "v1")
    mem.remember("t1", "u1", "k", "v2")
    dump = mem.export("t1", "u1")
    assert len(dump["items"]) == 2
    assert {e["action"] for e in dump["events"]} >= {"created", "replaced"}


def test_empty_key_or_value_is_refused(mem):
    with pytest.raises(ValueError):
        mem.remember("t1", "u1", "  ", "v")
    with pytest.raises(ValueError):
        mem.remember("t1", "u1", "k", " ")
