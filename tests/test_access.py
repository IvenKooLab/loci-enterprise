"""Roles, tenant isolation, scoped keys and ownership."""
from __future__ import annotations

import time

import pytest

from loci_enterprise.access import (ROLE_LEVELS, AccessControl, AccessDenied,
                                    Principal)
from loci_enterprise.store import Store


@pytest.fixture
def ac():
    return AccessControl(Store(":memory:"))


def test_role_matrix_is_ordered_and_complete(ac):
    m = ac.matrix()
    assert set(m) == set(ROLE_LEVELS)
    # every role can read; only admin and above may administer
    assert all(m[r]["read"] for r in m)
    assert not m["viewer"]["write"] and m["editor"]["write"]
    assert not m["editor"]["admin"] and m["admin"]["admin"]
    assert m["admin"]["delete"] and not m["editor"]["delete"]


def test_membership_grants_role_only_in_its_tenant(ac):
    ac.grant("t1", "alice", "editor")
    assert ac.principal("t1", "alice").role == "editor"
    assert ac.principal("t2", "alice") is None


def test_tenant_mismatch_is_denied_and_audited(ac):
    ac.grant("t1", "alice", "admin")
    p = ac.principal("t1", "alice")
    with pytest.raises(AccessDenied):
        ac.require(p, "read", tenant_id="t2")
    actions = [a["action"] for a in ac.audit_tail("t2", 5)]
    assert "rbac:deny:read" in actions


def test_viewer_cannot_write(ac):
    ac.grant("t1", "bob", "viewer")
    with pytest.raises(AccessDenied):
        ac.require(ac.principal("t1", "bob"), "write")


def test_editor_cannot_touch_someone_elses_resource(ac):
    ac.grant("t1", "bob", "editor")
    ac.own("t1", "doc-1", "alice")
    with pytest.raises(AccessDenied):
        ac.require(ac.principal("t1", "bob"), "write", resource_id="doc-1")
    # but may edit their own
    ac.own("t1", "doc-2", "bob")
    assert ac.require(ac.principal("t1", "bob"), "write", resource_id="doc-2")


def test_admin_may_edit_anyones_resource(ac):
    ac.grant("t1", "carol", "admin")
    ac.own("t1", "doc-1", "alice")
    assert ac.require(ac.principal("t1", "carol"), "write", resource_id="doc-1")


def test_anonymous_is_always_denied(ac):
    from loci_enterprise.access import ANONYMOUS
    with pytest.raises(AccessDenied):
        ac.require(ANONYMOUS, "read")


def test_key_roundtrip_and_scopes(ac):
    issued = ac.issue_key("t1", "etl", role="editor", scopes=["read", "write"])
    p = ac.authenticate(issued.token)
    assert p is not None and p.role == "editor" and p.key_id == issued.key_id
    assert p.can("write") and not p.can("admin")


def test_key_plaintext_is_never_stored(ac):
    issued = ac.issue_key("t1", "etl", role="viewer")
    row = ac.store.conn.execute("SELECT * FROM api_keys WHERE id=?",
                                (issued.key_id,)).fetchone()
    dumped = " ".join(str(v) for v in tuple(row))
    assert issued.token not in dumped
    secret = issued.token.rsplit("_", 1)[-1]
    assert secret not in dumped


def test_wrong_secret_is_rejected(ac):
    issued = ac.issue_key("t1", "etl", role="viewer")
    bad = issued.token[:-4] + "0000"
    assert ac.authenticate(bad) is None


def test_revoked_key_stops_working(ac):
    issued = ac.issue_key("t1", "etl", role="viewer")
    assert ac.authenticate(issued.token)
    assert ac.revoke_key("t1", issued.key_id, actor="alice")
    assert ac.authenticate(issued.token) is None


def test_expired_key_stops_working(ac):
    issued = ac.issue_key("t1", "etl", role="viewer", expires_in_days=1)
    assert ac.authenticate(issued.token)
    # move the clock past the expiry rather than sleeping for a day
    ac.store.conn.execute("UPDATE api_keys SET expires_at=? WHERE id=?",
                          (time.time() - 1, issued.key_id))
    assert ac.authenticate(issued.token) is None


def test_scope_limits_a_powerful_role(ac):
    issued = ac.issue_key("t1", "ro-bot", role="admin", scopes=["read"])
    p = ac.authenticate(issued.token)
    assert p.can("read") and not p.can("write") and not p.can("admin")


def test_unknown_scope_is_rejected_at_issue_time(ac):
    with pytest.raises(ValueError):
        ac.issue_key("t1", "x", role="viewer", scopes=["teleport"])


def test_allow_and_deny_both_reach_the_audit_log(ac):
    ac.grant("t1", "alice", "editor")
    p = ac.principal("t1", "alice")
    ac.require(p, "read")
    with pytest.raises(AccessDenied):
        ac.require(p, "admin")
    actions = [a["action"] for a in ac.audit_tail("t1", 10)]
    assert "rbac:allow:read" in actions
    assert "rbac:deny:admin" in actions
