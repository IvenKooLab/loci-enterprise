"""Migration ordering, idempotency, rollback and downgrade refusal."""
from __future__ import annotations

import sqlite3

import pytest

from loci_enterprise import migrations as mg
from loci_enterprise.store import Store


def test_fresh_store_migrates_to_the_latest_version():
    s = Store(":memory:")
    assert mg.current(s) == 0
    done = mg.apply_all(s)
    assert [d["version"] for d in done] == [m.version for m in mg.MIGRATIONS]
    assert mg.status(s)["up_to_date"] is True


def test_apply_is_idempotent():
    s = Store(":memory:")
    mg.apply_all(s)
    assert mg.apply_all(s) == []
    assert len(mg.applied(s)) == len(mg.MIGRATIONS)


def test_pending_lists_what_would_run():
    s = Store(":memory:")
    assert [m.version for m in mg.pending(s)] == [m.version for m in mg.MIGRATIONS]
    mg.apply_all(s)
    assert mg.pending(s) == []


def test_can_stop_at_an_explicit_target():
    s = Store(":memory:")
    mg.apply_all(s, target=1)
    assert mg.current(s) == 1
    assert len(mg.pending(s)) == len(mg.MIGRATIONS) - 1


def test_a_failing_migration_rolls_back_and_is_not_recorded(monkeypatch):
    s = Store(":memory:")
    broken = mg.Migration(1, "broken", lambda conn: (_ for _ in ()).throw(
        sqlite3.OperationalError("boom")))
    monkeypatch.setattr(mg, "MIGRATIONS", [broken])
    with pytest.raises(mg.MigrationError):
        mg.apply_all(s)
    assert mg.current(s) == 0
    assert mg.applied(s) == []


def test_downgrade_is_refused(monkeypatch):
    s = Store(":memory:")
    mg.apply_all(s)
    monkeypatch.setattr(mg, "MIGRATIONS", [mg.Migration(1, "older", lambda c: None)])
    with pytest.raises(mg.MigrationError):
        mg.apply_all(s)


def test_require_at_least_reports_the_gap():
    s = Store(":memory:")
    with pytest.raises(mg.MigrationError):
        mg.require_at_least(s, 3)
    mg.apply_all(s)
    mg.require_at_least(s, 3)


def test_status_shape():
    s = Store(":memory:")
    st = mg.status(s)
    assert set(st) == {"current", "latest", "pending", "up_to_date"}
    assert st["current"] == 0 and st["latest"] >= 1
