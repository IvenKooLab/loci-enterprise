"""Schema migrations: the difference between an upgrade and a gamble.

``CREATE TABLE IF NOT EXISTS`` gets a fresh install to a working schema, but it
cannot change an existing one. As soon as a column or an index is added, a
deployed database needs an ordered, recorded upgrade path — otherwise the only
options are "delete the data" or "hope".

Rules this module enforces:

* **Every migration has a number and a name**, and is applied at most once.
  The applied set is a table, not a log line.
* **Each migration runs in a transaction with its version record.** A failure
  rolls back both the change and the record, so a retry starts from the same
  known state instead of a half-applied schema.
* **Downgrades are refused.** Opening a database written by a newer build is a
  configuration error, not something to guess about.
* **The whole plan is inspectable before running**: ``pending()`` lists what
  would happen.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from .store import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _sql(*statements: str) -> Callable[[sqlite3.Connection], None]:
    def run(conn: sqlite3.Connection) -> None:
        for statement in statements:
            conn.execute(statement)
    return run


# Forward-only, ordered. Never edit an entry that has shipped: add a new one.
MIGRATIONS: list[Migration] = [
    Migration(1, "index documents by library",
              _sql("CREATE INDEX IF NOT EXISTS documents_library"
                   " ON documents(tenant_id, library)")),
    Migration(2, "index chunks by scope and document",
              _sql("CREATE INDEX IF NOT EXISTS chunks_scope_doc"
                   " ON chunks(tenant_id, library, doc_id)")),
    Migration(3, "index audit by tenant and time",
              _sql("CREATE INDEX IF NOT EXISTS audit_tenant_ts"
                   " ON audit(tenant_id, id DESC)")),
]


class MigrationError(RuntimeError):
    pass


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (name,)).fetchone()
    return row is not None


def current(store: Store) -> int:
    if not _table_exists(store.conn, "schema_migrations"):
        return 0
    row = store.conn.execute("SELECT MAX(version) v FROM schema_migrations").fetchone()
    return int(row["v"] or 0)


def applied(store: Store) -> list[dict]:
    if not _table_exists(store.conn, "schema_migrations"):
        return []
    return [dict(r) for r in store.conn.execute(
        "SELECT version, name, applied_at FROM schema_migrations ORDER BY version")]


def pending(store: Store) -> list[Migration]:
    have = current(store)
    return [m for m in MIGRATIONS if m.version > have]


def apply_all(store: Store, *, target: int | None = None) -> list[dict]:
    """Apply pending migrations in order. Returns what was applied."""
    store.conn.executescript(SCHEMA)
    have = current(store)
    want = max([m.version for m in MIGRATIONS] if target is None else
               [target])
    if have > want:
        raise MigrationError(
            f"database schema v{have} is newer than this build (v{want}); "
            f"upgrade the application instead of downgrading the data")
    done: list[dict] = []
    for migration in pending(store):
        if migration.version > want:
            continue
        # explicit transaction per migration: DDL + the record commit together
        store.conn.execute("BEGIN")
        try:
            migration.apply(store.conn)
            store.conn.execute(
                "INSERT INTO schema_migrations(version, name, applied_at)"
                " VALUES (?,?,datetime('now'))", (migration.version, migration.name))
            store.conn.execute("COMMIT")
        except Exception as exc:                     # noqa: BLE001 - re-raised below
            store.conn.execute("ROLLBACK")
            raise MigrationError(
                f"migration {migration.version} ({migration.name}) failed: {exc}") from exc
        done.append({"version": migration.version, "name": migration.name})
    return done


def require_at_least(store: Store, version: int) -> None:
    have = current(store)
    if have < version:
        raise MigrationError(
            f"database schema v{have} is older than required v{version}; "
            f"run the migrations before starting the service")


def status(store: Store) -> dict:
    have = current(store)
    latest = max(m.version for m in MIGRATIONS) if MIGRATIONS else 0
    return {"current": have, "latest": latest,
            "pending": [m.version for m in pending(store)],
            "up_to_date": have == latest}
