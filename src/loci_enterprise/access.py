"""Access control: who may do what, in which tenant, to whose resource.

Three things a knowledge base gets wrong before it gets anything else wrong:

1. **Role is not enough.** "editor" in tenant A must mean nothing in tenant B.
   Every check therefore takes an explicit tenant, and a membership row is the
   only thing that grants a role there.
2. **A key is not a user.** Service keys are issued by a person, act with a
   fixed role and an explicit scope list, expire, and can be revoked without
   touching anyone's account. Storing the plaintext key is how a database
   backup becomes a credential leak, so only a salted hash is kept.
3. **Ownership is a fact about the row, not about the request.** A resource
   carries its creator; editors may change their own, admins may change
   anyone's, viewers may change nothing.

The audit trail is written here rather than by callers, so an action that was
allowed and an action that was refused both leave a record.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field

from .store import Store, now_iso

# Ordered: a role satisfies every action its level covers.
ROLE_LEVELS = {"viewer": 1, "editor": 2, "admin": 3, "owner": 4}
ACTION_LEVELS = {"read": 1, "write": 2, "admin": 3, "delete": 3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS memberships (
    tenant_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL,
    scopes      TEXT NOT NULL DEFAULT '',
    salt        TEXT NOT NULL,
    digest      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    created_by  TEXT NOT NULL DEFAULT '',
    expires_at  REAL,
    revoked_at  REAL
);
CREATE INDEX IF NOT EXISTS api_keys_tenant ON api_keys(tenant_id);

CREATE TABLE IF NOT EXISTS resource_owners (
    resource_id TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    owner_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

KEY_PREFIX = "lek"


class AccessDenied(PermissionError):
    """Raised when a principal may not perform an action."""


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    role: str = "viewer"
    scopes: tuple[str, ...] = ()
    key_id: str = ""
    authenticated: bool = True

    def can(self, action: str) -> bool:
        level = ACTION_LEVELS.get(action)
        if level is None:
            raise ValueError(f"unknown action {action!r}")
        if level > ROLE_LEVELS.get(self.role, 0):
            return False
        if self.scopes and action not in self.scopes and "*" not in self.scopes:
            return False
        return True


ANONYMOUS = Principal(tenant_id="", user_id="", role="", authenticated=False)


def _hash(secret: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt),
                               120_000).hex()


@dataclass
class IssuedKey:
    key_id: str
    token: str                      # shown exactly once
    name: str
    role: str
    scopes: list[str] = field(default_factory=list)
    expires_at: float | None = None


class AccessControl:
    def __init__(self, store: Store):
        self.store = store
        self.store.conn.executescript(SCHEMA)

    # --------------------------------------------------------- memberships
    def grant(self, tenant_id: str, user_id: str, role: str, *,
              actor: str = "system") -> None:
        if role not in ROLE_LEVELS:
            raise ValueError(f"unknown role {role!r}")
        self.store.conn.execute(
            "INSERT INTO memberships(tenant_id, user_id, role, created_at, created_by)"
            " VALUES (?,?,?,?,?) ON CONFLICT(tenant_id, user_id) DO UPDATE SET role=excluded.role",
            (tenant_id, user_id, role, now_iso(), actor))
        self.store.audit(tenant_id, actor, f"rbac:grant:{role}", user_id, "")

    def revoke(self, tenant_id: str, user_id: str, *, actor: str = "system") -> bool:
        cur = self.store.conn.execute(
            "DELETE FROM memberships WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id))
        if cur.rowcount:
            self.store.audit(tenant_id, actor, "rbac:revoke", user_id, "")
        return bool(cur.rowcount)

    def members(self, tenant_id: str) -> list[dict]:
        rows = self.store.conn.execute(
            "SELECT tenant_id, user_id, role, created_at FROM memberships"
            " WHERE tenant_id=? ORDER BY user_id", (tenant_id,)).fetchall()
        return [dict(r) for r in rows]

    def principal(self, tenant_id: str, user_id: str) -> Principal | None:
        row = self.store.conn.execute(
            "SELECT role FROM memberships WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id)).fetchone()
        if row is None:
            return None
        return Principal(tenant_id=tenant_id, user_id=user_id, role=row["role"])

    # ------------------------------------------------------------- API keys
    def issue_key(self, tenant_id: str, name: str, *, role: str,
                  scopes: list[str] | None = None, expires_in_days: float | None = None,
                  actor: str = "system") -> IssuedKey:
        if role not in ROLE_LEVELS:
            raise ValueError(f"unknown role {role!r}")
        scopes = list(scopes or ["read"])
        bad = [s for s in scopes if s not in ACTION_LEVELS and s != "*"]
        if bad:
            raise ValueError(f"unknown scope(s): {bad}")
        key_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(24)
        salt = secrets.token_hex(16)
        expires_at = time.time() + expires_in_days * 86400 if expires_in_days else None
        self.store.conn.execute(
            "INSERT INTO api_keys(id, tenant_id, name, role, scopes, salt, digest,"
            " created_at, created_by, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (key_id, tenant_id, name, role, ",".join(scopes), salt,
             _hash(secret, salt), now_iso(), actor, expires_at))
        self.store.audit(tenant_id, actor, "apikey:issue", key_id,
                         f"{name} role={role} scopes={','.join(scopes)}")
        return IssuedKey(key_id=key_id, token=f"{KEY_PREFIX}_{key_id}_{secret}",
                         name=name, role=role, scopes=scopes, expires_at=expires_at)

    def authenticate(self, token: str) -> Principal | None:
        """Resolve `lek_<id>_<secret>`; constant-time compare, expiry enforced."""
        if not token or not token.startswith(KEY_PREFIX + "_"):
            return None
        try:
            _, key_id, secret = token.split("_", 2)
        except ValueError:
            return None
        row = self.store.conn.execute(
            "SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        if row is None or row["revoked_at"]:
            return None
        if row["expires_at"] and row["expires_at"] < time.time():
            return None
        if not hmac.compare_digest(_hash(secret, row["salt"]), row["digest"]):
            return None
        scopes = tuple(s for s in row["scopes"].split(",") if s)
        return Principal(tenant_id=row["tenant_id"], user_id=f"key:{row['name']}",
                         role=row["role"], scopes=scopes, key_id=key_id)

    def revoke_key(self, tenant_id: str, key_id: str, *, actor: str = "system") -> bool:
        cur = self.store.conn.execute(
            "UPDATE api_keys SET revoked_at=? WHERE id=? AND tenant_id=? AND revoked_at IS NULL",
            (time.time(), key_id, tenant_id))
        if cur.rowcount:
            self.store.audit(tenant_id, actor, "apikey:revoke", key_id, "")
        return bool(cur.rowcount)

    def keys(self, tenant_id: str) -> list[dict]:
        rows = self.store.conn.execute(
            "SELECT id, name, role, scopes, created_at, expires_at, revoked_at"
            " FROM api_keys WHERE tenant_id=? ORDER BY created_at DESC",
            (tenant_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ ownership
    def own(self, tenant_id: str, resource_id: str, owner_id: str) -> None:
        self.store.conn.execute(
            "INSERT INTO resource_owners(resource_id, tenant_id, owner_id, created_at)"
            " VALUES (?,?,?,?) ON CONFLICT(resource_id) DO UPDATE SET owner_id=excluded.owner_id",
            (resource_id, tenant_id, owner_id, now_iso()))

    def owner_of(self, resource_id: str) -> str | None:
        row = self.store.conn.execute(
            "SELECT owner_id FROM resource_owners WHERE resource_id=?",
            (resource_id,)).fetchone()
        return row["owner_id"] if row else None

    # -------------------------------------------------------------- the gate
    def require(self, principal: Principal, action: str, *,
                resource_id: str | None = None, tenant_id: str | None = None) -> Principal:
        """Allow, or raise AccessDenied with the reason recorded in the audit log."""
        tenant = tenant_id or principal.tenant_id
        reason = ""
        allowed = True

        if not principal.authenticated:
            allowed, reason = False, "anonymous"
        elif principal.tenant_id != tenant:
            allowed, reason = False, "tenant mismatch"
        elif not principal.can(action):
            allowed, reason = False, f"role {principal.role!r} lacks {action}"
        elif action in ("write", "delete") and resource_id:
            owner = self.owner_of(resource_id)
            if owner and owner != principal.user_id \
                    and ROLE_LEVELS.get(principal.role, 0) < ROLE_LEVELS["admin"]:
                allowed, reason = False, f"not the owner of {resource_id}"

        self.store.audit(
            tenant, principal.user_id or "-",
            ("rbac:allow:" if allowed else "rbac:deny:") + action,
            resource_id or "-", reason)
        if not allowed:
            raise AccessDenied(reason or "denied")
        return principal

    # ------------------------------------------------------------ reporting
    def audit_tail(self, tenant_id: str, limit: int = 100) -> list[dict]:
        return self.store.audit_tail(tenant_id, limit)

    def matrix(self) -> dict[str, dict[str, bool]]:
        """The whole permission model, printable, so it can be reviewed once."""
        return {role: {action: ROLE_LEVELS[role] >= ACTION_LEVELS[action]
                       for action in ACTION_LEVELS}
                for role in ROLE_LEVELS}
