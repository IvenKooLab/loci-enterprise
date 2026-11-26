# Operations

## Running it

Three processes, one store:

| Mode | Command | Use |
|---|---|---|
| CLI | `le --db /data/le.db ...` | one-off maintenance, scripts |
| HTTP | `le --db /data/le.db serve --host 0.0.0.0 --port 8971` | services and internal tools |
| MCP | `le --db /data/le.db mcp` | an assistant host launches this per session |

The HTTP API has **no anonymous mode by default**: every request needs
`Authorization: Bearer lek_...`. Loopback-only consoles may opt into
`--allow-anonymous`, which grants the viewer role and nothing more.

## First run

```bash
# 1. create the database and a tenant owner key
le --db /data/le.db key issue bootstrap --role admin --scopes read,write,admin,delete
#    -> prints the token once; store it in your secret manager

# 2. pull a source in
le --db /data/le.db --tenant default sync local /srv/wiki --library hr --name wiki

# 3. compile the wiki and check what can no longer be trusted
le --db /data/le.db --tenant default wiki build --library hr
le --db /data/le.db --tenant default scan --library hr
```

## Keys and roles

Roles are ordered `viewer < editor < admin < owner`. A key carries a role *and*
an explicit scope list; the effective permission is the intersection. Issue one
key per integration, never share the bootstrap key.

```bash
le key issue etl --role editor --scopes read,write --expires-in-days 90
le key list
le key revoke --key-id <id>
le perm                       # print the whole matrix for review
```

## Backup and restore

The whole state is a single SQLite file (plus WAL during writes).

```bash
sqlite3 /data/le.db ".backup '/backup/le-$(date +%F).db'"
```

Restore by stopping the service and replacing the file. Do not copy a live
`.db` without `.backup` — a file copy can miss the WAL.

## Upgrade

```bash
docker pull loci-enterprise:<new>
# stop, snapshot, start; the schema is additive (CREATE TABLE IF NOT EXISTS)
```

There are no destructive migrations in this codebase yet; if one is added it
must ship with a backup instruction and a downgrade note.

## Health and observability

- `GET /health` — liveness, no auth
- `GET /stats` — document and chunk counts
- traces: `Tracer(path=...)` appends one JSON object per span; ship that file
  to your tracing backend (the shape is service/trace_id/span tree)
- job queue: `le queue stats` — `pending`, `running`, `dead`. A non-zero `dead`
  is the signal to look, not to restart.

## Multi-tenancy

Every query takes a tenant. There is no "all tenants" code path, and the HTTP
API derives the tenant from the key — never from the request body. Provisioning
a tenant is `le --tenant <t> member grant <user> owner`.
