"""HTTP API: the same governance layer, reachable by anything that speaks HTTP.

Authentication is an API key (``lek_...``) resolved by ``AccessControl``; there
is no anonymous mode unless the caller explicitly constructs the server with
``allow_anonymous=True`` for a loopback-only console. Authorisation is checked
per route with the role/action matrix, and the audit trail records both the
allowed and refused calls.

Routes are intentionally coarse — this is a service for agents and internal
tools, not a public API:

    GET  /health                       liveness (no auth)
    GET  /stats                        counts
    POST /search                       {query, library?, top_k?}
    GET  /documents?library=           list documents
    GET  /document?path=&library=      one document
    POST /documents                    create/update {path,title,body,library,acl}
    POST /supersede                    {old,new,library,note?}
    GET  /governance?library=          retirement candidates + conflicts
    POST /memory                       {subject_id,key,value,ttl_days?}
    GET  /memory?subject_id=&key=      recall
    GET  /audit                        admin only
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .access import AccessControl, AccessDenied, Principal
from .memory import MemoryStore
from .retrieval import RetrievalConfig, Retriever
from .store import Document, Store


class App:
    """Route table + business calls, independent of the HTTP plumbing."""

    def __init__(self, store: Store, *, allow_anonymous: bool = False,
                 anonymous_role: str = "viewer"):
        self.store = store
        self.access = AccessControl(store)
        self.memory = MemoryStore(store)
        self.retriever = Retriever(store, config=RetrievalConfig(top_k=5))
        self.allow_anonymous = allow_anonymous
        self.anonymous_role = anonymous_role

    def principal(self, authorization: str) -> Principal | None:
        token = ""
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token:
            return self.access.authenticate(token)
        if self.allow_anonymous:
            return Principal(tenant_id="default", user_id="anonymous",
                             role=self.anonymous_role, scopes=("*",))
        return None

    # ------------------------------------------------------------- handlers
    def handle(self, method: str, path: str, query: dict[str, list[str]],
               body: dict, principal: Principal | None) -> tuple[int, dict]:
        try:
            if path == "/health":
                return 200, {"status": "ok", "version": "0.1.0"}
            if principal is None:
                return 401, {"error": "unauthorized"}
            tenant = principal.tenant_id

            if path == "/stats":
                self._require(principal, "read")
                return 200, self.store.counts(tenant)

            if path == "/search" and method == "POST":
                self._require(principal, "read")
                hits = self.retriever.search(
                    str(body.get("query", "")), tenant_id=tenant,
                    principals=[principal.user_id, *self._groups(principal)],
                    library=body.get("library"))
                if body.get("top_k"):
                    hits = hits[: int(body["top_k"])]
                return 200, {"hits": [{"chunk_id": h.chunk_id, "path": h.path,
                                       "text": h.text, "score": round(h.score, 6),
                                       "status": h.status} for h in hits]}

            if path == "/documents" and method == "GET":
                self._require(principal, "read")
                docs = self.store.list_documents(tenant, _one(query, "library"))
                return 200, {"documents": [
                    {"path": d.path, "title": d.title, "status": d.status,
                     "version": d.version, "updated_at": d.updated_at}
                    for d in docs]}

            if path == "/document" and method == "GET":
                self._require(principal, "read")
                doc = self.store.get_document(tenant, _one(query, "library"),
                                              _one(query, "path"))
                if doc is None:
                    return 404, {"error": "not found"}
                return 200, {"path": doc.path, "title": doc.title, "body": doc.body,
                             "status": doc.status, "version": doc.version}

            if path == "/documents" and method == "POST":
                self._require(principal, "write")
                if not body.get("path") or not body.get("body"):
                    return 400, {"error": "path and body are required"}
                doc = Document(id="", tenant_id=tenant,
                               library=body.get("library") or "default",
                               path=str(body["path"]),
                               title=str(body.get("title") or body["path"]),
                               body=str(body["body"]),
                               acl=list(body.get("acl") or ["public"]),
                               effective_from=body.get("effective_from"),
                               written_by=principal.user_id)
                saved = self.store.put_document(doc, note="http")
                self.access.own(tenant, f"{saved.library}/{saved.path}",
                                principal.user_id)
                return 200, {"path": saved.path, "version": saved.version}

            if path == "/supersede" and method == "POST":
                self._require(principal, "write")
                from .lifecycle import supersede
                old, new = supersede(self.store, tenant, str(body["old"]),
                                     str(body["new"]),
                                     library=body.get("library") or "default",
                                     actor=principal.user_id,
                                     note=str(body.get("note") or ""))
                return 200, {"superseded": old.path, "by": new.path}

            if path == "/governance" and method == "GET":
                self._require(principal, "read")
                from .lifecycle import conflict_scan, retirement_candidates
                lib = _one(query, "library")
                cands = retirement_candidates(self.store, tenant, library=lib)
                return 200, {"retirement": [c.__dict__ for c in cands],
                             "conflicts": conflict_scan(self.store, tenant, lib)}

            if path == "/memory" and method == "POST":
                self._require(principal, "write")
                res = self.memory.remember(
                    tenant, str(body["subject_id"]), str(body["key"]),
                    str(body["value"]), source=str(body.get("source") or "http"),
                    ttl_days=body.get("ttl_days"), actor=principal.user_id,
                    requires_review=bool(body.get("requires_review")))
                return 200, {"status": res.status,
                             "id": res.item.id if res.item else None}

            if path == "/memory" and method == "GET":
                self._require(principal, "read")
                items = self.memory.recall(tenant, _one(query, "subject_id") or "",
                                           _one(query, "key"),
                                           include_history=_truthy(_one(query, "history")))
                return 200, {"items": [i.as_dict() for i in items]}

            if path == "/audit":
                self._require(principal, "admin")
                return 200, {"audit": self.store.audit_tail(tenant, 200)}

            return 404, {"error": f"no route {method} {path}"}
        except AccessDenied as exc:
            return 403, {"error": f"forbidden: {exc}"}
        except KeyError as exc:
            return 400, {"error": f"missing field {exc}"}
        except (ValueError, TypeError) as exc:
            return 400, {"error": str(exc)}

    def _require(self, principal: Principal, action: str) -> None:
        self.access.require(principal, action)

    @staticmethod
    def _groups(principal: Principal) -> list[str]:
        """ACL principals implied by an authenticated caller.

        ``public`` is universal; ``role:<name>`` and ``user:<id>`` let a
        document be scoped to a role or a single person without inventing a
        separate group service.
        """
        return ["public", f"role:{principal.role}", f"user:{principal.user_id}"]


def _one(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name) or []
    return values[0] if values else None


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in ("1", "true", "yes")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "loci-enterprise/0.1"

        def log_message(self, fmt, *args):        # noqa: A003 - quiet by default
            return

        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, method: str) -> None:
            split = urlsplit(self.path)
            query = parse_qs(split.query)
            body: dict = {}
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid JSON body"})
            principal = app.principal(self.headers.get("Authorization", ""))
            code, payload = app.handle(method, split.path, query, body, principal)
            self._send(code, payload)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def serve(store: Store, host: str = "127.0.0.1", port: int = 8971, *,
          allow_anonymous: bool = False) -> ThreadingHTTPServer:
    app = App(store, allow_anonymous=allow_anonymous)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    return server


def run(store: Store, host: str = "127.0.0.1", port: int = 8971, *,
        allow_anonymous: bool = False) -> None:
    server = serve(store, host, port, allow_anonymous=allow_anonymous)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
