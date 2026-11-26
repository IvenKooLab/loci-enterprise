"""MCP server over stdio: the governance layer as tools an assistant can call.

Protocol: JSON-RPC 2.0, one message per line, stdout carries protocol only
(anything human-facing goes to stderr). That last rule is not cosmetic — a
stray ``print`` corrupts the stream and the host reports "connection failed"
with no clue why.

The tool surface is deliberately small and read-biased:

    kb_search        hybrid retrieval, ACL-scoped
    kb_open          one document by path
    kb_stats         what is indexed
    kb_governance    what can no longer be trusted
    kb_supersede     declare A replaced by B (write)
    kb_remember      store a durable fact (write)
    kb_recall        read facts, current or historical
    kb_wiki          compiled pages and dead links

Every write goes through the same store the CLI uses, so the audit trail is
identical whether a human or an assistant made the change.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from . import lifecycle
from .memory import MemoryStore
from .retrieval import RetrievalConfig, Retriever
from .store import Store
from .wiki import WikiStore

PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {"name": "kb_search",
     "description": "Search the knowledge base. Returns cited excerpts the caller is "
                    "allowed to read.",
     "inputSchema": {"type": "object", "required": ["query"],
                     "properties": {"query": {"type": "string"},
                                    "library": {"type": "string"},
                                    "top_k": {"type": "integer"}}}},
    {"name": "kb_open",
     "description": "Read one document by its path, as recorded in search results.",
     "inputSchema": {"type": "object", "required": ["path"],
                     "properties": {"path": {"type": "string"},
                                    "library": {"type": "string"}}}},
    {"name": "kb_stats",
     "description": "Counts of documents and chunks in this tenant.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "kb_governance",
     "description": "Read-only scan: superseded documents still indexed, duplicate "
                    "active titles, and retirement candidates.",
     "inputSchema": {"type": "object", "properties": {"library": {"type": "string"}}}},
    {"name": "kb_supersede",
     "description": "Declare that one document replaced another. Both directions are "
                    "written and the old one stops being returned.",
     "inputSchema": {"type": "object", "required": ["old", "new"],
                     "properties": {"old": {"type": "string"}, "new": {"type": "string"},
                                    "library": {"type": "string"},
                                    "note": {"type": "string"}}}},
    {"name": "kb_remember",
     "description": "Store a durable fact for a subject. A different value for the same "
                    "key supersedes the old one; identical writes are no-ops.",
     "inputSchema": {"type": "object",
                     "required": ["subject_id", "key", "value"],
                     "properties": {"subject_id": {"type": "string"},
                                    "key": {"type": "string"},
                                    "value": {"type": "string"},
                                    "ttl_days": {"type": "number"},
                                    "requires_review": {"type": "boolean"}}}},
    {"name": "kb_recall",
     "description": "Read a subject's facts. By default only currently-true ones; "
                    "include_history shows superseded and expired rows.",
     "inputSchema": {"type": "object", "required": ["subject_id"],
                     "properties": {"subject_id": {"type": "string"},
                                    "key": {"type": "string"},
                                    "include_history": {"type": "boolean"}}}},
    {"name": "kb_graph",
     "description": "Knowledge graph over the corpus: neighbours of a node, a path "
                    "between two nodes, or graph stats (isolated notes, broken links).",
     "inputSchema": {"type": "object",
                     "properties": {"action": {"type": "string",
                                               "enum": ["stats", "neighbors", "path"]},
                                    "node": {"type": "string"},
                                    "to": {"type": "string"},
                                    "library": {"type": "string"}}}},
    {"name": "kb_wiki",
     "description": "List compiled wiki pages and any dead links.",
     "inputSchema": {"type": "object", "properties": {"library": {"type": "string"}}}},
]


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class MCPServer:
    def __init__(self, store: Store, *, tenant_id: str = "default",
                 principals: list[str] | None = None):
        self.store = store
        self.tenant_id = tenant_id
        self.principals = principals or ["public"]
        self.memory = MemoryStore(store)
        self.retriever = Retriever(store, config=RetrievalConfig(top_k=5))

    # ------------------------------------------------------------- protocol
    def handle(self, message: dict) -> dict | None:
        """One request -> one response, or None for a notification."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _err(None, -32600, "invalid request")
        method = message.get("method")
        msg_id = message.get("id")
        if method is None:
            return _err(msg_id, -32600, "missing method")
        if msg_id is None:                       # notification: never reply
            return None
        try:
            if method == "initialize":
                return _ok(msg_id, {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {"name": "loci-enterprise", "version": "0.1.0"},
                    "capabilities": {"tools": {},
                                     "resources": {},
                                     "prompts": {}}})
            if method == "ping":
                return _ok(msg_id, {})
            if method == "tools/list":
                return _ok(msg_id, {"tools": TOOLS})
            if method == "tools/call":
                params = message.get("params") or {}
                return _ok(msg_id, self.call_tool(params.get("name", ""),
                                                 params.get("arguments") or {}))
            if method == "resources/list":
                return _ok(msg_id, {"resources": self.resources()})
            if method == "prompts/list":
                return _ok(msg_id, {"prompts": [
                    {"name": "knowledge-briefing",
                     "description": "Summarise what the knowledge base says about a topic"}]})
            raise MCPError(-32601, f"method not found: {method}")
        except MCPError as exc:
            return _err(msg_id, exc.code, exc.message)
        except Exception as exc:                 # noqa: BLE001 - protocol must stay alive
            return _err(msg_id, -32603, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _text(value: Any) -> dict:
        return {"content": [{"type": "text", "text": value}]}

    def call_tool(self, name: str, args: dict[str, Any]) -> dict:
        known = {t["name"] for t in TOOLS}
        if name not in known:
            raise MCPError(-32602, f"unknown tool: {name}")

        if name == "kb_search":
            hits = self.retriever.search(
                str(args["query"]), tenant_id=self.tenant_id,
                principals=self.principals, library=args.get("library"))
            if args.get("top_k"):
                hits = hits[: int(args["top_k"])]
            if not hits:
                return self._text("No results.")
            return self._text("\n\n".join(
                f"[{i + 1}] {h.path} ({h.status})\n{h.text[:600]}"
                for i, h in enumerate(hits)))

        if name == "kb_open":
            doc = self.store.get_document(self.tenant_id, args.get("library"),
                                          str(args["path"]))
            if doc is None:
                raise MCPError(-32602, f"no document at {args['path']}")
            if not set(doc.acl) & set(self.principals):
                raise MCPError(-32602, "not permitted")
            return self._text(doc.body)

        if name == "kb_stats":
            return self._text(json.dumps(self.store.counts(self.tenant_id),
                                         ensure_ascii=False))

        if name == "kb_governance":
            cands = lifecycle.retirement_candidates(
                self.store, self.tenant_id, library=args.get("library"))
            conflicts = lifecycle.conflict_scan(self.store, self.tenant_id,
                                                args.get("library"))
            return self._text(json.dumps(
                {"retirement": [c.__dict__ for c in cands], "conflicts": conflicts},
                ensure_ascii=False, indent=2))

        if name == "kb_supersede":
            old, new = lifecycle.supersede(
                self.store, self.tenant_id, str(args["old"]), str(args["new"]),
                library=args.get("library") or "default", actor="mcp",
                note=str(args.get("note") or ""))
            return self._text(f"{old.path} superseded by {new.path}")

        if name == "kb_remember":
            res = self.memory.remember(
                self.tenant_id, str(args["subject_id"]), str(args["key"]),
                str(args["value"]), source="mcp", ttl_days=args.get("ttl_days"),
                actor="mcp", requires_review=bool(args.get("requires_review")))
            return self._text(json.dumps({"status": res.status,
                                          "id": res.item.id if res.item else None},
                                         ensure_ascii=False))

        if name == "kb_recall":
            items = self.memory.recall(self.tenant_id, str(args["subject_id"]),
                                       args.get("key"),
                                       include_history=bool(args.get("include_history")))
            if not items:
                return self._text("No facts.")
            return self._text("\n".join(f"{i.key} = {i.value} [{i.status}]"
                                        for i in items))

        if name == "kb_graph":
            from .graph import GraphStore
            g = GraphStore(self.store)
            library = args.get("library") or "default"
            action = args.get("action", "stats")
            if action == "stats":
                return self._text(json.dumps(g.stats(self.tenant_id, library),
                                             ensure_ascii=False))
            if action == "neighbors":
                return self._text(json.dumps(
                    g.neighbors(self.tenant_id, library, str(args.get("node", ""))),
                    ensure_ascii=False))
            if action == "path":
                found = g.path(self.tenant_id, library, str(args.get("node", "")),
                               str(args.get("to", "")))
                return self._text(" -> ".join(found) if found else "(no path)")
            raise MCPError(-32602, f"unknown graph action {action}")

        if name == "kb_wiki":
            wiki = WikiStore(self.store)
            library = args.get("library") or "default"
            pages = wiki.pages(self.tenant_id, library)
            dead = wiki.dead_links(self.tenant_id, library)
            return self._text(json.dumps(
                {"pages": [{"slug": p.slug, "title": p.title, "version": p.version,
                            "citations": p.citations} for p in pages],
                 "dead_links": dead}, ensure_ascii=False, indent=2))

        raise MCPError(-32602, f"tool not implemented: {name}")

    def resources(self) -> list[dict]:
        return [{"uri": f"kb://document/{d.path}", "name": d.title or d.path,
                 "mimeType": "text/markdown"}
                for d in self.store.list_documents(self.tenant_id)
                if set(d.acl) & set(self.principals)]

    # ----------------------------------------------------------------- stdio
    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stdout.write(json.dumps(_err(None, -32700, "parse error")) + "\n")
                stdout.flush()
                continue
            response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()


def _ok(msg_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}
