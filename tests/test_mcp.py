"""MCP protocol behaviour and tool dispatch."""
from __future__ import annotations

import io
import json

import pytest

from loci_enterprise.mcp import TOOLS, MCPServer
from loci_enterprise.store import Document, Store


def server():
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="default", library="lib", path="policy.md",
                            title="差旅", body="# 差旅\n\n住宿上限 800 元",
                            acl=["public"], written_by="test"))
    s.put_document(Document(id="", tenant_id="default", library="lib", path="secret.md",
                            title="机密", body="# 机密\n\n工资表", acl=["finance"],
                            written_by="test"))
    return MCPServer(s, tenant_id="default", principals=["public"])


def call(srv, name, args=None, msg_id=1):
    return srv.handle({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                       "params": {"name": name, "arguments": args or {}}})


def test_initialize_reports_a_server_identity():
    r = server().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert r["result"]["serverInfo"]["name"] == "loci-enterprise"
    assert "tools" in r["result"]["capabilities"]


def test_notifications_get_no_reply():
    assert server().handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_is_complete_and_well_formed():
    srv = server()
    r = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in r["result"]["tools"]]
    assert names == [t["name"] for t in TOOLS]
    assert all(t["inputSchema"]["type"] == "object" for t in r["result"]["tools"])


def test_ping():
    r = server().handle({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    assert r["result"] == {}


def test_unknown_method_is_32601():
    r = server().handle({"jsonrpc": "2.0", "id": 4, "method": "nope"})
    assert r["error"]["code"] == -32601


def test_bad_jsonrpc_version_is_rejected():
    r = server().handle({"jsonrpc": "1.0", "id": 5, "method": "ping"})
    assert r["error"]["code"] == -32600


def test_invalid_json_on_the_wire_is_32700():
    srv = server()
    out = io.StringIO()
    srv.serve(stdin=io.StringIO("{not json\n"), stdout=out)
    assert json.loads(out.getvalue())["error"]["code"] == -32700


def test_search_returns_cited_text():
    r = call(server(), "kb_search", {"query": "住宿上限"})
    text = r["result"]["content"][0]["text"]
    assert "policy.md" in text and "800" in text


def test_search_never_returns_unreadable_documents():
    r = call(server(), "kb_search", {"query": "工资表"})
    assert r["result"]["content"][0]["text"] == "No results."


def test_open_respects_permissions():
    srv = server()
    assert "800" in call(srv, "kb_open", {"path": "policy.md",
                                          "library": "lib"})["result"]["content"][0]["text"]
    r = call(srv, "kb_open", {"path": "secret.md", "library": "lib"})
    assert "error" in r and "not permitted" in r["error"]["message"]


def test_supersede_changes_what_search_returns():
    srv = server()
    srv.store.put_document(Document(id="", tenant_id="default", library="lib",
                                    path="policy-v2.md", title="差旅",
                                    body="# 差旅\n\n住宿上限 1200 元", acl=["public"],
                                    written_by="test"))
    call(srv, "kb_supersede", {"old": "policy.md", "new": "policy-v2.md",
                               "library": "lib"})
    text = call(srv, "kb_search", {"query": "住宿上限"})["result"]["content"][0]["text"]
    assert "1200" in text and "800" not in text


def test_remember_then_recall_supersedes():
    srv = server()
    call(srv, "kb_remember", {"subject_id": "u1", "key": "budget", "value": "500"})
    call(srv, "kb_remember", {"subject_id": "u1", "key": "budget", "value": "800"})
    text = call(srv, "kb_recall", {"subject_id": "u1"})["result"]["content"][0]["text"]
    assert "800" in text and "500" not in text
    hist = call(srv, "kb_recall", {"subject_id": "u1",
                                   "include_history": True})["result"]["content"][0]["text"]
    assert "500" in hist and "superseded" in hist


def test_governance_reports_retirement_candidates():
    srv = server()
    srv.store.put_document(Document(id="", tenant_id="default", library="lib",
                                    path="policy-v2.md", title="差旅",
                                    body="# 差旅\n\n新", acl=["public"], written_by="t"))
    call(srv, "kb_supersede", {"old": "policy.md", "new": "policy-v2.md", "library": "lib"})
    text = call(srv, "kb_governance", {"library": "lib"})["result"]["content"][0]["text"]
    assert "policy.md" in text


def test_stats_reports_counts():
    text = call(server(), "kb_stats")["result"]["content"][0]["text"]
    assert json.loads(text)["documents"] == 2


def test_unknown_tool_is_an_error_response():
    r = call(server(), "kb_nope")
    assert r["error"]["code"] == -32602


def test_stdio_transport_round_trip():
    srv = server()
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    out = io.StringIO()
    srv.serve(stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)
    replies = [json.loads(l) for l in out.getvalue().splitlines()]
    assert len(replies) == 2          # the notification produced nothing
    assert replies[0]["id"] == 1 and replies[1]["id"] == 2
