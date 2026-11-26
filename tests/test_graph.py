"""Graph derivation and traversal."""
from __future__ import annotations

import pytest

from loci_enterprise.graph import GraphStore
from loci_enterprise.lifecycle import supersede
from loci_enterprise.store import Document, Store


def put(store, path, title, body, **kw):
    return store.put_document(Document(id="", tenant_id="t1", library="lib",
                                       path=path, title=title, body=body,
                                       acl=["public"], written_by="t", **kw))


def build():
    s = Store(":memory:")
    put(s, "a.md", "差旅", "# 差旅\n\n见 [[年假]]。")
    put(s, "b.md", "年假", "# 年假\n\n15 天。")
    put(s, "c.md", "旧政策", "# 旧政策\n\n旧")
    put(s, "d.md", "新政策", "# 新政策\n\n新")
    supersede(s, "t1", "c.md", "d.md", library="lib", actor="t")
    return s


def test_build_creates_document_nodes_and_link_edges():
    s = build()
    g = GraphStore(s)
    report = g.build("t1", "lib")
    assert report.nodes == 4
    kinds = {e.kind for e in g.edges("t1", "lib")}
    assert "links" in kinds


def test_unresolved_link_becomes_a_stub_not_a_dropped_edge():
    s = Store(":memory:")
    put(s, "a.md", "A", "# A\n\n见 [[不存在]]")
    g = GraphStore(s)
    report = g.build("t1", "lib")
    assert report.stubs == 1
    assert [n.kind for n in g.nodes("t1", "lib")].count("stub") == 1
    assert report.broken[0]["to"] == "不存在"


def test_supersede_becomes_an_edge():
    s = build()
    g = GraphStore(s)
    g.build("t1", "lib")
    assert any(e.kind == "supersedes" and e.src == "d.md" and e.dst == "c.md"
               for e in g.edges("t1", "lib"))


def test_duplicate_titles_are_linked():
    s = Store(":memory:")
    put(s, "x.md", "同一标题", "内容一")
    put(s, "y.md", "同一标题", "内容二")
    g = GraphStore(s)
    g.build("t1", "lib")
    dup = [e for e in g.edges("t1", "lib") if e.kind == "duplicate"]
    assert dup and {dup[0].src, dup[0].dst} == {"x.md", "y.md"}


def test_neighbors_direction():
    s = build()
    g = GraphStore(s)
    g.build("t1", "lib")
    out = g.neighbors("t1", "lib", "a.md", direction="out")
    assert any(n["node"] == "b.md" for n in out)
    assert g.neighbors("t1", "lib", "b.md", direction="in")


def test_subgraph_depth():
    s = Store(":memory:")
    put(s, "a.md", "A", "# A\n\n[[B]]")
    put(s, "b.md", "B", "# B\n\n[[C]]")
    put(s, "c.md", "C", "# C\n\nend")
    g = GraphStore(s)
    g.build("t1", "lib")
    one = g.subgraph("t1", "lib", "a.md", depth=1)
    two = g.subgraph("t1", "lib", "a.md", depth=2)
    assert {n["node_id"] for n in one["nodes"]} == {"a.md", "b.md"}
    assert "c.md" in {n["node_id"] for n in two["nodes"]}


def test_path_between_documents():
    s = Store(":memory:")
    put(s, "a.md", "A", "# A\n\n[[B]]")
    put(s, "b.md", "B", "# B\n\n[[C]]")
    put(s, "c.md", "C", "# C\n\nend")
    g = GraphStore(s)
    g.build("t1", "lib")
    assert g.path("t1", "lib", "a.md", "c.md") == ["a.md", "b.md", "c.md"]


def test_path_returns_none_when_disconnected():
    s = Store(":memory:")
    put(s, "a.md", "A", "# A")
    put(s, "z.md", "Z", "# Z")
    g = GraphStore(s)
    g.build("t1", "lib")
    assert g.path("t1", "lib", "a.md", "z.md") is None


def test_stats_report_isolated_documents():
    s = Store(":memory:")
    put(s, "lonely.md", "Lonely", "# Lonely")
    g = GraphStore(s)
    g.build("t1", "lib")
    assert "lonely.md" in g.stats("t1", "lib")["isolated"]


def test_rebuild_is_deterministic():
    s = build()
    g = GraphStore(s)
    g.build("t1", "lib")
    first = g.export("t1", "lib")
    g.build("t1", "lib")
    second = g.export("t1", "lib")
    assert first == second


def test_invalid_direction_is_refused():
    s = build()
    g = GraphStore(s)
    g.build("t1", "lib")
    with pytest.raises(ValueError):
        g.neighbors("t1", "lib", "a.md", direction="sideways")
