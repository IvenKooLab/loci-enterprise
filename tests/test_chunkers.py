"""Chunking strategies: structure, coverage, overlap, rebuild."""
from __future__ import annotations

import pytest

from loci_enterprise import chunkers
from loci_enterprise.store import Document, Store

BODY = """# 差旅政策

## 住宿

住宿上限 800 元。旺季可上浮 20%。

## 交通

高铁二等座实报实销。飞机经济舱需提前审批。
"""


def test_heading_strategy_returns_parent_chunks_only():
    cs = chunkers.chunk(BODY, "差旅", "heading")
    assert cs and all(c.level == "parent" for c in cs)


def test_fixed_window_covers_every_character():
    text = "".join(chr(0x4E00 + i) for i in range(1000))   # all distinct
    cs = chunkers.fixed_window(text, size=300, overlap=50)
    # windows start at 0, 250, 500, 750 and the last reaches the end
    assert cs[1].text == text[250:550]
    assert cs[-1].text.endswith(text[-1])
    assert len(cs) == 4


def test_fixed_window_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunkers.fixed_window("x", size=0)
    with pytest.raises(ValueError):
        chunkers.fixed_window("x", size=100, overlap=100)


def test_parent_child_links_children_to_their_section():
    cs = chunkers.parent_child(BODY, child_size=120, overlap=20)
    parents = [c for c in cs if c.level == "parent"]
    children = [c for c in cs if c.level == "child"]
    assert len(parents) >= 2 and children
    parent_ids = {str(p.ord) for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_parent_child_keeps_the_heading_path_on_children():
    cs = chunkers.parent_child(BODY, child_size=80, overlap=10)
    child = next(c for c in cs if c.level == "child" and "住宿" in c.text)
    assert "住宿" in child.heading_path


def test_parent_child_rejects_tiny_children():
    with pytest.raises(ValueError):
        chunkers.parent_child(BODY, child_size=40)


def test_unknown_strategy_lists_the_options():
    with pytest.raises(ValueError) as exc:
        chunkers.chunk(BODY, strategy="magic")
    assert "parent-child" in str(exc.value)


def test_rebuild_library_replaces_chunks_and_keeps_acl():
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="t1", library="lib", path="a.md",
                            title="差旅", body=BODY, acl=["public", "hr"],
                            written_by="t"))
    before = s.counts("t1")["chunks"]
    report = chunkers.rebuild_library(s, "t1", "lib", "parent-child",
                                      child_size=120, overlap=20)
    assert report["documents"] == 1 and report["chunks"] > before
    acl = {r["principal"] for r in s.conn.execute(
        "SELECT DISTINCT principal FROM chunk_acl").fetchall()}
    assert acl == {"public", "hr"}


def test_rebuild_is_idempotent_in_count():
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="t1", library="lib", path="a.md",
                            title="T", body=BODY, acl=["public"], written_by="t"))
    first = chunkers.rebuild_library(s, "t1", "lib", "fixed", size=200, overlap=20)
    second = chunkers.rebuild_library(s, "t1", "lib", "fixed", size=200, overlap=20)
    assert first["chunks"] == second["chunks"]


def test_rebuild_audits_the_change():
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="t1", library="lib", path="a.md",
                            title="T", body=BODY, acl=["public"], written_by="t"))
    chunkers.rebuild_library(s, "t1", "lib", "heading")
    actions = [a["action"] for a in s.audit_tail("t1", 10)]
    assert "rechunk:heading" in actions
