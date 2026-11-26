"""Wiki compilation, dedup, links and rollback."""
from __future__ import annotations

import pytest

from loci_enterprise.store import Document, Store
from loci_enterprise.wiki import WikiBuilder, WikiPage, WikiStore, slugify


def doc(store, path, title, body, *, status="active", library="lib", tenant="t1",
        authority=1.0):
    d = Document(id="", tenant_id=tenant, library=library, path=path, title=title,
                 body=body, status=status, authority=authority,
                 written_by="test", acl=["public"])
    return store.put_document(d)


def test_slugify_keeps_cjk_and_collapses_punctuation():
    assert slugify("差旅 报销 / 政策") == "差旅-报销-政策"
    assert slugify("  ") == "untitled"


def test_build_creates_one_page_per_topic():
    s = Store(":memory:")
    doc(s, "hr/travel.md", "差旅报销", "# 差旅报销\n\n住宿上限 800 元。")
    doc(s, "hr/leave.md", "年假", "# 年假\n\n15 天。")
    report = WikiBuilder(s).build("t1", "lib")
    assert set(report.created) == {"差旅报销", "年假"}
    pages = WikiStore(s).pages("t1", "lib")
    assert {p.title for p in pages} == {"差旅报销", "年假"}


def test_page_body_is_the_document_not_a_paraphrase():
    s = Store(":memory:")
    body = "# 差旅报销\n\n住宿上限 800 元。\n\n## 例外\n\n旺季可上浮。"
    doc(s, "a.md", "差旅报销", body)
    WikiBuilder(s).build("t1", "lib")
    page = WikiStore(s).page("t1", "lib", "差旅报销")
    assert "住宿上限 800 元。" in page.body
    assert "旺季可上浮。" in page.body


def test_rebuild_without_changes_creates_no_new_revision():
    s = Store(":memory:")
    doc(s, "a.md", "政策", "# 政策\n\nv1")
    w = WikiStore(s)
    first = WikiBuilder(s, w).build("t1", "lib")
    second = WikiBuilder(s, w).build("t1", "lib")
    assert first.created and second.unchanged
    assert len(w.revisions("t1", "lib", "政策")) == 1


def test_changed_document_bumps_the_version():
    s = Store(":memory:")
    doc(s, "a.md", "政策", "# 政策\n\nv1")
    w = WikiStore(s)
    WikiBuilder(s, w).build("t1", "lib")
    doc(s, "a.md", "政策", "# 政策\n\nv2")
    report = WikiBuilder(s, w).build("t1", "lib")
    assert report.updated == ["政策"]
    assert w.page("t1", "lib", "政策").version == 2


def test_superseded_documents_do_not_become_pages():
    s = Store(":memory:")
    doc(s, "old.md", "政策", "# 政策\n\n旧")
    doc(s, "new.md", "政策", "# 政策\n\n新")
    from loci_enterprise import lifecycle
    lifecycle.supersede(s, "t1", "old.md", "new.md", library="lib", actor="hr")
    report = WikiBuilder(s).build("t1", "lib")
    page = WikiStore(s).page("t1", "lib", "政策")
    assert "新" in page.body and "旧" not in page.body
    assert "old.md" in report.skipped


def test_sibling_documents_become_a_see_also_list():
    s = Store(":memory:")
    doc(s, "a.md", "差旅报销", "# 差旅报销\n\n主政策", authority=1.5)
    doc(s, "b.md", "差旅报销", "# 差旅报销\n\n补充说明", authority=1.0)
    WikiBuilder(s).build("t1", "lib")
    page = WikiStore(s).page("t1", "lib", "差旅报销")
    assert "主政策" in page.body
    assert "## Sources" in page.body and "补充说明" not in page.body
    assert page.citations == ["a.md", "b.md"]


def test_dead_links_are_reported():
    s = Store(":memory:")
    doc(s, "a.md", "政策", "# 政策\n\n参见 [[不存在的页面]]。")
    report = WikiBuilder(s).build("t1", "lib")
    assert report.dead_links == [{"from": "政策", "to": "不存在的页面"}]


def test_resolvable_links_are_not_reported():
    s = Store(":memory:")
    doc(s, "a.md", "年假", "# 年假\n\n见 [[政策]]。")
    doc(s, "b.md", "政策", "# 政策\n\n正文")
    assert WikiBuilder(s).build("t1", "lib").dead_links == []


def test_rollback_creates_a_new_revision_with_old_bytes():
    s = Store(":memory:")
    doc(s, "a.md", "政策", "# 政策\n\nv1")
    w = WikiStore(s)
    WikiBuilder(s, w).build("t1", "lib")
    doc(s, "a.md", "政策", "# 政策\n\nv2")
    WikiBuilder(s, w).build("t1", "lib")
    restored = w.rollback("t1", "lib", "政策", 1, actor="alice")
    assert "v1" in restored.body
    assert restored.version == 3            # rollback is additive, never a rewind
    assert [r["version"] for r in w.revisions("t1", "lib", "政策")] == [1, 2, 3]


def test_diff_between_revisions():
    s = Store(":memory:")
    doc(s, "a.md", "政策", "# 政策\n\nv1")
    w = WikiStore(s)
    WikiBuilder(s, w).build("t1", "lib")
    doc(s, "a.md", "政策", "# 政策\n\nv2")
    WikiBuilder(s, w).build("t1", "lib")
    d = w.diff("t1", "lib", "政策", 1, 2)
    assert "-v1" in d["unified"] and "+v2" in d["unified"]


def test_libraries_and_tenants_are_isolated():
    s = Store(":memory:")
    doc(s, "a.md", "政策", "# 政策\n\nA", library="l1")
    doc(s, "a.md", "政策", "# 政策\n\nB", library="l2")
    WikiBuilder(s).build("t1", "l1")
    WikiBuilder(s).build("t1", "l2")
    assert "A" in WikiStore(s).page("t1", "l1", "政策").body
    assert "B" in WikiStore(s).page("t1", "l2", "政策").body


def test_page_with_no_title_falls_back_to_filename():
    s = Store(":memory:")
    doc(s, "misc/notes.md", "", "# 随手记\n\n内容")
    WikiBuilder(s).build("t1", "lib")
    assert WikiStore(s).page("t1", "lib", "notes") is not None
