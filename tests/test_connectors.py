"""Connectors and incremental reconciliation."""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest

from loci_enterprise import connectors
from loci_enterprise.connectors import (LocalDirConnector, RssConnector,
                                        SourceDoc, StaticConnector, SyncEngine)
from loci_enterprise.store import Store


# ------------------------------------------------------------------ local
def test_local_connector_reads_supported_files_only(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# A\n\n内容", encoding="utf-8")
    (tmp_path / "notes" / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "notes" / "skip.exe").write_bytes(b"MZ")
    docs = LocalDirConnector(tmp_path / "notes").list()
    assert {d.key for d in docs} == {"a.md", "b.csv"}


def test_local_connector_keeps_subdirectories_as_keys(tmp_path):
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "policy.md").write_text("# 政策", encoding="utf-8")
    docs = LocalDirConnector(tmp_path).list()
    assert docs[0].key == "hr/policy.md"


def test_local_connector_raises_for_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalDirConnector(tmp_path / "nope").list()


# ------------------------------------------------------------------- sync
def test_sync_adds_then_skips_unchanged(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nv1", encoding="utf-8")
    store = Store(":memory:")
    engine = SyncEngine(store)
    conn = LocalDirConnector(notes)

    first = engine.sync(conn, tenant_id="t1", library="lib")
    assert (first.added, first.updated, first.unchanged) == (1, 0, 0)
    second = engine.sync(conn, tenant_id="t1", library="lib")
    assert (second.added, second.updated, second.unchanged) == (0, 0, 1)


def test_sync_updates_changed_file(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    p = notes / "a.md"
    p.write_text("# A\n\nv1", encoding="utf-8")
    store, engine, conn = Store(":memory:"), None, None
    engine = SyncEngine(store)
    conn = LocalDirConnector(notes)
    engine.sync(conn, tenant_id="t1", library="lib")
    p.write_text("# A\n\nv2", encoding="utf-8")
    report = engine.sync(conn, tenant_id="t1", library="lib")
    assert report.updated == 1
    doc = store.get_document("t1", "lib", "local/a.md")
    assert "v2" in doc.body


def test_sync_retires_documents_that_vanished_at_source(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    a, b = notes / "a.md", notes / "b.md"
    a.write_text("# A\n\nx", encoding="utf-8")
    b.write_text("# B\n\ny", encoding="utf-8")
    store = Store(":memory:")
    engine = SyncEngine(store)
    conn = LocalDirConnector(notes)
    engine.sync(conn, tenant_id="t1", library="lib")
    b.unlink()
    report = engine.sync(conn, tenant_id="t1", library="lib")
    assert report.pruned == 1
    assert store.get_document("t1", "lib", "local/b.md").status == "retired"


def test_sync_does_not_prune_other_connectors_documents(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nx", encoding="utf-8")
    store = Store(":memory:")
    engine = SyncEngine(store)
    engine.sync(StaticConnector([SourceDoc("keep.md", "Keep", "stable")],
                                name="other"), tenant_id="t1", library="lib")
    engine.sync(LocalDirConnector(notes), tenant_id="t1", library="lib")
    assert store.get_document("t1", "lib", "other/keep.md").status == "active"


def test_sync_records_a_watermark(tmp_path):
    store = Store(":memory:")
    engine = SyncEngine(store)
    engine.sync(StaticConnector([SourceDoc("a", "A", "text")]),
                tenant_id="t1", library="lib")
    state = engine.state()
    assert state[0]["tenant_id"] == "t1" and state[0]["last_sync"]


def test_sync_reports_list_errors_without_raising():
    class Broken(connectors.Connector):
        name = "broken"
        source_key = "broken:1"

        def list(self):
            raise RuntimeError("network down")

    report = SyncEngine(Store(":memory:")).sync(
        Broken(), tenant_id="t1", library="lib")
    assert report.errors[0]["stage"] == "list"
    assert report.added == 0


def test_empty_documents_are_skipped_not_indexed():
    store = Store(":memory:")
    SyncEngine(store).sync(StaticConnector([SourceDoc("a", "A", "   ")]),
                           tenant_id="t1", library="lib")
    assert store.counts("t1")["documents"] == 0


# -------------------------------------------------------------------- rss
RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Policy feed</title>
 <item><title>差旅政策更新</title><link>https://example.com/a</link>
       <description>&lt;p&gt;住宿上限改为 800 元&lt;/p&gt;</description>
       <pubDate>Tue, 25 Nov 2026 10:00:00 GMT</pubDate></item>
 <item><title>年假调整</title><link>https://example.com/b</link>
       <description>15 天</description></item>
</channel></rss>"""


def test_rss_connector_parses_items_and_strips_html(tmp_path):
    p = tmp_path / "feed.xml"
    p.write_text(RSS, encoding="utf-8")
    docs = RssConnector(p.as_uri()).list()
    assert [d.title for d in docs] == ["差旅政策更新", "年假调整"]
    assert "住宿上限改为 800 元" in docs[0].text
    assert "<p>" not in docs[0].text
    assert docs[0].key == "https://example.com/a"


ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>t</title>
 <entry><title>Entry A</title><link href="https://example.com/x"/>
        <updated>2026-11-25T10:00:00Z</updated>
        <summary>Summary text</summary></entry>
</feed>"""


def test_atom_entries_are_supported(tmp_path):
    p = tmp_path / "atom.xml"
    p.write_text(ATOM, encoding="utf-8")
    docs = RssConnector(p.as_uri()).list()
    assert docs[0].title == "Entry A" and docs[0].text == "Summary text"


def test_rss_feeds_that_are_not_xml_raise_cleanly(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text("not xml", encoding="utf-8")
    with pytest.raises(Exception):
        RssConnector(p.as_uri()).list()


# -------------------------------------------------------------------- web
def test_web_page_connector_extracts_text(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><head><title>页面</title></head><body><h1>标题</h1>"
                 "<p>正文</p></body></html>", encoding="utf-8")
    docs = connectors.WebPageConnector([p.as_uri()]).list()
    assert docs[0].title == "页面"
    assert "# 标题" in docs[0].text and "正文" in docs[0].text


def test_web_page_connector_keeps_going_when_one_page_fails(tmp_path):
    good = tmp_path / "ok.html"
    good.write_text("<html><body><p>ok</p></body></html>", encoding="utf-8")
    docs = connectors.WebPageConnector([good.as_uri(),
                                        (tmp_path / "missing.html").as_uri()]).list()
    assert len(docs) == 1 and "ok" in docs[0].text


def test_sitemap_expands_to_pages(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><head><title>S</title></head><body>正文</body></html>",
                    encoding="utf-8")
    sm = tmp_path / "sitemap.xml"
    sm.write_text(f"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{page.as_uri()}</loc></url>
</urlset>""", encoding="utf-8")
    docs = connectors.SitemapConnector(sm.as_uri()).list()
    assert len(docs) == 1 and "正文" in docs[0].text


# --------------------------------------------------------------- json api
def test_json_api_connector_maps_configured_fields(tmp_path):
    payload = {"data": {"pages": [
        {"page_id": "p1", "name": "差旅政策", "content": {"body": "住宿上限 800 元"}},
        {"page_id": "p2", "name": "年假", "content": {"body": "15 天"}},
    ]}}
    p = tmp_path / "api.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    conn = connectors.JsonApiConnector(
        p.as_uri(), items_path="data.pages", id_field="page_id",
        title_field="name", body_field="content.body")
    docs = conn.list()
    assert [d.key for d in docs] == ["p1", "p2"]
    assert docs[0].title == "差旅政策" and "800" in docs[0].text


def test_json_api_connector_rejects_a_non_list_items_path(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"items": {"not": "a list"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        connectors.JsonApiConnector(p.as_uri()).list()


def test_json_api_connector_honours_max_items(tmp_path):
    p = tmp_path / "many.json"
    p.write_text(json.dumps({"items": [{"id": i, "body": "x"} for i in range(10)]}),
                 encoding="utf-8")
    assert len(connectors.JsonApiConnector(p.as_uri(), max_items=3).list()) == 3


def test_json_api_connector_accepts_a_bare_array(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text(json.dumps([{"id": "a", "title": "A", "body": "text"}]),
                 encoding="utf-8")
    docs = connectors.JsonApiConnector(p.as_uri(), items_path="").list()
    assert docs[0].key == "a"


def test_json_api_connector_serialises_non_string_bodies(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"items": [{"id": "a", "body": {"nested": 1}}]}),
                 encoding="utf-8")
    assert '"nested": 1' in connectors.JsonApiConnector(p.as_uri()).list()[0].text
