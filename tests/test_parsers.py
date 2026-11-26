"""Parsers read real containers without third-party libraries."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from loci_enterprise import parsers

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


# ------------------------------------------------------------------ simple
def test_text_and_markdown(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# 标题\n\n正文内容", encoding="utf-8")
    doc = parsers.extract(p)
    assert doc.kind == "text" and "正文内容" in doc.text


def test_csv_becomes_a_pipe_table(tmp_path):
    p = tmp_path / "budget.csv"
    p.write_text("城市,住宿上限\n上海,800\n北京,700\n", encoding="utf-8")
    doc = parsers.extract(p)
    assert doc.kind == "csv"
    assert "城市 | 住宿上限" in doc.text
    assert "上海 | 800" in doc.text
    assert doc.meta["rows"] == 2


def test_json_is_pretty_printed(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1, "b": ["c"]}), encoding="utf-8")
    doc = parsers.extract(p)
    assert doc.kind == "json" and '"a": 1' in doc.text


def test_jsonl_falls_back_line_by_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\nnot-json\n{"b":2}\n', encoding="utf-8")
    doc = parsers.extract(p)
    assert doc.kind == "jsonl"
    assert '"a": 1' in doc.text and "not-json" in doc.text


def test_html_keeps_headings_lists_and_title(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("""<html><head><title>政策页</title></head><body>
      <script>ignore()</script>
      <h1>差旅</h1><p>住宿上限 800 元</p><ul><li>提前 5 天申请</li></ul>
    </body></html>""", encoding="utf-8")
    doc = parsers.extract(p)
    assert doc.title == "政策页"
    assert "# 差旅" in doc.text
    assert "- 提前 5 天申请" in doc.text
    assert "ignore()" not in doc.text


# ------------------------------------------------------------------ office
def _zip(tmp_path, name, files: dict[str, str]) -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        for arc, content in files.items():
            zf.writestr(arc, content)
    return p


def test_docx_reads_headings_paragraphs_and_tables(tmp_path):
    body = f"""<?xml version="1.0"?>
<w:document xmlns:w="{W}"><w:body>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>差旅政策</w:t></w:r></w:p>
  <w:p><w:r><w:t>住宿上限 800 元。</w:t></w:r></w:p>
  <w:tbl>
    <w:tr><w:tc><w:p><w:r><w:t>城市</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>上限</w:t></w:r></w:p></w:tc></w:tr>
    <w:tr><w:tc><w:p><w:r><w:t>上海</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>800</w:t></w:r></w:p></w:tc></w:tr>
  </w:tbl>
</w:body></w:document>"""
    p = _zip(tmp_path, "policy.docx", {"word/document.xml": body})
    doc = parsers.extract(p)
    assert doc.kind == "docx"
    assert "# 差旅政策" in doc.text
    assert "住宿上限 800 元。" in doc.text
    assert "城市 | 上限" in doc.text and "上海 | 800" in doc.text


def test_pptx_keeps_slide_boundaries(tmp_path):
    slide = f"""<?xml version="1.0"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="{A}"><p:cSld><p:spTree>
  <p:sp><p:txBody><a:p><a:r><a:t>第一页标题</a:t></a:r></a:p>
             <a:p><a:r><a:t>要点一</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>"""
    p = _zip(tmp_path, "deck.pptx", {"ppt/slides/slide1.xml": slide})
    doc = parsers.extract(p)
    assert doc.kind == "pptx"
    assert "## Slide 1" in doc.text
    assert "第一页标题" in doc.text and "要点一" in doc.text


def test_xlsx_resolves_shared_strings_per_sheet(tmp_path):
    shared = f"""<?xml version="1.0"?><s:sst xmlns:s="{S}">
      <s:si><t>城市</t></s:si><s:si><t>上限</t></s:si>
      <s:si><t>上海</t></s:si><s:si><t>800</t></s:si></s:sst>"""
    sheet = f"""<?xml version="1.0"?><worksheet xmlns:s="{S}"><s:sheetData>
      <s:row r="1"><s:c t="s"><s:v>0</s:v></s:c><s:c t="s"><s:v>1</s:v></s:c></s:row>
      <s:row r="2"><s:c t="s"><s:v>2</s:v></s:c><s:c t="s"><s:v>3</s:v></s:c></s:row>
    </s:sheetData></worksheet>"""
    workbook = f"""<?xml version="1.0"?><s:workbook xmlns:s="{S}">
      <s:sheets><s:sheet name="差旅" sheetId="1"/></s:sheets></s:workbook>"""
    p = _zip(tmp_path, "budget.xlsx", {
        "xl/sharedStrings.xml": shared,
        "xl/workbook.xml": workbook,
        "xl/worksheets/sheet1.xml": sheet})
    doc = parsers.extract(p)
    assert doc.kind == "xlsx"
    assert "## 差旅" in doc.text
    assert "城市 | 上限" in doc.text and "上海 | 800" in doc.text


def test_malformed_docx_raises_unsupported(tmp_path):
    p = _zip(tmp_path, "broken.docx", {"word/other.xml": "<x/>"})
    with pytest.raises(parsers.Unsupported):
        parsers.extract(p)


# -------------------------------------------------------------------- epub
def test_epub_follows_spine_order(tmp_path):
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
 <rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>"""
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
 <manifest>
  <item id="c2" href="ch2.xhtml"/>
  <item id="c1" href="ch1.xhtml"/>
 </manifest>
 <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"""
    p = _zip(tmp_path, "book.epub", {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": container,
        "OEBPS/content.opf": opf,
        "OEBPS/ch1.xhtml": "<html><body><h1>第一章</h1><p>甲</p></body></html>",
        "OEBPS/ch2.xhtml": "<html><body><h1>第二章</h1><p>乙</p></body></html>",
    })
    doc = parsers.extract(p)
    assert doc.kind == "epub"
    assert doc.text.index("第一章") < doc.text.index("第二章")


# ----------------------------------------------------------------- declined
def test_pdf_is_declined_with_a_reason_not_silently_empty(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(parsers.Unsupported) as exc:
        parsers.extract(p)
    assert "pdf" in str(exc.value).lower()


def test_unknown_extension_lists_what_is_supported(tmp_path):
    p = tmp_path / "x.zzz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(parsers.Unsupported):
        parsers.extract(p)


def test_missing_file_is_a_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        parsers.extract(tmp_path / "nope.md")


def test_xlsx_with_default_namespace_is_read(tmp_path):
    """Real spreadsheetml files declare a default namespace, not a prefix.

    This is the shape Excel and most generators emit; a parser that only
    understood prefixed tags would silently return zero sheets.
    """
    shared = f"""<?xml version="1.0"?><sst xmlns="{S}">
      <si><t>城市</t></si><si><t>上限</t></si><si><t>上海</t></si><si><t>800</t></si>
    </sst>"""
    sheet = f"""<?xml version="1.0"?><worksheet xmlns="{S}"><sheetData>
      <row r="1"><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>
      <row r="2"><c t="s"><v>2</v></c><c t="s"><v>3</v></c></row>
    </sheetData></worksheet>"""
    workbook = f"""<?xml version="1.0"?><workbook xmlns="{S}">
      <sheets><sheet name="差旅" sheetId="1"/></sheets></workbook>"""
    p = _zip(tmp_path, "default.xlsx", {
        "xl/sharedStrings.xml": shared,
        "xl/workbook.xml": workbook,
        "xl/worksheets/sheet1.xml": sheet})
    doc = parsers.extract(p)
    assert "## 差旅" in doc.text
    assert "城市 | 上限" in doc.text and "上海 | 800" in doc.text


def test_docx_rich_text_runs_are_joined(tmp_path):
    body = f"""<?xml version="1.0"?>
<w:document xmlns:w="{W}"><w:body>
  <w:p><w:r><w:t>上限</w:t></w:r><w:r><w:t> 800 元</w:t></w:r></w:p>
</w:body></w:document>"""
    p = _zip(tmp_path, "runs.docx", {"word/document.xml": body})
    assert "上限 800 元" in parsers.extract(p).text
