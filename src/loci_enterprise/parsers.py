"""Read the formats a company actually stores knowledge in.

Everything here is standard library. That is a deliberate constraint, not a
stylistic one: a governance layer has to be installable into an air-gapped
build with no wheel resolution, so every format that needs a native binary
(PDF with layout, OCR) is either declined with a clear message or handled by
an optional extra — never silently approximated.

Formats:
    text   .txt .md .markdown .rst .log
    data   .csv .tsv .json .jsonl
    web    .html .htm .mht
    office .docx .pptx .xlsx   (OOXML = zip + XML, no external tool)
    books  .epub
    pdf    → raises Unsupported (use an OCR/parse extra, do not fake it)

Office extraction is structure-aware where it is cheap: docx keeps heading
levels and table rows, pptx keeps one block per slide, xlsx keeps one block
per sheet with the header row intact. Flattening a spreadsheet into running
prose destroys the only thing that made it useful.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

TEXT = {".txt", ".md", ".markdown", ".rst", ".log", ".org", ".adoc"}
DATA = {".csv", ".tsv", ".json", ".jsonl"}
WEB = {".html", ".htm", ".mht", ".mhtml"}
OFFICE = {".docx", ".pptx", ".xlsx"}
BOOKS = {".epub"}
DECLINED = {".pdf": "PDF needs a layout-aware parser; install the pdf extra "
                     "(pymupdf) rather than indexing a lossy text guess.",
            ".doc": "Legacy binary .doc is not readable without a converter; "
                    "save as .docx.",
            ".xls": "Legacy binary .xls is not readable without a converter; "
                    "save as .xlsx.",
            ".ppt": "Legacy binary .ppt is not readable without a converter; "
                    "save as .pptx."}

SUPPORTED = TEXT | DATA | WEB | OFFICE | BOOKS


class Unsupported(Exception):
    pass


@dataclass
class Extracted:
    title: str
    text: str
    kind: str
    meta: dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------- html
class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style", "nav", "footer"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag in ("p", "div", "br", "tr"):
            self.parts.append("\n")
        elif tag in ("li",):
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style", "nav", "footer") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "tr", "table"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        self.parts.append(data)

    def result(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _html(path: Path) -> Extracted:
    parser = _HTMLText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = parser.result()
    return Extracted(title=parser.title.strip() or path.stem, text=text,
                     kind="html", meta={"chars": len(text)})


# --------------------------------------------------------------------- data
def _csv(path: Path, delimiter: str = ",") -> Extracted:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig",
                                                         errors="replace")),
                           delimiter=delimiter))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return Extracted(path.stem, "", "csv")
    header = [c.strip() for c in rows[0]]
    body = rows[1:]
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for r in body:
        cells = [(r[i] if i < len(r) else "").strip() for i in range(len(header))]
        lines.append(" | ".join(cells))
    return Extracted(path.stem, "\n".join(lines), "csv",
                     {"rows": len(body), "columns": len(header)})


def _json(path: Path) -> Extracted:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        data = json.loads(raw)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        kind = "json"
    except json.JSONDecodeError:
        # JSONL: one object per line, keep them readable
        objs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                objs.append({"line": line})
        text = "\n".join(json.dumps(o, ensure_ascii=False) for o in objs)
        kind = "jsonl"
    return Extracted(path.stem, text, kind, {"chars": len(text)})


# ------------------------------------------------------------------- office
_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _zip_text(zf: zipfile.ZipFile, name: str) -> ET.Element | None:
    try:
        return ET.fromstring(zf.read(name))
    except (KeyError, ET.ParseError):
        return None


def _local(tag: str) -> str:
    """Local part of a Clark-notation tag: {ns}p -> p."""
    return tag.rsplit("}", 1)[-1]


def _kids(el, name: str):
    return [c for c in list(el) if _local(c.tag) == name]


def _desc(el, name: str):
    return [e for e in el.iter() if _local(e.tag) == name]


def _attr(el, name: str, default: str = "") -> str:
    """Attribute by local name, regardless of the namespace it was written in."""
    for key, value in el.attrib.items():
        if _local(key) == name:
            return value
    return default


def _text(el) -> str:
    return "".join(t.text or "" for t in _desc(el, "t"))


def _docx(path: Path) -> Extracted:
    """Headings, paragraphs and table rows, namespace-prefix agnostic."""
    with zipfile.ZipFile(path) as zf:
        root = _zip_text(zf, "word/document.xml")
    if root is None:
        raise Unsupported("docx: word/document.xml missing or unreadable")
    body = next((e for e in root.iter() if _local(e.tag) == "body"), None)
    if body is None:
        raise Unsupported("docx: empty document body")
    lines: list[str] = []
    for el in list(body):
        tag = _local(el.tag)
        if tag == "p":
            text = _text(el).strip()
            if not text:
                continue
            level = 0
            for ppr in _kids(el, "pPr"):
                for style in _kids(ppr, "pStyle"):
                    m = re.search(r"(\d+)", _attr(style, "val"))
                    if m:
                        level = min(6, int(m.group(1)))
            lines.append(("#" * level + " " if level else "") + text)
        elif tag == "tbl":
            for row in _kids(el, "tr"):
                cells = [_text(cell).strip() for cell in _kids(row, "tc")]
                if any(cells):
                    lines.append(" | ".join(cells))
            lines.append("")
    text = "\n".join(lines).strip()
    return Extracted(path.stem, text, "docx", {"blocks": len(lines)})


def _pptx(path: Path) -> Extracted:
    slides: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist()
                 if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        for name in sorted(names, key=lambda n: int(re.search(r"(\d+)", n).group(1))):
            root = _zip_text(zf, name)
            if root is None:
                continue
            idx = int(re.search(r"(\d+)", name).group(1))
            body = "\n".join(t.strip() for t in
                             (e.text or "" for e in _desc(root, "t")) if t.strip())
            if body:
                slides.append((idx, body))
    text = "\n\n".join(f"## Slide {i}\n{body}" for i, body in slides)
    return Extracted(path.stem, text, "pptx", {"slides": len(slides)})


def _xlsx(path: Path) -> Extracted:
    """One block per sheet, header row intact. Handles default-namespace files."""
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        sroot = _zip_text(zf, "xl/sharedStrings.xml")
        if sroot is not None:
            for si in _desc(sroot, "si"):
                # join every <t> under the <si> (rich text splits a string)
                shared.append("".join((t.text or "") for t in _desc(si, "t")))

        names: list[str] = []
        wb = _zip_text(zf, "xl/workbook.xml")
        if wb is not None:
            names = [_attr(sheet, "name", f"Sheet{i + 1}")
                     for i, sheet in enumerate(_desc(wb, "sheet"))]

        sheet_files = [n for n in zf.namelist()
                       if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
        blocks: list[str] = []
        for i, name in enumerate(sorted(
                sheet_files, key=lambda n: int(re.search(r"(\d+)", n).group(1)))):
            root = _zip_text(zf, name)
            if root is None:
                continue
            rows: list[list[str]] = []
            for row in _desc(root, "row"):
                cells: list[str] = []
                for c in _kids(row, "c"):
                    ctype = _attr(c, "t")
                    v = next((x for x in _kids(c, "v")), None)
                    val = (v.text if v is not None else "") or ""
                    if ctype == "s" and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    elif ctype == "inlineStr":
                        val = "".join((t.text or "") for t in _desc(c, "t"))
                    cells.append(val.strip())
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            width = max(len(r) for r in rows)
            padded = [r + [""] * (width - len(r)) for r in rows]
            title = names[i] if i < len(names) else f"Sheet{i + 1}"
            blocks.append(f"## {title}\n" + "\n".join(" | ".join(r) for r in padded))
    return Extracted(path.stem, "\n\n".join(blocks), "xlsx", {"sheets": len(blocks)})


# -------------------------------------------------------------------- ebook
def _epub(path: Path) -> Extracted:
    with zipfile.ZipFile(path) as zf:
        html_names = [n for n in zf.namelist()
                      if n.lower().endswith((".xhtml", ".html", ".htm"))]
        # reading order: OPF spine if present, else archive order
        order: list[str] = []
        opf = next((n for n in zf.namelist() if n.lower().endswith(".opf")), None)
        if opf:
            root = _zip_text(zf, opf)
            if root is not None:
                base = opf.rsplit("/", 1)[0] + "/" if "/" in opf else ""
                manifest = {}
                for item in root.iter():
                    if item.tag.split("}")[-1] == "item":
                        manifest[item.get("id")] = item.get("href")
                for ref in root.iter():
                    if ref.tag.split("}")[-1] == "ref":
                        href = manifest.get(ref.get("idref"))
                        if href:
                            order.append(base + href)
        names = order or sorted(html_names)
        chunks = []
        for n in names:
            try:
                raw = zf.read(n).decode("utf-8", errors="replace")
            except KeyError:
                continue
            p = _HTMLText()
            p.feed(raw)
            t = p.result()
            if t.strip():
                chunks.append(t)
    text = "\n\n".join(chunks)
    return Extracted(path.stem, text, "epub", {"sections": len(chunks)})


# --------------------------------------------------------------------- main
def extract(path: str | Path) -> Extracted:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    if suffix in DECLINED:
        raise Unsupported(DECLINED[suffix])
    if suffix in TEXT:
        text = p.read_text(encoding="utf-8", errors="replace")
        return Extracted(p.stem, text, "text", {"chars": len(text)})
    if suffix in (".csv", ".tsv"):
        return _csv(p, "\t" if suffix == ".tsv" else ",")
    if suffix in DATA:
        return _json(p)
    if suffix in WEB:
        return _html(p)
    if suffix == ".docx":
        return _docx(p)
    if suffix == ".pptx":
        return _pptx(p)
    if suffix == ".xlsx":
        return _xlsx(p)
    if suffix in BOOKS:
        return _epub(p)
    raise Unsupported(f"{suffix or '(no extension)'} is not a supported format; "
                      f"supported: {', '.join(sorted(SUPPORTED))}")
