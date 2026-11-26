"""Connectors: getting a team's documents in, and keeping them current.

A connector is not a crawler. It is a contract with three obligations:

    list()    what exists now, with a stable key and a content hash
    fetch()   the current text of one item
    sync()    reconcile that against the store: add, update, skip, or prune

The third one is where most integrations quietly fail. If a page is deleted at
the source and the index keeps answering from it, the knowledge base now
contradicts the system people trust. ``SyncEngine`` therefore treats "vanished
from the source" as a deletion, and records a per-source watermark so an
interrupted run resumes instead of re-embedding everything.

Network access is urllib only, with a timeout and a descriptive User-Agent;
tests use ``file://`` URLs and a static connector so they never touch the
network.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from . import parsers
from .store import Document, Store, now_iso

UA = "loci-enterprise-connector/1.0 (+local knowledge base)"
STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS connector_state (
    source_key TEXT PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    library    TEXT NOT NULL,
    last_sync  TEXT NOT NULL,
    items      INTEGER NOT NULL DEFAULT 0,
    cursor     TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class SourceDoc:
    key: str                     # stable identity inside the connector
    title: str
    text: str
    mtime: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


class Connector:
    name = "connector"
    source_key = ""

    def list(self) -> list[SourceDoc]:
        raise NotImplementedError


# ------------------------------------------------------------------- local
class LocalDirConnector(Connector):
    """Every supported file under a directory, recursively."""

    def __init__(self, root: str | Path, *, name: str = "local",
                 recursive: bool = True, max_file_bytes: int = 32 << 20):
        self.root = Path(root).expanduser()
        self.name = name
        self.source_key = f"local:{self.root}"
        self.recursive = recursive
        self.max_file_bytes = max_file_bytes

    def list(self) -> list[SourceDoc]:
        if not self.root.is_dir():
            raise FileNotFoundError(str(self.root))
        pattern = "**/*" if self.recursive else "*"
        out: list[SourceDoc] = []
        for p in sorted(self.root.glob(pattern)):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() not in parsers.SUPPORTED:
                continue
            if p.stat().st_size > self.max_file_bytes:
                continue
            try:
                doc = parsers.extract(p)
            except parsers.Unsupported:
                continue
            rel = p.relative_to(self.root).as_posix()
            out.append(SourceDoc(key=rel, title=doc.title or p.stem, text=doc.text,
                                 mtime=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime(p.stat().st_mtime)),
                                 meta={"kind": doc.kind, **doc.meta}))
        return out


class StaticConnector(Connector):
    """An in-memory connector — for tests and for handing over a batch."""

    def __init__(self, docs: list[SourceDoc], *, name: str = "static"):
        self._docs = list(docs)
        self.name = name
        self.source_key = f"static:{name}"

    def list(self) -> list[SourceDoc]:
        return list(self._docs)


# -------------------------------------------------------------------- http
def _get(url: str, timeout: float = 20.0, max_bytes: int = 8 << 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{url}: response exceeds {max_bytes} bytes")
    return data


def _decode(data: bytes, content_type: str = "") -> str:
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    for enc in ([m.group(1)] if m else []) + ["utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


class WebPageConnector(Connector):
    """One or more pages, each becoming a document."""

    def __init__(self, urls: list[str], *, name: str = "web"):
        self.urls = list(urls)
        self.name = name
        self.source_key = f"web:{name}"

    def list(self) -> list[SourceDoc]:
        out = []
        for url in self.urls:
            try:
                resp = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": UA}),
                    timeout=20)
                data = resp.read(8 << 20)
                ctype = resp.headers.get("Content-Type", "")
            except Exception as exc:              # one bad page must not kill the run
                out.append(SourceDoc(key=url, title=url, text="",
                                     meta={"error": str(exc)}))
                continue
            html = _decode(data, ctype)
            p = parsers._HTMLText()
            p.feed(html)
            out.append(SourceDoc(key=url, title=p.title.strip() or url,
                                 text=p.result(), meta={"url": url}))
        return [d for d in out if d.text.strip()]


class RssConnector(Connector):
    """RSS 2.0 and Atom, each entry becoming one document."""

    def __init__(self, url: str, *, name: str = "rss", fetch_full: bool = False):
        self.url = url
        self.name = name
        self.source_key = f"rss:{url}"
        self.fetch_full = fetch_full

    @staticmethod
    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def list(self) -> list[SourceDoc]:
        root = ET.fromstring(_get(self.url))
        out: list[SourceDoc] = []
        for item in root.iter():
            if self._local(item.tag) not in ("item", "entry"):
                continue
            fields: dict[str, str] = {}
            link = ""
            for child in item:
                tag = self._local(child.tag)
                if tag == "link":
                    link = child.get("href") or (child.text or "") or link
                elif tag in ("title", "description", "summary", "content",
                             "encoded", "pubDate", "updated", "published", "id"):
                    fields[tag] = "".join(child.itertext()).strip()
            title = fields.get("title") or link or "(untitled)"
            body = (fields.get("content") or fields.get("encoded")
                    or fields.get("description") or fields.get("summary") or "")
            body = re.sub(r"<[^>]+>", " ", body)          # strip HTML from feed bodies
            body = re.sub(r"\s+", " ", body).strip()
            if self.fetch_full and link:
                try:
                    page = WebPageConnector([link]).list()
                    if page:
                        body, title = page[0].text, page[0].title or title
                except Exception:
                    pass                                  # fall back to the summary
            out.append(SourceDoc(key=link or title, title=title, text=body,
                                 mtime=fields.get("updated") or fields.get("pubDate") or "",
                                 meta={"link": link, "feed": self.url}))
        return out


class SitemapConnector(Connector):
    """Expand a sitemap.xml (or sitemap index) into page documents."""

    def __init__(self, url: str, *, name: str = "sitemap", limit: int = 200):
        self.url = url
        self.name = name
        self.source_key = f"sitemap:{url}"
        self.limit = limit

    def list(self) -> list[SourceDoc]:
        root = ET.fromstring(_get(self.url))
        urls = [e.text.strip() for e in root.iter() if e.tag.endswith("loc")
                and e.text and e.text.strip().startswith(("http://", "https://",
                                                          "file://"))]
        pages = WebPageConnector(urls[: self.limit], name=self.name).list()
        return pages


class JsonApiConnector(Connector):
    """A REST endpoint that returns JSON, mapped by configuration.

    Most enterprise systems expose "list of records as JSON" behind an API:
    Confluence pages, a Feishu document list, an internal CMS. Writing a
    connector per vendor is mostly repetition of the same five decisions, so
    this one makes them configuration:

        url          GET endpoint
        items_path   dot path to the array ("" = the response IS the array)
        id_field / title_field / body_field
        headers      auth headers (caller supplies; never logged)
        max_items    hard cap so a misconfigured path cannot ingest a database

    Field values may themselves be dot paths, so ``content.body`` works.
    """

    def __init__(self, url: str, *, name: str = "api", items_path: str = "items",
                 id_field: str = "id", title_field: str = "title",
                 body_field: str = "body", headers: dict | None = None,
                 max_items: int = 1000):
        self.url = url
        self.name = name
        self.source_key = f"api:{url}"
        self.items_path = items_path
        self.id_field = id_field
        self.title_field = title_field
        self.body_field = body_field
        self.headers = headers or {}
        self.max_items = max_items

    @staticmethod
    def _dig(obj, path: str):
        cur = obj
        for part in [p for p in (path or "").split(".") if p]:
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list) and part.isdigit():
                idx = int(part)
                cur = cur[idx] if idx < len(cur) else None
            else:
                return None
        return cur

    def list(self) -> list[SourceDoc]:
        req = urllib.request.Request(self.url, headers={"User-Agent": UA,
                                                        **self.headers})
        with urllib.request.urlopen(req, timeout=30) as resp:
            # file:// responses carry no status; urllib has already raised for
            # a real HTTP error by this point
            status = getattr(resp, "status", None)
            if status is not None and status >= 400:
                raise ValueError(f"{self.url}: HTTP {status}")
            payload = json.loads(resp.read(16 << 20).decode("utf-8", errors="replace"))
        items = self._dig(payload, self.items_path)
        if items is None and not self.items_path:
            items = payload
        if not isinstance(items, list):
            raise ValueError(f"{self.url}: items_path {self.items_path!r} did not "
                             f"yield a list (got {type(items).__name__})")
        out: list[SourceDoc] = []
        for raw in items[: self.max_items]:
            if not isinstance(raw, dict):
                continue
            ident = self._dig(raw, self.id_field)
            title = self._dig(raw, self.title_field)
            body = self._dig(raw, self.body_field)
            if body is None:
                body = ""
            if not isinstance(body, str):
                body = json.dumps(body, ensure_ascii=False)
            out.append(SourceDoc(key=str(ident or title or len(out)),
                                 title=str(title or ident or "(untitled)"),
                                 text=body,
                                 meta={"source": self.url}))
        return out


# -------------------------------------------------------------------- sync
@dataclass
class SyncReport:
    source_key: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    pruned: int = 0
    errors: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"source_key": self.source_key, "added": self.added,
                "updated": self.updated, "unchanged": self.unchanged,
                "pruned": self.pruned, "errors": self.errors}


class SyncEngine:
    """Reconcile a connector against the store, incrementally and safely."""

    def __init__(self, store: Store):
        self.store = store
        self.store.conn.executescript(STATE_SCHEMA)

    def sync(self, connector: Connector, *, tenant_id: str, library: str,
             acl: list[str] | None = None, actor: str = "") -> SyncReport:
        acl = acl or ["public"]
        report = SyncReport(connector.source_key)
        try:
            docs = connector.list()
        except Exception as exc:
            report.errors.append({"stage": "list", "error": str(exc)})
            return report

        live_keys: set[str] = set()
        for src in docs:
            if not src.text.strip():
                if src.meta.get("error"):
                    report.errors.append({"key": src.key, "error": src.meta["error"]})
                continue
            path = f"{connector.name}/{src.key}" if not src.key.startswith(
                f"{connector.name}/") else src.key
            live_keys.add(path)
            existing = self.store.get_document(tenant_id, library, path)
            if existing is not None and existing.content_hash == _digest(src.text) \
                    and existing.title == src.title:
                report.unchanged += 1
                continue
            doc = Document(id="", tenant_id=tenant_id, library=library, path=path,
                           title=src.title, body=src.text, acl=acl,
                           written_by=actor or f"connector:{connector.name}",
                           owner=actor)
            self.store.put_document(doc, note=f"sync:{connector.name}")
            if existing is None:
                report.added += 1
            else:
                report.updated += 1

        # vanished at the source -> leave the active index
        prefix = f"{connector.name}/"
        for doc in self.store.list_documents(tenant_id, library):
            if doc.path.startswith(prefix) and doc.path not in live_keys:
                self.store.set_status(doc.id, "retired", actor=actor or "sync",
                                      reason=f"removed at source ({connector.name})")
                report.pruned += 1

        self.store.conn.execute(
            "INSERT INTO connector_state(source_key, tenant_id, library, last_sync, items)"
            " VALUES (?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET"
            " last_sync=excluded.last_sync, items=excluded.items",
            (connector.source_key, tenant_id, library, now_iso(),
             report.added + report.updated + report.unchanged))
        return report

    def state(self, source_key: str | None = None) -> list[dict]:
        if source_key:
            rows = self.store.conn.execute(
                "SELECT * FROM connector_state WHERE source_key=?",
                (source_key,)).fetchall()
        else:
            rows = self.store.conn.execute(
                "SELECT * FROM connector_state ORDER BY last_sync DESC").fetchall()
        return [dict(r) for r in rows]


def _digest(text: str) -> str:
    from .store import digest
    return digest(text)
