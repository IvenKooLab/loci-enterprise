"""Wiki: compile documents into pages people can read, and keep the history.

A wiki built by a language model is pleasant and unaccountable. This one is
compiled deterministically from the corpus, which buys three things a
generated wiki cannot guarantee:

* **every statement has a source** — each section carries the document paths
  it was built from, and the page keeps that list;
* **the same input produces the same page**, so a diff means the corpus
  changed, not that the model felt different today;
* **rollback is exact** — pages are versioned, and restoring v3 produces a new
  v4 with v3's bytes.

Deduplication is content-based: rebuilding an unchanged corpus creates no new
revision. A link that points at a page that does not exist is reported, not
silently rendered as text.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from . import lifecycle
from .store import Store, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS wiki_pages (
    tenant_id  TEXT NOT NULL,
    library    TEXT NOT NULL,
    slug       TEXT NOT NULL,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    citations  TEXT NOT NULL DEFAULT '[]',
    version    INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, library, slug)
);
CREATE TABLE IF NOT EXISTS wiki_revisions (
    tenant_id  TEXT NOT NULL,
    library    TEXT NOT NULL,
    slug       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    citations  TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant_id, library, slug, version)
);
"""

_LINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
_HEADING = re.compile(r"^(#{2,4})\s+(.*)$")


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", (text or "").strip().lower())
    return s.strip("-") or "untitled"


@dataclass
class WikiPage:
    slug: str
    title: str
    body: str
    citations: list[str] = field(default_factory=list)
    version: int = 1
    updated_at: str = ""

    @property
    def content_hash(self) -> str:
        return hashlib.sha256((self.title + "\n" + self.body).encode()).hexdigest()[:16]

    def links(self) -> list[str]:
        return [slugify(m.group(1)) for m in _LINK.finditer(self.body)]


@dataclass
class BuildReport:
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    dead_links: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"created": self.created, "updated": self.updated,
                "unchanged": self.unchanged, "skipped": self.skipped,
                "dead_links": self.dead_links}


class WikiStore:
    def __init__(self, store: Store):
        self.store = store
        self.store.conn.executescript(SCHEMA)

    # ---------------------------------------------------------------- reads
    def page(self, tenant_id: str, library: str, slug: str) -> WikiPage | None:
        row = self.store.conn.execute(
            "SELECT * FROM wiki_pages WHERE tenant_id=? AND library=? AND slug=?",
            (tenant_id, library, slug)).fetchone()
        if row is None:
            return None
        return WikiPage(slug=row["slug"], title=row["title"], body=row["body"],
                        citations=_json_list(row["citations"]), version=row["version"],
                        updated_at=row["updated_at"])

    def pages(self, tenant_id: str, library: str) -> list[WikiPage]:
        return [WikiPage(slug=r["slug"], title=r["title"], body=r["body"],
                         citations=_json_list(r["citations"]), version=r["version"],
                         updated_at=r["updated_at"])
                for r in self.store.conn.execute(
                    "SELECT * FROM wiki_pages WHERE tenant_id=? AND library=?"
                    " ORDER BY title", (tenant_id, library)).fetchall()]

    def revisions(self, tenant_id: str, library: str, slug: str) -> list[dict]:
        return [dict(r) for r in self.store.conn.execute(
            "SELECT version, title, created_at, note, length(body) AS chars"
            " FROM wiki_revisions WHERE tenant_id=? AND library=? AND slug=?"
            " ORDER BY version", (tenant_id, library, slug)).fetchall()]

    def diff(self, tenant_id: str, library: str, slug: str,
             a: int, b: int) -> dict:
        rows = {r["version"]: r for r in self.store.conn.execute(
            "SELECT * FROM wiki_revisions WHERE tenant_id=? AND library=? AND slug=?"
            " AND version IN (?,?)", (tenant_id, library, slug, a, b)).fetchall()}
        if a not in rows or b not in rows:
            raise LookupError(f"missing revision {a} or {b}")
        old, new = rows[a]["body"].splitlines(), rows[b]["body"].splitlines()
        import difflib
        return {"from": a, "to": b,
                "unified": "\n".join(difflib.unified_diff(
                    old, new, fromfile=f"v{a}", tofile=f"v{b}", lineterm=""))}

    # --------------------------------------------------------------- writes
    def put(self, tenant_id: str, library: str, page: WikiPage,
            note: str = "") -> str:
        """Insert or update; identical content is a no-op. Returns the action."""
        existing = self.page(tenant_id, library, page.slug)
        if existing is not None and existing.content_hash == page.content_hash:
            return "unchanged"
        version = (existing.version + 1) if existing else 1
        now = now_iso()
        self.store.conn.execute(
            "INSERT INTO wiki_pages(tenant_id, library, slug, title, body, citations,"
            " version, updated_at) VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(tenant_id, library, slug) DO UPDATE SET"
            " title=excluded.title, body=excluded.body, citations=excluded.citations,"
            " version=excluded.version, updated_at=excluded.updated_at",
            (tenant_id, library, page.slug, page.title, page.body,
             _json_dump(page.citations), version, now))
        self.store.conn.execute(
            "INSERT INTO wiki_revisions(tenant_id, library, slug, version, title, body,"
            " citations, created_at, note) VALUES (?,?,?,?,?,?,?,?,?)",
            (tenant_id, library, page.slug, version, page.title, page.body,
             _json_dump(page.citations), now, note))
        self.store.audit(tenant_id, "wiki", f"wiki:{'update' if existing else 'create'}",
                         f"{library}/{page.slug}", f"v{version} {note}".strip())
        return "updated" if existing else "created"

    def rollback(self, tenant_id: str, library: str, slug: str, version: int,
                 *, actor: str = "wiki") -> WikiPage:
        row = self.store.conn.execute(
            "SELECT * FROM wiki_revisions WHERE tenant_id=? AND library=? AND slug=?"
            " AND version=?", (tenant_id, library, slug, version)).fetchone()
        if row is None:
            raise LookupError(f"no revision v{version} for {slug}")
        page = WikiPage(slug=slug, title=row["title"], body=row["body"],
                        citations=_json_list(row["citations"]))
        # put() will create a *new* revision carrying the old bytes
        self.put(tenant_id, library, page, note=f"rollback to v{version} by {actor}")
        return self.page(tenant_id, library, slug)

    # ----------------------------------------------------------- link check
    def dead_links(self, tenant_id: str, library: str) -> list[dict]:
        known = {p.slug for p in self.pages(tenant_id, library)}
        out = []
        for page in self.pages(tenant_id, library):
            for target in page.links():
                if target and target not in known:
                    out.append({"from": page.slug, "to": target})
        return out


class WikiBuilder:
    """Compile the active corpus into one page per topic."""

    def __init__(self, store: Store, wiki: WikiStore | None = None):
        self.store = store
        self.wiki = wiki or WikiStore(store)

    def build(self, tenant_id: str, library: str, *,
              note: str = "") -> BuildReport:
        report = BuildReport()
        groups: dict[str, list] = {}
        for doc in self.store.list_documents(tenant_id, library):
            verdict = lifecycle.evaluate(doc)
            if verdict.status not in ("active", "stale"):
                report.skipped.append(doc.path)
                continue
            fallback = doc.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            key = slugify(doc.title or fallback)
            groups.setdefault(key, []).append((verdict.weight, doc))

        for slug, items in sorted(groups.items()):
            # the most authoritative live document leads; others become sources
            items.sort(key=lambda pair: (-pair[0], pair[1].path))
            lead = items[0][1]
            body = self._compose(lead, [d for _, d in items])
            fallback_title = lead.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            page = WikiPage(slug=slug, title=lead.title or fallback_title, body=body,
                            citations=[d.path for _, d in items])
            action = self.wiki.put(tenant_id, library, page,
                                   note=note or f"compiled from {len(items)} doc(s)")
            getattr(report, action).append(slug)

        report.dead_links = self.wiki.dead_links(tenant_id, library)
        return report

    @staticmethod
    def _compose(lead, sources) -> str:
        """Lead document, plus a See-also list when the topic has siblings.

        The body is the document's own markdown: compiled, not paraphrased, so
        nothing here can invent a fact the corpus does not contain.
        """
        body = lead.body.strip()
        if len(sources) > 1:
            others = [d for d in sources if d.path != lead.path]
            if others:
                body += "\n\n## Sources\n" + "\n".join(
                    f"- [[{slugify(d.title or d.path)}]] — `{d.path}`"
                    for d in others)
        return body


def _json_list(text: str) -> list[str]:
    import json
    try:
        v = json.loads(text or "[]")
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _json_dump(value: list[str]) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)
