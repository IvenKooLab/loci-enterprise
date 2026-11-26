"""Chunking strategies, because one size does not fit every corpus.

The default heading-aware chunker is right for prose organised under headings.
Two other shapes matter in practice:

* **Parent-child** — retrieval matches small children (precise), but the answer
  gets the whole parent section (enough context). This is how you avoid the
  classic failure where a 200-character chunk wins the match and contains no
  usable answer.
* **Fixed window** — for material with no headings at all: transcripts, OCR
  output, exported chat logs.

Every strategy returns the same structure, ``Chunk`` records with an optional
``parent_id``, so the store and the retriever do not care which one produced
them. ``rebuild_library`` applies a strategy to an existing library, preserving
ACL rows and document identity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import chunk_document
from .store import Store

MAX_CHARS = 700
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    ord: int
    parent_id: str | None = None
    level: str = "parent"          # parent | child
    heading_path: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def heading_chunks(body: str, title: str = "") -> list[Chunk]:
    """The default: one chunk per heading section (delegates to chunking)."""
    return [Chunk(text=t, ord=i, level="parent")
            for i, t in enumerate(chunk_document(body, title))]


def fixed_window(body: str, title: str = "", *, size: int = MAX_CHARS,
                 overlap: int = 80) -> list[Chunk]:
    """No headings required; windows overlap so a cut sentence is recoverable."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = re.sub(r"\s+", " ", body or "").strip()
    if not text:
        return []
    step = size - overlap
    out: list[Chunk] = []
    for i, start in enumerate(range(0, max(1, len(text)), step)):
        piece = text[start:start + size]
        if not piece.strip():
            break
        out.append(Chunk(text=piece, ord=i, level="parent"))
        if start + size >= len(text):
            break
    return out


def parent_child(body: str, title: str = "", *, child_size: int = 300,
                 overlap: int = 60) -> list[Chunk]:
    """Heading sections become parents; their text is split into child windows.

    Children carry ``parent_id`` = the parent chunk's ordinal (as a string), so
    a hit can be expanded to the section that contains it.
    """
    if child_size < 80:
        raise ValueError("child_size must be >= 80")
    if overlap >= child_size:
        raise ValueError("overlap must be smaller than child_size")

    sections: list[tuple[list[str], str]] = []
    trail: list[str] = []
    buf: list[str] = []
    for line in (body or "").splitlines():
        m = _HEADING.match(line)
        if m:
            text = "\n".join(buf).strip()
            if text:
                sections.append((list(trail), text))
            buf = []
            level = len(m.group(1))
            trail = trail[: level - 1] + [m.group(2).strip()]
            continue
        buf.append(line)
    tail = "\n".join(buf).strip()
    if tail:
        sections.append((list(trail), tail))
    if not sections and (body or "").strip():
        sections = [([], (body or "").strip())]

    out: list[Chunk] = []
    parent_ord = 0
    for path, text in sections:
        header = " > ".join(path)
        parent_text = (header + "\n" if header else "") + text
        out.append(Chunk(text=parent_text, ord=parent_ord, level="parent",
                         heading_path=path))
        parent_id = str(parent_ord)
        parent_ord += 1
        step = child_size - overlap
        for start in range(0, max(1, len(text)), step):
            piece = text[start:start + child_size].strip()
            if not piece:
                break
            child = (header + "\n" if header else "") + piece
            out.append(Chunk(text=child, ord=len(out), level="child",
                             parent_id=parent_id, heading_path=path))
            if start + child_size >= len(text):
                break
    return out


STRATEGIES = {
    "heading": heading_chunks,
    "fixed": fixed_window,
    "parent-child": parent_child,
}


def chunk(body: str, title: str = "", strategy: str = "heading", **kw) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from "
                         f"{', '.join(STRATEGIES)}")
    return STRATEGIES[strategy](body, title, **kw)


def rebuild_library(store: Store, tenant_id: str, library: str,
                    strategy: str = "heading", **kw) -> dict:
    """Re-chunk every document in a library, preserving ACL and ids."""
    docs = store.list_documents(tenant_id, library)
    chunks_written = 0
    for doc in docs:
        pieces = chunk(doc.body, doc.title, strategy, **kw)
        store.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.id,))
        acl = doc.acl or ["public"]
        for i, piece in enumerate(pieces):
            cid = f"{doc.id}::{i}"
            store.conn.execute(
                "INSERT INTO chunks(id, doc_id, tenant_id, library, ord, text)"
                " VALUES (?,?,?,?,?,?)",
                (cid, doc.id, tenant_id, library, i, piece.text))
            for principal in acl:
                store.conn.execute(
                    "INSERT OR IGNORE INTO chunk_acl(chunk_id, principal) VALUES (?,?)",
                    (cid, principal))
            chunks_written += 1
    store.audit(tenant_id, "chunkers", f"rechunk:{strategy}", library,
                f"{len(docs)} docs, {chunks_written} chunks")
    return {"documents": len(docs), "chunks": chunks_written, "strategy": strategy}
