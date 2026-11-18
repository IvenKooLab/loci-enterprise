"""Heading-aware chunking, with the section path kept in the text.

The breadcrumb is not decoration: an answer that cites `policy.md > 报销 >
额度` is verifiable, while one that cites `policy.md` alone is not, and the
difference matters most exactly when the document has been superseded.
"""
from __future__ import annotations

import re

MAX_CHARS = 700
MIN_CHARS = 120
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _fence_after(line: str, fence: str | None) -> str | None:
    """Fence state after this line, per CommonMark: a fence opens on three or
    more backticks (info string without backticks) or tildes, and closes only
    on a run of the same character, at least as long, with nothing but
    trailing whitespace after it. A naive toggle on any ``` line lets an inner
    triple-backtick example close an outer four-backtick fence, and then the
    example's literal `# heading` lines hijack the section trail."""
    match = _FENCE.match(line)
    if not match:
        return fence
    marker, info = match.groups()
    if fence:
        if marker[0] == fence[0] and len(marker) >= len(fence) and not info.strip(" \t"):
            return None
        return fence
    return marker if marker[0] == "~" or "`" not in info else None


def chunk_document(body: str, title: str = "") -> list[str]:
    """Split markdown into heading-scoped chunks, each prefixed by its path."""
    lines = body.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    trail: list[str] = []
    buf: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((list(trail), text))
        buf.clear()

    for line in lines:
        next_fence = _fence_after(line, fence)
        if next_fence != fence:
            fence = next_fence
            buf.append(line)
            continue
        m = None if fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            name = m.group(2).strip()
            trail = trail[: level - 1] + [name]
            continue
        buf.append(line)
    flush()

    if not sections and title:
        return [f"{title}\n\n{body}".strip()] if body.strip() else []

    out: list[str] = []
    for trail_path, text in sections:
        header = " > ".join(trail_path)
        prefix = f"{header}\n" if header else ""
        if len(text) <= MAX_CHARS:
            out.append((prefix + text).strip())
            continue
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) + 2 > MAX_CHARS:
                if len(cur) >= MIN_CHARS:
                    out.append((prefix + cur).strip())
                    cur = p
                else:
                    cur = f"{cur}\n\n{p}"
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur.strip():
            out.append((prefix + cur).strip())
    return out
