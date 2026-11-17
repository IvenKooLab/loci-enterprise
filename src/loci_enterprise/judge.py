"""Answer verification: is each claim actually supported by the sources?

Two failure modes matter and they need different checks:

* **unsupported claim** — the answer says something none of the excerpts say.
  Classic hallucination; cheap to catch by asking a model to compare.
* **licensed individually, wrong together** — every sentence is quoted
  correctly, but the *join* across two documents is not stated anywhere.
  This is the one that survives ordinary citation checking, and it is the one
  that hurts most when a policy changed and the answer blends old and new.

Both are scored on a sampled basis and fed to the monitor, never to the
ranking directly: a judge that quietly reorders results is a second, hidden
retriever.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .providers import Provider, ProviderError

PROMPT = """You verify an answer against the source excerpts it cites.

Rules:
- A claim is supported only if the excerpts state it, not merely imply it.
- If the claim combines facts from DIFFERENT excerpts, say whether the
  combination itself is stated. Two true citations can still be joined into
  a wrong conclusion.
- Do not use outside knowledge.

Reply with JSON only:
{{"faithfulness": <0.0-1.0>, "unsupported_claims": ["..."],
 "cross_file": "ok|unstated|n/a", "reason": "<one sentence>"}}

QUESTION:
{question}

ANSWER:
{answer}

EXCERPTS:
{excerpts}
"""

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class Verdict:
    faithfulness: float
    unsupported: list[str] = field(default_factory=list)
    cross_file: str = "n/a"
    reason: str = ""
    model: str = ""
    error: str = ""

    @property
    def acceptable(self) -> bool:
        return self.faithfulness >= 0.5 and self.cross_file != "unstated"

    def as_dict(self) -> dict:
        return {"faithfulness": round(self.faithfulness, 3),
                "unsupported": self.unsupported, "cross_file": self.cross_file,
                "reason": self.reason, "model": self.model, "error": self.error}


def _excerpt_block(hits: list) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        if hasattr(h, "path"):
            lines.append(f"[{i}] {h.path}\n{h.text}")
        else:
            lines.append(f"[{i}] {h.get('path', '?')}\n{h.get('text', '')}")
    return "\n\n".join(lines)


def verify(provider: Provider, question: str, answer: str, hits: list,
           *, max_excerpt_chars: int = 1200) -> Verdict:
    blocks = []
    for i, h in enumerate(hits, 1):
        path = getattr(h, "path", None) or (h.get("path") if isinstance(h, dict) else "?")
        text = getattr(h, "text", None) or (h.get("text") if isinstance(h, dict) else "")
        blocks.append(f"[{i}] {path}\n{text[:max_excerpt_chars]}")
    prompt = PROMPT.format(question=question, answer=answer,
                           excerpts="\n\n".join(blocks))
    try:
        result = provider.chat([{"role": "user", "content": prompt}],
                               temperature=0.0, max_tokens=400)
    except ProviderError as exc:
        return Verdict(0.0, error=str(exc))
    m = _JSON.search(result.text or "")
    if not m:
        return Verdict(0.0, reason="judge did not return JSON",
                       model=result.model, error="unparseable")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(0.0, reason="judge returned invalid JSON", model=result.model,
                       error="unparseable")
    try:
        score = float(data.get("faithfulness", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return Verdict(
        faithfulness=max(0.0, min(1.0, score)),
        unsupported=[str(x) for x in (data.get("unsupported_claims") or [])],
        cross_file=str(data.get("cross_file") or "n/a"),
        reason=str(data.get("reason") or ""),
        model=result.model)


def sample(every: int) -> bool:
    """Cheap deterministic sampling: judge 1 in `every` requests."""
    if every <= 1:
        return True
    return (hash_stable(str(id(object()))) % every) == 0


def hash_stable(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h & 0x7FFFFFFF
