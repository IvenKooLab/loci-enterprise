"""Reranking: a second, cheaper opinion on the top of the list.

Retrieval optimises for recall; the first N can afford to be generous. Rerank
is where precision is bought back — but only over a small candidate set,
because a cross-encoder over everything is slower than the search it is
supposed to improve.

Three implementations, in increasing cost:

* ``LexicalReranker`` — IDF-weighted term overlap plus a phrase bonus. No model,
  deterministic, and surprisingly strong on identifier-shaped queries
  (``SHT45``, ``user_id``) where embeddings are weakest.
* ``ProviderReranker`` — asks a chat model for a 0–10 relevance score per
  candidate. Fail-open: a provider error returns the original order, because a
  broken reranker must not turn a good search into an outage.
* ``NoopReranker`` — explicit "off", so callers never branch on None.

All of them return candidates in descending score with the original index kept,
so a tie is stable and a diff between runs is meaningful.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9_]+", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    out = [m.group(0).lower() for m in _WORD.finditer(text or "")]
    out.extend(_CJK.findall(text or ""))
    for m in re.finditer(r"[\u4e00-\u9fff]+", text or ""):
        run = m.group(0)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


@dataclass
class Candidate:
    id: str
    text: str
    score: float = 0.0
    meta: dict | None = None


class Reranker:
    name = "reranker"

    def rerank(self, query: str, candidates: list[Candidate],
               top_k: int | None = None) -> list[Candidate]:
        raise NotImplementedError


class NoopReranker(Reranker):
    name = "noop"

    def rerank(self, query, candidates, top_k=None):
        return candidates[:top_k] if top_k else list(candidates)


class LexicalReranker(Reranker):
    """IDF-weighted overlap + adjacent-pair bonus + length normalisation."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[Candidate],
               top_k: int | None = None) -> list[Candidate]:
        q = _tokens(query)
        if not q or not candidates:
            return candidates[:top_k] if top_k else list(candidates)
        docs = [_tokens(c.text) for c in candidates]
        n = len(docs)
        df: dict[str, int] = {}
        for toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(1 + n / (1 + df.get(t, 0))) for t in set(q)}
        phrase = " ".join(q)
        scored: list[tuple[float, int, Candidate]] = []
        for idx, (c, toks) in enumerate(zip(candidates, docs)):
            if not toks:
                continue
            present = set(toks)
            overlap = sum(idf[t] for t in set(q) if t in present) / (sum(idf.values()) or 1)
            pairs = len(set(zip(q, q[1:]))) if len(q) > 1 else 0
            adjacency = sum(1 for a, b in zip(q, q[1:]) if a in present and b in present)
            adj_score = (adjacency / pairs) if pairs else 0.0
            phrase_bonus = 0.25 if len(phrase) > 3 and phrase in " ".join(toks) else 0.0
            length_penalty = 1.0 / (1 + math.log(1 + len(toks) / max(len(q), 1)))
            total = (0.6 * overlap + 0.25 * adj_score + phrase_bonus) * (0.5 + 0.5 * length_penalty)
            scored.append((total, idx, c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = [Candidate(id=c.id, text=c.text,
                         score=round(self._blend(c.score, s), 6),
                         meta=c.meta) for s, _, c in scored]
        return out[:top_k] if top_k else out

    @staticmethod
    def _blend(original: float, rerank: float) -> float:
        # keep the retriever's ordering as a prior; rerank breaks near-ties
        return original * 0.55 + rerank * 0.45


PROMPT = """Score how well each passage answers the question.

Question: {query}

Passages:
{passages}

Reply with JSON only: {{"scores": [<0-10>, ...]}} in the same order, one score
per passage. 0 = irrelevant, 10 = directly answers it."""


class ProviderReranker(Reranker):
    """Pointwise LLM scoring; fail-open on any provider or parsing error."""

    name = "provider"

    def __init__(self, provider, *, max_candidates: int = 20,
                 weight: float = 0.6):
        self.provider = provider
        self.max_candidates = max_candidates
        self.weight = weight

    def rerank(self, query, candidates, top_k=None):
        head = candidates[: self.max_candidates]
        tail = candidates[self.max_candidates:]
        if not head:
            return []
        passages = "\n".join(f"[{i + 1}] {c.text[:500]}" for i, c in enumerate(head))
        try:
            result = self.provider.chat(
                [{"role": "user", "content": PROMPT.format(query=query,
                                                           passages=passages)}],
                temperature=0.0, max_tokens=300)
            data = json.loads(_extract_json(result.text))
            scores = [float(x) for x in data.get("scores", [])]
            if len(scores) != len(head):
                raise ValueError("score count mismatch")
        except Exception:                      # noqa: BLE001 - fail-open by design
            return NoopReranker().rerank(query, candidates, top_k)
        ranked = []
        for idx, (c, raw) in enumerate(zip(head, scores)):
            normalised = max(0.0, min(1.0, raw / 10.0))
            ranked.append((c.score * (1 - self.weight) + normalised * self.weight, idx, c))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        out = [Candidate(id=c.id, text=c.text, score=round(s, 6), meta=c.meta)
               for s, _, c in ranked] + tail
        return out[:top_k] if top_k else out


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reranker reply")
    return text[start:end + 1]
