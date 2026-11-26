"""Hybrid retrieval where permission and lifecycle are part of the ranking.

Order of operations is the whole point:

    ACL join -> lifecycle filter -> BM25 over THAT corpus -> vector search
             -> RRF -> lifecycle weight -> feedback weight -> top-k

Two consequences worth stating plainly:

* **IDF is computed inside the permission domain.** A stranger's documents
  cannot change your term weights, which is both a leak fix and a quality fix.
* **Superseded documents are weighted out before fusion**, not filtered after
  it. Filtering after top-k is how an old policy keeps a slot that a current
  one should have had.
"""
from __future__ import annotations

import math
import re
from contextlib import nullcontext
from dataclasses import dataclass

from . import lifecycle
from .store import Store

_WORD = re.compile(r"[a-z0-9_]+", re.I)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Latin/identifier runs whole; CJK as overlapping bigrams.

    Two choices, both load-bearing:

    * `user_id` stays one token — splitting on `_` is what makes two different
      columns rank identically.
    * CJK is bigrammed rather than split into single characters. Single chars
      make every Chinese sentence match every other one on 是/的/在, which on a
      small corpus is worse than no lexical signal at all. Bigrams keep
      `住宿上限` from matching `病假证明`.
    """
    out: list[str] = [m.group(0).lower() for m in _WORD.finditer(text)]
    for m in _CJK_RUN.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            out.append(run)
            continue
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def bm25_scores(docs: list[str], query: str, *, k1: float = 1.5,
                b: float = 0.75) -> list[float]:
    """Okapi BM25 over exactly the documents passed in (the visible corpus)."""
    n = len(docs)
    if n == 0:
        return []
    tokenized = [tokenize(d) for d in docs]
    avgdl = sum(len(t) for t in tokenized) / n or 1.0
    q_terms = tokenize(query)
    df: dict[str, int] = {}
    for toks in tokenized:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    scores = []
    for toks in tokenized:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(toks) / avgdl)
            s += idf * (tf[term] * (k1 + 1)) / denom
        scores.append(s)
    return scores


class HashEmbedder:
    """Deterministic bag-of-tokens vector — offline, stable, not semantic.

    It exists so tests and demos never touch a network and never depend on a
    provider's drift. Swap in a real embedder for production quality; the
    ranking plumbing is identical.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = abs(hash_stable(tok)) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def hash_stable(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h if h < 2**31 else h - 2**32


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    path: str
    library: str
    text: str
    status: str
    score: float
    lexical: float = 0.0
    vector: float = 0.0
    reason: str = ""


@dataclass
class RetrievalConfig:
    top_k: int = 5
    rrf_k: int = 60
    hybrid: bool = True
    apply_lifecycle: bool = True
    apply_feedback: bool = True
    min_score: float = 0.0


class Retriever:
    def __init__(self, store: Store, *, embedder=None, config: RetrievalConfig | None = None,
                 feedback_scores: dict[str, float] | None = None, tracer=None):
        self.store = store
        self.embedder = embedder or HashEmbedder()
        self.config = config or RetrievalConfig()
        self.feedback_scores = feedback_scores or {}
        self.tracer = tracer

    def _span(self, name: str, **attrs):
        return self.tracer.span(name, **attrs) if self.tracer else nullcontext()

    def search(self, query: str, *, tenant_id: str, principals: list[str],
               library: str | None = None, on=None) -> list[Hit]:
        if self.tracer:
            with self.tracer.span("search", query=query[:80]) as root:
                hits = self._search(query, tenant_id=tenant_id,
                                    principals=principals, library=library, on=on)
                root.attrs["hits"] = len(hits)
                return hits
        return self._search(query, tenant_id=tenant_id, principals=principals,
                            library=library, on=on)

    def _search(self, query: str, *, tenant_id: str, principals: list[str],
                library: str | None = None, on=None) -> list[Hit]:
        cfg = self.config
        rows = self.store.visible_chunks(tenant_id, principals, library)
        if not rows:
            return []

        # ---- lifecycle gate, before scoring -------------------------------
        kept: list[dict] = []
        weights: dict[str, tuple[float, str]] = {}
        for r in rows:
            doc = self.store.get_document(tenant_id, doc_id=r["doc_id"])
            if doc is None:
                continue
            verdict = lifecycle.evaluate(doc, on=on)
            if not cfg.apply_lifecycle:
                weights[r["id"]] = (1.0, "lifecycle disabled")
                kept.append(r)
                continue
            if verdict.weight <= 0.0:
                continue
            weights[r["id"]] = (verdict.weight, verdict.reason)
            kept.append(r)
        if not kept:
            return []

        # ---- lexical over the visible corpus only -------------------------
        docs = [r["text"] for r in kept]
        with self._span("bm25", docs=len(docs)):
            lex = bm25_scores(docs, query)
        lex_norm = _minmax(lex)

        # ---- vector -------------------------------------------------------
        if cfg.hybrid:
            with self._span("vector", docs=len(docs)):
                qv = self.embedder.embed(query)
                vec = _minmax([cosine(qv, self.embedder.embed(d)) for d in docs])
        else:
            vec = [0.0] * len(docs)

        # ---- RRF over the two ranked lists --------------------------------
        with self._span("fuse", kept=len(kept)):
            lex_rank = _ranks(lex)
            vec_rank = _ranks(vec)
            fused: list[tuple[dict, float, float, float]] = []
            for i, r in enumerate(kept):
                score = 0.0
                if lex[i] > 0:
                    score += 1.0 / (cfg.rrf_k + lex_rank[i] + 1)
                if cfg.hybrid and vec[i] > 0:
                    score += 1.0 / (cfg.rrf_k + vec_rank[i] + 1)
                fused.append((r, score, lex_norm[i], vec[i]))

        # ---- lifecycle + authority + feedback weights ---------------------
        auth = {r["id"]: (r.get("authority") or 1.0) for r in kept}
        out: list[Hit] = []
        for r, s, lnorm, vscore in fused:
            w, reason = weights[r["id"]]
            s *= w
            s *= max(0.1, min(3.0, float(auth.get(r["id"], 1.0))))
            if cfg.apply_feedback:
                s *= self.feedback_scores.get(r["id"], 1.0)
            if s <= cfg.min_score:
                continue
            out.append(Hit(chunk_id=r["id"], doc_id=r["doc_id"], path=r["path"],
                           library=r["library"], text=r["text"], status=r["status"],
                           score=s, lexical=lnorm, vector=vscore, reason=reason))
        out.sort(key=lambda h: -h.score)
        return out[: cfg.top_k]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for pos, idx in enumerate(order):
        ranks[idx] = pos
    return ranks
