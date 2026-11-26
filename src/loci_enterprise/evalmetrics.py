"""Evaluation: the numbers that decide whether a change ships.

Two families, because they fail differently:

* **Retrieval metrics** (precision / recall / NDCG / MRR / MAP) answer "did the
  right document come back, and how high". They are cheap, deterministic and
  suitable for a CI gate.
* **Generation metrics** (BLEU / ROUGE-L / exact match) answer "does the text
  look like the reference". They are a weak proxy — a paraphrase scores badly
  and a confident hallucination can score well — so they are reported but
  never allowed to *block* a release on their own.

The report is a plain dict with a stable shape, because the only useful thing
to do with an eval is compare it to a stored one.

Chinese text is tokenised as CJK characters plus latin words; scoring it with
whitespace splitting would give every sentence n-gram precision of zero.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[a-z0-9_]+", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokens(text: str) -> list[str]:
    """Latin words whole, CJK one token per character, punctuation dropped."""
    out: list[str] = [m.group(0).lower() for m in _WORD.finditer(text or "")]
    out.extend(_CJK.findall(text or ""))
    return out


# --------------------------------------------------------------- retrieval
def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0                      # nothing to find
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 1.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def mean_average_precision(cases: list[tuple[list[str], set[str]]]) -> float:
    if not cases:
        return 0.0
    return sum(average_precision(r, rel) for r, rel in cases) / len(cases)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], relevance: dict[str, float], k: int) -> float:
    """Graded NDCG. A doc missing from `relevance` has gain 0."""
    gains = [relevance.get(d, 0.0) for d in ranked[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 1.0


# -------------------------------------------------------------- generation
def _ngrams(toks: list[str], n: int) -> dict[tuple, int]:
    out: dict[tuple, int] = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """Smoothed sentence BLEU with brevity penalty (0..1)."""
    ref, cand = tokens(reference), tokens(candidate)
    if not cand or not ref:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ng, cand_ng = _ngrams(ref, n), _ngrams(cand, n)
        if not cand_ng:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, ref_ng.get(g, 0)) for g, c in cand_ng.items())
        precisions.append(overlap / sum(cand_ng.values()))
    if any(p <= 0 for p in precisions):
        # add-one smoothing: without it a single missing 4-gram zeroes the score
        precisions = [(p if p > 0 else 1e-9) for p in precisions]
    log_mean = sum(math.log(p) for p in precisions) / max_n
    bp = min(1.0, math.exp(1 - len(ref) / len(cand))) if len(cand) > len(ref) else 1.0
    return bp * math.exp(log_mean)


def rouge_l(reference: str, candidate: str) -> dict[str, float]:
    """LCS-based ROUGE-L precision/recall/F."""
    ref, cand = tokens(reference), tokens(candidate)
    if not ref or not cand:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    prev = [0] * (len(cand) + 1)
    for i in range(1, len(ref) + 1):
        cur = [0] * (len(cand) + 1)
        for j in range(1, len(cand) + 1):
            cur[j] = prev[j - 1] + 1 if ref[i - 1] == cand[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(cand), lcs / len(ref)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f": f}


def exact_match(reference: str, candidate: str) -> float:
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return 1.0 if norm(reference) == norm(candidate) else 0.0


# ------------------------------------------------------------------ runner
@dataclass
class Case:
    query: str
    relevant: list[str] = field(default_factory=list)     # doc ids
    relevance: dict[str, float] = field(default_factory=dict)  # graded, optional
    answer: str | None = None
    generated: str | None = None
    principals: list[str] = field(default_factory=lambda: ["public"])
    library: str | None = None

    def graded(self) -> dict[str, float]:
        if self.relevance:
            return self.relevance
        return {d: 1.0 for d in self.relevant}


@dataclass
class Report:
    n: int
    metrics: dict[str, float]
    per_case: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
                "per_case": self.per_case}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")


def evaluate(cases: list[Case], search, *, k: int = 10) -> Report:
    """`search(query, principals, library) -> list[doc_id]`, best first."""
    if not cases:
        return Report(0, {})
    rows: list[dict] = []
    p1 = p5 = nd5 = nd10 = rr = ap = 0.0
    bleu_s = rouge_s = em_s = 0.0
    generated_n = 0

    for c in cases:
        ranked = search(c.query, c.principals, c.library) or []
        rel = set(c.relevant) or set(c.graded())
        p1 += precision_at_k(ranked, rel, 1)
        p5 += precision_at_k(ranked, rel, 5)
        nd5 += ndcg_at_k(ranked, c.graded(), 5)
        nd10 += ndcg_at_k(ranked, c.graded(), 10)
        rr += reciprocal_rank(ranked, rel)
        ap += average_precision(ranked, rel)
        row = {"query": c.query, "ranked": ranked[:k], "relevant": sorted(rel)}
        if c.answer is not None and c.generated is not None:
            generated_n += 1
            b, r, e = bleu(c.answer, c.generated), rouge_l(c.answer, c.generated)["f"], \
                exact_match(c.answer, c.generated)
            bleu_s += b
            rouge_s += r
            em_s += e
            row.update({"bleu": round(b, 4), "rouge_l": round(r, 4), "exact": e})
        rows.append(row)

    n = len(cases)
    metrics = {
        "precision_at_1": p1 / n,
        "precision_at_5": p5 / n,
        "ndcg_at_5": nd5 / n,
        "ndcg_at_10": nd10 / n,
        "mrr": rr / n,
        "map": ap / n,
    }
    if generated_n:
        metrics.update({"bleu": bleu_s / generated_n,
                        "rouge_l": rouge_s / generated_n,
                        "exact_match": em_s / generated_n})
    return Report(n, metrics, rows)


# ------------------------------------------------------------------- gates
RETRIEVAL_BLOCKING = ("precision_at_1", "precision_at_5", "ndcg_at_5", "mrr", "map")
GENERATION_ADVISORY = ("bleu", "rouge_l", "exact_match")


def gate(current: Report, baseline: dict, *, tolerance: float = 0.02,
         blocking: tuple[str, ...] = RETRIEVAL_BLOCKING) -> dict:
    """Compare to a stored report; retrieval regressions fail, generation warns."""
    base_metrics = baseline.get("metrics", baseline)
    failures, warnings = [], []
    for name in blocking:
        if name not in base_metrics or name not in current.metrics:
            continue
        drop = base_metrics[name] - current.metrics[name]
        if drop > tolerance:
            failures.append({"metric": name, "baseline": base_metrics[name],
                             "current": current.metrics[name], "drop": round(drop, 4)})
    for name in GENERATION_ADVISORY:
        if name in base_metrics and name in current.metrics:
            drop = base_metrics[name] - current.metrics[name]
            if drop > tolerance * 2:
                warnings.append({"metric": name, "baseline": base_metrics[name],
                                 "current": current.metrics[name], "drop": round(drop, 4)})
    return {"passed": not failures, "failures": failures, "warnings": warnings}


def load_cases(path: str | Path) -> list[Case]:
    out: list[Case] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(Case(query=d["query"],
                        relevant=d.get("relevant", d.get("expect_doc_ids", [])),
                        relevance=d.get("relevance", {}),
                        answer=d.get("answer"),
                        generated=d.get("generated"),
                        principals=d.get("principals", ["public"]),
                        library=d.get("library")))
    return out
