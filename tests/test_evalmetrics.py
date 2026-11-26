"""Metric correctness and gate behaviour."""
from __future__ import annotations

import json

import pytest

from loci_enterprise import evalmetrics as em


# --------------------------------------------------------------- retrieval
def test_precision_and_recall_at_k():
    ranked = ["a", "x", "b", "y"]
    rel = {"a", "b"}
    assert em.precision_at_k(ranked, rel, 1) == 1.0
    assert em.precision_at_k(ranked, rel, 2) == 0.5
    assert em.recall_at_k(ranked, rel, 2) == 0.5
    assert em.recall_at_k(ranked, rel, 3) == 1.0


def test_recall_of_empty_relevance_is_one_not_zero():
    assert em.recall_at_k(["a"], set(), 5) == 1.0


def test_average_precision_and_mrr():
    ranked = ["x", "a", "b"]
    rel = {"a", "b"}
    assert em.average_precision(ranked, rel) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert em.reciprocal_rank(ranked, rel) == pytest.approx(0.5)
    assert em.reciprocal_rank(["z"], rel) == 0.0


def test_ndcg_rewards_putting_the_best_document_first():
    rel = {"a": 3.0, "b": 1.0}
    good = em.ndcg_at_k(["a", "b"], rel, 5)
    bad = em.ndcg_at_k(["b", "a"], rel, 5)
    assert good == pytest.approx(1.0)
    assert bad < good


def test_ndcg_with_no_relevance_is_neutral():
    assert em.ndcg_at_k(["a"], {}, 5) == 1.0


# -------------------------------------------------------------- generation
def test_bleu_is_one_for_identical_text():
    assert em.bleu("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_bleu_penalises_a_short_candidate():
    full = "the cat sat on the mat today"
    assert em.bleu(full, "the cat") < 0.5


def test_bleu_handles_chinese_without_spaces():
    ref = "单次住宿上限八百元"
    assert em.bleu(ref, ref) == pytest.approx(1.0)
    assert em.bleu(ref, "单次住宿上限五百元") < 0.9


def test_rouge_l_scores_overlap():
    r = em.rouge_l("the quick brown fox jumps", "the quick brown fox")
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.8)
    assert 0 < r["f"] <= 1


def test_exact_match_ignores_case_and_spacing():
    assert em.exact_match("Hello  World", " hello world ") == 1.0
    assert em.exact_match("a", "b") == 0.0


# ------------------------------------------------------------------ runner
def make_cases():
    return [
        em.Case(query="q1", relevant=["d1"], answer="alpha beta gamma delta",
                generated="alpha beta gamma delta"),
        em.Case(query="q2", relevant=["d2"], answer="one two three four",
                generated="one two"),
        em.Case(query="q3", relevant=["d3"]),
    ]


def test_evaluate_aggregates_and_exposes_per_case():
    index = {"q1": ["d1", "dx"], "q2": ["dx", "d2"], "q3": ["d3"]}
    report = em.evaluate(make_cases(), lambda q, p, l: index[q])
    assert report.n == 3
    assert report.metrics["precision_at_1"] == pytest.approx(2 / 3)
    assert report.metrics["mrr"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert "bleu" in report.metrics          # only cases with both texts
    assert len(report.per_case) == 3


def test_report_saves_stable_json(tmp_path):
    report = em.evaluate([em.Case(query="q", relevant=["d"])], lambda q, p, l: ["d"])
    path = tmp_path / "r.json"
    report.save(path)
    data = json.loads(path.read_text())
    assert data["n"] == 1 and "metrics" in data and "per_case" in data


# ------------------------------------------------------------------- gate
def test_gate_passes_when_metrics_hold():
    report = em.Report(1, {"precision_at_1": 0.9, "mrr": 0.9, "map": 0.9,
                           "precision_at_5": 0.9, "ndcg_at_5": 0.9})
    verdict = em.gate(report, {"metrics": dict(report.metrics)})
    assert verdict["passed"] and not verdict["failures"]


def test_gate_fails_on_a_retrieval_regression():
    report = em.Report(1, {"precision_at_1": 0.6, "mrr": 0.9, "map": 0.9,
                           "precision_at_5": 0.9, "ndcg_at_5": 0.9})
    verdict = em.gate(report, {"metrics": {"precision_at_1": 0.9, "mrr": 0.9,
                                           "map": 0.9, "precision_at_5": 0.9,
                                           "ndcg_at_5": 0.9}})
    assert not verdict["passed"]
    assert verdict["failures"][0]["metric"] == "precision_at_1"


def test_gate_only_warns_on_generation_drop():
    report = em.Report(1, {"precision_at_1": 0.9, "mrr": 0.9, "map": 0.9,
                           "precision_at_5": 0.9, "ndcg_at_5": 0.9, "bleu": 0.2})
    verdict = em.gate(report, {"metrics": {"precision_at_1": 0.9, "mrr": 0.9,
                                           "map": 0.9, "precision_at_5": 0.9,
                                           "ndcg_at_5": 0.9, "bleu": 0.8}})
    assert verdict["passed"]                    # generation never blocks
    assert verdict["warnings"][0]["metric"] == "bleu"


def test_load_cases_accepts_both_relevance_shapes(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join([
        json.dumps({"query": "a", "relevant": ["d1"]}),
        json.dumps({"query": "b", "expect_doc_ids": ["d2"]}),
        json.dumps({"query": "c", "relevance": {"d3": 2.0}}),
    ]) + "\n", encoding="utf-8")
    cases = em.load_cases(p)
    assert cases[0].relevant == ["d1"]
    assert cases[1].relevant == ["d2"]
    assert cases[2].graded() == {"d3": 2.0}
