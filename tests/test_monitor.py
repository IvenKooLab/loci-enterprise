"""Drift detection, alert severity and recovery."""
from __future__ import annotations

from loci_enterprise import monitor as mon
from loci_enterprise.judge import Verdict, verify
from loci_enterprise.providers import ProviderError, StubProvider


def obs(i, *, judged=0.9, hits=3, score=0.04, fb=None, lib="policy"):
    return mon.Observation(ts=float(i), tenant_id="t1", library=lib,
                           query=f"q{i}", n_hits=hits, top_score=score,
                           judged=judged, feedback=fb)


def seed(m, n, **kw):
    for i in range(n):
        m.record(obs(i, **kw))


# ------------------------------------------------------------- statistics
def test_window_accuracy_counts_only_judged_requests():
    stats = mon.summarize([obs(0, judged=0.9), obs(1, judged=0.2), obs(2, judged=None)])
    assert stats.n == 3 and stats.judged_n == 2
    assert stats.accuracy == 0.5


def test_explicit_feedback_overrides_the_judge():
    assert mon.summarize([obs(0, judged=0.99, fb="bad")]).accuracy == 0.0
    assert mon.summarize([obs(0, judged=0.01, fb="good")]).accuracy == 1.0


def test_zero_hit_counts_as_unacceptable():
    assert mon.summarize([obs(0, judged=None, hits=0)]).accuracy == 0.0


# -------------------------------------------------------------- detection
def test_accuracy_drop_raises_a_critical_alert():
    m = mon.Monitor(min_samples=10)
    seed(m, 40, judged=0.9)
    m.pin_baseline("t1", "policy")
    seed(m, 40, judged=0.3)
    alerts = m.check("t1", "policy")
    kinds = {a.kind: a for a in alerts}
    assert "accuracy_drop" in kinds
    assert kinds["accuracy_drop"].severity == "critical"


def test_stale_truth_is_called_out_when_scores_are_unchanged():
    """Retrieval confidence flat, answers worse: the corpus, not the ranker."""
    m = mon.Monitor(min_samples=10)
    seed(m, 40, judged=0.9, score=0.04)
    m.pin_baseline("t1", "policy")
    seed(m, 40, judged=0.4, score=0.04)
    kinds = {a.kind for a in m.check("t1", "policy")}
    assert "stale_truth" in kinds


def test_no_alert_on_thin_samples():
    m = mon.Monitor(min_samples=20)
    seed(m, 5, judged=0.1)
    assert m.check("t1", "policy") == []


def test_zero_hit_spike_is_a_warning():
    m = mon.Monitor(min_samples=10)
    seed(m, 40, judged=0.9, hits=3)
    m.pin_baseline("t1", "policy")
    for i in range(40, 80):
        m.record(obs(i, judged=None, hits=0))
    kinds = {a.kind for a in m.check("t1", "policy")}
    assert "zero_hit_spike" in kinds


def test_mitigation_actions_are_ordered_and_reversible():
    m = mon.Monitor(min_samples=10)
    seed(m, 40, judged=0.9)
    m.pin_baseline("t1", "policy")
    seed(m, 40, judged=0.2)
    m.check("t1", "policy")
    actions = m.propose_actions("t1", "policy")
    assert actions and all(a.reversible for a in actions)
    assert actions[0].kind == "quarantine_superseded"


def test_recovery_is_detected_against_the_pinned_baseline():
    m = mon.Monitor(min_samples=10)
    seed(m, 40, judged=0.9)
    m.pin_baseline("t1", "policy")
    seed(m, 40, judged=0.2)
    assert not m.recovered("t1", "policy")
    seed(m, 60, judged=0.9)
    assert m.recovered("t1", "policy")


def test_observations_persist_across_processes(tmp_path):
    log = tmp_path / "m.jsonl"
    m = mon.Monitor(path=log)
    seed(m, 5)
    again = mon.Monitor(path=log)
    assert again.stats("t1", "policy").n == 5


# ------------------------------------------------------------------ judge
def test_judge_parses_a_stub_verdict():
    v = verify(StubProvider(score=0.9), "q", "a", [{"path": "a.md", "text": "x"}])
    assert v.faithfulness == 0.9 and v.acceptable


def test_judge_rejects_cross_file_inference_even_when_quotes_are_right():
    class CrossFile(StubProvider):
        def chat(self, messages, **kw):
            from loci_enterprise.providers import ChatResult
            return ChatResult('{"faithfulness": 0.95, "cross_file": "unstated",'
                              ' "unsupported_claims": [], "reason": "join not stated"}',
                              model="stub")

    v = verify(CrossFile(), "q", "a", [{"path": "a.md", "text": "x"}])
    assert v.faithfulness == 0.95 and not v.acceptable


def test_judge_survives_a_provider_error():
    class Broken(StubProvider):
        def chat(self, messages, **kw):
            raise ProviderError("timeout")

    v = verify(Broken(), "q", "a", [])
    assert v.faithfulness == 0.0 and v.error


def test_verdict_serialises_for_the_ledger():
    assert set(Verdict(0.8).as_dict()) >= {"faithfulness", "cross_file", "model"}
