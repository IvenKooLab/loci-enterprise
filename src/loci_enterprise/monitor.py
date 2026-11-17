"""Retrieval quality monitoring: notice the drift before the users do.

The 85% -> 60% incident ran for three months because nothing measured it. A
RAG system has no error rate in the usual sense — every request returns 200 —
so the only early signal is a *slow change in the shape of the answers*:
more zero-hit queries, lower top scores, more "I don't know", more thumbs
down on the same library.

This module keeps a rolling window per (tenant, library), compares it against
a pinned baseline, and escalates. Detection is a two-proportion z-test on the
"answered acceptably" rate — deliberately boring statistics, because the
failure mode being watched for is a step change, not an anomaly shape.

Mitigation is separated from detection on purpose: detection must be
trustworthy and cheap, mitigation must be reversible and attributable
(see `Action` and the audit trail).
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ACCEPT_THRESHOLD = 0.5      # a judge score below this counts as a bad answer


def _now() -> float:
    return time.time()


@dataclass
class Observation:
    ts: float
    tenant_id: str
    library: str
    query: str
    n_hits: int
    top_score: float
    judged: float | None = None      # 0..1 faithfulness, None = not judged
    feedback: str | None = None      # "good" | "bad" | None
    latency_ms: float = 0.0

    @property
    def acceptable(self) -> bool | None:
        """Did this request produce something we would stand behind?"""
        if self.feedback == "bad":
            return False
        if self.feedback == "good":
            return True
        if self.judged is not None:
            return self.judged >= ACCEPT_THRESHOLD
        if self.n_hits == 0:
            return False
        return None


@dataclass
class WindowStats:
    n: int
    judged_n: int
    acceptable: int
    accuracy: float
    zero_hit_rate: float
    avg_top_score: float
    avg_latency_ms: float
    feedback_rate: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "judged_n": self.judged_n,
            "accuracy": round(self.accuracy, 4),
            "zero_hit_rate": round(self.zero_hit_rate, 4),
            "avg_top_score": round(self.avg_top_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "feedback_rate": round(self.feedback_rate, 4),
        }


def summarize(observations: list[Observation]) -> WindowStats:
    n = len(observations)
    if n == 0:
        return WindowStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    decided = [o.acceptable for o in observations if o.acceptable is not None]
    ok = sum(1 for d in decided if d)
    zero = sum(1 for o in observations if o.n_hits == 0)
    return WindowStats(
        n=n,
        judged_n=len(decided),
        acceptable=ok,
        accuracy=(ok / len(decided)) if decided else 0.0,
        zero_hit_rate=zero / n,
        avg_top_score=sum(o.top_score for o in observations) / n,
        avg_latency_ms=sum(o.latency_ms for o in observations) / n,
        feedback_rate=sum(1 for o in observations if o.feedback) / n,
    )


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """z for "is the recent rate really lower than the baseline?"."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return 0.0
    return (p1 - p2) / se


@dataclass
class Alert:
    tenant_id: str
    library: str
    severity: str            # critical | warning | info
    kind: str
    message: str
    baseline: dict = field(default_factory=dict)
    recent: dict = field(default_factory=dict)
    recommended: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "severity": self.severity, "kind": self.kind, "library": self.library,
            "message": self.message, "baseline": self.baseline,
            "recent": self.recent, "recommended": self.recommended,
        }


@dataclass
class Action:
    kind: str
    target: str
    detail: str
    reversible: bool = True


class Monitor:
    """Rolling observations + pinned baselines, per library."""

    def __init__(self, window: int = 500, *, path: str | Path | None = None,
                 z_threshold: float = 2.5, min_samples: int = 20):
        self.window = window
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.path = Path(path) if path else None
        self._obs: dict[tuple[str, str], deque[Observation]] = {}
        self._baseline: dict[tuple[str, str], dict] = {}
        self.alerts: list[Alert] = []
        if self.path and self.path.exists():
            self._load()

    # ----------------------------------------------------------- recording
    def record(self, obs: Observation) -> None:
        key = (obs.tenant_id, obs.library)
        self._obs.setdefault(key, deque(maxlen=self.window)).append(obs)
        if self.path:
            self._append(obs)

    def _append(self, obs: Observation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": obs.ts, "tenant_id": obs.tenant_id, "library": obs.library,
                "query": obs.query, "n_hits": obs.n_hits,
                "top_score": obs.top_score, "judged": obs.judged,
                "feedback": obs.feedback, "latency_ms": obs.latency_ms,
            }, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self._obs.setdefault((d["tenant_id"], d["library"]), deque(
                maxlen=self.window)).append(Observation(**d))

    # ------------------------------------------------------------ baselines
    def pin_baseline(self, tenant_id: str, library: str, *,
                     accuracy: float | None = None) -> dict:
        """Freeze the current window as the reference to regress against."""
        stats = summarize(list(self._obs.get((tenant_id, library), [])))
        entry = stats.as_dict()
        if accuracy is not None:
            entry["accuracy"] = accuracy
        self._baseline[(tenant_id, library)] = entry
        return entry

    def baseline(self, tenant_id: str, library: str) -> dict | None:
        return self._baseline.get((tenant_id, library))

    def stats(self, tenant_id: str, library: str, n: int | None = None) -> WindowStats:
        obs = list(self._obs.get((tenant_id, library), []))
        if n:
            obs = obs[-n:]
        return summarize(obs)

    # ------------------------------------------------------------- detection
    def check(self, tenant_id: str, library: str, *, recent_n: int = 50) -> list[Alert]:
        obs = list(self._obs.get((tenant_id, library), []))
        if len(obs) < self.min_samples:
            return []
        base = self._baseline.get((tenant_id, library))
        recent = summarize(obs[-recent_n:])
        found: list[Alert] = []

        # 1. step change in accepted-answer rate
        if base and base.get("judged_n", 0) >= self.min_samples \
                and recent.judged_n >= self.min_samples:
            k1 = int(round(recent.accuracy * recent.judged_n))
            k2 = int(round(base["accuracy"] * base["judged_n"]))
            z = two_proportion_z(k1, recent.judged_n, k2, base["judged_n"])
            if z < -self.z_threshold:
                drop = base["accuracy"] - recent.accuracy
                sev = "critical" if drop >= 0.15 else "warning"
                found.append(Alert(
                    tenant_id, library, sev, "accuracy_drop",
                    f"accepted-answer rate fell from {base['accuracy']:.0%} "
                    f"to {recent.accuracy:.0%} (z={z:.1f})",
                    base, recent.as_dict(),
                    ["check for superseded documents still indexed",
                     "review stale documents in this library",
                     "compare the eval set against the current corpus"]))

        # 2. rising zero-hit rate — corpus gap or over-tight ACL
        if base and recent.zero_hit_rate - base.get("zero_hit_rate", 0) >= 0.10 \
                and recent.n >= self.min_samples:
            found.append(Alert(
                tenant_id, library, "warning", "zero_hit_spike",
                f"zero-hit queries {base.get('zero_hit_rate', 0):.0%} -> "
                f"{recent.zero_hit_rate:.0%}",
                base, recent.as_dict(),
                ["check connector ingest lag",
                 "check whether ACL groups changed recently"]))

        # 3. same top score but worse answers — the pure "truth moved" case
        if base and recent.judged_n >= self.min_samples \
                and base.get("avg_top_score") and recent.avg_top_score \
                >= base["avg_top_score"] * 0.95 and base.get("accuracy") \
                and recent.accuracy < base["accuracy"] - 0.10:
            found.append(Alert(
                tenant_id, library, "critical", "stale_truth",
                "retrieval confidence is unchanged but answers got worse: "
                "the index is confidently returning outdated sources",
                base, recent.as_dict(),
                ["run lifecycle scan and retire superseded documents",
                 "declare supersedes/effective on the new policy",
                 "do NOT add more documents — it will not help"]))

        self.alerts.extend(found)
        return found

    # ------------------------------------------------------------ mitigation
    def propose_actions(self, tenant_id: str, library: str) -> list[Action]:
        """What to do *now*, ordered safest-first. Caller applies, this audits."""
        actions: list[Action] = []
        for alert in self.alerts:
            if alert.library != library:
                continue
            if alert.kind in ("stale_truth", "accuracy_drop"):
                actions.append(Action(
                    "quarantine_superseded", library,
                    "drop superseded/expired documents from the active index"))
                actions.append(Action(
                    "freeze_baseline", library,
                    "pin the pre-incident window so recovery is measurable"))
                actions.append(Action(
                    "require_review", library,
                    "route answers in this library through human review until recovered"))
            elif alert.kind == "zero_hit_spike":
                actions.append(Action("requeue_ingest", library,
                                      "re-run the connector for this library"))
        seen, unique = set(), []
        for a in actions:
            if a.kind not in seen:
                seen.add(a.kind)
                unique.append(a)
        return unique

    def recovered(self, tenant_id: str, library: str, *, target: float | None = None,
                  recent_n: int = 50) -> bool:
        base = self._baseline.get((tenant_id, library))
        if not base:
            return False
        recent = self.stats(tenant_id, library, recent_n)
        if recent.judged_n < self.min_samples:
            return False
        return recent.accuracy >= (target if target is not None else base["accuracy"]) - 0.02

    def report(self, tenant_id: str) -> dict:
        libs = sorted({lib for (t, lib) in self._obs if t == tenant_id})
        return {
            lib: {
                "recent": self.stats(tenant_id, lib).as_dict(),
                "baseline": self.baseline(tenant_id, lib),
                "alerts": [a.as_dict() for a in self.alerts
                           if a.tenant_id == tenant_id and a.library == lib],
            }
            for lib in libs
        }
