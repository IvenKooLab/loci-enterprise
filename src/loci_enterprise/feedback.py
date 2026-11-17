"""Feedback loop: turn "that answer was wrong" into a ranking change.

Two loops, and they must not be confused with each other:

* **fast loop** — a thumbs-down demotes the chunks that were actually used, so
  the same wrong answer is less likely to be assembled again. Bounded and
  decaying, because a single angry click should not bury a document forever.
* **slow loop** — accepted/rejected pairs become eval cases, so the change is
  measurable against a pinned baseline instead of argued about.

The fast loop must never be the only loop: unmeasured feedback drifts
whichever way the loudest user pushes it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .store import Store, now_iso

# Ranking multipliers. Bad feedback sinks slowly, good feedback lifts slowly:
# asymmetric on purpose, because a false negative costs less than a false
# positive that keeps a wrong citation in the answer.
BAD_PENALTY = 0.82
GOOD_BONUS = 1.06
FLOOR, CEIL = 0.35, 1.5


@dataclass
class FeedbackEvent:
    tenant_id: str
    library: str
    query: str
    verdict: str                     # "good" | "bad" | "wrong_source" | "missing"
    chunk_ids: list[str]
    doc_ids: list[str]
    note: str = ""
    actor: str = ""
    ts: float = 0.0


class FeedbackLedger:
    def __init__(self, store: Store):
        self.store = store

    def record(self, ev: FeedbackEvent) -> None:
        ev.ts = ev.ts or time.time()
        self.store.conn.execute(
            "INSERT INTO feedback(ts, tenant_id, library, query, verdict,"
            " chunk_ids, doc_ids, note, actor) VALUES (?,?,?,?,?,?,?,?,?)",
            (now_iso(), ev.tenant_id, ev.library, ev.query, ev.verdict,
             json.dumps(ev.chunk_ids), json.dumps(ev.doc_ids), ev.note, ev.actor))
        action = "feedback:good" if ev.verdict == "good" else f"feedback:{ev.verdict}"
        self.store.audit(ev.tenant_id, ev.actor or "user", action, ev.library,
                         f"{ev.query[:80]} -> {len(ev.chunk_ids)} chunk(s)")

    def events(self, tenant_id: str, library: str | None = None,
               limit: int = 500) -> list[FeedbackEvent]:
        sql = "SELECT * FROM feedback WHERE tenant_id=?"
        args: list = [tenant_id]
        if library:
            sql += " AND library=?"
            args.append(library)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.store.conn.execute(sql, args).fetchall():
            out.append(FeedbackEvent(
                tenant_id=r["tenant_id"], library=r["library"], query=r["query"],
                verdict=r["verdict"], chunk_ids=json.loads(r["chunk_ids"]),
                doc_ids=json.loads(r["doc_ids"]), note=r["note"],
                actor=r["actor"], ts=0.0))
        return out

    # ------------------------------------------------------------- fast loop
    def chunk_weights(self, tenant_id: str, library: str | None = None) -> dict[str, float]:
        """Multiplier per chunk, from accepted/rejected verdicts."""
        weights: dict[str, float] = {}
        for ev in reversed(self.events(tenant_id, library, limit=2000)):
            if ev.verdict == "good":
                for cid in ev.chunk_ids:
                    weights[cid] = min(CEIL, weights.get(cid, 1.0) * GOOD_BONUS)
            elif ev.verdict in ("bad", "wrong_source"):
                for cid in ev.chunk_ids:
                    weights[cid] = max(FLOOR, weights.get(cid, 1.0) * BAD_PENALTY)
        return weights

    def corrected_documents(self, tenant_id: str) -> set[str]:
        """Documents that produced a *wrong* answer — retirement candidates."""
        out: set[str] = set()
        for ev in self.events(tenant_id, limit=5000):
            if ev.verdict in ("bad", "wrong_source"):
                out.update(ev.doc_ids)
        return out

    def cited_documents(self, tenant_id: str) -> set[str]:
        out: set[str] = set()
        for ev in self.events(tenant_id, limit=5000):
            out.update(ev.doc_ids)
        return out

    # ------------------------------------------------------------- slow loop
    def to_eval_cases(self, tenant_id: str, library: str | None = None,
                      out_path: str | Path | None = None) -> list[dict]:
        """Accepted queries become regression cases with their expected docs.

        A "good" verdict says: this query, these documents. That is exactly an
        eval case, so the corpus that produced a good answer is pinned as the
        expectation and a later change that breaks it fails CI.
        """
        cases: list[dict] = []
        for ev in self.events(tenant_id, library, limit=2000):
            if ev.verdict != "good" or not ev.doc_ids:
                continue
            cases.append({
                "query": ev.query,
                "expect_doc_ids": sorted(set(ev.doc_ids)),
                "library": ev.library,
                "source": "feedback",
            })
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
                         encoding="utf-8")
        return cases

    def summary(self, tenant_id: str, library: str | None = None) -> dict:
        evs = self.events(tenant_id, library, limit=5000)
        counts: dict[str, int] = {}
        for ev in evs:
            counts[ev.verdict] = counts.get(ev.verdict, 0) + 1
        return {"total": len(evs), "by_verdict": counts,
                "documents_cited": len(self.cited_documents(tenant_id)),
                "documents_corrected": len(self.corrected_documents(tenant_id))}
