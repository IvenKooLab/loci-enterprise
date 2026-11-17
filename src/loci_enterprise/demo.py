"""Reproduce the 85% -> 60% incident, then show what actually fixes it.

The scenario is deliberately boring, because the incident is boring:

  a company runs a policy wiki. Fourteen questions are answerable at 86%.
  Over a quarter, four policies change. Nobody retires the old pages — the new
  ones are simply added, which is what everyone does. Accuracy falls to ~57%
  while retrieval "works fine": the top hit is still a confident, well-formed,
  correctly-cited policy. It is just no longer true.

Why the *old* page keeps winning: it has been cited for a year, so it carries
more authority than the page published last week. That is a real ranking
signal, and it is exactly what makes stale documents so sticky.

Then we try the two things people try:

  A. add more documents        -> accuracy goes DOWN, not up
  B. retire what was superseded -> accuracy comes back

Run it with `python -m loci_enterprise.demo`.
"""
from __future__ import annotations

import sys
import textwrap

from . import governance, lifecycle
from .retrieval import RetrievalConfig, Retriever
from .store import Document, Store

TENANT = "acme"
LIB = "policy"

# topic, title, body, question, current fact
BASELINE = [
    ("reimburse", "差旅报销", "单次住宿上限 500 元。", "住宿上限是多少", "500"),
    ("leave", "年假", "入职满一年享 10 天年假。", "年假多少天", "10"),
    ("overtime", "加班", "工作日加班按 1.5 倍计薪。", "加班几倍", "1.5"),
    ("laptop", "设备", "笔记本更换周期为 36 个月。", "笔记本几年一换", "36"),
    ("travel", "出差", "国内出差需提前 3 天申请。", "国内出差提前几天", "3"),
    ("expense", "报销时限", "费用需在 30 天内提交。", "报销多久内提交", "30"),
    ("security", "密码", "密码复杂度要求至少 12 位。", "密码多少位", "12"),
    ("wifi", "访客网络", "访客网络每 8 小时失效。", "访客网络多久失效", "8"),
    ("parking", "停车", "每月停车补贴 200 元。", "停车补贴多少", "200"),
    ("training", "培训", "每人每年培训预算 3000 元。", "培训预算多少", "3000"),
    ("sick", "病假", "病假需提供二级以上医院证明。", "病假证明要求", "二级"),
    ("notice", "离职", "离职需提前 30 天通知。", "离职通知提前几天", "30"),
]

# The two questions that never worked: they need two documents joined, and no
# single chunk states the answer. A honest baseline is not 100%.
HARD = [
    ("住宿和停车补贴合计上限", "700", "cross-doc"),
    ("出差总成本上限", "无明确上限", "unstated"),
]

# topic -> (title, new body, new fact) — the world changed on 2026-10-01
POLICY_CHANGES = {
    "reimburse": ("差旅报销", "单次住宿上限 800 元。", "800"),
    "leave": ("年假", "入职满一年享 15 天年假。", "15"),
    "overtime": ("加班", "工作日加班按 2 倍计薪。", "2"),
    "travel": ("出差", "国内出差需提前 5 天申请。", "5"),
}

# Documents a well-meaning team adds to "improve coverage". They are all
# plausible, several are near-duplicates, and every one of them competes for
# the same top-k slot.
NOISE = [
    ("reimburse-faq", "差旅报销常见问题", "住宿上限按城市分档，一线城市可上浮。"),
    ("leave-old", "年假（历史版本）", "入职满一年享 8 天年假。"),
    ("overtime-note", "加班补充说明", "加班按 1.2 倍计薪（试行）。"),
    ("travel-guide", "出差指南", "国内出差需提前 2 天申请。"),
    ("reimburse-2019", "差旅报销 2019 版", "单次住宿上限 400 元。"),
    ("leave-draft", "年假政策草案", "拟调整为 12 天年假。"),
    ("laptop-note", "设备管理补充", "笔记本更换周期为 24 个月。"),
    ("parking-notice", "停车补贴通知", "每月停车补贴 150 元。"),
]


def _doc(topic: str, title: str, body: str, *, path: str | None = None,
         effective: str | None = None, authority: float = 1.0,
         written_by: str = "hr-bot") -> Document:
    return Document(id="", tenant_id=TENANT, library=LIB, path=path or f"{topic}.md",
                    title=title, body=f"# {title}\n\n{body}",
                    effective_from=effective, owner="hr", authority=authority,
                    stale_after_days=365, written_by=written_by, acl=["public"])


def questions_baseline() -> list[tuple[str, str, str]]:
    """The question set with the fact that was true *before* the change."""
    return [(q, f, t) for t, _ti, _b, q, f in BASELINE] + HARD


def questions_now() -> list[tuple[str, str, str]]:
    """The question set, with the fact that is true *today*."""
    out = []
    for topic, _t, _b, query, fact in BASELINE:
        changed = POLICY_CHANGES.get(topic)
        out.append((query, changed[2] if changed else fact, topic))
    out.extend(HARD)
    return out


def accuracy(store: Store, retriever: Retriever, questions) -> tuple[float, list]:
    """Top-1 must contain the *currently correct* fact. That is the metric.

    Deliberately not "did we retrieve something": the incident is exactly the
    case where we retrieve something, confidently, and it is stale.
    """
    ok, rows = 0, []
    for query, fact, topic in questions:
        hits = retriever.search(query, tenant_id=TENANT, principals=["public"])
        top = hits[0] if hits else None
        hit_ok = bool(top and fact in top.text)
        ok += hit_ok
        rows.append((query, fact, top.path if top else "—",
                     round(top.score, 4) if top else 0.0, hit_ok,
                     top.status if top else "no hit"))
    return ok / len(questions), rows


def run(verbose: bool = True) -> dict:
    store = Store(":memory:")
    gov = governance.Governor(store)
    retriever = Retriever(store, config=RetrievalConfig(top_k=3))
    results: dict[str, float] = {}

    # Authority models "this page has been cited for a year" — which is true,
    # and which is precisely why a stale page outranks a fresh one.
    for i, (topic, title, body, _q, _f) in enumerate(BASELINE):
        gov.write(_doc(topic, title, body, effective="2026-01-01", authority=1.3),
                  on_conflict="merge")

    acc, rows = accuracy(store, retriever, questions_baseline())
    results["1_healthy"] = acc
    if verbose:
        _head("Stage 1 — healthy corpus (12 policies, 2 questions never worked)")
        _table(rows)

    for topic, (title, body, _fact) in POLICY_CHANGES.items():
        gov.write(_doc(topic, title, body, path=f"{topic}-2026q4.md",
                       effective="2026-10-01", authority=1.3),
                  on_conflict="merge")
    acc, rows = accuracy(store, retriever, questions_now())
    results["2_added_without_retiring"] = acc
    if verbose:
        _head("Stage 2 — four policies changed; new pages added, old ones kept")
        _table(rows)

    for topic, title, body in NOISE:
        gov.write(_doc(topic, title, body, authority=1.35), on_conflict="merge")
    acc, rows = accuracy(store, retriever, questions_now())
    results["3_added_more_documents"] = acc
    if verbose:
        _head("Stage 3 — added six more documents to 'improve coverage'")
        _table(rows)

    for topic in POLICY_CHANGES:
        lifecycle.supersede(store, TENANT, f"{topic}.md", f"{topic}-2026q4.md",
                            library=LIB, actor="hr-admin", note="Q4 policy change")
    for topic, _title, _body in NOISE:
        doc = store.get_document(TENANT, LIB, f"{topic}.md")
        if doc:
            store.set_status(doc.id, "retired", actor="hr-admin",
                             reason="restates a superseded policy")
    acc, rows = accuracy(store, retriever, questions_now())
    results["4_retired_superseded"] = acc
    if verbose:
        _head("Stage 4 — superseded and duplicate pages retired from the index")
        _table(rows)
        _summary(results)
    return results


def _summary(results: dict[str, float]) -> None:
    labels = {
        "1_healthy": "healthy baseline",
        "2_added_without_retiring": "policy changed, only added docs",
        "3_added_more_documents": "added even more documents",
        "4_retired_superseded": "retired superseded documents",
    }
    _head("Summary")
    for key, value in results.items():
        print(f"  {labels[key]:<42} {value:6.1%}")
    print(textwrap.dedent("""
      Adding documents cannot fix a knowledge-base problem: it adds more
      plausible candidates to a similarity score that has no notion of
      "still true" — and more authority for the stale page to hide behind.
      Retiring what was superseded does fix it.

      Note the two questions that never reached 100%: a corpus is not graded
      on the questions you wrote, and a governance tool that reports 100%
      is reporting on itself.
    """))


def _head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _table(rows) -> None:
    print(f"  {'query':<20}{'want':<6}{'top hit':<22}{'score':<9}ok")
    print(f"  {'-' * 20}{'-' * 6}{'-' * 22}{'-' * 9}{'-' * 3}")
    for query, fact, path, score, ok, status in rows:
        flag = "✓" if ok else "✗"
        print(f"  {query:<20}{fact:<6}{path:<22}{score:<9}{flag} {status}")


if __name__ == "__main__":
    run()
    sys.exit(0)
