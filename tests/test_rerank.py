"""Reranker behaviour: order, stability, fail-open."""
from __future__ import annotations

import json

from loci_enterprise.providers import ChatResult
from loci_enterprise.rerank import (Candidate, LexicalReranker, NoopReranker,
                                    ProviderReranker)


def cands():
    return [Candidate("a", "住宿上限是 800 元", 0.5),
            Candidate("b", "年假 15 天", 0.5),
            Candidate("c", "住宿 报销 政策", 0.5)]


def test_noop_preserves_order_and_truncates():
    out = NoopReranker().rerank("q", cands(), top_k=2)
    assert [c.id for c in out] == ["a", "b"]


def test_lexical_promotes_the_matching_passage():
    out = LexicalReranker().rerank("年假几天", cands())
    assert out[0].id == "b"


def test_lexical_keeps_original_prior_as_a_tiebreaker():
    same = [Candidate("x", "irrelevant one", 0.9),
            Candidate("y", "irrelevant two", 0.1)]
    out = LexicalReranker().rerank("nothing matches here", same)
    assert [c.id for c in out] == ["x", "y"]


def test_lexical_handles_empty_input():
    assert LexicalReranker().rerank("q", []) == []


def test_lexical_is_deterministic():
    a = [c.id for c in LexicalReranker().rerank("住宿上限", cands())]
    b = [c.id for c in LexicalReranker().rerank("住宿上限", cands())]
    assert a == b


class GoodProvider:
    def __init__(self, scores):
        self.scores = scores

    def chat(self, messages, **kw):
        return ChatResult(text=json.dumps({"scores": self.scores}), model="stub")


class BadProvider:
    def chat(self, messages, **kw):
        raise RuntimeError("provider down")


class GarbageProvider:
    def chat(self, messages, **kw):
        return ChatResult(text="not json at all", model="stub")


def test_provider_reranker_uses_scores():
    out = ProviderReranker(GoodProvider([0, 10, 5])).rerank("年假", cands())
    assert out[0].id == "b"


def test_provider_reranker_is_fail_open_on_error():
    out = ProviderReranker(BadProvider()).rerank("年假", cands())
    assert [c.id for c in out] == ["a", "b", "c"]


def test_provider_reranker_is_fail_open_on_garbage():
    out = ProviderReranker(GarbageProvider()).rerank("年假", cands())
    assert [c.id for c in out] == ["a", "b", "c"]


def test_provider_reranker_ignores_mismatched_score_count():
    out = ProviderReranker(GoodProvider([10])).rerank("年假", cands())
    assert [c.id for c in out] == ["a", "b", "c"]


def test_provider_reranker_keeps_tail_after_the_scored_head():
    many = [Candidate(str(i), "text", 0.5) for i in range(25)]
    out = ProviderReranker(GoodProvider([5] * 20), max_candidates=20).rerank("q", many)
    assert len(out) == 25


def test_retriever_applies_an_optional_reranker():
    from loci_enterprise.retrieval import Retriever
    from loci_enterprise.store import Document, Store

    store = Store(":memory:")
    for path, body in [("a.md", "# A\n\n住宿上限 800 元"), ("b.md", "# B\n\n年假 15 天")]:
        store.put_document(Document(id="", tenant_id="t1", library="lib", path=path,
                                    title=path, body=body, acl=["public"],
                                    written_by="t"))
    plain = Retriever(store).search("年假", tenant_id="t1", principals=["public"])
    reranked = Retriever(store, reranker=LexicalReranker()).search(
        "年假", tenant_id="t1", principals=["public"])
    assert reranked[0].path == "b.md"
    assert len(reranked) == len(plain)
