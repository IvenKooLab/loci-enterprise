"""The backend contract, executed against every implementation."""
from __future__ import annotations

import pytest

from loci_enterprise.backends import (IndexRecord, MemoryBackend, Query,
                                      SQLiteBackend)


@pytest.fixture(params=["memory", "sqlite"])
def backend(request):
    return MemoryBackend() if request.param == "memory" else SQLiteBackend()


def rec(chunk_id, doc_id, text, *, tenant="t1", library="lib", acl=("public",),
        status="active", authority=1.0, vector=None):
    return IndexRecord(chunk_id=chunk_id, doc_id=doc_id, tenant_id=tenant,
                       library=library, text=text, acl=list(acl), status=status,
                       authority=authority, vector=vector)


def q(text, *, principals=("public",), tenant="t1", library="lib", **kw):
    return Query(text=text, tenant_id=tenant, principals=list(principals),
                 library=library, **kw)


# ------------------------------------------------------------------ contract
def test_upsert_and_search(backend):
    backend.upsert([rec("c1", "d1", "住宿上限 800 元"),
                    rec("c2", "d2", "年假 15 天")])
    hits = backend.search(q("住宿上限"))
    assert hits and hits[0].chunk_id == "c1"


def test_acl_is_enforced_before_scoring(backend):
    backend.upsert([rec("c1", "d1", "工资表 机密", acl=("finance",)),
                    rec("c2", "d2", "公开政策", acl=("public",))])
    assert [h.chunk_id for h in backend.search(q("工资表", principals=["public"]))] == []
    assert [h.chunk_id for h in backend.search(q("工资表", principals=["finance"]))] == ["c1"]


def test_tenant_isolation(backend):
    backend.upsert([rec("c1", "d1", "tenant one", tenant="t1"),
                    rec("c2", "d2", "tenant two", tenant="t2")])
    assert [h.chunk_id for h in backend.search(q("tenant"))] == ["c1"]


def test_library_scope(backend):
    backend.upsert([rec("c1", "d1", "policy", library="hr"),
                    rec("c2", "d2", "policy", library="eng")])
    hits = backend.search(q("policy", library="hr"))
    assert [h.chunk_id for h in hits] == ["c1"]


def test_upsert_is_idempotent_per_chunk(backend):
    backend.upsert([rec("c1", "d1", "v1")])
    backend.upsert([rec("c1", "d1", "v2")])
    assert backend.stats("t1")["chunks"] == 1
    assert backend.search(q("v2"))[0].chunk_id == "c1"


def test_delete_doc_removes_all_its_chunks(backend):
    backend.upsert([rec("c1", "d1", "a"), rec("c2", "d1", "b"), rec("c3", "d2", "c")])
    assert backend.delete_doc("t1", "d1") == 2
    assert backend.stats("t1")["chunks"] == 1


def test_inactive_records_are_not_candidates(backend):
    backend.upsert([rec("c1", "d1", "retired text", status="retired")])
    assert backend.search(q("retired")) == []


def test_allowed_docs_narrows_the_candidate_set(backend):
    backend.upsert([rec("c1", "d1", "alpha"), rec("c2", "d2", "alpha")])
    hits = backend.search(q("alpha", allowed_docs={"d2"}))
    assert [h.chunk_id for h in hits] == ["c2"]


def test_empty_allowed_docs_returns_nothing(backend):
    backend.upsert([rec("c1", "d1", "alpha")])
    assert backend.search(q("alpha", allowed_docs=set())) == []


def test_authority_influences_order(backend):
    backend.upsert([rec("c1", "d1", "policy 800", authority=1.0),
                    rec("c2", "d2", "policy 800", authority=2.5)])
    assert backend.search(q("policy 800"))[0].chunk_id == "c2"


def test_ranking_is_identical_across_backends():
    records = [rec("c1", "d1", "住宿上限 800 元"), rec("c2", "d2", "住宿上限 500 元"),
               rec("c3", "d3", "年假 15 天")]
    mem, sql = MemoryBackend(), SQLiteBackend()
    mem.upsert(records)
    sql.upsert(records)
    mem_hits = [(h.chunk_id, round(h.score, 9)) for h in mem.search(q("住宿上限"))]
    sql_hits = [(h.chunk_id, round(h.score, 9)) for h in sql.search(q("住宿上限"))]
    assert mem_hits == sql_hits


def test_stats_report_counts_without_content(backend):
    backend.upsert([rec("c1", "d1", "secret text"), rec("c2", "d1", "more")])
    stats = backend.stats("t1")
    assert stats["chunks"] == 2 and stats["documents"] == 1
    assert "secret" not in str(stats)


def test_vector_search_uses_cosine_when_supplied(backend):
    backend.upsert([rec("c1", "d1", "irrelevant", vector=[1.0, 0.0]),
                    rec("c2", "d2", "irrelevant", vector=[0.0, 1.0])])
    hits = backend.search(q("anything", vector=[1.0, 0.0]))
    assert hits[0].chunk_id == "c1"
