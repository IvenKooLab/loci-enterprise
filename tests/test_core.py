"""Store, lifecycle and retrieval behaviour."""
from __future__ import annotations

import datetime as dt
import pytest

from loci_enterprise import lifecycle
from loci_enterprise.retrieval import Retriever, RetrievalConfig, tokenize
from loci_enterprise.store import Document, Store


def doc(path="a.md", *, body="# T\n\nhello world", tenant="t1", library="lib",
        title="T", acl=("public",), **kw) -> Document:
    return Document(id="", tenant_id=tenant, library=library, path=path,
                    title=title, body=body, acl=list(acl), **kw)


# ------------------------------------------------------------------ ACL
def test_acl_is_applied_before_scoring():
    s = Store(":memory:")
    s.put_document(doc("secret.md", body="# S\n\nbudget 999", acl=("finance",)))
    s.put_document(doc("public.md", body="# P\n\nbudget 1", acl=("public",)))
    r = Retriever(s)
    assert [h.path for h in r.search("budget", tenant_id="t1", principals=["public"])] \
        == ["public.md"]
    assert [h.path for h in r.search("budget", tenant_id="t1", principals=["finance"])] \
        == ["secret.md"]


def test_principals_union_does_not_duplicate_hits():
    s = Store(":memory:")
    s.put_document(doc("x.md", acl=("public", "eng")))
    hits = Retriever(s).search("hello", tenant_id="t1", principals=["public", "eng"])
    assert len(hits) == 1


def test_tenant_isolation():
    s = Store(":memory:")
    s.put_document(doc("a.md", tenant="t1"))
    s.put_document(doc("a.md", tenant="t2"))
    hits = Retriever(s).search("hello", tenant_id="t1", principals=["public"])
    assert all(h.path == "a.md" for h in hits)
    assert len(s.list_documents("t2")) == 1


def test_idf_is_computed_within_the_permission_domain():
    """A stranger's documents must not change your term weights."""
    s = Store(":memory:")
    s.put_document(doc("mine.md", body="# M\n\nzebra crossing rules"))
    r = Retriever(s)
    before = r.search("zebra", tenant_id="t1", principals=["public"])[0].score
    for i in range(20):
        s.put_document(doc(f"theirs{i}.md", body=f"# X\n\nzebra {i}",
                           acl=("other-team",)))
    after = r.search("zebra", tenant_id="t1", principals=["public"])[0].score
    assert before == pytest.approx(after)


# ------------------------------------------------------------- lifecycle
def test_effective_window_expires_a_document():
    s = Store(":memory:")
    s.put_document(doc("old.md", effective_to="2026-01-31"))
    d = s.get_document("t1", "lib", "old.md")
    v = lifecycle.evaluate(d, on=dt.date(2026, 3, 1))
    assert v.status == "expired" and v.weight == 0.0


def test_not_yet_in_force_is_excluded():
    s = Store(":memory:")
    s.put_document(doc("future.md", effective_from="2030-01-01"))
    d = s.get_document("t1", "lib", "future.md")
    assert lifecycle.evaluate(d, on=dt.date(2026, 1, 1)).status == "expired"


def test_supersede_links_both_directions():
    s = Store(":memory:")
    s.put_document(doc("v1.md", title="Policy"))
    s.put_document(doc("v2.md", title="Policy"))
    old, new = lifecycle.supersede(s, "t1", "v1.md", "v2.md",
                                   library="lib", actor="hr")
    assert old.superseded_by == "v2.md" and old.status == "superseded"
    assert new.supersedes == "v1.md" and new.status == "active"


def test_chain_and_current_walk_the_lineage():
    s = Store(":memory:")
    for name in ("v1.md", "v2.md", "v3.md"):
        s.put_document(doc(name, title="Policy"))
    lifecycle.supersede(s, "t1", "v1.md", "v2.md", library="lib", actor="a")
    lifecycle.supersede(s, "t1", "v2.md", "v3.md", library="lib", actor="a")
    assert [c["path"] for c in lifecycle.chain(s, "t1", "lib", "v1.md")] \
        == ["v1.md", "v2.md", "v3.md"]
    assert lifecycle.current(s, "t1", "lib", "v1.md")["path"] == "v3.md"


def test_chain_survives_a_cycle():
    s = Store(":memory:")
    s.put_document(doc("a.md"))
    s.put_document(doc("b.md"))
    lifecycle.supersede(s, "t1", "a.md", "b.md", library="lib", actor="x")
    s.conn.execute("UPDATE documents SET superseded_by='a.md' WHERE path='b.md'")
    assert len(lifecycle.chain(s, "t1", "lib", "a.md")) <= 4


def test_superseded_document_is_weighted_out_before_fusion():
    s = Store(":memory:")
    s.put_document(doc("old.md", body="# P\n\nlimit 500", title="Policy",
                       authority=1.5))
    s.put_document(doc("new.md", body="# P\n\nlimit 800", title="Policy"))
    r = Retriever(s)
    assert r.search("limit", tenant_id="t1", principals=["public"])[0].path == "old.md"
    lifecycle.supersede(s, "t1", "old.md", "new.md", library="lib", actor="hr")
    top = r.search("limit", tenant_id="t1", principals=["public"])[0]
    assert top.path == "new.md", "stale document must not survive in top-k"


def test_staleness_decays_weight():
    s = Store(":memory:")
    s.put_document(doc("a.md", stale_after_days=30))
    s.conn.execute("UPDATE documents SET updated_at='2020-01-01T00:00:00Z'")
    d = s.get_document("t1", "lib", "a.md")
    v = lifecycle.evaluate(d, on=dt.date(2026, 1, 1))
    assert v.status == "stale" and 0 < v.weight < 1


def test_retirement_candidates_and_conflict_scan():
    s = Store(":memory:")
    s.put_document(doc("old.md", title="Policy"))
    s.put_document(doc("new.md", title="Policy"))
    s.put_document(doc("dup.md", title="Other", body="x"))
    s.put_document(doc("dup2.md", title="Other", body="y"))
    lifecycle.supersede(s, "t1", "old.md", "new.md", library="lib", actor="a")
    cands = lifecycle.retirement_candidates(s, "t1")
    assert any(c.path == "old.md" and c.severity == "high" for c in cands)
    conflicts = lifecycle.conflict_scan(s, "t1")
    assert conflicts and conflicts[0]["title"] == "other"


# ------------------------------------------------------------- retrieval
def test_tokenizer_keeps_identifiers_and_bigrams_cjk():
    assert "user_id" in tokenize("select user_id")
    assert "住宿" in tokenize("住宿上限")


def test_cjk_bigram_does_not_match_unrelated_text():
    s = Store(":memory:")
    s.put_document(doc("reimburse.md", body="# 报销\n\n单次住宿上限 500 元。"))
    s.put_document(doc("sick.md", body="# 病假\n\n病假需提供二级以上医院证明。"))
    top = Retriever(s).search("住宿上限是多少", tenant_id="t1",
                              principals=["public"])[0]
    assert top.path == "reimburse.md"


def test_lifecycle_can_be_disabled_for_legacy_behaviour():
    s = Store(":memory:")
    s.put_document(doc("old.md", body="# P\n\nlimit 500", title="Policy",
                       authority=1.5))
    s.put_document(doc("new.md", body="# P\n\nlimit 800", title="Policy"))
    lifecycle.supersede(s, "t1", "old.md", "new.md", library="lib", actor="hr")
    r = Retriever(s, config=RetrievalConfig(apply_lifecycle=False))
    assert r.search("limit", tenant_id="t1", principals=["public"])[0].path == "old.md"


def test_chunks_are_rebuilt_on_update():
    s = Store(":memory:")
    s.put_document(doc("a.md", body="# A\n\nalpha"))
    n1 = s.counts("t1")["chunks"]
    s.put_document(doc("a.md", body="# A\n\nalpha\n\n# B\n\nbeta"))
    assert s.counts("t1")["chunks"] > n1
