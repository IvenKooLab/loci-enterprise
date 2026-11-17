"""Multi-writer races, conflict policy, and the feedback loop."""
from __future__ import annotations

import pytest

from loci_enterprise import governance
from loci_enterprise.feedback import FeedbackEvent, FeedbackLedger
from loci_enterprise.store import Conflict, Document, Store


def doc(path="a.md", body="# T\n\nv1", title="T", **kw) -> Document:
    kw.setdefault("tenant_id", "t1")
    kw.setdefault("library", "lib")
    kw.setdefault("acl", ["public"])
    return Document(id="", tenant_id=kw.pop("tenant_id"), library=kw.pop("library"),
                    path=path, title=title, body=body, acl=kw.pop("acl"), **kw)


# ------------------------------------------------------- concurrency
def test_stale_expected_version_is_rejected():
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc(body="# T\n\nv1"))
    with pytest.raises(Conflict) as exc:
        g.write(doc(body="# T\n\nv2"), expected_version=99)
    assert exc.value.actual == 1


def test_merge_policy_keeps_both_edits():
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc(body="# T\n\nAlice: refund within 7 days"))
    g.write(doc(body="# T\n\nBob: refund within 10 days"),
            expected_version=0, on_conflict="merge")
    body = s.get_document("t1", "lib", "a.md").body
    assert "Alice" in body and "Bob" in body


def test_last_write_wins_is_audited_as_such():
    """Bob read v1, Alice already wrote v2; Bob explicitly chooses to overwrite."""
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc(body="# T\n\nold"))
    g.write(doc(body="# T\n\nalice"), expected_version=1)     # -> v2
    res = g.write(doc(body="# T\n\nbob"), expected_version=1,  # stale read
                  on_conflict="last_write_wins")
    assert res.status == "updated"
    assert "bob" in s.get_document("t1", "lib", "a.md").body
    actions = [a["action"] for a in s.audit_tail("t1", 10)]
    assert "write:last_write_wins" in actions


def test_conflict_reject_leaves_content_untouched():
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc(body="# T\n\noriginal"))
    with pytest.raises(Conflict):
        g.write(doc(body="# T\n\nintruder"), expected_version=0)
    assert "original" in s.get_document("t1", "lib", "a.md").body


def test_identical_duplicate_write_is_a_noop():
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc(body="# T\n\nsame"))
    before = s.get_document("t1", "lib", "a.md").version
    res = g.write(doc(body="# T\n\nsame"))
    assert res.status == "noop"
    assert s.get_document("t1", "lib", "a.md").version == before


def test_policy_change_lands_new_and_retires_old_atomically():
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc("v1.md", body="# P\n\nlimit 500", title="Policy"))
    res = g.apply_policy_change(
        tenant_id="t1", library="lib",
        new=doc("v2.md", body="# P\n\nlimit 800", title="Policy"),
        old_path="v1.md", effective="2026-10-01", actor="hr")
    assert res.status == "created"
    old = s.get_document("t1", "lib", "v1.md")
    new = s.get_document("t1", "lib", "v2.md")
    assert old.status == "superseded" and old.superseded_by == "v2.md"
    assert new.supersedes == "v1.md"


def test_versions_are_kept_for_audit():
    s = Store(":memory:")
    g = governance.Governor(s)
    g.write(doc(body="# T\n\nv1"))
    g.write(doc(body="# T\n\nv2"))
    g.write(doc(body="# T\n\nv3"))
    versions = s.versions(s.get_document("t1", "lib", "a.md").id)
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert "v1" in versions[0]["body"]


# ------------------------------------------------------------ feedback
def test_bad_feedback_sinks_the_chunk_that_was_used():
    s = Store(":memory:")
    s.put_document(doc("a.md", body="# T\n\nrefund 7 days", title="A"))
    s.put_document(doc("b.md", body="# T\n\nrefund 30 days", title="B"))
    ledger = FeedbackLedger(s)
    from loci_enterprise.retrieval import Retriever
    r = Retriever(s)
    top = r.search("refund", tenant_id="t1", principals=["public"])[0]
    ledger.record(FeedbackEvent("t1", "lib", "refund", "bad",
                                [top.chunk_id], [top.doc_id], actor="alice"))
    weights = ledger.chunk_weights("t1")
    assert weights[top.chunk_id] < 1.0


def test_feedback_to_eval_cases_roundtrip(tmp_path):
    s = Store(":memory:")
    ledger = FeedbackLedger(s)
    ledger.record(FeedbackEvent("t1", "lib", "how long is refund", "good",
                                ["c1"], ["d1"], actor="bob"))
    out = tmp_path / "cases.jsonl"
    cases = ledger.to_eval_cases("t1", out_path=out)
    assert cases[0]["expect_doc_ids"] == ["d1"]
    assert out.read_text().strip()


def test_corrected_documents_are_retirement_candidates():
    s = Store(":memory:")
    ledger = FeedbackLedger(s)
    ledger.record(FeedbackEvent("t1", "lib", "q", "wrong_source", ["c"], ["d9"]))
    assert "d9" in ledger.corrected_documents("t1")
