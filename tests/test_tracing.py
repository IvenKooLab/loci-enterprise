"""Span trees, token accounting and export."""
from __future__ import annotations

import json
import time

import pytest

from loci_enterprise.tracing import Tracer


def test_span_tree_nests_and_times():
    t = Tracer()
    t.start_trace("ask", query="住宿上限")
    with t.span("retrieve") as ret:
        ret.attrs["hits"] = 5
        with t.span("bm25"):
            time.sleep(0.002)
        with t.span("vector"):
            time.sleep(0.002)
    with t.span("generate"):
        time.sleep(0.002)

    summary = t.summary()
    assert summary["spans"] == 5
    assert summary["errors"] == 0
    assert summary["duration_ms"] > 0
    assert {s["name"] for s in summary["stages"]} == {"retrieve", "generate"}


def test_child_time_is_inside_the_parent_not_added_to_it():
    t = Tracer()
    t.start_trace("root")
    with t.span("parent"):
        with t.span("child"):
            time.sleep(0.02)
    root = t.spans[0]
    parent = next(s for s in t.spans if s.name == "parent")
    child = next(s for s in t.spans if s.name == "child")
    assert parent.duration_ms >= child.duration_ms
    assert root.duration_ms >= parent.duration_ms


def test_error_is_recorded_and_reraised():
    t = Tracer()
    t.start_trace("root")
    with pytest.raises(ValueError):
        with t.span("explode"):
            raise ValueError("bad input")
    failed = next(s for s in t.spans if s.name == "explode")
    assert failed.status == "error" and "ValueError" in failed.error
    assert t.summary()["errors"] == 1


def test_tokens_are_summed_from_spans_not_estimated():
    t = Tracer()
    t.start_trace("ask")
    with t.span("generate") as span:
        t.record_tokens(span, prompt=1000, completion=200, model="deepseek-flash")
    with t.span("verify") as span:
        t.record_tokens(span, prompt=300, completion=20)
    s = t.summary()
    assert s["prompt_tokens"] == 1300 and s["completion_tokens"] == 220


def test_tree_rendering_is_indented_and_flags_errors():
    t = Tracer()
    t.start_trace("root")
    with t.span("retrieve") as r:
        r.attrs["hits"] = 3
        try:
            with t.span("rerank"):
                raise RuntimeError("model down")
        except RuntimeError:
            pass
    text = t.tree()
    assert "- retrieve" in text and "  - rerank" in text
    assert "!!" in text and "hits=3" in text


def test_export_is_json_serialisable():
    t = Tracer()
    t.start_trace("root")
    with t.span("step") as s:
        t.record_tokens(s, prompt=1, completion=2)
    payload = json.dumps(t.export(), ensure_ascii=False)
    assert '"trace_id"' in payload and '"spans"' in payload


def test_spans_are_written_to_the_jsonl_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    t = Tracer(path=path)
    t.start_trace("root")
    with t.span("step"):
        pass
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) >= 2
    assert {l["name"] for l in lines} == {"root", "step"}


def test_slowest_returns_descending_order():
    t = Tracer()
    t.start_trace("root")
    with t.span("fast"):
        pass
    with t.span("slow"):
        time.sleep(0.01)
    names = [s["name"] for s in t.slowest(2)]
    assert names[0] == "slow"


def test_empty_summary_is_safe():
    assert Tracer().summary()["spans"] == 0


def test_new_trace_resets_state():
    t = Tracer()
    t.start_trace("a")
    with t.span("x"):
        pass
    t.start_trace("b")
    assert len(t.spans) == 1 and t.spans[0].name == "b"


def test_retriever_reports_its_pipeline_stages():
    from loci_enterprise.retrieval import Retriever
    from loci_enterprise.store import Document, Store
    s = Store(":memory:")
    s.put_document(Document(id="", tenant_id="t1", library="lib", path="a.md",
                            title="A", body="# A\n\n住宿上限 800 元", acl=["public"]))
    tracer = Tracer()
    tracer.start_trace("query")
    hits = Retriever(s, tracer=tracer).search("住宿上限", tenant_id="t1",
                                              principals=["public"])
    assert hits
    names = {sp.name for sp in tracer.spans}
    assert {"search", "bm25", "vector", "fuse"} <= names
    search_span = next(sp for sp in tracer.spans if sp.name == "search")
    assert search_span.attrs["hits"] == len(hits)
