"""Agent loop: tool dispatch, guardrails, tracing."""
from __future__ import annotations

import json

import pytest

from loci_enterprise.agent import (Agent, AgentContext, ScriptedProvider, Tool,
                                   ToolError, ToolRegistry, knowledge_tools)


def ctx(**kw):
    return AgentContext(tenant_id="t1", principals=["public"], library="lib", **kw)


def reg_with(handler=None, *, destructive=False):
    reg = ToolRegistry()
    reg.register(Tool("lookup", "look something up",
                      {"type": "object", "required": ["q"],
                       "properties": {"q": {"type": "string"}}},
                      handler or (lambda args, c: f"result for {args['q']}"),
                      destructive=destructive))
    return reg


def test_tool_schema_is_rendered_for_the_prompt():
    text = reg_with().describe()
    assert "lookup(q: string)" in text


def test_agent_calls_a_tool_then_answers():
    provider = ScriptedProvider([
        json.dumps({"tool": "lookup", "args": {"q": "budget"}}),
        json.dumps({"answer": "800", "citations": ["policy.md"]}),
    ])
    result = Agent(provider, reg_with()).run("what is the budget?", ctx())
    assert result.answer == "800" and result.citations == ["policy.md"]
    assert [s.kind for s in result.steps] == ["call", "answer"]
    assert "result for budget" in result.steps[0].observation


def test_unknown_tool_is_refused_and_reported():
    provider = ScriptedProvider([
        json.dumps({"tool": "delete_everything", "args": {}}),
        json.dumps({"answer": "sorry"}),
    ])
    result = Agent(provider, reg_with()).run("q", ctx())
    assert result.steps[0].kind == "error"
    assert "unknown tool" in result.steps[0].error


def test_destructive_tool_is_blocked_in_read_only_mode():
    calls = []
    reg = reg_with(lambda a, c: calls.append(1) or "done", destructive=True)
    provider = ScriptedProvider([
        json.dumps({"tool": "lookup", "args": {"q": "x"}}),
        json.dumps({"answer": "ok"}),
    ])
    result = Agent(provider, reg).run("q", ctx(read_only=True))
    assert calls == []                       # handler never ran
    assert "read-only" in result.steps[0].error


def test_write_mode_allows_destructive_tools():
    reg = reg_with(lambda a, c: "written", destructive=True)
    provider = ScriptedProvider([
        json.dumps({"tool": "lookup", "args": {"q": "x"}}),
        json.dumps({"answer": "ok"}),
    ])
    result = Agent(provider, reg).run("q", ctx(read_only=False))
    assert result.steps[0].kind == "call" and result.steps[0].observation == "written"


def test_missing_required_argument_is_refused():
    provider = ScriptedProvider([
        json.dumps({"tool": "lookup", "args": {}}),
        json.dumps({"answer": "ok"}),
    ])
    result = Agent(provider, reg_with()).run("q", ctx())
    assert "missing arguments" in result.steps[0].error


def test_unexpected_argument_is_refused():
    provider = ScriptedProvider([
        json.dumps({"tool": "lookup", "args": {"q": "x", "admin": True}}),
        json.dumps({"answer": "ok"}),
    ])
    result = Agent(provider, reg_with()).run("q", ctx())
    assert "unexpected arguments" in result.steps[0].error


def test_step_limit_stops_a_looping_agent():
    provider = ScriptedProvider([json.dumps({"tool": "lookup", "args": {"q": "x"}})] * 20)
    result = Agent(provider, reg_with(), max_steps=3).run("q", ctx())
    assert result.stopped == "max_steps"
    assert len([s for s in result.steps if s.kind == "call"]) == 3


def test_non_json_reply_stops_cleanly():
    result = Agent(ScriptedProvider(["I think the answer is 42"]),
                   reg_with()).run("q", ctx())
    assert result.stopped == "error"
    assert result.steps[-1].error.startswith("model did not return JSON")


def test_json_inside_a_code_fence_is_accepted():
    provider = ScriptedProvider([
        '```json\n{"tool": "lookup", "args": {"q": "x"}}\n```',
        '```json\n{"answer": "done"}\n```',
    ])
    result = Agent(provider, reg_with()).run("q", ctx())
    assert result.answer == "done"


def test_agent_traces_llm_and_tool_spans():
    from loci_enterprise.tracing import Tracer
    tracer = Tracer()
    tracer.start_trace("agent")
    provider = ScriptedProvider([
        json.dumps({"tool": "lookup", "args": {"q": "x"}}),
        json.dumps({"answer": "done"}),
    ])
    Agent(provider, reg_with(), tracer=tracer).run("q", ctx())
    names = [s.name for s in tracer.spans]
    assert names.count("llm") == 2 and "tool" in names
    assert tracer.summary()["prompt_tokens"] == 20


def test_knowledge_tools_search_and_open():
    def search(query, principals, library):
        return [{"path": "policy.md", "text": "住宿上限 800 元"},
                {"path": "leave.md", "text": "年假 15 天"}]

    reg = knowledge_tools(search, get_doc=lambda p: f"body of {p}")
    ctx_ = ctx()
    assert "policy.md" in reg.call("search_knowledge", {"query": "住宿"}, ctx_)
    assert reg.call("open_document", {"path": "policy.md"}, ctx_) == "body of policy.md"


def test_knowledge_tools_report_no_results_honestly():
    reg = knowledge_tools(lambda q, p, l: [])
    assert reg.call("search_knowledge", {"query": "zzz"}, ctx()) == "No results."


def test_tool_registry_rejects_bad_names():
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register(Tool("has space", "", {}, lambda a, c: ""))
