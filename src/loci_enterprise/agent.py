"""A small tool-using agent loop, with the guardrails that matter.

The loop is deliberately plain — plan, call a tool, observe, repeat, answer —
because the interesting engineering is not the prompt, it is the boundary:

* **A tool is declared once**, with a JSON schema, and dispatch is by exact
  name. The model cannot invent a tool or smuggle arguments past validation.
* **Every call is bounded**: max steps, max tool calls per step, a per-run
  timeout budget, and an explicit allowlist. A model that loops forever costs
  money, and one that can call anything is a privilege-escalation bug.
* **Read-only mode is a policy, not a prompt.** Destructive tools are refused
  by the runner even if the model asks for them, and the refusal is recorded.
* **Every step is traced**, including the tool name, its duration and the
  observation size — so "the agent was slow" can be attributed to a tool.

`ScriptedProvider` exists so the loop is testable without a model: tests feed
exact tool-call turns and assert on the transcript.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .providers import ChatResult, Provider

SYSTEM = """You are a knowledge-base agent. Answer from the tools, never from memory.

Available tools:
{tools}

Reply with JSON only, one of:
  {{"tool": "<name>", "args": {{...}}}}   to call a tool
  {{"answer": "<text>", "citations": ["<source>"]}}   when you can answer

Rules:
- Call a tool before answering. Do not guess.
- If a tool returns nothing useful, say so instead of inventing an answer.
- Keep the answer short and cite the sources you used.
"""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict, "AgentContext"], str]
    destructive: bool = False

    def schema_line(self) -> str:
        props = ", ".join(f"{k}: {v.get('type', 'string')}"
                          for k, v in (self.parameters.get("properties") or {}).items())
        return f"- {self.name}({props}): {self.description}"


@dataclass
class AgentContext:
    tenant_id: str
    principals: list[str]
    library: str | None = None
    read_only: bool = True
    state: dict = field(default_factory=dict)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name.isidentifier():
            raise ValueError(f"bad tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(self._tools[n].schema_line() for n in self.names())

    # ------------------------------------------------------------ validation
    def call(self, name: str, args: dict, ctx: AgentContext) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.names())}")
        if ctx.read_only and tool.destructive:
            raise ToolError(f"tool {name!r} is destructive and this run is read-only")
        if not isinstance(args, dict):
            raise ToolError("tool arguments must be an object")
        required = tool.parameters.get("required") or []
        missing = [r for r in required if r not in args]
        if missing:
            raise ToolError(f"tool {name!r} missing arguments: {', '.join(missing)}")
        allowed = set((tool.parameters.get("properties") or {}).keys())
        extra = [k for k in args if k not in allowed] if allowed else []
        if extra:
            raise ToolError(f"tool {name!r} got unexpected arguments: {', '.join(extra)}")
        out = tool.handler(args, ctx)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


class ToolError(RuntimeError):
    pass


@dataclass
class Step:
    kind: str                  # call | answer | error | limit
    tool: str = ""
    args: dict = field(default_factory=dict)
    observation: str = ""
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    error: str = ""
    ms: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RunResult:
    answer: str
    citations: list[str]
    steps: list[Step]
    stopped: str               # answer | max_steps | error
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict:
        return {"answer": self.answer, "citations": self.citations,
                "stopped": self.stopped, "elapsed_ms": round(self.elapsed_ms, 2),
                "steps": [s.as_dict() for s in self.steps]}


class Agent:
    def __init__(self, provider: Provider, registry: ToolRegistry, *,
                 max_steps: int = 8, max_tool_calls: int = 2,
                 timeout_seconds: float = 120.0, tracer=None):
        self.provider = provider
        self.registry = registry
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout_seconds
        self.tracer = tracer

    def run(self, question: str, ctx: AgentContext) -> RunResult:
        started = time.perf_counter()
        steps: list[Step] = []
        messages = [{"role": "system",
                     "content": SYSTEM.format(tools=self.registry.describe())},
                    {"role": "user", "content": question}]
        tool_calls = 0

        for _ in range(self.max_steps):
            if time.perf_counter() - started > self.timeout:
                steps.append(Step(kind="limit", error="time budget exhausted"))
                return RunResult("", [], steps, "max_steps",
                                 (time.perf_counter() - started) * 1000)
            with self._span("llm") as span:
                reply = self.provider.chat(messages, temperature=0.0, max_tokens=600)
                self._tokens(span, reply)

            parsed = _parse(reply.text)
            if parsed is None:
                steps.append(Step(kind="error", error="model did not return JSON"))
                return RunResult("", [], steps, "error",
                                 (time.perf_counter() - started) * 1000)

            if "answer" in parsed:
                answer = str(parsed.get("answer") or "")
                cites = [str(c) for c in (parsed.get("citations") or [])]
                steps.append(Step(kind="answer", answer=answer, citations=cites))
                return RunResult(answer, cites, steps, "answer",
                                 (time.perf_counter() - started) * 1000)

            name = str(parsed.get("tool") or "")
            args = parsed.get("args") or {}
            if tool_calls >= self.max_tool_calls * self.max_steps:
                steps.append(Step(kind="limit", error="tool-call budget exhausted"))
                return RunResult("", [], steps, "max_steps",
                                 (time.perf_counter() - started) * 1000)
            tool_calls += 1
            t0 = time.perf_counter()
            try:
                with self._span("tool", tool=name) as span:
                    observation = self.registry.call(name, args, ctx)
                    span.attrs["observation_chars"] = len(observation)
            except ToolError as exc:
                observation = f"ERROR: {exc}"
                steps.append(Step(kind="error", tool=name, args=args,
                                  error=str(exc), ms=(time.perf_counter() - t0) * 1000))
            else:
                steps.append(Step(kind="call", tool=name, args=args,
                                  observation=observation[:4000],
                                  ms=(time.perf_counter() - t0) * 1000))
            messages.append({"role": "assistant", "content": reply.text})
            messages.append({"role": "user",
                             "content": f"Observation from {name}:\n{observation[:4000]}"})

        steps.append(Step(kind="limit", error="step limit reached"))
        return RunResult("", [], steps, "max_steps",
                         (time.perf_counter() - started) * 1000)

    def _span(self, name: str, **attrs):
        if self.tracer:
            return self.tracer.span(name, **attrs)
        return _NullSpan()

    @staticmethod
    def _tokens(span, reply: ChatResult) -> None:
        span.attrs["prompt_tokens"] = reply.prompt_tokens
        span.attrs["completion_tokens"] = reply.completion_tokens
        if reply.model:
            span.attrs["model"] = reply.model


class _NullSpan:
    """Stand-in when no tracer is attached, so call sites stay branch-free."""

    def __init__(self):
        self.attrs: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _parse(text: str) -> dict | None:
    """Accept a bare JSON object or one wrapped in prose/code fences."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class ScriptedProvider(Provider):
    """Deterministic provider: one canned reply per turn, for tests."""

    name = "scripted"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls = 0

    def chat(self, messages, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        text = self._replies.pop(0) if self._replies else '{"answer": "(out of script)"}'
        return ChatResult(text=text, model="scripted", prompt_tokens=10,
                          completion_tokens=5)


# ---------------------------------------------------------------- built-ins
def knowledge_tools(search: Callable[[str, list[str], str | None], list[dict]],
                    get_doc: Callable[[str], str] | None = None) -> ToolRegistry:
    """The minimal useful toolset: search the corpus, optionally open one hit."""
    reg = ToolRegistry()

    def _search(args, ctx):
        hits = search(str(args["query"]), ctx.principals, ctx.library)
        if not hits:
            return "No results."
        return "\n".join(f"[{i + 1}] {h['path']} :: {h['text'][:300]}"
                         for i, h in enumerate(hits[:5]))

    reg.register(Tool("search_knowledge",
                      "Search the knowledge base and return cited excerpts.",
                      {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string"}}},
                      _search))

    if get_doc is not None:
        def _open(args, ctx):
            return get_doc(str(args["path"]))[:6000]

        reg.register(Tool("open_document", "Read one document by its path.",
                          {"type": "object", "required": ["path"],
                           "properties": {"path": {"type": "string"}}},
                          _open))
    return reg
