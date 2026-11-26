"""Tracing: where a request's time and tokens actually went.

A retrieval request is a pipeline — parse, embed, search, rerank, generate,
verify — and an average latency number tells you nothing about which stage
regressed. Spans give the shape:

    trace: ask "住宿上限"
      ├─ retrieve          42 ms   hits=5
      │   ├─ bm25          8 ms    docs=1204
      │   └─ vector        31 ms
      ├─ rerank            88 ms
      └─ generate         1204 ms  in=1830 out=210 tokens

Two rules make the data usable:

* **Spans are a tree, not a flat log.** A child's time is inside its parent's,
  so "slow" can be attributed instead of summed.
* **Token counts live on the span that spent them.** Provider calls record
  ``prompt_tokens`` / ``completion_tokens``; the trace total is a sum, never an
  estimate from character counts.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Span:
    id: str
    trace_id: str
    name: str
    parent_id: str | None
    start: float
    end: float | None = None
    attrs: dict = field(default_factory=dict)
    status: str = "ok"
    error: str = ""

    @property
    def duration_ms(self) -> float:
        # an unfinished span reports elapsed-so-far; the root of a live trace
        # is unfinished by definition and must not read as zero
        end = self.end if self.end is not None else time.perf_counter()
        return round((end - self.start) * 1000, 2)

    def as_dict(self) -> dict:
        return {"id": self.id, "trace_id": self.trace_id, "name": self.name,
                "parent_id": self.parent_id, "start": self.start, "end": self.end,
                "duration_ms": self.duration_ms, "attrs": self.attrs,
                "status": self.status, "error": self.error}


class Tracer:
    def __init__(self, *, path: str | Path | None = None,
                 service: str = "loci-enterprise"):
        self.path = Path(path) if path else None
        self.service = service
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self.trace_id = ""

    # --------------------------------------------------------------- tracing
    def start_trace(self, name: str = "trace", **attrs) -> str:
        self.trace_id = uuid.uuid4().hex
        self.spans.clear()
        self._stack.clear()
        root = self._new(name, None, attrs)
        self._stack.append(root)
        self._flush(root)
        return self.trace_id

    def end_trace(self) -> dict:
        """Close the root span and return the summary.

        Optional: summary() works without it by using the union of finished
        spans, but an explicit close makes the trace's own duration honest
        even when the last child finishes earlier than the caller.
        """
        if self.spans and self.spans[0].end is None:
            self.spans[0].end = time.perf_counter()
            self._flush(self.spans[0])
        self._stack.clear()
        return self.summary()

    def _new(self, name: str, parent_id: str | None, attrs: dict) -> Span:
        span = Span(id=uuid.uuid4().hex[:16], trace_id=self.trace_id or "-",
                    name=name, parent_id=parent_id, start=time.perf_counter(),
                    attrs=dict(attrs))
        self.spans.append(span)
        return span

    @contextmanager
    def span(self, name: str, **attrs):
        """Context manager; records the exception and re-raises."""
        parent = self._stack[-1].id if self._stack else None
        span = self._new(name, parent, attrs)
        self._stack.append(span)
        try:
            yield span
        except Exception as exc:                     # noqa: BLE001 - recorded, re-raised
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._stack.pop()
            span.end = time.perf_counter()
            self._flush(span)

    def record_tokens(self, span: Span, *, prompt: int = 0, completion: int = 0,
                      model: str = "") -> None:
        span.attrs["prompt_tokens"] = int(prompt)
        span.attrs["completion_tokens"] = int(completion)
        if model:
            span.attrs["model"] = model
        self._flush(span)

    def _flush(self, span: Span) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"service": self.service, **span.as_dict()},
                                ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- reporting
    def summary(self) -> dict:
        if not self.spans:
            return {"spans": 0, "duration_ms": 0, "errors": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "stages": []}
        root = self.spans[0]
        if root.end is None:
            # a trace still open: report the window covered so far
            finished = [s.end for s in self.spans if s.end]
            root.end = max(finished) if finished else root.start
        prompt = sum(int(s.attrs.get("prompt_tokens", 0)) for s in self.spans)
        completion = sum(int(s.attrs.get("completion_tokens", 0)) for s in self.spans)
        children = [s for s in self.spans if s.parent_id == root.id and s.end]
        return {
            "trace_id": self.trace_id,
            "spans": len(self.spans),
            "duration_ms": root.duration_ms,
            "errors": sum(1 for s in self.spans if s.status == "error"),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "stages": [{"name": s.name, "ms": s.duration_ms, **{
                k: v for k, v in s.attrs.items()
                if k in ("hits", "docs", "model", "status")}}
                for s in sorted(children, key=lambda x: -x.duration_ms)],
        }

    def slowest(self, n: int = 5) -> list[dict]:
        return [s.as_dict() for s in
                sorted([s for s in self.spans if s.end], key=lambda s: -s.duration_ms)[:n]]

    def tree(self) -> str:
        """Indented rendering, useful in a bug report."""
        lines: list[str] = []
        by_parent: dict[str | None, list[Span]] = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)

        def walk(parent: str | None, depth: int) -> None:
            for s in by_parent.get(parent, []):
                extra = " ".join(f"{k}={v}" for k, v in s.attrs.items()
                                 if k in ("hits", "docs", "model", "prompt_tokens",
                                          "completion_tokens"))
                flag = " !!" if s.status == "error" else ""
                lines.append(f"{'  ' * depth}- {s.name:<16} {s.duration_ms:>8.2f} ms"
                             f" {extra}{flag}")
                walk(s.id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)

    def export(self) -> dict:
        return {"service": self.service, "trace_id": self.trace_id,
                "summary": self.summary(), "spans": [s.as_dict() for s in self.spans]}
