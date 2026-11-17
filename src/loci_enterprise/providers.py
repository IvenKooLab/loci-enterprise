"""Pluggable model providers — one interface, several backends.

The governance layer must not care which model produced an answer, but it
does need one thing from every provider: a *judge* call that is cheap enough
to run on a sample of live traffic. That is the only reason this abstraction
exists.

Defaults target DeepSeek's OpenAI-compatible API (base_url
`https://api.deepseek.com`). The model *names* move: as of 2026-11 the
catalogue is `deepseek-flash` and `deepseek-v4-pro` (1M context, JSON output),
and the older `deepseek-chat` / `deepseek-reasoner` aliases are retired. Pin
the name in config or `DEEPSEEK_MODEL` rather than trusting a default.
Note the asymmetry that trips people up: that endpoint serves chat
completions, **not** embeddings — configure `[embed]` separately, point it at
an embedding-capable provider or a local model, and keep the two independent.
`StubProvider` exists so tests and demos never touch the network.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class ProviderError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict | None = None


class Provider:
    name = "provider"

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> ChatResult:
        raise NotImplementedError


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, *, base_url: str, api_key: str, model: str,
                 name: str = "openai-compat", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name
        self.timeout = timeout

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> ChatResult:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape") from exc
        usage = body.get("usage") or {}
        return ChatResult(text=text, model=body.get("model", self.model),
                          prompt_tokens=usage.get("prompt_tokens", 0),
                          completion_tokens=usage.get("completion_tokens", 0),
                          raw=body)


def deepseek(*, api_key: str | None = None, model: str | None = None,
             base_url: str = "https://api.deepseek.com") -> OpenAICompatProvider:
    """DeepSeek chat provider.

    `deepseek-flash` is the sensible judge default: judging is short-context,
    high-volume work, and the pro model costs an order of magnitude more per
    output token for a verdict. Pass `deepseek-v4-pro` only to adjudicate
    hard cross-file inferences.
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ProviderError("DEEPSEEK_API_KEY is not set")
    chosen = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-flash"
    return OpenAICompatProvider(base_url=base_url, api_key=key, model=chosen,
                                name="deepseek")


class StubProvider(Provider):
    """Deterministic offline provider for tests and `--offline` demos.

    It does not pretend to be smart. It returns structured JSON when the
    prompt asks for JSON, which is enough to exercise the judge plumbing.
    """
    name = "stub"

    def __init__(self, score: float = 0.9):
        self.score = score
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> ChatResult:
        self.calls.append(messages)
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        if "JSON" in user or "json" in user:
            text = json.dumps({
                "faithfulness": self.score,
                "unsupported_claims": [] if self.score >= 0.5 else ["(stub)"],
                "reason": "stub verdict",
            })
        else:
            text = "stub answer"
        return ChatResult(text=text, model="stub")


class EmbeddingProvider:
    """`/embeddings` for providers that have one; separate from chat."""

    def __init__(self, *, base_url: str, api_key: str, model: str,
                 name: str = "embed", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        return [item["embedding"] for item in body["data"]]
