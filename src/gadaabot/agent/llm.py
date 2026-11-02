"""Thin client for the vLLM OpenAI-compatible server that serves the generator.

Qwen3.5 runs in non-thinking mode (chat_template_kwargs.enable_thinking = false) with the
recommended sampling for answers (D23: temperature 0.7, top_p 0.8, top_k 20,
presence_penalty 1.5) and near-greedy settings for JSON judgements.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

ANSWER = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
JUDGE = {"temperature": 0.1, "top_p": 0.9, "top_k": 20, "presence_penalty": 0.0}


class LLM:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.http = httpx.Client(timeout=timeout)
        self.calls = 0
        self.seconds = 0.0

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 700, sampling: dict[str, float] | None = None) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "chat_template_kwargs": {"enable_thinking": False}, **(sampling or ANSWER)}
        t0 = time.time()
        for attempt in range(3):
            try:
                r = self.http.post(self.url, json=body)
                r.raise_for_status()
                self.calls += 1
                self.seconds += time.time() - t0
                return str(r.json()["choices"][0]["message"]["content"]).strip()
            except (httpx.HTTPError, KeyError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"LLM call failed: {exc}") from exc
                time.sleep(2 * (attempt + 1))
        return ""

    def chat_json(self, messages: list[dict[str, str]], max_tokens: int = 400) -> dict[str, Any]:
        """JSON judgement; one retry with an explicit reminder if the first reply does not parse."""
        for attempt in range(2):
            raw = self.chat(messages, max_tokens=max_tokens, sampling=JUDGE)
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    out = json.loads(m.group(0))
                    if isinstance(out, dict):
                        return out
                except json.JSONDecodeError:
                    pass
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": "Return only valid JSON, nothing else."}]
        return {}
