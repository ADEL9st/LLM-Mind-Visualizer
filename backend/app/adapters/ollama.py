from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
import re

import httpx

from app.adapters.base import ModelAdapter, event
from app.schemas import RunRequest


class OllamaAdapter(ModelAdapter):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        model = request.model or "qwen2.5:1.5b"
        yield event(
            "run_started",
            {
                "adapter": self.name,
                "model": model,
                "layer_count": 0,
                "head_count": 0,
                "prompt_tokens": None,
                "output_policy": request.output_policy,
            },
        )
        prompt = request.prompt
        if request.response_language != "en":
            lang_map = {"tr": "Turkish", "de": "German", "es": "Spanish"}
            lang_name = lang_map.get(request.response_language, request.response_language)
            prompt = f"Please reply in {lang_name}.\n\n" + prompt

        # Use /api/chat so prior turns ship as native messages — Ollama then handles
        # the model's chat template internally. With an empty history this is just
        # a single-turn chat call.
        messages: list[dict[str, str]] = []
        for turn in request.history:
            if turn.content:
                messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_new_tokens,
            },
        }
        generated = ""
        token_index = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
                        piece = (chunk.get("message") or {}).get("content", "") or chunk.get("response", "")
                        if piece:
                            if request.output_policy == "redacted":
                                piece = re.sub(r'[^\s\.,;!?-]', '█', piece)
                                
                            generated += piece
                            yield event(
                                "token",
                                {
                                    "index": token_index,
                                    "text": piece,
                                    "generated_text": generated,
                                    "entropy": None,
                                    "safety_state": "black_box",
                                },
                            )
                            token_index += 1
                        if chunk.get("done"):
                            yield event(
                                "black_box_metrics",
                                {
                                    "prompt_eval_count": chunk.get("prompt_eval_count"),
                                    "eval_count": chunk.get("eval_count"),
                                    "total_duration": chunk.get("total_duration"),
                                    "load_duration": chunk.get("load_duration"),
                                    "prompt_eval_duration": chunk.get("prompt_eval_duration"),
                                    "eval_duration": chunk.get("eval_duration"),
                                },
                            )
            except Exception as exc:  # noqa: BLE001 - surfaced as stream event
                yield event("error", {"message": f"Ollama request failed: {exc}"})
                return
        yield event(
            "run_completed",
            {
                "generated_text": generated,
                "finish_reason": "stop",
                "output_tokens": token_index,
            },
        )
