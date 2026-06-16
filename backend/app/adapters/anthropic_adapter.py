import time
from typing import AsyncGenerator
from anthropic import AsyncAnthropic
from app.schemas import RunRequest


def _ev(type_: str, data: dict) -> dict:
    return {"type": type_, "ts": time.perf_counter(), "data": data}


class AnthropicAdapter:
    async def stream(self, request: RunRequest) -> AsyncGenerator[dict, None]:
        if not request.api_key:
            yield _ev("error", {"message": "Anthropic API key is missing. Please enter it in the settings."})
            return

        yield _ev("run_started", {"prompt_tokens": 0})

        try:
            client = AsyncAnthropic(api_key=request.api_key)
            messages = [{"role": turn.role, "content": turn.content} for turn in request.history]
            messages.append({"role": "user", "content": request.prompt})

            async with client.messages.stream(
                model=request.model or "claude-3-5-sonnet-20241022",
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                messages=messages,
            ) as stream:
                full_text = ""
                async for text in stream.text_stream:
                    full_text += text
                    yield _ev("token", {"text": text, "generated_text": full_text})

        except Exception as e:
            yield _ev("error", {"message": str(e)})

        yield _ev("run_completed", {"refused": None, "state": "black_box", "errors": []})
