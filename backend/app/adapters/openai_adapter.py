import time
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.schemas import RunRequest


def _ev(type_: str, data: dict) -> dict:
    return {"type": type_, "ts": time.perf_counter(), "data": data}


class OpenaiAdapter:
    async def stream(self, request: RunRequest) -> AsyncGenerator[dict, None]:
        if not request.api_key:
            yield _ev("error", {"message": "OpenAI API key is missing. Please enter it in the settings."})
            return

        yield _ev("run_started", {"prompt_tokens": 0})

        try:
            client = AsyncOpenAI(api_key=request.api_key)
            messages = [{"role": turn.role, "content": turn.content} for turn in request.history]
            messages.append({"role": "user", "content": request.prompt})

            response = await client.chat.completions.create(
                model=request.model or "gpt-4o",
                messages=messages,
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                stream=True,
            )

            full_text = ""
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    yield _ev("token", {"text": text, "generated_text": full_text})

        except Exception as e:
            yield _ev("error", {"message": str(e)})

        yield _ev("run_completed", {"refused": None, "state": "black_box", "errors": []})
