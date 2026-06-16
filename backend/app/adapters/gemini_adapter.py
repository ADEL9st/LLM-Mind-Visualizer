import time
from typing import AsyncGenerator
from google import genai
from app.schemas import RunRequest


def _ev(type_: str, data: dict) -> dict:
    return {"type": type_, "ts": time.perf_counter(), "data": data}


class GeminiAdapter:
    async def stream(self, request: RunRequest) -> AsyncGenerator[dict, None]:
        if not request.api_key:
            yield _ev("error", {"message": "Gemini API key is missing. Please enter it in the settings."})
            return

        yield _ev("run_started", {"prompt_tokens": 0})

        try:
            client = genai.Client(api_key=request.api_key)

            contents = []
            for turn in request.history:
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.content}]})
            contents.append({"role": "user", "parts": [{"text": request.prompt}]})

            response = await client.aio.models.generate_content_stream(
                model=request.model or "gemini-1.5-pro",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                ),
            )

            full_text = ""
            async for chunk in response:
                if chunk.text:
                    text = chunk.text
                    full_text += text
                    yield _ev("token", {"text": text, "generated_text": full_text})

        except Exception as e:
            yield _ev("error", {"message": str(e)})

        yield _ev("run_completed", {"refused": None, "state": "black_box", "errors": []})
