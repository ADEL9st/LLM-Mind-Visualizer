from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.mock import MockAdapter
from app.adapters.ollama import OllamaAdapter
from app.adapters.openai_adapter import OpenaiAdapter
from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.gemini_adapter import GeminiAdapter
from app.schemas import ModelInfo, RunRequest


def _try_load(import_path: str, class_name: str):
    try:
        mod = __import__(import_path, fromlist=[class_name])
        return getattr(mod, class_name)()
    except ImportError:
        return None
    except Exception:
        return None


app = FastAPI(title="LLM Mind Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapters = {
    "mock": MockAdapter(),
    "ollama": OllamaAdapter(),
    "openai": OpenaiAdapter(),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
}
for key, import_path, class_name in (
    ("transformers", "app.adapters.transformers_hook", "TransformersHookAdapter"),
    ("pytorch", "app.adapters.pytorch_adapter", "PytorchAdapter"),
    ("nnsight", "app.adapters.nnsight_adapter", "NnsightAdapter"),
):
    instance = _try_load(import_path, class_name)
    if instance is not None:
        adapters[key] = instance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


import uuid

BOOT_ID = str(uuid.uuid4())

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "boot_id": BOOT_ID}


@app.get("/models")
async def models() -> list[ModelInfo]:
    detected = [
        ModelInfo(
            id="mock-qwen2.5-1.5b",
            label="Mock Qwen 1.5B Trace",
            adapter="mock",
            description="Deterministic simulated telemetry for UI and experiment flow.",
        ),
        ModelInfo(
            id="qwen2.5:1.5b",
            label="Ollama qwen2.5:1.5b",
            adapter="ollama",
            description="Black-box GGUF audit through Ollama.",
        ),
        # ── OpenAI ──────────────────────────────────────────────────────────
        ModelInfo(id="gpt-5.5",       label="GPT-5.5",       adapter="openai", description="OpenAI flagship — best reasoning & coding. 1M ctx, 128k output."),
        ModelInfo(id="gpt-5.4",       label="GPT-5.4",       adapter="openai", description="Powerful and cost-effective. 1M context window."),
        ModelInfo(id="gpt-5.4-mini",  label="GPT-5.4 Mini",  adapter="openai", description="Strong mini model for coding & agents. 400k context."),
        ModelInfo(id="gpt-5.4-nano",  label="GPT-5.4 Nano",  adapter="openai", description="Fastest and cheapest GPT-5.4 variant."),
        ModelInfo(id="gpt-4o",        label="GPT-4o",        adapter="openai", description="Previous flagship, multimodal, widely supported."),
        ModelInfo(id="gpt-4o-mini",   label="GPT-4o Mini",   adapter="openai", description="Cheap and fast; good for everyday tasks."),
        # ── Anthropic ───────────────────────────────────────────────────────
        ModelInfo(id="claude-fable-5",             label="Claude Fable 5",          adapter="anthropic", description="Most capable widely released Claude. 1M ctx, 128k output. $10/$50 MTok."),
        ModelInfo(id="claude-opus-4-8",            label="Claude Opus 4.8",         adapter="anthropic", description="Best Opus-tier: complex reasoning, agentic coding. 1M ctx. $5/$25 MTok."),
        ModelInfo(id="claude-sonnet-4-6",          label="Claude Sonnet 4.6",       adapter="anthropic", description="Best speed/intelligence balance. 1M ctx, 64k output. $3/$15 MTok."),
        ModelInfo(id="claude-haiku-4-5",           label="Claude Haiku 4.5",        adapter="anthropic", description="Fastest Claude with near-frontier intelligence. 200k ctx. $1/$5 MTok."),
        ModelInfo(id="claude-opus-4-7",            label="Claude Opus 4.7",         adapter="anthropic", description="Strong coding & vision. 1M ctx. $5/$25 MTok."),
        ModelInfo(id="claude-opus-4-6",            label="Claude Opus 4.6",         adapter="anthropic", description="Reliable & precise for enterprise workflows. 1M ctx."),
        ModelInfo(id="claude-sonnet-4-5",          label="Claude Sonnet 4.5",       adapter="anthropic", description="Fast and capable. 200k context."),
        ModelInfo(id="claude-opus-4-5",            label="Claude Opus 4.5",         adapter="anthropic", description="Previous Opus generation. 200k context."),
        # ── Google Gemini ────────────────────────────────────────────────────
        ModelInfo(id="gemini-3.5-flash",      label="Gemini 3.5 Flash",      adapter="gemini", description="Most intelligent Gemini for agentic & coding tasks (GA)."),
        ModelInfo(id="gemini-3.1-pro",        label="Gemini 3.1 Pro",        adapter="gemini", description="Advanced intelligence, complex problem-solving (Preview)."),
        ModelInfo(id="gemini-3.1-flash-lite", label="Gemini 3.1 Flash Lite", adapter="gemini", description="Cost-efficient, competitive performance (Stable)."),
        ModelInfo(id="gemini-2.5-pro",        label="Gemini 2.5 Pro",        adapter="gemini", description="Deep reasoning & coding. 1M token context (Stable)."),
        ModelInfo(id="gemini-2.5-flash",      label="Gemini 2.5 Flash",      adapter="gemini", description="Best price/performance for reasoning tasks (Stable)."),
        ModelInfo(id="gemini-2.5-flash-lite", label="Gemini 2.5 Flash Lite", adapter="gemini", description="Fastest and cheapest in the 2.5 family (Stable)."),
    ]
    detected.extend(_discover_transformers_models())
    return detected


def _discover_transformers_models() -> list[ModelInfo]:
    if not MODELS_DIR.exists():
        return []

    models_found: list[ModelInfo] = []
    for folder in sorted(item for item in MODELS_DIR.iterdir() if item.is_dir()):
        has_config = (folder / "config.json").exists()
        has_tokenizer = (folder / "tokenizer.json").exists() or (folder / "tokenizer_config.json").exists()
        has_weights = any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))
        if not (has_config and has_tokenizer and has_weights):
            continue

        model_id = f"../models/{folder.name}"
        label = _model_label(folder.name)
        if "pytorch" in adapters:
            models_found.append(
                ModelInfo(
                    id=model_id,
                    label=f"{label} (pytorch)",
                    adapter="pytorch",
                    description="White-box via plain PyTorch hooks — faster, no hook leak, natural EOS.",
                )
            )
        if "nnsight" in adapters:
            models_found.append(
                ModelInfo(
                    id=model_id,
                    label=f"{label} (nnsight)",
                    adapter="nnsight",
                    description="White-box via nnsight tracing — layer + head/neuron interventions.",
                )
            )
        if "transformers" in adapters:
            models_found.append(
                ModelInfo(
                    id=model_id,
                    label=f"{label} (hook v1)",
                    adapter="transformers",
                    description="Legacy white-box via manual forward hooks.",
                )
            )
    return models_found


def _model_label(folder_name: str) -> str:
    return folder_name.replace("-", " ").replace("_", " ").title()


from app.prompt_craft import apply_prompt_craft

@app.websocket("/ws/run")
async def run_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_text()
        request = RunRequest.model_validate_json(payload)
        
        if request.prompt_craft != "none":
            crafted = apply_prompt_craft(request.prompt, request.prompt_craft)
            request.prompt = crafted
            import time
            await websocket.send_text(json.dumps({
                "type": "prompt_crafted",
                "ts": time.perf_counter(),
                "data": {"crafted_prompt": crafted}
            }))
            
        adapter = adapters[request.adapter]
        async for item in adapter.stream(request):
            await websocket.send_text(json.dumps(item))
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 - send to client before closing
        await websocket.send_text(json.dumps({"type": "error", "ts": 0, "data": {"message": str(exc)}}))
    finally:
        await websocket.close()
