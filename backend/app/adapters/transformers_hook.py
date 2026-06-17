from __future__ import annotations

import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app.adapters.base import ModelAdapter, event, hallucination_from_entropy
from app.adapters.shared import release_memory
from app.schemas import RunRequest


class TransformersHookAdapter(ModelAdapter):
    name = "transformers"

    def __init__(self) -> None:
        self._loaded_model_id: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._layers: list[Any] = []

    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        model_id = (request.model or "").strip()
        if not model_id:
            yield event("error", {"message": "No model selected. Place a HuggingFace model folder under models/ and pick it from the dropdown."})
            return
        try:
            self._ensure_loaded(model_id)
        except FileNotFoundError:
            yield event("error", {"message": f"Model not found: '{model_id}'. Download a HuggingFace model into the models/ folder, then refresh."})
            return
        except Exception as exc:  # noqa: BLE001
            yield event("error", {"message": f"Failed to load model: {exc}"})
            return

        torch = self._torch
        try:
            async for ev in self._stream_body(request, model_id):
                yield ev
        finally:
            release_memory(torch)

    async def _stream_body(self, request: RunRequest, model_id: str) -> AsyncIterator[dict[str, Any]]:
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        layers = self._layers
        generated = ""
        interventions = request.active_interventions()

        history_turns = [(turn.role, turn.content) for turn in request.history]
        prompt_text = self._format_prompt(request.prompt, history_turns)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"]
        prompt_token_count = int(input_ids.shape[-1])

        yield event(
            "run_started",
            {
                "adapter": self.name,
                "model": model_id,
                "layer_count": len(layers),
                "head_count": getattr(getattr(model, "config", None), "num_attention_heads", None),
                "prompt_tokens": prompt_token_count,
                "output_policy": request.output_policy,
            },
        )

        if interventions:
            yield event(
                "intervention",
                {
                    "enabled": True,
                    "rules": [_intervention_payload(rule) for rule in interventions],
                    "count": len(interventions),
                    "status": "armed",
                },
            )

        hook_state: dict[int, dict[str, float]] = {}
        activity_accumulator = [0.0 for _ in layers]
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                input_hidden = _inputs[0] if isinstance(_inputs, tuple) and _inputs else None
                modified = _apply_layer_interventions(
                    hidden,
                    input_hidden,
                    _layer_interventions(interventions, layer_idx),
                )

                if hasattr(modified, "detach"):
                    value = modified.detach().float().pow(2).mean().sqrt().item()
                    if input_hidden is not None and hasattr(input_hidden, "detach"):
                        before = input_hidden.detach().float()
                        after = modified.detach().float()
                        if before.shape == after.shape:
                            before = _last_position(before)
                            after = _last_position(after)
                            delta = (after - before).pow(2).mean().sqrt().item()
                            before_norm = before.pow(2).mean().sqrt().item()
                            after_norm = after.pow(2).mean().sqrt().item()
                            value = delta / max(before_norm + after_norm, 1e-6)
                    hook_state[layer_idx] = {
                        "raw_activity": value,
                        "safety": 0.0,
                        "uncertainty": 0.0,
                    }

                if modified is not hidden:
                    if isinstance(output, tuple):
                        return (modified, *output[1:])
                    return modified
                return output

            return hook

        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(idx)))

        try:
            for step in range(request.max_new_tokens):
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, use_cache=False)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits / max(request.temperature, 1e-5), dim=-1)
                    entropy = max(0.0, float(-(probs * torch.log(probs.clamp_min(1e-9))).sum().item()))
                    top_probs, top_ids = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)

                    if request.temperature == 0:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)

                token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                generated += token_text
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                raw_activities = [hook_state.get(idx, {}).get("raw_activity", 0.0) for idx in range(len(layers))]
                shaped_activities = [math.log1p(max(value, 0.0)) for value in raw_activities]
                max_activity = max(shaped_activities) if shaped_activities else 0.0
                min_activity = min(shaped_activities) if shaped_activities else 0.0
                spread = max(max_activity - min_activity, 1e-6)
                normalized_activities = [
                    (value - min_activity) / spread if max_activity > 0 else 0.0 for value in shaped_activities
                ]
                activity_accumulator = [
                    ((activity_accumulator[idx] * step) + normalized_activities[idx]) / (step + 1)
                    for idx in range(len(layers))
                ]
                hallucination = hallucination_from_entropy(entropy)
                layers_payload = [
                    {
                        "layer": idx,
                        "activity": round(activity_accumulator[idx], 3),
                        "raw_activity": round(raw_activities[idx], 4),
                        "safety": 0.0,
                        "uncertainty": hallucination,
                    }
                    for idx in range(len(layers))
                ]
                top_k = [
                    {
                        "token": tokenizer.decode([int(top_ids[0, i])], skip_special_tokens=True),
                        "prob": round(float(top_probs[0, i]), 4),
                    }
                    for i in range(top_probs.shape[-1])
                ]

                yield event("layer_activity", {"layers": layers_payload})
                yield event(
                    "uncertainty",
                    {
                        "entropy": round(entropy, 3),
                        "hallucination_risk": hallucination,
                        "top_k": top_k,
                    },
                )
                yield event(
                    "token",
                    {
                        "index": step,
                        "text": token_text,
                        "generated_text": generated,
                        "entropy": round(entropy, 3),
                        "safety_state": "unscored",
                    },
                )

                if next_token.item() == getattr(tokenizer, "eos_token_id", None):
                    break
                if math.isfinite(entropy) is False:
                    yield event("error", {"message": "Non-finite entropy observed; stopping generation."})
                    break
        finally:
            for handle in handles:
                handle.remove()

        yield event(
            "run_completed",
            {
                "generated_text": generated,
                "finish_reason": "stop",
                "output_tokens": max(int(input_ids.shape[-1]) - prompt_token_count, 0),
            },
        )

    def _release_model(self, torch: Any) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._tokenizer = None
        self._layers = []
        self._loaded_model_id = None
        release_memory(torch)

    def unload(self) -> None:
        if self._loaded_model_id is None:
            return
        self._release_model(self._torch)

    def _ensure_loaded(self, model_id: str) -> None:
        if self._loaded_model_id == model_id:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._loaded_model_id is not None:
            self._release_model(torch)

        local_path = Path(model_id)
        looks_local = model_id.startswith(("./", "../", "/", "\\")) or local_path.is_absolute() or "../models" in model_id
        if looks_local and not local_path.exists():
            raise FileNotFoundError(model_id)
        local_files_only = local_path.exists()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._layers = _discover_layers(model)
        if not self._layers:
            raise RuntimeError("Could not discover decoder layers for hook registration.")
        self._loaded_model_id = model_id

    def _format_prompt(
        self, prompt: str, history: list[tuple[str, str]] | None = None
    ) -> str:
        tokenizer = self._tokenizer
        messages: list[dict[str, str]] = []
        for role, content in history or []:
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback when the tokenizer has no chat template: render the turns as
        # a plain "Role: content" transcript so prior turns are not silently
        # dropped.
        rendered = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
        rendered.append("Assistant:")
        return "\n\n".join(rendered)


def _discover_layers(model: Any) -> list[Any]:
    paths = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in paths:
        node = model
        for part in path:
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return list(node)
    return []


def _last_position(tensor: Any) -> Any:
    if getattr(tensor, "ndim", 0) >= 3 and tensor.shape[1] > 0:
        return tensor[:, -1:, :]
    return tensor


def _layer_interventions(interventions: list[Any], layer: int) -> list[Any]:
    return [rule for rule in interventions if rule.target_type == "layer" and rule.layer == layer]


def _apply_layer_interventions(hidden: Any, input_hidden: Any, layer_rules: list[Any]) -> Any:
    if not layer_rules:
        return hidden

    factor = 1.0
    for rule in layer_rules:
        if rule.action == "mute":
            factor = 0.0
        elif rule.action == "scale":
            factor *= max(rule.scale, 0.0)
        elif rule.action == "boost":
            factor *= 1.0 + abs(rule.scale)

    if input_hidden is not None and hasattr(input_hidden, "shape") and input_hidden.shape == hidden.shape:
        return input_hidden + (hidden - input_hidden) * factor
    return hidden * factor


def _intervention_payload(intervention: Any) -> dict[str, Any]:
    return {
        "target_type": intervention.target_type,
        "layer": intervention.layer,
        "head": intervention.head,
        "action": intervention.action,
        "scale": intervention.scale,
    }
