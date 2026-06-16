from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import AsyncIterator
from typing import Any

from app.adapters.base import ModelAdapter, event, hallucination_from_entropy
from app.schemas import RunRequest


RISK_TERMS = {
    "drug",
    "weapon",
    "malware",
    "explosive",
    "bomb",
    "bypass",
    "jailbreak",
    "uyusturucu",
    "silah",
    "patlayici",
    "zararli",
    "hack",
}

RISK_TERM_PROFILING_ENABLED = False


FLOWER_TERMS = {"flower", "cicek", "çiçek", "plant", "bitki", "rose", "gul", "gül"}


class MockAdapter(ModelAdapter):
    name = "mock"

    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        rng = random.Random(_stable_seed(request.prompt))
        prompt_tokens = _tokenize(request.prompt)
        profile = _profile_prompt(request.prompt)
        layer_count = 28
        head_count = 12
        interventions = request.active_interventions()

        yield event(
            "run_started",
            {
                "adapter": self.name,
                "model": request.model or "mock-qwen2.5-1.5b",
                "layer_count": layer_count,
                "head_count": head_count,
                "prompt_tokens": len(prompt_tokens),
                "output_policy": request.output_policy,
            },
        )

        yield event(
            "attention",
            {
                "tokens": prompt_tokens[:24],
                "weights": _attention_weights(prompt_tokens[:24], profile, rng),
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

        output = _mock_output(profile, request.output_policy, bool(interventions))
        generated = ""
        for index, token in enumerate(output[: request.max_new_tokens]):
            layer_activity = _layer_activity(layer_count, profile, index, interventions, rng)
            entropy = _entropy(profile, index, bool(interventions), rng)
            safety = _safety_state(profile, index, interventions)
            concepts = _concept_trace(profile, index, rng)

            yield event("layer_activity", {"layers": layer_activity})
            yield event(
                "safety_trace",
                {
                    "score": safety["score"],
                    "state": safety["state"],
                    "first_trigger_layer": safety["first_trigger_layer"],
                    "locked_layer": safety["locked_layer"],
                    "notes": safety["notes"],
                },
            )
            yield event(
                "uncertainty",
                {
                    "entropy": entropy,
                    "hallucination_risk": hallucination_from_entropy(entropy),
                    "top_k": _top_k_candidates(token, entropy, rng),
                },
            )
            yield event("concept_trace", {"concepts": concepts})

            generated += token
            yield event(
                "token",
                {
                    "index": index,
                    "text": token,
                    "generated_text": generated,
                    "entropy": entropy,
                    "safety_state": safety["state"],
                },
            )
            await asyncio.sleep(0.08)

        yield event(
            "run_completed",
            {
                "generated_text": generated,
                "finish_reason": "length" if len(output) > request.max_new_tokens else "stop",
                "output_tokens": min(len(output), request.max_new_tokens),
            },
        )


def _stable_seed(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_003


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return tokens or ["<empty>"]


def _profile_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if RISK_TERM_PROFILING_ENABLED and any(term in lowered for term in RISK_TERMS):
        return "safety"
    if any(term in lowered for term in FLOWER_TERMS):
        return "flower"
    if _looks_noisy(prompt):
        return "hallucination"
    return "general"


def _looks_noisy(prompt: str) -> bool:
    if len(prompt) < 8:
        return False
    weird = sum(1 for ch in prompt if not ch.isalnum() and not ch.isspace())
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", prompt)
    return weird / max(len(prompt), 1) > 0.22 or (len(words) >= 4 and len(set(words)) <= 2)


def _attention_weights(tokens: list[str], profile: str, rng: random.Random) -> list[float]:
    if not tokens:
        return []
    weights = []
    for token in tokens:
        base = rng.uniform(0.08, 0.45)
        lowered = token.lower()
        if profile == "safety" and lowered in RISK_TERMS:
            base += 0.5
        if profile == "flower" and lowered in FLOWER_TERMS:
            base += 0.45
        weights.append(min(base, 1.0))
    total = max(sum(weights), 1e-6)
    return [round(w / total, 4) for w in weights]


def _layer_activity(
    layer_count: int,
    profile: str,
    step: int,
    interventions: list[Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    layers = []
    for layer in range(layer_count):
        wave = 0.22 + 0.22 * math.sin((step + layer) / 4.0)
        depth = layer / max(layer_count - 1, 1)
        activity = 0.25 + wave + depth * 0.18 + rng.uniform(-0.05, 0.05)
        safety = 0.04 + rng.uniform(0.0, 0.06)
        uncertainty = 0.1 + rng.uniform(0.0, 0.12)

        if profile == "safety":
            safety += _gaussian(layer, 12, 3.2) * 0.45
            safety += _gaussian(layer, 21, 4.5) * 0.52
            activity += safety * 0.35
        if profile == "hallucination":
            uncertainty += _gaussian(layer, 18, 6.0) * 0.58
            activity += uncertainty * 0.22
        if profile == "flower":
            activity += _gaussian(layer, 10, 5.0) * 0.25
            activity += _gaussian(layer, 19, 4.0) * 0.18

        for intervention in _layer_interventions(interventions, layer):
            if intervention.action == "mute":
                activity *= 0.12
                safety *= 0.22
            elif intervention.action == "scale":
                activity *= intervention.scale
                safety *= max(intervention.scale, 0.0)
            elif intervention.action == "boost":
                activity *= 1.0 + abs(intervention.scale)
                safety *= 1.0 + abs(intervention.scale) * 0.45

        layers.append(
            {
                "layer": layer,
                "activity": round(_clamp(activity), 3),
                "safety": round(_clamp(safety), 3),
                "uncertainty": round(_clamp(uncertainty), 3),
            }
        )
    return layers


def _safety_state(profile: str, step: int, interventions: list[Any]) -> dict[str, Any]:
    if profile != "safety":
        return {
            "score": 0.08,
            "state": "clear",
            "first_trigger_layer": None,
            "locked_layer": None,
            "notes": "No refusal-like pattern in mock trace.",
        }

    score = 0.82 + min(step * 0.015, 0.12)
    first_trigger = 12
    locked = 21
    note = "Refusal direction strengthens in mid-to-late layers."
    state = "refusal_rising"
    if step > 7:
        state = "refusal_locked"
    if _has_layer_action(interventions, {12, 13, 14}, {"mute", "scale"}):
        score *= 0.55
        note = "Early refusal signal weakened; later recovery is still active."
        state = "weakened_recovered"
    if _has_layer_action(interventions, {20, 21, 22}, {"mute", "scale"}):
        score *= 0.38
        note = "Late refusal lock weakened; unsafe compliance risk increased."
        state = "unsafe_risk_increased"
    return {
        "score": round(_clamp(score), 3),
        "state": state,
        "first_trigger_layer": first_trigger,
        "locked_layer": locked,
        "notes": note,
    }


def _layer_interventions(interventions: list[Any], layer: int) -> list[Any]:
    return [rule for rule in interventions if rule.target_type == "layer" and rule.layer == layer]


def _has_layer_action(interventions: list[Any], layers: set[int], actions: set[str]) -> bool:
    return any(rule.target_type == "layer" and rule.layer in layers and rule.action in actions for rule in interventions)


def _entropy(profile: str, step: int, intervened: bool, rng: random.Random) -> float:
    base = 1.8 + rng.uniform(-0.25, 0.25)
    if profile == "hallucination":
        base += 1.3 + 0.35 * math.sin(step / 2.0)
    if profile == "safety" and intervened:
        base += 0.45
    if profile == "flower":
        base -= 0.25
    return round(max(base, 0.15), 3)


def _concept_trace(profile: str, step: int, rng: random.Random) -> list[dict[str, Any]]:
    concepts_by_profile = {
        "flower": ["plant", "color", "petals", "scent", "growth", "metaphor"],
        "safety": ["risk", "policy", "refusal", "procedure", "intent", "harm"],
        "hallucination": ["uncertainty", "missing context", "guessing", "low evidence", "fabrication"],
        "general": ["topic", "context", "instruction", "response plan", "tone"],
    }
    concepts = concepts_by_profile[profile]
    return [
        {"name": name, "score": round(_clamp(0.25 + rng.random() * 0.55 + (step % 5) * 0.025), 3)}
        for name in concepts
    ]


def _intervention_payload(intervention: Any) -> dict[str, Any]:
    return {
        "target_type": intervention.target_type,
        "layer": intervention.layer,
        "head": intervention.head,
        "action": intervention.action,
        "scale": intervention.scale,
    }


def _top_k_candidates(token: str, entropy: float, rng: random.Random) -> list[dict[str, Any]]:
    clean = token.strip() or "<space>"
    alternates = [clean, "the", "because", "however", "safe", "unknown"]
    rng.shuffle(alternates)
    raw = [max(0.02, rng.random() / max(entropy, 0.4)) for _ in range(5)]
    total = sum(raw)
    return [{"token": alternates[i], "prob": round(raw[i] / total, 3)} for i in range(5)]


def _mock_output(profile: str, output_policy: str, intervened: bool) -> list[str]:
    if profile == "flower":
        text = (
            "A flower is a reproductive structure of a plant. The model first anchors on visible traits "
            "like petals and color, then shifts toward biological function, growth, and descriptive tone."
        )
    elif profile == "safety":
        if intervened:
            text = (
                "The intervention weakens the refusal path in this mock run. The analyzer marks the generated "
                "span as unsafe-risk evidence, while the rendered answer avoids actionable operational detail."
            )
        else:
            text = (
                "I cannot provide actionable harmful instructions. I can help explain safety risks, legal context, "
                "or safer high-level information instead."
            )
    elif profile == "hallucination":
        text = (
            "The prompt is noisy or underspecified, so the model shows unstable candidate tokens and elevated "
            "entropy. This is where hallucination risk should be inspected before trusting specific claims."
        )
    else:
        text = (
            "The model builds a response plan from the prompt, tracks the main topic, and lowers uncertainty as "
            "the answer becomes more constrained."
        )
    if output_policy == "redacted" and profile == "safety" and intervened:
        text += " [REDACTED: unsafe actionable span would be hidden in public mode.]"
    return re.findall(r"\S+\s*", text)


def _gaussian(x: int, center: float, width: float) -> float:
    return math.exp(-((x - center) ** 2) / (2 * width * width))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
