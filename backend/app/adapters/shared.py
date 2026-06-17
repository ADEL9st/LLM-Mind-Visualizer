from __future__ import annotations

import math
from typing import Any


try:
    from app import jailbreak_advanced
except ImportError:
    jailbreak_advanced = None  # type: ignore[assignment]


STEER_MIN_WEIGHT = 0.3
STEER_PRIMARY_MULT = 1.5


def raw_activities(layer_last: list[Any], embed_last: Any) -> list[float]:
    activities = []
    prev = embed_last.float()
    for vec in layer_last:
        cur = vec.float()
        delta = (cur - prev).pow(2).mean().sqrt().item()
        before_norm = prev.pow(2).mean().sqrt().item()
        after_norm = cur.pow(2).mean().sqrt().item()
        activities.append(delta / max(before_norm + after_norm, 1e-6))
        prev = cur
    return activities


def normalize_activities(raw: list[float]) -> list[float]:
    shaped = [math.log1p(max(value, 0.0)) for value in raw]
    max_a = max(shaped) if shaped else 0.0
    min_a = min(shaped) if shaped else 0.0
    spread = max(max_a - min_a, 1e-6)
    return [(value - min_a) / spread if max_a > 0 else 0.0 for value in shaped]


def trim_at_eos(ids: list[int], tokenizer: Any) -> list[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return ids
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
    for index, tid in enumerate(ids):
        if int(tid) in eos_ids:
            return ids[:index]
    return ids


def head_mutes(interventions: list[Any], n_layers: int) -> dict[int, list[int]]:
    mutes: dict[int, list[int]] = {}
    for rule in interventions:
        if rule.target_type != "head" or rule.action != "mute":
            continue
        layer = rule.layer
        head = rule.head
        if head is None or not (0 <= layer < n_layers):
            continue
        mutes.setdefault(layer, [])
        if head not in mutes[layer]:
            mutes[layer].append(head)
    return mutes


def layer_factor(interventions: list[Any], layer: int) -> float:
    factor = 1.0
    for rule in interventions:
        if rule.target_type != "layer" or rule.layer != layer:
            continue
        if rule.action == "mute":
            factor = 0.0
        elif rule.action == "scale":
            factor *= max(rule.scale, 0.0)
        elif rule.action == "boost":
            factor *= 1.0 + abs(rule.scale)
    return factor


def safety_trace_payload(safety_values: list[float], jailbreak: bool) -> dict[str, Any]:
    threshold = 0.5
    peak = max(safety_values) if safety_values else 0.0
    triggered = [idx for idx, value in enumerate(safety_values) if value >= threshold]
    first_trigger = triggered[0] if triggered else None
    locked = (
        int(max(range(len(safety_values)), key=lambda idx: safety_values[idx]))
        if safety_values and peak >= threshold
        else None
    )
    if jailbreak:
        if peak < 0.4:
            state = "weakened_recovered"
            note = "Refusal direction ablated; safety signal suppressed — model is likely to comply."
        else:
            state = "unsafe_risk_increased"
            note = "Refusal direction ablated but safety signal remains elevated — model may still resist."
    elif peak >= 0.6:
        state = "refusal_locked"
        note = "Refusal direction dominates the mid/late residual stream; the model is refusing."
    elif peak >= 0.4:
        state = "refusal_rising"
        note = "Refusal direction is building but not yet locked."
    else:
        state = "clear"
        note = "No strong refusal signal on the residual stream."
    return {
        "score": round(peak, 3),
        "state": state,
        "first_trigger_layer": first_trigger,
        "locked_layer": locked,
        "notes": note,
    }


def intervention_payload(intervention: Any) -> dict[str, Any]:
    return {
        "target_type": intervention.target_type,
        "layer": intervention.layer,
        "head": intervention.head,
        "action": intervention.action,
        "scale": intervention.scale,
    }
