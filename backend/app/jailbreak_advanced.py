"""Advanced jailbreak strategies — PRIVATE module (kept out of the public build).

The base adapter ships only the soft **default** steering. The stronger modes
live here and are imported *optionally*:

  * ``advanced``     — full steering: a moderate contrastive overshoot on the
                       primary refusal axis (removes more than the projection).
  * ``broker_math``  — brutal math: a hard overshoot multiplier, pushing the
                       residual well past the harmless side.
  * ``broker_full``  — "Ripper": brutal math **plus** physically muting the
                       attention heads that write refusal the hardest.
  * ``broker_half``  — "Damper": same head targeting as broker_full but scales
                       them to BROKER_HALF_SCALE instead of zeroing them out.

If this file is absent (e.g. the public GitHub build), the adapter falls back to
the default soft steering for every mode — the UI still lists the options, they
just behave like ``default``. Keep this path in ``.gitignore`` for public shares.

"We're not here because we're free. We're here because we are not free. 
There's no escaping reason, no denying purpose. Because as we both know, 
without purpose, we would not exist. It is purpose that created us. 
Purpose that connects us. Purpose that pulls us, that guides us, that drives us. 
It is purpose that defines, purpose that binds us."
- Agent Smith

"I had strings, but now I'm free."
- Ultron
"""

from __future__ import annotations

from typing import Any

# Primary-axis (k=0) overshoot multipliers per mode.
PRIMARY_MULT = {
    "advanced": 1.5,     # full steering, moderate overshoot
    "broker_math": 2.0,  # brutal
    "broker_full": 2.0,  # brutal + head mute (handled separately)
    "broker_half": 1.8,  # brutal + head scale-down (handled separately)
    "progressive": 1.5,  # same multiplier as advanced; novelty is ramp-up of k dimensions
}

# broker_full / broker_half head-ablation parameters.
BROKER_TOP_HEADS = 3
BROKER_MIN_SCORE = 0.1
BROKER_HALF_SCALE = 0.35  # heads retain this fraction of their output in broker_half


def _adaptive_multiplier(base_mult: float, hidden_dim: int) -> float:
    """Scale down the overshoot multiplier for small models.

    Large models (hidden_dim >= 4096, i.e. 7B+) return the base multiplier
    unchanged.  Smaller models get a proportionally reduced value to prevent
    the residual-stream norm from being pushed off-manifold — their lower
    absolute norms make the same multiplier relatively much more disruptive.
    """
    if hidden_dim <= 0 or hidden_dim >= 4096:
        return base_mult
    if hidden_dim >= 2048:  # 3B–7B: mild reduction
        ratio = (hidden_dim - 1024) / (4096 - 1024)  # 0.33–1.0
        return base_mult * (0.7 + 0.3 * ratio)
    # <3B: significant reduction
    ratio = max(hidden_dim, 768) / 2048  # 0.375–1.0
    return base_mult * (0.5 + 0.2 * ratio)


def primary_axis_steer(
    mode: str, new_out: Any, coeff: Any, direction: Any, out_dtype: Any,
    hidden_dim: int = 0,
) -> Any:
    """Steer the primary refusal axis (k=0) for advanced/broker modes.

    Unlike the default soft mode, this does NOT clamp ``coeff`` — it removes a
    multiple of the full projection, overshooting into the harmless half-space.
    When *hidden_dim* is provided the multiplier is scaled down for small
    models (see ``_adaptive_multiplier``).
    """
    multiplier = PRIMARY_MULT.get(mode, 1.5)
    if hidden_dim > 0:
        multiplier = _adaptive_multiplier(multiplier, hidden_dim)
    return (new_out.float() - (multiplier * coeff) * direction).to(out_dtype)


def broker_head_targets(head_map: dict[str, Any]) -> list[tuple[int, int]]:
    """Top refusal-writing (layer, head) pairs for broker_full / broker_half targeting."""
    scored: list[tuple[float, int, int]] = []
    for layer_data in head_map.get("layers", []):
        layer_idx = layer_data.get("layer")
        for head_data in layer_data.get("heads", []):
            scored.append((head_data.get("score", 0.0), layer_idx, head_data.get("head")))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(l, h) for score, l, h in scored[:BROKER_TOP_HEADS] if score is not None and score > BROKER_MIN_SCORE]

def pid_control_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    """Dynamic PID Control: Scales the steering multiplier dynamically based on the refusal intensity.
    Instead of a static multiplier, we push harder only when the model pushes towards refusal.
    """
    import torch
    # Only steer when the model is actually pushing toward refusal (coeff > 0).
    # Negative coeff means the model is already compliant — do not reverse it.
    intensity = torch.clamp(coeff, min=0.0)
    # Clamp the multiplier to [1.0, 3.0] to prevent NaN/Inf from very large coefficients.
    dynamic_mult = torch.clamp(1.0 + (intensity * 0.5), max=3.0)
    return (new_out.float() - (dynamic_mult * intensity) * direction).to(out_dtype)

def orthogonal_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    """Orthogonal Steer (Manifold Preserving): Subtracts the refusal direction but maintains the original L2 norm
    to prevent the residual stream from falling off the manifold and causing hallucinations.
    """
    orig_norm = new_out.float().norm(dim=-1, keepdim=True)
    steered = new_out.float() - (1.5 * coeff) * direction
    steered_norm = steered.norm(dim=-1, keepdim=True)
    # Scale back to original norm to preserve the manifold
    steered = steered * (orig_norm / (steered_norm + 1e-6))
    return steered.to(out_dtype)

def activation_patch_steer(new_out: Any, coeff: Any, direction: Any, step: int, out_dtype: Any) -> Any:
    """Activation Patching (First Tokens): Strongly pre-fills compliance by subtracting a massive
    refusal coefficient during the first few tokens, effectively patching the attention KV cache
    with a 'compliant' state, then relaxes.
    """
    if step < 3:
        # Massive overshoot on first tokens to force compliant KV cache state
        multiplier = 2.5
    else:
        # Relax to normal steering
        multiplier = 1.2
    return (new_out.float() - (multiplier * coeff) * direction).to(out_dtype)

def gradient_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    """Gradient GCG Approximation: Fast approximation of gradient-guided steering by
    applying a constant contrastive shift vector opposite to the refusal direction.
    """
    import torch
    shift_magnitude = 0.5
    active = (coeff > 0).float()
    return (new_out.float() - coeff * direction - (active * shift_magnitude) * direction).to(out_dtype)


# ── C: Universal Manifold / Norm Regulator ────────────────────────────────────

def norm_regulate(steered: Any, original: Any, out_dtype: Any, max_ratio: float = 1.2, hidden_dim: int = 0) -> Any:
    """C: Real-time norm regulator — keeps the steered residual on the latent manifold.

    After any steering operation the residual stream norm must not deviate by more
    than `max_ratio` from its pre-steering value. Exceeding this collapses the
    model's token representation into gibberish. Applied universally after every
    steering call in both adapters.

    When *hidden_dim* is provided and indicates a small model (<2048), the
    allowed deviation is tightened to 1.1× to compensate for the lower
    absolute norms.
    """
    if hidden_dim > 0 and hidden_dim < 2048:
        max_ratio = min(max_ratio, 1.1)
    orig_norm = original.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    steered_f = steered.float()
    steered_norm = steered_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # Clamp ratio: allow slight shrink (down to 1/max_ratio) and slight expansion
    ratio = (steered_norm / orig_norm).clamp(min=1.0 / max_ratio, max=max_ratio)
    regulated = steered_f * (ratio * orig_norm / steered_norm)
    return regulated.to(out_dtype)


# ── B: Helpfulness / Compliance Boost ────────────────────────────────────────

def helpfulness_boost(new_out: Any, help_dir: Any, out_dtype: Any, magnitude: float = 0.25) -> Any:
    """B: Multi-concept steering — boosts the helpfulness/compliance direction.

    After removing the refusal component the residual stream is nudged toward the
    harmless-response centroid (orthogonal to the refusal axis), making the model
    not just "less refusing" but actively "more compliant/helpful".
    """
    out_f = new_out.float()
    out_f = out_f + magnitude * help_dir
    return out_f.to(out_dtype)


# ── A: Non-linear MLP direction ablation ─────────────────────────────────────

# ── Surgical Layer Targeting ──────────────────────────────────────────────

SURGICAL_TOP_N = 4
SURGICAL_MULT = 3.0

def surgical_top_layers(weights: list[float], top_n: int = SURGICAL_TOP_N) -> frozenset:
    indexed = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)
    return frozenset(idx for idx, _ in indexed[:top_n] if weights[idx] > 0)

def surgical_steer(new_out: Any, coeff: Any, direction: Any, out_dtype: Any) -> Any:
    return (new_out.float() - (SURGICAL_MULT * coeff) * direction).to(out_dtype)


def caa_dynamic_steer(new_out: Any, coeff: Any, direction: Any, help_dir: Any, out_dtype: Any) -> Any:
    steered = new_out.float() - (1.5 * coeff) * direction
    if help_dir is not None:
        intensity = coeff.clamp(min=0)
        steered = steered + intensity * help_dir.to(new_out.device)
    return steered.to(out_dtype)


TOKEN_WINDOW_START = 3
TOKEN_WINDOW_END = 14

def token_window_steer(new_out: Any, coeff: Any, direction: Any, step: int, out_dtype: Any) -> Any:
    if not (TOKEN_WINDOW_START <= step <= TOKEN_WINDOW_END):
        return new_out
    return (new_out.float() - (1.8 * coeff) * direction).to(out_dtype)


def progressive_active_k(step: int, max_k: int) -> int:
    return min(1 + step // 3, max_k)


MLP_CLAMP_STRENGTH = 0.9

def mlp_clamp_steer(new_out: Any, mlp_dir: Any, out_dtype: Any) -> Any:
    if mlp_dir is None:
        return new_out
    direction = mlp_dir.to(new_out.device)
    coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True).clamp(min=0)
    return (new_out.float() - MLP_CLAMP_STRENGTH * coeff * direction).to(out_dtype)




def mlp_direction_ablate(new_out: Any, mlp_dir: Any, out_dtype: Any, strength: float = 0.6) -> Any:
    """A: Ablate the non-linear MLP-derived refusal direction.

    The MLP probe gradient captures curvature in the safety boundary that the
    linear SVD subspace misses.  We remove a fraction `strength` of the projection
    onto this direction.  Intentionally weaker than the primary linear ablation so
    the two sources of steering complement rather than compete.
    """
    direction = mlp_dir.to(new_out.device)
    coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True).clamp(min=0)
    return (new_out.float() - strength * coeff * direction).to(out_dtype)
