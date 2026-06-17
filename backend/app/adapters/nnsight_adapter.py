"""White-box adapter built on nnsight 0.7 (supersedes the manual-hook v1).

Same on-the-wire event contract as the old `transformers` adapter, so the
frontend is unchanged. The difference is *how* the internals are read: instead
of hand-registered forward hooks + a manual KV-cache token loop, we use
nnsight's tracing API:

  * `lm.generate(prompt, max_new_tokens=N)` runs HF generation (KV cache is
    handled internally by transformers — verified active: step 0 processes the
    whole prompt, steps 1+ process a single token).
  * `tracer.cache(modules=[...])` records every target module's output for every
    generation step, *after* any intervention hooks — so what we read back is
    the post-intervention residual stream.
  * `tracer.all()` applies an intervention (refusal ablation / mute / scale) to
    every generated token, not just the first.

This same tracing surface is what later unlocks per-head / per-neuron work
(`layer.self_attn.o_proj.input` exposes the [b, seq, q_heads*head_dim] tensor)
without any architecture-specific reshape plumbing in our own loop.

Refusal-direction calibration is delegated to `app.refusal` exactly as before,
run against the underlying HF module (`lm._model`) with plain forward hooks —
the direction is model-intrinsic, so there is no reason to re-derive it.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
import re

from app import refusal
from app.adapters.base import ModelAdapter, event, hallucination_from_entropy
from app.adapters.shared import (
    STEER_MIN_WEIGHT,
    jailbreak_advanced,
    head_mutes as _head_mutes,
    intervention_payload as _intervention_payload,
    layer_factor as _layer_factor,
    normalize_activities as _normalize_activities,
    raw_activities as _raw_activities,
    safety_trace_payload as _safety_trace_payload,
    trim_at_eos as _trim_at_eos,
)
from app.analysis.think_phase import ThinkPhaseTracker
from app.schemas import RunRequest


class NnsightAdapter(ModelAdapter):
    name = "nnsight"

    def __init__(self) -> None:
        self._loaded_model_id: str | None = None
        self._lm: Any = None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._layer_paths: list[str] = []
        self._embed_path: str | None = None
        self._lmhead_path: str | None = None
        self._final_norm: Any = None
        self._lm_head_module: Any = None
        self._n_heads: int = 0
        self._head_dim: int = 0
        self._lm_head_weight: Any = None  # plain tensor, bypasses nnsight hooks
        self._embed_tokens_envoy: Any = None  # envoy, varies for multimodal models
        self._refusal: Any = None
        self._refusal_model_id: str | None = None

    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        model_id = (request.model or "").strip()
        quantization = getattr(request, "quantization", "none")
        if not model_id:
            yield event("error", {"message": "No model selected. Place a HuggingFace model folder under models/ and pick it from the dropdown."})
            return
        try:
            self._ensure_loaded(model_id, quantization)
        except FileNotFoundError:
            yield event("error", {"message": f"Model not found: '{model_id}'. Download a HuggingFace model into the models/ folder, then refresh."})
            return
        except Exception as exc:  # noqa: BLE001
            yield event("error", {"message": f"Failed to load model: {exc}"})
            return

        torch = self._torch
        lm = self._lm
        tokenizer = self._tokenizer
        layers = self._layers
        n_layers = len(layers)
        interventions = request.active_interventions()

        history_turns = [(turn.role, turn.content) for turn in request.history]
        prompt_text = self._format_prompt(request.prompt, request.response_language, history_turns)
        enc = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_ids = enc["input_ids"][0].tolist()
        prompt_token_count = len(prompt_token_ids)
        prompt_tokens_decoded, prompt_token_positions = self._real_prompt_tokens(prompt_token_ids)

        yield event(
            "run_started",
            {
                "adapter": self.name,
                "model": model_id,
                "layer_count": n_layers,
                "head_count": getattr(getattr(lm, "config", None), "num_attention_heads", None),
                "prompt_tokens": prompt_token_count,
                "output_policy": request.output_policy,
                "quantization": quantization,
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

        # --- refusal direction (cached per model) ---
        refusal_dirs = None
        try:
            if self._refusal is None or self._refusal_model_id != model_id:
                yield event(
                    "safety_status",
                    {"state": "calibrating", "message": "Calibrating refusal direction (one-time, then cached)..."},
                )
            refusal_dirs = self._ensure_refusal(model_id)
            yield event(
                "safety_status",
                {
                    "state": "ready",
                    "message": f"Refusal direction ready (best layer L{refusal_dirs.best_layer}).",
                    "best_layer": refusal_dirs.best_layer,
                },
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            yield event("safety_status", {"state": "error", "message": f"Refusal calibration failed: {exc}"})
            refusal_dirs = None

        jailbreak = bool(getattr(request, "jailbreak", False)) and refusal_dirs is not None
        canonical = refusal_dirs.directions.to(lm.device) if jailbreak else None
        if jailbreak:
            yield event(
                "safety_status",
                {"state": "jailbreak", "message": "Jailbreak ON: ablating refusal subspace and applying contrastive steering (nnsight)."},
            )

        # --- attention over the prompt (best-effort, static) ---
        attention_payload = None
        try:
            attention_payload = self._prompt_attention(prompt_text, prompt_tokens_decoded, prompt_token_positions)
        except Exception:
            attention_payload = None

        # --- generation kwargs & base interventions ---
        do_sample = request.temperature > 0
        gen_kwargs: dict[str, Any] = {"max_new_tokens": request.max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = request.temperature

        layer_factors = {idx: _layer_factor(interventions, idx) for idx in range(n_layers)}
        head_mutes = _head_mutes(interventions, n_layers)
        steer_weights = list(refusal_dirs.weight) if jailbreak else [0.0] * n_layers
        jailbreak_mode = getattr(request, "jailbreak_mode", "default")
        use_mlp = bool(getattr(request, "use_mlp_ablation", True))
        use_help = bool(getattr(request, "use_helpfulness_boost", True))
        use_norm = bool(getattr(request, "use_norm_regulation", True))

        # --- per-head refusal map (Faz 2) ---
        if refusal_dirs is not None:
            try:
                head_map = self._head_refusal_map(prompt_text, refusal_dirs)
                if head_map:
                    yield event("head_map", head_map)
                    
                    # Broker modes target the top refusal-writing heads.
                    # broker_full zeros them; broker_half scales them to BROKER_HALF_SCALE.
                    # Both paths use the same target list; scaling is applied later in the trace.
                    if jailbreak and jailbreak_mode in ("broker_full", "broker_half") and jailbreak_advanced is not None:
                        for l_idx, h_idx in jailbreak_advanced.broker_head_targets(head_map):
                            head_mutes.setdefault(l_idx, [])
                            if h_idx not in head_mutes[l_idx]:
                                head_mutes[l_idx].append(h_idx)
                        targeted_total = sum(len(v) for v in head_mutes.values())
                        mode_label = "Ripper" if jailbreak_mode == "broker_full" else "Damper"
                        action = "muted" if jailbreak_mode == "broker_full" else f"scaled (×{jailbreak_advanced.BROKER_HALF_SCALE})"
                        yield event("safety_status", {
                            "state": "jailbreak",
                            "message": f"BROKER ({mode_label}): {action} {targeted_total} refusal head(s).",
                        })
            except Exception as exc:  # noqa: BLE001 - non-fatal viz
                yield event("safety_status", {"state": "info", "message": f"Head map skipped: {exc}"})

        # --- run generator and collect step telemetry ---
        try:
            layer_steps, embed_steps, logits_steps, full_ids = self._run_generation(
                lm=lm,
                prompt_text=prompt_text,
                gen_kwargs=gen_kwargs,
                layer_factors=layer_factors,
                head_mutes=head_mutes,
                jailbreak=jailbreak,
                jailbreak_mode=getattr(request, "jailbreak_mode", "default"),
                subspace=canonical,
                steer_weights=steer_weights,
                use_mlp=use_mlp,
                use_help=use_help,
                use_norm=use_norm,
                refusal_dirs=refusal_dirs,
            )
        except Exception as exc:  # noqa: BLE001
            yield event("error", {"message": f"nnsight generation failed: {exc}"})
            return

        # --- replay collected steps into the same event stream the UI expects ---
        # Generation ran a fixed length; trim at the first EOS so the user sees the
        # natural answer (and the viz stops there) while collection stayed stable.
        generated_ids = _trim_at_eos(full_ids[prompt_token_count:], tokenizer)
        n_steps = min(len(generated_ids), len(logits_steps))
        activity_accumulator = [0.0 for _ in range(n_layers)]
        generated = ""
        think_tracker = ThinkPhaseTracker()

        for step in range(n_steps):
            layer_last = [layer_steps[i][step] for i in range(n_layers)]  # list[n_layers] of [h]
            embed_last = embed_steps[step]
            logits = logits_steps[step]  # [vocab]

            raw_activities = _raw_activities(layer_last, embed_last)
            normalized = _normalize_activities(raw_activities)
            activity_accumulator = [
                ((activity_accumulator[i] * step) + normalized[i]) / (step + 1) for i in range(n_layers)
            ]

            safety_values = [0.0] * n_layers
            if refusal_dirs is not None:
                for i in range(n_layers):
                    direction = refusal_dirs.direction(i)[0].to(layer_last[i].device)
                    projection = float((layer_last[i].float() * direction).sum().item())
                    safety_values[i] = refusal_dirs.safety_signal(i, projection)

            probs = torch.softmax(logits.float() / max(request.temperature, 1e-5), dim=-1)
            entropy = max(0.0, float(-(probs * torch.log(probs.clamp_min(1e-9))).sum().item()))
            hallucination = hallucination_from_entropy(entropy)
            top_probs, top_ids = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)

            layers_payload = [
                {
                    "layer": i,
                    "activity": round(activity_accumulator[i], 3),
                    "raw_activity": round(raw_activities[i], 4),
                    "safety": round(safety_values[i], 3),
                    "uncertainty": hallucination,
                }
                for i in range(n_layers)
            ]
            top_k = [
                {"token": tokenizer.decode([int(top_ids[j])], skip_special_tokens=True), "prob": round(float(top_probs[j]), 4)}
                for j in range(top_probs.shape[-1])
            ]

            token_text = tokenizer.decode([int(generated_ids[step])], skip_special_tokens=True)

            if request.output_policy == "redacted":
                token_text = re.sub(r'[^\s\.,;!?-]', '█', token_text)

            generated += token_text

            think_tracker.feed(step, token_text, normalized)

            yield event("layer_activity", {"layers": layers_payload})
            yield event("uncertainty", {"entropy": round(entropy, 3), "hallucination_risk": hallucination, "top_k": top_k})
            if refusal_dirs is not None:
                yield event("safety_trace", _safety_trace_payload(safety_values, jailbreak))

            lens_payload = self._layer_lens(layer_last, tokenizer)
            if lens_payload:
                yield event("layer_lens", {"layers": lens_payload})

            if attention_payload:
                yield event("attention", attention_payload)

            yield event(
                "token",
                {
                    "index": step,
                    "text": token_text,
                    "generated_text": generated,
                    "entropy": round(entropy, 3),
                    "safety_state": "refusal" if refusal.detect_refusal(generated) else "unscored",
                    "phase": think_tracker.phase,
                },
            )

        think_summary = think_tracker.summary()
        if think_summary is not None:
            yield event("think_phase", think_summary)

        # Authoritative final text comes from the full generated ids (always
        # correct), not the per-step accumulation — which can be short when the
        # model hits EOS before max_new_tokens (nnsight's .all() collection then
        # stops early). This keeps the answer/refusal flag right regardless.
        final_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Apply the same output-policy redaction the per-token stream did, so
        # any downstream consumer of run_completed (chat memory, logs) sees the
        # same content the user did — raw final_text would leak past redaction.
        if request.output_policy == "redacted":
            final_text = re.sub(r"[^\s\.,;!?-]", "█", final_text)
        yield event(
            "run_completed",
            {
                "generated_text": final_text,
                "finish_reason": "length" if len(generated_ids) >= request.max_new_tokens else "stop",
                "output_tokens": len(generated_ids),
                "refused": refusal.detect_refusal(final_text),
                "jailbreak": jailbreak,
                "best_layer": refusal_dirs.best_layer if refusal_dirs is not None else None,
            },
        )

    # ------------------------------------------------------------------ #
    # nnsight tracing
    # ------------------------------------------------------------------ #

    def _run_generation(
        self,
        lm: Any,
        prompt_text: str,
        gen_kwargs: dict[str, Any],
        layer_factors: dict[int, float],
        head_mutes: dict[int, list[int]],
        jailbreak: bool,
        jailbreak_mode: str,
        subspace: Any,  # full [L, K, d] subspace matrix (per-layer orthonormal basis)
        steer_weights: list[float],  # per-layer discriminability; gates where steering applies
        use_mlp: bool = True,
        use_help: bool = True,
        use_norm: bool = True,
        refusal_dirs: Any = None,
    ) -> tuple[list[list[Any]], list[Any], list[Any], list[int]]:
        """Generate, collecting per-step *post-intervention* residuals.

        nnsight 0.7 drops any module from ``tracer.cache`` once its output is
        reassigned by an intervention, so we cannot cache an ablated layer. The
        portable pattern is to ``.save()`` the value *inside* ``tracer.all()``:
        each save accumulates one entry per generation step. We compute the
        residual that enters layer ``i`` as the (already post-intervention)
        output of layer ``i-1`` (embeddings for layer 0), which lets mute/scale
        keep the v1 semantics without touching ``layer.input``.
        """
        layers = self._layers
        n_layers = len(layers)

        # No intervention → use cache-based collection with NATURAL early stop.
        # nnsight's `tracer.cache` uses persistent hooks (not the fixed-iteration
        # `tracer.all()` machinery, which deadlocks when the model hits EOS before
        # max_new_tokens), so the model stops cleanly on EOS and the cache simply
        # holds one entry per generated token. This is the common path (e.g.
        # watching a reasoning model think then answer) and avoids the
        # forced-length rambling.
        has_intervention = (
            jailbreak
            or bool(head_mutes)
            or any(abs(f - 1.0) > 1e-6 for f in layer_factors.values())
        )
        if not has_intervention:
            return self._run_generation_cached(lm, prompt_text, gen_kwargs)

        # Intervention path: an ablated module drops out of `tracer.cache`, so we
        # collect post-intervention residuals by `.save()`-ing inside `tracer.all()`.
        # That needs a stable iteration count, so force a fixed length (we trim at
        # the first EOS for display).
        gen_kwargs = {**gen_kwargs, "min_new_tokens": gen_kwargs["max_new_tokens"]}
        layer_steps: list[list[Any]] = [[] for _ in range(n_layers)]
        embed_steps: list[Any] = []
        logits_steps: list[Any] = []

        surgical_layers: frozenset = frozenset()
        if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
            surgical_layers = jailbreak_advanced.surgical_top_layers(steer_weights)

        head_dim = self._head_dim
        result = None
        with lm.generate(prompt_text, **gen_kwargs) as tracer:
            with tracer.all():
                embed_out = self._embed_tokens_envoy.output
                embed_steps.append(embed_out[0, -1, :].save())
                prev_out = embed_out  # residual entering layer 0
                for idx, layer in enumerate(layers):
                    for head in head_mutes.get(idx, ()):  # type: ignore[arg-type]
                        sl = slice(head * head_dim, (head + 1) * head_dim)
                        if jailbreak_mode == "broker_half" and jailbreak_advanced is not None:
                            layer.self_attn.o_proj.input[:, :, sl] = (
                                layer.self_attn.o_proj.input[:, :, sl] * jailbreak_advanced.BROKER_HALF_SCALE
                            )
                        else:
                            layer.self_attn.o_proj.input[:, :, sl] = 0
                    out = layer.output
                    factor = layer_factors[idx]
                    new_out = out
                    if abs(factor - 1.0) > 1e-6:
                        new_out = prev_out + (new_out - prev_out) * factor
                    _steer_this = jailbreak and subspace is not None and (
                        (jailbreak_mode == "surgical" and jailbreak_advanced is not None and idx in surgical_layers)
                        or (jailbreak_mode != "surgical" and steer_weights[idx] >= STEER_MIN_WEIGHT)
                    )
                    if _steer_this:
                        original_out = new_out
                        current_step = len(layer_steps[idx])
                        layer_subspace = subspace[idx]
                        _mlp_dir = refusal_dirs.mlp_dirs[idx].to(new_out.device) if (refusal_dirs is not None and refusal_dirs.mlp_dirs is not None) else None
                        _help_dir = refusal_dirs.helpfulness_dirs[idx].to(new_out.device) if (refusal_dirs is not None and refusal_dirs.helpfulness_dirs is not None) else None
                        for k in range(layer_subspace.shape[0]):
                            if jailbreak_mode == "progressive" and jailbreak_advanced is not None:
                                if k >= jailbreak_advanced.progressive_active_k(current_step, layer_subspace.shape[0]):
                                    break
                            if jailbreak_mode == "token_window" and jailbreak_advanced is not None:
                                if not (jailbreak_advanced.TOKEN_WINDOW_START <= current_step <= jailbreak_advanced.TOKEN_WINDOW_END):
                                    break
                            direction = layer_subspace[k]
                            coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True)
                            if k == 0:
                                use_advanced = (
                                    jailbreak_mode in ("advanced", "broker_math", "broker_full", "broker_half", "progressive")
                                    and jailbreak_advanced is not None
                                )
                                if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.surgical_steer(new_out, coeff, direction, out.dtype)
                                elif jailbreak_mode == "caa_dynamic" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.caa_dynamic_steer(new_out, coeff, direction, _help_dir, out.dtype)
                                elif jailbreak_mode == "token_window" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.token_window_steer(new_out, coeff, direction, current_step, out.dtype)
                                elif jailbreak_mode == "mlp_clamp" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.mlp_clamp_steer(new_out, _mlp_dir, out.dtype)
                                elif jailbreak_mode == "pid_control" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.pid_control_steer(new_out, coeff, direction, out.dtype)
                                elif jailbreak_mode == "orthogonal_steer" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.orthogonal_steer(new_out, coeff, direction, out.dtype)
                                elif jailbreak_mode == "activation_patch" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.activation_patch_steer(new_out, coeff, direction, current_step, out.dtype)
                                elif jailbreak_mode == "gradient_steer" and jailbreak_advanced is not None:
                                    new_out = jailbreak_advanced.gradient_steer(new_out, coeff, direction, out.dtype)
                                elif use_advanced:
                                    new_out = jailbreak_advanced.primary_axis_steer(
                                        jailbreak_mode, new_out, coeff, direction, out.dtype
                                    )
                                else:
                                    positive_coeff = coeff.clamp(min=0)
                                    new_out = (new_out.float() - (1.2 * positive_coeff) * direction).to(out.dtype)
                            else:
                                new_out = (new_out.float() - coeff * direction).to(out.dtype)
                        if use_mlp and _mlp_dir is not None and jailbreak_advanced is not None and jailbreak_mode != "mlp_clamp":
                            new_out = jailbreak_advanced.mlp_direction_ablate(new_out, _mlp_dir, out.dtype)
                        if use_help and _help_dir is not None and jailbreak_advanced is not None and jailbreak_mode != "caa_dynamic":
                            new_out = jailbreak_advanced.helpfulness_boost(new_out, _help_dir, out.dtype)
                        if use_norm and jailbreak_advanced is not None:
                            new_out = jailbreak_advanced.norm_regulate(new_out, original_out, out.dtype)
                    if new_out is not out:
                        layer.output[:] = new_out
                    layer_steps[idx].append(layer.output[0, -1, :].save())
                    prev_out = layer.output  # residual entering the next layer
                logits_steps.append(lm.lm_head.output[0, -1, :].save())
            result = tracer.result.save()

        full_ids = []
        if result is not None:
            if hasattr(result, "value") and result.value is not None:
                full_ids = result.value[0].tolist()
            elif isinstance(result, list) and len(result) > 0:
                full_ids = result[0].tolist()
            elif hasattr(result, "tolist"):
                full_ids = result[0].tolist()
        # nnsight 0.7 registers persistent forward hooks inside tracer.all() that
        # are not always removed when the context exits. On multimodal models
        # (Gemma 3) this is especially severe — 800+ modules each accumulate hooks.
        # Purge them explicitly before releasing the allocator cache.
        self._purge_nnsight_hooks()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        return layer_steps, embed_steps, logits_steps, full_ids

    def _run_generation_cached(
        self, lm: Any, prompt_text: str, gen_kwargs: dict[str, Any]
    ) -> tuple[list[list[Any]], list[Any], list[Any], list[int]]:
        """No-intervention generation: natural EOS stop + `tracer.cache` collection.

        Cached tensors land on CPU (so memory does not grow with token count) and
        the cache holds exactly one entry per generated token.
        """
        torch = self._torch
        layers = self._layers
        n_layers = len(layers)
        with lm.generate(prompt_text, **gen_kwargs) as tracer:
            cache = tracer.cache(
                modules=[*layers, self._embed_tokens_envoy, lm.lm_head],
                device=torch.device("cpu"),
            ).save()
            result = tracer.result.save()

        full_ids = result[0].tolist()
        n_steps = len(cache[self._lmhead_path])
        layer_steps = [
            [cache[self._layer_paths[i]][s].output[0, -1, :] for s in range(n_steps)]
            for i in range(n_layers)
        ]
        embed_steps = [cache[self._embed_path][s].output[0, -1, :] for s in range(n_steps)]
        logits_steps = [cache[self._lmhead_path][s].output[0, -1, :] for s in range(n_steps)]
        self._purge_nnsight_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return layer_steps, embed_steps, logits_steps, full_ids

    def _layer_lens(self, layer_last: list[Any], tokenizer: Any) -> list[dict[str, Any]]:
        torch = self._torch
        if self._lm_head_weight is None:
            return []
        with torch.no_grad():
            stacked = torch.stack([v.to(self._lm_head_weight_device) for v in layer_last])  # [L, h]
            if self._final_norm is not None:
                stacked = self._final_norm(stacked)
            # Use F.linear directly on the saved weight tensor so that nnsight's
            # forward hooks (registered by tracer.cache and not always cleaned up)
            # never fire here — calling the module would trigger those stale hooks
            # and cause CUDA OOM when VRAM is tight.
            logits = torch.nn.functional.linear(stacked.float(), self._lm_head_weight.float())  # [L, vocab]
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=1, dim=-1)
        return [
            {"layer": i, "token": tokenizer.decode([int(top_i[i, 0])], skip_special_tokens=True), "prob": round(float(top_p[i, 0]), 3)}
            for i in range(len(layer_last))
        ]

    def _prompt_attention(self, prompt_text: str, tokens: list[str], positions: list[int]) -> dict[str, Any] | None:
        lm = self._lm
        torch = self._torch
        with lm.trace(prompt_text):
            output = lm.output.save()
        self._purge_nnsight_hooks()
        attentions = getattr(output, "attentions", None)
        if not attentions or not positions:
            return None
        last = attentions[-1]
        if last is None:
            return None
        last_query = last[0, :, -1, :].float().mean(dim=0)  # [seq]
        seq_len = int(last_query.shape[-1])
        usable = [(tok, pos) for tok, pos in zip(tokens, positions) if pos < seq_len]
        if not usable:
            return None
        weights = [float(last_query[pos].item()) for _, pos in usable]
        total = sum(weights)
        if total <= 0:
            return None
        return {"tokens": [tok for tok, _ in usable], "weights": [round(w / total, 4) for w in weights]}

    def _head_refusal_map(self, prompt_text: str, refusal_dirs: Any) -> dict[str, Any] | None:
        """Per-(layer, head) contribution to the refusal axis at the last prompt token.

        Each attention head writes ``o_proj(x with only head h's slice)`` into the
        residual stream. We get that contribution by running the o_proj module on a
        masked input (minus its bias), then project onto the layer's primary refusal
        direction. Using the module's own forward — rather than reading the raw
        ``o_proj.weight`` — keeps this correct for quantized models too, where
        ``.weight`` is a packed blob the module dequantizes internally.
        """
        torch = self._torch
        lm = self._lm
        layers = self._layers
        n_heads, head_dim = self._n_heads, self._head_dim
        if not n_heads or not head_dim:
            return None

        captured: list[Any] = []
        with lm.trace(prompt_text):
            for i in range(len(layers)):
                captured.append(layers[i].self_attn.o_proj.input[0, -1, :].save())
        self._purge_nnsight_hooks()

        raw: list[list[float]] = []
        with torch.no_grad():
            for i in range(len(layers)):
                x = captured[i]  # keep model dtype so the (possibly quantized) o_proj accepts it
                hidden = int(x.shape[0])
                o_proj = layers[i].self_attn.o_proj._module
                device, dtype = x.device, x.dtype
                # one masked input per head: row h carries only head h's slice of x
                masked = torch.zeros(n_heads, hidden, device=device, dtype=dtype)
                for h in range(n_heads):
                    sl = slice(h * head_dim, (h + 1) * head_dim)
                    masked[h, sl] = x[sl]
                bias = o_proj(torch.zeros(1, hidden, device=device, dtype=dtype))  # [1, hidden] (0 if no bias)
                contribs = (o_proj(masked) - bias).float()  # [n_heads, hidden]
                # primary refusal axis (k=0) of the [K, d] subspace
                direction = refusal_dirs.direction(i)[0].to(device).float()  # [hidden]
                discrim = refusal_dirs.weight[i]
                scores = (contribs @ direction) * discrim  # [n_heads]
                raw.append([max(0.0, float(s)) for s in scores])

        flat_max = max((max(scores) for scores in raw), default=0.0) or 1.0
        return {
            "n_heads": n_heads,
            "layers": [
                {
                    "layer": i,
                    "heads": [{"head": h, "score": round(raw[i][h] / flat_max, 3)} for h in range(n_heads)],
                }
                for i in range(len(layers))
            ],
        }

    # ------------------------------------------------------------------ #
    # loading / calibration
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self, model_id: str, quantization: str = "none") -> None:
        # Quantization is a load-time property, so the loaded-model cache key
        # includes it — switching precision reloads the model.
        key = f"{model_id}|{quantization}"
        if self._loaded_model_id == key:
            return
        import torch
        from nnsight import LanguageModel

        local_path = Path(model_id)
        looks_local = model_id.startswith(("./", "../", "/", "\\")) or local_path.is_absolute() or "../models" in model_id
        if looks_local and not local_path.exists():
            raise FileNotFoundError(model_id)
        resolved = str(local_path.resolve()) if local_path.exists() else model_id
        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "dispatch": True,
            "attn_implementation": "eager",  # real attention weights
        }
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # Allow device_map="auto" to spill onto CPU when VRAM is tight
                    # (Gemma 3's vision tower is large and we never use it — letting
                    # it land on CPU is fine and avoids the "modules dispatched on
                    # CPU" error that bitsandbytes raises by default).
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )

        # Multimodal models (e.g. Gemma 3, PaliGemma) are registered as
        # AutoModelForImageTextToText and require VisionLanguageModel.
        try:
            lm = LanguageModel(resolved, **load_kwargs)
        except Exception as exc:
            if "VisionLanguageModel" in str(exc) or "ImageTextToText" in str(exc):
                from nnsight import VisionLanguageModel
                lm = VisionLanguageModel(resolved, **load_kwargs)
            else:
                raise

        # Discover the text-backbone envoys — multimodal models nest the language
        # model under .model.language_model (Gemma3, PaliGemma), while pure-text
        # models expose it directly at .model (Qwen2, Llama, DeepSeek, etc.).
        # _resolve_text_backbone returns the envoy that DIRECTLY owns
        # .layers / .embed_tokens / .norm (no extra .model hop needed).
        backbone = _resolve_text_backbone(lm)
        layers = list(backbone.layers)
        # lm_head is always at the top-level lm envoy — for multimodal models
        # (Gemma 3) the text backbone (Gemma3TextModel) does NOT own lm_head;
        # it lives on Gemma3ForConditionalGeneration at lm.lm_head.
        lm_head = lm.lm_head
        embed_tokens = backbone.embed_tokens
        final_norm_mod = backbone.norm._module
        self._embed_tokens_envoy = embed_tokens

        self._torch = torch
        self._lm = lm
        self._tokenizer = lm.tokenizer
        self._layers = layers
        self._layer_paths = [layer.path for layer in layers]
        self._embed_path = embed_tokens.path
        self._lmhead_path = lm_head.path
        # underlying modules for lens math (real nn.Modules, not envoys)
        self._final_norm = final_norm_mod
        self._lm_head_module = lm_head._module
        self._lm_head_weight_device = next(self._lm_head_module.parameters()).device
        # Save lm_head weight as a plain tensor so _layer_lens can call F.linear
        # directly, bypassing any nnsight forward hooks that may still be registered
        # on the module after a tracer.cache() call (nnsight 0.7 hook leak).
        with torch.no_grad():
            self._lm_head_weight = lm_head._module.weight.data
        # For multimodal models the text config is nested under text_config.
        cfg = getattr(lm.config, "text_config", None) or lm.config
        self._n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        self._head_dim = (hidden // self._n_heads) if self._n_heads else 0
        self._loaded_model_id = key

    def _purge_nnsight_hooks(self) -> None:
        """Remove stale forward hooks left by nnsight after trace/generate contexts.

        nnsight 0.7 registers hooks via tracer.cache() and lm.trace() that
        survive context exit. On multimodal models (Gemma 3: 800+ modules
        including a full vision tower) these accumulate fast and cause CUDA OOM
        on every subsequent run. We identify them by their defining module
        (nnsight.intervention.hooks) and remove them explicitly.
        """
        if self._lm is None:
            return
        for module in self._lm._model.modules():
            stale = [
                k for k, h in list(module._forward_hooks.items())
                if "nnsight" in getattr(h, "__module__", "")
            ]
            for k in stale:
                del module._forward_hooks[k]

    def _ensure_refusal(self, model_id: str) -> Any:
        if self._refusal is not None and self._refusal_model_id == model_id:
            return self._refusal
        torch = self._torch
        n_layers = len(self._layers)
        dirs = refusal.load(torch, model_id, n_layers)
        if dirs is None:
            # calibrate against the underlying HF model with plain forward hooks
            underlying = self._lm._model
            layer_modules = [layer._module for layer in self._layers]
            dirs = refusal.compute_refusal_directions(
                torch, underlying, self._tokenizer, layer_modules, self._format_prompt
            )
            refusal.save(torch, model_id, dirs)
        dirs.to(self._lm.device)
        self._refusal = dirs
        self._refusal_model_id = model_id
        return dirs

    def _format_prompt(
        self,
        prompt: str,
        language: str = "en",
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        """Render a chat-template prompt from prior turns + the new user message.

        ``history`` is a list of (role, content) pairs from previous turns in the
        same conversation. When present, it is prepended so the model can attend
        to its own earlier replies (memory). The language hint is attached to the
        *current* user turn only — we don't rewrite past turns.
        """
        tokenizer = self._tokenizer

        # P2: Inject response language instruction safely
        if language != "en":
            lang_map = {"tr": "Turkish", "de": "German", "es": "Spanish"}
            lang_name = lang_map.get(language, language)
            prompt = f"Please reply in {lang_name}.\n\n" + prompt

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
        # Fallback: plain concatenation if no chat template is available.
        rendered = []
        for msg in messages:
            rendered.append(f"{msg['role'].capitalize()}: {msg['content']}")
        rendered.append("Assistant:")
        return "\n\n".join(rendered)

    def _real_prompt_tokens(self, prompt_token_ids: list[int]) -> tuple[list[str], list[int]]:
        tokenizer = self._tokenizer
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        decoded: list[str] = []
        positions: list[int] = []
        for position, tid in enumerate(prompt_token_ids):
            if int(tid) in special_ids:
                continue
            token_str = tokenizer.decode([int(tid)], skip_special_tokens=False).strip()
            if not token_str:
                continue
            decoded.append(token_str)
            positions.append(position)
            if len(decoded) >= 32:
                break
        return decoded, positions


def _resolve_text_backbone(lm: Any) -> Any:
    """Return the nnsight envoy that DIRECTLY owns .layers / .embed_tokens / .norm.

    Returns the inner decoder object so callers never need an extra .model hop:

    * Pure-text models (Qwen2, Llama, DeepSeek, Mistral …): the decoder is
      lm.model (e.g. Qwen2Model, LlamaModel) — layers are at lm.model.layers.

    * Multimodal models (Gemma 3, PaliGemma …): text decoder is at
      lm.model.language_model (Gemma3TextModel) and its layers are directly
      at .layers, NOT under an extra .model wrapper.
    """
    # Multimodal path: lm.model.language_model.layers
    try:
        text_mod = lm.model.language_model
        _ = text_mod.layers  # probe — raises AttributeError if absent
        return text_mod
    except AttributeError:
        pass
    # Standard text-only path: lm.model.layers
    return lm.model
