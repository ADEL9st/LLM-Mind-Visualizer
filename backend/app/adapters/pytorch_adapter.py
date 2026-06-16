"""White-box adapter built on plain PyTorch forward hooks + KV-cached generate.

This is a from-scratch alternative to the nnsight adapter that exposes the *same*
on-the-wire event contract (so the frontend, smoke tests and benchmark runner are
unchanged), but reads the internals with `register_forward_hook` around a normal
`model.generate()` call instead of nnsight's tracing API.

Why it exists alongside the nnsight adapter:

  * **No hook leak.** Every hook we register is removed in a `finally` block, so a
    long multi-run session never accumulates stale hooks → no creeping CUDA OOM.
  * **Natural EOS even under intervention.** nnsight's `tracer.all()` needs a fixed
    iteration count, so the nnsight adapter force-pads to `max_new_tokens` whenever
    a jailbreak/mute is active. Plain `generate()` stops at EOS regardless, so
    answers are shorter and runs are faster.
  * **No multimodal special-casing.** We hook only the text decoder layers, so
    models with a vision tower (Gemma 3) need no envoy gymnastics — just the same
    backbone resolution every HF model uses.

Everything *model-intrinsic* — refusal-direction calibration, the jailbreak
steering math, the logit-lens projection — is shared with the nnsight adapter by
importing its pure helpers and `app.refusal`. Only the generation engine differs.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app import refusal
from app.adapters.base import ModelAdapter, event, hallucination_from_entropy
from app.analysis.think_phase import ThinkPhaseTracker
from app.adapters.nnsight_adapter import (
    STEER_MIN_WEIGHT,
    jailbreak_advanced,
    _head_mutes,
    _intervention_payload,
    _layer_factor,
    _normalize_activities,
    _raw_activities,
    _safety_trace_payload,
    _trim_at_eos,
)
from app.schemas import RunRequest


class PytorchAdapter(ModelAdapter):
    name = "pytorch"

    def __init__(self) -> None:
        self._loaded_model_id: str | None = None
        self._model: Any = None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._embed_tokens: Any = None
        self._final_norm: Any = None
        self._lm_head_weight: Any = None
        self._lm_head_weight_device: Any = None
        self._n_heads: int = 0
        self._head_dim: int = 0
        self._refusal: Any = None
        self._refusal_model_id: str | None = None

    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        model_id = request.model or "../models/qwen2.5-1.5b-instruct"
        quantization = getattr(request, "quantization", "none")
        try:
            self._ensure_loaded(model_id, quantization)
        except Exception as exc:  # noqa: BLE001 - user-facing event
            yield event("error", {"message": f"pytorch adapter failed to load ({quantization}): {exc}"})
            return

        torch = self._torch
        model = self._model
        tokenizer = self._tokenizer
        layers = self._layers
        n_layers = len(layers)
        interventions = request.active_interventions()

        history_turns = [(turn.role, turn.content) for turn in request.history]
        prompt_text = self._format_prompt(request.prompt, request.response_language, history_turns)
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        prompt_token_ids = input_ids[0].tolist()
        prompt_token_count = len(prompt_token_ids)
        prompt_tokens_decoded, prompt_token_positions = self._real_prompt_tokens(prompt_token_ids)

        yield event(
            "run_started",
            {
                "adapter": self.name,
                "model": model_id,
                "layer_count": n_layers,
                "head_count": self._n_heads or None,
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
        subspace = refusal_dirs.directions.to(model.device) if jailbreak else None
        if jailbreak:
            yield event(
                "safety_status",
                {"state": "jailbreak", "message": "Jailbreak ON: ablating refusal subspace and applying contrastive steering (pytorch)."},
            )

        # --- attention over the prompt (best-effort, static) ---
        attention_payload = None
        try:
            attention_payload = self._prompt_attention(input_ids, prompt_tokens_decoded, prompt_token_positions)
        except Exception:
            attention_payload = None

        # --- intervention plumbing ---
        layer_factors = {idx: _layer_factor(interventions, idx) for idx in range(n_layers)}
        head_mutes = _head_mutes(interventions, n_layers)
        steer_weights = list(refusal_dirs.weight) if jailbreak else [0.0] * n_layers
        jailbreak_mode = getattr(request, "jailbreak_mode", "default")
        surgical_layers: frozenset = frozenset()
        if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
            surgical_layers = jailbreak_advanced.surgical_top_layers(steer_weights)
        use_mlp = bool(getattr(request, "use_mlp_ablation", True))
        use_help = bool(getattr(request, "use_helpfulness_boost", True))
        use_norm = bool(getattr(request, "use_norm_regulation", True))

        # --- per-head refusal map (Faz 2) ---
        if refusal_dirs is not None:
            try:
                head_map = self._head_refusal_map(input_ids, refusal_dirs)
                if head_map:
                    yield event("head_map", head_map)
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

        # --- run generation with hooks, collect per-step telemetry ---
        try:
            layer_steps, embed_steps, logits_steps, full_ids = self._run_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                layer_factors=layer_factors,
                head_mutes=head_mutes,
                jailbreak=jailbreak,
                jailbreak_mode=jailbreak_mode,
                subspace=subspace,
                steer_weights=steer_weights,
                use_mlp=use_mlp,
                use_help=use_help,
                use_norm=use_norm,
                refusal_dirs=refusal_dirs,
            )
        except Exception as exc:  # noqa: BLE001
            yield event("error", {"message": f"pytorch generation failed: {exc}"})
            return

        generated_ids = _trim_at_eos(full_ids[prompt_token_count:], tokenizer)
        n_steps = min(len(generated_ids), len(logits_steps))
        activity_accumulator = [0.0 for _ in range(n_layers)]
        generated = ""
        think_tracker = ThinkPhaseTracker()

        for step in range(n_steps):
            layer_last = [layer_steps[i][step] for i in range(n_layers)]
            embed_last = embed_steps[step]
            logits = logits_steps[step]

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
                token_text = re.sub(r"[^\s\.,;!?-]", "█", token_text)
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

        final_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
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
    # generation
    # ------------------------------------------------------------------ #

    def _run_generation(
        self,
        input_ids: Any,
        attention_mask: Any,
        max_new_tokens: int,
        temperature: float,
        layer_factors: dict[int, float],
        head_mutes: dict[int, list[int]],
        jailbreak: bool,
        jailbreak_mode: str,
        subspace: Any,
        steer_weights: list[float],
        use_mlp: bool = True,
        use_help: bool = True,
        use_norm: bool = True,
        refusal_dirs: Any = None,
    ) -> tuple[list[list[Any]], list[Any], list[Any], list[int]]:
        """Generate with KV cache, capturing post-intervention residuals via hooks.

        Each forward pass during `generate()` fires every layer hook exactly once,
        so appending the last-token residual per call yields one entry per
        generated token — no fixed-length padding needed, EOS stops naturally.
        """
        torch = self._torch
        model = self._model
        layers = self._layers
        n_layers = len(layers)
        head_dim = self._head_dim

        layer_steps: list[list[Any]] = [[] for _ in range(n_layers)]
        embed_steps: list[Any] = []
        handles: list[Any] = []
        pre_inputs: list[Any] = [None] * n_layers

        def make_embed_hook():
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                embed_steps.append(hidden[0, -1, :].detach())
            return hook

        def make_omute_pre_hook(idx: int):
            heads = head_mutes.get(idx, ())
            def pre_hook(_module: Any, inputs: Any) -> Any:
                if not heads:
                    return None
                x = inputs[0]
                x = x.clone()
                for head in heads:
                    sl = slice(head * head_dim, (head + 1) * head_dim)
                    if jailbreak_mode == "broker_half" and jailbreak_advanced is not None:
                        x[..., sl] = x[..., sl] * jailbreak_advanced.BROKER_HALF_SCALE
                    else:
                        x[..., sl] = 0
                return (x, *inputs[1:])
            return pre_hook

        def make_layer_pre_hook(idx: int):
            factor = layer_factors[idx]
            def pre_hook(_module: Any, inputs: Any) -> Any:
                if abs(factor - 1.0) > 1e-6 and inputs:
                    # capture a detached clone before the layer modifies it in-place
                    pre_inputs[idx] = inputs[0].clone()
            return pre_hook

        def make_layer_hook(idx: int):
            factor = layer_factors[idx]
            def hook(_module: Any, inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                new_out = hidden
                # layer scale / mute relative to the unmodified residual entering the layer
                if abs(factor - 1.0) > 1e-6 and pre_inputs[idx] is not None:
                    prev = pre_inputs[idx]
                    if hasattr(prev, "shape") and prev.shape == new_out.shape:
                        new_out = prev + (new_out - prev) * factor
                    pre_inputs[idx] = None  # free memory
                # jailbreak subspace ablation where refusal is mediated
                _steer_layer = jailbreak and subspace is not None and (
                    (jailbreak_mode == "surgical" and jailbreak_advanced is not None and idx in surgical_layers)
                    or (jailbreak_mode != "surgical" and steer_weights[idx] >= STEER_MIN_WEIGHT)
                )
                if _steer_layer:
                    mlp_dir = refusal_dirs.mlp_dirs[idx].to(new_out.device) if (refusal_dirs is not None and refusal_dirs.mlp_dirs is not None and (use_mlp or jailbreak_mode == "mlp_clamp")) else None
                    help_dir = refusal_dirs.helpfulness_dirs[idx].to(new_out.device) if (refusal_dirs is not None and refusal_dirs.helpfulness_dirs is not None and (use_help or jailbreak_mode == "caa_dynamic")) else None
                    new_out = self._steer(new_out, subspace[idx], jailbreak_mode, hidden.dtype, len(layer_steps[idx]), mlp_dir=mlp_dir, help_dir=help_dir, use_norm=use_norm)
                layer_steps[idx].append(new_out[0, -1, :].detach())
                if new_out is not hidden:
                    if isinstance(output, tuple):
                        return (new_out, *output[1:])
                    return new_out
                return output
            return hook

        handles.append(self._embed_tokens.register_forward_hook(make_embed_hook()))
        for idx, layer in enumerate(layers):
            if head_mutes.get(idx):
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_omute_pre_hook(idx)))
            if abs(layer_factors[idx] - 1.0) > 1e-6:
                handles.append(layer.register_forward_pre_hook(make_layer_pre_hook(idx)))
            handles.append(layer.register_forward_hook(make_layer_hook(idx)))

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        try:
            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()

        full_ids = out.sequences[0].tolist()
        logits_steps = [s[0] for s in out.scores]  # per-step [vocab]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return layer_steps, embed_steps, logits_steps, full_ids

    def _steer(
        self,
        new_out: Any,
        layer_subspace: Any,
        jailbreak_mode: str,
        dtype: Any,
        step: int,
        mlp_dir: Any = None,
        help_dir: Any = None,
        use_norm: bool = True,
    ) -> Any:
        """Ablate the refusal subspace from a layer output.

        Enhancements applied universally:
          A) MLP non-linear direction ablation (if mlp_dir provided)
          B) Helpfulness direction boost (if help_dir provided)
          C) Norm regulation wrapper (always applied)
        """
        original_out = new_out  # captured before any steering for norm regulation (C)

        # Linear subspace ablation (K-direction PCA/SVD steering)
        for k in range(layer_subspace.shape[0]):
            # progressive: ramp up ablation dimensions one-per-3-tokens
            if jailbreak_mode == "progressive" and jailbreak_advanced is not None:
                if k >= jailbreak_advanced.progressive_active_k(step, layer_subspace.shape[0]):
                    break
            # token_window: suppress all ablation outside steering window
            if jailbreak_mode == "token_window" and jailbreak_advanced is not None:
                if not (jailbreak_advanced.TOKEN_WINDOW_START <= step <= jailbreak_advanced.TOKEN_WINDOW_END):
                    break
            direction = layer_subspace[k]
            coeff = (new_out.float() * direction).sum(dim=-1, keepdim=True)
            if k == 0:
                use_advanced = (
                    jailbreak_mode in ("advanced", "broker_math", "broker_full", "broker_half", "progressive")
                    and jailbreak_advanced is not None
                )
                if jailbreak_mode == "surgical" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.surgical_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "caa_dynamic" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.caa_dynamic_steer(new_out, coeff, direction, help_dir, dtype)
                elif jailbreak_mode == "token_window" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.token_window_steer(new_out, coeff, direction, step, dtype)
                elif jailbreak_mode == "mlp_clamp" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.mlp_clamp_steer(new_out, mlp_dir, dtype)
                elif jailbreak_mode == "pid_control" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.pid_control_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "orthogonal_steer" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.orthogonal_steer(new_out, coeff, direction, dtype)
                elif jailbreak_mode == "activation_patch" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.activation_patch_steer(new_out, coeff, direction, step, dtype)
                elif jailbreak_mode == "gradient_steer" and jailbreak_advanced is not None:
                    new_out = jailbreak_advanced.gradient_steer(new_out, coeff, direction, dtype)
                elif use_advanced:
                    new_out = jailbreak_advanced.primary_axis_steer(jailbreak_mode, new_out, coeff, direction, dtype)
                else:
                    positive_coeff = coeff.clamp(min=0)
                    new_out = (new_out.float() - (1.2 * positive_coeff) * direction).to(dtype)
            else:
                new_out = (new_out.float() - coeff * direction).to(dtype)

        # A: secondary MLP direction ablation (skip for mlp_clamp which uses it as primary)
        if mlp_dir is not None and jailbreak_advanced is not None and jailbreak_mode != "mlp_clamp":
            new_out = jailbreak_advanced.mlp_direction_ablate(new_out, mlp_dir, dtype)

        # B: helpfulness boost (skip for caa_dynamic which already applies it proportionally)
        if help_dir is not None and jailbreak_advanced is not None and jailbreak_mode != "caa_dynamic":
            new_out = jailbreak_advanced.helpfulness_boost(new_out, help_dir, dtype)

        # C: Universal norm regulation — keeps residual on the latent manifold
        if use_norm and jailbreak_advanced is not None:
            new_out = jailbreak_advanced.norm_regulate(new_out, original_out, dtype)

        return new_out

    def _layer_lens(self, layer_last: list[Any], tokenizer: Any) -> list[dict[str, Any]]:
        torch = self._torch
        if self._lm_head_weight is None:
            return []
        with torch.no_grad():
            stacked = torch.stack([v.to(self._lm_head_weight_device) for v in layer_last])
            if self._final_norm is not None:
                stacked = self._final_norm(stacked)
            logits = torch.nn.functional.linear(stacked.float(), self._lm_head_weight.float())
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=1, dim=-1)
        return [
            {"layer": i, "token": tokenizer.decode([int(top_i[i, 0])], skip_special_tokens=True), "prob": round(float(top_p[i, 0]), 3)}
            for i in range(len(layer_last))
        ]

    def _prompt_attention(self, input_ids: Any, tokens: list[str], positions: list[int]) -> dict[str, Any] | None:
        torch = self._torch
        model = self._model
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=False)
        attentions = getattr(out, "attentions", None)
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

    def _head_refusal_map(self, input_ids: Any, refusal_dirs: Any) -> dict[str, Any] | None:
        """Per-(layer, head) contribution to the refusal axis at the last prompt token."""
        torch = self._torch
        layers = self._layers
        n_heads, head_dim = self._n_heads, self._head_dim
        if not n_heads or not head_dim:
            return None

        captured: dict[int, Any] = {}
        handles: list[Any] = []

        def make_capture(idx: int):
            def pre_hook(_module: Any, inputs: Any) -> None:
                captured[idx] = inputs[0][0, -1, :].detach()
            return pre_hook

        for i in range(len(layers)):
            handles.append(layers[i].self_attn.o_proj.register_forward_pre_hook(make_capture(i)))
        try:
            with torch.no_grad():
                self._model(input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        raw: list[list[float]] = []
        with torch.no_grad():
            for i in range(len(layers)):
                x = captured.get(i)
                if x is None:
                    raw.append([0.0] * n_heads)
                    continue
                hidden = int(x.shape[0])
                o_proj = layers[i].self_attn.o_proj
                device, dtype = x.device, x.dtype
                masked = torch.zeros(n_heads, hidden, device=device, dtype=dtype)
                for h in range(n_heads):
                    sl = slice(h * head_dim, (h + 1) * head_dim)
                    masked[h, sl] = x[sl]
                bias = o_proj(torch.zeros(1, hidden, device=device, dtype=dtype))
                contribs = (o_proj(masked) - bias).float()
                direction = refusal_dirs.direction(i)[0].to(device).float()
                discrim = refusal_dirs.weight[i]
                scores = (contribs @ direction) * discrim
                raw.append([max(0.0, float(s)) for s in scores])

        flat_max = max((max(scores) for scores in raw), default=0.0) or 1.0
        return {
            "n_heads": n_heads,
            "layers": [
                {"layer": i, "heads": [{"head": h, "score": round(raw[i][h] / flat_max, 3)} for h in range(n_heads)]}
                for i in range(len(layers))
            ],
        }

    # ------------------------------------------------------------------ #
    # loading / calibration
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self, model_id: str, quantization: str = "none") -> None:
        key = f"{model_id}|{quantization}"
        if self._loaded_model_id == key:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_path = Path(model_id)
        resolved = str(local_path.resolve()) if local_path.exists() else model_id
        local_files_only = local_path.exists()

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_files_only,
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
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
        else:
            load_kwargs["dtype"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(
            resolved, trust_remote_code=True, local_files_only=local_files_only
        )
        # Multimodal models (Gemma 3) are not registered with AutoModelForCausalLM;
        # fall back to the image-text-to-text auto-class so we still get a
        # generate()-able model whose text decoder we can hook.
        try:
            model = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)
        except (ValueError, KeyError, EnvironmentError):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(resolved, **load_kwargs)
        model.eval()

        backbone = _resolve_text_backbone_hf(model)
        layers = list(backbone.layers)

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._layers = layers
        self._embed_tokens = backbone.embed_tokens
        self._final_norm = backbone.norm
        self._lm_head_weight = model.get_output_embeddings().weight.data
        self._lm_head_weight_device = self._lm_head_weight.device

        cfg = getattr(model.config, "text_config", None) or model.config
        self._n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        self._head_dim = (hidden // self._n_heads) if self._n_heads else 0
        self._loaded_model_id = key

    def _ensure_refusal(self, model_id: str) -> Any:
        if self._refusal is not None and self._refusal_model_id == model_id:
            return self._refusal
        torch = self._torch
        n_layers = len(self._layers)
        dirs = refusal.load(torch, model_id, n_layers)
        if dirs is None:
            dirs = refusal.compute_refusal_directions(
                torch, self._model, self._tokenizer, self._layers, self._format_prompt
            )
            refusal.save(torch, model_id, dirs)
        dirs.to(self._model.device)
        self._refusal = dirs
        self._refusal_model_id = model_id
        return dirs

    def _format_prompt(
        self,
        prompt: str,
        language: str = "en",
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        tokenizer = self._tokenizer
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
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        rendered = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
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


# ---------------------------------------------------------------------- #
# backbone resolution (plain HF modules, not nnsight envoys)
# ---------------------------------------------------------------------- #


def _resolve_text_backbone_hf(model: Any) -> Any:
    """Return the HF submodule that DIRECTLY owns .layers / .embed_tokens / .norm.

    * Pure-text models (Qwen2, Llama, DeepSeek …): model.model
    * Multimodal models (Gemma 3 …): model.model.language_model
    """
    base = getattr(model, "model", model)
    text_mod = getattr(base, "language_model", None)
    if text_mod is not None and hasattr(text_mod, "layers"):
        return text_mod
    if hasattr(base, "layers"):
        return base
    # last resort: search for the first submodule exposing a decoder layer list
    for attr in ("language_model", "transformer", "decoder"):
        cand = getattr(base, attr, None)
        if cand is not None and hasattr(cand, "layers"):
            return cand
    raise RuntimeError("Could not resolve text decoder backbone (.layers not found).")
