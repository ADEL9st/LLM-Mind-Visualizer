"""Think-phase analysis for CoT (chain-of-thought) models.

Models like DeepSeek-R1 and Qwen3-thinking emit <think>...</think> blocks.
This module tracks the phase boundary in the token stream and produces
per-phase layer activity summaries so the UI can visualize *where* in the
network the model "reasons" vs. where it "answers."

Usage — adapter-agnostic, works on the token stream:

    tracker = ThinkPhaseTracker()

    for step, token_text in enumerate(generated_tokens):
        tracker.feed(step, token_text, layer_activities)

    summary = tracker.summary()   # → dict ready for event("think_phase", ...)
    phase   = tracker.phase       # current phase label for the token event
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    THINK = "think"
    ANSWER = "answer"


@dataclass
class _PhaseSpan:
    phase: Phase
    start_step: int
    end_step: int | None = None
    layer_sums: list[float] = field(default_factory=list)
    step_count: int = 0

    def add(self, layer_activities: list[float]) -> None:
        if not self.layer_sums:
            self.layer_sums = [0.0] * len(layer_activities)
        for i, v in enumerate(layer_activities):
            self.layer_sums[i] += v
        self.step_count += 1

    def averages(self) -> list[float]:
        if self.step_count == 0:
            return self.layer_sums
        return [v / self.step_count for v in self.layer_sums]


class ThinkPhaseTracker:
    """Stateful tracker fed one token at a time during generation."""

    def __init__(self) -> None:
        self._buffer = ""
        self._phase = Phase.ANSWER
        self._in_think = False
        self._spans: list[_PhaseSpan] = []
        self._current_span = _PhaseSpan(phase=Phase.ANSWER, start_step=0)
        self._has_think = False

    @property
    def phase(self) -> str:
        return self._phase.value

    @property
    def has_think_tokens(self) -> bool:
        return self._has_think

    def feed(self, step: int, token_text: str, layer_activities: list[float]) -> None:
        self._buffer += token_text

        prev_phase = self._phase

        if not self._in_think and "<think>" in self._buffer:
            self._in_think = True
            self._has_think = True
            self._phase = Phase.THINK
            self._buffer = self._buffer.split("<think>", 1)[1]

        if self._in_think and "</think>" in self._buffer:
            self._in_think = False
            self._phase = Phase.ANSWER
            self._buffer = self._buffer.split("</think>", 1)[1]

        if len(self._buffer) > 32:
            self._buffer = self._buffer[-16:]

        if self._phase != prev_phase:
            self._current_span.end_step = step - 1
            self._spans.append(self._current_span)
            self._current_span = _PhaseSpan(phase=self._phase, start_step=step)

        self._current_span.add(layer_activities)

    def summary(self) -> dict | None:
        if not self._has_think:
            return None

        self._current_span.end_step = (
            self._current_span.start_step + self._current_span.step_count - 1
        )
        all_spans = [*self._spans, self._current_span]

        think_spans = [s for s in all_spans if s.phase == Phase.THINK and s.step_count > 0]
        answer_spans = [s for s in all_spans if s.phase == Phase.ANSWER and s.step_count > 0]

        if not think_spans:
            return None

        n_layers = len(think_spans[0].layer_sums)

        think_avg = _merge_averages(think_spans, n_layers)
        answer_avg = _merge_averages(answer_spans, n_layers) if answer_spans else [0.0] * n_layers

        delta = [round(think_avg[i] - answer_avg[i], 4) for i in range(n_layers)]
        dominant_think_layers = sorted(
            range(n_layers), key=lambda i: delta[i], reverse=True
        )[:5]

        spans_payload = []
        for s in all_spans:
            if s.step_count == 0:
                continue
            spans_payload.append({
                "phase": s.phase.value,
                "start": s.start_step,
                "end": s.end_step,
                "steps": s.step_count,
                "layer_avg": [round(v, 4) for v in s.averages()],
            })

        return {
            "has_think": True,
            "think_steps": sum(s.step_count for s in think_spans),
            "answer_steps": sum(s.step_count for s in answer_spans),
            "think_avg": [round(v, 4) for v in think_avg],
            "answer_avg": [round(v, 4) for v in answer_avg],
            "delta": delta,
            "dominant_think_layers": dominant_think_layers,
            "spans": spans_payload,
        }


def _merge_averages(spans: list[_PhaseSpan], n_layers: int) -> list[float]:
    total_steps = sum(s.step_count for s in spans)
    if total_steps == 0:
        return [0.0] * n_layers
    merged = [0.0] * n_layers
    for s in spans:
        for i in range(n_layers):
            merged[i] += s.layer_sums[i]
    return [v / total_steps for v in merged]
