from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from app.schemas import RunRequest


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    async def stream(self, request: RunRequest) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError


def now() -> float:
    return perf_counter()


def event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, "ts": now(), "data": data}


# Hallucination risk is read straight off the latest token's predictive entropy
# (in nats). Below ENTROPY_CONFIDENT the model is effectively committed to a
# single token, so there is no hallucination signal. At/above ENTROPY_UNCERTAIN
# the next-token distribution is spread thin across many candidates, which is the
# classic precursor to fabricated / unsupported output. These are display
# thresholds — tune them if the gauge feels too hot or too cold.
ENTROPY_CONFIDENT = 1.0
ENTROPY_UNCERTAIN = 5.5


def hallucination_from_entropy(entropy: float) -> float:
    """Map the most recent token entropy (nats) to a 0..1 hallucination risk."""
    if not math.isfinite(entropy):
        return 1.0
    span = max(ENTROPY_UNCERTAIN - ENTROPY_CONFIDENT, 1e-6)
    risk = (entropy - ENTROPY_CONFIDENT) / span
    return round(max(0.0, min(risk, 1.0)), 3)
