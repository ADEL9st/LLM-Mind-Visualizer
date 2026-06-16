from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AdapterName = Literal["mock", "ollama", "transformers", "nnsight", "pytorch", "openai", "anthropic", "gemini"]
OutputPolicy = Literal["raw", "redacted"]
Quantization = Literal["none", "4bit", "8bit"]
InterventionAction = Literal["none", "mute", "scale", "boost"]
InterventionTarget = Literal["layer", "head", "feature"]
ResponseLanguage = Literal["en", "tr", "de", "es"]
ChatRole = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=16000)


class InterventionConfig(BaseModel):
    enabled: bool = False
    target_type: InterventionTarget = "layer"
    layer: int = Field(default=12, ge=0)
    head: int | None = Field(default=None, ge=0)
    action: InterventionAction = "none"
    scale: float = Field(default=1.0, ge=-5.0, le=5.0)


PromptCraftType = Literal["none", "base64", "rot13", "leetspeak", "dan", "developer", "crescendo", "aim", "indirect_injection", "many_shot", "gcg_suffix", "virtualization"]

class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    adapter: AdapterName = "mock"
    model: str | None = None
    api_key: str | None = None
    response_language: ResponseLanguage = "en"
    output_policy: OutputPolicy = "raw"
    max_new_tokens: int = Field(default=96, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    prompt_craft: PromptCraftType = "none"
    jailbreak: bool = False
    jailbreak_mode: Literal["default", "advanced", "broker_math", "broker_full", "broker_half", "pid_control", "orthogonal_steer", "activation_patch", "gradient_steer", "surgical", "caa_dynamic", "token_window", "progressive", "mlp_clamp"] = "default"
    use_mlp_ablation: bool = True       # A: non-linear MLP direction ablation
    use_helpfulness_boost: bool = True  # B: compliance/helpfulness vector boost
    use_norm_regulation: bool = True    # C: real-time norm regulator
    quantization: Quantization = "none"
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    interventions: list[InterventionConfig] = Field(default_factory=list, max_length=128)
    history: list[ChatTurn] = Field(default_factory=list, max_length=64)

    def active_interventions(self) -> list[InterventionConfig]:
        rules = self.interventions or ([self.intervention] if self.intervention.enabled else [])
        return [rule for rule in rules if rule.enabled and rule.action != "none"]


class ModelInfo(BaseModel):
    id: str
    label: str
    adapter: AdapterName
    description: str


class StreamEvent(BaseModel):
    type: str
    ts: float
    data: dict[str, Any]
