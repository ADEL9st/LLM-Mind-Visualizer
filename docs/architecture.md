# Architecture Notes

The project is built around one stable event stream. Every backend adapter sends the same event shapes to the dashboard.

## Backend

```text
FastAPI
  /ws/run
    RunRequest
      adapter: mock | ollama | transformers
      intervention config
      output policy
    StreamEvent[]
```

Adapters:

- `mock`: deterministic simulated telemetry for UI development.
- `ollama`: black-box GGUF audit through Ollama's local API.
- `transformers`: optional PyTorch/Hugging Face hook mode for `safetensors` models.

## Event Types

- `run_started`: model metadata and prompt token count.
- `attention`: prompt token weights.
- `layer_activity`: per-layer activity, safety, and uncertainty values.
- `safety_trace`: refusal/safety signal summary.
- `uncertainty`: entropy, hallucination risk, and top-k candidates.
- `concept_trace`: human-readable concept scores.
- `token`: streamed text.
- `black_box_metrics`: Ollama timing/token stats.
- `intervention`: selected intervention status.
- `run_completed`: final text and finish reason.

## Intervention Model

The first supported intervention engine is layer-level scaling with multiple rules:

- `mute`: multiply selected layer output by `0`.
- `scale`: multiply selected layer output by `scale`.
- `boost`: multiply selected layer output by `1 + abs(scale)`.

The frontend can add layer sets like `8,12,16-20`; each selected layer becomes an intervention rule. Rules are applied in order during the forward hook.

The current generic hook adapter targets decoder layers exposed as:

- `model.layers`
- `model.model.layers`
- `transformer.h`
- `gpt_neox.layers`

Qwen/Llama-style models usually expose `model.layers` or `model.model.layers`.

## Safety Positioning

Internal development can inspect raw output. Public/demo builds should use `restricted` or `redacted` output policy and still show:

- affected layers
- safety/refusal score deltas
- entropy changes
- intervention target and result
- unsafe span type

This keeps the useful mechanism visible without making the UI a harmful output browser.
