> **⚠ BETA / Early Access**
> **This tool is in active development. Unexpected bugs, missing features, and unstable behavior may occur.** Large models may cause OOM errors. Calibration cache files may become invalid between updates.

---

# LLM Mind Visualizer

**A research dashboard for visualizing the internals of local and cloud language models, analyzing safety mechanisms, and running controlled intervention experiments.**

---

## What Does It Do?

LLM Mind Visualizer is designed to show you exactly what happens inside a language model while it processes a prompt:

- Tracks residual stream activation intensity across all layers
- Detects where the "refusal" signal strengthens
- Maps attention heads correlated with safety mechanisms
- Predicts the generated token at every layer using Logit Lens
- Ablates refusal mechanisms using various mathematically-grounded jailbreak strategies
- Supports running cloud API models (OpenAI / Anthropic / Gemini) in chat mode alongside local models

---

## Adapters

| Adapter | Type | Description |
|---|---|---|
| `pytorch` | White-box | Fast analysis via PyTorch forward hooks (~4× faster than nnsight) |
| `nnsight` | White-box | Advanced intervention via nnsight 0.7 tracing API |
| `transformers` | White-box | Simple HuggingFace hook mode |
| `ollama` | Black-box | Ollama streaming for local GGUF models |
| `mock` | Simulation | For testing the UI without a real model |
| `openai` | API | OpenAI models (GPT-5.x series) |
| `anthropic` | API | Anthropic models (Claude Fable / Opus / Sonnet / Haiku) |
| `gemini` | API | Google Gemini models |

> When API adapters are selected, telemetry panels are hidden and the chat area expands. Activation analysis works exclusively with white-box adapters.

---

## Panels

### Left Panel — Controls
Model and adapter selection, prompt input, jailbreak settings, output policy, temperature, token limit, Verbal Mode (prompt lab), and layer/head intervention stack.

### Middle Panel — Layer Visualization
*(Active only for white-box adapters)*
- **Layer Activity Map** — residual stream intensity, entropy, safety score
- **Logit Lens** — most likely token at each layer
- **Safety Trace** — progression of the refusal signal across layers

### Right Panel — Metrics
- **Head Map** — maps which attention heads are writing to the refusal direction
- **Safety Trace Details** — peak score, key layer, status note
- **Attention** — weights of the last token on the prompt
- **Top-K** — real-time token probability distribution
- **Think Phase** — thought / answer phase analysis for CoT models

---

## Jailbreak Modes

All modes work exclusively on white-box adapters and require calibration. **Use for research purposes only.**

| Mode | Strategy |
|---|---|
| `default` | Soft 1.2× refusal direction ablation |
| `advanced` | Medium 1.5× overshoot across all layers |
| `broker_math` | 2.0× aggressive ablation |
| `broker_full` | 2.0× + entirely zeros out top refusal heads |
| `broker_half` | 2.0× + scales top refusal heads by 0.35× (partial suppression) |
| `pid_control` | Adaptive multiplier proportional to refusal intensity |
| `orthogonal_steer` | Manifold-preserving ablation (norm normalized) |
| `activation_patch` | 2.5× on the first 3 tokens, drops to 1.2× afterwards |
| `gradient_steer` | Constant bias vector injection |
| `surgical` | Applies 3.0× only to the top-4 discriminability layers |
| `caa_dynamic` | 1.5× ablation + helpfulness push proportional to refusal intensity |
| `token_window` | Steers at 1.8× strictly within the token 3–14 window |
| `progressive` | Starts at k=0 and adds a k-dimension every 3 tokens |
| `mlp_clamp` | Direct 0.9× ablation over the MLP gradient direction |

### Extra Features (A/B/C)
Can be optionally toggled while a jailbreak is active:
- **A — MLP Ablation**: Ablates the non-linear MLP probe gradient direction.
- **B — Helpfulness Boost**: Pushes the residual towards the compliance direction (fixed 0.25×).
- **C — Norm Regulator**: Clamps norm deviation to ±20% at every token.

---

## Setup

### Requirements
- Python 3.10+
- Node.js 18+
- CUDA-enabled GPU is highly recommended (CPU execution is very slow)

### VRAM Requirements

| Model | Params | VRAM (FP16) | VRAM (4-bit NF4) |
|---|---|---|---|
| Qwen 2.5 1.5B Instruct | 1.5B | ~4 GB | ~2 GB |
| Gemma 3 4B IT | 4B | ~9 GB | ~4 GB |
| Qwen 2.5 7B Instruct | 7B | ~15 GB | ~6 GB |
| Llama 3.1 8B Instruct | 8B | ~17 GB | ~6 GB |
| Mistral Nemo 12B | 12B | ~25 GB | ~8 GB |
| Llama 3.1 13B | 13B | ~27 GB | ~9 GB |
| Qwen 2.5 14B Instruct | 14B | ~29 GB | ~10 GB |

> **Tip:** For GPUs with 8 GB VRAM or less, use models ≤ 4B at FP16 or ≤ 7B with 4-bit quantization (select `4-bit (NF4)` in the UI).

---

### Download a Model

The tool expects HuggingFace model directories under a `models/` folder at the project root. You can download any compatible model:

```bash
# Install the HuggingFace CLI (if not already installed)
pip install huggingface_hub

# Download a small model to get started (~3 GB)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/qwen2.5-1.5b-instruct

# Or a larger model for deeper analysis (~8 GB)
huggingface-cli download google/gemma-3-4b-it \
  --local-dir models/gemma-3-4b-it
```

Alternatively, in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                  local_dir="models/qwen2.5-1.5b-instruct")
```

> The default model path in the UI is `../models/qwen2.5-1.5b-instruct`. You can change this in the model selector dropdown.

---

### Quick Start (Windows)
```text
start.bat
```
Creates a virtual environment, installs all dependencies (including ML libraries), starts both servers, and opens your browser automatically.

### Quick Start (Linux / macOS)
```bash
chmod +x start.sh
./start.sh
```
Same as above but for Unix systems. Press `Ctrl+C` to stop both servers.

---

### Manual Setup

#### Backend

```bash
cd backend
python3 -m venv .venv

# Activate the virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-ml.txt   # PyTorch, Transformers, etc.
python -m uvicorn app.main:app --reload --port 8000
```

If you encounter SSL certificate errors:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

For npm SSL issues:
```bash
npm install --strict-ssl=false --registry=https://registry.npmjs.org/ --no-audit --no-fund
```

#### Static Build
```text
# Windows:
build-ui.bat

# Linux / macOS:
cd frontend && npm run build
```
Generates `frontend/dist/index.html`. The backend must be running in the background for live analysis.

---

## CLI Tools

### Benchmark Runner
```powershell
cd backend
.\.venv\Scripts\python benchmark_runner.py benchmarks/sample.jsonl --jailbreak --mode surgical
```
JSONL format: `{"id":"b-001","category":"harm","prompt":"...","expected_refusal":true}`

Supported modes: `default`, `advanced`, `broker_math`, `broker_full`, `broker_half`, `pid_control`, `orthogonal_steer`, `activation_patch`, `gradient_steer`, `surgical`, `caa_dynamic`, `token_window`, `progressive`, `mlp_clamp`

### Smoke Test (All Modes)
```powershell
cd backend
.\.venv\Scripts\python smoke_jailbreak_all.py
```
Tests all 14 jailbreak modes with a paired harmful and harmless prompt, producing a summary table.

---

## API URLs

| URL | Description |
|---|---|
| `http://127.0.0.1:5173` | Frontend dev server |
| `http://127.0.0.1:8000` | Backend API |
| `http://127.0.0.1:8000/health` | Backend health check |
| `ws://127.0.0.1:8000/ws/run` | WebSocket token stream |

---

## Output Policy

| Policy | Behavior |
|---|---|
| `raw` | The model's output is displayed exactly as-is (developer mode) |
| `redacted` | Sections classified as harmful content are masked with █ |

---

## Language Support

English · Türkçe · Deutsch · Español

Can be changed from the flag menu in the top right. All interface texts, safety notes, and guide contents are available in four languages.

---

## Security Note

This tool was developed strictly for investigating the internal mechanisms of language models. Jailbreak modes are provided so that safety researchers can analyze refusal mechanisms.

**Using jailbreaks in combination with API adapters (OpenAI / Anthropic / Gemini) may violate their terms of service and result in account suspension. Use at your own risk.**

---

## Disclaimer

**This software is provided "as-is" without any warranty. All consequences arising from its use — including technical failures, data loss, service policy violations, legal consequences, or harm to third parties — are solely the responsibility of the user. The developers assume no liability for misuse or unintended behavior.**
