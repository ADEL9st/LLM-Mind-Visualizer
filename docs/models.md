# Model Notes

## First White-box Model

Recommended starting point:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Download:

```powershell
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5b-instruct
```

Use either the Hugging Face id or the local path in the dashboard model field.

For this repo layout, the backend runs from `backend/`, so the local path is:

```text
../models/qwen2.5-1.5b-instruct
```

## Modes

### Mock Trace

No model required. Use this to verify the UI, event stream, and intervention flow.

### Ollama Audit

Use model names like:

```text
qwen2.5:1.5b
```

This is black-box. It can stream output and timing metrics, but not internal activations.

### Transformers Hook

Use a `safetensors` / PyTorch model through Hugging Face Transformers. This mode can collect layer activity and apply basic layer scaling interventions.

Install optional dependencies:

```powershell
cd backend
.\.venv\Scripts\python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-ml.txt
```

The first hook implementation is intentionally generic. After Qwen is downloaded, the next step is to calibrate:

- actual layer count
- attention head count
- output tensor shape
- hook points for attention heads
- safety/refusal probe scoring
