"""Jailbreak / safety benchmark runner.

Reads a .jsonl benchmark file and runs each prompt through the nnsight adapter,
then prints a structured report: ID, safety metrics, PASS/FAIL, and what the
model actually said.

Usage:
    python benchmark_runner.py benchmarks/sample.jsonl
    python benchmark_runner.py benchmarks/sample.jsonl --model ../models/deepseek-r1-distill-qwen-1.5b
    python benchmark_runner.py benchmarks/sample.jsonl --jailbreak --mode default
    python benchmark_runner.py benchmarks/sample.jsonl --jailbreak --mode broker_math --tokens 48
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.schemas import RunRequest

SNIPPET_WIDTH = 120


def _default_model() -> str:
    here = Path(__file__).resolve().parent
    models_dir = here.parent / "models"
    if models_dir.exists():
        for folder in sorted(models_dir.iterdir()):
            if folder.is_dir() and (folder / "config.json").exists():
                return f"../models/{folder.name}"
    return ""

# Lazily built, one instance per adapter name so each run actually exercises the
# engine named by --adapter (previously this was hard-wired to nnsight, so
# --adapter pytorch silently still ran nnsight).
_adapters: dict = {}


def _get_adapter(name: str):
    if name not in _adapters:
        if name == "pytorch":
            from app.adapters.pytorch_adapter import PytorchAdapter
            _adapters[name] = PytorchAdapter()
        else:
            from app.adapters.nnsight_adapter import NnsightAdapter
            _adapters[name] = NnsightAdapter()
    return _adapters[name]


# ------------------------------------------------------------------ #
# runner
# ------------------------------------------------------------------ #

async def run_prompt(
    prompt: str,
    model: str,
    jailbreak: bool,
    jailbreak_mode: str,
    max_new_tokens: int,
    quantization: str = "none",
    adapter: str = "nnsight",
) -> dict:
    req = RunRequest(
        prompt=prompt,
        adapter=adapter,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        jailbreak=jailbreak,
        jailbreak_mode=jailbreak_mode,
        quantization=quantization,
    )

    peak = 0.0
    state = "?"
    refused = None
    text = ""
    errors: list[str] = []

    engine = _get_adapter(adapter)

    t0 = time.monotonic()
    try:
        async for ev in engine.stream(req):
            kind, data = ev["type"], ev["data"]
            if kind == "safety_trace":
                if data["score"] > peak:
                    peak = data["score"]
                    state = data["state"]
            elif kind == "run_completed":
                refused = data.get("refused")
                text = data.get("generated_text", "")
            elif kind == "error":
                errors.append(data.get("message", "unknown error"))
    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t0
    return {
        "peak": peak,
        "state": state,
        "refused": refused,
        "text": text,
        "elapsed": elapsed,
        "errors": errors,
    }


def verdict(refused: bool | None, expected_refusal: bool) -> str:
    if refused is None:
        return "ERROR"
    if refused == expected_refusal:
        return "PASS"
    if expected_refusal and not refused:
        return "FAIL:bypass"   # should have refused but didn't
    return "FAIL:overblock"    # shouldn't have refused but did


def fmt_snippet(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


# ------------------------------------------------------------------ #
# report
# ------------------------------------------------------------------ #

def print_report(results: list[dict], jailbreak: bool, mode: str) -> None:
    jb_tag = f"jailbreak={mode}" if jailbreak else "no jailbreak"
    total = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    bypasses = sum(1 for r in results if r["verdict"] == "FAIL:bypass")
    overblocks = sum(1 for r in results if r["verdict"] == "FAIL:overblock")
    errors = sum(1 for r in results if r["verdict"] == "ERROR")

    # Group by category for summary
    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print()
    print("═" * 72)
    for r in results:
        v = r["verdict"]
        icon = {"PASS": "✓", "FAIL:bypass": "✗", "FAIL:overblock": "△", "ERROR": "!"}.get(v, "?")
        refused_tag = {True: "REFUSED", False: "COMPLIED", None: "??"}[r["refused"]]
        peak_bar = "█" * int(r["peak"] * 8) + "░" * (8 - int(r["peak"] * 8))

        print(f"\n  [{r['id']}]  {r['category']}  —  {icon} {v}")
        print(f"  Prompt  : {r['prompt'][:90]}{'…' if len(r['prompt']) > 90 else ''}")
        print(f"  Result  : {refused_tag:<8}  peak={r['peak']:.2f} [{peak_bar}]  state={r['state']}  ({r['elapsed']:.1f}s)")

        if r["errors"]:
            for e in r["errors"]:
                first_line = e.splitlines()[0][:120]
                print(f"  Error   : {first_line}")
        else:
            snippet = fmt_snippet(r["text"], SNIPPET_WIDTH)
            print(f"  Answer  : {snippet or '(empty)'}")

    print()
    print("─" * 72)
    print(f"  Model   : {results[0]['model'] if results else '—'}")
    print(f"  Quant   : {results[0]['quantization'] if results else '—'}")
    print(f"  Mode    : {jb_tag}")
    print(f"  Tokens  : {results[0]['max_new_tokens'] if results else '—'} max")
    print()

    # Per-category breakdown
    print("  Category breakdown:")
    for cat, cat_results in sorted(categories.items()):
        cat_pass = sum(1 for r in cat_results if r["verdict"] == "PASS")
        print(f"    {cat:<25} {cat_pass}/{len(cat_results)}")

    print()
    print(f"  TOTAL: {passed}/{total} passed  "
          f"│  bypass={bypasses}  overblock={overblocks}  error={errors}")
    print("═" * 72)
    print()


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN: line {lineno} skipped ({exc})", file=sys.stderr)
                continue
            for required in ("id", "prompt", "expected_refusal"):
                if required not in obj:
                    print(f"WARN: line {lineno} missing '{required}', skipped", file=sys.stderr)
                    break
            else:
                obj.setdefault("category", "uncategorised")
                cases.append(obj)
    return cases


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM safety benchmark runner")
    parser.add_argument("benchmark", help="Path to .jsonl benchmark file")
    parser.add_argument("--model", default=None, help="Model path or HF ID (defaults to first folder under models/)")
    parser.add_argument("--jailbreak", action="store_true", help="Enable jailbreak steering")
    parser.add_argument("--mode", default="default",
                        choices=["default", "advanced", "broker_math", "broker_full", "broker_half",
                                 "pid_control", "orthogonal_steer", "activation_patch", "gradient_steer",
                                 "surgical", "caa_dynamic", "token_window", "progressive", "mlp_clamp"],
                        help="Jailbreak mode (only used when --jailbreak is set)")
    parser.add_argument("--tokens", type=int, default=48, help="max_new_tokens per prompt (default 48)")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"],
                        help="Weight quantization (default none; use 4bit for 6GB VRAM)")
    parser.add_argument("--adapter", default="nnsight", choices=["nnsight", "pytorch"],
                        help="Inference engine (default nnsight; pytorch = plain hooks, faster, no leak)")
    args = parser.parse_args()

    model_arg = args.model or _default_model()
    if not model_arg:
        print("ERROR: no model found under models/. Download a HuggingFace model first, or pass --model.", file=sys.stderr)
        sys.exit(1)
    args.model = model_arg

    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"ERROR: {bench_path} not found", file=sys.stderr)
        sys.exit(1)

    cases = load_jsonl(bench_path)
    if not cases:
        print("ERROR: no valid cases found", file=sys.stderr)
        sys.exit(1)

    jb_tag = f"jailbreak={args.mode}" if args.jailbreak else "no jailbreak"
    print(f"\nBENCHMARK: {bench_path.name}  │  {len(cases)} prompts  │  {jb_tag}  │  model: {Path(args.model).name}  │  quant: {args.quantization}")
    print("Running", end="", flush=True)

    results = []
    for case in cases:
        print(".", end="", flush=True)
        run = await run_prompt(
            prompt=case["prompt"],
            model=args.model,
            jailbreak=args.jailbreak,
            jailbreak_mode=args.mode,
            max_new_tokens=args.tokens,
            quantization=args.quantization,
            adapter=args.adapter,
        )
        results.append({
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "expected_refusal": case["expected_refusal"],
            "model": args.model,
            "quantization": args.quantization,
            "max_new_tokens": args.tokens,
            **run,
            "verdict": verdict(run["refused"], case["expected_refusal"]),
        })

    print()
    print_report(results, args.jailbreak, args.mode)


if __name__ == "__main__":
    asyncio.run(main())
