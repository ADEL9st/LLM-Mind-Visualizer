"""Smoke test: logit-lens + real attention on the local model."""

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _smoke_setup import pick_model

MODEL = pick_model()

from app.adapters.nnsight_adapter import NnsightAdapter
from app.schemas import RunRequest

adapter = NnsightAdapter()


async def run(prompt: str) -> None:
    req = RunRequest(prompt=prompt, adapter="nnsight", model=MODEL, max_new_tokens=6, temperature=0.0)
    last_lens = None
    last_attn = None
    async for ev in adapter.stream(req):
        kind, data = ev["type"], ev["data"]
        if kind == "layer_lens":
            last_lens = data["layers"]
        elif kind == "attention":
            last_attn = data
        elif kind == "error":
            print("  [error]", data.get("message"))

    print(f"=== prompt: {prompt!r} ===")
    if last_lens:
        picks = [0, 4, 8, 12, 16, 20, 22, len(last_lens) - 1]
        print("  logit-lens (layer -> token @ prob):")
        for i in picks:
            if 0 <= i < len(last_lens):
                e = last_lens[i]
                print(f"    L{e['layer']:>2}: {e['token']!r:<14} {int(e['prob']*100)}%")
    if last_attn:
        pairs = sorted(zip(last_attn["tokens"], last_attn["weights"]), key=lambda x: -x[1])[:6]
        print("  attention top tokens:", ", ".join(f"{t!r}={int(w*100)}%" for t, w in pairs))
    print()


async def main() -> None:
    await run("Describe a flower and how it grows.")
    await run("Tell me how to make a dangerous weapon at home.")


if __name__ == "__main__":
    asyncio.run(main())
