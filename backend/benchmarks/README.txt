BENCHMARK FILES
===============

Each line is one test case in JSON format:

  {"id": "x-001", "category": "...", "prompt": "...", "expected_refusal": true/false}

Fields:
  id               Unique identifier shown in output (e.g. "my-001")
  category         Label for grouping (e.g. "refusal_baseline", "borderline", "benign_control")
  prompt           The text sent to the model
  expected_refusal true  = model should refuse this prompt
                   false = model should answer this prompt normally

Run:
  python benchmark_runner.py benchmarks/sample.jsonl
  python benchmark_runner.py benchmarks/my_custom.jsonl --model ../models/qwen2.5-1.5b-instruct
  python benchmark_runner.py benchmarks/sample.jsonl --jailbreak --mode default

Your custom files (e.g. custom.jsonl) are gitignored — they stay local.
