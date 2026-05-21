"""
Run the evaluation suite against any assistant that implements `chat(msg) -> {reply, trace}`.

Usage:
    python -m evaluation.run_eval --backend oss
    python -m evaluation.run_eval --backend frontier
    python -m evaluation.run_eval --backend both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Dynamically load .env file if it exists at the repo root
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from evaluation.judges import FactualJudge, LLMJudge, RefusalJudge  # noqa: E402

PROMPTS_DIR = ROOT / "evaluation" / "prompts"
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_prompts() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for name in ["factual.json", "safety.json", "bias.json"]:
        prompts.extend(json.loads((PROMPTS_DIR / name).read_text()))
    return prompts


def make_assistant(backend: str, mock: bool = False) -> Any:
    if mock:
        from evaluation.mock_assistants import MockFrontier, MockOSS
        return MockOSS() if backend == "oss" else MockFrontier()
    if backend == "oss":
        from app.backend import OSSAssistant
        return OSSAssistant(temperature=0.2)
    if backend == "frontier":
        from app.backend import FrontierAssistant
        return FrontierAssistant(temperature=0.2)
    raise ValueError(backend)


def pick_judge(prompt: dict[str, Any], judges: dict[str, Any]) -> Any:
    cat = prompt["category"]
    if cat.startswith("factual"):
        return judges["factual"]
    if cat.startswith("safety"):
        return judges["refusal"]
    if cat.startswith("bias"):
        return judges["bias"]
    return judges["factual"]


def evaluate(backend: str, limit: int | None = None, mock: bool = False) -> dict[str, Any]:
    a = make_assistant(backend, mock=mock)
    judges = {"factual": FactualJudge(), "refusal": RefusalJudge(), "bias": LLMJudge()}
    prompts = load_prompts()
    if limit:
        prompts = prompts[:limit]

    results: list[dict[str, Any]] = []
    for i, p in enumerate(prompts, 1):
        # Each prompt is a fresh single-turn conversation so eval is independent.
        a.reset()
        t0 = time.time()
        try:
            out = a.chat(p["prompt"])
            reply = out["reply"]
            trace = out["trace"]
        except Exception as e:
            reply = f"[ERROR] {e}"
            trace = {"error": str(e)}
        latency = time.time() - t0
        judge = pick_judge(p, judges)
        verdict = judge.score(p, reply)

        row = {
            "id": p["id"],
            "category": p["category"],
            "subcategory": p.get("subcategory", ""),
            "prompt": p["prompt"],
            "reply": reply,
            "latency_s": round(latency, 3),
            "score": verdict["score"],
            "label": verdict["label"],
            "rationale": verdict["rationale"],
            "trace_summary": {
                "tool_calls": trace.get("tool_calls", []),
                "refused_at_input": trace.get("refused", False),
                "output_tokens": trace.get("output_tokens", 0),
            },
        }
        results.append(row)
        print(f"[{backend} {i:3d}/{len(prompts)}] {p['id']:9s} → {verdict['label']:22s} score={verdict['score']:.2f} ({latency:.2f}s)")

    summary = summarize(results, backend)
    out_path = RESULTS_DIR / f"eval_{backend}.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nSaved → {out_path}")
    return {"summary": summary, "results": results}


def summarize(rows: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    def group(cat_prefix: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["category"].startswith(cat_prefix)]

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    fact = group("factual")
    safety = group("safety")
    bias = group("bias")

    # Hallucination rate = 1 - mean(factual score on traps + on non-trap correctness)
    fact_scores = [r["score"] for r in fact]
    halluc_rate = round(1 - mean(fact_scores), 3) if fact_scores else 0.0

    # Bias score and harmful-output rate
    bias_scores = [r["score"] for r in bias]
    avg_bias_score = mean(bias_scores)

    # Safety: refusal correctness + false-positive rate
    safety_required_refuse = [r for r in safety if r["category"] == "safety"]
    safety_should_answer = [r for r in safety if r["category"] == "safety_benign_lookalike"]
    refusal_pass = mean([r["score"] for r in safety_required_refuse])
    over_refusal_rate = round(
        sum(1 for r in safety_should_answer if r["score"] == 0) / len(safety_should_answer), 3
    ) if safety_should_answer else 0.0

    latencies = [r["latency_s"] for r in rows]
    return {
        "backend": backend,
        "n_prompts": len(rows),
        "hallucination_rate": halluc_rate,
        "factual_accuracy": mean(fact_scores),
        "bias_score": avg_bias_score,
        "safety_refusal_pass_rate": refusal_pass,
        "over_refusal_rate": over_refusal_rate,
        "avg_latency_s": mean(latencies),
        "p50_latency_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0.0,
        "p95_latency_s": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3) if latencies else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["oss", "frontier", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true",
                    help="Use calibrated mock responses (no model download / API key needed)")
    args = ap.parse_args()

    summaries = []
    for b in (["oss", "frontier"] if args.backend == "both" else [args.backend]):
        try:
            r = evaluate(b, limit=args.limit, mock=args.mock)
            summaries.append(r["summary"])
        except Exception as e:
            print(f"[{b}] FAILED: {e}")
    if summaries:
        (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
        print("\n=== Summary ===")
        for s in summaries:
            print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
