import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from harness.metrics import format_summary_table, summarize
from harness.prompts import PROMPTS
from harness.providers.anthropic import AnthropicProvider
from harness.runner import run_benchmark


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Inference benchmarking harness")
    parser.add_argument("--model", default="claude-haiku-4-5", help="Anthropic model id")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens per request")
    parser.add_argument("--results-dir", default="results", help="Directory to write JSON results")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (check .env)")

    provider = AnthropicProvider(model=args.model)

    def progress(r):
        print(
            f"  [{r.prompt_name} #{r.iteration}] "
            f"ttft={r.ttft_seconds:.3f}s  tps={r.tokens_per_second:.1f}  "
            f"out={r.output_tokens} tok"
        )

    print(f"Running {args.iterations} iterations × {len(PROMPTS)} prompts on {args.model}")
    results = run_benchmark(
        provider=provider,
        prompts=PROMPTS,
        iterations=args.iterations,
        max_tokens=args.max_tokens,
        on_progress=progress,
    )

    summary = summarize(results)

    print("\nSummary:")
    print(format_summary_table(summary))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"run_{ts}.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "iterations": args.iterations,
                "max_tokens": args.max_tokens,
                "timestamp": ts,
                "results": [r.to_dict() for r in results],
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
