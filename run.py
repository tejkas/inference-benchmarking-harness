import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.load_runner import sweep_concurrency
from harness.plot import plot_ttft_vs_throughput
from harness.prompts import PROMPTS
from harness.providers.llama_cpp import LlamaCppProvider


def main():
    parser = argparse.ArgumentParser(description="TTFT vs throughput sweep")
    parser.add_argument("--model", required=True, help="Model name (label for llama.cpp)")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt", default="medium", choices=[p.name for p in PROMPTS])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    prompt = next(p for p in PROMPTS if p.name == args.prompt)
    provider = LlamaCppProvider(model=args.model, base_url=args.base_url)

    def progress(r):
        print(f"  [n={r['concurrency']}] ttft={r['mean_ttft']:.3f}s  tps={r['throughput_tps']:.1f}")

    print(f"Sweeping concurrency {args.levels} on {args.model}")
    results = sweep_concurrency(provider, prompt, args.levels, args.max_tokens, progress)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = results_dir / f"sweep_{ts}.json"
    with json_path.open("w") as f:
        json.dump({"model": args.model, "prompt": args.prompt, "results": results}, f, indent=2)
    print(f"Wrote {json_path}")

    plot_path = str(results_dir / f"sweep_{ts}.png")
    plot_ttft_vs_throughput(results, args.model, plot_path)


if __name__ == "__main__":
    main()