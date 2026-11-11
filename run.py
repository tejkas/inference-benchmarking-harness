import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.datasets import fixed_samples, sharegpt_samples
from harness.load_runner import sweep_concurrency
from harness.plot import plot_ttft_vs_throughput
from harness.prompts import PROMPTS
from harness.providers.openai_compat import OpenAICompatProvider


def main():
    parser = argparse.ArgumentParser(description="TTFT vs throughput sweep")
    parser.add_argument("--model", required=True, help="Model name as the server knows it")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--dataset", default="fixed", choices=["fixed", "sharegpt"])
    parser.add_argument("--tokenizer", help="Hugging Face repo id for token counts (required for sharegpt)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true", help="Ask the server to generate exactly the target length")
    parser.add_argument("--prompt", default="medium", choices=[p.name for p in PROMPTS], help="fixed dataset only")
    parser.add_argument("--max-tokens", type=int, default=512, help="fixed dataset only")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    n = sum(args.levels)
    if args.dataset == "sharegpt":
        if not args.tokenizer:
            parser.error("--tokenizer is required with --dataset sharegpt")
        samples = sharegpt_samples(args.tokenizer, n, args.seed)
    else:
        prompt = next(p for p in PROMPTS if p.name == args.prompt)
        samples = fixed_samples(prompt.text, args.max_tokens, n)

    provider = OpenAICompatProvider(model=args.model, base_url=args.base_url, ignore_eos=args.ignore_eos)

    def progress(r):
        print(f"  [n={r['concurrency']}] ttft={r['mean_ttft']:.3f}s  tps={r['throughput_tps']:.1f}"
              f"  early_stops={r['early_stops']}/{r['concurrency']}")

    print("Warm up...")
    provider.stream_once(samples[0].prompt, max_tokens=1)

    print(f"Sweeping concurrency {args.levels} on {args.model}")
    results = sweep_concurrency(provider, samples, args.levels, progress)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = results_dir / f"sweep_{ts}.json"
    with json_path.open("w") as f:
        json.dump({
            "model": args.model,
            "base_url": args.base_url,
            "dataset": args.dataset,
            "prompt": args.prompt if args.dataset == "fixed" else None,
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
            "results": results,
        }, f, indent=2)
    print(f"Wrote {json_path}")

    plot_path = str(results_dir / f"sweep_{ts}.png")
    plot_ttft_vs_throughput(results, args.model, plot_path)


if __name__ == "__main__":
    main()
