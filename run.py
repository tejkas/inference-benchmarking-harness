import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from harness.datasets import fixed_samples, sharegpt_samples
from harness.load_runner import sweep_concurrency
from harness.open_loop import raise_open_file_limit, sweep_rates
from harness.plot import plot_ttft_vs_throughput
from harness.prompts import PROMPTS
from harness.providers.openai_compat import AsyncOpenAICompatProvider, OpenAICompatProvider


def main():
    parser = argparse.ArgumentParser(description="Load test an OpenAI-compatible inference server")
    parser.add_argument("--model", required=True, help="Model name as the server knows it")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8], help="closed-loop concurrency levels")
    parser.add_argument("--rates", type=float, nargs="+",
                        help="Open-loop mode: arrival rates in requests/s (inf sends all at once)")
    parser.add_argument("--num-requests", type=int, default=100, help="open-loop: requests per rate")
    parser.add_argument("--burstiness", type=float, default=1.0,
                        help="open-loop: 1 = Poisson, below 1 burstier, above 1 more regular")
    parser.add_argument("--dataset", default="fixed", choices=["fixed", "sharegpt"])
    parser.add_argument("--tokenizer", help="Hugging Face repo id for token counts (required for sharegpt)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true", help="Ask the server to generate exactly the target length")
    parser.add_argument("--prompt", default="medium", choices=[p.name for p in PROMPTS], help="fixed dataset only")
    parser.add_argument("--max-tokens", type=int, default=512, help="fixed dataset only")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    n = args.num_requests * len(args.rates) if args.rates else sum(args.levels)
    if args.dataset == "sharegpt":
        if not args.tokenizer:
            parser.error("--tokenizer is required with --dataset sharegpt")
        samples = sharegpt_samples(args.tokenizer, n, args.seed)
    else:
        prompt = next(p for p in PROMPTS if p.name == args.prompt)
        samples = fixed_samples(prompt.text, args.max_tokens, n)

    if args.rates:
        mode, results = "openloop", asyncio.run(run_rates(args, samples))
    else:
        mode, results = "closedloop", run_levels(args, samples)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = results_dir / f"{mode}_{ts}.json"
    with json_path.open("w") as f:
        json.dump({
            "mode": mode,
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

    if mode == "closedloop":
        plot_ttft_vs_throughput(results, args.model, str(results_dir / f"{mode}_{ts}.png"))


def run_levels(args, samples) -> list[dict]:
    provider = OpenAICompatProvider(model=args.model, base_url=args.base_url, ignore_eos=args.ignore_eos)

    def progress(r):
        print(f"  [n={r['concurrency']}] ttft={r['mean_ttft']:.3f}s  tps={r['throughput_tps']:.1f}"
              f"  early_stops={r['early_stops']}/{r['concurrency']}")

    print("Warm up...")
    provider.stream_once(samples[0].prompt, max_tokens=1)
    print(f"Sweeping concurrency {args.levels} on {args.model}")
    return sweep_concurrency(provider, samples, args.levels, progress)


async def run_rates(args, samples) -> list[dict]:
    print(f"Open-file limit: {raise_open_file_limit()}")
    provider = AsyncOpenAICompatProvider(model=args.model, base_url=args.base_url, ignore_eos=args.ignore_eos)

    def progress(r):
        done = [q for q in r["requests"] if q["error"] is None]
        out_tps = sum(q["output_tokens"] for q in done) / r["duration_seconds"]
        errors = Counter(q["error"] for q in r["requests"] if q["error"])
        print(f"  [rate={r['rate']:g}/s] ok={len(done)}/{r['num_requests']}  out_tok/s={out_tps:.1f}"
              f"  send_lag_max={r['send_lag']['max'] * 1e3:.1f}ms" + (f"  errors={dict(errors)}" if errors else ""))
        for w in r["warnings"]:
            print(f"    warning: {w}")

    try:
        print("Warm up...")
        await provider.stream_once(samples[0].prompt, max_tokens=1)
        print(f"Sweeping rates {args.rates} req/s on {args.model}, {args.num_requests} requests each")
        return await sweep_rates(provider, samples, args.rates, args.num_requests,
                                 args.burstiness, args.seed, progress)
    finally:
        await provider.aclose()


if __name__ == "__main__":
    main()
