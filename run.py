import argparse
import asyncio
import itertools
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from harness.datasets import Sample, fixed_samples, iter_sharegpt_samples, sharegpt_samples
from harness.knee import DEFAULT_ATTAINMENT, find_knee
from harness.load_runner import sweep_concurrency
from harness.metrics import format_summary, summarize, summary_header
from harness.open_loop import raise_open_file_limit, sweep_rates
from harness.plot import plot_rate_sweep, plot_ttft_vs_throughput, sweep_subtitle
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
    parser.add_argument("--trim", type=float, default=0.1,
                        help="open-loop: fraction of requests dropped from each end (by send order) for metrics")
    parser.add_argument("--slo-ttft-ms", type=float, help="open-loop: TTFT target for goodput")
    parser.add_argument("--slo-tpot-ms", type=float, help="open-loop: TPOT target for goodput")
    parser.add_argument("--find-knee", action="store_true",
                        help="Search for the highest sustainable rate (with SLOs: the highest meeting them)")
    parser.add_argument("--attainment", type=float, default=DEFAULT_ATTAINMENT,
                        help="knee: share of requests that must meet the SLOs")
    parser.add_argument("--knee-start", type=float,
                        help="knee: first rate to probe (default: 1 / latency of a lone request)")
    parser.add_argument("--dataset", default="fixed", choices=["fixed", "sharegpt"])
    parser.add_argument("--tokenizer", help="Hugging Face repo id for token counts (required for sharegpt)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true", help="Ask the server to generate exactly the target length")
    parser.add_argument("--prompt", default="medium", choices=[p.name for p in PROMPTS], help="fixed dataset only")
    parser.add_argument("--max-tokens", type=int, default=512, help="fixed dataset only")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.dataset == "sharegpt" and not args.tokenizer:
        parser.error("--tokenizer is required with --dataset sharegpt")
    prompt = next(p for p in PROMPTS if p.name == args.prompt)
    extra = {}
    if args.find_knee:
        # The search decides how many prompts it needs, so hand it an endless supply
        if args.dataset == "sharegpt":
            stream = iter_sharegpt_samples(args.tokenizer, args.seed)
        else:
            stream = itertools.repeat(Sample(prompt.text, None, args.max_tokens))
        mode = "knee"
        extra["knee"], results = asyncio.run(run_knee(args, stream))
    else:
        n = args.num_requests * len(args.rates) if args.rates else sum(args.levels)
        if args.dataset == "sharegpt":
            samples = sharegpt_samples(args.tokenizer, n, args.seed)
        else:
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
            **extra,
            "results": results,
        }, f, indent=2)
    print(f"Wrote {json_path}")

    plot_path = str(results_dir / f"{mode}_{ts}.png")
    if mode == "closedloop":
        plot_ttft_vs_throughput(results, args.model, plot_path)
    elif mode == "knee":
        knee = extra["knee"]
        latest = {r["rate"]: r for r in results}  # a confirmed rate was probed twice; plot the repeat
        plot_rate_sweep([(args.model, sorted(latest.values(), key=lambda r: r["rate"]))],
                        f"{args.model}: knee search", sweep_subtitle(results, args.dataset), plot_path,
                        knee=(knee["lo"], knee["hi"]))
    elif any(r["rate"] != float("inf") for r in results):
        plot_rate_sweep([(args.model, results)], f"{args.model} under open-loop load",
                        sweep_subtitle(results, args.dataset), plot_path)


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

    slo_ttft = args.slo_ttft_ms / 1e3 if args.slo_ttft_ms is not None else None
    slo_tpot = args.slo_tpot_ms / 1e3 if args.slo_tpot_ms is not None else None

    def progress(r):
        # Stored next to the raw records; analyze.py can recompute it later
        r["summary"] = summarize(r, args.trim, slo_ttft, slo_tpot)
        print(format_summary(r["rate"], r["summary"]))
        for w in r["warnings"]:
            print(f"{'':>8}- warning: {w}")

    try:
        print("Warm up...")
        await provider.stream_once(samples[0].prompt, max_tokens=1)
        print(f"Sweeping rates {args.rates} req/s on {args.model}, {args.num_requests} requests each")
        print(summary_header())
        return await sweep_rates(provider, samples, args.rates, args.num_requests,
                                 args.burstiness, args.seed, progress)
    finally:
        await provider.aclose()


async def run_knee(args, stream) -> tuple[dict, list[dict]]:
    print(f"Open-file limit: {raise_open_file_limit()}")
    provider = AsyncOpenAICompatProvider(model=args.model, base_url=args.base_url, ignore_eos=args.ignore_eos)
    slo_ttft = args.slo_ttft_ms / 1e3 if args.slo_ttft_ms is not None else None
    slo_tpot = args.slo_tpot_ms / 1e3 if args.slo_tpot_ms is not None else None

    def progress(r):
        kp = r["knee_probe"]
        print(f"{'':>8}{kp['role']} -> {'pass' if kp['pass'] else 'FAIL'}")
        print(format_summary(r["rate"], r["summary"]))
        if r["aborted"]:
            a = kp["abort"]
            print(f"{'':>8}- stopped after {r['num_requests']}/{r['num_planned']} sent: {a['certain_misses']} certain"
                  f" SLO misses of {a['window']}, only {a['allowed']} allowed (row shows the requests sent so far)")
        for w in r["warnings"]:
            print(f"{'':>8}- warning: {w}")

    try:
        print("Warm up...")
        await provider.stream_once(next(stream).prompt, max_tokens=1)
        start = args.knee_start
        if start is None:
            # A server running one request at a time sustains 1 / latency; real capacity is higher
            lat = []
            for s in itertools.islice(stream, 3):
                m = await provider.stream_once(s.prompt, s.target_output_tokens)
                lat.append(m.end_time - m.send_time)
            start = 1 / statistics.median(lat)
        print(f"Knee search on {args.model}, starting at {start:.3g} req/s")
        print(summary_header())
        knee, probes = await find_knee(provider, stream, start, slo_ttft, slo_tpot, args.attainment,
                                       args.trim, args.burstiness, args.seed, progress)
    finally:
        await provider.aclose()
    hi = f"{knee['hi']:.3g}" if knee["hi"] is not None else "not found"
    lo = f"{knee['lo']:.3g}" if knee["lo"] is not None else "none passed"
    print(f"Knee ({knee['criterion']}): passes at {lo} req/s, fails at {hi} req/s"
          f"{'' if knee['confirmed'] else '  (not confirmed on repeat)'}")
    return knee, probes


if __name__ == "__main__":
    main()
