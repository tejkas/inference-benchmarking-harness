"""Recompute metrics for saved open-loop results, e.g. with a different trim or SLOs.

Each file is re-analyzed with the settings it was recorded with (its "config" block; built-in
defaults for older files), then --config and flags on top. Several files can be compared on one
plot, e.g. the same model on two engines.
"""
import argparse
import json

from harness.config import Config, load_config
from harness.metrics import format_summary, summarize, summary_header
from harness.plot import plot_rate_sweep, sweep_subtitle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="results/openloop_*.json")
    parser.add_argument("--labels", nargs="+", help="one per file for the plot legend (default: model name)")
    parser.add_argument("--config", help="TOML file overriding the saved settings (see harness/config.py)")
    parser.add_argument("--trim", type=float, help="fraction dropped from each end, by send order")
    parser.add_argument("--slo-ttft-ms", type=float, help="TTFT target for goodput")
    parser.add_argument("--slo-tpot-ms", type=float, help="TPOT target for goodput")
    parser.add_argument("--plot", help="write a TTFT / TPOT / goodput vs rate plot to this PNG")
    parser.add_argument("--save", action="store_true", help="write the new summaries back into each file")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.paths):
        parser.error("--labels needs one label per file")

    slo_ttft = args.slo_ttft_ms / 1e3 if args.slo_ttft_ms is not None else None
    slo_tpot = args.slo_tpot_ms / 1e3 if args.slo_tpot_ms is not None else None
    runs = []
    for i, path in enumerate(args.paths):
        with open(path) as f:
            data = json.load(f)
        if data.get("mode") not in ("openloop", "knee"):
            parser.error(f"{path}: only open-loop results have the per-request send times these metrics need")
        label = args.labels[i] if args.labels else data["model"]
        cfg = load_config(args.config, base=Config.from_dict(data.get("config", {})), trim=args.trim)
        print(f"\n{label}  ({path})")
        print(summary_header())
        for result in data["results"]:
            result["summary"] = summarize(result, cfg, slo_ttft, slo_tpot)
            print(format_summary(result["rate"], result["summary"]))
        runs.append((label, data["results"]))
        if args.save:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated {path}")

    if args.plot:
        title = runs[0][0] if len(runs) == 1 else " vs ".join(label for label, _ in runs)
        plot_rate_sweep(runs, f"{title} under open-loop load", sweep_subtitle(runs[0][1], data["dataset"]), args.plot)


if __name__ == "__main__":
    main()
