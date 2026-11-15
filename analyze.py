"""Recompute metrics for a saved open-loop results file, e.g. with a different trim or SLOs."""
import argparse
import json

from harness.metrics import format_summary, summarize, summary_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="results/openloop_*.json")
    parser.add_argument("--trim", type=float, default=0.1, help="fraction dropped from each end, by send order")
    parser.add_argument("--slo-ttft-ms", type=float, help="TTFT target for goodput")
    parser.add_argument("--slo-tpot-ms", type=float, help="TPOT target for goodput")
    parser.add_argument("--save", action="store_true", help="write the new summaries back into the file")
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)
    if data.get("mode") != "openloop":
        parser.error("only open-loop results have the per-request send times these metrics need")

    slo_ttft = args.slo_ttft_ms / 1e3 if args.slo_ttft_ms is not None else None
    slo_tpot = args.slo_tpot_ms / 1e3 if args.slo_tpot_ms is not None else None
    print(summary_header())
    for result in data["results"]:
        result["summary"] = summarize(result, args.trim, slo_ttft, slo_tpot)
        print(format_summary(result["rate"], result["summary"]))

    if args.save:
        with open(args.path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {args.path}")


if __name__ == "__main__":
    main()
