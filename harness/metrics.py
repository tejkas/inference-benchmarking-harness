"""Serving metrics from an open-loop run's raw request records.

Definitions match vLLM's `bench serve` so results can be checked against it. Everything works
on the saved JSON dicts, so old results can be re-analyzed with a different trim or SLOs.
"""
import math
from collections import Counter

import numpy as np

from .config import Config

METRICS_VERSION = 3
TREND_POINTS = 200  # samples of the in-flight count for the stability trend fit


def request_metrics(r: dict) -> dict | None:
    """Per-request latencies in seconds; None for failed requests."""
    if r["error"] is not None:
        return None
    chunks = r["chunk_times"]
    e2e = r["end_time"] - r["send_time"]
    ttft = chunks[0] - r["send_time"] if chunks else None
    n = r["output_tokens"]
    return {
        "ttft": ttft,
        "e2e": e2e,
        # vLLM's TPOT: decode time spread over the tokens after the first
        "tpot": (e2e - ttft) / (n - 1) if ttft is not None and n > 1 else None,
        "itl": np.diff(chunks).tolist(),
        "tokens_per_chunk": n / len(chunks) if chunks else None,
        # Allow one token with no chunk: servers count the EOS token but send no text for it
        "excess_tokens_per_chunk": (n - 1) / len(chunks) if chunks else None,
    }


def meets_slo(m: dict | None, slo_ttft: float | None, slo_tpot: float | None) -> bool:
    """m is request_metrics() output; failed or contentless requests never meet an SLO."""
    if m is None or m["ttft"] is None:
        return False
    if slo_ttft is not None and m["ttft"] > slo_ttft:
        return False
    return slo_tpot is None or m["tpot"] is None or m["tpot"] <= slo_tpot


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99))}


def _finish_time(r: dict) -> float:
    return r["end_time"] if r["error"] is None else r["error_time"]


def _backlog_trend(requests: list[dict], start: float, end: float) -> float:
    """Least-squares slope of the in-flight count over [start, end], in requests per second.

    In flight at t = launched by t - finished by t. A fitted line uses the whole window; the
    difference between the counts at the two ends would carry the noise of both moments.
    """
    launches = np.sort([r["launch_time"] for r in requests])
    finishes = np.sort([_finish_time(r) for r in requests])
    ts = np.linspace(start, end, TREND_POINTS)
    in_flight = np.searchsorted(launches, ts, side="right") - np.searchsorted(finishes, ts, side="right")
    return float(np.polyfit(ts, in_flight, 1)[0])


def summarize(run: dict, cfg: Config, slo_ttft: float | None = None, slo_tpot: float | None = None) -> dict:
    """Metrics over the steady-state window of one open-loop run. SLOs are in seconds.

    The window drops the first and last `trim` fraction of requests by send order. Selecting by
    send time only matters: selecting by finish time would keep the fast requests above
    saturation and hide the tail.
    """
    reqs = sorted(run["requests"], key=lambda r: r["launch_time"])
    k = int(len(reqs) * cfg.trim)
    window = reqs[k:len(reqs) - k] or reqs
    per = {r["index"]: request_metrics(r) for r in reqs}
    done = [per[r["index"]] for r in window if per[r["index"]] is not None]
    notes = []

    ttfts = [m["ttft"] for m in done if m["ttft"] is not None]
    if len(done) > len(ttfts):
        notes.append(f"{len(done) - len(ttfts)} requests finished without any content")
    if len(ttfts) < cfg.min_n_for_p99:
        notes.append(f"only {len(ttfts)} requests in the window: p99 is close to the max")
    tpc = [m["tokens_per_chunk"] for m in done if m["tokens_per_chunk"] is not None]
    median_tpc = float(np.median(tpc)) if tpc else None
    excess = [m["excess_tokens_per_chunk"] for m in done if m["excess_tokens_per_chunk"] is not None]
    if excess and np.median(excess) > cfg.multi_token_chunks:
        notes.append(f"~{median_tpc:.1f} tokens per chunk: ITL is per chunk, not per token")
    # Fewer tokens than asked for: the server stopped at EOS (ignore_eos off or not honored)
    early_stops = sum(r["error"] is None and r["output_tokens"] < r["target_output_tokens"] for r in window)
    if early_stops:
        notes.append(f"{early_stops}/{len(done)} requests stopped before their target length")
    missing = sum(r["error"] is None and r.get("usage_missing") for r in window)
    if missing:
        notes.append(f"{missing} requests had no usage event: token counts are chunk counts")

    start, end = window[0]["launch_time"], window[-1]["launch_time"]
    span = end - start
    timed = not math.isinf(run["rate"]) and span > 0  # rate=inf sends everything at once

    stability = None
    window_tps = None
    if timed:
        arrivals = len(window)
        completions = sum(start <= _finish_time(r) <= end for r in reqs)
        trend = _backlog_trend(reqs, start, end)
        allowed = cfg.stability_tolerance * arrivals
        judgeable = allowed >= 1  # a tolerance under one request can't tell growth from noise
        if not judgeable:
            notes.append(f"only {arrivals} requests in the window: too few to judge stability")
        stability = {
            # Unstable: the backlog's fitted growth over the window exceeds the tolerance
            "stable": trend * span <= allowed if judgeable else None,
            "offered_rps": arrivals / span,
            "served_rps": completions / span,
            "backlog_trend_rps": trend,
            "backlog_growth": trend * span,
            "allowed_growth": allowed,
        }
        # Count tokens as they arrive inside the window, so long requests aren't credited all at once
        window_tokens = sum(
            m["tokens_per_chunk"] * sum(start <= t <= end for t in r["chunk_times"])
            for r in reqs
            if (m := per[r["index"]]) is not None and m["tokens_per_chunk"] is not None
        )
        window_tps = window_tokens / span

    # Secondary signal: TTFT rising from the first half of the window to the second
    half = len(ttfts) // 2
    ttft_trend = float(np.median(ttfts[half:]) / np.median(ttfts[:half])) if half else None

    goodput = None
    if slo_ttft is not None or slo_tpot is not None:
        n_good = sum(meets_slo(per[r["index"]], slo_ttft, slo_tpot) for r in window)
        goodput = {
            "slo_ttft": slo_ttft,
            "slo_tpot": slo_tpot,
            "good": n_good,
            # Failures stay in the denominator, or dropping requests would look like meeting SLOs
            "attainment": n_good / len(window),
            "rps": n_good / span if timed else None,
        }

    completed = [m for r in reqs if (m := per[r["index"]]) is not None]
    run_tokens = sum(r["output_tokens"] for r in reqs if r["error"] is None)
    return {
        "metrics_version": METRICS_VERSION,
        "trim": cfg.trim,
        "stability_tolerance": cfg.stability_tolerance,
        "window_requests": len(window),
        "window_completed": len(done),
        "failures": dict(Counter(r["error"] for r in window if r["error"])),
        "stability": stability,
        "ttft_trend": ttft_trend,
        "ttft": distribution(ttfts),
        "tpot": distribution([m["tpot"] for m in done if m["tpot"] is not None]),
        "itl": distribution([g for m in done for g in m["itl"]]),
        "e2e": distribution([m["e2e"] for m in done]),
        "tokens_per_chunk": median_tpc,
        "early_stops": early_stops,
        "output_tps": {"run": run_tokens / run["duration_seconds"], "window": window_tps},
        "completed_total": len(completed),
        "goodput": goodput,
        "notes": notes,
    }


def summary_header() -> str:
    return (f"{'rate':>6} {'ok':>8} {'stable':>6} {'TTFT p50/p99 ms':>16} {'TPOT p50/p99 ms':>16}"
            f" {'ITL p99':>8} {'tok/s':>7} {'good':>6}")


def format_summary(rate: float, s: dict) -> str:
    def ms(d, key):
        return f"{d[key] * 1e3:.0f}" if d else "-"
    st = s["stability"]
    stable = "n/a" if st is None or st["stable"] is None else ("yes" if st["stable"] else "NO")
    tps = s["output_tps"]["window"] or s["output_tps"]["run"]
    good = f"{s['goodput']['attainment']:.0%}" if s["goodput"] else "-"
    line = (f"{rate:>6g} {s['window_completed']:>4}/{s['window_requests']:<3} {stable:>6}"
            f" {ms(s['ttft'], 'p50') + '/' + ms(s['ttft'], 'p99'):>16}"
            f" {ms(s['tpot'], 'p50') + '/' + ms(s['tpot'], 'p99'):>16}"
            f" {ms(s['itl'], 'p99'):>8} {tps:>7.1f} {good:>6}")
    extra = list(s["notes"])
    if s["failures"]:
        extra.insert(0, f"failures: {s['failures']}")
    return "\n".join([line] + [f"{'':>8}- {n}" for n in extra])
