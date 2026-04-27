import statistics
from dataclasses import dataclass, asdict


@dataclass
class RunResult:
    provider: str
    model: str
    prompt_name: str
    iteration: int
    ttft_seconds: float
    generation_seconds: float
    output_tokens: int
    input_tokens: int
    tokens_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(results: list[RunResult]) -> list[dict]:
    """Group by (model, prompt_name) and compute mean / p50 / p95 of TTFT and tokens/sec."""
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.model, r.prompt_name), []).append(r)

    summary = []
    for (model, prompt_name), runs in groups.items():
        ttfts = [r.ttft_seconds for r in runs]
        tps = [r.tokens_per_second for r in runs]
        summary.append({
            "model": model,
            "prompt": prompt_name,
            "n": len(runs),
            "ttft_mean": statistics.mean(ttfts),
            "ttft_p50": _percentile(ttfts, 50),
            "ttft_p95": _percentile(ttfts, 95),
            "tps_mean": statistics.mean(tps),
            "tps_p50": _percentile(tps, 50),
            "tps_p95": _percentile(tps, 95),
        })
    return summary


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def format_summary_table(summary: list[dict]) -> str:
    headers = ["model", "prompt", "n", "ttft_mean", "ttft_p50", "ttft_p95", "tps_mean", "tps_p50", "tps_p95"]
    rows = [headers]
    for s in summary:
        rows.append([
            s["model"],
            s["prompt"],
            str(s["n"]),
            f"{s['ttft_mean']:.3f}",
            f"{s['ttft_p50']:.3f}",
            f"{s['ttft_p95']:.3f}",
            f"{s['tps_mean']:.1f}",
            f"{s['tps_p50']:.1f}",
            f"{s['tps_p95']:.1f}",
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
