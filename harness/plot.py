import matplotlib.pyplot as plt


def plot_ttft_vs_throughput(results: list[dict], model: str, output_path: str) -> None:
    throughputs = [r["throughput_tps"] for r in results]
    ttfts = [r["mean_ttft"] for r in results]
    concurrencies = [r["concurrency"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(throughputs, ttfts, marker="o", linewidth=1.5)

    for x, y, c in zip(throughputs, ttfts, concurrencies):
        ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("Mean TTFT (seconds)")
    ax.set_title(f"TTFT vs Throughput — {model}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

# Reference palette: chart chrome plus the first three categorical slots, which stay
# distinguishable for colorblind readers in every pairing (checked with the validator)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS, labelcolor=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("offered rate (req/s)", color=INK_2)


def _series(ax, points, color, linestyle) -> None:
    """points: (rate, value, stable). Unstable points are hollow, so the flag doesn't rely on color."""
    points = [p for p in points if p[1] is not None]
    if not points:
        return
    xs, ys, stable = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=2, linestyle=linestyle, solid_capstyle="round",
            dash_capstyle="round", zorder=2)
    for x, y, ok in zip(xs, ys, stable):
        ax.plot([x], [y], marker="o", markersize=9, linestyle="none", zorder=3, markeredgewidth=2,
                markerfacecolor=color if ok else SURFACE, markeredgecolor=SURFACE if ok else color)


def _passed(r: dict) -> bool:
    if "knee_probe" in r:
        return r["knee_probe"]["pass"]
    return r["summary"]["stability"] is None or r["summary"]["stability"]["stable"]


def _ms_ticks(v, _):
    return f"{v:,.0f}" if v < 10_000 else f"{v / 1000:,.0f}K"


def plot_rate_sweep(runs: list[tuple[str, list[dict]]], title: str, subtitle: str, output_path: str,
                    knee: tuple[float | None, float | None] | None = None) -> None:
    """TTFT, TPOT and goodput (or throughput) against offered rate, one color per run.

    runs: (label, open-loop results with a "summary" each). rate=inf points are skipped.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if len(runs) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} runs per plot stay distinguishable; split the comparison")
    runs = [(label, [r for r in res if r["rate"] != float("inf")]) for label, res in runs]
    use_goodput = all(r["summary"]["goodput"] for _, res in runs for r in res)

    fig, (ax_ttft, ax_tpot, ax_out) = plt.subplots(1, 3, figsize=(13, 4.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax_ttft, "TTFT (ms, log scale)")
    _style(ax_tpot, "TPOT (ms)")
    _style(ax_out, "Goodput (req/s)" if use_goodput else "Output throughput (tok/s)")

    for (label, res), color in zip(runs, SERIES):
        def pts(get):
            return [(r["rate"], get(r["summary"]), _passed(r)) for r in res]
        for q, ls in (("p50", "-"), ("p99", "--")):
            _series(ax_ttft, pts(lambda s: s["ttft"] and s["ttft"][q] * 1e3), color, ls)
            _series(ax_tpot, pts(lambda s: s["tpot"] and s["tpot"][q] * 1e3), color, ls)
        if use_goodput:
            # attainment x nominal rate, not the measured rps: with few requests the random arrivals
            # drift from the nominal rate, and 100% attainment should sit on the "offered" line
            _series(ax_out, [(x, x * r["summary"]["goodput"]["attainment"], ok)
                             for (x, _, ok), r in zip(pts(lambda s: 0), res)], color, "-")
        else:
            _series(ax_out, pts(lambda s: s["output_tps"]["window"] or s["output_tps"]["run"]), color, "-")

    if use_goodput:
        # Perfect service: every offered request meets the SLOs
        rates = sorted({r["rate"] for _, res in runs for r in res})
        if rates:
            ax_out.plot([0, rates[-1]], [0, rates[-1]], color=MUTED, linewidth=1, zorder=1)
            ax_out.annotate("offered", (rates[-1], rates[-1]), color=MUTED, fontsize=9,
                            textcoords="offset points", xytext=(-4, -14), ha="right")

    ax_ttft.set_yscale("log")
    ax_ttft.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_ttft.yaxis.set_major_formatter(FuncFormatter(_ms_ticks))
    ax_ttft.yaxis.set_minor_formatter(NullFormatter())
    for ax in (ax_tpot, ax_out):
        ax.set_ylim(bottom=0)
    for ax in (ax_ttft, ax_tpot, ax_out):
        ax.set_xlim(left=0)

    if knee and knee[0] is not None:
        # Shade between the highest passing and lowest failing rate: the knee is somewhere in here
        for ax in (ax_ttft, ax_tpot, ax_out):
            ax.axvspan(knee[0], knee[1] if knee[1] is not None else knee[0], color=GRID, alpha=0.6, zorder=0,
                       linewidth=0)

    hollow = "failed probe" if any("knee_probe" in r for _, res in runs for r in res) else "unstable (backlog growing)"
    handles = [Line2D([], [], color=c, linewidth=2, label=label) for (label, _), c in zip(runs, SERIES)]
    handles += [
        Line2D([], [], color=INK_2, linewidth=2, label="p50"),
        Line2D([], [], color=INK_2, linewidth=2, linestyle="--", label="p99"),
        Line2D([], [], marker="o", linestyle="none", markersize=9, markeredgewidth=2,
               markerfacecolor=SURFACE, markeredgecolor=INK_2, label=hollow),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.9), ncol=len(handles),
               frameon=False, labelcolor=INK_2, fontsize=9)
    fig.text(0.01, 0.97, title, color=INK, fontsize=13, va="top")
    fig.text(0.01, 0.925, subtitle, color=INK_2, fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def sweep_subtitle(results: list[dict], dataset: str) -> str:
    s = results[0]["summary"]
    sizes = sorted({r["num_planned"] if "num_planned" in r else r["num_requests"] for r in results})
    per_rate = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]}"
    parts = [f"{dataset} prompts", f"{per_rate} requests per rate",
             f"middle {1 - 2 * s['trim']:.0%} by send order"]
    g = s["goodput"]
    if g:
        slos = [f"TTFT ≤ {g['slo_ttft'] * 1e3:g} ms" if g["slo_ttft"] is not None else None,
                f"TPOT ≤ {g['slo_tpot'] * 1e3:g} ms" if g["slo_tpot"] is not None else None]
        parts.append("goodput SLO " + ", ".join(x for x in slos if x))
    return " · ".join(parts)
