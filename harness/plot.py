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