"""Every methodology setting in one place, each with where its default comes from.

  convention  matches published tools or papers, so results are comparable
  statistics  follows from what a measurement needs to mean anything
  judgment    a reasonable choice, not derived; tune it if you have a reason
  safety      bounds the run itself, doesn't change what a result means

The settings are saved into every results file, so a result always records how it was made.
Override them with a TOML file (run.py --config) whose keys are these field names; command-line
flags win over the file. Per-experiment inputs (model, SLOs, rates, seed) stay on the command line.
"""
import math
import tomllib
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    # --- Load generation ---
    # convention: 1 = Poisson arrivals, the standard model of independent users. Spread of the
    # gaps between requests (std / mean): 0 = evenly spaced, above 1 = bursty.
    # vLLM's --burstiness is 1 / arrival_cv**2.
    arrival_cv: float = 1.0

    # --- Workload: ShareGPT length filters ---
    # convention: vLLM's ShareGPT sampling, so prompts match `vllm bench serve`
    sharegpt_min_len: int = 4
    sharegpt_max_prompt_len: int = 1024
    sharegpt_max_total_len: int = 2048

    # --- Steady-state window ---
    # judgment: fraction of requests dropped from each end, by send order, to skip ramp-up and drain
    trim: float = 0.1

    # --- Stability ---
    # judgment: unstable if the in-flight count's fitted trend grows by more than this fraction
    # of arrivals over the window. Stable vLLM probes stayed within ±2.3%, overloaded ones
    # were at 6.9% or more (H100 PCIe, 2026-11-26).
    stability_tolerance: float = 0.05

    # --- Goodput and knee search ---
    # convention: DistServe's goodput definition: at least 90% of requests meet the SLOs
    attainment: float = 0.9
    # judgment: stop bisecting when the failing rate is within this factor of the passing rate;
    # repeat probes at one rate differ by a few percent, so finer brackets mostly measure noise
    knee_resolution: float = 1.1
    # statistics: at least 100 latencies, so p99 isn't just the slowest request
    probe_min_requests: int = 100
    # judgment: long enough for a slowly growing backlog to show, at any rate
    probe_min_seconds: float = 60
    # safety: total probes per search
    max_probes: int = 16
    # safety: re-probes at lower rates when the final passing rate fails on repeat
    max_step_downs: int = 3
    # judgment: lone requests timed to pick the first rate (1 / median latency)
    knee_start_samples: int = 3

    # --- Measurement checks ---
    # judgment: event-loop or send lag above this (p99) makes chunk timestamps suspect
    lag_warn_ms: float = 5.0
    # statistics: fewer latencies than this and p99 is effectively the max (flagged in notes)
    min_n_for_p99: int = 100
    # judgment: median tokens per chunk above this means ITL is per chunk, not per token
    multi_token_chunks: float = 1.05

    def __post_init__(self):
        checks = [
            (self.arrival_cv >= 0, "arrival_cv must be >= 0"),
            (0 <= self.trim < 0.5, "trim must be in [0, 0.5)"),
            (self.stability_tolerance > 0, "stability_tolerance must be > 0"),
            (0 < self.attainment <= 1, "attainment must be in (0, 1]"),
            (self.knee_resolution > 1, "knee_resolution must be > 1"),
            (self.probe_min_requests > 0 and self.max_probes > 0, "probe counts must be > 0"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(f"config: {message}")

    def probe_size(self, rate: float) -> int:
        """Requests per knee probe: enough for p99, and long enough in time for trends to show."""
        return max(self.probe_min_requests, math.ceil(rate * self.probe_min_seconds))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """For configs saved in results files; keys from other versions are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_config(path: str | None = None, base: Config | None = None, **overrides) -> Config:
    """base (default: built-in defaults), then the TOML file, then overrides that aren't None."""
    cfg = base or Config()
    if path:
        with open(path, "rb") as f:
            values = tomllib.load(f)
        unknown = set(values) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
        cfg = replace(cfg, **values)
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
