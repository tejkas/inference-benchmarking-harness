"""Find the highest arrival rate a server sustains: double until a probe fails, then bisect.

With SLOs, a probe passes if enough requests meet them and the backlog isn't growing.
Without, it passes if the backlog isn't growing.
"""
import itertools
import math
from collections.abc import Iterable, Iterator

from .config import Config
from .datasets import Sample
from .metrics import meets_slo, request_metrics, summarize
from .open_loop import LiveRequest, run_open_loop
from .providers.base import AsyncProvider


class SloAbort:
    """Stops a probe once it can no longer reach the attainment target.

    Counts only certain misses inside the window summarize() will use: failed requests,
    finished requests over a target, and requests already waiting longer than the TTFT target.
    """

    def __init__(self, n_planned: int, trim: float, attainment: float,
                 slo_ttft: float | None, slo_tpot: float | None):
        k = int(n_planned * trim)
        self.lo, self.hi = (k, n_planned - k) if n_planned - 2 * k > 0 else (0, n_planned)
        n_window = self.hi - self.lo
        self.budget = n_window - math.ceil(attainment * n_window - 1e-9)  # misses allowed
        self.slo_ttft, self.slo_tpot = slo_ttft, slo_tpot
        self.misses = 0
        self.late = 0  # in-flight requests already past the TTFT target, as of the last check

    def __call__(self, newly_finished: list[LiveRequest], in_flight: Iterable[LiveRequest], now: float) -> bool:
        for live in newly_finished:
            if self.lo <= live.index < self.hi and not meets_slo(_metrics(live), self.slo_ttft, self.slo_tpot):
                self.misses += 1
        late = 0
        if self.slo_ttft is not None:
            for live in in_flight:
                if live.launch_time is None or not self.lo <= live.index < self.hi:
                    continue
                if (live.first_token_time or now) - live.launch_time > self.slo_ttft:
                    late += 1
        self.late = late
        return self.misses + late > self.budget


def _metrics(live: LiveRequest) -> dict | None:
    r = live.result
    if r is None or r.error is not None:
        return None
    return request_metrics({**r.measurement.to_record(0.0), "error": None})


def probe_passes(result: dict, attainment: float, use_slo: bool) -> bool:
    s = result["summary"]
    if result["aborted"]:
        return False
    # Even with enough good requests, a growing backlog means attainment would keep falling.
    # No stability block (rate=inf: everything sent at once) doesn't fail a probe.
    stable = s["stability"] is None or s["stability"]["stable"]
    if use_slo:
        return stable and s["goodput"]["attainment"] >= attainment
    return stable


async def find_knee(
    provider: AsyncProvider,
    samples: Iterator[Sample],
    start_rate: float,
    cfg: Config,
    slo_ttft: float | None = None,
    slo_tpot: float | None = None,
    seed: int = 0,
    on_probe=None,
) -> tuple[dict, list[dict]]:
    """Returns (knee, probes). knee["lo"] is the highest rate that passed, knee["hi"] the lowest
    that failed (None if none failed within the probe budget)."""
    use_slo = slo_ttft is not None or slo_tpot is not None
    probes: list[dict] = []

    async def probe(rate: float, role: str) -> bool:
        n = cfg.probe_size(rate)
        chunk = list(itertools.islice(samples, n))
        if len(chunk) < n:
            raise ValueError(f"dataset ran out: needed {n} more prompts for a probe at {rate:g} req/s")
        abort = SloAbort(n, cfg.trim, cfg.attainment, slo_ttft, slo_tpot) if use_slo else None
        result = await run_open_loop(provider, chunk, rate, cfg, seed, abort_when=abort)
        result["summary"] = summarize(result, cfg, slo_ttft, slo_tpot)
        ok = probe_passes(result, cfg.attainment, use_slo)
        result["knee_probe"] = {"role": role, "pass": ok}
        if result["aborted"]:
            # The summary covers only the requests sent before the stop; this is what decided the probe
            result["knee_probe"]["abort"] = {"certain_misses": abort.misses + abort.late, "allowed": abort.budget,
                                             "window": abort.hi - abort.lo}
        probes.append(result)
        if on_probe:
            on_probe(result)
        return ok

    lo = hi = None
    rate = start_rate
    # Bracket: double while passing, halve while failing, until one of each
    while (lo is None or hi is None) and len(probes) < cfg.max_probes:
        if await probe(rate, "bracket"):
            lo = rate
            rate *= 2
        else:
            hi = rate
            rate /= 2
    # Bisect geometrically: capacity spans orders of magnitude across hardware
    while lo is not None and hi is not None and hi / lo > cfg.knee_resolution and len(probes) < cfg.max_probes:
        mid = math.sqrt(lo * hi)
        if await probe(mid, "bisect"):
            lo = mid
        else:
            hi = mid
    # Confirm: one false pass near the knee would otherwise be the answer
    confirmed = False
    for _ in range(cfg.max_step_downs):
        if lo is None:
            break
        if await probe(lo, "confirm"):
            confirmed = True
            break
        hi, lo = lo, lo / cfg.knee_resolution

    knee = {
        "criterion": f"attainment >= {cfg.attainment:.0%} and stable" if use_slo else "stable",
        "attainment": cfg.attainment if use_slo else None,
        "slo_ttft": slo_ttft,
        "slo_tpot": slo_tpot,
        "lo": lo,
        "hi": hi,
        "confirmed": confirmed,
        "probes": len(probes),
    }
    return knee, probes
