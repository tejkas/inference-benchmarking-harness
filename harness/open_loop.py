import asyncio
import errno
import json
import math
import random
import resource
import time
from dataclasses import dataclass

import httpx

from .datasets import Sample
from .providers.base import AsyncProvider, StreamMeasurement

LAG_PROBE_INTERVAL = 0.010  # seconds between event-loop lag probes
LAG_WARN_SECONDS = 0.005  # p99 lag above this makes per-chunk timestamps suspect


def arrival_offsets(rate: float, n: int, burstiness: float = 1.0, seed: int = 0) -> list[float]:
    """Send times in seconds from the start of the run, for n requests averaging `rate` per second.

    Gaps between arrivals are gamma-distributed with mean 1/rate. burstiness=1 gives
    exponential gaps (a Poisson process); below 1 is burstier, above 1 more regular.
    rate=inf sends everything at once.
    """
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    shape = burstiness
    scale = 1.0 / (rate * burstiness)  # keeps the mean gap at shape * scale = 1/rate
    offsets = []
    t = 0.0
    for _ in range(n):
        offsets.append(t)
        t += rng.gammavariate(shape, scale)
    return offsets


@dataclass
class RequestResult:
    # Absolute time.perf_counter() values, on the same clock as StreamMeasurement
    index: int
    scheduled_time: float  # when the arrival schedule said to send
    launch_time: float  # when the request was actually started
    measurement: StreamMeasurement | None  # None if the request failed
    error: str | None
    error_time: float | None  # when the failure surfaced

    def to_record(self, origin: float, sample: Sample) -> dict:
        record = {
            "index": self.index,
            "scheduled_time": self.scheduled_time - origin,
            "launch_time": self.launch_time - origin,
            "error": self.error,
            "error_time": self.error_time - origin if self.error_time is not None else None,
            "dataset_prompt_tokens": sample.prompt_tokens,
            "target_output_tokens": sample.target_output_tokens,
        }
        if self.measurement:
            record.update(self.measurement.to_record(origin))
        return record


def classify_error(exc: Exception) -> str:
    """Short, stable label for why a request failed, so failures can be grouped in analysis."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        # Running out of file descriptors is a client limit, not the server refusing connections
        if _caused_by_errno(exc, errno.EMFILE):
            return "client_fd_limit"
        return "connect"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"  # no bytes within the timeout, usually stuck in the server's queue
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError)):
        return "disconnected"  # connection closed or reset mid-response
    if isinstance(exc, json.JSONDecodeError):
        return "bad_sse"
    return type(exc).__name__


def _caused_by_errno(exc: BaseException, code: int) -> bool:
    """httpx wraps the underlying OSError, so walk the exception chain looking for it."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno == code:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def _run_one(provider: AsyncProvider, index: int, sample: Sample, scheduled_time: float) -> RequestResult:
    launch_time = time.perf_counter()
    try:
        m = await provider.stream_once(sample.prompt, sample.target_output_tokens)
        return RequestResult(index, scheduled_time, launch_time, m, None, None)
    # Exception, not BaseException: CancelledError must still propagate so Ctrl-C stops the run
    except Exception as exc:
        return RequestResult(index, scheduled_time, launch_time, None, classify_error(exc), time.perf_counter())


async def _probe_loop_lag(lags: list[float]) -> None:
    """Sleep in short steps and record how late each wakeup is: how busy the event loop is."""
    while True:
        target = time.perf_counter() + LAG_PROBE_INTERVAL
        await asyncio.sleep(LAG_PROBE_INTERVAL)
        lags.append(time.perf_counter() - target)


def _summary(values: list[float]) -> dict | None:
    if not values:
        return None
    s = sorted(values)
    def pick(q: float) -> float:
        return s[min(len(s) - 1, int(q * len(s)))]
    return {"p50": pick(0.5), "p99": pick(0.99), "max": s[-1]}


async def run_open_loop(
    provider: AsyncProvider,
    samples: list[Sample],
    rate: float,
    burstiness: float = 1.0,
    seed: int = 0,
) -> dict:
    """Send one request per sample on a gamma arrival schedule, never waiting for earlier ones.

    Records every request, including failures and the ramp-up and drain at either end.
    Deciding which requests count toward a metric is left to analysis.
    """
    offsets = arrival_offsets(rate, len(samples), burstiness, seed)
    loop_lags: list[float] = []
    monitor = asyncio.create_task(_probe_loop_lag(loop_lags))
    # The event loop only holds weak references to tasks, so keep them here or they can be collected
    tasks: list[asyncio.Task] = []
    origin = time.perf_counter()
    try:
        for i, (sample, offset) in enumerate(zip(samples, offsets)):
            scheduled = origin + offset
            # Sleep toward the absolute schedule so overshoots don't accumulate.
            # If we're already late, send now (catch up) and let launch_time show the lag.
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(_run_one(provider, i, sample, scheduled)))
        results = await asyncio.gather(*tasks)
    finally:
        monitor.cancel()
        for t in tasks:
            t.cancel()  # no-op for finished tasks; stops in-flight ones on Ctrl-C
    end = time.perf_counter()

    send_lags = [r.launch_time - r.scheduled_time for r in results]
    loop_lag = _summary(loop_lags)
    warnings = []
    if loop_lag and loop_lag["p99"] > LAG_WARN_SECONDS:
        warnings.append(f"event loop p99 lag {loop_lag['p99'] * 1e3:.1f} ms: chunk timestamps may be late by this much")
    if max(send_lags, default=0) > LAG_WARN_SECONDS:
        warnings.append(f"max send lag {max(send_lags) * 1e3:.1f} ms: arrivals did not follow the schedule")

    return {
        "rate": rate,
        "burstiness": burstiness,
        "seed": seed,
        "num_requests": len(samples),
        "num_failed": sum(r.error is not None for r in results),
        "duration_seconds": end - origin,
        "send_lag": _summary(send_lags),
        "loop_lag": loop_lag,
        "warnings": warnings,
        "requests": [r.to_record(origin, s) for r, s in zip(results, samples)],
    }


def raise_open_file_limit(target: int = 65536) -> int:
    """Each in-flight request holds a socket; many Linux clusters default to 1024 open files."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except (ValueError, OSError):
            pass
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


async def sweep_rates(
    provider: AsyncProvider,
    samples: list[Sample],
    rates: list[float],
    num_requests: int,
    burstiness: float = 1.0,
    seed: int = 0,
    on_progress=None,
) -> list[dict]:
    """Each rate takes the next num_requests samples, so no prompt is sent twice.

    Every rate reuses the same seed, so the gap pattern is identical, just scaled by 1/rate.
    Differences between rates then come from the load, not from a luckier burst pattern.
    """
    results = []
    for i, rate in enumerate(rates):
        chunk = samples[i * num_requests:(i + 1) * num_requests]
        result = await run_open_loop(provider, chunk, rate, burstiness, seed)
        results.append(result)
        if on_progress:
            on_progress(result)
    return results
