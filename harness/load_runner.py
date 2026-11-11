import time
from concurrent.futures import ThreadPoolExecutor

from .datasets import Sample
from .providers.base import Provider


def run_concurrency_level(provider: Provider, samples: list[Sample]) -> dict:
    """Send one request per sample, all at once, and wait for every one to finish."""
    concurrency = len(samples)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        start = time.perf_counter()
        futures = [
            executor.submit(provider.stream_once, s.prompt, s.target_output_tokens)
            for s in samples
        ]
        measurements = [f.result() for f in futures]
        end = time.perf_counter() # calculate end within thread

    wall_clock = end - start
    total_tokens = sum(m.output_tokens for m in measurements)
    return {
        "concurrency": concurrency,
        "mean_ttft": sum(m.ttft_seconds for m in measurements) / len(measurements),
        "throughput_tps": total_tokens / wall_clock,
        "wall_clock_seconds": wall_clock,
        "total_tokens": total_tokens,
        # Requests that generated fewer tokens than asked for (server ignored ignore_eos, or it was off)
        "early_stops": sum(m.output_tokens < s.target_output_tokens for m, s in zip(measurements, samples)),
        "requests": [
            {
                **m.to_record(start),
                "dataset_prompt_tokens": s.prompt_tokens,
                "target_output_tokens": s.target_output_tokens,
            }
            for m, s in zip(measurements, samples)
        ],
    }

def sweep_concurrency(
      provider: Provider,
      samples: list[Sample],
      levels: list[int],
      on_progress=None,
  ) -> list[dict]:
      """Each level takes the next len(level) samples, so no prompt is sent twice."""
      results = []
      offset = 0
      for concurrency in levels:
          result = run_concurrency_level(provider, samples[offset:offset + concurrency])
          offset += concurrency
          results.append(result)
          if on_progress:
              on_progress(result)
      return results
