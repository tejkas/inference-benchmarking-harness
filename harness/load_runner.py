import time
from concurrent.futures import ThreadPoolExecutor

from .providers.base import Provider
from .prompts import Prompt


def run_concurrency_level(
    provider: Provider,
    prompt: Prompt,
    concurrency: int,
    max_tokens: int,
) -> dict:
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        start = time.perf_counter()
        futures = [
            executor.submit(provider.stream_once, prompt.text, max_tokens)
            for _ in range(concurrency)
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
    }

def sweep_concurrency(
      provider: Provider,
      prompt: Prompt,
      levels: list[int],
      max_tokens: int,
      on_progress=None,
  ) -> list[dict]:
      results = []
      for concurrency in levels:
          result = run_concurrency_level(provider, prompt, concurrency, max_tokens)
          results.append(result)
          if on_progress:
              on_progress(result)
      return results