from .metrics import RunResult
from .prompts import Prompt
from .providers.base import Provider


def run_benchmark(
    provider: Provider,
    prompts: list[Prompt],
    iterations: int,
    max_tokens: int,
    on_progress=None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for prompt in prompts:
        for i in range(iterations):
            m = provider.stream_once(prompt.text, max_tokens=max_tokens)
            result = RunResult(
                provider=provider.name,
                model=provider.model,
                prompt_name=prompt.name,
                iteration=i,
                ttft_seconds=m.ttft_seconds,
                generation_seconds=m.generation_seconds,
                output_tokens=m.output_tokens,
                input_tokens=m.input_tokens,
                tokens_per_second=m.tokens_per_second,
            )
            results.append(result)
            if on_progress:
                on_progress(result)
    return results
