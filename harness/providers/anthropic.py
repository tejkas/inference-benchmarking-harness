import time

import anthropic

from .base import StreamMeasurement


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model
        self._client = anthropic.Anthropic()

    def stream_once(self, prompt: str, max_tokens: int) -> StreamMeasurement:
        start = time.perf_counter()
        ttft: float | None = None

        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for _ in stream.text_stream:
                if ttft is None:
                    ttft = time.perf_counter() - start
            final = stream.get_final_message()

        end = time.perf_counter()

        if ttft is None:
            # No text was streamed (empty response). Treat TTFT as full duration.
            ttft = end - start

        return StreamMeasurement(
            ttft_seconds=ttft,
            generation_seconds=(end - start) - ttft,
            output_tokens=final.usage.output_tokens,
            input_tokens=final.usage.input_tokens,
        )
