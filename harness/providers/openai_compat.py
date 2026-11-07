import time
from openai import OpenAI
from .base import StreamMeasurement


class OpenAICompatProvider:
    """Any server exposing the OpenAI chat completions API (Ollama, llama.cpp, vLLM, SGLang)."""

    name = "openai_compat"

    def __init__(self, model: str, base_url: str, ignore_eos: bool = False):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key="local")
        # Non-standard field that not every server honors; compare output_tokens against the target
        self._extra_body = {"ignore_eos": True} if ignore_eos else None

    def stream_once(self, prompt: str, max_tokens: int) -> StreamMeasurement:
        send_time = time.perf_counter()
        chunk_times: list[float] = []

        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self._extra_body,
        )

        last_chunk = None # Needed to capture usage statistics
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_times.append(time.perf_counter())
            last_chunk = chunk

        end_time = time.perf_counter()
        usage = last_chunk.usage if last_chunk else None

        return StreamMeasurement(
            send_time=send_time,
            chunk_times=chunk_times,
            end_time=end_time,
            output_tokens=usage.completion_tokens if usage else 0,
            input_tokens=usage.prompt_tokens if usage else 0,
        )
