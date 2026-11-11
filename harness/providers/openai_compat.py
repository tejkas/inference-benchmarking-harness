import json
import time

import httpx
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


class AsyncOpenAICompatProvider:
    """Async version for open-loop load. Parses the SSE stream directly instead of using the
    openai client, which costs ~37 us per chunk building objects vs ~1.5 us for json.loads.
    All streams share one event loop, so that cost would delay every other stream's timestamps.
    """

    name = "openai_compat"

    def __init__(self, model: str, base_url: str, ignore_eos: bool = False, timeout: float = 600.0):
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._ignore_eos = ignore_eos
        self._client = httpx.AsyncClient(
            # No connection cap: a pool limit queues requests on the client, which closes the loop again
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def stream_once(self, prompt: str, max_tokens: int) -> StreamMeasurement:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._ignore_eos:
            body["ignore_eos"] = True

        chunk_times: list[float] = []
        usage = None
        send_time = time.perf_counter()
        async with self._client.stream("POST", self._url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                now = time.perf_counter()  # stamp before parsing, so parse time doesn't skew it
                if not line.startswith("data:"):
                    continue  # blank separators and ": keep-alive" comments
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices")
                if choices and choices[0].get("delta", {}).get("content"):
                    chunk_times.append(now)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        end_time = time.perf_counter()

        return StreamMeasurement(
            send_time=send_time,
            chunk_times=chunk_times,
            end_time=end_time,
            output_tokens=usage["completion_tokens"] if usage else 0,
            input_tokens=usage["prompt_tokens"] if usage else 0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
