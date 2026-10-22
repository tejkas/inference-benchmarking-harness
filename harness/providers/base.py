from dataclasses import dataclass
from typing import Protocol


@dataclass
class StreamMeasurement:
    # Absolute time.perf_counter() values, so requests in one run share a clock
    send_time: float
    chunk_times: list[float]  # arrival of each content-bearing chunk
    end_time: float
    output_tokens: int
    input_tokens: int

    @property
    def ttft_seconds(self) -> float:
        if not self.chunk_times:
            return self.end_time - self.send_time
        return self.chunk_times[0] - self.send_time

    @property
    def generation_seconds(self) -> float:
        return (self.end_time - self.send_time) - self.ttft_seconds

    @property
    def tokens_per_second(self) -> float:
        if self.generation_seconds <= 0:
            return 0.0
        return self.output_tokens / self.generation_seconds

    def to_record(self, origin: float) -> dict:
        """Raw per-request record with times in seconds relative to origin."""
        return {
            "send_time": self.send_time - origin,
            "chunk_times": [t - origin for t in self.chunk_times],
            "end_time": self.end_time - origin,
            "output_tokens": self.output_tokens,
            "input_tokens": self.input_tokens,
        }


class Provider(Protocol):
    name: str
    model: str

    def stream_once(self, prompt: str, max_tokens: int) -> StreamMeasurement:
        """Send one streaming request and return timing + token measurements."""
        ...
