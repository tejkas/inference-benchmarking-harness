from dataclasses import dataclass
from typing import Protocol


@dataclass
class StreamMeasurement:
    ttft_seconds: float
    generation_seconds: float
    output_tokens: int
    input_tokens: int

    @property
    def tokens_per_second(self) -> float:
        if self.generation_seconds <= 0:
            return 0.0
        return self.output_tokens / self.generation_seconds


class Provider(Protocol):
    name: str
    model: str

    def stream_once(self, prompt: str, max_tokens: int) -> StreamMeasurement:
        """Send one streaming request and return timing + token measurements."""
        ...
