import time
from openai import OpenAI
from .base import StreamMeasurement


class LlamaCppProvider:
    name = "llama_cpp"

    # init to ollama localhost 
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key="local")
    
    def stream_once(self, prompt: str, max_tokens: int) -> StreamMeasurement:
        start = time.perf_counter()
        ttft: float | None = None

        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - start
        
        end = time.perf_counter()
        usage = stream.get_final_completion().usage
        if ttft is None:
            ttft = end - start
        
        return StreamMeasurement(
            ttft_seconds=ttft,
            generation_seconds=(end - start) - ttft,
            output_tokens=usage.output_tokens,
            input_tokens=usage.input_tokens,
        )