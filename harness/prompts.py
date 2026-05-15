from dataclasses import dataclass


@dataclass
class Prompt:
    name: str
    text: str

# Three prompts of different complexity
PROMPTS: list[Prompt] = [
    Prompt(
        name="short",
        text="What is the capital of France? Answer in one sentence.",
    ),
    Prompt(
        name="medium",
        text=(
            "Explain how a transformer language model generates text, "
            "covering tokenization, attention, and sampling. Aim for about 200 words."
        ),
    ),
    Prompt(
        name="long",
        text=(
            "Write a detailed technical comparison of three approaches to scaling "
            "inference for large language models: tensor parallelism, pipeline "
            "parallelism, and continuous batching. For each, describe how it works, "
            "what hardware it requires, the latency/throughput tradeoffs, and which "
            "deployment scenarios it fits best. Aim for about 600 words."
        ),
    ),
]
