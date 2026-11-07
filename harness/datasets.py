import json
import random
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Length filters match vLLM's ShareGPT sampling so results stay comparable
MIN_LEN = 4
MAX_PROMPT_LEN = 1024
MAX_TOTAL_LEN = 2048


@dataclass
class Sample:
    prompt: str
    prompt_tokens: int | None  # tokenized without chat template; None for fixed prompts
    target_output_tokens: int  # sent as max_tokens


def fixed_samples(prompt: str, max_tokens: int, n: int) -> list[Sample]:
    return [Sample(prompt, None, max_tokens) for _ in range(n)]


def sharegpt_samples(tokenizer_id: str, n: int, seed: int = 0) -> list[Sample]:
    """First user turn as the prompt, first reply's token length as the target output length."""
    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    with open(path) as f:
        conversations = [d["conversations"] for d in json.load(f) if len(d["conversations"]) >= 2]
    random.Random(seed).shuffle(conversations)

    tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    samples: list[Sample] = []
    # Tokenize lazily: only as many conversations as it takes to fill n samples
    for conv in conversations:
        prompt, completion = conv[0]["value"], conv[1]["value"]
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        output_len = len(tokenizer.encode(completion, add_special_tokens=False).ids)
        if prompt_len < MIN_LEN or output_len < MIN_LEN:
            continue
        if prompt_len > MAX_PROMPT_LEN or prompt_len + output_len > MAX_TOTAL_LEN:
            continue
        samples.append(Sample(prompt, prompt_len, output_len))
        if len(samples) == n:
            return samples
    raise ValueError(f"Only {len(samples)} ShareGPT samples passed the length filters; asked for {n}")
