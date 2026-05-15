# inference_benchmarking_harness

A harness for measuring LLM inference latency and throughput across providers.

**Current metrics:** time-to-first-token (TTFT), tokens/sec.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
python run.py --model claude-sonnet-4-6 --iterations 5 --max-tokens 2048
```

Output: a printed summary table plus a JSON file under `results/run_<UTC timestamp>.json`
containing every individual run and the aggregated stats.

## How metrics are measured

- **TTFT**: wall-clock time from request send to the first streamed text token.
- **tokens/sec**: `output_tokens / (total_duration - ttft)` — pure generation throughput
  with the prefill latency excluded (TTFT already captures that).
- Token counts come from the API's `usage` field (server truth), not local tokenization.

## Layout

```
harness/
├── providers/
│   ├── base.py          # Provider Protocol + StreamMeasurement
│   └── anthropic.py     # AnthropicProvider (streaming via SDK)
├── prompts.py           # fixed short / medium / long prompt set
├── metrics.py           # RunResult, summarize, format_summary_table
└── runner.py            # sequential run loop
run.py                   # CLI entry point
```
