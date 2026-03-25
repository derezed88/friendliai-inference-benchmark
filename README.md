# Inference Engine Benchmark: FriendliAI vs vLLM

A reproducible benchmarking tool that compares inference performance between the FriendliAI Engine and vLLM (or any OpenAI-compatible inference server).

## Quick Start

### Prerequisites

```bash
pip install aiohttp matplotlib numpy
```

Both inference engines must be deployed and accessible via their OpenAI-compatible `/v1/chat/completions` endpoints before running.

### Run

```bash
python benchmark.py \
    --friendli-url https://api.friendli.ai/serverless/v1 \
    --friendli-key flp_YOUR_TOKEN \
    --friendli-model meta-llama-3.1-8b-instruct \
    --vllm-url http://localhost:8000/v1 \
    --vllm-model meta-llama/Llama-3.1-8B-Instruct
```

**Note:** FriendliAI Serverless uses model IDs (e.g. `meta-llama-3.1-8b-instruct`) while vLLM typically uses HuggingFace names (e.g. `meta-llama/Llama-3.1-8B-Instruct`). Run `curl -H "Authorization: Bearer $TOKEN" https://api.friendli.ai/serverless/v1/models` to list available model IDs.

### Optional Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrency` | `1,4,8,16,32` | Concurrency levels to test |
| `--requests-per-level` | `16` | Total requests per concurrency level |
| `--max-tokens` | `256` | Max output tokens per request |
| `--output` | `benchmark_results.png` | Output chart filename |
| `--vllm-key` | *(empty)* | vLLM API key, if authentication is enabled |

## Sample Run

The following run demonstrates the benchmark using two models on FriendliAI Serverless (Llama 3.1 8B as "FriendliAI" and Llama 3.3 70B as "vLLM" stand-in) to validate the tooling. In a real evaluation, both engines would serve the same model.

```
$ python benchmark.py \
    --friendli-url https://api.friendli.ai/serverless/v1 \
    --friendli-key flp_... \
    --friendli-model meta-llama-3.1-8b-instruct \
    --vllm-url https://api.friendli.ai/serverless/v1 \
    --vllm-key flp_... \
    --vllm-model meta-llama-3.3-70b-instruct \
    --concurrency 1,4,8 --requests-per-level 8 --max-tokens 128

Warming up engines...
  FriendliAI: OK
  vLLM: OK

============================================================
  Benchmarking: FriendliAI Engine
  Endpoint:     https://api.friendli.ai/serverless/v1
  Model:        meta-llama-3.1-8b-instruct
============================================================
  Concurrency   1: sending 8 requests... done (8 ok, 0 err) | TTFT p50=0.096s | throughput=354.7 tok/s
  Concurrency   4: sending 8 requests... done (8 ok, 0 err) | TTFT p50=0.122s | throughput=316.6 tok/s
  Concurrency   8: sending 8 requests... done (8 ok, 0 err) | TTFT p50=0.162s | throughput=259.9 tok/s

============================================================
  Benchmarking: vLLM
  Endpoint:     https://api.friendli.ai/serverless/v1
  Model:        meta-llama-3.3-70b-instruct
============================================================
  Concurrency   1: sending 8 requests... done (8 ok, 0 err) | TTFT p50=0.114s | throughput=140.5 tok/s
  Concurrency   4: sending 8 requests... done (8 ok, 0 err) | TTFT p50=0.156s | throughput=134.3 tok/s
  Concurrency   8: sending 8 requests... done (8 ok, 0 err) | TTFT p50=0.175s | throughput=125.1 tok/s

Chart saved to benchmark_results.png
```

### Output Chart

![Benchmark Results](benchmark_results.png)

## Metrics

The benchmark measures three metrics across increasing concurrency levels:

**Time to First Token (TTFT)** — Time from sending the request to receiving the first generated token. This is the user-perceived latency — how long they wait before seeing output. Reported as median (p50) and p90. TTFT directly reflects prefill efficiency: how fast the engine processes the input prompt and begins generation.

**Per-Request Output Throughput (tokens/sec)** — Output tokens divided by decode time (first token to last token) for each individual request. This measures how fast a single user sees tokens stream in. It reflects the engine's decode-phase efficiency and is the metric most noticeable to interactive users.

**Aggregate Throughput (total tokens/sec)** — Total output tokens generated across all concurrent requests divided by total decode wall-clock time. This measures the engine's ability to serve multiple users simultaneously — the metric that determines cost-per-token at scale.

## Why These Metrics

- **TTFT** answers: "How responsive is the engine?" — critical for interactive and agentic use cases where latency compounds across multiple turns.
- **Per-request throughput** answers: "How fast does a single user get their response?" — the direct user experience metric.
- **Aggregate throughput** answers: "How efficiently does the engine use hardware under load?" — the metric that drives infrastructure cost decisions.

Together, they reveal whether an engine is fast at low load (TTFT), maintains quality of service under pressure (per-request throughput at high concurrency), and maximizes hardware utilization (aggregate throughput scaling).

## Why This Visualization

The three-panel chart provides an at-a-glance comparison:

1. **Bar chart for TTFT** — grouped bars with p50/p90 markers make absolute differences and tail latency immediately visible across concurrency levels.
2. **Bar chart for per-request throughput** — side-by-side bars show the user-experience gap at each concurrency level.
3. **Line chart for aggregate throughput** — shows scaling behavior: how total system throughput grows (or saturates) as concurrency increases. The slope reveals whether the engine is batching efficiently.

A single image with all three panels lets the viewer draw conclusions without cross-referencing separate charts.

## Fairness Controls

- **Same model**: Both engines should serve the same model architecture and weights.
- **Same prompts**: Identical prompt set in identical order.
- **Same parameters**: `max_tokens`, temperature (default), and system prompt are identical.
- **Warm-up**: One request is sent to each engine before measurement begins to avoid cold-start bias.
- **Streaming**: Both engines are measured via streaming SSE, the mode used in production agentic workflows.
