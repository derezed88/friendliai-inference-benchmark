#!/usr/bin/env python3
"""
Inference Engine Benchmark: FriendliAI vs vLLM

Measures Time to First Token (TTFT), output throughput, and end-to-end latency
across increasing concurrency levels. Generates a comparison chart.

Usage:
    python benchmark.py --friendli-url URL --friendli-key KEY --friendli-model MODEL \
                        --vllm-url URL [--vllm-key KEY] --vllm-model MODEL
"""

import argparse
import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field

import aiohttp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Prompts — varied lengths and tasks for realistic workload
# ---------------------------------------------------------------------------
PROMPTS = [
    {"role": "user", "content": "Write a Python function that checks if a string is a palindrome. Include docstring and examples."},
    {"role": "user", "content": "Explain the difference between a stack and a queue. Give a real-world analogy for each."},
    {"role": "user", "content": "Write a SQL query to find the top 5 customers by total order value from tables 'customers' and 'orders'."},
    {"role": "user", "content": "What are the SOLID principles in software engineering? Briefly explain each with one sentence."},
    {"role": "user", "content": "Write a bash one-liner that finds all files larger than 100MB in the current directory tree."},
    {"role": "user", "content": "Explain how a hash table works, including how collisions are handled."},
    {"role": "user", "content": "Write a Python async function that fetches 10 URLs concurrently using aiohttp and returns results as a list."},
    {"role": "user", "content": "Compare TCP and UDP. When would you choose one over the other?"},
]

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = 16  # total requests fired at each concurrency level
MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    ttft: float              # seconds to first token
    e2e_latency: float       # total seconds
    output_tokens: int       # tokens generated
    throughput: float         # output tokens / decode time
    success: bool = True
    error: str = ""


@dataclass
class LevelResult:
    concurrency: int
    results: list = field(default_factory=list)

    @property
    def successful(self):
        return [r for r in self.results if r.success]

    def median_ttft(self):
        vals = [r.ttft for r in self.successful]
        return statistics.median(vals) if vals else float("nan")

    def p90_ttft(self):
        vals = sorted(r.ttft for r in self.successful)
        if not vals:
            return float("nan")
        idx = int(len(vals) * 0.9)
        return vals[min(idx, len(vals) - 1)]

    def median_throughput(self):
        vals = [r.throughput for r in self.successful]
        return statistics.median(vals) if vals else float("nan")

    def aggregate_throughput(self):
        """Total tokens generated / total wall-clock decode time across all requests."""
        total_tokens = sum(r.output_tokens for r in self.successful)
        total_decode = sum(r.e2e_latency - r.ttft for r in self.successful)
        return total_tokens / total_decode if total_decode > 0 else 0


# ---------------------------------------------------------------------------
# Core benchmark: stream a single request, measure timing
# ---------------------------------------------------------------------------
async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    model: str,
    prompt: dict,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, prompt],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_token = None
    output_tokens = 0

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(0, 0, 0, 0, success=False, error=f"HTTP {resp.status}: {body[:200]}")

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check for first content token
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if t_first_token is None:
                        t_first_token = time.perf_counter()

                # Capture usage from final chunk
                usage = chunk.get("usage")
                if usage and "completion_tokens" in usage:
                    output_tokens = usage["completion_tokens"]

    except Exception as e:
        return RequestResult(0, 0, 0, 0, success=False, error=str(e)[:200])

    t_end = time.perf_counter()

    if t_first_token is None:
        t_first_token = t_end  # no tokens received

    ttft = t_first_token - t_start
    e2e = t_end - t_start
    decode_time = t_end - t_first_token
    throughput = output_tokens / decode_time if decode_time > 0 else 0

    return RequestResult(
        ttft=ttft,
        e2e_latency=e2e,
        output_tokens=output_tokens,
        throughput=throughput,
    )


# ---------------------------------------------------------------------------
# Run one concurrency level
# ---------------------------------------------------------------------------
async def run_level(
    url: str,
    headers: dict,
    model: str,
    concurrency: int,
    num_requests: int,
) -> LevelResult:
    sem = asyncio.Semaphore(concurrency)
    level = LevelResult(concurrency=concurrency)

    async def bounded_request(prompt):
        async with sem:
            return await stream_request(session, url, headers, model, prompt)

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
        tasks = [bounded_request(p) for p in prompts]
        level.results = await asyncio.gather(*tasks)

    return level


# ---------------------------------------------------------------------------
# Run full benchmark for one engine
# ---------------------------------------------------------------------------
async def benchmark_engine(name: str, base_url: str, api_key: str, model: str):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"  Endpoint:     {base_url}")
    print(f"  Model:        {model}")
    print(f"{'='*60}")

    levels = []
    for c in CONCURRENCY_LEVELS:
        print(f"  Concurrency {c:>3d}: sending {REQUESTS_PER_LEVEL} requests...", end="", flush=True)
        level = await run_level(url, headers, model, c, REQUESTS_PER_LEVEL)
        ok = len(level.successful)
        fail = len(level.results) - ok
        print(f" done ({ok} ok, {fail} err) "
              f"| TTFT p50={level.median_ttft():.3f}s "
              f"| throughput={level.median_throughput():.1f} tok/s")
        levels.append(level)

    return levels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_chart(friendli_levels, vllm_levels, output_path="benchmark_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Inference Engine Benchmark: FriendliAI vs vLLM", fontsize=14, fontweight="bold")

    concurrencies = CONCURRENCY_LEVELS

    def extract(levels, fn):
        return [fn(l) for l in levels]

    # --- Panel 1: TTFT (median + p90) ---
    ax = axes[0]
    f_ttft_med = extract(friendli_levels, lambda l: l.median_ttft() * 1000)
    v_ttft_med = extract(vllm_levels, lambda l: l.median_ttft() * 1000)
    f_ttft_p90 = extract(friendli_levels, lambda l: l.p90_ttft() * 1000)
    v_ttft_p90 = extract(vllm_levels, lambda l: l.p90_ttft() * 1000)

    x = np.arange(len(concurrencies))
    w = 0.35
    bars1 = ax.bar(x - w/2, f_ttft_med, w, label="FriendliAI (p50)", color="#2563eb")
    bars2 = ax.bar(x + w/2, v_ttft_med, w, label="vLLM (p50)", color="#dc2626")
    # p90 markers
    ax.scatter(x - w/2, f_ttft_p90, marker="_", color="#1e40af", s=200, linewidths=2, zorder=5, label="FriendliAI (p90)")
    ax.scatter(x + w/2, v_ttft_p90, marker="_", color="#991b1b", s=200, linewidths=2, zorder=5, label="vLLM (p90)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Time to First Token (ms)")
    ax.set_title("TTFT — Lower is Better")
    ax.set_xticks(x)
    ax.set_xticklabels(concurrencies)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: Per-request throughput (median tok/s) ---
    ax = axes[1]
    f_thr = extract(friendli_levels, lambda l: l.median_throughput())
    v_thr = extract(vllm_levels, lambda l: l.median_throughput())

    ax.bar(x - w/2, f_thr, w, label="FriendliAI", color="#2563eb")
    ax.bar(x + w/2, v_thr, w, label="vLLM", color="#dc2626")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output Tokens / sec")
    ax.set_title("Per-Request Throughput — Higher is Better")
    ax.set_xticks(x)
    ax.set_xticklabels(concurrencies)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: Aggregate throughput (total tok/s across all concurrent requests) ---
    ax = axes[2]
    f_agg = extract(friendli_levels, lambda l: l.aggregate_throughput())
    v_agg = extract(vllm_levels, lambda l: l.aggregate_throughput())

    ax.plot(concurrencies, f_agg, "o-", color="#2563eb", linewidth=2, markersize=8, label="FriendliAI")
    ax.plot(concurrencies, v_agg, "s-", color="#dc2626", linewidth=2, markersize=8, label="vLLM")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Total Tokens / sec")
    ax.set_title("Aggregate Throughput — Higher is Better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark FriendliAI Engine vs vLLM inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FriendliAI Serverless vs local vLLM
  python benchmark.py \\
      --friendli-url https://api.friendli.ai/serverless/v1 \\
      --friendli-key flp_... \\
      --friendli-model meta-llama/Llama-3.3-70B-Instruct \\
      --vllm-url http://localhost:8000/v1 \\
      --vllm-model meta-llama/Llama-3.3-70B-Instruct

  # Adjust workload
  python benchmark.py ... --concurrency 1,4,8 --requests-per-level 8 --max-tokens 128
        """,
    )
    p.add_argument("--friendli-url", required=True, help="FriendliAI base URL (e.g. https://api.friendli.ai/serverless/v1)")
    p.add_argument("--friendli-key", required=True, help="FriendliAI API token")
    p.add_argument("--friendli-model", required=True, help="Model ID on FriendliAI")
    p.add_argument("--vllm-url", required=True, help="vLLM base URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--vllm-key", default="", help="vLLM API key (if required)")
    p.add_argument("--vllm-model", required=True, help="Model ID on vLLM")
    p.add_argument("--concurrency", default=",".join(str(c) for c in CONCURRENCY_LEVELS),
                    help=f"Comma-separated concurrency levels (default: {','.join(str(c) for c in CONCURRENCY_LEVELS)})")
    p.add_argument("--requests-per-level", type=int, default=REQUESTS_PER_LEVEL,
                    help=f"Requests per concurrency level (default: {REQUESTS_PER_LEVEL})")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help=f"Max output tokens per request (default: {MAX_TOKENS})")
    p.add_argument("--output", default="benchmark_results.png", help="Output chart filename")
    return p.parse_args()


async def main():
    args = parse_args()

    global CONCURRENCY_LEVELS, REQUESTS_PER_LEVEL, MAX_TOKENS
    CONCURRENCY_LEVELS = [int(x) for x in args.concurrency.split(",")]
    REQUESTS_PER_LEVEL = args.requests_per_level
    MAX_TOKENS = args.max_tokens

    # Warm up: send one request to each engine to avoid cold-start skew
    print("Warming up engines...")
    for name, url, key, model in [
        ("FriendliAI", args.friendli_url, args.friendli_key, args.friendli_model),
        ("vLLM", args.vllm_url, args.vllm_key, args.vllm_model),
    ]:
        warmup = await run_level(
            f"{url.rstrip('/')}/chat/completions",
            {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
            model, concurrency=1, num_requests=1,
        )
        ok = len(warmup.successful)
        print(f"  {name}: {'OK' if ok else 'FAILED — ' + warmup.results[0].error if warmup.results else 'no response'}")

    friendli_levels = await benchmark_engine(
        "FriendliAI Engine",
        args.friendli_url,
        args.friendli_key,
        args.friendli_model,
    )

    vllm_levels = await benchmark_engine(
        "vLLM",
        args.vllm_url,
        args.vllm_key,
        args.vllm_model,
    )

    generate_chart(friendli_levels, vllm_levels, args.output)


if __name__ == "__main__":
    asyncio.run(main())
