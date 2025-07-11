import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional, Tuple
import dataclasses

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from vllm import LLM, SamplingParams, RequestOutput, AsyncLLMEngine
from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser


def get_requests(
    num_requests: int,
    input_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
):
    # This is a simplified request generator.
    # You can replace this with a more sophisticated one based on your needs.
    requests = []
    for _ in range(num_requests):
        prompt = " ".join(["word"] * input_len)
        requests.append(
            (prompt, input_len, output_len)
        )
    return requests

async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    engine_args: AsyncEngineArgs,
    batch_size: int,
    num_warmup_runs: int,
    output_format: str,
) -> Tuple[List[RequestOutput], float]:
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Warm-up
    if output_format == "text":
        print(f"Warming up the engine with {num_warmup_runs} batch(es)...")
    if requests:
        for i in range(num_warmup_runs):
            # Take a slice for the warmup batch. If we run out of requests, just reuse from the start.
            start_idx = (i * batch_size) % len(requests)
            end_idx = start_idx + batch_size
            warmup_requests = requests[start_idx:end_idx]
            if not warmup_requests:
                break
                
            prompts = [prompt for prompt, _, _ in warmup_requests]
            sampling_params = [
                SamplingParams(
                    temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len
                )
                for _, _, output_len in warmup_requests
            ]
            
            async def warmup_generate(prompt, sp, request_id):
                await engine.generate(prompt, sp, request_id)

            tasks = [
                asyncio.create_task(warmup_generate(prompts[j], sampling_params[j], f"warmup-{i}-{j}"))
                for j in range(len(prompts))
            ]
            await asyncio.gather(*tasks)
    if output_format == "text":
        print("Warm-up complete.")
    
    all_outputs = []
    start_time = time.perf_counter()

    for i in tqdm(range(0, len(requests), batch_size), desc="Processing requests"):
        batch = requests[i:i+batch_size]
        prompts = [prompt for prompt, _, _ in batch]
        sampling_params = [
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
            ) for _, _, output_len in batch
        ]

        async def generate(prompt, sp, request_id):
            return await engine.generate(prompt, sp, request_id)

        tasks = [
            asyncio.create_task(generate(prompts[j], sampling_params[j], str(i+j)))
            for j in range(len(prompts))
        ]
        batch_outputs = await asyncio.gather(*tasks)
        all_outputs.extend(batch_outputs)

    end_time = time.perf_counter()
    return all_outputs, end_time - start_time


def run_vllm(
    requests: List[Tuple[str, int, int]],
    engine_args: EngineArgs,
    batch_size: int,
    num_warmup_runs: int,
    output_format: str,
) -> Tuple[List[RequestOutput], float]:
    llm = LLM(**dataclasses.asdict(engine_args))
    
    # Warm-up run
    if output_format == "text":
        print(f"Warming up the engine with {num_warmup_runs} batch(es)...")
    if requests:
        for i in range(num_warmup_runs):
            # Take a slice for the warmup batch. If we run out of requests, just reuse from the start.
            start_idx = (i * batch_size) % len(requests)
            end_idx = start_idx + batch_size
            warmup_requests = requests[start_idx:end_idx]
            if not warmup_requests:
                break

            prompts = [prompt for prompt, _, _ in warmup_requests]
            sampling_params = [
                SamplingParams(
                    temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len
                )
                for _, _, output_len in warmup_requests
            ]
            llm.generate(prompts, sampling_params)
    if output_format == "text":
        print("Warm-up complete.")

    all_outputs = []
    start_time = time.perf_counter()

    # Main benchmark run
    main_requests = requests
    for i in tqdm(range(0, len(main_requests), batch_size), desc="Processing requests"):
        batch = main_requests[i:i+batch_size]
        prompts = [prompt for prompt, _, _ in batch]
        sampling_params = [
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
            ) for _, _, output_len in batch
        ]

        # The llm.generate call here is implicitly batching these.
        # This loop is more for logical batching if we were to use a different backend
        # or wanted to control the batching more explicitly. For vLLM, it handles
        # the batching internally. The loop is kept for conceptual clarity and
        # future extensions.
        outputs = llm.generate(prompts, sampling_params)
        all_outputs.extend(outputs)

    end_time = time.perf_counter()
    return all_outputs, end_time - start_time


def main(args: argparse.Namespace):
    random.seed(args.seed)

    # Generate requests
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    if args.dataset:
        with open(args.dataset) as f:
            # This assumes a list of [prompt, input_len, output_len]
            requests = [tuple(req) for req in json.load(f)]
    else:
        requests = get_requests(args.num_prompts, args.input_len, args.output_len, tokenizer)
    
    if args.async_engine:
        engine_args = AsyncEngineArgs.from_cli_args(args)
        outputs, elapsed_time = asyncio.run(run_vllm_async(requests, engine_args, args.batch_size, args.num_warmup_runs, args.output_format))
    else:
        engine_args = EngineArgs.from_cli_args(args)
        outputs, elapsed_time = run_vllm(requests, engine_args, args.batch_size, args.num_warmup_runs, args.output_format)


    # Collect metrics
    total_output_tokens = 0
    total_prompt_tokens = 0
    ttfts = []
    tpots = []
    itls = []
    prefill_times = []
    decode_times = []

    for i, output in enumerate(outputs):
        total_prompt_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)

        if not output.metrics:
            continue

        # Handle different metric field names between v0 and v1 engine
        is_v0_metrics = hasattr(output.metrics, 'first_token_time')
        
        if is_v0_metrics:
            first_token = output.metrics.first_token_time
            last_token = output.metrics.last_token_time
            arrival = output.metrics.arrival_time
        else:
            first_token = output.metrics.first_token_ts
            last_token = output.metrics.last_token_ts
            arrival = output.metrics.queued_ts

        if (not first_token or not last_token or not arrival
                or not output.metrics.token_timestamps):
            continue

        

        ttft = first_token - arrival
        ttfts.append(ttft)
        
        if output.metrics.prefill_time:
            prefill_times.append(output.metrics.prefill_time)
        
        if output.metrics.decode_time:
            decode_times.append(output.metrics.decode_time)

        if len(output.outputs[0].token_ids) > 1:
            tpot = (last_token - first_token) / (len(output.outputs[0].token_ids) - 1)
            tpots.append(tpot)

        if len(output.metrics.token_timestamps) > 1:
            itl = np.diff(output.metrics.token_timestamps).tolist()
            itls.extend(itl)
    
    # Calculate and print metrics
    request_throughput = len(requests) / elapsed_time
    system_throughput = (total_prompt_tokens + total_output_tokens) / elapsed_time

    if args.output_format == "text":
        print(f"\nTotal time: {elapsed_time:.2f} s")

        # Throughput
        print("\n--- Throughput ---")
        print(f"Request throughput: {request_throughput:.2f} requests/s")
        print(f"System throughput: {system_throughput:.2f} tokens/s")
        
        # Latency
        print("\n--- Latency ---")
        if prefill_times:
            print("\nAverage Prefill Time:")
            print(f"  mean: {np.mean(prefill_times):.4f} s")

        if decode_times:
            print("\nAverage Decode Time:")
            print(f"  mean: {np.mean(decode_times):.4f} s")

        if ttfts:
            print("\nTime to First Token (TTFT):")
            print(f"  mean: {np.mean(ttfts):.4f} s")

        if tpots:
            print("\nTime Per Output Token (TPOT):")
            print(f"  mean: {np.mean(tpots):.4f} s/token")

        if itls:
            print("\nInter-Token Latency (ITL):")
            print(f"  mean: {np.mean(itls):.4f} s")
            print(f"  median: {np.median(itls):.4f} s")
            print(f"  p99: {np.percentile(itls, 99):.4f} s")

    if args.output_format == 'json':
        results = {
            "total_time": elapsed_time,
            "throughput": {
                "request_throughput": request_throughput,
                "system_throughput": system_throughput,
            },
            "latency": {
                "avg_prefill_time": np.mean(prefill_times) if prefill_times else None,
                "avg_decode_time": np.mean(decode_times) if decode_times else None,
                "ttft": {
                    "mean": np.mean(ttfts) if ttfts else None,
                },
                "tpot": {
                    "mean": np.mean(tpots) if tpots else None,
                },
                "itl": {
                    "mean": np.mean(itls) if itls else None,
                    "median": np.median(itls) if itls else None,
                    "p99": np.percentile(itls, 99) if itls else None,
                }
            }
        }
        print(json.dumps(results))

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the performance of vLLM.")
    
    # Add EngineArgs
    parser = EngineArgs.add_cli_args(parser)
    
    # Benchmark-specific arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the benchmark.")
    parser.add_argument("--input-len", type=int, default=32, help="Input sequence length for synthetic data.")
    parser.add_argument("--output-len", type=int, default=128, help="Output sequence length for synthetic data.")
    parser.add_argument("--num-prompts", type=int, default=256, help="Number of prompts to process for synthetic data.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON file with prompts. Each entry should be a list of [prompt, input_len, output_len].")
    parser.add_argument("--async-engine", action="store_true", help="Use async engine.")
    parser.add_argument("--num-warmup-runs", type=int, default=1, help="Number of warm-up batches to run before benchmarking.")
    parser.add_argument("--output-format", type=str, default="text", choices=["text", "json"], help="Output format.")
    
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    if args.output_format == 'text':
        print(args)

    if args.dataset is None and args.output_format == 'text':
        print(f"Dataset not specified, generating {args.num_prompts} random prompts...")
    
    main(args) 