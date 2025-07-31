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
from generate_realistic_dataset import SyntheticDatasetGenerator


def get_requests(
    num_requests: int,
    input_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    use_realistic: bool = True,
):
    """Generate requests using the SyntheticDatasetGenerator."""
    generator = SyntheticDatasetGenerator(tokenizer)
    reqs = generator.generate_requests(
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        variety=use_realistic
    )
    generator.save_to_file(reqs, filename = "mixtral_dataset.json")
    return reqs
# TODO: This is a copy of the run_vllm function in compare_async_vs_batch.py.
# We should refactor this to avoid code duplication.
async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    engine_args: AsyncEngineArgs,
    max_concurrent_requests: int,
    num_warmup_runs: int,
    output_format: str,
) -> Tuple[List[RequestOutput], float]:
    """
    Run vLLM async engine with proper continuous batching.
    
    This implementation correctly uses continuous batching where:
    - Requests are submitted continuously without waiting for batch completion
    - The engine dynamically batches requests based on available resources
    - New requests can join ongoing batches as slots become available
    
    Args:
        max_concurrent_requests: Maximum number of concurrent requests in flight
                                (for memory management, not artificial batching)
    """
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Warm-up with continuous submission
    if output_format == "text":
        print(f"Warming up the engine with {num_warmup_runs} request(s)...")
    
    if requests and num_warmup_runs > 0:
        warmup_count = min(num_warmup_runs, len(requests))
        warmup_tasks = []
        
        for i in range(warmup_count):
            request_idx = i % len(requests)
            prompt, _, output_len = requests[request_idx]
            sampling_params = SamplingParams(
                temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len
            )
            
            async def warmup_generate(p, sp, rid):
                async for _ in engine.generate(p, sp, rid):
                    pass  # Consume all outputs
                    
            task = asyncio.create_task(
                warmup_generate(prompt, sampling_params, f"warmup-{i}")
            )
            warmup_tasks.append(task)
        
        await asyncio.gather(*warmup_tasks)
    
    if output_format == "text":
        print("Warm-up complete. Starting continuous batching benchmark...")
    
    # Continuous batching implementation
    all_outputs = []
    completed_requests = 0
    total_requests = len(requests)
    request_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    
    # Fill the request queue
    for i, (prompt, _, output_len) in enumerate(requests):
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len
        )
        await request_queue.put((i, prompt, sampling_params))
    
    start_time = time.perf_counter()
    
    async def request_submitter():
        """Continuously submit requests up to max_concurrent_requests"""
        active_tasks = set()
        request_id = 0
        
        while completed_requests < total_requests or active_tasks:
            # Submit new requests if we have capacity and requests remaining
            while (len(active_tasks) < max_concurrent_requests and 
                   not request_queue.empty()):
                try:
                    req_id, prompt, sampling_params = await asyncio.wait_for(
                        request_queue.get(), timeout=0.1
                    )
                    
                    async def process_request(rid, p, sp):
                        final_output = None
                        async for output in engine.generate(p, sp, f"req-{rid}"):
                            final_output = output
                        # Ensure we only put the final, complete output
                        if final_output and final_output.finished:
                            await result_queue.put(final_output)
                    
                    task = asyncio.create_task(
                        process_request(request_id, prompt, sampling_params)
                    )
                    active_tasks.add(task)
                    request_id += 1
                    
                except asyncio.TimeoutError:
                    break
            
            # Check for completed tasks
            if active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1
                )
                for task in done:
                    try:
                        await task  # This will raise any exceptions
                    except Exception as e:
                        print(f"Request failed: {e}")
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.001)
    
    async def result_collector():
        """Collect results as they become available"""
        nonlocal completed_requests
        pbar = tqdm(total=total_requests, desc="Processing requests") if output_format == "text" else None
        
        while completed_requests < total_requests:
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
                all_outputs.append(result)
                completed_requests += 1
                if pbar:
                    pbar.update(1)
            except asyncio.TimeoutError:
                continue
        
        if pbar:
            pbar.close()
    
    # Run both submitter and collector concurrently
    await asyncio.gather(
        request_submitter(),
        result_collector()
    )

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
            data = json.load(f)
            # Handle both old format [prompt, input_len, output_len] and new format {"prompt": ..., "input_length": ..., "output_length": ...}
            if data and isinstance(data[0], dict):
                # New format from generate_realistic_dataset.py
                requests = [(item['prompt'], item['input_length'], item['output_length']) for item in data]
            else:
                # Old format
                requests = [tuple(req) for req in data]
    else:
        requests = get_requests(
            args.num_prompts, 
            args.input_len, 
            args.output_len, 
            tokenizer,
            use_realistic=not args.simple_prompts
        )
    
    if args.async_engine:
        engine_args = AsyncEngineArgs.from_cli_args(args)
        # Use continuous batching with max concurrent requests
        outputs, elapsed_time = asyncio.run(run_vllm_async(requests, engine_args, args.max_concurrent_requests, args.num_warmup_runs, args.output_format))
    else:
        engine_args = EngineArgs.from_cli_args(args)
        # Use traditional batch processing
        outputs, elapsed_time = run_vllm(requests, engine_args, args.batch_size, args.num_warmup_runs, args.output_format)


    # Collect metrics
    total_output_tokens = 0
    total_prompt_tokens = 0
    ttfts = []
    tpots = []
    itls = []
    prefill_times = []
    decode_times = []
    inference_times = []
    queued_times = []
    cache_hit_rates = []
    total_cached_tokens = 0

    for i, output in enumerate(outputs):
        total_prompt_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)

        if not output.metrics:
            continue

        # Handle different metric field names between v0 and v1 engine
        is_v0_metrics = hasattr(output.metrics, 'first_token_time')

        if is_v0_metrics:
            print("***********   Using V0 engine ***********")
        else:
            print("***********   Using V1 engine ***********")
        
        if is_v0_metrics:
            first_token = output.metrics.first_token_time
            last_token = output.metrics.last_token_time
            arrival = output.metrics.arrival_time

            print("FTT", end=" : ")
            print(first_token)
            print("LTT", end=" : ")
            print(last_token)
            print("Arrival time", end=" : ")
            print(arrival)

        else:
            first_token = output.metrics.first_token_ts
            last_token = output.metrics.last_token_ts
            arrival = output.metrics.queued_ts

            print("FTT", end=" : ")
            print(first_token)
            print("LTT", end=" : ")
            print(last_token)
            print("Arrival time", end=" : ")
            print(arrival)

        if (not first_token or not last_token or not arrival
                or not output.metrics.token_timestamps):
            continue

        ttft = first_token - arrival
        ttfts.append(ttft)
        
        if output.metrics.prefill_time:
            prefill_times.append(output.metrics.prefill_time)
        
        if output.metrics.decode_time:
            decode_times.append(output.metrics.decode_time)

        if output.metrics.inference_time:
            inference_times.append(output.metrics.inference_time)

        if output.metrics.queued_time:
            queued_times.append(output.metrics.queued_time)

        if len(output.outputs[0].token_ids) > 1:
            tpot = (last_token - first_token) / (len(output.outputs[0].token_ids) - 1)
            tpots.append(tpot)

        if len(output.metrics.token_timestamps) > 1:
            itl = np.diff(output.metrics.token_timestamps).tolist()
            itls.extend(itl)

        # Collect cache metrics
        if hasattr(output, 'num_cached_tokens'):
            cached_tokens = output.num_cached_tokens
            total_cached_tokens += cached_tokens
            if cached_tokens > 0:
              cache_hit_rate = cached_tokens / len(output.prompt_token_ids)
              cache_hit_rates.append(cache_hit_rate)

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

        if inference_times:
            print("\nAverage Inference Time:")
            print(f"  mean: {np.mean(inference_times):.4f} s")

        if queued_times:
            print("\nAverage Queued Time:")
            print(f"  mean: {np.mean(queued_times):.4f} s")

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

        if cache_hit_rates:
            print("\nCache Hit Rate:")
            print(f"  mean: {np.mean(cache_hit_rates):.4f}")
            print(f"  median: {np.median(cache_hit_rates):.4f}")
            print(f"  p99: {np.percentile(cache_hit_rates, 99):.4f}")

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
                "avg_inference_time": np.mean(inference_times) if inference_times else None,
                "avg_queued_time": np.mean(queued_times) if queued_times else None,
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
                },
            },
            "cache_hit_rate": {
                    "mean": np.mean(cache_hit_rates) if cache_hit_rates else None,
                    "median": np.median(cache_hit_rates) if cache_hit_rates else None,
                    "p99": np.percentile(cache_hit_rates, 99) if cache_hit_rates else None,
                }     
        }
        print(json.dumps(results))

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the performance of vLLM.")
    
    # Add AsyncEngineArgs (which includes EngineArgs)
    parser = AsyncEngineArgs.add_cli_args(parser)
    
    # Benchmark-specific arguments
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="For offline mode: batch size. For async mode: max concurrent requests for continuous batching.")
    parser.add_argument("--max-concurrent-requests", type=int, default=None,
                       help="Maximum concurrent requests for async engine (overrides batch-size for async mode)")
    parser.add_argument("--input-len", type=int, default=32, help="Input sequence length for synthetic data.")
    parser.add_argument("--output-len", type=int, default=128, help="Output sequence length for synthetic data.")
    parser.add_argument("--num-prompts", type=int, default=256, help="Number of prompts to process for synthetic data.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON file with prompts. Each entry should be a list of [prompt, input_len, output_len].")
    parser.add_argument("--async-engine", action="store_true", 
                       help="Use async engine with continuous batching (vs offline batch processing).")
    parser.add_argument("--num-warmup-runs", type=int, default=1, help="Number of warm-up requests to run before benchmarking.")
    parser.add_argument("--output-format", type=str, default="text", choices=["text", "json"], help="Output format.")
    parser.add_argument("--simple-prompts", action="store_true", help="Use simple word repetition prompts instead of realistic synthetic prompts.")
    
    # Scheduler control arguments (these are already part of EngineArgs but we can document them)
    # Note: These are automatically added by EngineArgs.add_cli_args() but we can add descriptions
    parser.add_argument("--help-scheduler", action="store_true",
                       help="Show scheduler-related parameters: --max-num-batched-tokens, --max-num-seqs, --scheduler-delay-factor")
    
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    # Set max_concurrent_requests for async mode if not specified
    if args.max_concurrent_requests is None:
        args.max_concurrent_requests = args.batch_size
    
    if args.output_format == 'text':
        print(args)

    if args.output_format == 'text':
        mode_desc = "async engine with continuous batching" if args.async_engine else "offline batch processing"
        print(f"Running benchmark using {mode_desc}")
        if args.async_engine:
            print(f"Max concurrent requests: {args.max_concurrent_requests}")
        else:
            print(f"Batch size: {args.batch_size}")
        
        if args.dataset is None:
            prompt_type = "simple" if args.simple_prompts else "realistic synthetic"
            print(f"Dataset not specified, generating {args.num_prompts} {prompt_type} prompts...")
    
    main(args) 