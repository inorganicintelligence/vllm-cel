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
from vllm.profiler.gpu_efficiency import (
    gpu_monitor_context, 
    calculate_efficiency_metrics,
    get_model_config_from_engine
)

# Import analytical model integration
try:
    from model_integration import (
        extract_vllm_metrics,
        create_analytical_config,
        integrate_analytical_model,
        validate_analytical_results
    )
    ANALYTICAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Analytical model not available: {e}")
    ANALYTICAL_MODEL_AVAILABLE = False


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
    enable_efficiency_monitoring: bool = False,
) -> Tuple[List[RequestOutput], float, dict]:
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
    gpu_metrics = {}
    
    # Start GPU monitoring if enabled
    if enable_efficiency_monitoring:
        monitor_context = gpu_monitor_context(device_id=0, sample_interval=0.1)
        gpu_monitor = monitor_context.__enter__()
    
    start_time = time.perf_counter()

    try:
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
        
    finally:
        # Stop GPU monitoring and collect metrics
        if enable_efficiency_monitoring:
            gpu_metrics = monitor_context.__exit__(None, None, None)
    
    return all_outputs, end_time - start_time, gpu_metrics


def run_vllm(
    requests: List[Tuple[str, int, int]],
    engine_args: EngineArgs,
    batch_size: int,
    num_warmup_runs: int,
    output_format: str,
    enable_efficiency_monitoring: bool = False,
) -> Tuple[List[RequestOutput], float, dict]:
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
    gpu_metrics = {}
    
    # Start GPU monitoring if enabled
    if enable_efficiency_monitoring:
        monitor_context = gpu_monitor_context(device_id=0, sample_interval=0.1)
        gpu_monitor = monitor_context.__enter__()
    
    start_time = time.perf_counter()

    try:
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
        
    finally:
        # Stop GPU monitoring and collect metrics
        if enable_efficiency_monitoring:
            gpu_metrics = monitor_context.__exit__(None, None, None)

    return all_outputs, end_time - start_time, gpu_metrics


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
        outputs, elapsed_time, gpu_metrics = asyncio.run(run_vllm_async(requests, engine_args, args.batch_size, args.num_warmup_runs, args.output_format, args.enable_efficiency_monitoring))
    else:
        engine_args = EngineArgs.from_cli_args(args)
        outputs, elapsed_time, gpu_metrics = run_vllm(requests, engine_args, args.batch_size, args.num_warmup_runs, args.output_format, args.enable_efficiency_monitoring)


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
        
        # Print efficiency metrics if available
        if args.enable_efficiency_monitoring and gpu_metrics:
            print("\n--- GPU Efficiency ---")
            
            gpu_name = gpu_metrics.get("gpu_name", "Unknown")
            print(f"GPU: {gpu_name}")
            
            gpu_util = gpu_metrics.get("gpu_utilization_percent", 0)
            print(f"GPU Utilization: {gpu_util:.1f}%")
            
            mem_util = gpu_metrics.get("memory_utilization_percent", 0)
            print(f"Memory Utilization: {mem_util:.1f}%")
            
            peak_mem = gpu_metrics.get("peak_memory_usage_gb", 0)
            print(f"Peak Memory Usage: {peak_mem:.1f} GB")
            
            # Calculate efficiency metrics for display
            model_config = None
            if not args.async_engine:
                try:
                    temp_engine_args = EngineArgs.from_cli_args(args)
                    temp_llm = LLM(**dataclasses.asdict(temp_engine_args))
                    model_config = get_model_config_from_engine(temp_llm.llm_engine)
                except:
                    pass
            
            efficiency_metrics = calculate_efficiency_metrics(
                system_throughput=system_throughput,
                total_tokens=total_output_tokens + total_prompt_tokens,
                total_time=elapsed_time,
                gpu_metrics=gpu_metrics,
                model_config=model_config
            )
            
            if efficiency_metrics:
                mfu = efficiency_metrics.get("mfu_percent", 0)
                mbu = efficiency_metrics.get("mbu_percent", 0)
                print(f"Model FLOPs Utilization (MFU): {mfu:.1f}%")
                print(f"Memory Bandwidth Utilization (MBU): {mbu:.1f}%")
                
        # Print analytical model results if available
        if args.use_analytical_model and ANALYTICAL_MODEL_AVAILABLE:
            try:
                # Extract metrics from vLLM outputs
                vllm_metrics = extract_vllm_metrics(outputs)
                
                # Create analytical model configuration
                analytical_config = create_analytical_config(
                    batch_size=args.batch_size,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
                    enable_expert_parallel=getattr(args, 'enable_expert_parallel', False)
                )
                
                # Run analytical model
                analytical_results = integrate_analytical_model(
                    vllm_metrics=vllm_metrics,
                    analytical_config=analytical_config,
                    enable_analytical=True
                )
                
                if analytical_results and "error" not in analytical_results:
                    print("\n--- Analytical Model Results ---")
                    print(f"MFU Prefill: {analytical_results.get('mfu_prefill_analytical', 0):.1f}%")
                    print(f"MBU Prefill: {analytical_results.get('mbu_prefill_analytical', 0):.1f}%")
                    print(f"MFU Decode:  {analytical_results.get('mfu_decode_analytical', 0):.1f}%")
                    print(f"MBU Decode:  {analytical_results.get('mbu_decode_analytical', 0):.1f}%")
                    print(f"MFU Total:   {analytical_results.get('mfu_total_analytical', 0):.1f}%")
                    print(f"MBU Total:   {analytical_results.get('mbu_total_analytical', 0):.1f}%")
                    print(f"Padding Overhead: {analytical_results.get('padding_overhead', 1.0):.2f}x")
                    print(f"Effective Token Ratio: {analytical_results.get('effective_token_ratio', 1.0):.1%}")
                    
                    # Print bottleneck analysis
                    prefill_bottleneck = analytical_results.get('prefill_bottleneck', 'unknown')
                    decode_bottleneck = analytical_results.get('decode_bottleneck', 'unknown')
                    print(f"Prefill Bottleneck: {prefill_bottleneck}")
                    print(f"Decode Bottleneck: {decode_bottleneck}")
                    
                    # Print recommendations
                    recommendations = analytical_results.get('optimization_recommendations', [])
                    if recommendations:
                        print("\n--- Optimization Recommendations ---")
                        for i, rec in enumerate(recommendations, 1):
                            print(f"{i}. {rec}")
                            
            except Exception as e:
                print(f"Warning: Analytical model failed: {e}")

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
        
        # Add efficiency metrics if GPU monitoring or analytical model was enabled
        efficiency_metrics = {}
        
        if args.enable_efficiency_monitoring and gpu_metrics:
            # Try to get model configuration for MFU calculation
            model_config = None
            if not args.async_engine:
                # For sync engine, try to extract from the LLM instance
                try:
                    # Create temporary engine to extract config
                    temp_engine_args = EngineArgs.from_cli_args(args)
                    temp_llm = LLM(**dataclasses.asdict(temp_engine_args))
                    model_config = get_model_config_from_engine(temp_llm.llm_engine)
                except Exception as e:
                    if args.output_format == "text":
                        print(f"Warning: Could not extract model config for MFU calculation: {e}")
            
            # Calculate efficiency metrics from GPU monitoring
            gpu_efficiency_metrics = calculate_efficiency_metrics(
                system_throughput=system_throughput,
                total_tokens=total_output_tokens + total_prompt_tokens,
                total_time=elapsed_time,
                gpu_metrics=gpu_metrics,
                model_config=model_config
            )
            
            if gpu_efficiency_metrics:
                efficiency_metrics.update(gpu_efficiency_metrics)
        
        # Add analytical model results if enabled
        if args.use_analytical_model and ANALYTICAL_MODEL_AVAILABLE:
            try:
                # Extract metrics from vLLM outputs
                vllm_metrics = extract_vllm_metrics(outputs)
                
                # Create analytical model configuration
                analytical_config = create_analytical_config(
                    batch_size=args.batch_size,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
                    enable_expert_parallel=getattr(args, 'enable_expert_parallel', False)
                )
                
                # Run analytical model
                analytical_results = integrate_analytical_model(
                    vllm_metrics=vllm_metrics,
                    analytical_config=analytical_config,
                    enable_analytical=True
                )
                
                if analytical_results:
                    efficiency_metrics["analytical"] = analytical_results
                    
                    # Validate against GPU monitoring if both available
                    if args.enable_efficiency_monitoring and gpu_metrics:
                        validation = validate_analytical_results(
                            gpu_monitoring=efficiency_metrics,
                            analytical=analytical_results,
                            tolerance=0.2
                        )
                        efficiency_metrics["validation"] = validation
                        
                if args.output_format == "text":
                    print("Analytical model results integrated successfully")
                    
            except Exception as e:
                if args.output_format == "text":
                    print(f"Warning: Analytical model integration failed: {e}")
                efficiency_metrics["analytical_error"] = str(e)
        
        elif args.use_analytical_model and not ANALYTICAL_MODEL_AVAILABLE:
            if args.output_format == "text":
                print("Warning: Analytical model requested but not available")
            efficiency_metrics["analytical_error"] = "Analytical model not available"
        
        # Add efficiency metrics to results if any were calculated
        if efficiency_metrics:
            results["efficiency"] = efficiency_metrics
        
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
    parser.add_argument("--enable-efficiency-monitoring", action="store_true", help="Enable GPU efficiency monitoring for MFU/MBU metrics.")
    parser.add_argument("--use-analytical-model", action="store_true", help="Use analytical model for MFU/MBU calculation.")
    
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    if args.output_format == 'text':
        print(args)

    if args.dataset is None and args.output_format == 'text':
        print(f"Dataset not specified, generating {args.num_prompts} random prompts...")
    
    main(args) 