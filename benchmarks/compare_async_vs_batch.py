#!/usr/bin/env python3
"""
Compare performance between async continuous batching and offline batch processing.

This script runs benchmark_performance.py with both modes and compares key metrics:
- Throughput (requests/second)
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- GPU utilization efficiency
"""

import subprocess
import json
import sys
from typing import Dict, Any
import argparse


def run_benchmark(mode: str, extra_args: list[str]) -> Dict[str, Any]:
    """Run benchmark_performance.py and capture the JSON output."""
    base_cmd = [
        sys.executable,
        "benchmark_performance.py",
        "--model", "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "--tensor-parallel-size", "2",
        "--enable-expert-parallel",
        "--num-prompts", "512",
        "--input-len", "1024", 
        "--output-len", "512",
        "--output-format", "json",
        "--dataset", "mixtral_dataset.json",
    ]
    
    if mode == "async":
        # Async mode with continuous batching
        cmd = base_cmd + [
            "--async-engine",
            "--max-concurrent-requests", "256",  # High concurrency for async
            "--max-num-batched-tokens", "16384",  # Large batch for GPU efficiency
            "--max-num-seqs", "256",
        ] + extra_args
    else:
        # Offline batch mode
        cmd = base_cmd + [
             "--batch-size", "64",  # Traditional batch processing
#            "--max-num-batched-tokens", "16384",
#            "--max-num-seqs", "256",
        ] + extra_args
    
    print(f"\nRunning {mode} mode benchmark...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON output
        output_lines = result.stdout.strip().split('\n')
        # Find the JSON line (last line should be JSON)
        for line in reversed(output_lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise ValueError("No valid JSON output found")
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def compare_results(async_results: Dict[str, Any], batch_results: Dict[str, Any]):
    """Compare and display the results."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: Async vs Batch")
    print("="*80)
    
    # Define metrics to compare
    metrics = [
        ("Total Time (s)", ["total_time"]),
        ("System Throughput (tokens/s)", ["throughput", "system_throughput"]),
        ("Avg Prefill Time (s)", ["latency", "avg_prefill_time"]),
        ("Avg Decode Time (s)", ["latency", "avg_decode_time"]),
        ("Avg Inference Time (s)", ["latency", "avg_inference_time"]),
        ("Avg Queued Time (s)", ["latency", "avg_queued_time"]),
        ("Mean TTFT (s)", ["latency", "ttft", "mean"]),
        ("Mean TPOT (s)", ["latency", "tpot", "mean"]),
        ("Mean ITL (s)", ["latency", "itl", "mean"]),
        ("P99 ITL (s)", ["latency", "itl", "p99"]),
    ]
    
    print(f"\n{'Metric':<30} {'Async':<15} {'Batch':<15} {'Difference':<15}")
    print("-" * 80)
    
    for metric_name, path in metrics:
        # Get values from nested dict
        async_val = async_results
        batch_val = batch_results
        
        for key in path:
            async_val = async_val.get(key, {}) if isinstance(async_val, dict) else 0
            batch_val = batch_val.get(key, {}) if isinstance(batch_val, dict) else 0
        
        # Convert to float if needed
        async_val = float(async_val) if async_val else 0
        batch_val = float(batch_val) if batch_val else 0
        
        # Calculate percentage difference
        if batch_val > 0:
            diff_pct = ((async_val - batch_val) / batch_val) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"
        
        print(f"{metric_name:<30} {async_val:<15.4f} {batch_val:<15.4f} {diff_str:<15}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare async vs batch performance")
    parser.add_argument("--scheduler-delay-factor", type=float, default=0.0,
                       help="Scheduler delay factor for async mode (default: 0.0)")
    #parser.add_argument("--num-warmup-runs", type=int, default=1,
    #                   help="Number of warmup runs (default: 1)")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                       choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
                       help="KV cache data type (default: auto)")
    
    args = parser.parse_args()
    
    # Build extra arguments
    extra_args = []
    if args.scheduler_delay_factor > 0:
        extra_args.extend(["--scheduler-delay-factor", str(args.scheduler_delay_factor)])
    #if args.num_warmup_runs != 3:
    #    extra_args.extend(["--num-warmup-runs", str(args.num_warmup_runs)])
    if args.kv_cache_dtype != "auto":
        extra_args.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    
    print("Starting performance comparison: Async Continuous Batching vs Offline Batch Mode")
    print(f"Additional arguments: {extra_args}")
    
    # Run benchmarks
    async_results = run_benchmark("async", extra_args)
    print("$$$$$$$$$   Printing Async Results $$$$$$$$$$")
    print("async results")
    print(async_results)

    batch_results = run_benchmark("batch", extra_args)
    print("$$$$$$$$$   Printing Batch Results $$$$$$$$$$")
    print("batch results")
    print(batch_results)

    # Compare and display results
    compare_results(async_results, batch_results)


if __name__ == "__main__":
    main()