#!/usr/bin/env python3
"""
model_integration.py

Integration utilities for connecting vLLM benchmark results with the analytical model.
Handles metric extraction, configuration mapping, and result merging.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from vllm import RequestOutput

try:
    from .utilization_analytical_model import UtilizationModel
except ImportError:
    # Handle relative import when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utilization_analytical_model import UtilizationModel

logger = logging.getLogger(__name__)


def extract_vllm_metrics(outputs: List[RequestOutput]) -> Dict[str, float]:
    """Extract timing metrics from vLLM RequestOutput objects.
    
    Args:
        outputs: List of RequestOutput from vLLM benchmark
        
    Returns:
        Dictionary with extracted timing metrics
    """
    prefill_times = []
    decode_times = []
    total_prefill_tokens = 0
    total_decode_tokens = 0
    
    for output in outputs:
        if not output.metrics:
            continue
            
        # Extract timing metrics (handle both v0 and v1 formats)
        if hasattr(output.metrics, 'prefill_time') and output.metrics.prefill_time:
            prefill_times.append(output.metrics.prefill_time)
            
        if hasattr(output.metrics, 'decode_time') and output.metrics.decode_time:
            decode_times.append(output.metrics.decode_time)
            
        # Count tokens
        total_prefill_tokens += len(output.prompt_token_ids)
        total_decode_tokens += len(output.outputs[0].token_ids) if output.outputs else 0
    
    # Calculate averages
    avg_prefill_time = np.mean(prefill_times) if prefill_times else 0.0
    avg_decode_time = np.mean(decode_times) if decode_times else 0.0
    
    return {
        'prefill_time': avg_prefill_time,
        'decode_time': avg_decode_time,
        'total_prefill_tokens': total_prefill_tokens,
        'total_decode_tokens': total_decode_tokens,
        'num_requests': len(outputs)
    }


def generate_sequence_distribution(
    target_length: int, 
    batch_size: int, 
    variance: float = 0.2
) -> List[int]:
    """Generate realistic sequence length distribution for analytical model.
    
    Args:
        target_length: Target sequence length (will be max after padding)
        batch_size: Number of sequences in batch
        variance: Relative variance in sequence lengths (0.0-1.0)
        
    Returns:
        List of actual sequence lengths before padding
    """
    # Generate lengths with normal distribution
    std_dev = target_length * variance
    lengths = np.random.normal(target_length * 0.85, std_dev, batch_size)
    
    # Ensure all lengths are valid (between 10% and 100% of target)
    min_length = max(1, int(target_length * 0.1))
    max_length = target_length
    
    lengths = np.clip(lengths, min_length, max_length).astype(int)
    
    return lengths.tolist()


def create_analytical_config(
    batch_size: int,
    input_len: int,
    output_len: int,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False
) -> Dict[str, Any]:
    """Create configuration dictionary for analytical model.
    
    Args:
        batch_size: Batch size for benchmark
        input_len: Input sequence length (prefill)
        output_len: Output sequence length (decode tokens to generate)
        tensor_parallel_size: Number of tensor parallel devices
        enable_expert_parallel: Whether expert parallelism is enabled
        
    Returns:
        Configuration dictionary for analytical model
    """
    # Generate realistic sequence length distribution
    actual_seq_lengths = generate_sequence_distribution(
        target_length=input_len,
        batch_size=batch_size,
        variance=0.2
    )
    
    return {
        'batch_size': batch_size,
        'actual_seq_lengths': actual_seq_lengths,
        'max_seq_length': input_len,  # Padded length
        'num_decode_tokens': output_len,
        'tensor_parallel_size': tensor_parallel_size,
        'enable_expert_parallel': enable_expert_parallel
    }


def detect_bottleneck(mfu: float, mbu: float, phase: str) -> str:
    """Detect performance bottleneck based on MFU/MBU metrics.
    
    Args:
        mfu: Model FLOPs Utilization (0.0-1.0)
        mbu: Memory Bandwidth Utilization (0.0-1.0)
        phase: 'prefill' or 'decode'
        
    Returns:
        Bottleneck description string
    """
    if phase == 'prefill':
        # Prefill should be compute-bound (high MFU, moderate MBU)
        if mfu > 0.7:
            return "compute_optimal"
        elif mfu > 0.4:
            return "compute_moderate"
        elif mbu > 0.8:
            return "memory_bound"
        else:
            return "underutilized"
    
    elif phase == 'decode':
        # Decode should be memory-bound (high MBU, low MFU)
        if mbu > 0.8:
            return "memory_optimal"
        elif mbu > 0.5:
            return "memory_moderate"
        elif mfu > 0.5:
            return "compute_bound"
        else:
            return "underutilized"
    
    return "unknown"


def generate_optimization_recommendations(
    mfu_prefill: float,
    mbu_prefill: float,
    mfu_decode: float,
    mbu_decode: float,
    padding_overhead: float
) -> List[str]:
    """Generate optimization recommendations based on utilization metrics.
    
    Args:
        mfu_prefill: Prefill MFU
        mbu_prefill: Prefill MBU
        mfu_decode: Decode MFU  
        mbu_decode: Decode MBU
        padding_overhead: Padding waste factor
        
    Returns:
        List of optimization recommendation strings
    """
    recommendations = []
    
    # Prefill optimizations (should be compute-bound)
    if mfu_prefill < 0.5:
        recommendations.append("Increase batch size to improve prefill compute utilization")
        recommendations.append("Enable Flash Attention for memory efficiency during prefill")
    
    # Decode optimizations (should be memory-bound)  
    if mbu_decode < 0.6:
        recommendations.append("Optimize weight quantization (consider FP8 or INT4)")
        recommendations.append("Improve KV cache layout for better memory access patterns")
    
    # Padding optimizations
    if padding_overhead > 1.3:
        recommendations.append("Reduce sequence length variance to minimize padding waste")
        recommendations.append("Consider dynamic batching to reduce padding overhead")
    
    # General optimizations
    if mfu_prefill < 0.3 and mfu_decode < 0.3:
        recommendations.append("Check for scheduling bottlenecks or resource contention")
    
    if len(recommendations) == 0:
        recommendations.append("Performance looks optimal for current configuration")
    
    return recommendations


def integrate_analytical_model(
    vllm_metrics: Dict[str, float],
    analytical_config: Dict[str, Any],
    enable_analytical: bool = True
) -> Dict[str, Any]:
    """Main integration function to run analytical model and merge results.
    
    Args:
        vllm_metrics: Timing metrics extracted from vLLM
        analytical_config: Configuration for analytical model
        enable_analytical: Whether to run analytical model
        
    Returns:
        Dictionary with analytical efficiency metrics
    """
    if not enable_analytical:
        return {}
    
    try:
        # Initialize analytical model
        model = UtilizationModel()
        
        # Prepare metrics for analytical model
        analytical_metrics = {
            'prefill_time': vllm_metrics['prefill_time'],
            'decode_time': vllm_metrics['decode_time']
        }
        
        # Run analytical model
        results = model.calculate_mfu_mbu(
            metrics=analytical_metrics,
            batch_size=analytical_config['batch_size'],
            actual_seq_lengths=analytical_config['actual_seq_lengths'],
            max_seq_length=analytical_config['max_seq_length'],
            num_decode_tokens=analytical_config['num_decode_tokens']
        )
        
        # Detect bottlenecks
        prefill_bottleneck = detect_bottleneck(
            results['mfu_prefill'], 
            results['mbu_prefill'], 
            'prefill'
        )
        decode_bottleneck = detect_bottleneck(
            results['mfu_decode'], 
            results['mbu_decode'], 
            'decode'
        )
        
        # Generate recommendations
        recommendations = generate_optimization_recommendations(
            results['mfu_prefill'],
            results['mbu_prefill'],
            results['mfu_decode'],
            results['mbu_decode'],
            results['details']['padding_overhead']
        )
        
        # Format results for integration
        analytical_results = {
            # Core utilization metrics (as percentages)
            'mfu_prefill_analytical': results['mfu_prefill'] * 100,
            'mbu_prefill_analytical': results['mbu_prefill'] * 100,
            'mfu_decode_analytical': results['mfu_decode'] * 100,
            'mbu_decode_analytical': results['mbu_decode'] * 100,
            'mfu_total_analytical': results['mfu_total'] * 100,
            'mbu_total_analytical': results['mbu_total'] * 100,
            
            # Efficiency factors
            'padding_overhead': results['details']['padding_overhead'],
            'effective_token_ratio': results['details']['effective_token_ratio'],
            'flash_attention_factor': results['details']['flash_attention_factor'],
            'tensor_core_efficiency': results['details']['tensor_core_efficiency'],
            
            # Bottleneck analysis
            'prefill_bottleneck': prefill_bottleneck,
            'decode_bottleneck': decode_bottleneck,
            'optimization_recommendations': recommendations,
            
            # Raw metrics for validation
            'raw_results': {
                'mfu_prefill_raw': results['mfu_prefill_raw'],
                'mbu_prefill_raw': results['mbu_prefill_raw'],
                'mfu_decode_raw': results['mfu_decode_raw'],
                'mbu_decode_raw': results['mbu_decode_raw']
            }
        }
        
        logger.info("Analytical model integration successful")
        return analytical_results
        
    except Exception as e:
        logger.error(f"Analytical model integration failed: {e}")
        return {
            'error': str(e),
            'mfu_prefill_analytical': 0.0,
            'mbu_prefill_analytical': 0.0,
            'mfu_decode_analytical': 0.0,
            'mbu_decode_analytical': 0.0,
            'mfu_total_analytical': 0.0,
            'mbu_total_analytical': 0.0,
            'padding_overhead': 1.0,
            'effective_token_ratio': 1.0
        }


def validate_analytical_results(
    gpu_monitoring: Dict[str, float],
    analytical: Dict[str, float],
    tolerance: float = 0.2
) -> Dict[str, Any]:
    """Validate analytical model results against GPU monitoring.
    
    Args:
        gpu_monitoring: Results from GPU monitoring
        analytical: Results from analytical model
        tolerance: Acceptable relative difference (0.0-1.0)
        
    Returns:
        Validation results and comparison metrics
    """
    validation = {
        'validation_enabled': True,
        'tolerance': tolerance,
        'comparisons': {}
    }
    
    # Compare MFU metrics
    if 'mfu_percent' in gpu_monitoring and 'mfu_total_analytical' in analytical:
        gpu_mfu = gpu_monitoring['mfu_percent']
        analytical_mfu = analytical['mfu_total_analytical']
        
        if gpu_mfu > 0:
            relative_diff = abs(gpu_mfu - analytical_mfu) / gpu_mfu
            validation['comparisons']['mfu'] = {
                'gpu_monitoring': gpu_mfu,
                'analytical': analytical_mfu,
                'relative_difference': relative_diff,
                'within_tolerance': relative_diff <= tolerance
            }
    
    # Compare MBU metrics
    if 'mbu_percent' in gpu_monitoring and 'mbu_total_analytical' in analytical:
        gpu_mbu = gpu_monitoring['mbu_percent']
        analytical_mbu = analytical['mbu_total_analytical']
        
        if gpu_mbu > 0:
            relative_diff = abs(gpu_mbu - analytical_mbu) / gpu_mbu
            validation['comparisons']['mbu'] = {
                'gpu_monitoring': gpu_mbu,
                'analytical': analytical_mbu,
                'relative_difference': relative_diff,
                'within_tolerance': relative_diff <= tolerance
            }
    
    return validation


if __name__ == "__main__":
    # Example usage
    print("Model Integration Utilities")
    print("Use this module to integrate vLLM benchmarks with analytical model")