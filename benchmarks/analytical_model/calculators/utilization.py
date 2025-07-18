"""Utilization calculators for MFU and MBU analysis."""

from typing import Dict, Tuple
from dataclasses import dataclass

from ..hardware.base import BaseHardware
from ..models.base import BaseModel, ParallelismConfig, ModelMetrics


@dataclass
class UtilizationMetrics:
    """Container for utilization metrics."""
    total_mfu: float
    total_mbu: float
    prefill_mfu: float
    prefill_mbu: float
    decode_mfu: float
    decode_mbu: float
    
    # Additional debugging info
    actual_total_flops_per_sec: float
    actual_prefill_flops_per_sec: float
    actual_decode_flops_per_sec: float
    actual_total_memory_bw_gb_s: float
    actual_prefill_memory_bw_gb_s: float
    actual_decode_memory_bw_gb_s: float


@dataclass
class BenchmarkResults:
    """Container for benchmark results from CSV."""
    input_len: int
    output_len: int
    batch_size: int
    tp_size: int
    ep_enabled: bool
    system_throughput: float  # tokens/sec
    request_throughput: float  # requests/sec
    avg_prefill_time: float  # seconds
    avg_decode_time: float  # seconds


class UtilizationCalculator:
    """Calculator for Model FLOPs Utilization (MFU) and Memory Bandwidth Utilization (MBU)."""
    
    def __init__(self, hardware: BaseHardware, model: BaseModel, dtype: str = 'fp16'):
        """Initialize the calculator.
        
        Args:
            hardware: Hardware specification
            model: Model specification
            dtype: Data type for computation
        """
        self.hardware = hardware
        self.model = model
        self.dtype = dtype
        
        # Validate dtype is supported by hardware
        self.hardware.validate_dtype(dtype)
        
        # Get peak performance specs
        self.peak_flops_tflops = self.hardware.get_peak_flops(dtype)
        self.peak_memory_bw_gb_s = self.hardware.peak_memory_bandwidth_gb_s
    
    def calculate_utilization(
        self,
        benchmark_results: BenchmarkResults
    ) -> UtilizationMetrics:
        """Calculate utilization metrics from benchmark results.
        
        Args:
            benchmark_results: Results from benchmark run
            
        Returns:
            UtilizationMetrics with MFU and MBU values
        """
        # Add debug info for problematic cases
        debug_info = (
            f"input_len={benchmark_results.input_len}, "
            f"batch_size={benchmark_results.batch_size}, "
            f"prefill_time={benchmark_results.avg_prefill_time:.4f}s"
        )
        # Create parallelism configuration
        parallelism_config = ParallelismConfig(
            tp_size=benchmark_results.tp_size,
            ep_enabled=benchmark_results.ep_enabled,
            use_flash_attention=True  # Assume Flash Attention is used
        )
        
        # Calculate theoretical model metrics
        model_metrics = self.model.calculate_all_metrics(
            input_len=benchmark_results.input_len,
            output_len=benchmark_results.output_len,
            batch_size=benchmark_results.batch_size,
            parallelism_config=parallelism_config
        )
        
        # Calculate actual FLOPs/sec
        actual_flops = self._calculate_actual_flops_per_sec(
            benchmark_results, model_metrics
        )
        
        # Calculate actual memory bandwidth
        actual_memory_bw = self._calculate_actual_memory_bw_gb_s(
            benchmark_results, model_metrics
        )
        
        # Calculate utilization percentages
        total_mfu = actual_flops.total / self.peak_flops_tflops
        prefill_mfu = actual_flops.prefill / self.peak_flops_tflops
        decode_mfu = actual_flops.decode / self.peak_flops_tflops
        
        total_mbu = actual_memory_bw.total / self.peak_memory_bw_gb_s
        prefill_mbu = actual_memory_bw.prefill / self.peak_memory_bw_gb_s
        decode_mbu = actual_memory_bw.decode / self.peak_memory_bw_gb_s
        
        # Validate utilization values - warn if they exceed 100%
        if total_mfu > 1.0:
            print(f"WARNING: Total MFU {total_mfu:.3f} exceeds 100% - this may indicate calculation errors")
        if prefill_mfu > 1.0:
            print(f"WARNING: Prefill MFU {prefill_mfu:.3f} exceeds 100% - actual {actual_flops.prefill:.1f} TFLOPS/sec vs peak {self.peak_flops_tflops:.1f} TFLOPS")
        if decode_mfu > 1.0:
            print(f"WARNING: Decode MFU {decode_mfu:.3f} exceeds 100% - this may indicate calculation errors")
        if total_mbu > 1.0:
            print(f"WARNING: Total MBU {total_mbu:.3f} exceeds 100% - this may indicate calculation errors")
        if prefill_mbu > 1.0:
            print(f"WARNING: Prefill MBU {prefill_mbu:.3f} exceeds 100% - this may indicate calculation errors")
        if decode_mbu > 1.0:
            print(f"WARNING: Decode MBU {decode_mbu:.3f} exceeds 100% - this may indicate calculation errors")
        
        return UtilizationMetrics(
            total_mfu=total_mfu,
            total_mbu=total_mbu,
            prefill_mfu=prefill_mfu,
            prefill_mbu=prefill_mbu,
            decode_mfu=decode_mfu,
            decode_mbu=decode_mbu,
            actual_total_flops_per_sec=actual_flops.total,
            actual_prefill_flops_per_sec=actual_flops.prefill,
            actual_decode_flops_per_sec=actual_flops.decode,
            actual_total_memory_bw_gb_s=actual_memory_bw.total,
            actual_prefill_memory_bw_gb_s=actual_memory_bw.prefill,
            actual_decode_memory_bw_gb_s=actual_memory_bw.decode,
        )
    
    def _calculate_actual_flops_per_sec(
        self,
        benchmark_results: BenchmarkResults,
        model_metrics: ModelMetrics
    ) -> Tuple[float, float, float]:
        """Calculate actual FLOPs/sec for total, prefill, and decode phases.
        
        Returns:
            Tuple of (total_flops_per_sec, prefill_flops_per_sec, decode_flops_per_sec)
        """
        # Calculate actual FLOPs/sec
        # The benchmark times are per-sequence, but our FLOP calculations are for the entire batch
        # So we need to divide FLOPs by batch_size to get per-sequence FLOPs, then divide by time
        
        # For prefill: FLOPs per sequence divided by time per sequence
        prefill_flops_per_sequence = model_metrics.prefill_flops / benchmark_results.batch_size
        actual_prefill_flops_per_sec = (
            prefill_flops_per_sequence / benchmark_results.avg_prefill_time
            if benchmark_results.avg_prefill_time > 0 else 0.0
        )
        
        # For decode: FLOPs per sequence divided by time per sequence
        decode_flops_per_sequence = model_metrics.decode_flops / benchmark_results.batch_size
        actual_decode_flops_per_sec = (
            decode_flops_per_sequence / benchmark_results.avg_decode_time
            if benchmark_results.avg_decode_time > 0 else 0.0
        )
        
        # Total: weighted average based on time spent
        total_time = benchmark_results.avg_prefill_time + benchmark_results.avg_decode_time
        if total_time > 0:
            prefill_weight = benchmark_results.avg_prefill_time / total_time
            decode_weight = benchmark_results.avg_decode_time / total_time
            actual_total_flops_per_sec = (
                prefill_weight * actual_prefill_flops_per_sec + 
                decode_weight * actual_decode_flops_per_sec
            )
        else:
            actual_total_flops_per_sec = 0.0
        
        return self._FlopsResult(
            total=actual_total_flops_per_sec,
            prefill=actual_prefill_flops_per_sec,
            decode=actual_decode_flops_per_sec
        )
    
    def _calculate_actual_memory_bw_gb_s(
        self,
        benchmark_results: BenchmarkResults,
        model_metrics: ModelMetrics
    ) -> Tuple[float, float, float]:
        """Calculate actual memory bandwidth usage for total, prefill, and decode phases.
        
        Returns:
            Tuple of (total_bw_gb_s, prefill_bw_gb_s, decode_bw_gb_s)
        """
        # Calculate actual memory bandwidth
        # The benchmark times are per-sequence, but our memory calculations are for the entire batch
        # So we need to divide memory by batch_size to get per-sequence memory, then divide by time
        
        # For prefill: memory per sequence divided by time per sequence
        prefill_memory_per_sequence = model_metrics.prefill_memory_gb / benchmark_results.batch_size
        actual_prefill_memory_bw = (
            prefill_memory_per_sequence / benchmark_results.avg_prefill_time
            if benchmark_results.avg_prefill_time > 0 else 0.0
        )
        
        # For decode: memory per sequence divided by time per sequence
        decode_memory_per_sequence = model_metrics.decode_memory_gb / benchmark_results.batch_size
        actual_decode_memory_bw = (
            decode_memory_per_sequence / benchmark_results.avg_decode_time
            if benchmark_results.avg_decode_time > 0 else 0.0
        )
        
        # Total: weighted average based on time spent
        total_time = benchmark_results.avg_prefill_time + benchmark_results.avg_decode_time
        if total_time > 0:
            prefill_weight = benchmark_results.avg_prefill_time / total_time
            decode_weight = benchmark_results.avg_decode_time / total_time
            actual_total_memory_bw = (
                prefill_weight * actual_prefill_memory_bw + 
                decode_weight * actual_decode_memory_bw
            )
        else:
            actual_total_memory_bw = 0.0
        
        return self._MemoryResult(
            total=actual_total_memory_bw,
            prefill=actual_prefill_memory_bw,
            decode=actual_decode_memory_bw
        )
    
    @dataclass
    class _FlopsResult:
        total: float
        prefill: float
        decode: float
    
    @dataclass 
    class _MemoryResult:
        total: float
        prefill: float
        decode: float