# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU hardware specifications for MFU/MBU calculation"""

import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

@dataclass
class GPUMetrics:
    """Container for GPU performance metrics"""
    name: str
    peak_tflops_fp16: float
    peak_tflops_fp8: Optional[float]
    peak_memory_bw_gb_s: float
    memory_size_gb: float

class GPUSpecs:
    """GPU hardware specifications for MFU/MBU calculation"""
    
    # Comprehensive GPU specifications including H200 and B200
    PEAK_FLOPS = {
        "NVIDIA A100": {
            "fp16": 312.0,
            "bf16": 312.0,
            "fp32": 19.5,
            "tf32": 156.0,
            "fp8": 624.0,  # Structured sparsity
        },
        "NVIDIA H100": {
            "fp16": 989.0,
            "bf16": 989.0,
            "fp32": 67.0,
            "tf32": 494.0,
            "fp8": 1979.0,
        },
        "NVIDIA H200": {  # Enhanced H100 with more memory
            "fp16": 989.0,   # Same compute as H100
            "bf16": 989.0,
            "fp32": 67.0,
            "tf32": 494.0,
            "fp8": 1979.0,
        },
        "NVIDIA B200": {  # Blackwell architecture estimates
            "fp16": 2250.0,  # Estimated 2.3x improvement over H100
            "bf16": 2250.0,
            "fp32": 150.0,
            "tf32": 1125.0,
            "fp8": 4500.0,
            "fp4": 9000.0,  # New precision for Blackwell
        },
        "NVIDIA V100": {
            "fp16": 125.0,
            "fp32": 15.7,
            "tf32": 62.5,
        },
        "NVIDIA A10": {
            "fp16": 125.0,
            "fp32": 31.2,
            "tf32": 62.5,
        },
        "NVIDIA L40": {
            "fp16": 362.0,
            "fp32": 90.5,
            "tf32": 181.0,
            "fp8": 724.0,
        },
        "NVIDIA L40S": {
            "fp16": 733.0,
            "fp32": 91.6,
            "tf32": 183.0,
            "fp8": 1466.0,
        },
        "NVIDIA T4": {
            "fp16": 65.1,
            "fp32": 8.1,
        },
        "NVIDIA RTX 4090": {
            "fp16": 165.0,
            "fp32": 83.0,
            "tf32": 166.0,
        },
        "NVIDIA RTX 3090": {
            "fp16": 71.0,
            "fp32": 36.0,
        },
    }
    
    # Memory bandwidth in GB/s
    MEMORY_BANDWIDTH = {
        "NVIDIA A100": 2039.0,    # HBM2e
        "NVIDIA H100": 3350.0,    # HBM3
        "NVIDIA H200": 4800.0,    # HBM3e - Higher bandwidth!
        "NVIDIA B200": 8000.0,    # HBM3e (estimated)
        "NVIDIA V100": 900.0,     # HBM2
        "NVIDIA A10": 600.0,      # GDDR6
        "NVIDIA L40": 864.0,      # GDDR6
        "NVIDIA L40S": 864.0,     # GDDR6X
        "NVIDIA T4": 320.0,       # GDDR6
        "NVIDIA RTX 4090": 1008.0, # GDDR6X
        "NVIDIA RTX 3090": 936.0,  # GDDR6X
    }
    
    # Memory sizes in GB
    MEMORY_SIZES = {
        "NVIDIA A100": 80.0,      # 40GB and 80GB variants
        "NVIDIA H100": 80.0,      # SXM and PCIe variants
        "NVIDIA H200": 141.0,     # Enhanced memory
        "NVIDIA B200": 192.0,     # Estimated
        "NVIDIA V100": 32.0,      # 16GB and 32GB variants
        "NVIDIA A10": 24.0,
        "NVIDIA L40": 48.0,
        "NVIDIA L40S": 48.0,
        "NVIDIA T4": 16.0,
        "NVIDIA RTX 4090": 24.0,
        "NVIDIA RTX 3090": 24.0,
    }
    
    @staticmethod
    def detect_gpu() -> Optional[str]:
        """Auto-detect GPU type with comprehensive support"""
        if not torch.cuda.is_available():
            return None
            
        gpu_name = torch.cuda.get_device_name(0)
        
        # Extended GPU mapping with multiple name variations
        gpu_mappings = [
            # Primary mappings
            ("A100", "NVIDIA A100"),
            ("H100", "NVIDIA H100"), 
            ("H200", "NVIDIA H200"),
            ("B200", "NVIDIA B200"),
            ("Blackwell", "NVIDIA B200"),  # Alternative name
            ("V100", "NVIDIA V100"),
            ("A10G", "NVIDIA A10"),  # AWS variant
            ("A10", "NVIDIA A10"),
            ("L40S", "NVIDIA L40S"),
            ("L40", "NVIDIA L40"),
            ("T4", "NVIDIA T4"),
            ("RTX 4090", "NVIDIA RTX 4090"),
            ("RTX 3090", "NVIDIA RTX 3090"),
        ]
        
        # Check for exact matches first
        for pattern, standard_name in gpu_mappings:
            if pattern in gpu_name:
                return standard_name
        
        # Special handling for H200 (might show as H100 variant with more memory)
        if "H100" in gpu_name:
            try:
                # Try to detect H200 by memory size
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_gb = total_memory / (1024**3)
                if memory_gb > 100:  # H200 has 141GB vs H100's 80GB
                    return "NVIDIA H200"
            except:
                pass
                
        return gpu_name  # Return raw name if not found
    
    @staticmethod
    def get_peak_tflops(gpu_name: str, dtype: str = "fp16") -> float:
        """Get peak TFLOPS for given GPU and dtype"""
        if gpu_name in GPUSpecs.PEAK_FLOPS:
            return GPUSpecs.PEAK_FLOPS[gpu_name].get(dtype, 0.0)
        return 0.0
    
    @staticmethod
    def get_memory_bandwidth(gpu_name: str) -> float:
        """Get peak memory bandwidth in GB/s"""
        return GPUSpecs.MEMORY_BANDWIDTH.get(gpu_name, 0.0)
    
    @staticmethod
    def get_memory_size(gpu_name: str) -> float:
        """Get memory size in GB"""
        return GPUSpecs.MEMORY_SIZES.get(gpu_name, 0.0)
    
    @staticmethod
    def get_supported_dtypes(gpu_name: str) -> list[str]:
        """Get supported data types for a GPU"""
        if gpu_name in GPUSpecs.PEAK_FLOPS:
            return list(GPUSpecs.PEAK_FLOPS[gpu_name].keys())
        return []
    
    @staticmethod
    def is_supported_gpu(gpu_name: str) -> bool:
        """Check if GPU is supported for MFU/MBU calculation"""
        return gpu_name in GPUSpecs.PEAK_FLOPS
    
    @staticmethod
    def get_gpu_info(gpu_name: Optional[str] = None) -> Dict[str, any]:
        """Get complete GPU information"""
        if gpu_name is None:
            gpu_name = GPUSpecs.detect_gpu()
            
        if not gpu_name or not GPUSpecs.is_supported_gpu(gpu_name):
            return {
                "error": f"GPU '{gpu_name}' not supported or not found",
                "supported_gpus": list(GPUSpecs.PEAK_FLOPS.keys())
            }
        
        return {
            "name": gpu_name,
            "peak_flops": GPUSpecs.PEAK_FLOPS[gpu_name],
            "memory_bandwidth_gb_s": GPUSpecs.get_memory_bandwidth(gpu_name),
            "memory_size_gb": GPUSpecs.get_memory_size(gpu_name),
            "supported_dtypes": GPUSpecs.get_supported_dtypes(gpu_name),
            "detected": gpu_name == GPUSpecs.detect_gpu()
        }
    
    @staticmethod
    def list_supported_gpus() -> list[str]:
        """List all supported GPU types"""
        return sorted(GPUSpecs.PEAK_FLOPS.keys())

# Convenience functions for backward compatibility
def detect_gpu() -> Optional[str]:
    """Detect current GPU type"""
    return GPUSpecs.detect_gpu()

def get_peak_tflops(gpu_name: str, dtype: str = "fp16") -> float:
    """Get peak TFLOPS for GPU and dtype"""
    return GPUSpecs.get_peak_tflops(gpu_name, dtype)

def get_memory_bandwidth(gpu_name: str) -> float:
    """Get memory bandwidth for GPU"""
    return GPUSpecs.get_memory_bandwidth(gpu_name)