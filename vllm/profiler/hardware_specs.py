"""GPU hardware specifications database for efficiency metrics calculation."""

from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# GPU specifications: {datatype: peak_flops_tflops, "memory_bandwidth_gb_s": val, "memory_size_gb": val}
GPU_SPECS = {
    # NVIDIA H100 Series
    "H100": {
        "fp32": 67.0,      # FP32 peak performance
        "fp16": 1979.0,    # FP16 Tensor Core performance  
        "bf16": 1979.0,    # BF16 Tensor Core performance
        "fp8": 3958.0,     # FP8 Tensor Core performance (2x FP16)
        "int8": 3958.0,    # INT8 Tensor Core performance
        "int4": 7916.0,    # INT4 Tensor Core performance (4x FP16) 
        "memory_bandwidth_gb_s": 3350.0,
        "memory_size_gb": 80.0
    },
    "H100-PCIe": {
        "fp32": 51.0,
        "fp16": 1513.0,
        "bf16": 1513.0, 
        "fp8": 3026.0,
        "int8": 3026.0,
        "int4": 6052.0,
        "memory_bandwidth_gb_s": 2000.0,
        "memory_size_gb": 80.0
    },
    
    # NVIDIA H200 Series
    "H200": {
        "fp32": 67.0,
        "fp16": 1979.0,
        "bf16": 1979.0,
        "fp8": 3958.0,
        "int8": 3958.0,
        "int4": 7916.0,
        "memory_bandwidth_gb_s": 4800.0,
        "memory_size_gb": 141.0
    },
    "H200-PCIe": {
        "fp32": 51.0,
        "fp16": 1513.0,
        "bf16": 1513.0,
        "fp8": 3026.0,
        "int8": 3026.0,
        "int4": 6052.0,
        "memory_bandwidth_gb_s": 4800.0,
        "memory_size_gb": 141.0
    },
    
    # NVIDIA B200 Series (Blackwell)
    "B200": {
        "fp32": 125.0,     # Estimated
        "fp16": 5000.0,    # Estimated 2.5x H100
        "bf16": 5000.0,
        "fp8": 10000.0,    # Estimated 2x FP16
        "int8": 10000.0,
        "int4": 20000.0,   # Estimated 4x FP16
        "memory_bandwidth_gb_s": 8000.0,
        "memory_size_gb": 192.0
    },
    "B200-PCIe": {
        "fp32": 100.0,     # Estimated
        "fp16": 4000.0,    # Estimated
        "bf16": 4000.0,
        "fp8": 8000.0,
        "int8": 8000.0,
        "int4": 16000.0,
        "memory_bandwidth_gb_s": 6400.0,
        "memory_size_gb": 192.0
    },
    
    # NVIDIA A100 Series  
    "A100": {
        "fp32": 19.5,
        "fp16": 624.0,     # Tensor Core FP16
        "bf16": 624.0,     # Tensor Core BF16
        "fp8": 1248.0,     # Sparsity-enabled
        "int8": 1248.0,
        "int4": 2496.0,
        "memory_bandwidth_gb_s": 2039.0,
        "memory_size_gb": 80.0
    },
    "A100-SXM4-40GB": {
        "fp32": 19.5,
        "fp16": 624.0,
        "bf16": 624.0,
        "fp8": 1248.0,
        "int8": 1248.0,
        "int4": 2496.0,
        "memory_bandwidth_gb_s": 2039.0,
        "memory_size_gb": 40.0
    },
    "A100-PCIe": {
        "fp32": 19.5,
        "fp16": 624.0,
        "bf16": 624.0,
        "fp8": 1248.0,
        "int8": 1248.0,
        "int4": 2496.0,
        "memory_bandwidth_gb_s": 1935.0,
        "memory_size_gb": 80.0
    },
    "A100-PCIe-40GB": {
        "fp32": 19.5,
        "fp16": 624.0,
        "bf16": 624.0,
        "fp8": 1248.0,
        "int8": 1248.0,
        "int4": 2496.0,
        "memory_bandwidth_gb_s": 1935.0,
        "memory_size_gb": 40.0
    },
    
    # NVIDIA V100 Series
    "V100": {
        "fp32": 15.7,
        "fp16": 112.0,     # Tensor Core FP16
        "bf16": 112.0,
        "fp8": 224.0,      # Estimated
        "int8": 224.0,
        "int4": 448.0,
        "memory_bandwidth_gb_s": 900.0,
        "memory_size_gb": 32.0
    },
    "V100-PCIe": {
        "fp32": 14.0,
        "fp16": 112.0,
        "bf16": 112.0,
        "fp8": 224.0,
        "int8": 224.0,
        "int4": 448.0,
        "memory_bandwidth_gb_s": 900.0,
        "memory_size_gb": 32.0
    },
    "V100S": {
        "fp32": 16.4,
        "fp16": 130.0,
        "bf16": 130.0,
        "fp8": 260.0,
        "int8": 260.0,
        "int4": 520.0,
        "memory_bandwidth_gb_s": 1134.0,
        "memory_size_gb": 32.0
    },
    
    # NVIDIA A6000/RTX Series (Ada Lovelace/Ampere)
    "RTX A6000": {
        "fp32": 38.7,
        "fp16": 154.8,
        "bf16": 154.8,
        "fp8": 309.6,
        "int8": 309.6,
        "int4": 619.2,
        "memory_bandwidth_gb_s": 768.0,
        "memory_size_gb": 48.0
    },
    "RTX 4090": {
        "fp32": 83.0,
        "fp16": 166.9,    # Tensor performance
        "bf16": 166.9,
        "fp8": 333.8,
        "int8": 333.8,
        "int4": 667.6,
        "memory_bandwidth_gb_s": 1008.0,
        "memory_size_gb": 24.0
    },
    "RTX 3090": {
        "fp32": 35.6,
        "fp16": 142.0,
        "bf16": 142.0,
        "fp8": 284.0,
        "int8": 284.0,
        "int4": 568.0,
        "memory_bandwidth_gb_s": 936.0,
        "memory_size_gb": 24.0
    },
    
    # NVIDIA L40 Series
    "L40": {
        "fp32": 90.5,
        "fp16": 181.0,
        "bf16": 181.0,
        "fp8": 362.0,
        "int8": 362.0,
        "int4": 724.0,
        "memory_bandwidth_gb_s": 864.0,
        "memory_size_gb": 48.0
    },
    "L40S": {
        "fp32": 91.6,
        "fp16": 362.0,    # Ada Lovelace improvements
        "bf16": 362.0,
        "fp8": 724.0,
        "int8": 724.0,
        "int4": 1448.0,
        "memory_bandwidth_gb_s": 864.0,
        "memory_size_gb": 48.0
    },
    
    # NVIDIA T4
    "T4": {
        "fp32": 8.1,
        "fp16": 65.0,
        "bf16": 65.0,
        "fp8": 130.0,
        "int8": 130.0,
        "int4": 260.0,
        "memory_bandwidth_gb_s": 320.0,
        "memory_size_gb": 16.0
    },
}

# Common GPU name mappings to standardized specs
GPU_NAME_MAPPINGS = {
    # H100 variants
    "NVIDIA H100": "H100",
    "NVIDIA H100 80GB HBM3": "H100",
    "H100-SXM5-80GB": "H100",
    "H100-PCIE-80GB": "H100-PCIe",
    
    # H200 variants
    "NVIDIA H200": "H200",
    "NVIDIA H200 141GB HBM3e": "H200",
    "H200-SXM5-141GB": "H200",
    "H200-PCIE-141GB": "H200-PCIe",
    
    # B200 variants
    "NVIDIA B200": "B200",
    "B200-SXM6-192GB": "B200",
    "B200-PCIE-192GB": "B200-PCIe",
    
    # A100 variants
    "NVIDIA A100-SXM4-80GB": "A100",
    "NVIDIA A100-SXM4-40GB": "A100-SXM4-40GB", 
    "NVIDIA A100-PCIE-80GB": "A100-PCIe",
    "NVIDIA A100-PCIE-40GB": "A100-PCIe-40GB",
    "A100-SXM-80GB": "A100",
    
    # V100 variants
    "Tesla V100-SXM2-32GB": "V100",
    "Tesla V100-PCIE-32GB": "V100-PCIe",
    "Tesla V100S-PCIE-32GB": "V100S",
    
    # RTX variants
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA RTX A6000": "RTX A6000",
    
    # L40 variants
    "NVIDIA L40": "L40",
    "NVIDIA L40S": "L40S",
    
    # T4
    "Tesla T4": "T4",
    "NVIDIA T4": "T4",
}


def get_gpu_specs(gpu_name: str, datatype: str = "fp16") -> Optional[Dict]:
    """Get GPU specifications by name and datatype.
    
    Args:
        gpu_name: GPU model name as returned by nvidia-ml-py
        datatype: Computation datatype ("fp32", "fp16", "bf16", "fp8", "int8", "int4")
        
    Returns:
        Dict with GPU specs including peak_flops_tflops for specified datatype,
        memory_bandwidth_gb_s, and memory_size_gb, or None if GPU not found
    """
    gpu_spec_dict = None
    
    # Try direct lookup first
    if gpu_name in GPU_SPECS:
        gpu_spec_dict = GPU_SPECS[gpu_name]
    
    # Try mapped lookup
    elif gpu_name in GPU_NAME_MAPPINGS:
        mapped_name = GPU_NAME_MAPPINGS[gpu_name]
        if mapped_name in GPU_SPECS:
            gpu_spec_dict = GPU_SPECS[mapped_name]
    
    # Try partial matching for common variants
    else:
        gpu_name_clean = gpu_name.upper().replace("-", "").replace(" ", "")
        for spec_name in GPU_SPECS:
            spec_name_clean = spec_name.upper().replace("-", "").replace(" ", "")
            if spec_name_clean in gpu_name_clean or gpu_name_clean in spec_name_clean:
                logger.info(f"Matched GPU '{gpu_name}' to spec '{spec_name}' via partial matching")
                gpu_spec_dict = GPU_SPECS[spec_name]
                break
    
    if gpu_spec_dict is None:
        logger.warning(f"GPU specifications not found for: {gpu_name}")
        return None
    
    # Check if requested datatype is available
    if datatype not in gpu_spec_dict:
        logger.warning(f"Datatype '{datatype}' not available for {gpu_name}, falling back to fp16")
        datatype = "fp16"
        if datatype not in gpu_spec_dict:
            logger.error(f"No supported datatypes found for {gpu_name}")
            return None
    
    return {
        "peak_flops_tflops": gpu_spec_dict[datatype],
        "memory_bandwidth_gb_s": gpu_spec_dict["memory_bandwidth_gb_s"],
        "memory_size_gb": gpu_spec_dict["memory_size_gb"],
        "datatype": datatype
    }


def estimate_transformer_flops(
    batch_size: int,
    seq_len: int, 
    hidden_size: int,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    intermediate_size: Optional[int] = None
) -> float:
    """Estimate FLOPs for transformer forward pass.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length  
        hidden_size: Hidden dimension
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: MLP intermediate size (default: 4 * hidden_size)
        
    Returns:
        Estimated FLOPs for forward pass
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    
    # Embedding layer: batch_size * seq_len * vocab_size * hidden_size
    embedding_flops = batch_size * seq_len * vocab_size * hidden_size
    
    # Per layer calculations
    layer_flops = 0
    
    # Self-attention
    # Q, K, V projections: 3 * (batch_size * seq_len * hidden_size * hidden_size)
    qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
    
    # Attention scores: batch_size * num_heads * seq_len * seq_len * (hidden_size / num_heads)
    attention_flops = batch_size * num_attention_heads * seq_len * seq_len * (hidden_size // num_attention_heads)
    
    # Attention output projection: batch_size * seq_len * hidden_size * hidden_size  
    output_proj_flops = batch_size * seq_len * hidden_size * hidden_size
    
    # MLP layers: 2 * (batch_size * seq_len * hidden_size * intermediate_size)
    mlp_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size
    
    layer_flops = qkv_flops + attention_flops + output_proj_flops + mlp_flops
    
    # Total FLOPs
    total_flops = embedding_flops + (num_layers * layer_flops)
    
    return total_flops


def calculate_mfu(
    observed_tokens_per_second: float,
    gpu_specs: Dict,
    model_params: Dict
) -> float:
    """Calculate Model FLOPs Utilization (MFU).
    
    Args:
        observed_tokens_per_second: Measured token generation rate
        gpu_specs: Dict with GPU specifications including peak_flops_tflops
        model_params: Dict with model parameters (batch_size, seq_len, etc.)
        
    Returns:
        MFU as percentage (0-100)
    """
    peak_flops_tflops = gpu_specs["peak_flops_tflops"]
    peak_flops_per_second = peak_flops_tflops * 1e12
    
    # Estimate FLOPs per token
    flops_per_forward = estimate_transformer_flops(**model_params)
    flops_per_token = flops_per_forward / (model_params['batch_size'] * model_params['seq_len'])
    
    # Calculate observed FLOP/s
    observed_flops_per_second = observed_tokens_per_second * flops_per_token
    
    # Calculate MFU
    mfu = (observed_flops_per_second / peak_flops_per_second) * 100
    
    return min(mfu, 100.0)  # Cap at 100%


def calculate_mbu(
    memory_throughput_gb_s: float,
    gpu_specs: Dict
) -> float:
    """Calculate Memory Bandwidth Utilization (MBU).
    
    Args:
        memory_throughput_gb_s: Observed memory throughput in GB/s
        gpu_specs: Dict with GPU specifications including memory_bandwidth_gb_s
        
    Returns:
        MBU as percentage (0-100)
    """
    peak_memory_bandwidth_gb_s = gpu_specs["memory_bandwidth_gb_s"]
    
    mbu = (memory_throughput_gb_s / peak_memory_bandwidth_gb_s) * 100
    
    return min(mbu, 100.0)  # Cap at 100%