#!/usr/bin/env python3
"""
utilization_analytical_model.py

Analytical model for calculating Model FLOPs Utilization (MFU) and Memory Bandwidth Utilization (MBU)
for DeepSeek-V3 inference on H200 GPUs with offline batch processing.

This model accounts for:
- Native FP8 weights (no quantization overhead)
- Flash Attention v2 optimizations
- Tensor Parallelism (TP=8) and Expert Parallelism (EP=8)
- vLLM kernel fusion
- Offline batch inference (with padding overhead)
"""

import numpy as np
from typing import Dict, List, Tuple


class DeepSeekV3Config:
    """Model configuration for DeepSeek-V3"""
    def __init__(self):
        self.hidden_size = 7168
        self.num_layers = 61
        self.num_attention_heads = 128
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.n_routed_experts = 256
        self.n_shared_experts = 1
        self.experts_per_tok = 8
        self.moe_intermediate_size = 2048
        self.routed_scaling_factor = 2.5
        self.intermediate_size = 18432
        self.vocab_size = 129280
        
        # Derived values
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.q_dim = self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        self.k_dim = self.num_attention_heads * self.qk_nope_head_dim
        self.v_dim = self.num_attention_heads * self.v_head_dim
        self.expert_hidden_dim = int(self.moe_intermediate_size * self.routed_scaling_factor)


class H200Config:
    """Hardware configuration for H200 GPU"""
    def __init__(self):
        self.peak_fp16_tflops = 1979e12
        self.peak_fp8_tflops = 3958e12  # 2x FP16
        self.memory_bandwidth_tb_s = 4.8
        self.memory_bandwidth_bytes_s = 4.8e12
        self.memory_capacity_gb = 141
        self.tensor_core_efficiency_fp8 = 0.90  # 90% efficiency for fixed-size offline batching


class ParallelismConfig:
    """Parallelism configuration"""
    def __init__(self):
        self.tensor_parallel = 8
        self.expert_parallel = 8
        self.num_gpus = 8


class UtilizationModel:
    """Analytical model for MFU and MBU calculation"""
    
    def __init__(self):
        self.model_config = DeepSeekV3Config()
        self.hardware_config = H200Config()
        self.parallel_config = ParallelismConfig()
    
    def calculate_attention_flops_prefill(self, batch_size: int, seq_length: int) -> float:
        """Calculate FLOPs for attention during prefill phase"""
        cfg = self.model_config
        
        flops_per_layer = batch_size * seq_length * (
            # Q, K, V projections
            2 * cfg.hidden_size * cfg.q_dim +
            2 * cfg.hidden_size * cfg.k_dim +
            2 * cfg.hidden_size * cfg.v_dim +
            
            # RoPE application
            4 * cfg.num_attention_heads * cfg.qk_rope_head_dim * seq_length +
            
            # QK^T computation
            2 * cfg.num_attention_heads * seq_length * seq_length * 
            (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) +
            
            # Softmax (approximate as 5 ops per element)
            5 * cfg.num_attention_heads * seq_length * seq_length +
            
            # Attention × V
            2 * cfg.num_attention_heads * seq_length * seq_length * cfg.v_head_dim +
            
            # Output projection
            2 * cfg.v_dim * cfg.hidden_size
        )
        
        return cfg.num_layers * flops_per_layer
    
    def calculate_attention_flops_decode(self, batch_size: int, seq_length: int, 
                                       position: int) -> float:
        """Calculate FLOPs for attention during decode phase at given position"""
        cfg = self.model_config
        total_seq = seq_length + position
        
        flops_per_layer = batch_size * (
            # Q, K, V projections for new token
            2 * cfg.hidden_size * cfg.q_dim +
            2 * cfg.hidden_size * cfg.k_dim +
            2 * cfg.hidden_size * cfg.v_dim +
            
            # RoPE for new token
            4 * cfg.num_attention_heads * cfg.qk_rope_head_dim +
            
            # QK^T with all cached positions
            2 * cfg.num_attention_heads * total_seq * 
            (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) +
            
            # Softmax
            5 * cfg.num_attention_heads * total_seq +
            
            # Attention × V (with cached values)
            2 * cfg.num_attention_heads * total_seq * cfg.v_head_dim +
            
            # Output projection
            2 * cfg.v_dim * cfg.hidden_size
        )
        
        return cfg.num_layers * flops_per_layer
    
    def calculate_moe_flops_prefill(self, batch_size: int, seq_length: int) -> float:
        """Calculate FLOPs for MoE layers during prefill"""
        cfg = self.model_config
        
        flops_per_layer = batch_size * seq_length * (
            # Router computation
            2 * cfg.hidden_size * cfg.n_routed_experts +
            
            # Top-k selection (approximate)
            cfg.n_routed_experts * np.log2(cfg.n_routed_experts) +
            
            # Routed experts (only top-k active)
            cfg.experts_per_tok * (
                2 * cfg.hidden_size * cfg.expert_hidden_dim +  # Up projection
                2 * cfg.expert_hidden_dim +                     # Activation (SiLU)
                2 * cfg.expert_hidden_dim * cfg.hidden_size     # Down projection
            ) +
            
            # Shared expert (always active)
            2 * cfg.hidden_size * cfg.expert_hidden_dim +       # Up projection
            2 * cfg.expert_hidden_dim +                          # Activation
            2 * cfg.expert_hidden_dim * cfg.hidden_size         # Down projection
        )
        
        return cfg.num_layers * flops_per_layer
    
    def calculate_moe_flops_decode(self, batch_size: int) -> float:
        """Calculate FLOPs for MoE layers during decode (single token)"""
        return self.calculate_moe_flops_prefill(batch_size, 1)
    
    def calculate_weight_memory_bytes(self) -> float:
        """Calculate total weight memory access in bytes"""
        cfg = self.model_config
        bytes_per_param = 1  # FP8
        
        attention_weights_per_layer = (
            cfg.hidden_size * cfg.q_dim +           # Q projection
            cfg.hidden_size * cfg.k_dim +           # K projection
            cfg.hidden_size * cfg.v_dim +           # V projection
            cfg.v_dim * cfg.hidden_size             # Output projection
        ) * bytes_per_param
        
        # MoE weights (only active experts loaded due to EP)
        moe_weights_per_layer = (
            cfg.hidden_size * cfg.n_routed_experts +     # Router
            cfg.experts_per_tok * (
                cfg.hidden_size * cfg.expert_hidden_dim +
                cfg.expert_hidden_dim * cfg.hidden_size
            ) +
            # Shared expert
            cfg.hidden_size * cfg.expert_hidden_dim + 
            cfg.expert_hidden_dim * cfg.hidden_size
        ) * bytes_per_param
        
        # Layer norm weights
        norm_weights_per_layer = 2 * cfg.hidden_size * bytes_per_param
        
        # Embedding weights
        embedding_weights = cfg.vocab_size * cfg.hidden_size * bytes_per_param
        
        total_weights = (
            cfg.num_layers * (attention_weights_per_layer + moe_weights_per_layer + 
                             norm_weights_per_layer) +
            embedding_weights
        )
        
        return total_weights
    
    def calculate_kv_cache_memory(self, batch_size: int, seq_length: int) -> float:
        """Calculate KV cache memory in bytes"""
        cfg = self.model_config
        bytes_per_element = 1  # FP8
        
        # K cache: B × S × L × num_heads × qk_nope_dim
        k_cache_size = (batch_size * seq_length * cfg.num_layers * 
                       cfg.num_attention_heads * cfg.qk_nope_head_dim * bytes_per_element)
        
        # V cache: B × S × L × num_heads × v_head_dim
        v_cache_size = (batch_size * seq_length * cfg.num_layers * 
                       cfg.num_attention_heads * cfg.v_head_dim * bytes_per_element)
        
        return k_cache_size + v_cache_size
    
    def calculate_activation_memory(self, batch_size: int, seq_length: int) -> float:
        """Calculate activation memory during forward pass"""
        cfg = self.model_config
        bytes_per_element = 1  # FP8
        
        activation_memory_per_layer = batch_size * seq_length * (
            # Attention activations
            cfg.hidden_size +                              # Input
            cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) +  # Q
            cfg.num_attention_heads * cfg.qk_nope_head_dim +  # K
            cfg.num_attention_heads * cfg.v_head_dim +        # V
            cfg.num_attention_heads * seq_length +            # Attention scores
            cfg.num_attention_heads * cfg.v_head_dim +        # Attention output
            cfg.hidden_size +                                  # Projection output
            
            # MoE activations
            cfg.experts_per_tok * cfg.expert_hidden_dim +     # Active expert intermediates
            cfg.expert_hidden_dim +                           # Shared expert intermediate
            cfg.hidden_size                                   # MoE output
        ) * bytes_per_element
        
        return cfg.num_layers * activation_memory_per_layer
    
    def calculate_tp_communication(self, batch_size: int, seq_length: int) -> float:
        """Calculate tensor parallelism communication overhead"""
        cfg = self.model_config
        pcfg = self.parallel_config
        bytes_per_element = 1  # FP8
        
        # AllReduce needed after attention and MoE outputs
        comm_volume_per_layer = batch_size * seq_length * cfg.hidden_size * bytes_per_element * 3
        
        # Ring AllReduce: 2(P-1)/P × data_size
        allreduce_factor = 2 * (pcfg.tensor_parallel - 1) / pcfg.tensor_parallel
        
        return cfg.num_layers * comm_volume_per_layer * allreduce_factor
    
    def calculate_ep_communication(self, batch_size: int, seq_length: int) -> float:
        """Calculate expert parallelism communication overhead"""
        cfg = self.model_config
        bytes_per_element = 1  # FP8
        
        # AllToAll for expert routing (forward and backward)
        comm_volume_per_layer = 2 * batch_size * seq_length * cfg.hidden_size * bytes_per_element
        
        return cfg.num_layers * comm_volume_per_layer
    
    def calculate_flash_attention_factor(self, seq_length: int) -> float:
        """Calculate Flash Attention v2 memory reduction factor"""
        if seq_length <= 512:
            return 1.5
        elif seq_length <= 2048:
            return 2.0
        elif seq_length <= 8192:
            return 2.5
        else:
            return 3.0
    
    def calculate_kernel_fusion_benefits(self) -> Dict[str, float]:
        """Calculate benefits from vLLM kernel fusion"""
        return {
            'compute_efficiency': 0.98,  # 2% overhead reduction
            'memory_efficiency': 0.85    # 15% bandwidth reduction
        }
    
    def calculate_mfu_mbu(
        self,
        metrics: Dict[str, float],
        batch_size: int,
        actual_seq_lengths: List[int],
        max_seq_length: int,
        num_decode_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate MFU and MBU for offline batch inference
        
        Args:
            metrics: Dictionary containing timing metrics
            batch_size: Batch size
            actual_seq_lengths: List of actual sequence lengths before padding
            max_seq_length: Maximum sequence length (padded)
            num_decode_tokens: Number of tokens to generate
            
        Returns:
            Dictionary with MFU/MBU metrics and detailed breakdown
        """
        # Extract timing
        prefill_time = metrics['prefill_time']
        decode_time = metrics['decode_time']
        
        # Calculate padding overhead
        avg_actual_length = np.mean(actual_seq_lengths)
        padding_overhead = max_seq_length / avg_actual_length
        effective_token_ratio = sum(actual_seq_lengths) / (batch_size * max_seq_length)
        
        # Calculate FLOPs (on padded sequences)
        prefill_flops = (
            self.calculate_attention_flops_prefill(batch_size, max_seq_length) +
            self.calculate_moe_flops_prefill(batch_size, max_seq_length)
        )
        
        # Decode FLOPs
        total_decode_flops = 0
        for pos in range(num_decode_tokens):
            total_decode_flops += (
                self.calculate_attention_flops_decode(batch_size, max_seq_length, pos) +
                self.calculate_moe_flops_decode(batch_size)
            )
        
        # Memory calculations
        weight_memory = self.calculate_weight_memory_bytes()
        
        # Get optimization factors
        flash_attn_factor = self.calculate_flash_attention_factor(max_seq_length)
        kernel_fusion = self.calculate_kernel_fusion_benefits()
        
        # Prefill memory
        prefill_memory_base = (
            weight_memory +
            self.calculate_activation_memory(batch_size, max_seq_length) +
            self.calculate_kv_cache_memory(batch_size, max_seq_length) +
            self.calculate_tp_communication(batch_size, max_seq_length) +
            self.calculate_ep_communication(batch_size, max_seq_length)
        )
        
        # Apply Flash Attention optimization
        attention_memory_fraction = 0.4
        prefill_memory = (
            prefill_memory_base * (1 - attention_memory_fraction) +
            prefill_memory_base * attention_memory_fraction / flash_attn_factor
        )
        
        # Decode memory
        total_decode_memory = 0
        for pos in range(num_decode_tokens):
            decode_memory_per_token = (
                weight_memory +
                self.calculate_kv_cache_memory(batch_size, max_seq_length + pos) * 0.5 +  # Read partial
                self.calculate_activation_memory(batch_size, 1) +
                self.calculate_tp_communication(batch_size, 1) +
                self.calculate_ep_communication(batch_size, 1)
            )
            
            # Apply Flash Attention (reduced impact in decode)
            decode_memory_per_token = (
                decode_memory_per_token * (1 - attention_memory_fraction * 0.5) +
                decode_memory_per_token * attention_memory_fraction * 0.5 / flash_attn_factor
            )
            
            total_decode_memory += decode_memory_per_token
        
        # Apply kernel fusion
        prefill_flops *= kernel_fusion['compute_efficiency']
        total_decode_flops *= kernel_fusion['compute_efficiency']
        prefill_memory *= kernel_fusion['memory_efficiency']
        total_decode_memory *= kernel_fusion['memory_efficiency']
        
        # Calculate utilization
        hcfg = self.hardware_config
        pcfg = self.parallel_config
        
        effective_flops_per_gpu = hcfg.peak_fp8_tflops * hcfg.tensor_core_efficiency_fp8
        batch_efficiency = 0.95  # High efficiency for offline batching
        
        # Raw utilization
        mfu_prefill_raw = (prefill_flops * batch_efficiency) / \
                         (effective_flops_per_gpu * pcfg.num_gpus * prefill_time)
        
        mbu_prefill_raw = (prefill_memory * batch_efficiency) / \
                         (hcfg.memory_bandwidth_bytes_s * pcfg.num_gpus * prefill_time)
        
        mfu_decode_raw = (total_decode_flops * batch_efficiency) / \
                        (effective_flops_per_gpu * pcfg.num_gpus * decode_time)
        
        mbu_decode_raw = (total_decode_memory * batch_efficiency) / \
                        (hcfg.memory_bandwidth_bytes_s * pcfg.num_gpus * decode_time)
        
        # Adjust for padding waste (effective utilization)
        mfu_prefill = mfu_prefill_raw * effective_token_ratio
        mbu_prefill = mbu_prefill_raw * effective_token_ratio
        mfu_decode = mfu_decode_raw * effective_token_ratio
        mbu_decode = mbu_decode_raw * effective_token_ratio
        
        # Total utilization
        total_time = prefill_time + decode_time
        total_flops = prefill_flops + total_decode_flops
        total_memory = prefill_memory + total_decode_memory
        
        mfu_total = (total_flops * batch_efficiency * effective_token_ratio) / \
                    (effective_flops_per_gpu * pcfg.num_gpus * total_time)
        
        mbu_total = (total_memory * batch_efficiency * effective_token_ratio) / \
                    (hcfg.memory_bandwidth_bytes_s * pcfg.num_gpus * total_time)
        
        return {
            # Primary metrics
            'mfu_prefill': mfu_prefill,
            'mbu_prefill': mbu_prefill,
            'mfu_decode': mfu_decode,
            'mbu_decode': mbu_decode,
            'mfu_total': mfu_total,
            'mbu_total': mbu_total,
            
            # Raw utilization (before padding adjustment)
            'mfu_prefill_raw': mfu_prefill_raw,
            'mbu_prefill_raw': mbu_prefill_raw,
            'mfu_decode_raw': mfu_decode_raw,
            'mbu_decode_raw': mbu_decode_raw,
            
            # Detailed breakdown
            'details': {
                'padding_overhead': padding_overhead,
                'effective_token_ratio': effective_token_ratio,
                'tensor_core_efficiency': hcfg.tensor_core_efficiency_fp8,
                'flash_attention_factor': flash_attn_factor,
                'batch_efficiency': batch_efficiency,
                'kernel_fusion': kernel_fusion,
                'theoretical_flops': {
                    'prefill': prefill_flops,
                    'decode': total_decode_flops,
                    'total': total_flops
                },
                'memory_bandwidth_bytes': {
                    'prefill': prefill_memory,
                    'decode': total_decode_memory,
                    'total': total_memory
                },
                'hardware': {
                    'peak_fp8_tflops_total': hcfg.peak_fp8_tflops * pcfg.num_gpus / 1e12,
                    'peak_bandwidth_tb_s_total': hcfg.memory_bandwidth_tb_s * pcfg.num_gpus,
                    'effective_fp8_tflops_total': effective_flops_per_gpu * pcfg.num_gpus / 1e12
                }
            }
        }


def main():
    """Example usage of the analytical model"""
    model = UtilizationModel()
    
    # Example metrics from vLLM benchmark
    metrics = {
        'prefill_time': 0.5,      # seconds
        'decode_time': 2.0,       # seconds
        'time_per_output_token': 0.02  # seconds
    }
    
    # Example batch configuration
    batch_size = 32
    actual_seq_lengths = [512, 768, 1024, 1536] * 8  # 32 sequences
    max_seq_length = 2048  # All padded to this length
    num_decode_tokens = 128
    
    # Calculate MFU and MBU
    results = model.calculate_mfu_mbu(
        metrics=metrics,
        batch_size=batch_size,
        actual_seq_lengths=actual_seq_lengths,
        max_seq_length=max_seq_length,
        num_decode_tokens=num_decode_tokens
    )
    
    # Print results
    print("=== DeepSeek-V3 MFU/MBU Analysis ===\n")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Average actual length: {np.mean(actual_seq_lengths):.1f}")
    print(f"Padding overhead: {results['details']['padding_overhead']:.2f}x")
    print(f"Effective token ratio: {results['details']['effective_token_ratio']:.2%}\n")
    
    print("=== Utilization Metrics ===")
    print(f"MFU Prefill: {results['mfu_prefill']:.1%} (raw: {results['mfu_prefill_raw']:.1%})")
    print(f"MBU Prefill: {results['mbu_prefill']:.1%} (raw: {results['mbu_prefill_raw']:.1%})")
    print(f"MFU Decode:  {results['mfu_decode']:.1%} (raw: {results['mfu_decode_raw']:.1%})")
    print(f"MBU Decode:  {results['mbu_decode']:.1%} (raw: {results['mbu_decode_raw']:.1%})")
    print(f"MFU Total:   {results['mfu_total']:.1%}")
    print(f"MBU Total:   {results['mbu_total']:.1%}\n")
    
    print("=== Hardware Capacity ===")
    hw = results['details']['hardware']
    print(f"Peak FP8 TFLOPS (8 GPUs): {hw['peak_fp8_tflops_total']:.1f}")
    print(f"Effective FP8 TFLOPS: {hw['effective_fp8_tflops_total']:.1f}")
    print(f"Peak Memory BW (8 GPUs): {hw['peak_bandwidth_tb_s_total']:.1f} TB/s\n")
    
    print("=== Optimization Impact ===")
    print(f"Flash Attention factor: {results['details']['flash_attention_factor']:.1f}x")
    print(f"Tensor Core efficiency: {results['details']['tensor_core_efficiency']:.0%}")
    print(f"Batch efficiency: {results['details']['batch_efficiency']:.0%}")
    print(f"Kernel fusion - compute: {results['details']['kernel_fusion']['compute_efficiency']:.0%}")
    print(f"Kernel fusion - memory: {results['details']['kernel_fusion']['memory_efficiency']:.0%}")


if __name__ == "__main__":
    main()
