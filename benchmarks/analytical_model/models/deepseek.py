"""DeepSeek V3 model implementation for analytical modeling."""

from typing import Dict
from .base import BaseModel, ParallelismConfig


class DeepSeekV3(BaseModel):
    """DeepSeek V3 model specification with Multi-Latent Attention (MLA)."""
    
    def __init__(self):
        super().__init__("DeepSeek-V3")
        
        # DeepSeek V3 model parameters
        self.hidden_size = 7168
        self.intermediate_size = 18432
        self.num_layers = 61
        self.num_attention_heads = 128
        self.num_key_value_heads = 128  # No GQA in DeepSeek V3
        self.head_dim = 56  # 7168 / 128
        self.num_experts = 256
        self.experts_per_token = 8  # Top-8 routing  
        self.vocab_size = 129024
        self.max_position_embeddings = 65536
        
        # MLA (Multi-Latent Attention) parameters
        self.q_lora_rank = 1536  # Lq - latent Q dimension
        self.kv_lora_rank = 512  # Lkv - latent KV dimension
        self.qk_nope_head_dim = 128  # P - nope dimension
        self.qk_rope_head_dim = 64   # R - rope dimension
        self.v_head_dim = 128        # V - value head dimension
        
        # Memory parameters (FP16)
        self.dtype_size_bytes = 2
    
    def get_model_params(self) -> Dict[str, any]:
        """Get DeepSeek V3 model parameters."""
        return {
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'head_dim': self.head_dim,
            'num_experts': self.num_experts,
            'experts_per_token': self.experts_per_token,
            'vocab_size': self.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'q_lora_rank': self.q_lora_rank,
            'kv_lora_rank': self.kv_lora_rank,
            'qk_nope_head_dim': self.qk_nope_head_dim,
            'qk_rope_head_dim': self.qk_rope_head_dim,
            'v_head_dim': self.v_head_dim,
        }
    
    def calculate_total_flops_per_gpu(
        self, 
        seq_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total FLOPs per GPU for DeepSeek V3."""
        total_flops = 0.0
        
        # MLA Attention FLOPs per layer
        mla_flops = self._calculate_mla_attention_flops(
            seq_len=seq_len,
            batch_size=batch_size,
            tp_size=parallelism_config.tp_size
        )
        
        # MoE FLOPs per layer (256 experts)
        moe_flops = self._calculate_moe_flops(
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
            tp_size=parallelism_config.tp_size,
            ep_enabled=parallelism_config.ep_enabled
        )
        
        # Total FLOPs across all layers
        total_flops = (mla_flops + moe_flops) * self.num_layers
        
        # Add embedding and output layer FLOPs
        embedding_flops = 2 * seq_len * batch_size * self.hidden_size * self.vocab_size / parallelism_config.tp_size
        total_flops += embedding_flops / 1e12
        
        return total_flops
    
    def calculate_prefill_flops_per_gpu(
        self, 
        input_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate prefill FLOPs per GPU for DeepSeek V3."""
        return self.calculate_total_flops_per_gpu(input_len, batch_size, parallelism_config)
    
    def calculate_decode_flops_per_gpu(
        self, 
        output_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate decode FLOPs per GPU for DeepSeek V3."""
        single_token_flops = self.calculate_total_flops_per_gpu(1, batch_size, parallelism_config)
        return single_token_flops * output_len
    
    def calculate_total_memory_per_gpu(
        self, 
        seq_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total memory access per GPU for DeepSeek V3."""
        memory_gb = 0.0
        
        # Model weights memory
        memory_gb += self._calculate_weights_memory_per_gpu(parallelism_config)
        
        # Activations memory
        memory_gb += self._calculate_activations_memory(seq_len, batch_size)
        
        # Compressed KV cache memory (MLA benefit)
        memory_gb += self._calculate_compressed_kv_cache_memory(seq_len, batch_size, parallelism_config.tp_size)
        
        return memory_gb
    
    def calculate_prefill_memory_per_gpu(
        self, 
        input_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate prefill memory access per GPU for DeepSeek V3."""
        memory_gb = 0.0
        
        # Model weights (loaded once during prefill)
        memory_gb += self._calculate_weights_memory_per_gpu(parallelism_config)
        
        # Prefill activations
        memory_gb += self._calculate_activations_memory(input_len, batch_size)
        
        # Compressed KV cache writes during prefill
        memory_gb += self._calculate_compressed_kv_cache_memory(input_len, batch_size, parallelism_config.tp_size)
        
        return memory_gb
    
    def calculate_decode_memory_per_gpu(
        self, 
        output_len: int, 
        batch_size: int, 
        kv_cache_size: int,
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate decode memory access per GPU for DeepSeek V3."""
        memory_gb = 0.0
        
        # Model weights (loaded once per decode step)
        weights_per_step = self._calculate_weights_memory_per_gpu(parallelism_config)
        memory_gb += weights_per_step * output_len
        
        # Decode activations (per step)
        decode_activations_per_step = self._calculate_activations_memory(1, batch_size)
        memory_gb += decode_activations_per_step * output_len
        
        # Compressed KV cache reads/writes during decode
        kv_memory_per_step = self._calculate_compressed_kv_cache_memory(kv_cache_size, batch_size, parallelism_config.tp_size)
        memory_gb += kv_memory_per_step * output_len
        
        return memory_gb
    
    def _calculate_mla_attention_flops(self, seq_len: int, batch_size: int, tp_size: int) -> float:
        """Calculate Multi-Latent Attention FLOPs per layer."""
        flops = 0.0
        
        # Q compression: hidden -> q_lora_rank (replicated)
        q_compression_flops = 2 * seq_len * batch_size * self.hidden_size * self.q_lora_rank
        
        # Q expansion: q_lora_rank -> num_heads * head_dim (sharded by TP)
        q_expansion_flops = 2 * seq_len * batch_size * self.q_lora_rank * self.num_attention_heads * self.head_dim / tp_size
        
        # KV compression: hidden -> kv_lora_rank + rope_dim (replicated)
        kv_compression_flops = 2 * seq_len * batch_size * self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
        
        # KV expansion: kv_lora_rank -> num_heads * (nope_dim + v_dim) (sharded by TP)
        kv_expansion_flops = 2 * seq_len * batch_size * self.kv_lora_rank * self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim) / tp_size
        
        # Attention computation: with compressed representations
        # Q * K^T: seq_len^2 * num_heads * head_dim
        qk_flops = 2 * seq_len * seq_len * batch_size * self.num_attention_heads * self.head_dim / tp_size
        
        # Attention * V: seq_len^2 * num_heads * v_head_dim  
        av_flops = 2 * seq_len * seq_len * batch_size * self.num_attention_heads * self.v_head_dim / tp_size
        
        # Output projection: num_heads * v_head_dim -> hidden (sharded by TP)
        out_flops = 2 * seq_len * batch_size * self.num_attention_heads * self.v_head_dim * self.hidden_size / tp_size
        
        total_flops = (q_compression_flops + q_expansion_flops + kv_compression_flops + 
                      kv_expansion_flops + qk_flops + av_flops + out_flops)
        
        return total_flops / 1e12  # Convert to TFLOPS
    
    def _calculate_weights_memory_per_gpu(self, parallelism_config: ParallelismConfig) -> float:
        """Calculate model weights memory per GPU in GB."""
        memory_bytes = 0
        
        # MLA attention weights per layer
        # Q compression (replicated) + Q expansion (sharded)
        q_weights = (
            self.hidden_size * self.q_lora_rank +  # q_a_proj (replicated)
            self.q_lora_rank * self.num_attention_heads * self.head_dim // parallelism_config.tp_size  # q_b_proj (sharded)
        )
        
        # KV compression (replicated) + KV expansion (sharded)
        kv_weights = (
            self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim) +  # kv_a_proj (replicated)
            self.kv_lora_rank * self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim) // parallelism_config.tp_size  # kv_b_proj (sharded)
        )
        
        # Output projection (sharded)
        o_weights = self.num_attention_heads * self.v_head_dim * self.hidden_size // parallelism_config.tp_size
        
        mla_weights = q_weights + kv_weights + o_weights
        
        # MoE weights per layer (256 experts)
        if parallelism_config.ep_enabled:
            # With EP, each GPU stores complete experts for its local experts
            experts_per_gpu = self.num_experts // parallelism_config.tp_size
            moe_weights = (
                experts_per_gpu * 3 * self.hidden_size * self.intermediate_size +  # gate, up, down
                self.hidden_size * self.num_experts  # router
            )
        else:
            # Without EP, weights are sharded across TP
            moe_weights = (
                self.num_experts * 3 * self.hidden_size * self.intermediate_size +  # gate, up, down  
                self.hidden_size * self.num_experts  # router
            ) // parallelism_config.tp_size
        
        # Layer norm weights (small, replicated)
        layernorm_weights = 3 * self.hidden_size  # attention + ffn layer norms + q_a_layernorm, kv_a_layernorm
        
        # Total weights per layer
        weights_per_layer = mla_weights + moe_weights + layernorm_weights
        
        # Total model weights
        total_weights = weights_per_layer * self.num_layers
        
        # Add embedding weights
        embedding_weights = self.vocab_size * self.hidden_size // parallelism_config.tp_size
        total_weights += embedding_weights
        
        # Convert to GB
        memory_bytes = total_weights * self.dtype_size_bytes
        return memory_bytes / (1024**3)
    
    def _calculate_activations_memory(self, seq_len: int, batch_size: int) -> float:
        """Calculate activations memory in GB."""
        # Activation memory for forward pass with MLA
        activations_per_token = (
            self.hidden_size +  # input hidden state
            self.q_lora_rank + self.kv_lora_rank +  # compressed representations
            self.num_attention_heads * self.v_head_dim +  # attention output
            self.experts_per_token * self.intermediate_size +  # MoE intermediate
            self.hidden_size  # final hidden state
        )
        
        total_activations = seq_len * batch_size * activations_per_token * self.num_layers
        memory_bytes = total_activations * self.dtype_size_bytes
        return memory_bytes / (1024**3)
    
    def _calculate_compressed_kv_cache_memory(self, seq_len: int, batch_size: int, tp_size: int) -> float:
        """Calculate compressed KV cache memory in GB.
        
        MLA uses compressed KV representations, significantly reducing memory usage.
        """
        # Compressed KV cache: stores latent representations instead of full KV
        # Format: Lkv + R (latent KV + rope dimension)
        compressed_kv_dim = self.kv_lora_rank + self.qk_rope_head_dim
        
        kv_cache_per_token = (
            compressed_kv_dim *  # Compressed representation  
            self.num_layers      # All layers
        )
        
        # Adjust for tensor parallelism (cache is typically replicated, but let's assume some sharding)
        kv_cache_per_token_per_gpu = kv_cache_per_token // tp_size
        
        total_kv_cache = seq_len * batch_size * kv_cache_per_token_per_gpu
        memory_bytes = total_kv_cache * self.dtype_size_bytes
        return memory_bytes / (1024**3)