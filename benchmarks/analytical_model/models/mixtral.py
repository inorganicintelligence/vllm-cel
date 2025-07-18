"""Mixtral 8x7B model implementation for analytical modeling."""

from typing import Dict
from .base import BaseModel, ParallelismConfig


class Mixtral8x7B(BaseModel):
    """Mixtral 8x7B model specification."""
    
    def __init__(self):
        super().__init__("Mixtral-8x7B")
        
        # Mixtral 8x7B model parameters
        self.hidden_size = 4096
        self.intermediate_size = 14336
        self.num_layers = 32
        self.num_attention_heads = 32
        self.num_key_value_heads = 8  # GQA
        self.head_dim = 128
        self.num_experts = 8
        self.experts_per_token = 2  # Top-2 routing
        self.vocab_size = 32000
        self.max_position_embeddings = 32768
        
        # Memory parameters (FP16)
        self.dtype_size_bytes = 2
    
    def get_model_params(self) -> Dict[str, any]:
        """Get Mixtral model parameters."""
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
        }
    
    def calculate_total_flops_per_gpu(
        self, 
        seq_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total FLOPs per GPU for Mixtral."""
        total_flops = 0.0
        
        # Attention FLOPs per layer
        attention_flops = self._calculate_attention_flops(
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            tp_size=parallelism_config.tp_size,
            use_flash_attention=parallelism_config.use_flash_attention
        )
        
        # MoE FLOPs per layer
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
        
        # Total FLOPs across all layers (attention_flops and moe_flops are already in TFLOPS)
        total_flops = (attention_flops + moe_flops) * self.num_layers
        
        # Add embedding and output layer FLOPs (minimal compared to main layers)
        embedding_flops = 2 * seq_len * batch_size * self.hidden_size * self.vocab_size / parallelism_config.tp_size
        total_flops += embedding_flops / 1e12  # Convert to TFLOPS
        
        return total_flops
    
    def calculate_prefill_flops_per_gpu(
        self, 
        input_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate prefill FLOPs per GPU for Mixtral."""
        return self.calculate_total_flops_per_gpu(input_len, batch_size, parallelism_config)
    
    def calculate_decode_flops_per_gpu(
        self, 
        output_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate decode FLOPs per GPU for Mixtral.
        
        Note: For decode, each step processes 1 token, so we multiply by output_len.
        """
        single_token_flops = self.calculate_total_flops_per_gpu(1, batch_size, parallelism_config)
        return single_token_flops * output_len
    
    def calculate_total_memory_per_gpu(
        self, 
        seq_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total memory access per GPU for Mixtral."""
        memory_gb = 0.0
        
        # Model weights memory
        memory_gb += self._calculate_weights_memory_per_gpu(parallelism_config)
        
        # Activations memory
        memory_gb += self._calculate_activations_memory(seq_len, batch_size)
        
        # KV cache memory
        memory_gb += self._calculate_kv_cache_memory(seq_len, batch_size, parallelism_config.tp_size)
        
        return memory_gb
    
    def calculate_prefill_memory_per_gpu(
        self, 
        input_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate prefill memory access per GPU for Mixtral."""
        memory_gb = 0.0
        
        # Model weights (loaded once during prefill)
        memory_gb += self._calculate_weights_memory_per_gpu(parallelism_config)
        
        # Prefill activations
        memory_gb += self._calculate_activations_memory(input_len, batch_size)
        
        # KV cache writes during prefill
        memory_gb += self._calculate_kv_cache_memory(input_len, batch_size, parallelism_config.tp_size)
        
        return memory_gb
    
    def calculate_decode_memory_per_gpu(
        self, 
        output_len: int, 
        batch_size: int, 
        kv_cache_size: int,
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate decode memory access per GPU for Mixtral."""
        memory_gb = 0.0
        
        # Model weights (loaded once per decode step)
        weights_per_step = self._calculate_weights_memory_per_gpu(parallelism_config)
        memory_gb += weights_per_step * output_len
        
        # Decode activations (per step)
        decode_activations_per_step = self._calculate_activations_memory(1, batch_size)
        memory_gb += decode_activations_per_step * output_len
        
        # KV cache reads/writes during decode
        kv_memory_per_step = self._calculate_kv_cache_memory(kv_cache_size, batch_size, parallelism_config.tp_size)
        memory_gb += kv_memory_per_step * output_len
        
        return memory_gb
    
    def _calculate_weights_memory_per_gpu(self, parallelism_config: ParallelismConfig) -> float:
        """Calculate model weights memory per GPU in GB."""
        memory_bytes = 0
        
        # Attention weights per layer
        # Q, K, V projections + output projection
        attn_weights = (
            self.hidden_size * self.num_attention_heads * self.head_dim +  # Q
            self.hidden_size * self.num_key_value_heads * self.head_dim +   # K  
            self.hidden_size * self.num_key_value_heads * self.head_dim +   # V
            self.num_attention_heads * self.head_dim * self.hidden_size     # O
        )
        
        # MoE weights per layer
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
        layernorm_weights = 2 * self.hidden_size  # attention + ffn layer norms
        
        # Total weights per layer
        weights_per_layer = attn_weights // parallelism_config.tp_size + moe_weights + layernorm_weights
        
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
        # Activation memory for forward pass
        # Main contributors: hidden states, attention outputs, MoE outputs
        activations_per_token = (
            self.hidden_size +  # input hidden state
            self.num_attention_heads * self.head_dim +  # attention output
            self.experts_per_token * self.intermediate_size +  # MoE intermediate
            self.hidden_size  # final hidden state
        )
        
        total_activations = seq_len * batch_size * activations_per_token * self.num_layers
        memory_bytes = total_activations * self.dtype_size_bytes
        return memory_bytes / (1024**3)
    
    def _calculate_kv_cache_memory(self, seq_len: int, batch_size: int, tp_size: int) -> float:
        """Calculate KV cache memory in GB."""
        # KV cache: K and V for each layer
        kv_cache_per_token = (
            2 *  # K and V
            self.num_key_value_heads * self.head_dim *  # GQA: fewer KV heads
            self.num_layers
        )
        
        # Adjust for tensor parallelism
        kv_cache_per_token_per_gpu = kv_cache_per_token // tp_size
        
        total_kv_cache = seq_len * batch_size * kv_cache_per_token_per_gpu
        memory_bytes = total_kv_cache * self.dtype_size_bytes
        return memory_bytes / (1024**3)