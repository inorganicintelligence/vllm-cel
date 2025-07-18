"""Base model class for analytical performance modeling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    total_flops: float
    prefill_flops: float
    decode_flops: float
    total_memory_gb: float
    prefill_memory_gb: float
    decode_memory_gb: float
    communication_overhead_flops: float = 0.0


@dataclass
class ParallelismConfig:
    """Configuration for model parallelism."""
    tp_size: int = 1
    ep_enabled: bool = False
    dp_size: int = 1
    use_flash_attention: bool = True


class BaseModel(ABC):
    """Abstract base class for model specifications.
    
    This class defines the interface for model specifications used in
    analytical performance modeling. Subclasses should implement specific
    model architectures (Mixtral, DeepSeek, etc.).
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, any]:
        """Get model parameters dictionary.
        
        Returns:
            Dictionary containing model parameters like hidden_size,
            num_layers, num_experts, etc.
        """
        pass
    
    @abstractmethod
    def calculate_total_flops_per_gpu(
        self, 
        seq_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total FLOPs per GPU for the entire sequence.
        
        Args:
            seq_len: Total sequence length (input + output)
            batch_size: Batch size
            parallelism_config: Parallelism configuration
            
        Returns:
            Total FLOPs per GPU in TFLOPS
        """
        pass
    
    @abstractmethod
    def calculate_prefill_flops_per_gpu(
        self, 
        input_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate prefill phase FLOPs per GPU.
        
        Args:
            input_len: Input sequence length
            batch_size: Batch size
            parallelism_config: Parallelism configuration
            
        Returns:
            Prefill FLOPs per GPU in TFLOPS
        """
        pass
    
    @abstractmethod
    def calculate_decode_flops_per_gpu(
        self, 
        output_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate decode phase FLOPs per GPU.
        
        Args:
            output_len: Output sequence length
            batch_size: Batch size
            parallelism_config: Parallelism configuration
            
        Returns:
            Decode FLOPs per GPU in TFLOPS
        """
        pass
    
    @abstractmethod
    def calculate_total_memory_per_gpu(
        self, 
        seq_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total memory access per GPU.
        
        Args:
            seq_len: Total sequence length
            batch_size: Batch size
            parallelism_config: Parallelism configuration
            
        Returns:
            Total memory access per GPU in GB
        """
        pass
    
    @abstractmethod
    def calculate_prefill_memory_per_gpu(
        self, 
        input_len: int, 
        batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate prefill phase memory access per GPU.
        
        Args:
            input_len: Input sequence length
            batch_size: Batch size
            parallelism_config: Parallelism configuration
            
        Returns:
            Prefill memory access per GPU in GB
        """
        pass
    
    @abstractmethod
    def calculate_decode_memory_per_gpu(
        self, 
        output_len: int, 
        batch_size: int, 
        kv_cache_size: int,
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate decode phase memory access per GPU.
        
        Args:
            output_len: Output sequence length  
            batch_size: Batch size
            kv_cache_size: KV cache size in tokens
            parallelism_config: Parallelism configuration
            
        Returns:
            Decode memory access per GPU in GB
        """
        pass
    
    def calculate_all_metrics(
        self,
        input_len: int,
        output_len: int,
        batch_size: int,
        parallelism_config: ParallelismConfig
    ) -> ModelMetrics:
        """Calculate all model metrics for a given configuration.
        
        Args:
            input_len: Input sequence length
            output_len: Output sequence length
            batch_size: Batch size
            parallelism_config: Parallelism configuration
            
        Returns:
            ModelMetrics object containing all calculated metrics
        """
        total_seq_len = input_len + output_len
        kv_cache_size = input_len + output_len
        
        return ModelMetrics(
            total_flops=self.calculate_total_flops_per_gpu(
                total_seq_len, batch_size, parallelism_config
            ),
            prefill_flops=self.calculate_prefill_flops_per_gpu(
                input_len, batch_size, parallelism_config
            ),
            decode_flops=self.calculate_decode_flops_per_gpu(
                output_len, batch_size, parallelism_config
            ),
            total_memory_gb=self.calculate_total_memory_per_gpu(
                total_seq_len, batch_size, parallelism_config
            ),
            prefill_memory_gb=self.calculate_prefill_memory_per_gpu(
                input_len, batch_size, parallelism_config
            ),
            decode_memory_gb=self.calculate_decode_memory_per_gpu(
                output_len, batch_size, kv_cache_size, parallelism_config
            ),
        )
    
    def _calculate_attention_flops(
        self, 
        seq_len: int, 
        batch_size: int, 
        hidden_size: int, 
        num_heads: int, 
        head_dim: int,
        tp_size: int = 1,
        use_flash_attention: bool = True
    ) -> float:
        """Helper method to calculate attention FLOPs.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Head dimension
            tp_size: Tensor parallel size
            use_flash_attention: Whether Flash Attention is used
            
        Returns:
            Attention FLOPs per GPU in TFLOPS
        """
        # QKV projection: 3 * 2 * seq_len * hidden * hidden
        qkv_flops = 3 * 2 * seq_len * batch_size * hidden_size * hidden_size
        
        # Attention computation: 2 * seq_len^2 * num_heads * head_dim
        attn_flops = 2 * seq_len * seq_len * batch_size * num_heads * head_dim
        
        # Output projection: 2 * seq_len * hidden * hidden  
        out_flops = 2 * seq_len * batch_size * hidden_size * hidden_size
        
        total_flops = qkv_flops + attn_flops + out_flops
        
        # Adjust for tensor parallelism
        total_flops = total_flops / tp_size
        
        # Flash Attention reduces memory but not FLOPs
        return total_flops / 1e12  # Convert to TFLOPS
    
    def _calculate_moe_flops(
        self,
        seq_len: int,
        batch_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        tp_size: int = 1,
        ep_enabled: bool = False
    ) -> float:
        """Helper method to calculate MoE FLOPs.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            hidden_size: Hidden dimension
            intermediate_size: Intermediate dimension
            num_experts: Total number of experts
            experts_per_token: Number of experts per token (top-k)
            tp_size: Tensor parallel size
            ep_enabled: Whether expert parallelism is enabled
            
        Returns:
            MoE FLOPs per GPU in TFLOPS
        """
        # Router computation: 2 * seq_len * hidden * num_experts
        router_flops = 2 * seq_len * batch_size * hidden_size * num_experts
        
        # Expert computation: experts_per_token * 2 * seq_len * (2 * hidden * intermediate + intermediate * hidden)
        # Gate and up projections + down projection
        expert_flops = (
            experts_per_token * 2 * seq_len * batch_size * 
            (2 * hidden_size * intermediate_size + intermediate_size * hidden_size)
        )
        
        total_flops = router_flops + expert_flops
        
        if ep_enabled:
            # With EP, each GPU processes fewer tokens but full expert computation
            # Assuming uniform routing, each GPU gets total_tokens / tp_size
            total_flops = total_flops / tp_size
        else:
            # Without EP, expert weights are sharded across TP dimension
            total_flops = total_flops / tp_size
        
        return total_flops / 1e12  # Convert to TFLOPS