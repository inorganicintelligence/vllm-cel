"""NVIDIA GPU hardware specifications for analytical modeling."""

from .base import BaseHardware


class A100Hardware(BaseHardware):
    """NVIDIA A100 hardware specification."""
    
    def __init__(self):
        super().__init__(
            name="A100",
            peak_memory_bandwidth_gb_s=2039.0,  # HBM2e
            memory_size_gb=80.0,
            tensor_core_version="3rd Gen",
            base_clock_ghz=1.41,
            memory_clock_ghz=1.2,
            memory_bus_width_bits=5120
        )
    
    def get_peak_flops(self, dtype: str) -> float:
        """Get peak FLOPS for A100.
        
        Peak FLOPS values for A100 with Tensor Cores:
        - FP16: 312 TFLOPS
        - BF16: 312 TFLOPS  
        - TF32: 156 TFLOPS
        - FP32: 19.5 TFLOPS
        """
        self.validate_dtype(dtype)
        
        flops_map = {
            'fp16': 312.0,
            'bf16': 312.0,
            'tf32': 156.0,
            'fp32': 19.5,
        }
        return flops_map[dtype]
    
    def get_supported_dtypes(self) -> list[str]:
        return ['fp16', 'bf16', 'tf32', 'fp32']


class H100Hardware(BaseHardware):
    """NVIDIA H100 hardware specification."""
    
    def __init__(self):
        super().__init__(
            name="H100",
            peak_memory_bandwidth_gb_s=3350.0,  # HBM3
            memory_size_gb=80.0,
            tensor_core_version="4th Gen",
            base_clock_ghz=1.98,
            memory_clock_ghz=2.62,
            memory_bus_width_bits=5120
        )
    
    def get_peak_flops(self, dtype: str) -> float:
        """Get peak FLOPS for H100.
        
        Peak FLOPS values for H100 with 4th Gen Tensor Cores:
        - FP8: 1979 TFLOPS (E4M3/E5M2)
        - FP16: 989 TFLOPS
        - BF16: 989 TFLOPS
        - TF32: 494 TFLOPS
        - FP32: 67 TFLOPS
        """
        self.validate_dtype(dtype)
        
        flops_map = {
            'fp8': 1979.0,
            'fp16': 989.0,
            'bf16': 989.0,
            'tf32': 494.0,
            'fp32': 67.0,
        }
        return flops_map[dtype]
    
    def get_supported_dtypes(self) -> list[str]:
        return ['fp8', 'fp16', 'bf16', 'tf32', 'fp32']


class H200Hardware(BaseHardware):
    """NVIDIA H200 hardware specification."""
    
    def __init__(self):
        super().__init__(
            name="H200",
            peak_memory_bandwidth_gb_s=4800.0,  # HBM3e
            memory_size_gb=141.0,
            tensor_core_version="4th Gen",
            base_clock_ghz=1.98,
            memory_clock_ghz=3.2,
            memory_bus_width_bits=5120
        )
    
    def get_peak_flops(self, dtype: str) -> float:
        """Get peak FLOPS for H200.
        
        Peak FLOPS values for H200 (similar to H100 with enhanced memory):
        - FP8: 1979 TFLOPS (E4M3/E5M2)
        - FP16: 989 TFLOPS
        - BF16: 989 TFLOPS
        - TF32: 494 TFLOPS
        - FP32: 67 TFLOPS
        """
        self.validate_dtype(dtype)
        
        flops_map = {
            'fp8': 1979.0,
            'fp16': 989.0,
            'bf16': 989.0,
            'tf32': 494.0,
            'fp32': 67.0,
        }
        return flops_map[dtype]
    
    def get_supported_dtypes(self) -> list[str]:
        return ['fp8', 'fp16', 'bf16', 'tf32', 'fp32']


class B200Hardware(BaseHardware):
    """NVIDIA B200 hardware specification (Blackwell)."""
    
    def __init__(self):
        super().__init__(
            name="B200",
            peak_memory_bandwidth_gb_s=8000.0,  # HBM3e (estimated)
            memory_size_gb=192.0,  # Estimated
            tensor_core_version="5th Gen",
            base_clock_ghz=2.2,  # Estimated
            memory_clock_ghz=4.0,  # Estimated
            memory_bus_width_bits=6144  # Estimated wider bus
        )
    
    def get_peak_flops(self, dtype: str) -> float:
        """Get peak FLOPS for B200.
        
        Estimated peak FLOPS values for B200 with 5th Gen Tensor Cores:
        - FP4: ~9000 TFLOPS (estimated)
        - FP8: ~4500 TFLOPS (estimated)
        - FP16: ~2250 TFLOPS (estimated)
        - BF16: ~2250 TFLOPS (estimated)
        - TF32: ~1125 TFLOPS (estimated)
        - FP32: ~150 TFLOPS (estimated)
        
        Note: These are estimated values based on expected improvements.
        """
        self.validate_dtype(dtype)
        
        flops_map = {
            'fp4': 9000.0,  # Estimated
            'fp8': 4500.0,  # Estimated
            'fp16': 2250.0,  # Estimated
            'bf16': 2250.0,  # Estimated
            'tf32': 1125.0,  # Estimated
            'fp32': 150.0,   # Estimated
        }
        return flops_map[dtype]
    
    def get_supported_dtypes(self) -> list[str]:
        return ['fp4', 'fp8', 'fp16', 'bf16', 'tf32', 'fp32']