"""Base hardware specification class for analytical modeling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BaseHardware(ABC):
    """Abstract base class for hardware specifications.
    
    This class defines the interface for hardware specifications used in
    analytical performance modeling. Subclasses should implement specific
    GPU hardware characteristics.
    """
    
    name: str
    peak_memory_bandwidth_gb_s: float
    memory_size_gb: float
    tensor_core_version: str
    base_clock_ghz: float
    memory_clock_ghz: float
    memory_bus_width_bits: int
    
    @abstractmethod
    def get_peak_flops(self, dtype: str) -> float:
        """Get peak FLOPS for the specified data type.
        
        Args:
            dtype: Data type ('fp8', 'fp16', 'bf16', 'fp32')
            
        Returns:
            Peak FLOPS in TFLOPS
            
        Raises:
            ValueError: If dtype is not supported by this hardware
        """
        pass
    
    @abstractmethod
    def get_supported_dtypes(self) -> list[str]:
        """Get list of supported data types.
        
        Returns:
            List of supported dtype strings
        """
        pass
    
    def get_theoretical_memory_bandwidth(self) -> float:
        """Calculate theoretical memory bandwidth from specifications.
        
        Returns:
            Theoretical memory bandwidth in GB/s
        """
        # Formula: (Memory Clock * Bus Width * 2) / 8 / 1e9
        # *2 for DDR, /8 for bits to bytes, /1e9 for GB/s
        return (self.memory_clock_ghz * 1e9 * self.memory_bus_width_bits * 2) / 8 / 1e9
    
    def validate_dtype(self, dtype: str) -> None:
        """Validate that the dtype is supported by this hardware.
        
        Args:
            dtype: Data type to validate
            
        Raises:
            ValueError: If dtype is not supported
        """
        if dtype not in self.get_supported_dtypes():
            raise ValueError(
                f"Data type '{dtype}' not supported by {self.name}. "
                f"Supported types: {self.get_supported_dtypes()}"
            )