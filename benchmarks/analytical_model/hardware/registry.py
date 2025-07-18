"""Hardware registry for managing different GPU specifications."""

from typing import Dict, Type
from .base import BaseHardware
from .nvidia_gpus import A100Hardware, H100Hardware, H200Hardware, B200Hardware


class HardwareRegistry:
    """Registry for managing hardware specifications."""
    
    _hardware_classes: Dict[str, Type[BaseHardware]] = {}
    
    @classmethod
    def register(cls, name: str, hardware_class: Type[BaseHardware]) -> None:
        """Register a hardware class.
        
        Args:
            name: Name identifier for the hardware
            hardware_class: Hardware class to register
        """
        cls._hardware_classes[name.lower()] = hardware_class
    
    @classmethod
    def get_hardware(cls, name: str) -> BaseHardware:
        """Get hardware instance by name.
        
        Args:
            name: Name of the hardware
            
        Returns:
            Hardware instance
            
        Raises:
            ValueError: If hardware name is not found
        """
        name_lower = name.lower()
        if name_lower not in cls._hardware_classes:
            available = list(cls._hardware_classes.keys())
            raise ValueError(
                f"Hardware '{name}' not found. Available: {available}"
            )
        
        return cls._hardware_classes[name_lower]()
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available hardware names.
        
        Returns:
            List of available hardware names
        """
        return list(cls._hardware_classes.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if hardware is available.
        
        Args:
            name: Hardware name to check
            
        Returns:
            True if hardware is available, False otherwise
        """
        return name.lower() in cls._hardware_classes


# Register all available hardware
HardwareRegistry.register("a100", A100Hardware)
HardwareRegistry.register("h100", H100Hardware)
HardwareRegistry.register("h200", H200Hardware)
HardwareRegistry.register("b200", B200Hardware)


def get_hardware(name: str) -> BaseHardware:
    """Convenience function to get hardware instance.
    
    Args:
        name: Hardware name
        
    Returns:
        Hardware instance
    """
    return HardwareRegistry.get_hardware(name)


def list_available_hardware() -> list[str]:
    """Convenience function to list available hardware.
    
    Returns:
        List of available hardware names
    """
    return HardwareRegistry.list_available()