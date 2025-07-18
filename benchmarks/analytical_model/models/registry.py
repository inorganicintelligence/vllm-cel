"""Model registry for managing different model specifications."""

from typing import Dict, Type
from .base import BaseModel
from .mixtral import Mixtral8x7B
from .deepseek import DeepSeekV3


class ModelRegistry:
    """Registry for managing model specifications."""
    
    _model_classes: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model class.
        
        Args:
            name: Name identifier for the model
            model_class: Model class to register
        """
        cls._model_classes[name.lower()] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> BaseModel:
        """Get model instance by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not found
        """
        name_lower = name.lower()
        if name_lower not in cls._model_classes:
            available = list(cls._model_classes.keys())
            raise ValueError(
                f"Model '{name}' not found. Available: {available}"
            )
        
        return cls._model_classes[name_lower]()
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available model names.
        
        Returns:
            List of available model names
        """
        return list(cls._model_classes.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if model is available.
        
        Args:
            name: Model name to check
            
        Returns:
            True if model is available, False otherwise
        """
        return name.lower() in cls._model_classes


# Register all available models
ModelRegistry.register("mixtral-8x7b", Mixtral8x7B)
ModelRegistry.register("mixtral8x7b", Mixtral8x7B)  # Alternative name
ModelRegistry.register("deepseek-v3", DeepSeekV3)
ModelRegistry.register("deepseekv3", DeepSeekV3)  # Alternative name


def get_model(name: str) -> BaseModel:
    """Convenience function to get model instance.
    
    Args:
        name: Model name
        
    Returns:
        Model instance
    """
    return ModelRegistry.get_model(name)


def list_available_models() -> list[str]:
    """Convenience function to list available models.
    
    Returns:
        List of available model names
    """
    return ModelRegistry.list_available()