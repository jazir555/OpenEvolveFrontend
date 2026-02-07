"""docling.datamodel.base_models module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class BaseModels:
    """Main class for docling.datamodel.base_models."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class BaseModelsConfig:
    """Configuration for BaseModels."""
    enabled: bool = True


class BaseModelsError(Exception):
    """Error for BaseModels."""
    pass


def create_base_models(*args, **kwargs):
    """Factory function."""
    return BaseModels(*args, **kwargs)
