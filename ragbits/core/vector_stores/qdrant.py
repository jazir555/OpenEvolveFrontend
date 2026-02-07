"""ragbits.core.vector_stores.qdrant module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class Qdrant:
    """Main class for ragbits.core.vector_stores.qdrant."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class QdrantConfig:
    """Configuration for Qdrant."""
    enabled: bool = True


class QdrantError(Exception):
    """Error for Qdrant."""
    pass


def create_qdrant(*args, **kwargs):
    """Factory function."""
    return Qdrant(*args, **kwargs)
