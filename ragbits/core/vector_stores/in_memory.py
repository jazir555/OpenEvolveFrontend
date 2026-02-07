"""ragbits.core.vector_stores.in_memory module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class InMemory:
    """Main class for ragbits.core.vector_stores.in_memory."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class InMemoryConfig:
    """Configuration for InMemory."""
    enabled: bool = True


class InMemoryError(Exception):
    """Error for InMemory."""
    pass


def create_in_memory(*args, **kwargs):
    """Factory function."""
    return InMemory(*args, **kwargs)
