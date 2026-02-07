"""ragbits.document_search.retrieval.rephrasers.noop module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class Noop:
    """Main class for ragbits.document_search.retrieval.rephrasers.noop."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class NoopConfig:
    """Configuration for Noop."""
    enabled: bool = True


class NoopError(Exception):
    """Error for Noop."""
    pass


def create_noop(*args, **kwargs):
    """Factory function."""
    return Noop(*args, **kwargs)
