"""opentelemetry.instrumentation.fastapi module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class Fastapi:
    """Main class for opentelemetry.instrumentation.fastapi."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class FastapiConfig:
    """Configuration for Fastapi."""
    enabled: bool = True


class FastapiError(Exception):
    """Error for Fastapi."""
    pass


def create_fastapi(*args, **kwargs):
    """Factory function."""
    return Fastapi(*args, **kwargs)
