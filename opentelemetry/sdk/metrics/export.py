"""opentelemetry.sdk.metrics.export module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class Export:
    """Main class for opentelemetry.sdk.metrics.export."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class ExportConfig:
    """Configuration for Export."""
    enabled: bool = True


class ExportError(Exception):
    """Error for Export."""
    pass


def create_export(*args, **kwargs):
    """Factory function."""
    return Export(*args, **kwargs)
