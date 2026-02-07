"""openevolve.openevolve.api module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class Api:
    """Main class for openevolve.openevolve.api."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class ApiConfig:
    """Configuration for Api."""
    enabled: bool = True


class ApiError(Exception):
    """Error for Api."""
    pass


def create_api(*args, **kwargs):
    """Factory function."""
    return Api(*args, **kwargs)
