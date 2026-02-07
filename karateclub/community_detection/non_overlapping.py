"""karateclub.community_detection.non_overlapping module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class NonOverlapping:
    """Main class for karateclub.community_detection.non_overlapping."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class NonOverlappingConfig:
    """Configuration for NonOverlapping."""
    enabled: bool = True


class NonOverlappingError(Exception):
    """Error for NonOverlapping."""
    pass


def create_non_overlapping(*args, **kwargs):
    """Factory function."""
    return NonOverlapping(*args, **kwargs)
