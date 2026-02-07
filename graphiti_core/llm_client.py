"""graphiti_core.llm_client module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class LlmClient:
    """Main class for graphiti_core.llm_client."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class LlmClientConfig:
    """Configuration for LlmClient."""
    enabled: bool = True


class LlmClientError(Exception):
    """Error for LlmClient."""
    pass


def create_llm_client(*args, **kwargs):
    """Factory function."""
    return LlmClient(*args, **kwargs)
