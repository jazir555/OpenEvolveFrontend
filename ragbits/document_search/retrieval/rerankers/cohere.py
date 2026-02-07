"""ragbits.document_search.retrieval.rerankers.cohere module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class Cohere:
    """Main class for ragbits.document_search.retrieval.rerankers.cohere."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class CohereConfig:
    """Configuration for Cohere."""
    enabled: bool = True


class CohereError(Exception):
    """Error for Cohere."""
    pass


def create_cohere(*args, **kwargs):
    """Factory function."""
    return Cohere(*args, **kwargs)
