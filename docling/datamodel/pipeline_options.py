"""docling.datamodel.pipeline_options module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class PipelineOptions:
    """Main class for docling.datamodel.pipeline_options."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class PipelineOptionsConfig:
    """Configuration for PipelineOptions."""
    enabled: bool = True


class PipelineOptionsError(Exception):
    """Error for PipelineOptions."""
    pass


def create_pipeline_options(*args, **kwargs):
    """Factory function."""
    return PipelineOptions(*args, **kwargs)
