"""opentelemetry.exporter.otlp.proto.grpc.metric_exporter module.

Auto-generated stub module.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class MetricExporter:
    """Main class for opentelemetry.exporter.otlp.proto.grpc.metric_exporter."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def process(self, data: Any = None) -> Any:
        return data


@dataclass
class MetricExporterConfig:
    """Configuration for MetricExporter."""
    enabled: bool = True


class MetricExporterError(Exception):
    """Error for MetricExporter."""
    pass


def create_metric_exporter(*args, **kwargs):
    """Factory function."""
    return MetricExporter(*args, **kwargs)
