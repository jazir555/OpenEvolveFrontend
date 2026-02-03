"""
OpenTelemetry Integration - License: Apache 2.0

Distributed tracing and metrics for OpenEvolve using OpenTelemetry.
All OpenTelemetry packages are Apache 2.0 licensed.

Dependencies (all Apache 2.0):
- opentelemetry-api: Apache 2.0
- opentelemetry-sdk: Apache 2.0
- opentelemetry-instrumentation-fastapi: Apache 2.0
- opentelemetry-exporter-otlp: Apache 2.0
- opentelemetry-instrumentation: Apache 2.0

Author: OpenEvolve
Date: 2026-02-02
"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager
from enum import Enum
import time

# OpenTelemetry - All Apache 2.0
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Tracer, Span, SpanKind, Status, StatusCode
    from opentelemetry.metrics import Meter, Counter, Histogram, ObservableGauge
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry not installed. Telemetry will be disabled.")

logger = logging.getLogger(__name__)


class TelemetryConfig:
    """Configuration for OpenTelemetry."""
    
    def __init__(
        self,
        service_name: str = "openevolve",
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        sample_rate: float = 1.0
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.sample_rate = sample_rate


class TelemetryManager:
    """
    Central manager for OpenTelemetry telemetry.
    
    Provides:
    - Distributed tracing
    - Metrics collection
    - Auto-instrumentation
    - Custom spans and metrics
    
    License: Apache 2.0
    """
    
    _instance: Optional['TelemetryManager'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if TelemetryManager._initialized:
            return
            
        self._tracer: Optional[Tracer] = None
        self._meter: Optional[Meter] = None
        self._config: Optional[TelemetryConfig] = None
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, ObservableGauge] = {}
        TelemetryManager._initialized = True
    
    def initialize(self, config: TelemetryConfig) -> bool:
        """
        Initialize OpenTelemetry with configuration.
        
        Args:
            config: Telemetry configuration
            
        Returns:
            True if initialized successfully
        """
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, telemetry disabled")
            return False
            
        self._config = config
        
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: config.service_name,
            SERVICE_VERSION: config.service_version,
        })
        
        # Initialize tracing
        if config.enable_tracing:
            trace_provider = TracerProvider(resource=resource)
            
            # Add OTLP exporter if endpoint configured
            if config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
                trace_provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
            
            # Add console exporter if enabled
            if config.console_export:
                console_exporter = ConsoleSpanExporter()
                trace_provider.add_span_processor(
                    BatchSpanProcessor(console_exporter)
                )
            
            trace.set_tracer_provider(trace_provider)
            self._tracer = trace.get_tracer(config.service_name, config.service_version)
            logger.info(f"Tracing initialized for {config.service_name}")
        
        # Initialize metrics
        if config.enable_metrics:
            readers = []
            
            if config.otlp_endpoint:
                otlp_exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint)
                readers.append(PeriodicExportingMetricReader(otlp_exporter))
            
            if config.console_export:
                console_exporter = ConsoleMetricExporter()
                readers.append(PeriodicExportingMetricReader(console_exporter))
            
            if readers:
                metric_provider = MeterProvider(resource=resource, metric_readers=readers)
                metrics.set_meter_provider(metric_provider)
                self._meter = metrics.get_meter(config.service_name, config.service_version)
                logger.info(f"Metrics initialized for {config.service_name}")
        
        return True
    
    def get_tracer(self) -> Optional[Tracer]:
        """Get the tracer instance."""
        return self._tracer
    
    def get_meter(self) -> Optional[Meter]:
        """Get the meter instance."""
        return self._meter
    
    def create_counter(
        self,
        name: str,
        description: str,
        unit: str = "1"
    ) -> Optional[Counter]:
        """Create or get a counter metric."""
        if not self._meter:
            return None
            
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._counters[name]
    
    def create_histogram(
        self,
        name: str,
        description: str,
        unit: str = "ms"
    ) -> Optional[Histogram]:
        """Create or get a histogram metric."""
        if not self._meter:
            return None
            
        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._histograms[name]
    
    def instrument_fastapi(self, app) -> None:
        """Instrument a FastAPI application."""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI application instrumented")
        except Exception as e:
            logger.error(f"Error instrumenting FastAPI: {e}")


# Global telemetry manager
telemetry = TelemetryManager()


# =============================================================================
# DECORATORS & CONTEXT MANAGERS
# =============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator to trace function execution.
    
    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Additional attributes
        
    Example:
        @traced(name="decompose_problem", attributes={"component": "decomposition"})
        async def decompose(problem): ...
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not telemetry.get_tracer():
                return await func(*args, **kwargs)
            
            with telemetry.get_tracer().start_as_current_span(
                span_name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                try:
                    # Add function parameters as attributes
                    for i, arg in enumerate(args):
                        if isinstance(arg, (str, int, float, bool)):
                            span.set_attribute(f"arg.{i}", arg)
                    
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"kwarg.{key}", value)
                    
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not telemetry.get_tracer():
                return func(*args, **kwargs)
            
            with telemetry.get_tracer().start_as_current_span(
                span_name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def timed(metric_name: str, description: Optional[str] = None):
    """
    Decorator to time function execution and record as histogram.
    
    Args:
        metric_name: Name of the histogram metric
        description: Metric description
        
    Example:
        @timed("decomposition.duration", "Time to decompose problem")
        async def decompose(problem): ...
    """
    def decorator(func: F) -> F:
        hist = telemetry.create_histogram(
            metric_name,
            description or f"Duration of {func.__name__}",
            unit="ms"
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                if hist:
                    duration_ms = (time.time() - start) * 1000
                    hist.record(duration_ms)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                if hist:
                    duration_ms = (time.time() - start) * 1000
                    hist.record(duration_ms)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@contextmanager
def span_context(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Context manager for creating spans.
    
    Example:
        with span_context("process_subproblem", attributes={"id": sp.id}) as span:
            process(sp)
    """
    tracer = telemetry.get_tracer()
    if not tracer:
        yield None
        return
    
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as span:
        yield span


# =============================================================================
# WORKFLOW-SPECIFIC TELEMETRY
# =============================================================================

class WorkflowTelemetry:
    """Telemetry helpers for workflow tracking."""
    
    def __init__(self):
        self.workflow_counter = telemetry.create_counter(
            "workflows.total",
            "Total number of workflows executed"
        )
        self.workflow_duration = telemetry.create_histogram(
            "workflows.duration",
            "Workflow execution duration",
            unit="ms"
        )
        self.decomposition_counter = telemetry.create_counter(
            "decompositions.total",
            "Total number of decompositions performed"
        )
        self.subproblem_counter = telemetry.create_counter(
            "subproblems.completed",
            "Total number of sub-problems completed"
        )
    
    def record_workflow_start(self, workflow_id: str, problem_type: str) -> None:
        """Record workflow start."""
        if self.workflow_counter:
            self.workflow_counter.add(1, {"type": problem_type})
        
        # Start a span for the workflow
        tracer = telemetry.get_tracer()
        if tracer:
            span = tracer.start_span(
                "workflow.execution",
                attributes={
                    "workflow.id": workflow_id,
                    "workflow.problem_type": problem_type
                }
            )
            span.set_attribute("workflow.status", "started")
            # Store span in context for later
            
    def record_workflow_complete(
        self,
        workflow_id: str,
        duration_ms: float,
        success: bool
    ) -> None:
        """Record workflow completion."""
        if self.workflow_duration:
            self.workflow_duration.record(
                duration_ms,
                {"success": str(success)}
            )
        
        tracer = telemetry.get_tracer()
        if tracer:
            # End the workflow span
            pass  # Would retrieve and end span from context


# Global workflow telemetry
workflow_telemetry = WorkflowTelemetry()


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_telemetry(
    service_name: str = "openevolve",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False
) -> bool:
    """
    Initialize telemetry with standard configuration.
    
    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP collector endpoint (optional)
        console_export: Whether to export to console
        
    Returns:
        True if initialized successfully
    """
    config = TelemetryConfig(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        console_export=console_export,
        enable_metrics=True,
        enable_tracing=True
    )
    
    return telemetry.initialize(config)


# Convenience imports
__all__ = [
    'telemetry',
    'TelemetryConfig',
    'TelemetryManager',
    'traced',
    'timed',
    'span_context',
    'WorkflowTelemetry',
    'workflow_telemetry',
    'init_telemetry',
]
