"""Minimal no-op tracing shim for the hermetic BubbleLab integration env.

The ``engines/other`` reliability stack imports :func:`initialize_tracer` and
uses it purely for observability (``start_as_current_span``). In environments
where a real tracing backend is not installed we provide a no-op implementation
so the analysis engines remain importable and functional.
"""

from contextlib import contextmanager
from typing import Iterator, Optional


class _NoopSpan:
    def set_attribute(self, key, value):  # pragma: no cover - no-op
        pass

    def add_event(self, name, attributes=None):  # pragma: no cover - no-op
        pass

    def record_exception(self, exc):  # pragma: no cover - no-op
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, *args, **kwargs) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    def start_span(self, name: str, *args, **kwargs) -> _NoopSpan:
        return _NoopSpan()


_TRACER = _NoopTracer()


def initialize_tracer(*args, **kwargs) -> _NoopTracer:
    return _TRACER


def get_tracer(name: Optional[str] = None) -> _NoopTracer:  # pragma: no cover
    return _TRACER
