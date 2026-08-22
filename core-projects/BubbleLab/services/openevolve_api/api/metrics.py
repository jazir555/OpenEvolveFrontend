"""
Dependency-free in-memory request metrics collector and Starlette middleware.

Only the Python standard library and Starlette are used, so this module can
never fail to import (no prometheus_client or other third-party dependencies).

The collector tracks, per route+method:
    * cumulative request count
    * rolling error count (5xx responses)
    * cumulative latency (seconds)

plus process-wide totals. All state is held in plain dicts guarded by a lock.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_lock = threading.Lock()

_stats: Dict[str, Any] = {
    "total_requests": 0,
    "error_count": 0,
    "latency_total": 0.0,
    "by_route": {},  # "METHOD path" -> {count, errors, latency_total}
}


def record_request(method: str, path: str, status_code: int, latency: float) -> None:
    """Record a single completed HTTP request."""
    is_error = 500 <= status_code < 600
    with _lock:
        _stats["total_requests"] += 1
        _stats["latency_total"] += latency
        if is_error:
            _stats["error_count"] += 1
        key = f"{method} {path}"
        route = _stats["by_route"].get(key)
        if route is None:
            route = {"count": 0, "errors": 0, "latency_total": 0.0}
            _stats["by_route"][key] = route
        route["count"] += 1
        route["latency_total"] += latency
        if is_error:
            route["errors"] += 1


def get_metrics() -> Dict[str, Any]:
    """Return a snapshot of the collected metrics."""
    with _lock:
        return {
            "total_requests": _stats["total_requests"],
            "error_count": _stats["error_count"],
            "latency_total": round(_stats["latency_total"], 6),
            "by_route": {k: dict(v) for k, v in _stats["by_route"].items()},
        }


class MetricsMiddleware:
    """Starlette ASGI middleware that records each HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status", status_code)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency = time.perf_counter() - start
            record_request(method, path, status_code, latency)
