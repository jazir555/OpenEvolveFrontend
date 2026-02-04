"""Tooling utilities for LoongFlow."""

from __future__ import annotations

from typing import Any, Callable, Optional


class BaseTool:
    """Minimal callable tool wrapper."""

    def __init__(self, func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None) -> None:
        self.func = func
        self.name = name or getattr(func, "__name__", "tool")
        self.description = description or getattr(func, "__doc__", "")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def function_tool(func: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to wrap a function into a BaseTool."""

    def _wrap(f: Callable[..., Any]) -> BaseTool:
        return BaseTool(f, name=name, description=description)

    if func is None:
        return _wrap
    return _wrap(func)


__all__ = ["BaseTool", "function_tool"]
