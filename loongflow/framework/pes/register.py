"""Runner registration utilities for PES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


_REGISTRY: Dict[str, Any] = {}


@dataclass
class Worker:
    name: str
    runner: Any


def register_runner(name: str, runner: Any) -> Worker:
    """Register a PES runner and return a Worker wrapper."""
    _REGISTRY[name] = runner
    return Worker(name=name, runner=runner)


__all__ = ["register_runner", "Worker"]
