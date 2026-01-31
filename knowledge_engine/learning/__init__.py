"""
OpenEvolve Knowledge Engine - Learning Module

Provides adaptive learning capabilities including:
- AdaptationEngine: Learns from experience and adapts system behavior
- ReflectionEngine: Periodically reflects on system performance and suggests improvements
"""

from .adaptation_engine import AdaptationEngine
from .reflection_engine import ReflectionEngine

__all__ = [
    "AdaptationEngine",
    "ReflectionEngine",
]
