"""
Graphiti Integration Package for OpenEvolve Knowledge Engine

This package provides integration with Graphiti temporal knowledge graph system,
enabling temporal queries, contradiction detection, and agent memory capabilities.

Components:
- GraphitiTemporalBridge: Main bridge to Graphiti system
- GraphitiHealthChecker: Health checking utilities
- GraphitiContradictionDetector: Contradiction detection
"""

try:
    from .graphiti_temporal_bridge import GraphitiTemporalBridge
except ImportError:
    GraphitiTemporalBridge = None

try:
    from .health_check import GraphitiHealthChecker
except ImportError:
    GraphitiHealthChecker = None

try:
    from .contradiction_detector import GraphitiContradictionDetector
except ImportError:
    GraphitiContradictionDetector = None

__all__ = [
    'GraphitiTemporalBridge',
    'GraphitiHealthChecker',
    'GraphitiContradictionDetector'
]