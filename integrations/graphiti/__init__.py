"""
Graphiti Integration for OpenEvolve

This package provides a decoupled adapter pattern for integrating Graphiti,
a temporally-aware knowledge graph, into the OpenEvolve knowledge engine.

Main Components:
- GraphitiAdapter: Implements KnowledgeGraphInterface for Graphiti
- GraphitiBridge: Connects adapter to OpenEvolve knowledge engine
- Config: YAML-based configuration management

Usage:
    from integrations.graphiti import get_bridge, GraphitiAdapter

    # Using bridge (recommended)
    bridge = await get_bridge("path/to/config.yaml")
    await bridge.initialize()
    await bridge.add_episode(...)
    results = await bridge.search("query")

    # Using adapter directly
    adapter = GraphitiAdapter()
    await adapter.initialize(config)
    await adapter.add_episode(...)
"""

from integrations.graphiti.adapter import (
    GraphitiAdapter,
    GRAPHITI_AVAILABLE,
)

from integrations.graphiti.bridge import (
    GraphitiBridge,
    get_bridge,
)

__all__ = [
    "GraphitiAdapter",
    "GraphitiBridge",
    "get_bridge",
    "GRAPHITI_AVAILABLE",
]

__version__ = "0.1.0"
