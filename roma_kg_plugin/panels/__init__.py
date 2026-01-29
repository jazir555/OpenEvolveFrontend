"""
ROMA Knowledge Graph Plugin - Panels

This package contains TUI panels for knowledge graph visualization.
All panels use dependency injection - no direct coupling to ROMA core.
"""

from .knowledge_graph_panel import KnowledgeGraphPanel
from .analytics_dashboard import AnalyticsDashboard

__all__ = [
    "KnowledgeGraphPanel",
    "AnalyticsDashboard"
]
