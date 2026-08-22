"""
Knowledge Graph Analytics subpackage.

Exposes the unified :class:`AnalyticsEngine` and the specialized analyzers
(centrality, community, path, pattern, temporal, spatial, ML).

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

from .base import AnalyticsRequest, build_nx_graph, AnalyticsError, BaseAnalyzer
from .centrality import CentralityAnalyzer
from .community import CommunityDetector
from .path import PathAnalyzer
from .pattern import PatternMiner
from .temporal import TemporalAnalyzer
from .spatial import SpatialAnalyzer
from .ml import MLIntegrator
from .engine import AnalyticsEngine

__all__ = [
    "AnalyticsRequest", "build_nx_graph", "AnalyticsError", "BaseAnalyzer",
    "CentralityAnalyzer", "CommunityDetector", "PathAnalyzer", "PatternMiner",
    "TemporalAnalyzer", "SpatialAnalyzer", "MLIntegrator", "AnalyticsEngine",
]
