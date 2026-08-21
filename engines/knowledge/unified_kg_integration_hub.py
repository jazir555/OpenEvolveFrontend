"""
Unified KG Integration Hub Module

Real, dependency-light hub that ties multiple knowledge-graph sources together
and exposes unified build / merge / query operations. Documented as the
orchestration layer over the unified knowledge graph. Optional integrations
(``integrations.graphiti_integration``) are imported lazily and guarded.

Author: OpenEvolve Team
Date: 2026-08
"""
from __future__ import annotations


import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unified_kg import UnifiedKG

logger = logging.getLogger(__name__)


@dataclass
class UnifiedKGConfig:
    """Configuration for the unified KG integration hub."""
    name: str = "unified_hub"
    graphiti_host: Optional[str] = None
    enable_graphiti: bool = False


class UnifiedKGIntegrationHub:
    """
    Hub that owns a primary :class:`UnifiedKG` and accepts subgraphs from
    multiple sources, merging them into one queryable graph.
    """
    def __init__(self, config: Optional[UnifiedKGConfig] = None):
        self.config = config or UnifiedKGConfig()
        self.graph = UnifiedKG(name=self.config.name)
        self._sources: List[str] = []
        self._graphiti = None
        if self.config.enable_graphiti:
            self._init_graphiti()

    def _init_graphiti(self) -> None:
        try:  # pragma: no cover - optional integration
            from integrations.graphiti_integration import GraphitiIntegration
            host = self.config.graphiti_host or "localhost"
            self._graphiti = GraphitiIntegration(host=host)
            logger.info("Graphiti integration enabled")
        except Exception as exc:
            logger.info("Graphiti unavailable, using local graph: %s", exc)
            self._graphiti = None

    def add_source(self, source_id: str, graph: UnifiedKG) -> None:
        self.graph.merge(graph)
        if source_id not in self._sources:
            self._sources.append(source_id)

    def build_from_extractions(self, entities: List[Any],
                               relations: List[Any]) -> None:
        self.graph.build_from_extractions(entities, relations)

    def merge(self, other: UnifiedKG) -> None:
        self.graph.merge(other)

    def query(self, node_id: str) -> List[Dict[str, Any]]:
        return self.graph.neighbors(node_id)

    def export(self) -> Dict[str, Any]:
        return self.graph.to_dict()

    @property
    def source_count(self) -> int:
        return len(self._sources)
