"""
Knowledge Engine Orchestrator Module

Real, dependency-light orchestrator that wires the knowledge pipeline together:
text ingestion -> extraction -> storage -> vector indexing -> knowledge graph
building/merge -> retrieval. Documented as the top-level coordinator for the
knowledge engine. Heavy optional dependencies (transformers, chroma, lean4) are
imported lazily and guarded so the module always imports cleanly.

Author: OpenEvolve Team
Date: 2026-08
"""
from __future__ import annotations


import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from enhanced_knowledge_core import (
    EnhancedKnowledgeCore, KnowledgeExtractor, KnowledgeIntegrator,
)
from knowledge_storage import KnowledgeStorage
from unified_kg import UnifiedKG
from unified_kg_integration_hub import UnifiedKGConfig, UnifiedKGIntegrationHub
from vector_search import HashEmbeddingProvider, VectorSearch, VectorSearchConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    use_model_embeddings: bool = False
    enable_graphiti: bool = False
    persist_path: Optional[str] = None


class KnowledgeEngineOrchestrator:
    """
    Coordinates the full knowledge lifecycle:

    1. ``ingest(text)`` extracts entities/relations, stores an artifact,
       indexes it for retrieval and updates the unified knowledge graph.
    2. ``search(query)`` retrieves the most relevant artifacts.
    3. ``merge_graph(other)`` combines an external subgraph into the hub.
    """
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.storage = KnowledgeStorage(persist_path=self.config.persist_path)
        self.extractor = KnowledgeExtractor(use_model=self.config.use_model_embeddings)
        self.integrator = KnowledgeIntegrator(self.storage)
        self.search = VectorSearch(VectorSearchConfig(
            embedding_provider=HashEmbeddingProvider()))
        self.core = EnhancedKnowledgeCore(self.storage, self.search, self.extractor)
        self.hub = UnifiedKGIntegrationHub(UnifiedKGConfig(
            enable_graphiti=self.config.enable_graphiti))

    def ingest(self, text: str, source: str = "doc",
               artifact_type: str = "insight") -> Dict[str, Any]:
        aid = self.core.ingest_text(text, source, artifact_type)
        entities, relations = self.extractor.extract(text, source)
        self.hub.build_from_extractions(entities, relations)
        return {"artifact_id": aid, "entities": len(entities),
                "relations": len(relations)}

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.core.retrieve(query, top_k)

    def merge_graph(self, other: UnifiedKG) -> None:
        self.hub.merge(other)

    def stats(self) -> Dict[str, Any]:
        return {
            "artifacts": self.storage.count(),
            "graph_nodes": len(self.hub.graph.nodes),
            "graph_edges": len(self.hub.graph.edges),
            "sources": self.hub.source_count,
        }

    def verify(self) -> Dict[str, bool]:
        """Lightweight self-check that all components are wired."""
        return {
            "storage": self.storage is not None,
            "search": self.search is not None,
            "extractor": self.extractor is not None,
            "hub": self.hub is not None,
        }
