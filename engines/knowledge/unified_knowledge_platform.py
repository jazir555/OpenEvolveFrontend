"""
Unified Knowledge Platform Module

Convenience facade over the knowledge engine orchestrator, exposing a single
high-level entry point for ingest / query / graph operations. Real and
dependency-light; delegates to :mod:`knowledge_engine_orchestrator`.

Author: OpenEvolve Team
Date: 2026-08
"""
from __future__ import annotations


import logging
from typing import Any, Dict, List, Optional

from knowledge_engine_orchestrator import (
    KnowledgeEngineOrchestrator, OrchestratorConfig,
)

logger = logging.getLogger(__name__)


class UnifiedKnowledgePlatform:
    """
    Single entry point bundling the orchestrator for callers that want one
    object to drive the whole knowledge lifecycle.
    """
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.orchestrator = KnowledgeEngineOrchestrator(config)

    def ingest(self, text: str, source: str = "doc", artifact_type: str = "insight") -> Dict[str, Any]:
        return self.orchestrator.ingest(text, source, artifact_type)

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.orchestrator.query(text, top_k)

    def stats(self) -> Dict[str, Any]:
        return self.orchestrator.stats()
