"""
ACE Workflow Knowledge Extractor Module

Extracts knowledge from workflows for ACE framework.

Author: OpenEvolve Team
Date: 2026-02-06
"""
from __future__ import annotations


import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# ML Knowledge Extraction Integration
try:
    from ml_pattern_clustering import MLKnowledgeExtraction
    ML_KNOWLEDGE_AVAILABLE = True
except ImportError:
    ML_KNOWLEDGE_AVAILABLE = False

# Stage 6: store learned patterns as knowledge artifacts
from ace_knowledge_artifacts import (
    KnowledgeArtifact,
    InMemoryArtifactStore,
    ArtifactType,
    ArtifactSource,
    ArtifactStatus,
    ArtifactMetadata,
)

logger = logging.getLogger(__name__)


class ACEWorkflowKnowledgeExtractor:
    """ACE Workflow Knowledge Extractor class"""

    def __init__(self):
        self.ml_extractor = None
        if ML_KNOWLEDGE_AVAILABLE:
            try:
                self.ml_extractor = MLKnowledgeExtraction()
                logger.info("ACE Workflow Knowledge Extractor initialized with ML capabilities")
            except Exception as e:
                logger.error(f"Failed to initialize ML extractor: {e}")
        else:
            logger.info("ACE Workflow Knowledge Extractor initialized (Basic Mode)")

        # Stage 6 in-memory store for learned patterns
        self.store = InMemoryArtifactStore()

    def extract(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge from workflow and store learned patterns (Stage 6)."""
        if not self.ml_extractor:
            knowledge: Dict[str, Any] = {}
        else:
            try:
                # Extract combined problem/solution text
                text = f"Problem: {workflow.get('problem_statement', '')}\nSolution: {workflow.get('final_solution', '')}"

                # Use ML extractor
                knowledge = self.ml_extractor.extract_from_text(
                    text,
                    domain=workflow.get('domain', 'general')
                )
            except Exception as e:
                logger.error(f"Knowledge extraction failed: {e}")
                knowledge = {}

        # Stage 6: persist the learned pattern as a knowledge artifact
        artifact_id = self._store_learned_pattern(workflow, knowledge)

        return {
            "knowledge": knowledge,
            "workflow": workflow,
            "artifact_id": artifact_id,
            "stored_artifacts": self.store.count(),
            "timestamp": datetime.now().isoformat(),
        }

    def _store_learned_pattern(
        self, workflow: Dict[str, Any], knowledge: Dict[str, Any]
    ) -> Optional[str]:
        """Build and store a KnowledgeArtifact for the learned pattern (Stage 6)."""
        problem = workflow.get("problem_statement", "")
        solution = workflow.get("final_solution", "")
        domain = workflow.get("domain", "general")

        summary = knowledge.get("summary") if isinstance(knowledge, dict) else None
        content_parts = []
        if isinstance(knowledge, dict):
            for key in ("patterns", "insights", "summary", "extracted_text"):
                value = knowledge.get(key)
                if value:
                    content_parts.append(f"{key}: {value}")
        if solution:
            content_parts.append(f"solution: {solution}")
        content = "\n".join(content_parts) or solution or problem

        if not content.strip():
            logger.debug("No learnable content; skipping artifact storage")
            return None

        metadata = ArtifactMetadata(
            artifact_type=ArtifactType.DOMAIN_KNOWLEDGE,
            source=ArtifactSource.WORKFLOW_PHASE,
            status=ArtifactStatus.DRAFT,
            domain=domain,
            tags=["stage6", "auto-extracted"],
        )
        artifact = KnowledgeArtifact(
            metadata=metadata,
            title=(summary or problem)[:200] or "Learned pattern",
            description=f"Auto-extracted learning from workflow in domain '{domain}'.",
            content=content,
            context=f"Problem: {problem}",
        )
        return self.store.add(artifact)


# Alias for compatibility
WorkflowKnowledgeExtractor = ACEWorkflowKnowledgeExtractor
