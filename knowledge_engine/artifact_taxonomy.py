"""
Ultra-Comprehensive Knowledge Artifact Taxonomy

This module now re-exports from knowledge_engine.schemas.base for backward compatibility.
The actual implementations have been consolidated into the unified schema system.

Defines 30+ artifact types covering all aspects of problem-solving,
system design, team dynamics, and operational excellence.
"""

# Re-export all artifact-related classes from unified schemas
from knowledge_engine.schemas.base import (
    ArtifactCategory,
    ArtifactType,
    KnowledgeArtifact,
    ArtifactTaxonomy,
    TOTAL_ARTIFACT_TYPES,
)

# Keep backward compatibility for any code importing from here
__all__ = [
    "ArtifactCategory",
    "ArtifactType",
    "KnowledgeArtifact",
    "ArtifactTaxonomy",
    "TOTAL_ARTIFACT_TYPES",
]
