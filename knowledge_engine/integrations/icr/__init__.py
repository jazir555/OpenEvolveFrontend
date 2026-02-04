"""Knowledge Engine ICR Integration.

Integrates ICR (Iterative Contextual Refinements) with the Knowledge Graph
for iterative refinement of extractions, queries, and schema inference.
"""

from knowledge_engine.integrations.icr.icr_integration import (
    ICRKGIntegration,
    RefinedExtraction,
    ImprovedQuery,
    RefinedEntities,
    OptimizedKG,
    RefinedSchema,
)

__version__ = "1.0.0"

__all__ = [
    "ICRKGIntegration",
    "RefinedExtraction",
    "ImprovedQuery",
    "RefinedEntities",
    "OptimizedKG",
    "RefinedSchema",
]
