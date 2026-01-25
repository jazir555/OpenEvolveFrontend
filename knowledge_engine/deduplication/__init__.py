"""
Unified Deduplication System for OpenEvolve Knowledge Engine

Integrates multiple deduplication strategies:
- SEMHASH: Fast rule-based deduplication (kg-gen)
- LM Cluster: ML-based clustering (kg-gen)
- Standardization: Entity normalization (ai-knowledge-graph)
- Semantic: LLM-based semantic matching (Graphiti)
"""

from .unified_manager import UnifiedDeduplicationManager
from .base import DeduplicationStrategy, DeduplicationResult, Entity

__all__ = [
    'UnifiedDeduplicationManager',
    'DeduplicationStrategy',
    'DeduplicationResult',
    'Entity'
]
