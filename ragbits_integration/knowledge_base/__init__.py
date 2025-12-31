"""
Enhanced Knowledge Base Module

Phase 4: Advanced RAG-powered knowledge base with automatic
knowledge extraction, vector indexing optimization, and
knowledge enrichment.
"""

from ragbits_integration.knowledge_base.extraction.knowledge_extractor import (
    KnowledgeExtractor,
    ExtractionResult,
    KnowledgeEntityType
)
from ragbits_integration.knowledge_base.enrichment.knowledge_enricher import (
    KnowledgeEnricher,
    EnrichmentResult
)
from ragbits_integration.knowledge_base.indexing.vector_optimizer import (
    VectorIndexOptimizer,
    IndexingStrategy,
    OptimizationReport
)
from ragbits_integration.knowledge_base.rag_engine.advanced_rag import (
    AdvancedRAGEngine,
    RAGQuery,
    RAGResult
)

__all__ = [
    # Extraction
    "KnowledgeExtractor",
    "ExtractionResult",
    "KnowledgeEntityType",

    # Enrichment
    "KnowledgeEnricher",
    "EnrichmentResult",

    # Indexing
    "VectorIndexOptimizer",
    "IndexingStrategy",
    "OptimizationReport",

    # RAG Engine
    "AdvancedRAGEngine",
    "RAGQuery",
    "RAGResult"
]
