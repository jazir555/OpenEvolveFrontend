"""
KG-Gen Integration Package for OpenEvolve Knowledge Engine

This package provides integration with KG-Gen knowledge extraction pipeline,
enabling entity and relationship extraction with advanced deduplication.

Components:
- ExtractionPipeline: Main extraction pipeline class
- KGGenPipelineIntegration: Complete 3-stage pipeline implementation
- DocumentChunker: Document chunking utilities
- ParallelChunkProcessor: Parallel processing utilities
- DeduplicationEngine: Entity deduplication using SEMHASH and LM clustering
- Neo4jGraphUploader: Neo4j upload functionality
"""

try:
    from .extraction_pipeline import ExtractionPipeline
except ImportError:
    ExtractionPipeline = None

try:
    from .kggen_pipeline import KGGenPipelineIntegration
except ImportError:
    KGGenPipelineIntegration = None

try:
    from .chunking import DocumentChunker
except ImportError:
    DocumentChunker = None

try:
    from .parallel_processing import ParallelChunkProcessor
except ImportError:
    ParallelChunkProcessor = None

try:
    from .deduplication import DeduplicationEngine
except ImportError:
    DeduplicationEngine = None

try:
    from .neo4j_integration import Neo4jGraphUploader
except ImportError:
    Neo4jGraphUploader = None

__all__ = [
    'ExtractionPipeline',
    'KGGenPipelineIntegration',
    'DocumentChunker',
    'ParallelChunkProcessor',
    'DeduplicationEngine',
    'Neo4jGraphUploader'
]