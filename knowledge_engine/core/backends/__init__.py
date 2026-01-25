"""
Knowledge Graph Backend Adapters

This package provides backend adapters for various knowledge graph storage systems.
Each adapter implements the KnowledgeGraphBackend interface for consistency.
"""

from .base import (
    KnowledgeGraphBackend,
    BackendType,
    OperationType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)

from .memory_backend import MemoryBackend
from .neo4j_backend import Neo4jBackend
from .qdrant_backend import QdrantBackend
from .mongodb_backend import MongoDBBackend
from .karateclub_backend import KarateClubBackend

__all__ = [
    "KnowledgeGraphBackend",
    "BackendType",
    "OperationType",
    "KnowledgeEntry",
    "SearchResults",
    "AnalysisResult",
    "GraphStatistics",
    "MemoryBackend",
    "Neo4jBackend",
    "QdrantBackend",
    "MongoDBBackend",
    "KarateClubBackend",
]
