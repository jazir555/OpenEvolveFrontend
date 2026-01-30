"""
Knowledge Graph Backend Adapters

This package provides backend adapters for various knowledge graph storage systems.
Each adapter implements the KnowledgeGraphBackend interface for consistency.

## Active Backends (Used in Code)
- PostgreSQLBackend: PostgreSQL License (permissive)
- MemgraphBackend: Apache 2.0 (permissive)
- QdrantBackend: Apache 2.0 (permissive)
- Redis: BSD (permissive)
- MemoryBackend: MIT (permissive)
- KarateClubBackend: MIT (permissive)

## Orphaned Backends (Not Used - Zero References)
The following backends exist as files but are NOT imported or used by any active code:
- Neo4jBackend: GPL license (copyleft) - orphaned, zero references
- MongoDBBackend: SSPL license (copyleft) - orphaned, zero references

All active code paths use only permissively-licensed backends.
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

"""
Knowledge Graph Backend Adapters

This package provides backend adapters for various knowledge graph storage systems.
Each adapter implements the KnowledgeGraphBackend interface for consistency.

## Active Backends (All Permissive Licenses)
- PostgreSQLBackend: PostgreSQL License
- MemgraphBackend: Apache 2.0
- QdrantBackend: Apache 2.0
- Redis: BSD
- MemoryBackend: MIT
- KarateClubBackend: MIT

## Removed Backends
The following backends were previously available but have been removed:
- Neo4jBackend: GPL license - removed
- MongoDBBackend: SSPL license - removed

Only permissive-licensed backends are supported.
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
from .memgraph_backend import MemgraphBackend  # Apache 2.0
from .qdrant_backend import QdrantBackend  # Apache 2.0
from .postgresql_backend import PostgreSQLBackend  # PostgreSQL License
from .karateclub_backend import KarateClubBackend  # MIT

# NOTE: Neo4jBackend and MongoDBBackend are intentionally NOT imported
# They are orphaned backends with non-permissive licenses (GPL/SSPL)
# The files exist but are not referenced by any active code

__all__ = [
    "KnowledgeGraphBackend",
    "BackendType",
    "OperationType",
    "KnowledgeEntry",
    "SearchResults",
    "AnalysisResult",
    "GraphStatistics",
    "MemoryBackend",
    "MemgraphBackend",
    "QdrantBackend",
    "PostgreSQLBackend",
    "KarateClubBackend",
]
