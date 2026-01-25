"""
Core Knowledge Engine Components

This package provides the core data structures and utilities for the Knowledge Engine.

Enhanced implementations following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Verify operations succeed
- IDEMPOTENCY: All operations safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs

Version: 2.0.0
"""

# Import enhanced implementations
from .entity_knowledge_graph import EntityKnowledgeGraph, Entity, Relationship
from .knowledge_state import KnowledgeState, KnowledgeTriple, StateSnapshot

__all__ = [
    'KnowledgeState',
    'EntityKnowledgeGraph',
    'Entity',
    'Relationship',
    'KnowledgeTriple',
    'StateSnapshot'
]
