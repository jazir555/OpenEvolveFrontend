"""
Ontology Mapping Components

Sub-modules for ontology mapping system.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

from .lexical_matcher import LexicalMatcher
from .semantic_matcher import SemanticMatcher, FallbackSemanticMatcher
from .graph_embedder import GraphEmbedder, FallbackGraphEmbedder
from .kg_validator import KGValidator, FallbackKGValidator

__all__ = [
    'LexicalMatcher',
    'SemanticMatcher',
    'FallbackSemanticMatcher',
    'GraphEmbedder',
    'FallbackGraphEmbedder',
    'KGValidator',
    'FallbackKGValidator',
]
