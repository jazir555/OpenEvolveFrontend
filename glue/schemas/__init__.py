"""
Glue Schemas - Canonical Data Models for OpenEvolve Frontend

This module provides canonical schemas for data exchange between
core projects and glue adapters.

Following CLAUDE.md principles:
- Anti-Corruption Layer: Transform to/from canonical format
- Schema-first: Define contracts before implementation
- Contract Testing: Validate schemas at startup
"""

__version__ = "1.0.0"

# Export RESE schemas
try:
    from .rese_schemas import (
        # Enums
        HypothesisStatus,
        PatternType,
        MCTSNodeState,
        ExplorationStrategy,
        ContradictionType,
        # Core schemas
        Hypothesis,
        SearchTreeNode,
        Pattern,
        MCTSSearchResult,
        # Configuration
        ExplorationConfig,
    )
    __all__ = [
        # Enums
        "HypothesisStatus",
        "PatternType",
        "MCTSNodeState",
        "ExplorationStrategy",
        "ContradictionType",
        # Core schemas
        "Hypothesis",
        "SearchTreeNode",
        "Pattern",
        "MCTSSearchResult",
        # Configuration
        "ExplorationConfig",
    ]
except ImportError:
    # RESE schemas not available
    __all__ = []
