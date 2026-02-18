"""
Z3 Knowledge Integration - Alias to Complete Implementation

This module provides the Z3 knowledge integration by re-exporting from
z3_knowledge_complete.py for backward compatibility.

Author: OpenEvolve
Date: 2026-02-17
"""

# Re-export everything from the complete implementation
try:
    from .z3_knowledge_complete import (
        # Main classes
        Z3KnowledgeManager,
        Z3KnowledgePersistence,
        Z3KnowledgeExtractor,
        FeatureExtractionPipeline,
        ConflictDetector,
        ExtractedFeatures,

        # Helper function
        get_z3_knowledge_manager,

        # Data structures from z3_knowledge_extraction
        ProofPattern,
        ConstraintPattern,
        SolutionStrategy,
        MathematicalInsight
    )
except ImportError:
    # Fallback for direct execution
    from z3_knowledge_complete import (
        # Main classes
        Z3KnowledgeManager,
        Z3KnowledgePersistence,
        Z3KnowledgeExtractor,
        FeatureExtractionPipeline,
        ConflictDetector,
        ExtractedFeatures,

        # Helper function
        get_z3_knowledge_manager,

        # Data structures from z3_knowledge_extraction
        ProofPattern,
        ConstraintPattern,
        SolutionStrategy,
        MathematicalInsight
    )

# Aliases for backward compatibility
Z3KnowledgeIntegration = Z3KnowledgeManager
Z3KnowledgeExtractionHook = Z3KnowledgePersistence
get_z3_knowledge_integration = get_z3_knowledge_manager
get_z3_knowledge_extractor = lambda: Z3KnowledgeExtractor()
