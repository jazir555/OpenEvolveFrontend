"""
OpenEvolve Knowledge Engine Integrations

This package contains integrations with various external systems and tools.
"""

# Try to import Z3 knowledge integration
try:
    from .z3_knowledge_integration import (
        Z3KnowledgeIntegration,
        get_z3_knowledge_integration,
        Z3KnowledgeExtractionHook,
        Z3KnowledgeEntry
    )
    Z3_INTEGRATION_AVAILABLE = True
except ImportError:
    Z3_INTEGRATION_AVAILABLE = False

try:
    from .z3_database_models import (
        Z3KnowledgeEntry as DBZ3KnowledgeEntry,
        Z3ProofPattern,
        Z3ConstraintPattern,
        Z3Strategy,
        Z3MathematicalInsight,
        Z3SolverResult,
        create_z3_tables
    )
    Z3_MODELS_AVAILABLE = True
except ImportError:
    Z3_MODELS_AVAILABLE = False

try:
    from .z3_auto_extraction import (
        Z3AutoExtractionManager,
        get_auto_extraction_manager,
        enable_auto_extraction,
        disable_auto_extraction,
        auto_extract_knowledge,
        Z3KnowledgeExtractorMixin
    )
    Z3_AUTO_EXTRACTION_AVAILABLE = True
except ImportError:
    Z3_AUTO_EXTRACTION_AVAILABLE = False

try:
    from .z3_api import (
        create_z3_knowledge_app,
        router as z3_knowledge_router
    )
    Z3_API_AVAILABLE = True
except ImportError:
    Z3_API_AVAILABLE = False


__all__ = [
    # Main integration
    "Z3KnowledgeIntegration",
    "get_z3_knowledge_integration",
    "Z3KnowledgeExtractionHook",
    
    # Database models
    "Z3ProofPattern",
    "Z3ConstraintPattern", 
    "Z3Strategy",
    "Z3MathematicalInsight",
    "Z3SolverResult",
    "create_z3_tables",
    
    # Auto-extraction
    "Z3AutoExtractionManager",
    "get_auto_extraction_manager",
    "enable_auto_extraction",
    "disable_auto_extraction",
    "auto_extract_knowledge",
    "Z3KnowledgeExtractorMixin",
    
    # API
    "create_z3_knowledge_app",
    "z3_knowledge_router",
    
    # Availability flags
    "Z3_INTEGRATION_AVAILABLE",
    "Z3_MODELS_AVAILABLE",
    "Z3_AUTO_EXTRACTION_AVAILABLE",
    "Z3_API_AVAILABLE",
]
