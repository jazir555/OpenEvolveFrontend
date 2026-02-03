"""
OpenEvolve Knowledge Engine Integrations

This package contains integrations with various external systems and tools.
"""

# Z3 Knowledge Integration
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
    from .z3_enhanced_knowledge import (
        EnhancedZ3KnowledgeIntegration,
        get_enhanced_z3_integration,
        MLPoweredPatternMatcher,
        AdaptiveStrategyOptimizer
    )
    Z3_ENHANCED_AVAILABLE = True
except ImportError:
    Z3_ENHANCED_AVAILABLE = False

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

# LeanAIDE Knowledge Integration
try:
    from .leanaide_knowledge_extraction import (
        LeanAideKnowledgeExtractor,
        get_leanaide_knowledge_extractor,
        TacticPattern,
        TheoremPattern,
        ProofStrategy,
        MathematicalConcept
    )
    LEANAIDE_KE_AVAILABLE = True
except ImportError:
    LEANAIDE_KE_AVAILABLE = False

try:
    from .leanaide_proof_integration import (
        LeanAideProofIntegration,
        get_leanaide_proof_integration,
        AutomatedProofSearcher,
        ProofAttempt,
        ProofSearchConfig
    )
    LEANAIDE_PROOF_AVAILABLE = True
except ImportError:
    LEANAIDE_PROOF_AVAILABLE = False

# Unified Bridge
try:
    from .unified_math_knowledge_bridge import (
        UnifiedMathKnowledgeBridge,
        get_unified_math_bridge,
        UnifiedMathProblem,
        UnifiedKnowledgePattern,
        ProblemClassifier,
        CrossSystemKnowledgeTransfer
    )
    UNIFIED_BRIDGE_AVAILABLE = True
except ImportError:
    UNIFIED_BRIDGE_AVAILABLE = False


__all__ = [
    # Z3 Knowledge Integration
    "Z3KnowledgeIntegration",
    "get_z3_knowledge_integration",
    "Z3KnowledgeExtractionHook",
    
    # Z3 Enhanced
    "EnhancedZ3KnowledgeIntegration",
    "get_enhanced_z3_integration",
    "MLPoweredPatternMatcher",
    "AdaptiveStrategyOptimizer",
    
    # Z3 Database models
    "Z3ProofPattern",
    "Z3ConstraintPattern", 
    "Z3Strategy",
    "Z3MathematicalInsight",
    "Z3SolverResult",
    "create_z3_tables",
    
    # Z3 Auto-extraction
    "Z3AutoExtractionManager",
    "get_auto_extraction_manager",
    "enable_auto_extraction",
    "disable_auto_extraction",
    "auto_extract_knowledge",
    "Z3KnowledgeExtractorMixin",
    
    # Z3 API
    "create_z3_knowledge_app",
    "z3_knowledge_router",
    
    # LeanAIDE Knowledge
    "LeanAideKnowledgeExtractor",
    "get_leanaide_knowledge_extractor",
    "TacticPattern",
    "TheoremPattern",
    "ProofStrategy",
    "MathematicalConcept",
    
    # LeanAIDE Proof
    "LeanAideProofIntegration",
    "get_leanaide_proof_integration",
    "AutomatedProofSearcher",
    "ProofAttempt",
    "ProofSearchConfig",
    
    # Unified Bridge
    "UnifiedMathKnowledgeBridge",
    "get_unified_math_bridge",
    "UnifiedMathProblem",
    "UnifiedKnowledgePattern",
    "ProblemClassifier",
    "CrossSystemKnowledgeTransfer",
    
    # Availability flags
    "Z3_INTEGRATION_AVAILABLE",
    "Z3_ENHANCED_AVAILABLE",
    "Z3_MODELS_AVAILABLE",
    "Z3_AUTO_EXTRACTION_AVAILABLE",
    "Z3_API_AVAILABLE",
    "LEANAIDE_KE_AVAILABLE",
    "LEANAIDE_PROOF_AVAILABLE",
    "UNIFIED_BRIDGE_AVAILABLE",
]
