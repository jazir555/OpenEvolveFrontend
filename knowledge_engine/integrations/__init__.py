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

# LoongFlow Integration
try:
    from .loongflow_integration import (
        LoongFlowKnowledgeExtractor,
        PESRunResults,
        KnowledgeArtifact as LoongFlowKnowledgeArtifact,
    )
    LOONGFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    LOONGFLOW_INTEGRATION_AVAILABLE = False
    LoongFlowKnowledgeExtractor = None

# Unified Evolution Integration
try:
    from .unified_evolution_integration import (
        UnifiedEvolutionKnowledgeExtractor,
    )
    UNIFIED_EVOLUTION_AVAILABLE = True
except ImportError:
    UNIFIED_EVOLUTION_AVAILABLE = False
    UnifiedEvolutionKnowledgeExtractor = None

# ROMA Integration
try:
    from .roma_integration import (
        ROMAIntegration,
        ROMAMetaAgent
    )
    ROMA_INTEGRATION_AVAILABLE = True
except ImportError:
    ROMA_INTEGRATION_AVAILABLE = False

# ROMA-Entity Knowledge Graph Integration
try:
    from .roma_entity_kg_integration import (
        ROMAEntityType,
        ROMARelationshipType,
        ROMAEntity,
        ROMARelationship,
        ROMAKnowledgeResult,
        SimilarDecomposition,
        ROMAEntityExtractor,
        ROMAKnowledgeWriter,
        ROMAKnowledgeReader,
        create_roma_ekg_integration
    )
    ROMA_EKG_INTEGRATION_AVAILABLE = True
except ImportError:
    ROMA_EKG_INTEGRATION_AVAILABLE = False

# Causal-Learn Integration (Optional)
try:
    from .causal_learn_integration import (
        CausalLearnIntegration,
        CausalDiscoveryEngine
    )
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    CausalLearnIntegration = None
    CausalDiscoveryEngine = None

# DeepKE Integration
try:
    from .deepke_integration import (
        DeepKEIntegration,
        DeepKEEnhancedExtractor,
        DEEPKE_INTEGRATION_AVAILABLE
    )
except ImportError:
    DEEPKE_INTEGRATION_AVAILABLE = False

# DSPy Integration
try:
    from .dspy_integration import (
        DSPyIntegration,
        DSPY_INTEGRATION_AVAILABLE
    )
except ImportError:
    DSPY_INTEGRATION_AVAILABLE = False

# Ragbits Integration
try:
    from .ragbits_integration import (
        RagbitsIntegration,
        RAGBITS_INTEGRATION_AVAILABLE
    )
except ImportError:
    RAGBITS_INTEGRATION_AVAILABLE = False

# Agentic Context Engine Integration
try:
    from .agentic_context_integration import (
        AgenticContextEngine,
        ACE_INTEGRATION_AVAILABLE
    )
except ImportError:
    ACE_INTEGRATION_AVAILABLE = False

# AgentJSON Integration
try:
    from .agentjson_integration import (
        AgentJSONIntegration,
        AGENTJSON_INTEGRATION_AVAILABLE
    )
except ImportError:
    AGENTJSON_INTEGRATION_AVAILABLE = False

# Research Quest Integration
try:
    from .research_quest_integration import (
        ResearchQuestIntegration,
        RESEARCH_QUEST_INTEGRATION_AVAILABLE
    )
except ImportError:
    RESEARCH_QUEST_INTEGRATION_AVAILABLE = False

# MCP Gateway Integration
try:
    from .mcp_gateway_integration import (
        MCPGatewayIntegration,
        MCP_GATEWAY_INTEGRATION_AVAILABLE
    )
except ImportError:
    MCP_GATEWAY_INTEGRATION_AVAILABLE = False

# OpenEvolve Integration Library
try:
    from .openevolve_integration_library import (
        OPENEVOLVE_INTEGRATION_AVAILABLE
    )
except ImportError:
    OPENEVOLVE_INTEGRATION_AVAILABLE = False

# New Advanced Integrations (2026-02-03)
# Outlines - Structured LLM output generation
try:
    from .outlines.outlines_integration import (
        OutlinesKGIntegration,
        OUTLINES_INTEGRATION_AVAILABLE
    )
except ImportError:
    OutlinesKGIntegration = None
    OUTLINES_INTEGRATION_AVAILABLE = False

# LMQL - Declarative query language for LLMs
try:
    from .lmql.lmql_integration import (
        LMQLKGIntegration,
        LMQL_INTEGRATION_AVAILABLE
    )
except ImportError:
    LMQLKGIntegration = None
    LMQL_INTEGRATION_AVAILABLE = False

# Neuromancer - Physics-informed neural operators
try:
    from .neuromancer.neuromancer_integration import (
        NeuromancerKGIntegration,
        NEUROMANCER_INTEGRATION_AVAILABLE
    )
except ImportError:
    NeuromancerKGIntegration = None
    NEUROMANCER_INTEGRATION_AVAILABLE = False

# Cognitive-Hydraulics - Hybrid neuro-symbolic reasoning
try:
    from .cognitive_hydraulics.cognitive_hydraulics_integration import (
        CognitiveHydraulicsKGIntegration,
        COGNITIVE_HYDRAULICS_INTEGRATION_AVAILABLE
    )
except ImportError:
    CognitiveHydraulicsKGIntegration = None
    COGNITIVE_HYDRAULICS_INTEGRATION_AVAILABLE = False


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

    # LoongFlow Integration
    "LoongFlowKnowledgeExtractor",
    "PESRunResults",
    "LoongFlowKnowledgeArtifact",
    "LOONGFLOW_INTEGRATION_AVAILABLE",

    # Unified Evolution Integration
    "UnifiedEvolutionKnowledgeExtractor",
    "UNIFIED_EVOLUTION_AVAILABLE",

    # ROMA Integration
    "ROMAIntegration",
    "ROMAMetaAgent",

    # ROMA-EKG Integration
    "ROMAEntityType",
    "ROMARelationshipType",
    "ROMAEntity",
    "ROMARelationship",
    "ROMAKnowledgeResult",
    "SimilarDecomposition",
    "ROMAEntityExtractor",
    "ROMAKnowledgeWriter",
    "ROMAKnowledgeReader",
    "create_roma_ekg_integration",

    # Availability flags
    "Z3_INTEGRATION_AVAILABLE",
    "Z3_ENHANCED_AVAILABLE",
    "Z3_MODELS_AVAILABLE",
    "Z3_AUTO_EXTRACTION_AVAILABLE",
    "Z3_API_AVAILABLE",
    "LEANAIDE_KE_AVAILABLE",
    "LEANAIDE_PROOF_AVAILABLE",
    "UNIFIED_BRIDGE_AVAILABLE",
    "LOONGFLOW_INTEGRATION_AVAILABLE",
    "UNIFIED_EVOLUTION_AVAILABLE",
    "ROMA_INTEGRATION_AVAILABLE",
    "ROMA_EKG_INTEGRATION_AVAILABLE",
    
    # Causal-Learn Integration
    "CausalLearnIntegration",
    "CausalDiscoveryEngine",
    "CAUSAL_LEARN_AVAILABLE",

    # DeepKE Integration
    "DeepKEIntegration",
    "DeepKEEnhancedExtractor",
    "DEEPKE_INTEGRATION_AVAILABLE",

    # DSPy Integration
    "DSPyIntegration",
    "DSPY_INTEGRATION_AVAILABLE",

    # Ragbits Integration
    "RagbitsIntegration",
    "RAGBITS_INTEGRATION_AVAILABLE",

    # Agentic Context Engine Integration
    "AgenticContextEngine",
    "ACE_INTEGRATION_AVAILABLE",

    # AgentJSON Integration
    "AgentJSONIntegration",
    "AGENTJSON_INTEGRATION_AVAILABLE",

    # Research Quest Integration
    "ResearchQuestIntegration",
    "RESEARCH_QUEST_INTEGRATION_AVAILABLE",

    # MCP Gateway Integration
    "MCPGatewayIntegration",
    "MCP_GATEWAY_INTEGRATION_AVAILABLE",

    # OpenEvolve Integration Library
    "OPENEVOLVE_INTEGRATION_AVAILABLE",
    
    # Outlines Integration (Structured LLM Output Generation)
    "OutlinesKGIntegration",
    "OUTLINES_INTEGRATION_AVAILABLE",
    
    # LMQL Integration (Declarative Query Language)
    "LMQLKGIntegration",
    "LMQL_INTEGRATION_AVAILABLE",
    
    # Neuromancer Integration (Physics-Informed Neural Operators)
    "NeuromancerKGIntegration",
    "NEUROMANCER_INTEGRATION_AVAILABLE",
    
    # Cognitive-Hydraulics Integration (Hybrid Neuro-Symbolic Reasoning)
    "CognitiveHydraulicsKGIntegration",
    "COGNITIVE_HYDRAULICS_INTEGRATION_AVAILABLE",
]
