"""Knowledge Engine DTS Integration.

Multi-turn conversation optimization for Knowledge Graph interactions.

This module wraps the core DTS engine with KG-specific functionality:
- Optimize KG query dialogs
- Simulate user interactions with KG queries
- Score KG conversation trajectories
- Generate explanation trees for KG results
- Multi-turn retrieval optimization

Example:
    >>> from knowledge_engine.integrations.dts import DTSKGIntegration
    >>> 
    >>> # Create integration
    >>> kg_dts = DTSKGIntegration()
    >>> 
    >>> # Optimize a KG query dialog
    >>> tree = kg_dts.optimize_kg_query_dialog(
    ...     context="Find companies in AI sector",
    ...     user_goal="Research AI companies"
    ... )
    >>> 
    >>> # Extract entities via optimized dialog
    >>> entities = kg_dts.extract_kg_via_dialog(
    ...     entity_query="AI companies founded after 2020"
    ... )
    >>> print(f"Extracted {entities.total_count} entities")
    >>> 
    >>> # Generate explanation for KG results
    >>> script = kg_dts.explain_kg_result_conversation(
    ...     kg_data={"entities": [...], "relations": [...]},
    ...     user_knowledge_level="intermediate"
    ... )
"""

# Import core DTS components for re-export
from integrations.dts import (
    # Data structures
    ConversationNode,
    ConversationTree,
    StrategyGenerator,
    
    # User simulation
    UserPersona,
    UserSimulator,
    IntentModel,
    IntentType,
    PREDEFINED_PERSONAS,
    
    # Scoring
    ScoreResult,
    Judge,
    TrajectoryScorer,
    CriterionType,
    
    # Search
    BeamState,
    BeamSearch,
    ParallelBeamSearch,
    
    # Engine
    DTSConfig,
    DTSResult,
    DTSEngine,
    DTSEngineBuilder,
)

# KG-specific components
from .dts_integration import (
    DTSKGIntegration,
    SimulatedResponse,
    ExtractedEntities,
    ConversationScript,
    OptimalPath,
)

__all__ = [
    # Core DTS - re-exported for convenience
    "ConversationNode",
    "ConversationTree",
    "StrategyGenerator",
    "UserPersona",
    "UserSimulator",
    "IntentModel",
    "IntentType",
    "PREDEFINED_PERSONAS",
    "ScoreResult",
    "Judge",
    "TrajectoryScorer",
    "CriterionType",
    "BeamState",
    "BeamSearch",
    "ParallelBeamSearch",
    "DTSConfig",
    "DTSResult",
    "DTSEngine",
    "DTSEngineBuilder",
    
    # KG-specific
    "DTSKGIntegration",
    "SimulatedResponse",
    "ExtractedEntities",
    "ConversationScript",
    "OptimalPath",
]
