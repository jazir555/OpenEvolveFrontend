"""Knowledge Engine LMQL Integration Package.

Declarative query interface for Knowledge Graph operations using LMQL.
This is a thin wrapper around the primary LMQL implementation.

Architecture: SSOT (Single Source of Truth)
- Primary implementation: integrations/lmql/
- This wrapper: knowledge_engine/integrations/lmql/

Example:
    >>> from knowledge_engine.integrations.lmql import LMQLKGIntegration
    >>> integration = LMQLKGIntegration()
    >>> result = integration.query_entities("Find companies founded by Steve Jobs")
    
    >>> # Register with unified hub
    >>> from knowledge_engine.integrations.lmql import register_with_hub
    >>> integration = register_with_hub("lmql")

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

from knowledge_engine.integrations.lmql.lmql_integration import (
    # Data classes
    EntityQueryResult,
    RelationQueryResult,
    SchemaInferenceResult,
    MultiHopResult,
    QueryExplanation,
    CypherGenerationResult,
    # Main classes
    LMQLKGIntegration,
    UnifiedKGIntegrationHub,
    # Functions
    get_default_hub,
    register_with_hub,
)

# Version
__version__ = "1.0.0"

# Consolidated exports
__all__ = [
    # Version
    "__version__",
    # Data classes
    "EntityQueryResult",
    "RelationQueryResult",
    "SchemaInferenceResult",
    "MultiHopResult",
    "QueryExplanation",
    "CypherGenerationResult",
    # Main classes
    "LMQLKGIntegration",
    "UnifiedKGIntegrationHub",
    # Functions
    "get_default_hub",
    "register_with_hub",
]
