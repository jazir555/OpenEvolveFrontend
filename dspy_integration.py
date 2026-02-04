"""
Global DSPy Integration Module for OpenEvolve

SSOT (Single Source of Truth): 
knowledge_engine/integrations/dspy_integration.py

This module re-exports DSPy integration components for backward compatibility.
New code should import directly from the SSOT.

Migration:
    OLD: from dspy_integration import DSPyIntegration, initialize_dspy
    NEW: from knowledge_engine.integrations.dspy_integration import (
        DSPyIntegration, initialize_dspy
    )

The SSOT contains:
- DSPyIntegration class (full implementation)
- DSPy signatures (KnowledgeExtraction, ContentEvaluation, etc.)
- Global instance helpers (get_global_dspy_instance, initialize_dspy)
- All DSPy configuration and methods
"""

import warnings
import sys
from pathlib import Path

# Add knowledge_engine to path for imports
_ke_path = Path(__file__).parent / "knowledge_engine"
if str(_ke_path) not in sys.path:
    sys.path.insert(0, str(_ke_path))

# Re-export all components from SSOT
from knowledge_engine.integrations.dspy_integration import (
    # Main class
    DSPyIntegration,
    DSPyResult,
    # Signatures
    KnowledgeExtractionSignature,
    ContentEvaluationSignature,
    StrategyGenerationSignature,
    SolutionPatternSignature,
    # Helpers
    get_global_dspy_instance,
    initialize_dspy,
    get_dspy_status,
    # Constants
    DSPY_INTEGRATION_AVAILABLE,
    DSPY_SIGNATURES_AVAILABLE,
)

__version__ = "2.0.0"
__all__ = [
    "DSPyIntegration",
    "DSPyResult",
    "KnowledgeExtractionSignature",
    "ContentEvaluationSignature",
    "StrategyGenerationSignature",
    "SolutionPatternSignature",
    "get_global_dspy_instance",
    "initialize_dspy",
    "get_dspy_status",
    "DSPY_INTEGRATION_AVAILABLE",
    "DSPY_SIGNATURES_AVAILABLE",
]

# Warn about deprecated import path
warnings.warn(
    "Importing from 'dspy_integration' is deprecated. "
    "Import from 'knowledge_engine.integrations.dspy_integration' instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)
