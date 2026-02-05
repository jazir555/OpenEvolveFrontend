"""
Z3-LeanAide Integration Bridge (DEPRECATED)

This module is preserved for backward compatibility.
The implementation has been migrated to CAV-NLP.

NEW LOCATION: openevolve.cav_nlp_integration

Migration Guide:
- Replace: from openevolve import z3_leanaide_bridge
- With: from openevolve.cav_nlp_integration import adapter

- Replace: z3_leanaide_bridge.Z3LeanAideBridge()
- With: adapter.Z3LeanAideBridge()
"""

import warnings
import sys

# Ensure deprecation warning is shown
if not sys.warnoptions:
    warnings.simplefilter("always", DeprecationWarning)

# Deprecation warning on module import
warnings.warn(
    "The z3_leanaide_bridge module is deprecated. "
    "Use openevolve.cav_nlp_integration.adapter instead. "
    "This module will be removed in version 2.0.0.",
    DeprecationWarning,
    stacklevel=2
)

# Version and metadata
__version__ = "2.0.0-cav-nlp-migration"
__deprecated_in__ = "2.0.0"
__removal_version__ = "3.0.0"
__replacement__ = "openevolve.cav_nlp_integration"

# Import all components from new location
from openevolve.cav_nlp_integration.adapter import (
    Z3LeanAideBridge,
    create_z3_lean_bridge,
    quick_verify,
)

from openevolve.cav_nlp_integration.data_structures import (
    TranslationDirection,
    ConstraintType,
    Z3Constraint,
    Lean4Constraint,
    TranslationResult,
    VerificationBridgeResult,
    HybridProofResult,
    CAVNLPContext,
    CanonicalizationResult,
)

from openevolve.cav_nlp_integration.mappings import (
    Z3_TO_LEAN_TYPES,
    LEAN_TO_Z3_TYPES,
    Z3_TO_LEAN_OPERATORS,
    LEAN_TO_Z3_OPERATORS,
    CONSTRAINT_TYPE_TACTICS,
    CANONICALIZATION_RULES,
)

__all__ = [
    # Main classes
    "Z3LeanAideBridge",
    "TranslationDirection",
    "ConstraintType",
    "Z3Constraint",
    "Lean4Constraint",
    "TranslationResult",
    "VerificationBridgeResult",
    "HybridProofResult",
    "CAVNLPContext",
    "CanonicalizationResult",
    # Functions
    "create_z3_lean_bridge",
    "quick_verify",
    # Mappings
    "Z3_TO_LEAN_TYPES",
    "LEAN_TO_Z3_TYPES",
    "Z3_TO_LEAN_OPERATORS",
    "LEAN_TO_Z3_OPERATORS",
    "CONSTRAINT_TYPE_TACTICS",
    "CANONICALIZATION_RULES",
    # Migration helper
    "migrate_code",
]

# Backward compatibility aliases for legacy code
Z3ToLeanTranslator = None  # Removed - use CAV-NLP translator
LeanToZ3Translator = None  # Removed - use CAV-NLP reverse translator
Z3LeanVerificationBridge = None  # Use adapter.Z3LeanAideBridge instead
HybridProofEngine = None  # Removed - use adapter.Z3LeanAideBridge.prove()


def _warn_deprecated_class(name):
    """Warn about deprecated class usage."""
    warnings.warn(
        f"{name} is deprecated. Use openevolve.cav_nlp_integration.adapter instead.",
        DeprecationWarning,
        stacklevel=3
    )


def migrate_code(old_code: str) -> str:
    """
    Helper to migrate old code to new API.
    
    Args:
        old_code: Old Python code using z3_leanaide_bridge
        
    Returns:
        Migrated code using cav_nlp_integration
    """
    replacements = [
        ("from openevolve import z3_leanaide_bridge", 
         "from openevolve.cav_nlp_integration import adapter"),
        ("import z3_leanaide_bridge", 
         "from openevolve.cav_nlp_integration import adapter as z3_leanaide_bridge"),
        ("z3_leanaide_bridge.Z3ToLeanTranslator()", 
         "# Removed - use adapter.Z3LeanAideBridge().z3_to_lean4()"),
        ("z3_leanaide_bridge.LeanToZ3Translator()", 
         "# Removed - use adapter.Z3LeanAideBridge().lean4_to_z3()"),
    ]
    
    result = old_code
    for old, new in replacements:
        result = result.replace(old, new)
    
    return result
