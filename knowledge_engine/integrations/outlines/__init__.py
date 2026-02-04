"""
Knowledge Engine Outlines Integration

Thin wrapper around SSOT implementations in integrations/outlines/
with KE-specific context and Memgraph-compatible output formats.

Version: 1.0.0
License: Apache-2.0

This module provides:
- Integration with UnifiedKGIntegrationHub
- Memgraph-compatible output formats
- Structured logging per CLAUDE.md
- KE-specific convenience methods
"""

import warnings
from datetime import datetime, timezone

__version__ = "1.0.0"

# Primary exports from SSOT
from integrations.outlines import (
    OutlinesAdapter,
    OutlinesConfig,
    OutlinesResult,
    ModelProvider,
    EntityExtractionSchema,
    RelationshipSchema,
    CypherQuerySchema,
    ValidationResultSchema,
    KnowledgeGraphConstraints,
    PromptTemplateManager,
    GenerationError,
    ValidationError,
)

# KE-specific wrapper
from .outlines_integration import OutlinesKGIntegration, KGExtractionResult

# Deprecation warnings for old imports
_old_imports = {}

def _warn_deprecated(old_name: str, new_name: str):
    """Warn about deprecated imports."""
    warnings.warn(
        f"'{old_name}' is deprecated. Use '{new_name}' from integrations.outlines instead.",
        DeprecationWarning,
        stacklevel=3
    )

__all__ = [
    # SSOT exports
    "OutlinesAdapter",
    "OutlinesConfig",
    "OutlinesResult",
    "ModelProvider",
    "EntityExtractionSchema",
    "RelationshipSchema",
    "CypherQuerySchema",
    "ValidationResultSchema",
    "KnowledgeGraphConstraints",
    "PromptTemplateManager",
    "GenerationError",
    "ValidationError",
    # KE-specific
    "OutlinesKGIntegration",
    "KGExtractionResult",
]
