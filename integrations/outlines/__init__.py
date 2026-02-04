"""
Outlines Integration for OpenEvolve

Structured LLM Output Generation with regex/JSON constraints.
Integrates with DSPy for optimized prompts + guaranteed valid outputs.

Version: 1.0.0
License: Apache-2.0
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"

# Core adapter
from .adapter import (
    OutlinesAdapter,
    OutlinesConfig,
    OutlinesResult,
    ModelProvider,
    GenerationError,
    ValidationError,
    ConstraintCompilationError,
)

# KG-specific constraints
from .kg_constraints import (
    EntityExtractionSchema,
    RelationshipSchema,
    CypherQuerySchema,
    ValidationResultSchema,
    PropertySchema,
    KnowledgeGraphConstraints,
)

# Prompt templates
from .prompt_templates import (
    ENTITY_EXTRACTION_TEMPLATE,
    RELATION_EXTRACTION_TEMPLATE,
    SCHEMA_VALIDATION_TEMPLATE,
    CYPHER_GENERATION_TEMPLATE,
    PromptTemplateManager,
)

__all__ = [
    # Adapter
    "OutlinesAdapter",
    "OutlinesConfig",
    "OutlinesResult",
    "ModelProvider",
    "GenerationError",
    "ValidationError",
    "ConstraintCompilationError",
    # KG Constraints
    "EntityExtractionSchema",
    "RelationshipSchema",
    "CypherQuerySchema",
    "ValidationResultSchema",
    "PropertySchema",
    "KnowledgeGraphConstraints",
    # Prompt Templates
    "ENTITY_EXTRACTION_TEMPLATE",
    "RELATION_EXTRACTION_TEMPLATE",
    "SCHEMA_VALIDATION_TEMPLATE",
    "CYPHER_GENERATION_TEMPLATE",
    "PromptTemplateManager",
]
