"""
Knowledge Engine Schema System

Provides unified entity schema management across all knowledge graph projects.
"""

from .base import (
    EntitySchema,
    EntityType,
    RelationshipType,
    PropertyDefinition,
    ValidationRule,
    Entity,
    Relationship
)

from .entity_schema_manager import EntitySchemaManager, ValidationResult

from .validators import SchemaValidator

from .openevolve_schemas import (
    SOFTWARE_ENGINEERING_SCHEMA,
    MATHEMATICAL_REASONING_SCHEMA,
    WORKFLOW_PROVENANCE_SCHEMA
)

__all__ = [
    # Base classes
    'EntitySchema',
    'EntityType',
    'RelationshipType',
    'PropertyDefinition',
    'ValidationRule',
    'Entity',
    'Relationship',

    # Manager
    'EntitySchemaManager',
    'ValidationResult',

    # Validators
    'SchemaValidator',

    # Predefined schemas
    'SOFTWARE_ENGINEERING_SCHEMA',
    'MATHEMATICAL_REASONING_SCHEMA',
    'WORKFLOW_PROVENANCE_SCHEMA',
]
