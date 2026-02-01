"""
Knowledge Engine Schema System

Provides unified data models for:
- Knowledge artifacts
- Entities and relationships
- Validation results
- Schema definitions

All models have been consolidated into schemas.base for consistency.
"""

from .base import (
    # Enums
    PropertyType,
    ArtifactType,
    ArtifactCategory,
    EntityType,
    RelationshipType,
    TOTAL_ARTIFACT_TYPES,
    
    # Core classes
    PropertyDefinition,
    ValidationRule,
    KnowledgeArtifact,
    Entity,
    Relationship,
    ValidationResult,
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    EntitySchema,
    ArtifactTaxonomy,
)

# Also export from entity_schema_manager
from .entity_schema_manager import EntitySchemaManager

__all__ = [
    # Enums
    "PropertyType",
    "ArtifactType",
    "ArtifactCategory",
    "EntityType",
    "RelationshipType",
    "TOTAL_ARTIFACT_TYPES",
    
    # Core classes
    "PropertyDefinition",
    "ValidationRule",
    "KnowledgeArtifact",
    "Entity",
    "Relationship",
    "ValidationResult",
    "EntityTypeDefinition",
    "RelationshipTypeDefinition",
    "EntitySchema",
    "ArtifactTaxonomy",
    "EntitySchemaManager",
]
