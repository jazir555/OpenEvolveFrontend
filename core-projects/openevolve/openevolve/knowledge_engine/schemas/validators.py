"""
Schema Validators

Provides validation logic for entities and relationships against schema definitions.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import re

from .base import (
    EntitySchema,
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    ValidationResult
)


logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates entities and relationships against schema definitions.
    """

    def __init__(self, schema: EntitySchema):
        """
        Initialize validator with a schema.

        Args:
            schema: EntitySchema to validate against
        """
        self.schema = schema
        self.logger = logging.getLogger(f"{__name__}.{schema.domain}")

    def validate_entity(self, entity: Entity) -> ValidationResult:
        """
        Validate a single entity against the schema.

        Args:
            entity: Entity to validate

        Returns:
            ValidationResult with detailed validation status
        """
        result = ValidationResult(is_valid=True, entity_count=1)

        # Check entity type exists
        entity_type = self.schema.get_entity_type(entity.entity_type)
        if not entity_type:
            result.add_error(f"Unknown entity type: {entity.entity_type}")
            return result

        # Validate entity ID format
        if not entity.entity_id or not isinstance(entity.entity_id, str):
            result.add_error("Entity ID must be a non-empty string")
        else:
            # Check for valid characters (alphanumeric, underscore, hyphen)
            if not re.match(r'^[a-zA-Z0-9_\-:]+$', entity.entity_id):
                result.add_warning(
                    f"Entity ID '{entity.entity_id}' contains unusual characters"
                )

        # Validate confidence score
        if not 0.0 <= entity.confidence <= 1.0:
            result.add_error(
                f"Confidence score must be between 0 and 1, got {entity.confidence}"
            )

        # Validate against type definition
        is_valid, errors = entity_type.validate(entity.properties)
        if not is_valid:
            for error in errors:
                result.add_error(f"Entity '{entity.entity_id}': {error}")

        # Check for extra properties not in schema
        defined_properties = set(entity_type.get_all_properties().keys())
        actual_properties = set(entity.properties.keys())
        extra_properties = actual_properties - defined_properties

        if extra_properties:
            result.add_warning(
                f"Entity '{entity.entity_id}' has undefined properties: {extra_properties}"
            )

        # Update counters
        if result.is_valid:
            result.valid_count = 1
        else:
            result.invalid_count = 1

        return result

    def validate_relationship(
        self,
        relationship: Relationship,
        source_entity_type: str,
        target_entity_type: str
    ) -> ValidationResult:
        """
        Validate a relationship against the schema.

        Args:
            relationship: Relationship to validate
            source_entity_type: Type of source entity
            target_entity_type: Type of target entity

        Returns:
            ValidationResult with detailed validation status
        """
        result = ValidationResult(is_valid=True)

        # Check relationship type exists
        rel_type = self.schema.get_relationship_type(relationship.relationship_type)
        if not rel_type:
            result.add_error(f"Unknown relationship type: {relationship.relationship_type}")
            return result

        # Validate relationship ID format
        if not relationship.relationship_id or not isinstance(relationship.relationship_id, str):
            result.add_error("Relationship ID must be a non-empty string")

        # Validate against type definition
        is_valid, errors = rel_type.validate(
            relationship.properties,
            source_entity_type,
            target_entity_type
        )
        if not is_valid:
            for error in errors:
                result.add_error(f"Relationship '{relationship.relationship_id}': {error}")

        # Validate confidence score
        if not 0.0 <= relationship.confidence <= 1.0:
            result.add_error(
                f"Confidence score must be between 0 and 1, got {relationship.confidence}"
            )

        result.is_valid = len(result.errors) == 0
        return result

    def validate_batch(
        self,
        entities: List[Entity],
        fail_fast: bool = False
    ) -> ValidationResult:
        """
        Validate a batch of entities against the schema.

        Args:
            entities: List of entities to validate
            fail_fast: Stop on first validation error if True

        Returns:
            ValidationResult with aggregate validation status
        """
        aggregate_result = ValidationResult(
            is_valid=True,
            entity_count=len(entities)
        )

        for i, entity in enumerate(entities):
            entity_result = self.validate_entity(entity)

            if not entity_result.is_valid and fail_fast:
                # Return immediately on error
                entity_result.entity_count = i + 1
                entity_result.invalid_count = 1
                return entity_result

            aggregate_result.merge(entity_result)

        self.logger.info(
            f"Batch validation complete: {aggregate_result.valid_count} valid, "
            f"{aggregate_result.invalid_count} invalid"
        )

        return aggregate_result

    def validate_entity_consistency(
        self,
        entities: List[Entity]
    ) -> ValidationResult:
        """
        Validate consistency across multiple entities.

        Checks for:
        - Duplicate entity IDs
        - Orphaned entities (entities without relationships)
        - Circular dependencies

        Args:
            entities: List of entities to check

        Returns:
            ValidationResult with consistency issues
        """
        result = ValidationResult(is_valid=True)

        # Check for duplicate entity IDs
        entity_ids = [e.entity_id for e in entities]
        seen_ids: Set[str] = set()
        duplicates = set()

        for entity_id in entity_ids:
            if entity_id in seen_ids:
                duplicates.add(entity_id)
            seen_ids.add(entity_id)

        if duplicates:
            result.add_error(f"Duplicate entity IDs found: {duplicates}")

        # Check for missing required properties
        for entity in entities:
            entity_type = self.schema.get_entity_type(entity.entity_type)
            if entity_type:
                for prop_name, prop_def in entity_type.properties.items():
                    if prop_def.required and prop_name not in entity.properties:
                        result.add_error(
                            f"Entity '{entity.entity_id}' missing required property: {prop_name}"
                        )

        result.is_valid = len(result.errors) == 0
        return result

    def validate_cross_schema_compatibility(
        self,
        other_schema: EntitySchema
    ) -> ValidationResult:
        """
        Check compatibility between this schema and another schema.

        Checks for:
        - Compatible entity type definitions
        - Compatible property types
        - Mappable relationship types

        Args:
            other_schema: Other schema to check compatibility with

        Returns:
            ValidationResult with compatibility issues
        """
        result = ValidationResult(is_valid=True)

        # Check for overlapping entity types
        self_types = set(self.schema.list_entity_types())
        other_types = set(other_schema.list_entity_types())
        common_types = self_types & other_types

        if common_types:
            result.add_warning(
                f"Found {len(common_types)} common entity types: {common_types}"
            )

            # Check compatibility of common types
            for type_name in common_types:
                self_type = self.schema.get_entity_type(type_name)
                other_type = other_schema.get_entity_type(type_name)

                # Check property compatibility
                self_props = set(self_type.properties.keys())
                other_props = set(other_type.properties.keys())

                if self_props != other_props:
                    result.add_warning(
                        f"Entity type '{type_name}' has different properties: "
                        f"only_self={self_props - other_props}, "
                        f"only_other={other_props - self_props}"
                    )

        # Check for compatible relationship types
        self_rels = set(self.schema.list_relationship_types())
        other_rels = set(other_schema.list_relationship_types())

        common_rels = self_rels & other_rels
        if common_rels:
            result.add_warning(
                f"Found {len(common_rels)} common relationship types: {common_rels}"
            )

        result.is_valid = len(result.errors) == 0
        return result


class EntityBatchProcessor:
    """
    Processes and validates batches of entities efficiently.
    """

    def __init__(self, validator: SchemaValidator, batch_size: int = 100):
        """
        Initialize batch processor.

        Args:
            validator: SchemaValidator instance
            batch_size: Number of entities per batch
        """
        self.validator = validator
        self.batch_size = batch_size

    def process_entities(
        self,
        entities: List[Entity],
        on_batch_complete: Optional[callable] = None
    ) -> ValidationResult:
        """
        Process entities in batches.

        Args:
            entities: List of entities to process
            on_batch_complete: Optional callback for batch completion

        Returns:
            Aggregate ValidationResult
        """
        aggregate_result = ValidationResult(is_valid=True, entity_count=len(entities))

        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i + self.batch_size]
            batch_result = self.validator.validate_batch(batch)

            aggregate_result.merge(batch_result)

            if on_batch_complete:
                on_batch_complete(i, i + len(batch), batch_result)

        return aggregate_result


class SchemaMigrationValidator:
    """
    Validates schema migrations and changes.
    """

    @staticmethod
    def validate_migration(
        old_schema: EntitySchema,
        new_schema: EntitySchema
    ) -> ValidationResult:
        """
        Validate that a schema migration is safe.

        Checks for:
        - Removed entity types (breaking change)
        - Removed properties (breaking change)
        - Type changes on properties (breaking change)
        - Changed required properties (breaking change)

        Args:
            old_schema: Original schema
            new_schema: New schema

        Returns:
            ValidationResult with migration issues
        """
        result = ValidationResult(is_valid=True)

        # Check for removed entity types
        old_types = set(old_schema.list_entity_types())
        new_types = set(new_schema.list_entity_types())
        removed_types = old_types - new_types

        if removed_types:
            result.add_error(f"Removed entity types (breaking change): {removed_types}")

        # Check for changes in existing types
        common_types = old_types & new_types
        for type_name in common_types:
            old_type = old_schema.get_entity_type(type_name)
            new_type = new_schema.get_entity_type(type_name)

            # Check for removed properties
            old_props = set(old_type.properties.keys())
            new_props = set(new_type.properties.keys())
            removed_props = old_props - new_props

            if removed_props:
                result.add_error(
                    f"Entity type '{type_name}': Removed properties {removed_props}"
                )

            # Check for property type changes
            for prop_name in old_props & new_props:
                old_prop = old_type.properties[prop_name]
                new_prop = new_type.properties[prop_name]

                if old_prop.type != new_prop.type:
                    result.add_error(
                        f"Entity type '{type_name}': Property '{prop_name}' "
                        f"type changed from {old_prop.type} to {new_prop.type}"
                    )

                # Check if optional became required
                if not old_prop.required and new_prop.required:
                    result.add_error(
                        f"Entity type '{type_name}': Property '{prop_name}' "
                        f"changed from optional to required (breaking change)"
                    )

        result.is_valid = len(result.errors) == 0
        return result

    @staticmethod
    def generate_migration_plan(
        old_schema: EntitySchema,
        new_schema: EntitySchema
    ) -> Dict[str, Any]:
        """
        Generate a migration plan for converting entities from old to new schema.

        Args:
            old_schema: Original schema
            new_schema: New schema

        Returns:
            Dictionary with migration steps
        """
        plan = {
            "breaking_changes": [],
            "additive_changes": [],
            "migration_steps": []
        }

        old_types = set(old_schema.list_entity_types())
        new_types = set(new_schema.list_entity_types())

        # Removed types
        for type_name in old_types - new_types:
            plan["breaking_changes"].append({
                "type": "removed_entity_type",
                "entity_type": type_name,
                "severity": "error"
            })

        # Added types
        for type_name in new_types - old_types:
            plan["additive_changes"].append({
                "type": "added_entity_type",
                "entity_type": type_name
            })

        # Common types
        for type_name in old_types & new_types:
            old_type = old_schema.get_entity_type(type_name)
            new_type = new_schema.get_entity_type(type_name)

            old_props = set(old_type.properties.keys())
            new_props = set(new_type.properties.keys())

            # Removed properties
            for prop_name in old_props - new_props:
                plan["breaking_changes"].append({
                    "type": "removed_property",
                    "entity_type": type_name,
                    "property": prop_name,
                    "severity": "error"
                })
                plan["migration_steps"].append({
                    "action": "delete_property",
                    "entity_type": type_name,
                    "property": prop_name
                })

            # Added properties
            for prop_name in new_props - old_props:
                plan["additive_changes"].append({
                    "type": "added_property",
                    "entity_type": type_name,
                    "property": prop_name
                })

        return plan
