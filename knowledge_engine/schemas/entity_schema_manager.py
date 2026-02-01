"""
Entity Schema Manager

Manages entity schemas across all knowledge graph projects.
Provides unified schema validation, mapping, and generation.

Uses unified ValidationResult from knowledge_engine.schemas.base.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .base import (
    EntitySchema,
    EntityTypeDefinition,  # Renamed from EntityType for clarity
    RelationshipTypeDefinition,  # Renamed from RelationshipType
    Entity,
    Relationship,
    PropertyDefinition,
    PropertyType,
    ValidationResult,  # Now from unified base
)


logger = logging.getLogger(__name__)


# Backward compatibility: ValidationResult is now from unified base
# which includes all fields from both previous versions:
# - is_valid, errors, warnings, entity_id, schema_name, timestamp (from schemas/base.py)
# - entity_count, valid_count, invalid_count (from entity_schema_manager.py)
# - validator, passed, score, feedback, improvements (from sovereign_data_models.py)


class EntitySchemaManager:
    """
    Manages entity schemas across all knowledge graph projects.

    Provides unified schema validation, mapping, and generation capabilities.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the schema manager.

        Args:
            config_path: Optional path to schema configuration file
        """
        self.schemas: Dict[str, EntitySchema] = {}
        self.schema_mappings: Dict[str, Dict[str, str]] = {}
        self.config: Dict[str, Any] = {}
        self.default_schema: Optional[str] = None

        if config_path:
            self._load_config(config_path)

        logger.info("EntitySchemaManager initialized")

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f) or {}

                self.default_schema = self.config.get('default_schema')
                logger.info(f"Loaded schema configuration from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")

    def register_schema(self, domain: str, schema_definition: Dict[str, Any]):
        """
        Register a schema for a domain.

        Args:
            domain: Domain name
            schema_definition: Schema definition dictionary or EntitySchema object
        """
        try:
            if isinstance(schema_definition, EntitySchema):
                schema = schema_definition
            else:
                schema = EntitySchema.from_dict({
                    "domain": domain,
                    **schema_definition
                })

            self.schemas[domain] = schema
            logger.info(f"Registered schema for domain: {domain}")
        except Exception as e:
            logger.error(f"Error registering schema for domain {domain}: {e}")
            raise

    def register_mapping(self, mapping_name: str, mapping: Dict[str, str]):
        """
        Register an entity type mapping between schemas.

        Args:
            mapping_name: Name of the mapping (e.g., 'knowledge_engine_to_graphiti')
            mapping: Dictionary mapping source types to target types
        """
        self.schema_mappings[mapping_name] = mapping
        logger.info(f"Registered mapping: {mapping_name}")

    def get_schema(self, domain: str) -> Optional[EntitySchema]:
        """
        Get schema by domain name.

        Args:
            domain: Domain name

        Returns:
            EntitySchema if found, None otherwise
        """
        return self.schemas.get(domain)

    def list_schemas(self) -> List[str]:
        """
        List all registered schema domains.

        Returns:
            List of domain names
        """
        return list(self.schemas.keys())

    def list_mappings(self) -> List[str]:
        """
        List all registered mappings.

        Returns:
            List of mapping names
        """
        return list(self.schema_mappings.keys())

    def map_entities(
        self,
        source_entities: List[Entity],
        target_schema: str,
        mapping_name: Optional[str] = None
    ) -> List[Entity]:
        """
        Map entities from one schema to another.

        Args:
            source_entities: List of source entities
            target_schema: Target schema domain name
            mapping_name: Optional specific mapping to use

        Returns:
            List of mapped entities
        """
        target_schema_obj = self.get_schema(target_schema)
        if not target_schema_obj:
            logger.error(f"Target schema not found: {target_schema}")
            return []

        # Determine mapping to use
        mapping = None
        if mapping_name:
            mapping = self.schema_mappings.get(mapping_name)
        elif source_entities:
            # Try to infer mapping from source
            source_domain = source_entities[0].source
            mapping_key = f"{source_domain}_to_{target_schema}"
            mapping = self.schema_mappings.get(mapping_key)

        if not mapping:
            logger.warning(f"No mapping found, using direct type conversion")
            mapping = {}

        mapped_entities = []
        for entity in source_entities:
            # Map entity type
            mapped_type = mapping.get(entity.entity_type, entity.entity_type)

            # Validate target type exists
            if not target_schema_obj.get_entity_type(mapped_type):
                logger.warning(
                    f"Target entity type '{mapped_type}' not found in schema '{target_schema}', "
                    f"skipping entity {entity.entity_id}"
                )
                continue

            # Create mapped entity
            mapped_entity = Entity(
                entity_id=entity.entity_id,
                entity_type=mapped_type,
                name=entity.name,
                properties=entity.properties.copy(),
                metadata={
                    **entity.metadata,
                    "original_type": entity.entity_type,
                    "original_source": entity.source,
                    "mapped_from": entity.source
                },
                source=target_schema,
                confidence=entity.confidence
            )

            mapped_entities.append(mapped_entity)

        logger.info(f"Mapped {len(mapped_entities)} entities to schema '{target_schema}'")
        return mapped_entities

    def validate_entity(
        self,
        entity: Entity,
        schema: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a single entity against a schema.

        Args:
            entity: Entity to validate
            schema: Schema domain name (uses default if not provided)

        Returns:
            ValidationResult with validation status (unified model)
        """
        schema_domain = schema or self.default_schema
        if not schema_domain:
            return ValidationResult(
                is_valid=False,
                errors=["No schema specified and no default schema set"]
            )

        schema_obj = self.get_schema(schema_domain)
        if not schema_obj:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema not found: {schema_domain}"]
            )

        result = ValidationResult(is_valid=True, entity_count=1)

        # Check entity type exists
        entity_type = schema_obj.get_entity_type(entity.entity_type)
        if not entity_type:
            result.add_error(f"Unknown entity type: {entity.entity_type}")
            result.invalid_count = 1
            return result

        # Validate against type definition
        is_valid, errors = entity_type.validate(entity.properties)
        if not is_valid:
            for error in errors:
                result.add_error(f"Entity {entity.entity_id}: {error}")
            result.invalid_count = 1
        else:
            result.valid_count = 1

        result.is_valid = len(result.errors) == 0
        result.passed = result.is_valid  # Sync with unified model
        return result

    def validate_entities(
        self,
        entities: List[Entity],
        schema: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate multiple entities against a schema.

        Args:
            entities: List of entities to validate
            schema: Schema domain name (uses default if not provided)

        Returns:
            ValidationResult with aggregate validation status
        """
        aggregate_result = ValidationResult(is_valid=True, entity_count=len(entities))

        for entity in entities:
            entity_result = self.validate_entity(entity, schema)
            aggregate_result.merge(entity_result)

        logger.info(
            f"Validated {aggregate_result.entity_count} entities: "
            f"{aggregate_result.valid_count} valid, {aggregate_result.invalid_count} invalid"
        )

        return aggregate_result

    def validate_relationship(
        self,
        relationship: Relationship,
        schema: Optional[str] = None,
        source_entity_type: Optional[str] = None,
        target_entity_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a relationship against a schema.

        Args:
            relationship: Relationship to validate
            schema: Schema domain name
            source_entity_type: Source entity type
            target_entity_type: Target entity type

        Returns:
            ValidationResult with validation status
        """
        schema_domain = schema or self.default_schema
        if not schema_domain:
            return ValidationResult(
                is_valid=False,
                errors=["No schema specified and no default schema set"]
            )

        schema_obj = self.get_schema(schema_domain)
        if not schema_obj:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema not found: {schema_domain}"]
            )

        result = ValidationResult(is_valid=True)

        # Check relationship type exists
        rel_type = schema_obj.get_relationship_type(relationship.relationship_type)
        if not rel_type:
            result.add_error(f"Unknown relationship type: {relationship.relationship_type}")
            return result

        # Validate if entity types provided
        if source_entity_type and target_entity_type:
            is_valid, errors = rel_type.validate(
                relationship.properties,
                source_entity_type,
                target_entity_type
            )
            if not is_valid:
                for error in errors:
                    result.add_error(f"Relationship {relationship.relationship_id}: {error}")

        result.is_valid = len(result.errors) == 0
        result.passed = result.is_valid
        return result

    def generate_schema_prompt(
        self,
        domain: str,
        include_examples: bool = True
    ) -> str:
        """
        Generate an LLM prompt for entity extraction based on a schema.

        Args:
            domain: Schema domain name
            include_examples: Whether to include example entities

        Returns:
            Prompt string for LLM
        """
        schema = self.get_schema(domain)
        if not schema:
            return f"Error: Schema '{domain}' not found"

        prompt_parts = [
            f"# Entity Extraction Task for {domain.upper()}\n",
            f"## Domain Description\n{schema.description}\n",
            "## Entity Types to Extract\n"
        ]

        for type_name, entity_type in schema.entity_types.items():
            prompt_parts.append(f"\n### {type_name}")
            prompt_parts.append(f"{entity_type.description}")

            if entity_type.properties:
                prompt_parts.append("\n**Properties:**")
                for prop_name, prop_def in entity_type.properties.items():
                    required = " (required)" if prop_def.required else " (optional)"
                    prompt_parts.append(
                        f"- `{prop_name}`: {prop_def.description}{required}"
                    )

            if include_examples and entity_type.examples:
                prompt_parts.append("\n**Example:**")
                for i, example in enumerate(entity_type.examples[:2], 1):
                    prompt_parts.append(f"\n{i}. {example}")

        prompt_parts.append("\n## Relationship Types\n")
        for rel_name, rel_type in schema.relationship_types.items():
            prompt_parts.append(f"\n### {rel_name}")
            prompt_parts.append(f"{rel_type.description}")
            prompt_parts.append(
                f"- Source types: {', '.join(rel_type.source_types)}"
            )
            prompt_parts.append(
                f"- Target types: {', '.join(rel_type.target_types)}"
            )

        prompt_parts.append(
            "\n## Instructions\n"
            "Extract entities and relationships from the provided text that match "
            "the schema definition above. Return results in a structured format."
        )

        return "\n".join(prompt_parts)

    def merge_cross_domain(
        self,
        entity_sets: List[Tuple[List[Entity], str]]
    ) -> List[Entity]:
        """
        Merge entities from different domains.

        Args:
            entity_sets: List of (entities, domain) tuples

        Returns:
            List of merged entities with deduplication
        """
        merged = {}
        entity_id_map = {}  # Track entity IDs across domains

        for entities, domain in entity_sets:
            schema = self.get_schema(domain)
            if not schema:
                logger.warning(f"Schema not found for domain: {domain}")
                continue

            for entity in entities:
                # Use entity ID as primary key
                if entity.entity_id in merged:
                    # Merge properties
                    existing = merged[entity.entity_id]
                    for key, value in entity.properties.items():
                        if key not in existing.properties:
                            existing.properties[key] = value

                    # Track sources
                    if "sources" not in existing.metadata:
                        existing.metadata["sources"] = []
                    if domain not in existing.metadata["sources"]:
                        existing.metadata["sources"].append(domain)

                    # Update confidence (use max)
                    existing.confidence = max(existing.confidence, entity.confidence)
                else:
                    # Add new entity
                    entity_copy = Entity(
                        entity_id=entity.entity_id,
                        entity_type=entity.entity_type,
                        name=entity.name,
                        properties=entity.properties.copy(),
                        metadata={
                            **entity.metadata,
                            "sources": [domain]
                        },
                        source=entity.source,
                        confidence=entity.confidence
                    )
                    merged[entity.entity_id] = entity_copy

        result = list(merged.values())
        logger.info(f"Merged {len(result)} unique entities from {len(entity_sets)} domains")
        return result

    def export_schema(self, domain: str, output_path: str):
        """
        Export a schema to a YAML file.

        Args:
            domain: Schema domain name
            output_path: Output file path
        """
        schema = self.get_schema(domain)
        if not schema:
            raise ValueError(f"Schema not found: {domain}")

        schema_dict = schema.to_dict()

        with open(output_path, 'w') as f:
            yaml.dump(schema_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported schema '{domain}' to {output_path}")

    def import_schema(self, domain: str, input_path: str):
        """
        Import a schema from a YAML file.

        Args:
            domain: Domain name to register
            input_path: Input file path
        """
        with open(input_path, 'r') as f:
            schema_dict = yaml.safe_load(f)

        self.register_schema(domain, schema_dict)
        logger.info(f"Imported schema '{domain}' from {input_path}")

    def get_statistics(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about registered schemas.

        Args:
            domain: Optional domain to get stats for (gets all if not provided)

        Returns:
            Dictionary with schema statistics
        """
        if domain:
            schemas = {domain: self.get_schema(domain)} if self.get_schema(domain) else {}
        else:
            schemas = self.schemas

        stats = {
            "total_schemas": len(schemas),
            "schemas": {}
        }

        for domain_name, schema in schemas.items():
            stats["schemas"][domain_name] = {
                "entity_types": len(schema.entity_types),
                "relationship_types": len(schema.relationship_types),
                "version": schema.version,
                "total_properties": sum(
                    len(et.properties) for et in schema.entity_types.values()
                )
            }

        return stats


# Export all
__all__ = [
    "ValidationResult",  # Unified from schemas.base
    "EntitySchemaManager",
]
