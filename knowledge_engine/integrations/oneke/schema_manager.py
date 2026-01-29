"""
OneKE Schema Management System
Task 3.3: Schema Management System

Implements:
- 3.3.1: Schema definition format (JSON/YAML)
- 3.3.2: Schema versioning
- 3.3.3: Schema validation with Pydantic
- 3.3.4: Dynamic schema updates
- 3.3.5: Schema migration tools
- 3.3.6: Schema library for common domains

Following CLAUDE.md Principles:
- RUNTIME TRUTH: Validate schemas at runtime
- IDEMPOTENCY: Schema operations are idempotent
- CONFIGURATION EXPLICITNESS: Schema paths via environment variables
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import hashlib
import copy

from pydantic import BaseModel, Field, field_validator, model_validator

# Structured logging
logger = logging.getLogger(__name__)


class SchemaFormat(Enum):
    """Schema file formats."""
    JSON = "json"
    YAML = "yaml"


@dataclass
class EntityType:
    """Entity type definition."""
    name: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "examples": self.examples,
            "attributes": self.attributes,
            "validation_rules": self.validation_rules
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityType":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            examples=data.get("examples", []),
            attributes=data.get("attributes", []),
            validation_rules=data.get("validation_rules", {})
        )


@dataclass
class RelationType:
    """Relation type definition."""
    name: str
    description: str = ""
    domain: Optional[str] = None  # Subject entity type
    range: Optional[str] = None   # Object entity type
    examples: List[str] = field(default_factory=list)
    symmetric: bool = False
    inverse_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "range": self.range,
            "examples": self.examples,
            "symmetric": self.symmetric,
            "inverse_of": self.inverse_of
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationType":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain"),
            range=data.get("range"),
            examples=data.get("examples", []),
            symmetric=data.get("symmetric", False),
            inverse_of=data.get("inverse_of")
        )


@dataclass
class EventType:
    """Event type definition."""
    name: str
    description: str = ""
    arguments: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
            "examples": self.examples
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventType":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            arguments=data.get("arguments", []),
            examples=data.get("examples", [])
        )


class SchemaDefinition(BaseModel):
    """
    Schema definition with Pydantic validation (Task 3.3.3).

    Environment Variables:
    - ONEKE_SCHEMA_DIR: Directory for schema storage
    - ONEKE_SCHEMA_VERSIONING: Enable versioning (default: true)
    """

    name: str = Field(..., description="Schema name")
    version: str = Field(default="1.0.0", description="Schema version")
    description: str = Field(default="", description="Schema description")
    entity_types: List[Dict[str, Any]] = Field(default_factory=list, description="Entity types")
    relation_types: List[Dict[str, Any]] = Field(default_factory=list, description="Relation types")
    event_types: List[Dict[str, Any]] = Field(default_factory=list, description="Event types")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate version format."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {v}, expected X.Y.Z")
        return v

    @field_validator("entity_types")
    @classmethod
    def validate_entity_types(cls, v):
        """Validate entity types."""
        names = set()
        for entity_type in v:
            if "name" not in entity_type:
                raise ValueError("Entity type must have 'name' field")
            if entity_type["name"] in names:
                raise ValueError(f"Duplicate entity type name: {entity_type['name']}")
            names.add(entity_type["name"])
        return v

    @field_validator("relation_types")
    @classmethod
    def validate_relation_types(cls, v):
        """Validate relation types."""
        for relation_type in v:
            if "name" not in relation_type:
                raise ValueError("Relation type must have 'name' field")
        return v

    @model_validator(mode='after')
    def validate_relations_reference_entities(self):
        """Validate that relations reference existing entities."""
        entity_names = {et["name"] for et in self.entity_types}
        relation_types = self.relation_types

        for relation_type in relation_types:
            if "domain" in relation_type and relation_type["domain"]:
                if relation_type["domain"] not in entity_names:
                    raise ValueError(f"Relation {relation_type['name']} references unknown domain entity: {relation_type['domain']}")

            if "range" in relation_type and relation_type["range"]:
                if relation_type["range"] not in entity_names:
                    raise ValueError(f"Relation {relation_type['name']} references unknown range entity: {relation_type['range']}")

        return self

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "name": "company_schema",
                "version": "1.0.0",
                "description": "Schema for company-related entities",
                "entity_types": [
                    {
                        "name": "Company",
                        "description": "A business organization",
                        "examples": ["Apple", "Microsoft", "Google"]
                    },
                    {
                        "name": "Person",
                        "description": "A human individual",
                        "examples": ["Steve Jobs", "Bill Gates"]
                    }
                ],
                "relation_types": [
                    {
                        "name": "founded_by",
                        "description": "Company was founded by person",
                        "domain": "Company",
                        "range": "Person"
                    }
                ]
            }
        }

    def get_hash(self) -> str:
        """Get schema hash for versioning."""
        schema_dict = self.dict(exclude={"created_at", "updated_at"})
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


class OneKESchemaManager:
    """
    Schema management system (Task 3.3).

    Implements:
    - Task 3.3.1: Schema definition format (JSON/YAML)
    - Task 3.3.2: Schema versioning
    - Task 3.3.3: Schema validation with Pydantic
    - Task 3.3.4: Dynamic schema updates
    - Task 3.3.5: Schema migration tools
    - Task 3.3.6: Schema library for common domains

    Following CLAUDE.md:
    - RUNTIME TRUTH: Validate schemas at load time
    - IDEMPOTENCY: Schema operations are idempotent
    - CONFIGURATION EXPLICITNESS: Schema directory via environment
    - UTC TIME: All timestamps in UTC
    - STRUCTURED LOGGING: JSON logs with correlation IDs
    """

    # Built-in schema library (Task 3.3.6)
    BUILTIN_SCHEMAS = {
        "general": {
            "name": "general",
            "version": "1.0.0",
            "description": "General purpose knowledge extraction schema",
            "entity_types": [
                {"name": "Person", "description": "A person", "examples": ["John Doe", "Jane Smith"]},
                {"name": "Organization", "description": "An organization", "examples": ["Apple", "UN"]},
                {"name": "Location", "description": "A location", "examples": ["New York", "Paris"]},
                {"name": "Date", "description": "A date or time", "examples": ["2024-01-01", "January"]},
                {"name": "Number", "description": "A numerical value", "examples": ["100", "3.14"]},
            ],
            "relation_types": [
                {"name": "located_in", "description": "Something is located in a place"},
                {"name": "works_for", "description": "Person works for organization"},
                {"name": "founded", "description": "Person founded organization"},
                {"name": "happened_on", "description": "Event happened on date"},
            ]
        },
        "biomedical": {
            "name": "biomedical",
            "version": "1.0.0",
            "description": "Biomedical domain schema",
            "entity_types": [
                {"name": "Gene", "description": "A gene", "examples": ["TP53", "BRCA1"]},
                {"name": "Protein", "description": "A protein", "examples": ["p53", "Hemoglobin"]},
                {"name": "Disease", "description": "A disease", "examples": ["Cancer", "Diabetes"]},
                {"name": "Drug", "description": "A drug or medication", "examples": ["Aspirin", "Insulin"]},
            ],
            "relation_types": [
                {"name": "associates_with", "description": "Gene associates with disease"},
                {"name": "interacts_with", "description": "Protein interacts with protein"},
                {"name": "treats", "description": "Drug treats disease"},
            ]
        },
        "legal": {
            "name": "legal",
            "version": "1.0.0",
            "description": "Legal domain schema",
            "entity_types": [
                {"name": "Court", "description": "A court", "examples": ["Supreme Court", "District Court"]},
                {"name": "Case", "description": "A legal case", "examples": ["Roe v. Wade"]},
                {"name": "Law", "description": "A law or statute", "examples": ["Constitution", "Bill of Rights"]},
                {"name": "Judge", "description": "A judge", "examples": ["Judge Smith"]},
            ],
            "relation_types": [
                {"name": "heard_in", "description": "Case was heard in court"},
                {"name": "presided_over", "description": "Judge presided over case"},
                {"name": "established_by", "description": "Case established law"},
            ]
        }
    }

    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize schema manager.

        Args:
            schema_dir: Directory for schema storage (default: from env or ./schemas)

        Environment Variables:
            ONEKE_SCHEMA_DIR: Schema storage directory
        """
        self.schema_dir = Path(schema_dir or os.getenv("ONEKE_SCHEMA_DIR", "./knowledge_engine/integrations/oneke/schemas"))
        self.schema_dir.mkdir(parents=True, exist_ok=True)

        # Schema cache: name -> {version -> SchemaDefinition}
        self._schemas: Dict[str, Dict[str, SchemaDefinition]] = {}

        # Version history
        self._version_history: Dict[str, List[str]] = {}

        logger.info({
            "msg": "Schema manager initialized",
            "schema_dir": str(self.schema_dir),
            "builtin_schemas": len(self.BUILTIN_SCHEMAS),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def load_schema(
        self,
        name: str,
        version: Optional[str] = None,
        format: SchemaFormat = SchemaFormat.JSON,
        correlation_id: Optional[str] = None
    ) -> SchemaDefinition:
        """
        Load schema from file or built-in library (Task 3.3.1).

        Args:
            name: Schema name (or path for file loading)
            version: Schema version (latest if not specified)
            format: File format
            correlation_id: Correlation ID for tracing

        Returns:
            SchemaDefinition

        Raises:
            FileNotFoundError: If schema file not found
            ValueError: If schema is invalid
        """
        correlation_id = correlation_id or f"schema_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Loading schema",
            "name": name,
            "version": version,
            "format": format.value,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Check built-in schemas first
        if name in self.BUILTIN_SCHEMAS:
            return await self._load_builtin_schema(name, correlation_id)

        # Load from file
        schema_path = self.schema_dir / f"{name}.{format.value}"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                if format == SchemaFormat.JSON:
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)

            # Validate with Pydantic (Task 3.3.3)
            schema_def = SchemaDefinition(**data)

            # Cache schema
            if schema_def.name not in self._schemas:
                self._schemas[schema_def.name] = {}

            target_version = version or schema_def.version
            self._schemas[schema_def.name][target_version] = schema_def

            # Update version history
            if schema_def.name not in self._version_history:
                self._version_history[schema_def.name] = []
            if target_version not in self._version_history[schema_def.name]:
                self._version_history[schema_def.name].append(target_version)

            logger.info({
                "msg": "Schema loaded successfully",
                "name": schema_def.name,
                "version": target_version,
                "hash": schema_def.get_hash(),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return schema_def

        except Exception as e:
            logger.error({
                "msg": "Failed to load schema",
                "error": str(e),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise ValueError(f"Schema validation failed: {e}") from e

    async def _load_builtin_schema(self, name: str, correlation_id: str) -> SchemaDefinition:
        """Load built-in schema from library (Task 3.3.6)."""
        if name not in self.BUILTIN_SCHEMAS:
            raise ValueError(f"Built-in schema not found: {name}")

        data = self.BUILTIN_SCHEMAS[name]
        schema_def = SchemaDefinition(**data)

        logger.info({
            "msg": "Built-in schema loaded",
            "name": name,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return schema_def

    async def save_schema(
        self,
        schema: SchemaDefinition,
        format: SchemaFormat = SchemaFormat.JSON,
        create_version: bool = True,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Save schema to file (Task 3.3.2: Versioning).

        Args:
            schema: Schema definition
            format: File format
            create_version: Create new version if exists
            correlation_id: Correlation ID

        Returns:
            Schema file path
        """
        correlation_id = correlation_id or f"save_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        # Update timestamp
        schema.updated_at = datetime.now(timezone.utc).isoformat()

        # Check if version exists
        schema_path = self.schema_dir / f"{schema.name}.{format.value}"

        if create_version and schema_path.exists():
            # Load existing and increment version
            try:
                existing = await self.load_schema(schema.name, format=format)
                major, minor, patch = map(int, existing.version.split("."))
                schema.version = f"{major}.{minor}.{patch + 1}"
                logger.info({
                    "msg": "Incremented schema version",
                    "old_version": existing.version,
                    "new_version": schema.version,
                    "correlation_id": correlation_id
                })
            except Exception:
                pass

        # Save schema
        with open(schema_path, "w", encoding="utf-8") as f:
            if format == SchemaFormat.JSON:
                json.dump(schema.dict(), f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(schema.dict(), f, default_flow_style=False, allow_unicode=True)

        # Cache schema
        if schema.name not in self._schemas:
            self._schemas[schema.name] = {}
        self._schemas[schema.name][schema.version] = schema

        logger.info({
            "msg": "Schema saved",
            "name": schema.name,
            "version": schema.version,
            "path": str(schema_path),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return str(schema_path)

    async def update_schema(
        self,
        name: str,
        updates: Dict[str, Any],
        create_version: bool = True,
        correlation_id: Optional[str] = None
    ) -> SchemaDefinition:
        """
        Dynamically update schema (Task 3.3.4).

        Args:
            name: Schema name
            updates: Updates to apply
            create_version: Create new version
            correlation_id: Correlation ID

        Returns:
            Updated schema
        """
        correlation_id = correlation_id or f"update_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        # Load current schema
        current = await self.load_schema(name)

        # Apply updates
        current_dict = current.dict()

        for key, value in updates.items():
            if key in current_dict:
                if isinstance(value, dict) and isinstance(current_dict[key], dict):
                    current_dict[key].update(value)
                else:
                    current_dict[key] = value
            else:
                current_dict[key] = value

        # Validate updated schema
        updated = SchemaDefinition(**current_dict)

        # Save updated schema
        await self.save_schema(updated, create_version=create_version, correlation_id=correlation_id)

        logger.info({
            "msg": "Schema updated",
            "name": name,
            "version": updated.version,
            "updates": list(updates.keys()),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return updated

    async def migrate_schema(
        self,
        name: str,
        from_version: str,
        to_version: str,
        migration_steps: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> SchemaDefinition:
        """
        Migrate schema between versions (Task 3.3.5).

        Args:
            name: Schema name
            from_version: Source version
            to_version: Target version
            migration_steps: Migration steps to apply
            correlation_id: Correlation ID

        Returns:
            Migrated schema
        """
        correlation_id = correlation_id or f"migrate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        # Load source schema
        source = await self.load_schema(name, version=from_version)

        logger.info({
            "msg": "Starting schema migration",
            "name": name,
            "from_version": from_version,
            "to_version": to_version,
            "steps": len(migration_steps),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Apply migration steps
        schema_dict = source.dict()

        for step in migration_steps:
            step_type = step.get("type")

            if step_type == "add_entity_type":
                entity_types = schema_dict.get("entity_types", [])
                entity_types.append(step["entity_type"])
                schema_dict["entity_types"] = entity_types

            elif step_type == "remove_entity_type":
                entity_types = schema_dict.get("entity_types", [])
                schema_dict["entity_types"] = [
                    et for et in entity_types if et["name"] != step["entity_name"]
                ]

            elif step_type == "add_relation_type":
                relation_types = schema_dict.get("relation_types", [])
                relation_types.append(step["relation_type"])
                schema_dict["relation_types"] = relation_types

            elif step_type == "rename_entity_type":
                entity_types = schema_dict.get("entity_types", [])
                for et in entity_types:
                    if et["name"] == step["old_name"]:
                        et["name"] = step["new_name"]

            elif step_type == "update_version":
                schema_dict["version"] = to_version

        # Validate migrated schema
        migrated = SchemaDefinition(**schema_dict)
        migrated.version = to_version

        # Save migrated schema
        await self.save_schema(migrated, correlation_id=correlation_id)

        logger.info({
            "msg": "Schema migration completed",
            "name": name,
            "from_version": from_version,
            "to_version": to_version,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return migrated

    async def list_schemas(self) -> List[Dict[str, Any]]:
        """
        List all available schemas.

        Returns:
            List of schema metadata
        """
        schemas = []

        # Built-in schemas
        for name, data in self.BUILTIN_SCHEMAS.items():
            schemas.append({
                "name": name,
                "version": data.get("version", "1.0.0"),
                "description": data.get("description", ""),
                "source": "builtin"
            })

        # File-based schemas
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                schema = await self.load_schema(schema_file.stem)
                schemas.append({
                    "name": schema.name,
                    "version": schema.version,
                    "description": schema.description,
                    "source": "file",
                    "path": str(schema_file)
                })
            except Exception:
                pass

        return schemas

    async def get_schema_versions(self, name: str) -> List[str]:
        """
        Get all versions of a schema (Task 3.3.2).

        Args:
            name: Schema name

        Returns:
            List of version strings
        """
        if name in self._version_history:
            return self._version_history[name].copy()
        return []

    def get_builtin_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get built-in schema library (Task 3.3.6)."""
        return copy.deepcopy(self.BUILTIN_SCHEMAS)
