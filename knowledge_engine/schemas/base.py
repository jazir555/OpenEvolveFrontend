"""
Base schema classes for the Knowledge Engine Schema System.

Defines the core data structures for entity schemas, types, and relationships.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import re


class PropertyType(Enum):
    """Supported property types for entity attributes."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


@dataclass
class PropertyDefinition:
    """
    Defines a property on an entity type.

    Attributes:
        name: Property name
        type: Property data type
        required: Whether this property is required
        description: Human-readable description
        default_value: Default value if not provided
        allowed_values: For enum types, list of allowed values
        validation_pattern: Regex pattern for string validation
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        min_length: Minimum length for strings/arrays
        max_length: Maximum length for strings/arrays
    """
    name: str
    type: PropertyType
    required: bool = False
    description: str = ""
    default_value: Any = None
    allowed_values: Optional[List[Any]] = None
    validation_pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a value against this property definition.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required
        if value is None:
            if self.required:
                return False, f"Required property '{self.name}' is missing"
            return True, None

        # Check type
        if self.type == PropertyType.STRING:
            if not isinstance(value, str):
                return False, f"Property '{self.name}' must be a string"
            if self.validation_pattern and not re.match(self.validation_pattern, value):
                return False, f"Property '{self.name}' does not match required pattern"
            if self.min_length and len(value) < self.min_length:
                return False, f"Property '{self.name}' length below minimum ({self.min_length})"
            if self.max_length and len(value) > self.max_length:
                return False, f"Property '{self.name}' length exceeds maximum ({self.max_length})"

        elif self.type == PropertyType.INTEGER:
            if not isinstance(value, int):
                return False, f"Property '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Property '{self.name}' below minimum value"
            if self.max_value is not None and value > self.max_value:
                return False, f"Property '{self.name}' exceeds maximum value"

        elif self.type == PropertyType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Property '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Property '{self.name}' below minimum value"
            if self.max_value is not None and value > self.max_value:
                return False, f"Property '{self.name}' exceeds maximum value"

        elif self.type == PropertyType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Property '{self.name}' must be a boolean"

        elif self.type == PropertyType.ARRAY:
            if not isinstance(value, list):
                return False, f"Property '{self.name}' must be an array"
            if self.min_length and len(value) < self.min_length:
                return False, f"Property '{self.name}' length below minimum"
            if self.max_length and len(value) > self.max_length:
                return False, f"Property '{self.name}' length exceeds maximum"

        elif self.type == PropertyType.ENUM:
            if self.allowed_values and value not in self.allowed_values:
                return False, f"Property '{self.name}' value not in allowed values: {self.allowed_values}"

        return True, None


@dataclass
class ValidationRule:
    """
    A custom validation rule for an entity type.

    Attributes:
        name: Rule name
        description: Human-readable description
        validator: Function that takes an entity and returns (is_valid, error_message)
        severity: 'error', 'warning', or 'info'
    """
    name: str
    description: str
    validator: Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]
    severity: str = "error"  # error, warning, info


@dataclass
class Entity:
    """
    Represents an individual entity instance.

    Attributes:
        entity_id: Unique identifier
        entity_type: Type name
        properties: Property values
        metadata: Additional metadata
        source: Source system/domain
        confidence: Confidence score (0-1)
    """
    entity_id: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "metadata": self.metadata,
            "source": self.source,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary representation."""
        return cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class Relationship:
    """
    Represents a relationship between two entities.

    Attributes:
        relationship_id: Unique identifier
        source_entity_id: Source entity ID
        target_entity_id: Target entity ID
        relationship_type: Type name
        properties: Property values
        metadata: Additional metadata
        confidence: Confidence score (0-1)
    """
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "relationship_id": self.relationship_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type,
            "properties": self.properties,
            "metadata": self.metadata,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create relationship from dictionary representation."""
        return cls(
            relationship_id=data["relationship_id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            relationship_type=data["relationship_type"],
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class EntityType:
    """
    Defines an entity type in a schema.

    Attributes:
        name: Type name
        description: Human-readable description
        properties: Property definitions
        validation_rules: Custom validation rules
        examples: Example entities
        base_type: Optional parent type for inheritance
    """
    name: str
    properties: Dict[str, PropertyDefinition] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    base_type: Optional[str] = None

    def get_all_properties(self) -> Dict[str, PropertyDefinition]:
        """Get all properties including inherited ones."""
        # Note: In a full implementation, this would merge with base_type properties
        return self.properties.copy()

    def validate(self, entity_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate entity data against this type definition.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Validate properties
        all_properties = self.get_all_properties()
        for prop_name, prop_def in all_properties.items():
            value = entity_data.get(prop_name)
            is_valid, error_msg = prop_def.validate(value)
            if not is_valid:
                errors.append(f"  - {error_msg}")

        # Run custom validation rules
        for rule in self.validation_rules:
            is_valid, error_msg = rule.validator(entity_data)
            if not is_valid:
                errors.append(f"  - [{rule.severity.upper()}] {error_msg}")

        return len(errors) == 0, errors


@dataclass
class RelationshipType:
    """
    Defines a relationship type in a schema.

    Attributes:
        name: Relationship type name
        description: Human-readable description
        source_types: Allowed source entity types
        target_types: Allowed target entity types
        properties: Property definitions
        inverse_relationship: Optional inverse relationship name
        directed: Whether the relationship is directed
    """
    name: str
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)
    properties: Dict[str, PropertyDefinition] = field(default_factory=dict)
    inverse_relationship: Optional[str] = None
    directed: bool = True
    description: str = ""

    def validate(
        self,
        relationship_data: Dict[str, Any],
        source_entity_type: str,
        target_entity_type: str
    ) -> tuple[bool, List[str]]:
        """
        Validate relationship data against this type definition.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Check source type
        if source_entity_type not in self.source_types:
            errors.append(
                f"Source entity type '{source_entity_type}' not in allowed types: {self.source_types}"
            )

        # Check target type
        if target_entity_type not in self.target_types:
            errors.append(
                f"Target entity type '{target_entity_type}' not in allowed types: {self.target_types}"
            )

        # Validate properties
        for prop_name, prop_def in self.properties.items():
            value = relationship_data.get(prop_name)
            is_valid, error_msg = prop_def.validate(value)
            if not is_valid:
                errors.append(f"  - {error_msg}")

        return len(errors) == 0, errors


@dataclass
class EntitySchema:
    """
    Complete schema definition for a domain.

    Attributes:
        domain: Domain name
        description: Schema description
        entity_types: Entity type definitions
        relationship_types: Relationship type definitions
        metadata: Additional metadata
        version: Schema version
    """
    domain: str
    entity_types: Dict[str, EntityType] = field(default_factory=dict)
    relationship_types: Dict[str, RelationshipType] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0.0"

    def get_entity_type(self, type_name: str) -> Optional[EntityType]:
        """Get entity type by name."""
        return self.entity_types.get(type_name)

    def get_relationship_type(self, type_name: str) -> Optional[RelationshipType]:
        """Get relationship type by name."""
        return self.relationship_types.get(type_name)

    def list_entity_types(self) -> List[str]:
        """List all entity type names."""
        return list(self.entity_types.keys())

    def list_relationship_types(self) -> List[str]:
        """List all relationship type names."""
        return list(self.relationship_types.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "domain": self.domain,
            "description": self.description,
            "version": self.version,
            "entity_types": {
                name: {
                    "description": et.description,
                    "properties": {
                        pname: {
                            "type": pdef.type.value,
                            "required": pdef.required,
                            "description": pdef.description
                        }
                        for pname, pdef in et.properties.items()
                    }
                }
                for name, et in self.entity_types.items()
            },
            "relationship_types": {
                name: {
                    "description": rt.description,
                    "source_types": rt.source_types,
                    "target_types": rt.target_types,
                    "directed": rt.directed
                }
                for name, rt in self.relationship_types.items()
            },
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntitySchema':
        """Create schema from dictionary representation."""
        schema = cls(
            domain=data["domain"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {})
        )

        # Reconstruct entity types
        for name, et_data in data.get("entity_types", {}).items():
            properties = {}
            for pname, pdata in et_data.get("properties", {}).items():
                prop_type = PropertyType(pdata["type"])
                properties[pname] = PropertyDefinition(
                    name=pname,
                    type=prop_type,
                    required=pdata.get("required", False),
                    description=pdata.get("description", "")
                )

            schema.entity_types[name] = EntityType(
                name=name,
                properties=properties,
                description=et_data.get("description", "")
            )

        # Reconstruct relationship types
        for name, rt_data in data.get("relationship_types", {}).items():
            schema.relationship_types[name] = RelationshipType(
                name=name,
                source_types=rt_data.get("source_types", []),
                target_types=rt_data.get("target_types", []),
                directed=rt_data.get("directed", True),
                description=rt_data.get("description", "")
            )

        return schema
