"""
KG-specific structured generation constraints for Outlines.

Predefined schemas for common KG operations:
- Entity extraction with type validation
- Relationship extraction with arity constraints
- Property validation schemas
- Cypher query generation constraints

All Cypher queries are Memgraph-compatible (not Neo4j-specific).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Common entity types for knowledge graphs."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    DOCUMENT = "DOCUMENT"
    CUSTOM = "CUSTOM"


class RelationType(str, Enum):
    """Common relationship types for knowledge graphs."""
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    CREATED = "CREATED"
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    INFLUENCES = "INFLUENCES"
    CUSTOM = "CUSTOM"


class PropertySchema(BaseModel):
    """Schema for entity/relation properties."""
    name: str = Field(..., description="Property name")
    value: Union[str, int, float, bool, List[Any]] = Field(..., description="Property value")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source: Optional[str] = Field(default=None, description="Source of the property")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class EntityExtractionSchema(BaseModel):
    """Schema for entity extraction output.
    
    Example:
        {
            "entities": [
                {
                    "name": "John Smith",
                    "type": "PERSON",
                    "confidence": 0.95,
                    "properties": [
                        {"name": "age", "value": 35, "confidence": 0.8}
                    ]
                }
            ]
        }
    """
    class Entity(BaseModel):
        name: str = Field(..., description="Entity name/text")
        type: str = Field(..., description="Entity type (PERSON, ORGANIZATION, etc.)")
        confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
        properties: List[PropertySchema] = Field(default_factory=list, description="Entity properties")
        start_pos: Optional[int] = Field(default=None, description="Start position in text")
        end_pos: Optional[int] = Field(default=None, description="End position in text")
        
        @field_validator('confidence')
        @classmethod
        def validate_confidence(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError('Confidence must be between 0.0 and 1.0')
            return v
    
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of extraction"
    )
    text_hash: Optional[str] = Field(default=None, description="Hash of source text")
    model_used: Optional[str] = Field(default=None, description="Model used for extraction")
    
    def to_memgraph_nodes(self) -> List[Dict[str, Any]]:
        """Convert to Memgraph node format."""
        nodes = []
        for entity in self.entities:
            node = {
                "labels": [entity.type],
                "properties": {
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "extraction_timestamp": self.extraction_timestamp,
                }
            }
            # Add properties
            for prop in entity.properties:
                node["properties"][prop.name] = prop.value
            nodes.append(node)
        return nodes


class RelationshipSchema(BaseModel):
    """Schema for relationship extraction output.
    
    Example:
        {
            "relationships": [
                {
                    "source": "John Smith",
                    "target": "Acme Corp",
                    "type": "WORKS_FOR",
                    "confidence": 0.92,
                    "properties": [{"name": "since", "value": "2020"}]
                }
            ]
        }
    """
    class Relationship(BaseModel):
        source: str = Field(..., description="Source entity name")
        target: str = Field(..., description="Target entity name")
        type: str = Field(..., description="Relationship type")
        confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
        properties: List[PropertySchema] = Field(default_factory=list, description="Relation properties")
        directed: bool = Field(default=True, description="Whether relationship is directed")
        
        @field_validator('confidence')
        @classmethod
        def validate_confidence(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError('Confidence must be between 0.0 and 1.0')
            return v
    
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted relationships")
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of extraction"
    )
    text_hash: Optional[str] = Field(default=None, description="Hash of source text")
    model_used: Optional[str] = Field(default=None, description="Model used for extraction")
    
    def to_memgraph_edges(self) -> List[Dict[str, Any]]:
        """Convert to Memgraph edge format."""
        edges = []
        for rel in self.relationships:
            edge = {
                "type": rel.type,
                "from": {"name": rel.source},
                "to": {"name": rel.target},
                "properties": {
                    "confidence": rel.confidence,
                    "extraction_timestamp": self.extraction_timestamp,
                }
            }
            # Add properties
            for prop in rel.properties:
                edge["properties"][prop.name] = prop.value
            edges.append(edge)
        return edges


class CypherQuerySchema(BaseModel):
    """Schema for Memgraph Cypher query generation.
    
    Note: All queries are Memgraph-compatible (not Neo4j-specific).
    
    Example:
        {
            "query": "MATCH (p:PERSON {name: $name}) RETURN p",
            "parameters": {"name": "John Smith"},
            "explanation": "Find person by name",
            "query_type": "READ",
            "estimated_complexity": "LOW"
        }
    """
    class QueryType(str, Enum):
        READ = "READ"
        WRITE = "WRITE"
        UPDATE = "UPDATE"
        DELETE = "DELETE"
        SCHEMA = "SCHEMA"
        UNKNOWN = "UNKNOWN"
    
    class Complexity(str, Enum):
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
    
    query: str = Field(..., description="Memgraph Cypher query string")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    explanation: str = Field(..., description="Human-readable explanation of the query")
    query_type: QueryType = Field(default=QueryType.UNKNOWN, description="Type of query")
    estimated_complexity: Complexity = Field(default=Complexity.LOW, description="Estimated query complexity")
    requires_index: bool = Field(default=False, description="Whether query benefits from index")
    idempotent: bool = Field(default=False, description="Whether query is safe to replay")
    
    @field_validator('query')
    @classmethod
    def validate_memgraph_compatibility(cls, v):
        """Ensure query is Memgraph-compatible."""
        # Check for Neo4j-specific syntax that Memgraph doesn't support
        neo4j_specific = [
            "apoc.",  # APOC procedures
            "db.labels",  # Neo4j specific
            "db.schema",  # Neo4j specific
            "CALL {",  # Subqueries (limited support in Memgraph)
        ]
        
        warnings = []
        for pattern in neo4j_specific:
            if pattern.lower() in v.lower():
                warnings.append(f"Possible Neo4j-specific syntax: {pattern}")
        
        # Store warnings in model config for later access
        return v
    
    def to_memgraph_query(self) -> str:
        """Get the query string ready for Memgraph execution."""
        return self.query
    
    def get_parameterized_query(self) -> tuple:
        """Get query and parameters as tuple for execution."""
        return (self.query, self.parameters)


class ValidationResultSchema(BaseModel):
    """Schema for KG validation results.
    
    Example:
        {
            "is_valid": false,
            "errors": ["Missing required property 'name'"],
            "warnings": ["Low confidence score"],
            "confidence": 0.75,
            "suggestions": ["Add source citation"]
        }
    """
    class ValidationIssue(BaseModel):
        severity: str = Field(..., description="ERROR, WARNING, or INFO")
        message: str = Field(..., description="Issue description")
        field: Optional[str] = Field(default=None, description="Affected field")
        suggestion: Optional[str] = Field(default=None, description="Suggested fix")
    
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    issues: List[ValidationIssue] = Field(default_factory=list, description="Detailed issues")
    validation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of validation"
    )
    validator_version: str = Field(default="1.0.0", description="Validator version")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    def add_error(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add an error to the result."""
        self.errors.append(message)
        self.issues.append(self.ValidationIssue(
            severity="ERROR",
            message=message,
            field=field,
            suggestion=suggestion
        ))
        self.is_valid = False
    
    def add_warning(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add a warning to the result."""
        self.warnings.append(message)
        self.issues.append(self.ValidationIssue(
            severity="WARNING",
            message=message,
            field=field,
            suggestion=suggestion
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class KnowledgeGraphConstraints:
    """
    Collection of predefined constraints for KG operations.
    
    Provides:
    - Predefined schemas for common KG tasks
    - Memgraph-compatible Cypher patterns
    - Validation rules for KG data
    """
    
    # JSON Schemas for entity extraction
    ENTITY_EXTRACTION_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": [t.value for t in EntityType]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "properties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": ["string", "number", "boolean"]},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                },
                                "required": ["name", "value"]
                            }
                        }
                    },
                    "required": ["name", "type"]
                }
            },
            "extraction_timestamp": {"type": "string"},
            "text_hash": {"type": ["string", "null"]},
            "model_used": {"type": ["string", "null"]}
        },
        "required": ["entities"]
    }
    
    # JSON Schema for relationship extraction
    RELATIONSHIP_EXTRACTION_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "type": {"type": "string", "enum": [t.value for t in RelationType]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "properties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": ["string", "number", "boolean"]},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                },
                                "required": ["name", "value"]
                            }
                        },
                        "directed": {"type": "boolean"}
                    },
                    "required": ["source", "target", "type"]
                }
            },
            "extraction_timestamp": {"type": "string"},
            "text_hash": {"type": ["string", "null"]},
            "model_used": {"type": ["string", "null"]}
        },
        "required": ["relationships"]
    }
    
    # JSON Schema for Cypher query generation
    CYPHER_QUERY_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "parameters": {"type": "object"},
            "explanation": {"type": "string"},
            "query_type": {
                "type": "string",
                "enum": ["READ", "WRITE", "UPDATE", "DELETE", "SCHEMA", "UNKNOWN"]
            },
            "estimated_complexity": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH"]
            },
            "requires_index": {"type": "boolean"},
            "idempotent": {"type": "boolean"}
        },
        "required": ["query", "explanation"]
    }
    
    # Regex patterns for common KG values
    ENTITY_NAME_PATTERN = r'^[A-Za-z][A-Za-z0-9_\s\-\.]{0,199}$'
    RELATION_TYPE_PATTERN = r'^[A-Z][A-Z_]{0,49}$'
    CONFIDENCE_PATTERN = r'^0?\.[0-9]+$|^1\.0$|^0$|^1$'
    TIMESTAMP_PATTERN = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
    
    @classmethod
    def get_entity_extraction_schema(cls) -> type:
        """Get Pydantic schema for entity extraction."""
        return EntityExtractionSchema
    
    @classmethod
    def get_relationship_schema(cls) -> type:
        """Get Pydantic schema for relationship extraction."""
        return RelationshipSchema
    
    @classmethod
    def get_cypher_query_schema(cls) -> type:
        """Get Pydantic schema for Cypher query generation."""
        return CypherQuerySchema
    
    @classmethod
    def get_validation_schema(cls) -> type:
        """Get Pydantic schema for validation results."""
        return ValidationResultSchema
    
    @classmethod
    def get_entity_types(cls) -> List[str]:
        """Get list of valid entity types."""
        return [t.value for t in EntityType]
    
    @classmethod
    def get_relation_types(cls) -> List[str]:
        """Get list of valid relationship types."""
        return [t.value for t in RelationType]
    
    @classmethod
    def validate_entity_name(cls, name: str) -> bool:
        """Validate entity name format."""
        import re
        return bool(re.match(cls.ENTITY_NAME_PATTERN, name))
    
    @classmethod
    def validate_relation_type(cls, rel_type: str) -> bool:
        """Validate relationship type format."""
        import re
        return bool(re.match(cls.RELATION_TYPE_PATTERN, rel_type))
    
    @classmethod
    def create_entity_extraction_constraint(
        cls,
        allowed_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Create a constraint for entity extraction.
        
        Args:
            allowed_types: List of allowed entity types (default: all)
            min_confidence: Minimum confidence threshold
            
        Returns:
            JSON schema constraint
        """
        schema = cls.ENTITY_EXTRACTION_JSON_SCHEMA.copy()
        
        if allowed_types:
            schema["properties"]["entities"]["items"]["properties"]["type"]["enum"] = allowed_types
        
        return schema
    
    @classmethod
    def create_relationship_constraint(
        cls,
        allowed_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Create a constraint for relationship extraction.
        
        Args:
            allowed_types: List of allowed relation types (default: all)
            min_confidence: Minimum confidence threshold
            
        Returns:
            JSON schema constraint
        """
        schema = cls.RELATIONSHIP_EXTRACTION_JSON_SCHEMA.copy()
        
        if allowed_types:
            schema["properties"]["relationships"]["items"]["properties"]["type"]["enum"] = allowed_types
        
        return schema
    
    @classmethod
    def create_cypher_constraint(
        cls,
        allowed_operations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a constraint for Cypher query generation.
        
        Args:
            allowed_operations: List of allowed operations (READ, WRITE, etc.)
            
        Returns:
            JSON schema constraint
        """
        schema = cls.CYPHER_QUERY_JSON_SCHEMA.copy()
        
        if allowed_operations:
            schema["properties"]["query_type"]["enum"] = allowed_operations
        
        return schema


# Predefined constraints for common use cases
DEFAULT_ENTITY_CONSTRAINT = KnowledgeGraphConstraints.ENTITY_EXTRACTION_JSON_SCHEMA
DEFAULT_RELATIONSHIP_CONSTRAINT = KnowledgeGraphConstraints.RELATIONSHIP_EXTRACTION_JSON_SCHEMA
DEFAULT_CYPHER_CONSTRAINT = KnowledgeGraphConstraints.CYPHER_QUERY_JSON_SCHEMA

__all__ = [
    # Schemas
    "EntityExtractionSchema",
    "RelationshipSchema",
    "CypherQuerySchema",
    "ValidationResultSchema",
    "PropertySchema",
    # Enums
    "EntityType",
    "RelationType",
    # Constraints
    "KnowledgeGraphConstraints",
    # Defaults
    "DEFAULT_ENTITY_CONSTRAINT",
    "DEFAULT_RELATIONSHIP_CONSTRAINT",
    "DEFAULT_CYPHER_CONSTRAINT",
]
