"""
Schema Mapping: Arbor Graph → Knowledge Engine Unified Graph

Converts Arbor's code graph schema to Knowledge Engine's entity-relationship model.

Following CLAUDE.md principles:
- TYPE SAFETY: Clear mappings with validation
- EXTENSIBILITY: Easy to add new language mappings
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from knowledge_engine.schemas.base import Entity, Relationship, EntityType
from .exceptions import ArborSchemaError

logger = logging.getLogger(__name__)


# Mapping from Arbor node kinds to KE entity types
ARBOR_KIND_TO_ENTITY_TYPE = {
    # Functions
    "function": "code_function",
    "method": "code_method",
    "lambda": "code_lambda",
    
    # Types/Classes
    "class": "code_class",
    "struct": "code_struct",
    "enum": "code_enum",
    "interface": "code_interface",
    "trait": "code_trait",
    "type_alias": "code_type_alias",
    
    # Modules
    "module": "code_module",
    "namespace": "code_namespace",
    "package": "code_package",
    
    # Variables/Constants
    "variable": "code_variable",
    "constant": "code_constant",
    "field": "code_field",
    "property": "code_property",
    
    # Imports/Exports
    "import": "code_import",
    "export": "code_export",
    "use": "code_use",
    
    # Macros/Annotations
    "macro": "code_macro",
    "decorator": "code_decorator",
    "attribute": "code_attribute",
    
    # Special
    "comment": "code_comment",
    "docstring": "code_docstring",
}

# Mapping from Arbor edge kinds to KE relationship types
ARBOR_EDGE_TO_RELATIONSHIP_TYPE = {
    "calls": "code_calls",
    "called_by": "code_called_by",
    "imports": "code_imports",
    "exports": "code_exports",
    "extends": "code_extends",
    "implements": "code_implements",
    "uses_type": "code_uses_type",
    "references": "code_references",
    "contains": "code_contains",
    "returns": "code_returns",
    "parameter": "code_has_parameter",
    "field_of": "code_field_of",
    "method_of": "code_method_of",
    "overrides": "code_overrides",
    "implements_trait": "code_implements_trait",
}


@dataclass
class MappingResult:
    """Result of schema mapping operation."""
    
    entity: Optional[Entity] = None
    relationships: List[Relationship] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
        if self.warnings is None:
            self.warnings = []


class ArborSchemaMapper:
    """
    Maps Arbor graph schema to Knowledge Engine unified schema.
    
    Handles:
    - Node type conversion (Arbor kind → KE entity_type)
    - Property mapping and transformation
    - Edge relationship conversion
    - ID namespacing to avoid collisions
    """
    
    def __init__(self, storage_prefix: str = "arbor"):
        """
        Initialize schema mapper.
        
        Args:
            storage_prefix: Prefix for IDs to namespace Arbor entities
        """
        self.storage_prefix = storage_prefix
        self._kind_mappings = ARBOR_KIND_TO_ENTITY_TYPE.copy()
        self._edge_mappings = ARBOR_EDGE_TO_RELATIONSHIP_TYPE.copy()
        
        logger.info({
            "msg": "ArborSchemaMapper initialized",
            "storage_prefix": storage_prefix
        })
    
    def namespace_id(self, arbor_id: str) -> str:
        """
        Create namespaced ID for Knowledge Engine.
        
        Args:
            arbor_id: Original Arbor node ID
            
        Returns:
            Namespaced ID (e.g., "arbor:node_123")
        """
        return f"{self.storage_prefix}:{arbor_id}"
    
    def extract_arbor_id(self, namespaced_id: str) -> str:
        """
        Extract original Arbor ID from namespaced ID.
        
        Args:
            namespaced_id: Namespaced ID
            
        Returns:
            Original Arbor ID
        """
        prefix = f"{self.storage_prefix}:"
        if namespaced_id.startswith(prefix):
            return namespaced_id[len(prefix):]
        return namespaced_id
    
    def map_node_kind(self, arbor_kind: str) -> str:
        """
        Map Arbor node kind to KE entity type.
        
        Args:
            arbor_kind: Arbor node kind (e.g., "function", "class")
            
        Returns:
            KE entity type
        """
        entity_type = self._kind_mappings.get(arbor_kind)
        if entity_type:
            return entity_type
        
        # Unknown kind - use generic with warning
        logger.warning(f"Unknown Arbor node kind: {arbor_kind}")
        return f"code_{arbor_kind}"
    
    def map_edge_kind(self, arbor_edge_kind: str) -> str:
        """
        Map Arbor edge kind to KE relationship type.
        
        Args:
            arbor_edge_kind: Arbor edge kind (e.g., "calls", "imports")
            
        Returns:
            KE relationship type
        """
        rel_type = self._edge_mappings.get(arbor_edge_kind)
        if rel_type:
            return rel_type
        
        # Unknown edge kind
        logger.warning(f"Unknown Arbor edge kind: {arbor_edge_kind}")
        return f"code_{arbor_edge_kind}"
    
    def convert_arbor_node(self, arbor_node: Dict[str, Any]) -> Entity:
        """
        Convert Arbor node to Knowledge Engine Entity.
        
        Args:
            arbor_node: Arbor node dictionary
            
        Returns:
            Knowledge Engine Entity
            
        Raises:
            ArborSchemaError: If conversion fails
        """
        try:
            # Extract required fields
            node_id = arbor_node.get("id")
            if not node_id:
                raise ArborSchemaError(
                    node_type="unknown",
                    message="Arbor node missing 'id' field",
                    data=arbor_node
                )
            
            name = arbor_node.get("name", node_id)
            kind = arbor_node.get("kind", "unknown")
            
            # Map to KE entity type
            entity_type = self.map_node_kind(kind)
            
            # Build properties
            properties = {
                "arbor_kind": kind,
                "arbor_id": node_id,
                "source_system": "arbor",
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            # Add optional fields if present
            if "qualifiedName" in arbor_node:
                properties["qualified_name"] = arbor_node["qualifiedName"]
            
            if "file" in arbor_node:
                properties["file_path"] = arbor_node["file"]
            
            if "lineStart" in arbor_node:
                properties["location"] = {
                    "line_start": arbor_node["lineStart"],
                    "line_end": arbor_node.get("lineEnd"),
                    "column": arbor_node.get("column")
                }
            
            if "signature" in arbor_node:
                properties["signature"] = arbor_node["signature"]
            
            if "visibility" in arbor_node:
                properties["visibility"] = arbor_node["visibility"]
            
            if "attributes" in arbor_node:
                properties["attributes"] = arbor_node["attributes"]
            
            if "docstring" in arbor_node:
                properties["docstring"] = arbor_node["docstring"]
            
            if "centrality" in arbor_node:
                properties["centrality_score"] = arbor_node["centrality"]
            
            # Add metadata
            metadata = {
                "source_system": "arbor",
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            # Extract language from file extension if available
            if "file" in arbor_node:
                file_path = arbor_node["file"]
                metadata["language"] = self._detect_language(file_path)
            
            return Entity(
                entity_id=self.namespace_id(node_id),
                name=name,
                entity_type=entity_type,
                properties=properties,
                metadata=metadata
            )
            
        except ArborSchemaError:
            raise
        except Exception as e:
            raise ArborSchemaError(
                node_type=arbor_node.get("kind", "unknown"),
                message=f"Failed to convert node: {str(e)}",
                data=arbor_node
            )
    
    def convert_arbor_edge(self, arbor_edge: Dict[str, Any]) -> Relationship:
        """
        Convert Arbor edge to Knowledge Engine Relationship.
        
        Args:
            arbor_edge: Arbor edge dictionary
            
        Returns:
            Knowledge Engine Relationship
        """
        # Extract fields
        from_id = arbor_edge.get("from")
        to_id = arbor_edge.get("to")
        kind = arbor_edge.get("kind", "references")
        
        if not from_id or not to_id:
            raise ArborSchemaError(
                node_type="edge",
                message="Arbor edge missing 'from' or 'to' field",
                data=arbor_edge
            )
        
        # Map relationship type
        rel_type = self.map_edge_kind(kind)
        
        # Build properties
        properties = {
            "arbor_kind": kind,
            "source_system": "arbor"
        }
        
        # Add location if present
        if "location" in arbor_edge:
            properties["location"] = arbor_edge["location"]
        
        return Relationship(
            source_id=self.namespace_id(from_id),
            target_id=self.namespace_id(to_id),
            relationship_type=rel_type,
            properties=properties,
            metadata={"source_system": "arbor"}
        )
    
    def convert_arbor_graph(
        self,
        arbor_graph: Dict[str, Any]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Convert complete Arbor graph to KE entities and relationships.
        
        Args:
            arbor_graph: Complete graph export from Arbor
            
        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        errors = []
        
        # Convert nodes
        for node in arbor_graph.get("nodes", []):
            try:
                entity = self.convert_arbor_node(node)
                entities.append(entity)
            except ArborSchemaError as e:
                errors.append(e.to_dict())
                logger.warning(f"Skipping invalid node: {e}")
        
        # Convert edges
        for edge in arbor_graph.get("edges", []):
            try:
                relationship = self.convert_arbor_edge(edge)
                relationships.append(relationship)
            except ArborSchemaError as e:
                errors.append(e.to_dict())
                logger.warning(f"Skipping invalid edge: {e}")
        
        logger.info({
            "msg": "Converted Arbor graph",
            "entities": len(entities),
            "relationships": len(relationships),
            "errors": len(errors)
        })
        
        return entities, relationships
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to source file
            
        Returns:
            Language name or None
        """
        extension_map = {
            ".py": "python",
            ".rs": "rust",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".dart": "dart",
        }
        
        file_lower = file_path.lower()
        for ext, lang in extension_map.items():
            if file_lower.endswith(ext):
                return lang
        
        return None
    
    def add_custom_mapping(self, arbor_kind: str, entity_type: str) -> None:
        """
        Add custom node kind mapping.
        
        Args:
            arbor_kind: Arbor node kind
            entity_type: KE entity type
        """
        self._kind_mappings[arbor_kind] = entity_type
        logger.info(f"Added custom mapping: {arbor_kind} -> {entity_type}")
    
    def add_custom_edge_mapping(self, arbor_edge: str, rel_type: str) -> None:
        """
        Add custom edge kind mapping.
        
        Args:
            arbor_edge: Arbor edge kind
            rel_type: KE relationship type
        """
        self._edge_mappings[arbor_edge] = rel_type
        logger.info(f"Added custom edge mapping: {arbor_edge} -> {rel_type}")


# Convenience function for one-off conversions
def convert_arbor_node(arbor_node: Dict[str, Any], prefix: str = "arbor") -> Entity:
    """
    Convert single Arbor node to Entity.
    
    Args:
        arbor_node: Arbor node dictionary
        prefix: ID prefix
        
    Returns:
        Knowledge Engine Entity
    """
    mapper = ArborSchemaMapper(storage_prefix=prefix)
    return mapper.convert_arbor_node(arbor_node)


def convert_arbor_edge(arbor_edge: Dict[str, Any], prefix: str = "arbor") -> Relationship:
    """
    Convert single Arbor edge to Relationship.
    
    Args:
        arbor_edge: Arbor edge dictionary
        prefix: ID prefix
        
    Returns:
        Knowledge Engine Relationship
    """
    mapper = ArborSchemaMapper(storage_prefix=prefix)
    return mapper.convert_arbor_edge(arbor_edge)
