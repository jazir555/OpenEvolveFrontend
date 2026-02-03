"""
KG-Gen Pipeline Mock for Contract Testing

This provides mock classes for testing contracts without requiring
the actual kg-gen installation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class KnowledgeGraph:
    """
    Mock Knowledge Graph for KG-Gen pipeline contract testing.
    """

    entities: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    def add_entity(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an entity to the graph."""
        if name not in self.entities:
            self.entities[name] = attributes or {}

    def merge(self, other: 'KnowledgeGraph'):
        """Merge another graph into this one."""
        self.entities.update(other.entities)
        self.relationships.extend(other.relationships)

    def get_entities(self) -> List[str]:
        """Get all entity names."""
        return list(self.entities.keys())


@dataclass
class UploadResult:
    """
    Result of uploading knowledge graph to Neo4j.
    """

    success: bool
    entities_uploaded: int
    relationships_uploaded: int
    error: Optional[str] = None
    duration_ms: float = 0.0
