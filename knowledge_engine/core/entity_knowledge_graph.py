"""
EntityKnowledgeGraph - Core entity graph implementation

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Verify operations succeed
- IDEMPOTENCY: All operations safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs

Uses unified Entity and Relationship from knowledge_engine.schemas.base.

Author: OpenEvolve Distinguished Engineer
Version: 2.1.0
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from threading import Lock, RLock
import uuid
import re

# Import unified models - use relative import
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from schemas.base import (
    Entity,
    Relationship,
    ValidationResult,
)


logger = logging.getLogger(__name__)

# Re-export unified Entity and Relationship for backward compatibility
__all__ = [
    "Entity",
    "Relationship",
    "EntityKnowledgeGraph",
    "ValidationResult",
]


class EntityKnowledgeGraph:
    """
    Thread-safe in-memory entity knowledge graph.
    
    Features:
    - Add entities with attributes
    - Add relationships between entities
    - Search/query entities
    - Serialize to/from JSON
    - Thread-safe operations
    - Idempotent operations
    - Structured logging with correlation IDs
    
    Uses unified Entity and Relationship models from schemas.base.
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize the knowledge graph.

        Args:
            correlation_id: Optional correlation ID for logging
        """
        self._entities: Dict[str, Entity] = {}
        self._relationships: List[Relationship] = []
        self._entity_types: Dict[str, Set[str]] = {}  # entity_type -> set of entity names
        self._lock = RLock()  # Use RLock (reentrant) to allow nested lock acquisition
        self._async_lock: Optional[asyncio.Lock] = None  # Lazy initialization

        # Correlation ID for structured logging
        self._correlation_id = correlation_id or str(uuid.uuid4())

        logger.info({
            "msg": "EntityKnowledgeGraph initialized",
            "correlation_id": self._correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_async_lock(self) -> asyncio.Lock:
        """Get or create async lock (lazy initialization for proper event loop binding)."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _log(self, level: str, message: str, **kwargs):
        """
        Structured logging with correlation ID.

        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
            **kwargs: Additional context
        """
        log_data = {
            "msg": message,
            "correlation_id": self._correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }

        log_func = getattr(logger, level, logger.info)
        log_func(json.dumps(log_data))

    @property
    def entities(self) -> Dict[str, Entity]:
        """
        Get all entities (backward compatibility property).

        Returns:
            Dictionary mapping entity names to Entity objects
        """
        with self._lock:
            return self._entities.copy()

    def add_entity(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        entity: Optional[Entity] = None  # Support passing Entity object directly
    ) -> bool:
        """
        Add an entity to the graph (synchronous).

        IDEMPOTENT: If entity exists, attributes are merged.

        Args:
            name: Unique entity identifier (maps to entity_id in unified model)
            entity_type: Type/category of entity (defaults to "Entity" if not provided)
            attributes: Optional key-value pairs (maps to properties in unified model)
            entity: Optional Entity object to add directly (takes precedence if provided)

        Returns:
            True if entity was added or updated, False on error
        """
        try:
            # Support passing Entity object directly
            if entity is not None:
                if not isinstance(entity, Entity):
                    raise ValueError("entity parameter must be an Entity instance")
                # Use entity.name (the human-readable name) as the key, not entity_id
                name = entity.name if entity.name else entity.entity_id
                entity_type = entity.entity_type.value if isinstance(entity.entity_type, Enum) else entity.entity_type
                attributes = entity.properties
            elif name is None:
                raise ValueError("Either name or entity parameter must be provided")

            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Entity name must be a non-empty string")

            # Default entity_type if not provided
            if not entity_type:
                entity_type = "Entity"

            if not isinstance(entity_type, str):
                # Convert enum to string if needed
                if isinstance(entity_type, Enum):
                    entity_type = entity_type.value
                else:
                    raise ValueError("Entity type must be a string")

            attributes = attributes or {}

            with self._lock:
                # Check if entity exists
                if name in self._entities:
                    # Merge attributes (idempotent update)
                    existing = self._entities[name]
                    existing.properties.update(attributes)
                    existing.updated_at = datetime.now(timezone.utc)

                    self._log("info", f"Updated entity: {name}", entity_type=entity_type)
                    return True
                else:
                    # Create new entity using unified model
                    new_entity = Entity(
                        entity_id=name,
                        name=name,
                        entity_type=entity_type,
                        properties=attributes,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    self._entities[name] = new_entity

                    # Update type index
                    if entity_type not in self._entity_types:
                        self._entity_types[entity_type] = set()
                    self._entity_types[entity_type].add(name)

                    self._log("info", f"Added entity: {name}", entity_type=entity_type)
                    return True

        except Exception as e:
            self._log("error", f"Failed to add entity: {name if name else 'unknown'}", error=str(e))
            return False

    async def add_entity_async(
        self,
        name: str,
        entity_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an entity to the graph (asynchronous).

        IDEMPOTENT: If entity exists, attributes are merged.

        Args:
            name: Unique entity identifier
            entity_type: Type/category of entity (defaults to "Entity" if not provided)
            attributes: Optional key-value pairs

        Returns:
            True if entity was added or updated, False on error
        """
        try:
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Entity name must be a non-empty string")

            # Default entity_type if not provided
            if not entity_type:
                entity_type = "Entity"

            if not isinstance(entity_type, str):
                raise ValueError("Entity type must be a string")

            attributes = attributes or {}

            async with self._get_async_lock():
                # Check if entity exists
                if name in self._entities:
                    # Merge attributes (idempotent update)
                    existing = self._entities[name]
                    existing.properties.update(attributes)
                    existing.updated_at = datetime.now(timezone.utc)

                    self._log("info", f"Updated entity: {name}", entity_type=entity_type)
                    return True
                else:
                    # Create new entity using unified model
                    entity = Entity(
                        entity_id=name,
                        name=name,
                        entity_type=entity_type,
                        properties=attributes,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    self._entities[name] = entity

                    # Update type index
                    if entity_type not in self._entity_types:
                        self._entity_types[entity_type] = set()
                    self._entity_types[entity_type].add(name)

                    self._log("info", f"Added entity: {name}", entity_type=entity_type)
                    return True

        except Exception as e:
            self._log("error", f"Failed to add entity: {name}", error=str(e))
            return False

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a relationship between entities (synchronous).

        IDEMPOTENT: Duplicate relationships are ignored.

        Args:
            source: Source entity name (maps to source_entity_id)
            target: Target entity name (maps to target_entity_id)
            relation_type: Type of relationship (deprecated, use relationship_type)
            relationship_type: Type of relationship
            attributes: Optional relationship properties (maps to properties)

        Returns:
            True if relationship was added, False on error or duplicate
        """
        try:
            # Support both parameter names for backward compatibility
            rel_type = relationship_type or relation_type
            if not rel_type:
                raise ValueError("Either relation_type or relationship_type must be provided")

            # Validate inputs
            if not source or not target:
                raise ValueError("Source and target must be non-empty strings")

            attributes = attributes or {}

            with self._lock:
                # Ensure entities exist (create empty ones if not)
                if source not in self._entities:
                    self.add_entity(source, "unknown")
                if target not in self._entities:
                    self.add_entity(target, "unknown")

                # Check for duplicate
                new_rel = Relationship(
                    source_entity_id=source,
                    target_entity_id=target,
                    relationship_type=rel_type,
                    properties=attributes
                )

                if new_rel in self._relationships:
                    self._log("debug", f"Relationship already exists: {source} -> {target}")
                    return True  # Idempotent - success even if exists

                # Add relationship
                self._relationships.append(new_rel)

                self._log("info", f"Added relationship: {source} -> {target}",
                          relation_type=rel_type)
                return True

        except Exception as e:
            self._log("error", f"Failed to add relationship: {source} -> {target}",
                     error=str(e))
            return False

    async def add_relationship_async(
        self,
        source: str,
        target: str,
        relation_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a relationship between entities (asynchronous).

        IDEMPOTENT: Duplicate relationships are ignored.

        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Type of relationship (deprecated, use relationship_type)
            relationship_type: Type of relationship
            attributes: Optional relationship properties

        Returns:
            True if relationship was added, False on error or duplicate
        """
        try:
            # Support both parameter names for backward compatibility
            rel_type = relationship_type or relation_type
            if not rel_type:
                raise ValueError("Either relation_type or relationship_type must be provided")

            # Validate inputs
            if not source or not target:
                raise ValueError("Source and target must be non-empty strings")

            attributes = attributes or {}

            async with self._get_async_lock():
                # Ensure entities exist (create empty ones if not)
                if source not in self._entities:
                    await self.add_entity_async(source, "unknown")
                if target not in self._entities:
                    await self.add_entity_async(target, "unknown")

                # Check for duplicate
                new_rel = Relationship(
                    source_entity_id=source,
                    target_entity_id=target,
                    relationship_type=rel_type,
                    properties=attributes
                )

                if new_rel in self._relationships:
                    self._log("debug", f"Relationship already exists: {source} -> {target}")
                    return True  # Idempotent - success even if exists

                # Add relationship
                self._relationships.append(new_rel)

                self._log("info", f"Added relationship: {source} -> {target}",
                          relation_type=rel_type)
                return True

        except Exception as e:
            self._log("error", f"Failed to add relationship: {source} -> {target}",
                     error=str(e))
            return False

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by name (synchronous).

        Args:
            name: Entity name

        Returns:
            Entity dictionary or None if not found
        """
        with self._lock:
            entity = self._entities.get(name)
            return entity.to_dict() if entity else None

    async def get_entity_async(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by name (asynchronous).

        Args:
            name: Entity name

        Returns:
            Entity dictionary or None if not found
        """
        async with self._get_async_lock():
            entity = self._entities.get(name)
            return entity.to_dict() if entity else None

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities matching criteria (synchronous).

        Args:
            entity_type: Filter by entity type
            attributes: Filter by attribute key-value pairs

        Returns:
            List of matching entity dictionaries
        """
        with self._lock:
            results = []

            for entity in self._entities.values():
                # Filter by type
                if entity_type and entity.entity_type != entity_type:
                    continue

                # Filter by attributes
                if attributes:
                    match = True
                    for key, value in attributes.items():
                        if key not in entity.properties or entity.properties[key] != value:
                            match = False
                            break
                    if not match:
                        continue

                results.append(entity.to_dict())

            return results

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Get all entities in the graph (synchronous).

        Returns:
            List of all entity dictionaries
        """
        with self._lock:
            return [entity.to_dict() for entity in self._entities.values()]

    async def get_all_entities_async(self) -> List[Dict[str, Any]]:
        """
        Get all entities in the graph (asynchronous).

        Returns:
            List of all entity dictionaries
        """
        async with self._get_async_lock():
            return [entity.to_dict() for entity in self._entities.values()]

    def get_related_entities(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to the given entity (synchronous).

        Args:
            entity_name: Name of the entity
            relation_type: Optional filter by relationship type
            direction: "incoming", "outgoing", or "both" (default)

        Returns:
            List of related entity dictionaries
        """
        with self._lock:
            related = []

            for rel in self._relationships:
                # Check if this relationship involves the entity
                source_matches = rel.source_entity_id == entity_name
                target_matches = rel.target_entity_id == entity_name

                if not (source_matches or target_matches):
                    continue

                # Filter by relationship type if specified
                if relation_type:
                    rel_type = rel.relationship_type.value if isinstance(rel.relationship_type, Enum) else rel.relationship_type
                    if rel_type != relation_type:
                        continue

                # Check direction
                if direction == "outgoing" and not source_matches:
                    continue
                if direction == "incoming" and not target_matches:
                    continue

                # Add the related entity
                if source_matches and rel.target_entity_id in self._entities:
                    related.append(self._entities[rel.target_entity_id].to_dict())
                elif target_matches and rel.source_entity_id in self._entities:
                    related.append(self._entities[rel.source_entity_id].to_dict())

            return related

    async def get_related_entities_async(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to the given entity (asynchronous).

        Args:
            entity_name: Name of the entity
            relation_type: Optional filter by relationship type
            direction: "incoming", "outgoing", or "both" (default)

        Returns:
            List of related entity dictionaries
        """
        async with self._get_async_lock():
            related = []

            for rel in self._relationships:
                # Check if this relationship involves the entity
                source_matches = rel.source_entity_id == entity_name
                target_matches = rel.target_entity_id == entity_name

                if not (source_matches or target_matches):
                    continue

                # Filter by relationship type if specified
                if relation_type:
                    rel_type = rel.relationship_type.value if isinstance(rel.relationship_type, Enum) else rel.relationship_type
                    if rel_type != relation_type:
                        continue

                # Check direction
                if direction == "outgoing" and not source_matches:
                    continue
                if direction == "incoming" and not target_matches:
                    continue

                # Add the related entity
                if source_matches and rel.target_entity_id in self._entities:
                    related.append(self._entities[rel.target_entity_id].to_dict())
                elif target_matches and rel.source_entity_id in self._entities:
                    related.append(self._entities[rel.source_entity_id].to_dict())

            return related

    async def find_entities_async(
        self,
        entity_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities matching criteria (asynchronous).

        Args:
            entity_type: Filter by entity type
            attributes: Filter by attribute key-value pairs

        Returns:
            List of matching entity dictionaries
        """
        async with self._get_async_lock():
            results = []

            for entity in self._entities.values():
                # Filter by type
                if entity_type and entity.entity_type != entity_type:
                    continue

                # Filter by attributes
                if attributes:
                    match = True
                    for key, value in attributes.items():
                        if key not in entity.properties or entity.properties[key] != value:
                            match = False
                            break
                    if not match:
                        continue

                results.append(entity.to_dict())

            return results

    def search_entities(self, query: str, limit: int = 100, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search entities by name or attributes (synchronous).

        Args:
            query: Search query string
            limit: Maximum number of results
            entity_type: Optional filter by entity type

        Returns:
            List of matching entity dictionaries
        """
        with self._lock:
            results = []
            query_lower = query.lower()

            for entity in self._entities.values():
                # Filter by entity type if specified
                if entity_type and entity.entity_type != entity_type:
                    continue

                # Search in name (entity_id is the unique identifier)
                if query_lower in entity.name.lower():
                    results.append(entity.to_dict())
                    continue

                # Search in properties (formerly attributes)
                for key, value in entity.properties.items():
                    if query_lower in str(value).lower():
                        results.append(entity.to_dict())
                        break

                if len(results) >= limit:
                    break

            return results

    async def search_entities_async(self, query: str, limit: int = 100, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search entities by name or attributes (asynchronous).

        Args:
            query: Search query string
            limit: Maximum number of results
            entity_type: Optional filter by entity type

        Returns:
            List of matching entity dictionaries
        """
        async with self._get_async_lock():
            results = []
            query_lower = query.lower()

            for entity in self._entities.values():
                # Filter by entity type if specified
                if entity_type and entity.entity_type != entity_type:
                    continue

                # Search in name
                if query_lower in entity.name.lower():
                    results.append(entity.to_dict())
                    continue

                # Search in properties
                for key, value in entity.properties.items():
                    if query_lower in str(value).lower():
                        results.append(entity.to_dict())
                        break

                if len(results) >= limit:
                    break

            return results

    def get_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity (synchronous).

        Args:
            entity_name: Entity name

        Returns:
            List of relationship dictionaries
        """
        with self._lock:
            results = []

            for rel in self._relationships:
                if rel.source_entity_id == entity_name or rel.target_entity_id == entity_name:
                    results.append(rel.to_dict())

            return results

    async def get_relationships_async(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity (asynchronous).

        Args:
            entity_name: Entity name

        Returns:
            List of relationship dictionaries
        """
        async with self._get_async_lock():
            results = []

            for rel in self._relationships:
                if rel.source_entity_id == entity_name or rel.target_entity_id == entity_name:
                    results.append(rel.to_dict())

            return results

    def to_json(self) -> str:
        """
        Serialize graph to JSON (synchronous).

        Returns:
            JSON string representation
        """
        with self._lock:
            data = {
                "entities": [e.to_dict() for e in self._entities.values()],
                "relationships": [r.to_dict() for r in self._relationships],
                "metadata": {
                    "entity_count": len(self._entities),
                    "relationship_count": len(self._relationships),
                    "correlation_id": self._correlation_id,
                    "exported_at": datetime.now(timezone.utc).isoformat()
                }
            }

            return json.dumps(data, indent=2)

    async def to_json_async(self) -> str:
        """
        Serialize graph to JSON (asynchronous).

        Returns:
            JSON string representation
        """
        async with self._get_async_lock():
            data = {
                "entities": [e.to_dict() for e in self._entities.values()],
                "relationships": [r.to_dict() for r in self._relationships],
                "metadata": {
                    "entity_count": len(self._entities),
                    "relationship_count": len(self._relationships),
                    "correlation_id": self._correlation_id,
                    "exported_at": datetime.now(timezone.utc).isoformat()
                }
            }

            return json.dumps(data, indent=2)

    def from_json(self, json_str: str) -> bool:
        """
        Load graph from JSON (synchronous).

        IDEMPOTENT: Can be called multiple times with same data.

        Args:
            json_str: JSON string representation

        Returns:
            True if successful, False on error
        """
        try:
            data = json.loads(json_str)

            with self._lock:
                # Clear existing data
                self._entities.clear()
                self._relationships.clear()
                self._entity_types.clear()

                # Load entities using unified model
                for entity_data in data.get("entities", []):
                    entity = Entity.from_dict(entity_data)
                    self._entities[entity.entity_id] = entity

                    # Update type index
                    if entity.entity_type not in self._entity_types:
                        self._entity_types[entity.entity_type] = set()
                    self._entity_types[entity.entity_type].add(entity.entity_id)

                # Load relationships using unified model
                for rel_data in data.get("relationships", []):
                    rel = Relationship.from_dict(rel_data)
                    self._relationships.append(rel)

            self._log("info", "Graph loaded from JSON",
                     entity_count=len(self._entities),
                     relationship_count=len(self._relationships))

            return True

        except Exception as e:
            self._log("error", "Failed to load from JSON", error=str(e))
            return False

    async def from_json_async(self, json_str: str) -> bool:
        """
        Load graph from JSON (asynchronous).

        IDEMPOTENT: Can be called multiple times with same data.

        Args:
            json_str: JSON string representation

        Returns:
            True if successful, False on error
        """
        try:
            data = json.loads(json_str)

            async with self._get_async_lock():
                # Clear existing data
                self._entities.clear()
                self._relationships.clear()
                self._entity_types.clear()

                # Load entities using unified model
                for entity_data in data.get("entities", []):
                    entity = Entity.from_dict(entity_data)
                    self._entities[entity.entity_id] = entity

                    # Update type index
                    if entity.entity_type not in self._entity_types:
                        self._entity_types[entity.entity_type] = set()
                    self._entity_types[entity.entity_type].add(entity.entity_id)

                # Load relationships using unified model
                for rel_data in data.get("relationships", []):
                    rel = Relationship.from_dict(rel_data)
                    self._relationships.append(rel)

            self._log("info", "Graph loaded from JSON",
                     entity_count=len(self._entities),
                     relationship_count=len(self._relationships))

            return True

        except Exception as e:
            self._log("error", "Failed to load from JSON", error=str(e))
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics (synchronous).

        Returns:
            Dictionary with graph metrics
        """
        with self._lock:
            return {
                "entity_count": len(self._entities),
                "relationship_count": len(self._relationships),
                "entity_types": {
                    entity_type: len(entities)
                    for entity_type, entities in self._entity_types.items()
                },
                "correlation_id": self._correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_statistics_async(self) -> Dict[str, Any]:
        """
        Get graph statistics (asynchronous).

        Returns:
            Dictionary with graph metrics
        """
        async with self._get_async_lock():
            return {
                "entity_count": len(self._entities),
                "relationship_count": len(self._relationships),
                "entity_types": {
                    entity_type: len(entities)
                    for entity_type, entities in self._entity_types.items()
                },
                "correlation_id": self._correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def clear(self):
        """Clear all entities and relationships (synchronous)."""
        with self._lock:
            self._entities.clear()
            self._relationships.clear()
            self._entity_types.clear()

            self._log("info", "Graph cleared")

    async def clear_async(self):
        """Clear all entities and relationships (asynchronous)."""
        async with self._get_async_lock():
            self._entities.clear()
            self._relationships.clear()
            self._entity_types.clear()

            self._log("info", "Graph cleared")

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Get all relationships in the graph (synchronous).

        Returns:
            List of all relationship dictionaries
        """
        with self._lock:
            return [rel.to_dict() for rel in self._relationships]

    async def get_all_relationships_async(self) -> List[Dict[str, Any]]:
        """
        Get all relationships in the graph (asynchronous).

        Returns:
            List of all relationship dictionaries
        """
        async with self._get_async_lock():
            return [rel.to_dict() for rel in self._relationships]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire graph to a dictionary representation.

        Returns:
            Dictionary with entities and relationships
        """
        with self._lock:
            return {
                "entities": [entity.to_dict() for entity in self._entities.values()],
                "relationships": [rel.to_dict() for rel in self._relationships],
                "statistics": self.get_statistics()
            }

    async def to_dict_async(self) -> Dict[str, Any]:
        """
        Convert the entire graph to a dictionary representation (asynchronous).

        Returns:
            Dictionary with entities and relationships
        """
        async with self._get_async_lock():
            return {
                "entities": [entity.to_dict() for entity in self._entities.values()],
                "relationships": [rel.to_dict() for rel in self._relationships],
                "statistics": await self.get_statistics_async()
            }
