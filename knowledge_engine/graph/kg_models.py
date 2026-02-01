"""
Knowledge Graph Models for Integration Hub

Additional data models for the Unified KG Integration Hub,
providing standardized structures for knowledge representation.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import json

# Optional imports from graph module (may not be available)
try:
    from .models import KnowledgeNode, KnowledgeEdge, NodeProperties, EdgeProperties
    _models_available = True
except ImportError:
    _models_available = False
    KnowledgeNode = None
    KnowledgeEdge = None
    NodeProperties = None
    EdgeProperties = None

try:
    from .schema import NodeType, EdgeType
    _schema_available = True
except ImportError:
    _schema_available = False
    NodeType = None
    EdgeType = None

logger = logging.getLogger(__name__)


class KnowledgeSource(Enum):
    """Source of knowledge entries."""
    EXTRACTION = "extraction"
    MANUAL = "manual"
    INFERENCE = "inference"
    IMPORT = "import"
    SYSTEM = "system"


class ConfidenceLevel(Enum):
    """Confidence level classification."""
    HIGH = "high"      # >= 0.9
    MEDIUM = "medium"  # >= 0.7
    LOW = "low"        # >= 0.5
    UNCERTAIN = "uncertain"  # < 0.5
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Get confidence level from score."""
        if score >= 0.9:
            return cls.HIGH
        elif score >= 0.7:
            return cls.MEDIUM
        elif score >= 0.5:
            return cls.LOW
        else:
            return cls.UNCERTAIN


@dataclass
class EntityReference:
    """Reference to an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityReference":
        return cls(**data)


@dataclass
class RelationshipDefinition:
    """Definition of a relationship type."""
    name: str
    description: str
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)
    symmetric: bool = False
    transitive: bool = False
    inverse: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipDefinition":
        return cls(**data)


@dataclass
class KnowledgeStatement:
    """
    A knowledge statement with provenance and confidence.
    
    This is a higher-level construct than a simple triple,
    including metadata about how the knowledge was obtained
    and its reliability.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 1.0
    source: KnowledgeSource = KnowledgeSource.SYSTEM
    source_detail: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source.value,
            "source_detail": self.source_detail,
            "timestamp": self.timestamp.isoformat(),
            "evidence": self.evidence,
            "metadata": self.metadata,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeStatement":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=KnowledgeSource(data.get("source", "system")),
            source_detail=data.get("source_detail"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None
        )
    
    def is_valid_at(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if statement is valid at given time."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level classification."""
        return ConfidenceLevel.from_score(self.confidence)
    
    def to_triple(self) -> tuple:
        """Convert to simple triple tuple."""
        return (self.subject, self.predicate, self.object)


@dataclass
class EntityProfile:
    """
    Comprehensive profile for an entity.
    
    Aggregates all known information about an entity,
    including properties, relationships, and provenance.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    types: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    sources: Set[str] = field(default_factory=set)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "types": self.types,
            "aliases": self.aliases,
            "properties": self.properties,
            "relationships": self.relationships,
            "sources": list(self.sources),
            "confidence_scores": self.confidence_scores,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityProfile":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            types=data.get("types", []),
            aliases=data.get("aliases", []),
            properties=data.get("properties", {}),
            relationships=data.get("relationships", []),
            sources=set(data.get("sources", [])),
            confidence_scores=data.get("confidence_scores", {}),
            first_seen=datetime.fromisoformat(data["first_seen"]) if "first_seen" in data else datetime.now(timezone.utc),
            last_updated=datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else datetime.now(timezone.utc),
            metadata=data.get("metadata", {})
        )
    
    def update_timestamp(self):
        """Update last_updated timestamp."""
        self.last_updated = datetime.now(timezone.utc)
    
    def add_relationship(
        self,
        predicate: str,
        target: str,
        confidence: float = 1.0,
        source: str = "unknown"
    ):
        """Add a relationship to the profile."""
        self.relationships.append({
            "predicate": predicate,
            "target": target,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.sources.add(source)
        self.update_timestamp()
    
    def get_avg_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_scores:
            return 1.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)


@dataclass
class GraphPattern:
    """
    A pattern in the knowledge graph.
    
    Represents a recurring structure or motif in the graph,
    useful for pattern mining and graph analytics.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    pattern_type: str = "subgraph"  # subgraph, path, star, etc.
    nodes: List[str] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    frequency: int = 0
    support: float = 0.0
    confidence: float = 0.0
    instances: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type,
            "nodes": self.nodes,
            "edges": self.edges,
            "frequency": self.frequency,
            "support": self.support,
            "confidence": self.confidence,
            "instances": self.instances,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphPattern":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data["description"],
            pattern_type=data.get("pattern_type", "subgraph"),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            frequency=data.get("frequency", 0),
            support=data.get("support", 0.0),
            confidence=data.get("confidence", 0.0),
            instances=data.get("instances", []),
            metadata=data.get("metadata", {}),
            discovered_at=datetime.fromisoformat(data["discovered_at"]) if "discovered_at" in data else datetime.now(timezone.utc)
        )


class KnowledgeGraphModels:
    """
    Knowledge Graph Models for Integration Hub.
    
    Provides standardized data models and utilities for knowledge
    representation in the Unified KG Integration Hub.
    
    Features:
    - Entity profile management
    - Knowledge statement handling
    - Relationship definitions
    - Graph pattern representation
    - Conversion utilities
    """
    
    def __init__(self):
        """Initialize the knowledge graph models."""
        self._entity_profiles: Dict[str, EntityProfile] = {}
        self._statements: Dict[str, KnowledgeStatement] = {}
        self._patterns: Dict[str, GraphPattern] = {}
        self._relationship_defs: Dict[str, RelationshipDefinition] = {}
        
        # Initialize default relationship definitions
        self._init_default_relationships()
        
        logger.info("KnowledgeGraphModels initialized")
    
    def _init_default_relationships(self):
        """Initialize default relationship definitions."""
        defaults = [
            RelationshipDefinition(
                name="is_a",
                description="Entity is a type of another entity",
                transitive=True
            ),
            RelationshipDefinition(
                name="part_of",
                description="Entity is part of another entity",
                transitive=True
            ),
            RelationshipDefinition(
                name="related_to",
                description="General relationship between entities",
                symmetric=True
            ),
            RelationshipDefinition(
                name="causes",
                description="Entity causes another entity",
                domain=["event", "action", "phenomenon"],
                range=["event", "effect", "outcome"]
            ),
            RelationshipDefinition(
                name="located_in",
                description="Entity is located in another entity",
                transitive=True
            ),
            RelationshipDefinition(
                name="produces",
                description="Entity produces or creates another entity"
            ),
            RelationshipDefinition(
                name="uses",
                description="Entity uses another entity"
            ),
            RelationshipDefinition(
                name="knows",
                description="Entity knows or is aware of another entity",
                symmetric=True
            )
        ]
        
        for rel in defaults:
            self._relationship_defs[rel.name] = rel
    
    # ========================================================================
    # Entity Profile Operations
    # ========================================================================
    
    def create_entity_profile(
        self,
        name: str,
        types: Optional[List[str]] = None,
        **kwargs
    ) -> EntityProfile:
        """
        Create a new entity profile.
        
        Args:
            name: Entity name
            types: Entity types
            **kwargs: Additional profile properties
            
        Returns:
            Created entity profile
        """
        profile = EntityProfile(
            name=name,
            types=types or [],
            **kwargs
        )
        self._entity_profiles[name] = profile
        return profile
    
    def get_entity_profile(self, name: str) -> Optional[EntityProfile]:
        """Get entity profile by name."""
        return self._entity_profiles.get(name)
    
    def update_entity_profile(
        self,
        name: str,
        updates: Dict[str, Any]
    ) -> Optional[EntityProfile]:
        """
        Update an entity profile.
        
        Args:
            name: Entity name
            updates: Dictionary of updates
            
        Returns:
            Updated profile or None if not found
        """
        profile = self._entity_profiles.get(name)
        if not profile:
            return None
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.update_timestamp()
        return profile
    
    def get_all_profiles(self) -> List[EntityProfile]:
        """Get all entity profiles."""
        return list(self._entity_profiles.values())
    
    # ========================================================================
    # Knowledge Statement Operations
    # ========================================================================
    
    def create_statement(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source: KnowledgeSource = KnowledgeSource.SYSTEM,
        **kwargs
    ) -> KnowledgeStatement:
        """
        Create a knowledge statement.
        
        Args:
            subject: Statement subject
            predicate: Statement predicate
            object: Statement object
            confidence: Confidence score
            source: Knowledge source
            **kwargs: Additional properties
            
        Returns:
            Created knowledge statement
        """
        statement = KnowledgeStatement(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            source=source,
            **kwargs
        )
        self._statements[statement.id] = statement
        return statement
    
    def get_statement(self, statement_id: str) -> Optional[KnowledgeStatement]:
        """Get statement by ID."""
        return self._statements.get(statement_id)
    
    def find_statements(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[KnowledgeStatement]:
        """
        Find statements matching criteria.
        
        Args:
            subject: Filter by subject
            predicate: Filter by predicate
            object: Filter by object
            min_confidence: Minimum confidence
            
        Returns:
            List of matching statements
        """
        results = list(self._statements.values())
        
        if subject:
            results = [s for s in results if s.subject == subject]
        if predicate:
            results = [s for s in results if s.predicate == predicate]
        if object:
            results = [s for s in results if s.object == object]
        
        results = [s for s in results if s.confidence >= min_confidence]
        
        return results
    
    def get_all_statements(self) -> List[KnowledgeStatement]:
        """Get all knowledge statements."""
        return list(self._statements.values())
    
    # ========================================================================
    # Pattern Operations
    # ========================================================================
    
    def create_pattern(
        self,
        name: str,
        pattern_type: str,
        nodes: List[str],
        edges: List[Dict[str, Any]],
        **kwargs
    ) -> GraphPattern:
        """
        Create a graph pattern.
        
        Args:
            name: Pattern name
            pattern_type: Type of pattern
            nodes: Pattern nodes
            edges: Pattern edges
            **kwargs: Additional properties
            
        Returns:
            Created graph pattern
        """
        pattern = GraphPattern(
            name=name,
            pattern_type=pattern_type,
            nodes=nodes,
            edges=edges,
            **kwargs
        )
        self._patterns[pattern.id] = pattern
        return pattern
    
    def get_pattern(self, pattern_id: str) -> Optional[GraphPattern]:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def find_patterns(self, pattern_type: Optional[str] = None) -> List[GraphPattern]:
        """Find patterns by type."""
        patterns = list(self._patterns.values())
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        return patterns
    
    # ========================================================================
    # Relationship Definition Operations
    # ========================================================================
    
    def get_relationship_def(self, name: str) -> Optional[RelationshipDefinition]:
        """Get relationship definition by name."""
        return self._relationship_defs.get(name)
    
    def add_relationship_def(self, definition: RelationshipDefinition):
        """Add a relationship definition."""
        self._relationship_defs[definition.name] = definition
    
    def get_all_relationship_defs(self) -> List[RelationshipDefinition]:
        """Get all relationship definitions."""
        return list(self._relationship_defs.values())
    
    # ========================================================================
    # Conversion Utilities
    # ========================================================================
    
    def statement_to_triple(
        self,
        statement: KnowledgeStatement
    ) -> tuple:
        """Convert statement to triple tuple."""
        return statement.to_triple()
    
    def statements_to_triples(
        self,
        statements: List[KnowledgeStatement]
    ) -> List[tuple]:
        """Convert statements to triple tuples."""
        return [s.to_triple() for s in statements]
    
    def profile_to_node(
        self,
        profile: EntityProfile,
        node_type: Any = None
    ) -> Any:
        """
        Convert entity profile to KnowledgeNode.
        
        Args:
            profile: Entity profile
            node_type: Node type for the KnowledgeNode (requires models module)
            
        Returns:
            KnowledgeNode representation or None if models not available
        """
        if not _models_available or not _schema_available:
            logger.warning("Cannot convert profile to node: models/schema not available")
            return None
        
        if node_type is None:
            node_type = NodeType.ENTITY
        
        properties = NodeProperties(
            name=profile.name,
            source=list(profile.sources)[0] if profile.sources else None,
            confidence=profile.get_avg_confidence(),
            metadata=profile.properties
        )
        
        return KnowledgeNode(
            id=profile.id,
            node_type=node_type,
            properties=properties,
            labels=profile.types
        )
    
    def export_all(self) -> Dict[str, Any]:
        """Export all models to dictionary."""
        return {
            "entity_profiles": {
                k: v.to_dict() for k, v in self._entity_profiles.items()
            },
            "statements": {
                k: v.to_dict() for k, v in self._statements.items()
            },
            "patterns": {
                k: v.to_dict() for k, v in self._patterns.items()
            },
            "relationship_definitions": {
                k: v.to_dict() for k, v in self._relationship_defs.items()
            },
            "export_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "profile_count": len(self._entity_profiles),
                "statement_count": len(self._statements),
                "pattern_count": len(self._patterns)
            }
        }
    
    def import_all(self, data: Dict[str, Any]) -> Dict[str, int]:
        """
        Import models from dictionary.
        
        Args:
            data: Dictionary with model data
            
        Returns:
            Counts of imported items
        """
        counts = {
            "profiles": 0,
            "statements": 0,
            "patterns": 0
        }
        
        if "entity_profiles" in data:
            for k, v in data["entity_profiles"].items():
                self._entity_profiles[k] = EntityProfile.from_dict(v)
                counts["profiles"] += 1
        
        if "statements" in data:
            for k, v in data["statements"].items():
                self._statements[k] = KnowledgeStatement.from_dict(v)
                counts["statements"] += 1
        
        if "patterns" in data:
            for k, v in data["patterns"].items():
                self._patterns[k] = GraphPattern.from_dict(v)
                counts["patterns"] += 1
        
        return counts
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "entity_profiles": len(self._entity_profiles),
            "knowledge_statements": len(self._statements),
            "graph_patterns": len(self._patterns),
            "relationship_definitions": len(self._relationship_defs)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check models health."""
        return {
            "status": "healthy",
            "profiles_loaded": len(self._entity_profiles),
            "statements_loaded": len(self._statements),
            "patterns_loaded": len(self._patterns),
            "relationship_defs_loaded": len(self._relationship_defs)
        }
