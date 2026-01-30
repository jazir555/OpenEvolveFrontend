"""
Knowledge Graph Schema Definitions

Defines the schema for nodes, edges, and properties in the knowledge graph.
Uses Pydantic for validation and type safety.

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

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    # Core entities
    CONCEPT = "Concept"
    ENTITY = "Entity"
    EVENT = "Event"
    DOCUMENT = "Document"
    CODE = "Code"
    
    # Project-specific
    PROJECT = "Project"
    TASK = "Task"
    TEAM = "Team"
    WORKFLOW = "Workflow"
    
    # Knowledge-specific
    FACT = "Fact"
    RULE = "Rule"
    PATTERN = "Pattern"
    STRATEGY = "Strategy"
    
    # Agent-specific
    AGENT = "Agent"
    ACTION = "Action"
    DECISION = "Decision"
    
    # Semantic
    TOPIC = "Topic"
    CATEGORY = "Category"
    TAG = "Tag"


class EdgeType(Enum):
    """Types of relationships between nodes"""
    # Hierarchical
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    
    # Causal
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    
    # Semantic
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    
    # Temporal
    FOLLOWS = "FOLLOWS"
    PRECEDES = "PRECEDES"
    
    # Project
    ASSIGNED_TO = "ASSIGNED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    BLOCKS = "BLOCKS"
    
    # Knowledge
    EVIDENCE_FOR = "EVIDENCE_FOR"
    REFUTES = "REFUTES"
    IMPLEMENTS = "IMPLEMENTS"
    
    # Agent
    PERFORMED_BY = "PERFORMED_BY"
    DECIDED_BY = "DECIDED_BY"
    TRIGGERED = "TRIGGERED"


class PropertyType(Enum):
    """Types of properties that can be stored"""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    DATETIME = auto()
    LIST = auto()
    DICT = auto()
    EMBEDDING = auto()  # Vector embedding


@dataclass
class PropertySchema:
    """Schema for a single property"""
    name: str
    type: PropertyType
    required: bool = False
    default: Any = None
    default_factory: Any = None
    indexed: bool = False
    unique: bool = False
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this schema"""
        if value is None:
            return not self.required
        
        type_checks = {
            PropertyType.STRING: lambda x: isinstance(x, str),
            PropertyType.INTEGER: lambda x: isinstance(x, int),
            PropertyType.FLOAT: lambda x: isinstance(x, (int, float)),
            PropertyType.BOOLEAN: lambda x: isinstance(x, bool),
            PropertyType.DATETIME: lambda x: isinstance(x, datetime),
            PropertyType.LIST: lambda x: isinstance(x, list),
            PropertyType.DICT: lambda x: isinstance(x, dict),
            PropertyType.EMBEDDING: lambda x: isinstance(x, (list, tuple)),
        }
        
        check = type_checks.get(self.type)
        if check:
            return check(value)
        return True


@dataclass
class NodeSchema:
    """Schema definition for a node type"""
    node_type: NodeType
    properties: List[PropertySchema] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        """Add default properties common to all nodes"""
        default_props = [
            PropertySchema("id", PropertyType.STRING, required=True, indexed=True, unique=True),
            PropertySchema("name", PropertyType.STRING, required=True, indexed=True),
            PropertySchema("created_at", PropertyType.DATETIME, required=True),
            PropertySchema("updated_at", PropertyType.DATETIME, required=True),
            PropertySchema("source", PropertyType.STRING),
            PropertySchema("confidence", PropertyType.FLOAT, default=1.0),
            PropertySchema("embedding", PropertyType.EMBEDDING),
            PropertySchema("metadata", PropertyType.DICT, default_factory=dict),
        ]
        
        existing_names = {p.name for p in self.properties}
        for prop in default_props:
            if prop.name not in existing_names:
                # Convert dataclass to avoid frozen issues
                new_prop = PropertySchema(
                    name=prop.name,
                    type=prop.type,
                    required=prop.required,
                    default=prop.default,
                    indexed=prop.indexed,
                    unique=prop.unique,
                    description=prop.description
                )
                self.properties.append(new_prop)
    
    def get_property(self, name: str) -> Optional[PropertySchema]:
        """Get property schema by name"""
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None
    
    def validate_properties(self, properties: Dict[str, Any]) -> List[str]:
        """Validate properties against schema, return list of errors"""
        errors = []
        
        # Check required properties
        for prop in self.properties:
            if prop.required and prop.name not in properties:
                errors.append(f"Required property '{prop.name}' is missing")
        
        # Validate provided properties
        for name, value in properties.items():
            prop = self.get_property(name)
            if prop is None:
                errors.append(f"Unknown property '{name}'")
            elif not prop.validate(value):
                errors.append(f"Property '{name}' has invalid type (expected {prop.type.name})")
        
        return errors


@dataclass
class EdgeSchema:
    """Schema definition for an edge type"""
    edge_type: EdgeType
    source_types: List[NodeType] = field(default_factory=list)
    target_types: List[NodeType] = field(default_factory=list)
    properties: List[PropertySchema] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        """Add default properties common to all edges"""
        default_props = [
            PropertySchema("id", PropertyType.STRING, required=True, indexed=True, unique=True),
            PropertySchema("created_at", PropertyType.DATETIME, required=True),
            PropertySchema("weight", PropertyType.FLOAT, default=1.0),
            PropertySchema("confidence", PropertyType.FLOAT, default=1.0),
            PropertySchema("source", PropertyType.STRING),
            PropertySchema("metadata", PropertyType.DICT, default_factory=dict),
        ]
        
        existing_names = {p.name for p in self.properties}
        for prop in default_props:
            if prop.name not in existing_names:
                self.properties.append(prop)


@dataclass
class GraphSchema:
    """Complete schema for the knowledge graph"""
    name: str
    version: str = "1.0"
    node_schemas: Dict[NodeType, NodeSchema] = field(default_factory=dict)
    edge_schemas: Dict[EdgeType, EdgeSchema] = field(default_factory=dict)
    description: str = ""
    
    def __post_init__(self):
        """Initialize default schemas if not provided"""
        if not self.node_schemas:
            self._init_default_node_schemas()
        if not self.edge_schemas:
            self._init_default_edge_schemas()
    
    def _init_default_node_schemas(self):
        """Initialize default node schemas"""
        for node_type in NodeType:
            self.node_schemas[node_type] = NodeSchema(
                node_type=node_type,
                description=f"{node_type.value} node type"
            )
    
    def _init_default_edge_schemas(self):
        """Initialize default edge schemas"""
        for edge_type in EdgeType:
            self.edge_schemas[edge_type] = EdgeSchema(
                edge_type=edge_type,
                description=f"{edge_type.value} relationship"
            )
    
    def get_node_schema(self, node_type: NodeType) -> Optional[NodeSchema]:
        """Get schema for a node type"""
        return self.node_schemas.get(node_type)
    
    def get_edge_schema(self, edge_type: EdgeType) -> Optional[EdgeSchema]:
        """Get schema for an edge type"""
        return self.edge_schemas.get(edge_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "node_types": [nt.value for nt in self.node_schemas.keys()],
            "edge_types": [et.value for et in self.edge_schemas.keys()],
        }
    
    def to_json(self) -> str:
        """Serialize schema to JSON"""
        return json.dumps(self.to_dict(), indent=2)


# Predefined schemas for common use cases
DEFAULT_SCHEMA = GraphSchema(
    name="OpenEvolve Knowledge Graph",
    description="Default schema for OpenEvolve knowledge graph"
)

PROJECT_SCHEMA = GraphSchema(
    name="Project Management Schema",
    description="Schema for project management and task tracking",
    node_schemas={
        NodeType.PROJECT: NodeSchema(
            node_type=NodeType.PROJECT,
            properties=[
                PropertySchema("status", PropertyType.STRING, required=True),
                PropertySchema("priority", PropertyType.STRING),
                PropertySchema("deadline", PropertyType.DATETIME),
                PropertySchema("owner", PropertyType.STRING),
            ],
            description="A project with tasks and teams"
        ),
        NodeType.TASK: NodeSchema(
            node_type=NodeType.TASK,
            properties=[
                PropertySchema("status", PropertyType.STRING, required=True),
                PropertySchema("priority", PropertyType.STRING),
                PropertySchema("assignee", PropertyType.STRING),
                PropertySchema("estimated_hours", PropertyType.INTEGER),
                PropertySchema("actual_hours", PropertyType.INTEGER),
            ],
            description="A task within a project"
        ),
        NodeType.TEAM: NodeSchema(
            node_type=NodeType.TEAM,
            properties=[
                PropertySchema("lead", PropertyType.STRING),
                PropertySchema("members", PropertyType.LIST),
            ],
            description="A team working on projects"
        ),
    },
    edge_schemas={
        EdgeType.CONTAINS: EdgeSchema(
            edge_type=EdgeType.CONTAINS,
            source_types=[NodeType.PROJECT],
            target_types=[NodeType.TASK],
            description="Project contains tasks"
        ),
        EdgeType.ASSIGNED_TO: EdgeSchema(
            edge_type=EdgeType.ASSIGNED_TO,
            source_types=[NodeType.TASK],
            target_types=[NodeType.TEAM, NodeType.AGENT],
            description="Task assigned to team or agent"
        ),
        EdgeType.DEPENDS_ON: EdgeSchema(
            edge_type=EdgeType.DEPENDS_ON,
            source_types=[NodeType.TASK],
            target_types=[NodeType.TASK],
            description="Task depends on another task"
        ),
    }
)
