"""
Knowledge Graph Pydantic Models

Data models for nodes, edges, and graphs with validation.

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

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
import json

from .schema import NodeType, EdgeType, PropertyType


class NodeProperties(BaseModel):
    """Properties common to all nodes"""
    model_config = {"extra": "allow"}
    
    name: str = Field(..., description="Human-readable name")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = Field(None, description="Source of this knowledge")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()


class EdgeProperties(BaseModel):
    """Properties common to all edges"""
    model_config = {"extra": "allow"}
    
    weight: float = Field(1.0, ge=0.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Optional[str] = Field(None, description="Source of this relationship")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeNode(BaseModel):
    """A node in the knowledge graph"""
    model_config = {"extra": "allow"}
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    properties: NodeProperties
    labels: List[str] = Field(default_factory=list)
    
    @validator('labels', pre=True, always=True)
    def ensure_type_label(cls, v, values):
        """Ensure node type is in labels"""
        if 'node_type' in values:
            type_label = values['node_type'].value
            if type_label not in v:
                v = [type_label] + v
        return v
    
    def get_label_string(self) -> str:
        """Get Cypher label string (e.g., :Concept:Entity)"""
        return ":" + ":".join(self.labels)
    
    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher format"""
        props = self.properties.model_dump()
        props['id'] = self.id
        
        # Convert datetime to timestamp
        for key in ['created_at', 'updated_at']:
            if key in props and isinstance(props[key], datetime):
                props[key] = props[key].isoformat()
        
        # Convert embedding to JSON string if present
        if 'embedding' in props and props['embedding'] is not None:
            props['embedding'] = json.dumps(props['embedding'])
        
        # Convert metadata to JSON string
        if 'metadata' in props:
            props['metadata'] = json.dumps(props['metadata'])
        
        return "{" + ", ".join(f"{k}: {repr(v)}" for k, v in props.items() if v is not None) + "}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "labels": self.labels,
            "properties": self.properties.model_dump()
        }


class KnowledgeEdge(BaseModel):
    """An edge/relationship in the knowledge graph"""
    model_config = {"extra": "allow"}
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    edge_type: EdgeType
    source_id: str = Field(..., description="ID of source node")
    target_id: str = Field(..., description="ID of target node")
    properties: EdgeProperties = Field(default_factory=EdgeProperties)
    
    def to_cypher_pattern(self, source_var: str = "a", target_var: str = "b") -> str:
        """Convert to Cypher relationship pattern"""
        props = self.properties.model_dump()
        props['id'] = self.id
        
        # Convert metadata to JSON string
        if 'metadata' in props:
            props['metadata'] = json.dumps(props['metadata'])
        
        props_str = "{" + ", ".join(f"{k}: {repr(v)}" for k, v in props.items() if v is not None) + "}"
        
        return f"({source_var})-[:{self.edge_type.value} {props_str}]->({target_var})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "edge_type": self.edge_type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": self.properties.model_dump()
        }


class KnowledgeGraph(BaseModel):
    """A collection of nodes and edges"""
    model_config = {"extra": "allow"}
    
    name: str = Field(..., description="Graph name")
    nodes: Dict[str, KnowledgeNode] = Field(default_factory=dict)
    edges: Dict[str, KnowledgeEdge] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.updated_at = datetime.utcnow()
        return node.id
    
    def add_edge(self, edge: KnowledgeEdge) -> str:
        """Add an edge to the graph"""
        # Validate nodes exist
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} does not exist")
        
        self.edges[edge.id] = edge
        self.updated_at = datetime.utcnow()
        return edge.id
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """Get an edge by ID"""
        return self.edges.get(edge_id)
    
    def get_neighbors(self, node_id: str) -> List[KnowledgeNode]:
        """Get all neighbor nodes"""
        neighbor_ids = set()
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbor_ids.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbor_ids.add(edge.source_id)
        
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]
    
    def get_outgoing_edges(self, node_id: str) -> List[KnowledgeEdge]:
        """Get all outgoing edges from a node"""
        return [e for e in self.edges.values() if e.source_id == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[KnowledgeEdge]:
        """Get all incoming edges to a node"""
        return [e for e in self.edges.values() if e.target_id == node_id]
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges"""
        if node_id not in self.nodes:
            return False
        
        # Remove connected edges
        edges_to_remove = [
            eid for eid, e in self.edges.items()
            if e.source_id == node_id or e.target_id == node_id
        ]
        for eid in edges_to_remove:
            del self.edges[eid]
        
        del self.nodes[node_id]
        self.updated_at = datetime.utcnow()
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge"""
        if edge_id not in self.edges:
            return False
        
        del self.edges[edge_id]
        self.updated_at = datetime.utcnow()
        return True
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[KnowledgeEdge]:
        """Get all edges of a specific type"""
        return [e for e in self.edges.values() if e.edge_type == edge_type]
    
    def search_nodes(self, query: str) -> List[KnowledgeNode]:
        """Simple text search in node names and properties"""
        query_lower = query.lower()
        results = []
        
        for node in self.nodes.values():
            # Search in name
            if query_lower in node.properties.name.lower():
                results.append(node)
                continue
            
            # Search in metadata
            for key, value in node.properties.metadata.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(node)
                    break
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to dictionary"""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        node_type_counts = {}
        for node in self.nodes.values():
            nt = node.node_type.value
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
        
        edge_type_counts = {}
        for edge in self.edges.values():
            et = edge.edge_type.value
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_type_counts,
            "edge_types": edge_type_counts,
            "average_degree": len(self.edges) * 2 / len(self.nodes) if self.nodes else 0,
        }
