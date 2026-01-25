"""
Functional Dependency Graph (FDG) Implementation

Captures causal structure of problem domains for isomorphism detection.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
import json


class EdgeType(Enum):
    """Types of relationships in FDG"""
    CAUSAL = "causal"           # Direct cause-effect
    CORRELATION = "correlation" # Statistical association
    CONSTRAINT = "constraint"   # Logical constraint
    FEEDBACK = "feedback"       # Bidirectional causal


@dataclass
class Node:
    """Node in Functional Dependency Graph"""
    id: str
    variable: str              # Variable name
    constraint_type: str       # Type of constraint
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'variable': self.variable,
            'constraint_type': self.constraint_type,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """Deserialize from dictionary"""
        return cls(**data)


@dataclass
class Edge:
    """Edge in Functional Dependency Graph"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        """Deserialize from dictionary"""
        data['edge_type'] = EdgeType(data['edge_type'])
        return cls(**data)


@dataclass
class CausalModel:
    """Structural Causal Model"""
    structural_equations: Dict[str, Any]  # X_i = f_i(pa(X_i), U_i)
    exogenous_distribution: Optional[Any] = None
    intervention_data: Optional[Dict] = None


class FunctionalDependencyGraph:
    """
    Functional Dependency Graph representation

    Captures causal structure of problem domain
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.causal_model: Optional[CausalModel] = None
        self.metadata: Dict[str, Any] = {}

    def add_node(self, node: Node) -> None:
        """Add node to FDG"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.metadata)

    def add_edge(self, edge: Edge) -> None:
        """Add edge to FDG"""
        self.edges[(edge.source, edge.target)] = edge
        self.graph.add_edge(
            edge.source,
            edge.target,
            type=edge.edge_type.value,
            weight=edge.weight,
            **edge.metadata
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """Get edge by source and target"""
        return self.edges.get((source, target))

    def get_causal_subgraph(self) -> nx.DiGraph:
        """Extract subgraph containing only causal edges"""
        causal_edges = [
            (s, t) for (s, t), e in self.edges.items()
            if e.edge_type == EdgeType.CAUSAL
        ]
        return self.graph.edge_subgraph(causal_edges)

    def get_feedback_loops(self) -> List[List[str]]:
        """Detect feedback loops in the graph"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except Exception:
            return []

    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendants of a node"""
        try:
            return nx.descendants(self.graph, node_id)
        except nx.NetworkXError:
            return set()

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestors of a node"""
        try:
            return nx.ancestors(self.graph, node_id)
        except nx.NetworkXError:
            return set()

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'causal_model': self.causal_model.structural_equations if self.causal_model else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FunctionalDependencyGraph':
        """Deserialize from dictionary"""
        fdg = cls()
        for node_data in data['nodes']:
            node = Node.from_dict(node_data)
            fdg.add_node(node)
        for edge_data in data['edges']:
            edge = Edge.from_dict(edge_data)
            fdg.add_edge(edge)
        if data.get('causal_model'):
            fdg.causal_model = CausalModel(structural_equations=data['causal_model'])
        fdg.metadata = data.get('metadata', {})
        return fdg

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"FDG(nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __eq__(self, other):
        """Check FDG equality"""
        if not isinstance(other, FunctionalDependencyGraph):
            return False
        return (
            len(self.nodes) == len(other.nodes) and
            len(self.edges) == len(other.edges)
        )
