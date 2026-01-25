"""
Unit tests for Functional Dependency Graph (FDG)

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from phase2.imech.core import (
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType,
    CausalModel
)


class TestNode:
    """Test Node class"""

    def test_node_creation(self):
        """Test creating a node"""
        node = Node(
            id="n1",
            variable="x",
            constraint_type="continuous"
        )
        assert node.id == "n1"
        assert node.variable == "x"
        assert node.constraint_type == "continuous"
        assert node.metadata == {}

    def test_node_with_metadata(self):
        """Test node with metadata"""
        node = Node(
            id="n1",
            variable="x",
            constraint_type="continuous",
            metadata={"range": [0, 10]}
        )
        assert node.metadata == {"range": [0, 10]}

    def test_node_hash(self):
        """Test node hashing"""
        node1 = Node(id="n1", variable="x", constraint_type="continuous")
        node2 = Node(id="n1", variable="y", constraint_type="discrete")
        assert hash(node1) == hash(node2)

    def test_node_serialization(self):
        """Test node serialization"""
        node = Node(id="n1", variable="x", constraint_type="continuous")
        data = node.to_dict()
        assert data['id'] == "n1"
        assert data['variable'] == "x"

        node2 = Node.from_dict(data)
        assert node2.id == node.id
        assert node2.variable == node.variable


class TestEdge:
    """Test Edge class"""

    def test_edge_creation(self):
        """Test creating an edge"""
        edge = Edge(
            source="n1",
            target="n2",
            edge_type=EdgeType.CAUSAL
        )
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.edge_type == EdgeType.CAUSAL
        assert edge.weight == 1.0

    def test_edge_with_weight(self):
        """Test edge with custom weight"""
        edge = Edge(
            source="n1",
            target="n2",
            edge_type=EdgeType.CORRELATION,
            weight=0.5
        )
        assert edge.weight == 0.5

    def test_edge_serialization(self):
        """Test edge serialization"""
        edge = Edge(
            source="n1",
            target="n2",
            edge_type=EdgeType.CAUSAL,
            weight=2.0
        )
        data = edge.to_dict()
        assert data['source'] == "n1"
        assert data['edge_type'] == "causal"
        assert data['weight'] == 2.0

        edge2 = Edge.from_dict(data)
        assert edge2.source == edge.source
        assert edge2.edge_type == edge.edge_type


class TestFunctionalDependencyGraph:
    """Test FDG class"""

    def test_fdg_creation(self):
        """Test creating an FDG"""
        fdg = FunctionalDependencyGraph()
        assert len(fdg) == 0
        assert len(fdg.nodes) == 0
        assert len(fdg.edges) == 0

    def test_add_node(self):
        """Test adding nodes to FDG"""
        fdg = FunctionalDependencyGraph()
        node = Node(id="n1", variable="x", constraint_type="continuous")
        fdg.add_node(node)

        assert len(fdg) == 1
        assert "n1" in fdg.nodes
        assert fdg.get_node("n1") == node

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes"""
        fdg = FunctionalDependencyGraph()
        for i in range(5):
            node = Node(
                id=f"n{i}",
                variable=f"x{i}",
                constraint_type="continuous"
            )
            fdg.add_node(node)

        assert len(fdg) == 5

    def test_add_edge(self):
        """Test adding edges to FDG"""
        fdg = FunctionalDependencyGraph()
        node1 = Node(id="n1", variable="x", constraint_type="continuous")
        node2 = Node(id="n2", variable="y", constraint_type="continuous")
        fdg.add_node(node1)
        fdg.add_node(node2)

        edge = Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL)
        fdg.add_edge(edge)

        assert len(fdg.edges) == 1
        assert fdg.get_edge("n1", "n2") == edge

    def test_causal_subgraph(self):
        """Test extracting causal subgraph"""
        fdg = FunctionalDependencyGraph()

        # Add nodes
        for i in range(3):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        # Add edges (mixed types)
        fdg.add_edge(Edge(source="n0", target="n1", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CORRELATION))

        causal_subgraph = fdg.get_causal_subgraph()
        assert len(list(causal_subgraph.edges())) == 1

    def test_feedback_loops(self):
        """Test feedback loop detection"""
        fdg = FunctionalDependencyGraph()

        # Create cycle
        for i in range(3):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        fdg.add_edge(Edge(source="n0", target="n1", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source="n2", target="n0", edge_type=EdgeType.FEEDBACK))

        loops = fdg.get_feedback_loops()
        assert len(loops) > 0

    def test_descendants(self):
        """Test getting descendants"""
        fdg = FunctionalDependencyGraph()

        # Chain: n0 -> n1 -> n2
        for i in range(3):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        fdg.add_edge(Edge(source="n0", target="n1", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL))

        descendants = fdg.get_descendants("n0")
        assert "n1" in descendants
        assert "n2" in descendants

    def test_ancestors(self):
        """Test getting ancestors"""
        fdg = FunctionalDependencyGraph()

        # Chain: n0 -> n1 -> n2
        for i in range(3):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        fdg.add_edge(Edge(source="n0", target="n1", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL))

        ancestors = fdg.get_ancestors("n2")
        assert "n0" in ancestors
        assert "n1" in ancestors

    def test_serialization(self):
        """Test FDG serialization"""
        fdg = FunctionalDependencyGraph()

        node1 = Node(id="n1", variable="x", constraint_type="continuous")
        node2 = Node(id="n2", variable="y", constraint_type="discrete")
        fdg.add_node(node1)
        fdg.add_node(node2)

        edge = Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL)
        fdg.add_edge(edge)

        data = fdg.to_dict()
        assert 'metadata' in data
        assert data['metadata'] == {}

        fdg2 = FunctionalDependencyGraph.from_dict(data)
        assert len(fdg2) == len(fdg)
        assert len(fdg2.edges) == len(fdg.edges)

    def test_equality(self):
        """Test FDG equality"""
        fdg1 = FunctionalDependencyGraph()
        node1 = Node(id="n1", variable="x", constraint_type="continuous")
        fdg1.add_node(node1)

        fdg2 = FunctionalDependencyGraph()
        node2 = Node(id="n2", variable="y", constraint_type="continuous")
        fdg2.add_node(node2)

        assert fdg1 == fdg2  # Same length

    def test_repr(self):
        """Test FDG string representation"""
        fdg = FunctionalDependencyGraph()
        node = Node(id="n1", variable="x", constraint_type="continuous")
        fdg.add_node(node)

        repr_str = repr(fdg)
        assert "FDG" in repr_str
        assert "nodes=1" in repr_str
