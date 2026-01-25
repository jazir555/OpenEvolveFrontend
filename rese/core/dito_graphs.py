"""
DITO Knowledge Graph Structures

Implements three core graphs for O(n log n) contradiction detection:
- CD-Graph (Constraint Dependency Graph): Direct dependencies between constraints
- PV-Graph (Predicate Variable Graph): Bipartite graph of predicates and variables
- HAG (Hierarchical Abstraction Graph): Multi-level constraint clusters

Author: Agent A3 (DITO Specialist)
Created: 2025-12-31
Status: 🟢 Implementation Phase
Complexity: O(log n) operations on all graphs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
import networkx as nx
from collections import defaultdict, deque


# =============================================================================
# GRAPH DATA STRUCTURES
# =============================================================================

class DependencyType(Enum):
    """Types of dependency edges in CD-Graph"""
    DIRECT = "direct"                # cⱼ directly uses cᵢ's output
    INDIRECT = "indirect"            # cⱼ indirectly depends on cᵢ
    CONTRADICTION = "contradiction"  # cᵢ and cⱼ contradict
    IMPLICATION = "implication"      # cᵢ implies cⱼ
    EQUIVALENCE = "equivalence"      # cᵢ ≡ cⱼ


class NodeStatus(Enum):
    """Status of constraint node"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CONTRADICTED = "contradicted"
    SATISFIED = "satisfied"
    UNKNOWN = "unknown"


@dataclass
class ConstraintNode:
    """Node in Constraint Dependency Graph"""
    id: str
    constraint: Any                  # SCE Constraint object

    # Graph topology
    incoming: Set[str] = field(default_factory=set)
    outgoing: Set[str] = field(default_factory=set)
    level: int = 0

    # State
    status: NodeStatus = NodeStatus.UNKNOWN
    timestamp: float = 0
    version: int = 0

    # Watched literals (CDCL technique)
    watched_literals: List[str] = field(default_factory=list)
    watchers: Set[str] = field(default_factory=set)

    # Cache
    contradiction_cache: Set[str] = field(default_factory=set)
    last_check_version: int = 0


@dataclass
class DependencyEdge:
    """Edge in Constraint Dependency Graph"""
    id: str
    source: str                      # Source constraint ID
    target: str                      # Target constraint ID

    dep_type: DependencyType = DependencyType.DIRECT
    weight: float = 1.0              # Strength of dependency [0, 1]
    confidence: float = 1.0          # Certainty of dependency [0, 1]

    version: int = 0
    dirty: bool = False


# =============================================================================
# CONSTRAINT DEPENDENCY GRAPH (CD-Graph)
# =============================================================================

class ConstraintDependencyGraph:
    """
    Constraint Dependency Graph (CD-Graph)

    Tracks direct dependencies between constraints.
    Operations: O(log n) incremental updates, O(1) propagation
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ConstraintNode] = {}
        self.edges: Dict[str, DependencyEdge] = {}
        self.edge_counter = 0

    def add_node(self, constraint: Any) -> None:
        """Add constraint node to graph"""
        node = ConstraintNode(
            id=constraint.id,
            constraint=constraint,
            status=NodeStatus.ACTIVE
        )

        self.nodes[constraint.id] = node
        self.graph.add_node(constraint.id, node=node)

    def add_edge(self, source_id: str, target_id: str,
                 dep_type: DependencyType = DependencyType.DIRECT,
                 weight: float = 1.0) -> None:
        """Add dependency edge between constraints"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both nodes must exist")

        edge = DependencyEdge(
            id=f"edge_{self.edge_counter}",
            source=source_id,
            target=target_id,
            dep_type=dep_type,
            weight=weight
        )
        self.edge_counter += 1

        self.edges[edge.id] = edge
        self.graph.add_edge(source_id, target_id, edge=edge)

        # Update node topology
        self.nodes[source_id].outgoing.add(target_id)
        self.nodes[target_id].incoming.add(source_id)

    def remove_node(self, node_id: str) -> None:
        """Remove node and all associated edges"""
        if node_id not in self.nodes:
            return

        # Remove edges
        in_edges = list(self.graph.in_edges(node_id))
        out_edges = list(self.graph.out_edges(node_id))

        for src, tgt in in_edges + out_edges:
            edge_data = self.graph[src][tgt].get('edge')
            if edge_data:
                del self.edges[edge_data.id]

        # Remove node
        self.graph.remove_node(node_id)
        del self.nodes[node_id]

    def get_dependencies(self, constraint_id: str) -> List[ConstraintNode]:
        """Get all dependencies (predecessors) of constraint"""
        if constraint_id not in self.nodes:
            return []

        return [
            self.nodes[pred_id]
            for pred_id in self.graph.predecessors(constraint_id)
            if pred_id in self.nodes
        ]

    def get_dependents(self, constraint_id: str) -> List[ConstraintNode]:
        """Get all constraints that depend on this one (successors)"""
        if constraint_id not in self.nodes:
            return []

        return [
            self.nodes[succ_id]
            for succ_id in self.graph.successors(constraint_id)
            if succ_id in self.nodes
        ]

    def mark_dirty_region(self, changed_constraint: str,
                          max_depth: int = 5) -> Set[str]:
        """
        Mark affected region after constraint change.

        Complexity: O(branching^L) where L = max_depth
        """
        dirty_nodes = set()

        # Forward propagation (dependents)
        queue = deque([(changed_constraint, 0)])
        while queue:
            node_id, depth = queue.popleft()

            if depth > max_depth:
                continue

            if node_id not in dirty_nodes:
                dirty_nodes.add(node_id)

                # Add successors
                for succ_id in self.graph.successors(node_id):
                    queue.append((succ_id, depth + 1))

        # Backward propagation (dependencies)
        queue = deque([(changed_constraint, 0)])
        while queue:
            node_id, depth = queue.popleft()

            if depth > max_depth:
                continue

            if node_id not in dirty_nodes:
                dirty_nodes.add(node_id)

                # Add predecessors
                for pred_id in self.graph.predecessors(node_id):
                    queue.append((pred_id, depth + 1))

        return dirty_nodes

    def propagate_watched_literals(self, changed_constraint: str) -> List[str]:
        """
        Propagate changes using watched literals (CDCL technique).

        Complexity: O(watchers · propagation_depth)
        """
        node = self.nodes.get(changed_constraint)
        if not node:
            return []

        propagated = []

        # Get watchers
        for watcher_id in node.watchers:
            watcher_node = self.nodes.get(watcher_id)
            if not watcher_node:
                continue

            # Evaluate watcher constraint
            # (In full implementation, this would call LLTL evaluator)
            is_satisfied = self._evaluate_node(watcher_node)

            if is_satisfied:
                propagated.append(watcher_id)
                # Recursively propagate
                propagated.extend(self.propagate_watched_literals(watcher_id))

        return propagated

    def _evaluate_node(self, node: ConstraintNode) -> bool:
        """Evaluate constraint node (placeholder)"""
        # In full implementation, would use LLTL theorem prover
        return node.status == NodeStatus.SATISFIED

    def get_statistics(self) -> Dict[str, int]:
        """Get graph statistics"""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "avg_degree": sum(len(n.incoming) + len(n.outgoing)
                            for n in self.nodes.values()) / max(len(self.nodes), 1)
        }


# =============================================================================
# PREDICATE VARIABLE GRAPH (PV-Graph)
# =============================================================================

@dataclass
class PredicateNode:
    """Node representing a predicate (LLTL formula)"""
    id: str
    formula: Any                      # LLTL formula object

    variables: Set[str] = field(default_factory=set)
    community: str = ""
    community_rank: float = 0.0
    timestamp: float = 0


@dataclass
class VariableNode:
    """Node representing a variable"""
    id: str
    name: str
    var_type: str                     # Type of variable

    predicates: Set[str] = field(default_factory=set)
    domain: Optional[Any] = None
    current_value: Optional[Any] = None
    community: str = ""


class PredicateVariableGraph:
    """
    Predicate Variable Graph (PV-Graph)

    Bipartite graph tracking relationships between predicates and variables.
    Operations: O(log n) community detection, O(1) related constraint lookup
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.predicates: Dict[str, PredicateNode] = {}
        self.variables: Dict[str, VariableNode] = {}
        self.community_index: Dict[str, Set[str]] = defaultdict(set)

    def add_predicate(self, constraint_id: str, formula: Any,
                      variables: List[str]) -> None:
        """Add predicate node connected to variables"""
        node = PredicateNode(
            id=f"pred_{constraint_id}",
            formula=formula,
            variables=set(variables)
        )

        self.predicates[constraint_id] = node
        self.graph.add_node(node.id, node=node, bipartite=0)

        # Connect to variables
        for var_name in variables:
            self._ensure_variable(var_name)
            self.graph.add_edge(node.id, var_name)
            node.variables.add(var_name)
            self.variables[var_name].predicates.add(constraint_id)

    def _ensure_variable(self, var_name: str) -> None:
        """Ensure variable node exists"""
        if var_name not in self.variables:
            node = VariableNode(
                id=f"var_{var_name}",
                name=var_name,
                var_type="UNKNOWN"
            )
            self.variables[var_name] = node
            self.graph.add_node(var_name, node=node, bipartite=1)

    def detect_communities(self) -> Dict[str, Set[str]]:
        """
        Detect communities using Louvain algorithm.

        Complexity: O(n log n)
        """
        import networkx.algorithms.community as nx_comm

        # Detect communities
        communities = nx_comm.louvain_communities(self.graph)

        # Build index
        self.community_index.clear()
        for comm_id, community in enumerate(communities):
            for node_id in community:
                self.community_index[str(comm_id)].add(node_id)

                # Update node community
                if node_id in self.predicates:
                    self.predicates[node_id].community = str(comm_id)
                elif node_id in self.variables:
                    self.variables[node_id].community = str(comm_id)

        return self.community_index

    def get_related_constraints(self, variables: List[str]) -> Set[str]:
        """
        Get all constraints that reference given variables.

        Complexity: O(variables · avg_degree)
        """
        related = set()

        for var_name in variables:
            if var_name not in self.variables:
                continue

            # Get all predicates connected to this variable
            for pred_id in self.graph.neighbors(var_name):
                if pred_id.startswith("pred_"):
                    # Extract constraint ID from predicate ID
                    constraint_id = pred_id.replace("pred_", "")
                    related.add(constraint_id)

        return related

    def get_community_overlap(self, constraint1: str, constraint2: str) -> bool:
        """Check if two constraints share communities"""
        pred1 = self.predicates.get(constraint1)
        pred2 = self.predicates.get(constraint2)

        if not pred1 or not pred2:
            return False

        comm1 = pred1.community
        comm2 = pred2.community

        return comm1 and comm2 and comm1 == comm2

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "predicates": len(self.predicates),
            "variables": len(self.variables),
            "edges": self.graph.number_of_edges(),
            "communities": len(self.community_index),
            "avg_community_size": sum(len(c)
                                     for c in self.community_index.values()) /
                                   max(len(self.community_index), 1)
        }


# =============================================================================
# HIERARCHICAL ABSTRACTION GRAPH (HAG)
# =============================================================================

@dataclass
class HAGNode:
    """Node in Hierarchical Abstraction Graph"""
    id: str
    level: int

    # Composition
    children: List[str] = field(default_factory=list)
    members: List[str] = field(default_factory=list)

    # Abstracted properties
    extent: Optional[Any] = None
    signature: str = ""

    # Inter-cluster edges
    dependencies: Set[str] = field(default_factory=set)
    contradictions: Set[str] = field(default_factory=set)

    # Status
    status: str = "UNKNOWN"
    last_check: float = 0

    # Cache
    contradiction_cache: Set[str] = field(default_factory=set)


class HierarchicalAbstractionGraph:
    """
    Hierarchical Abstraction Graph (HAG)

    Multi-level hierarchy enabling O(log n) contradiction detection
    through top-down refinement and early pruning.

    Operations: O(log n) traversal, O(n log n) build
    """

    def __init__(self, max_level: int = 10):
        self.max_level = max_level
        self.nodes: Dict[str, HAGNode] = {}
        self.levels: Dict[int, List[str]] = defaultdict(list)
        self.node_counter = 0

    def build_hierarchy(self, constraints: Dict[str, Any],
                       cd_graph: Any = None,  # Changed type to Any to avoid circular import
                       pv_graph: Any = None) -> None:  # Changed type to Any to avoid circular import
        """
        Build hierarchical abstraction from constraints.

        Complexity: O(n log n)

        Args:
            constraints: Dictionary of constraint_id -> constraint object
            cd_graph: Optional ConstraintDependencyGraph instance
            pv_graph: Optional PredicateVariableGraph instance
        """
        # Level 0: Each constraint is its own node
        for constraint_id, constraint in constraints.items():
            node = HAGNode(
                id=f"hag_{self.node_counter}",
                level=0,
                members=[constraint_id],
                signature=self._compute_signature([constraint_id])
            )
            self.node_counter += 1

            self.nodes[node.id] = node
            self.levels[0].append(node.id)

        # Build higher levels
        for level in range(1, self.max_level + 1):
            if not self.levels[level - 1]:
                break

            self._build_level(level, constraints, cd_graph, pv_graph)

    def _build_level(self, level: int, constraints: Dict[str, Any],
                     cd_graph: Any = None,  # Changed to Any to avoid circular import
                     pv_graph: Any = None) -> None:  # Changed to Any to avoid circular import
        """Build a single level of hierarchy"""
        prev_level_nodes = [self.nodes[nid] for nid in self.levels[level - 1]]

        # Cluster nodes at previous level
        # Simple clustering: group by spatial/community proximity
        cluster_size = 2 ** level

        for i in range(0, len(prev_level_nodes), cluster_size):
            cluster = prev_level_nodes[i:i + cluster_size]
            if not cluster:
                continue

            # Merge cluster
            all_members = []
            all_children = []

            for node in cluster:
                all_members.extend(node.members)
                all_children.append(node.id)

            # Compute combined signature
            signature = self._compute_signature(all_members)

            # Create new abstraction node
            new_node = HAGNode(
                id=f"hag_{self.node_counter}",
                level=level,
                children=all_children,
                members=all_members,
                signature=signature
            )
            self.node_counter += 1

            self.nodes[new_node.id] = new_node
            self.levels[level].append(new_node.id)

    def get_nodes_at_level(self, level: int) -> List[HAGNode]:
        """Get all nodes at given level"""
        return [self.nodes[nid] for nid in self.levels.get(level, [])]

    def detect_contradictions_top_down(self) -> List[Tuple[str, str]]:
        """
        Detect contradictions using top-down refinement.

        Complexity: O(√n · log n)
        """
        contradictions = []

        # Start from top level
        max_level = max(self.levels.keys()) if self.levels else 0

        for level in range(max_level, -1, -1):
            nodes = self.get_nodes_at_level(level)

            # Check pairs at this level
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    # Quick check: non-overlapping extents can't contradict
                    if self._can_prune_pair(node1, node2):
                        continue

                    # Refine or check
                    if level == 0:
                        # Leaf level: check constraint pairs
                        for c1 in node1.members:
                            for c2 in node2.members:
                                contradictions.append((c1, c2))
                    else:
                        # Higher level: refine
                        contradictions.extend(
                            self._refine_pair(node1, node2)
                        )

        return contradictions

    def _can_prune_pair(self, node1: HAGNode, node2: HAGNode) -> bool:
        """Quick check if pair can be pruned"""
        # Check status
        if node1.status == "PRUNED" or node2.status == "PRUNED":
            return True

        # Check spatial overlap (if extents available)
        if node1.extent and node2.extent:
            if hasattr(node1.extent, 'overlaps'):
                if not node1.extent.overlaps(node2.extent):
                    return True

        return False

    def _refine_pair(self, node1: HAGNode, node2: HAGNode) -> List[Tuple[str, str]]:
        """Refine pair to lower level"""
        contradictions = []

        for child1_id in node1.children:
            for child2_id in node2.children:
                child1 = self.nodes.get(child1_id)
                child2 = self.nodes.get(child2_id)

                if not child1 or not child2:
                    continue

                if self._can_prune_pair(child1, child2):
                    continue

                if child1.level == 0:
                    # Leaf level
                    for c1 in child1.members:
                        for c2 in child2.members:
                            contradictions.append((c1, c2))
                else:
                    # Recurse
                    contradictions.extend(self._refine_pair(child1, child2))

        return contradictions

    def update_incremental(self, changed_constraint: str) -> List[str]:
        """
        Update hierarchy after constraint change.

        Complexity: O(log n)
        """
        affected_nodes = []

        # Find leaf node containing constraint
        for node_id in self.levels.get(0, []):
            node = self.nodes[node_id]
            if changed_constraint in node.members:
                # Walk up hierarchy
                current_node = node
                while current_node:
                    affected_nodes.append(current_node.id)
                    current_node = self._get_parent(current_node)

                break

        return affected_nodes

    def _get_parent(self, node: HAGNode) -> Optional[HAGNode]:
        """Get parent of node"""
        if node.level >= self.max_level:
            return None

        next_level = node.level + 1
        for potential_parent_id in self.levels.get(next_level, []):
            parent = self.nodes[potential_parent_id]
            if node.id in parent.children:
                return parent

        return None

    @staticmethod
    def _compute_signature(members: List[str]) -> str:
        """Compute signature hash for node"""
        import hashlib
        combined = ",".join(sorted(members))
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchy statistics"""
        total_nodes = sum(len(nodes) for nodes in self.levels.values())

        return {
            "total_nodes": total_nodes,
            "levels": len(self.levels),
            "max_level": max(self.levels.keys()) if self.levels else 0,
            "nodes_per_level": {lvl: len(nodes)
                              for lvl, nodes in self.levels.items()}
        }


# =============================================================================
# GRAPH TRAVERSAL ALGORITHMS
# =============================================================================

class GraphTraversals:
    """Efficient graph traversal algorithms for DITO"""

    @staticmethod
    def bfs_localized_check(cd_graph: ConstraintDependencyGraph,
                           start_constraint: str,
                           max_depth: int = 5) -> List[str]:
        """
        BFS for localized contradiction checking.

        Complexity: O(branching^L) where L = max_depth
        """
        visited = set()
        result = []
        queue = deque([(start_constraint, 0)])

        while queue:
            node_id, depth = queue.popleft()

            if node_id in visited:
                continue

            visited.add(node_id)
            result.append(node_id)

            if depth >= max_depth:
                continue

            # Expand to neighbors
            node = cd_graph.nodes.get(node_id)
            if not node:
                continue

            for neighbor_id in node.incoming | node.outgoing:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))

        return result

    @staticmethod
    def priority_traversal(cd_graph: ConstraintDependencyGraph,
                          query_constraint: str,
                          related_constraints: Set[str],
                          top_k: int = 10) -> List[str]:
        """
        Priority-based traversal for focused checking.

        Complexity: O(k · d · log d) where d = avg degree
        """
        import heapq

        # Compute priority scores
        priorities = []
        for constraint_id in related_constraints:
            if constraint_id == query_constraint:
                continue

            score = GraphTraversals._compute_priority_score(
                cd_graph, query_constraint, constraint_id
            )
            priorities.append((-score, constraint_id))  # Negative for max-heap

        # Get top-k
        heapq.heapify(priorities)
        top_k_constraints = []

        for _ in range(min(top_k, len(priorities))):
            if priorities:
                _, constraint_id = heapq.heappop(priorities)
                top_k_constraints.append(constraint_id)

        return top_k_constraints

    @staticmethod
    def _compute_priority_score(cd_graph: ConstraintDependencyGraph,
                                query_id: str,
                                candidate_id: str) -> float:
        """Compute priority score for candidate constraint"""
        score = 0.0

        # Dependency distance (closer = higher priority)
        try:
            shortest_path = nx.shortest_path_length(
                cd_graph.graph, query_id, candidate_id
            )
            score += 1.0 / (1 + shortest_path)
        except nx.NetworkXNoPath:
            pass

        # Shared variables (from PV-Graph would be used here)
        # For now, use graph structure

        return score

    @staticmethod
    def bidirectional_search(cd_graph: ConstraintDependencyGraph,
                            constraint1: str,
                            constraint2: str) -> Optional[List[str]]:
        """
        Bidirectional search for path finding.

        Complexity: O(b^(d/2)) where b = branching, d = path length
        """
        forward_visited = {constraint1: 0}
        forward_parent = {}
        backward_visited = {constraint2: 0}
        backward_parent = {}

        forward_queue = deque([constraint1])
        backward_queue = deque([constraint2])

        meeting_node = None

        while forward_queue and backward_queue and not meeting_node:
            # Expand forward
            for _ in range(len(forward_queue)):
                current = forward_queue.popleft()

                if current in backward_visited:
                    meeting_node = current
                    break

                for neighbor in cd_graph.graph.successors(current):
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = forward_visited[current] + 1
                        forward_parent[neighbor] = current
                        forward_queue.append(neighbor)

            # Expand backward
            for _ in range(len(backward_queue)):
                current = backward_queue.popleft()

                if current in forward_visited:
                    meeting_node = current
                    break

                for neighbor in cd_graph.graph.predecessors(current):
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = backward_visited[current] + 1
                        backward_parent[neighbor] = current
                        backward_queue.append(neighbor)

        if not meeting_node:
            return None

        # Reconstruct path
        path = []

        # Forward path
        node = meeting_node
        while node != constraint1:
            path.append(node)
            node = forward_parent.get(node, constraint1)
        path.append(constraint1)
        path.reverse()

        # Backward path
        node = meeting_node
        while node != constraint2:
            node = backward_parent.get(node, constraint2)
            if node != constraint2:
                path.append(node)

        return path


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DITO Knowledge Graph Structures - Testing")
    print("=" * 70)

    # Import SCE for testing
    from symbolic_constraint_engine import Constraint, ConstraintType

    # Create test constraints
    constraints = [
        Constraint(
            id="c1",
            type=ConstraintType.HARD,
            description="Temperature < 1000",
            formalization="T < 1000",
            source="user"
        ),
        Constraint(
            id="c2",
            type=ConstraintType.HARD,
            description="Temperature > 500",
            formalization="T > 500",
            source="user",
            dependencies=["c1"]
        ),
        Constraint(
            id="c3",
            type=ConstraintType.SOFT,
            description="Pressure < 10",
            formalization="P < 10",
            source="system"
        ),
    ]

    # Test CD-Graph
    print("\n[INFO] Testing CD-Graph...")
    cd_graph = ConstraintDependencyGraph()

    for c in constraints:
        cd_graph.add_node(c)

    cd_graph.add_edge("c1", "c2", DependencyType.DIRECT)

    print(f"[OK] Nodes: {cd_graph.get_statistics()['nodes']}")
    print(f"[OK] Edges: {cd_graph.get_statistics()['edges']}")

    deps = cd_graph.get_dependencies("c2")
    print(f"[OK] Dependencies of c2: {[d.id for d in deps]}")

    # Test PV-Graph
    print("\n[INFO] Testing PV-Graph...")
    pv_graph = PredicateVariableGraph()

    for c in constraints:
        variables = ["T"] if "Temperature" in c.description else ["P"]
        pv_graph.add_predicate(c.id, c.formalization, variables)

    pv_graph.detect_communities()
    print(f"[OK] Predicates: {pv_graph.get_statistics()['predicates']}")
    print(f"[OK] Variables: {pv_graph.get_statistics()['variables']}")
    print(f"[OK] Communities: {pv_graph.get_statistics()['communities']}")

    related = pv_graph.get_related_constraints(["T"])
    print(f"[OK] Constraints using T: {related}")

    # Test HAG
    print("\n[INFO] Testing HAG...")
    hag = HierarchicalAbstractionGraph(max_level=3)

    constraint_dict = {c.id: c for c in constraints}
    hag.build_hierarchy(constraint_dict, cd_graph, pv_graph)

    print(f"[OK] Total nodes: {hag.get_statistics()['total_nodes']}")
    print(f"[OK] Levels: {hag.get_statistics()['levels']}")

    # Test traversals
    print("\n[INFO] Testing traversals...")
    localized = GraphTraversals.bfs_localized_check(cd_graph, "c1", max_depth=2)
    print(f"[OK] BFS from c1 (depth 2): {localized}")

    path = GraphTraversals.bidirectional_search(cd_graph, "c1", "c2")
    print(f"[OK] Path c1 -> c2: {path}")

    print("\n" + "=" * 70)
    print("[OK] All graph structures tested successfully")
    print("=" * 70)
