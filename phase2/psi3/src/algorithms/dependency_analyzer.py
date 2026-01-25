"""
Stage 2: Dependency Analysis for Ψ₃

Builds functional dependency graph and detects implications.
"""

from typing import List, Set, Dict, Tuple
import time
import networkx as nx

from ..core.constraint import Constraint
from ..core.expression import Expr
from ..solvers.sat_wrapper import SATInterface, SatResult


class DependencyGraph:
    """
    Dependency graph representing constraint implications.

    Nodes: Constraints
    Edges: Implications (c1 ⊨ c2)
    """

    def __init__(self, constraints: List[Constraint]):
        """Initialize graph from constraints"""
        self.constraints = {c.id: c for c in constraints}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([c.id for c in constraints])

        # Cached computations
        self._transitive_closure: Dict[int, Set[int]] = None
        self._sccs: List[Set[int]] = None

    def add_implication(self, source_id: int, target_id: int):
        """Add implication edge: source ⊨ target"""
        self.graph.add_edge(source_id, target_id)
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidate cached computations"""
        self._transitive_closure = None
        self._sccs = None

    def compute_transitive_closure(self) -> Dict[int, Set[int]]:
        """Compute transitive closure using NetworkX"""
        if self._transitive_closure is not None:
            return self._transitive_closure

        closure = nx.transitive_closure(self.graph)

        self._transitive_closure = {
            node: set(closure.successors(node))
            for node in self.graph.nodes()
        }

        return self._transitive_closure

    def find_strongly_connected_components(self) -> List[Set[int]]:
        """Find SCCs (equivalence classes of mutual implication)"""
        if self._sccs is not None:
            return self._sccs

        self._sccs = [
            set(comp)
            for comp in nx.strongly_connected_components(self.graph)
        ]

        return self._sccs

    def transitive_reduction(self) -> 'DependencyGraph':
        """
        Remove transitive edges.

        If a → b → c exists, remove direct edge a → c
        """
        reduced_graph = nx.transitive_reduction(self.graph)

        result = DependencyGraph(list(self.constraints.values()))
        result.graph = reduced_graph

        return result

    def get_redundant_constraints(self) -> Set[int]:
        """Find constraints implied by other constraints"""
        redundant = set()

        for node in self.graph.nodes():
            predecessors = set(self.graph.predecessors(node))

            # If any predecessor implies this node, it's redundant
            for pred in predecessors:
                if nx.has_path(self.graph, pred, node):
                    redundant.add(node)
                    break

        return redundant


class DependencyAnalysisResult:
    """Result of dependency analysis"""

    def __init__(
        self,
        graph: DependencyGraph,
        implications_found: int,
        sccs_found: int,
        transitive_edges_removed: int,
        runtime_seconds: float
    ):
        self.graph = graph
        self.implications_found = implications_found
        self.sccs_found = sccs_found
        self.transitive_edges_removed = transitive_edges_removed
        self.runtime_seconds = runtime_seconds


def build_dependency_graph(
    constraints: List[Constraint],
    solver: SATInterface,
    verbose: bool = False
) -> DependencyAnalysisResult:
    """
    Stage 2: Build functional dependency graph.

    Detects implications using SAT solver and builds graph structure.

    Complexity: O(k² · SAT(k)) where k = |constraints|

    Args:
        constraints: Constraints to analyze
        solver: SAT solver interface
        verbose: Enable logging

    Returns:
        DependencyAnalysisResult with implication graph
    """
    start_time = time.time()

    if verbose:
        print(f"[Stage 2] Building dependency graph for {len(constraints)} constraints")

    # Initialize graph
    graph = DependencyGraph(constraints)

    # Detect implications between all pairs
    implications_found = 0
    from itertools import combinations

    for i, c1 in enumerate(constraints):
        if verbose and i % 10 == 0:
            print(f"  [Stage 2] Processing constraint {i+1}/{len(constraints)}")

        for c2 in constraints:
            if c1.id == c2.id:
                continue

            # Check if c1 implies c2
            if solver.check_implication(c1.expr, c2.expr):
                graph.add_implication(c1.id, c2.id)
                implications_found += 1

                if verbose:
                    print(f"    [Implication] {c1.id} ⊨ {c2.id}")

    # Compute transitive closure
    if verbose:
        print(f"  [Stage 2] Computing transitive closure...")
    graph.compute_transitive_closure()

    # Find SCCs
    if verbose:
        print(f"  [Stage 2] Finding strongly connected components...")
    sccs = graph.find_strongly_connected_components()
    sccs_found = len(sccs)

    # Transitive reduction
    if verbose:
        print(f"  [Stage 2] Computing transitive reduction...")
    edges_before = graph.graph.number_of_edges()
    graph_reduced = graph.transitive_reduction()
    edges_after = graph_reduced.graph.number_of_edges()
    transitive_removed = edges_before - edges_after

    total_time = time.time() - start_time

    if verbose:
        print(f"[Stage 2] Complete: {implications_found} implications, "
              f"{sccs_found} SCCs, {transitive_removed} transitive edges removed "
              f"({total_time:.3f}s)")

    return DependencyAnalysisResult(
        graph=graph_reduced,
        implications_found=implications_found,
        sccs_found=sccs_found,
        transitive_edges_removed=transitive_removed,
        runtime_seconds=total_time
    )


def find_redundant_constraints(
    constraints: List[Constraint],
    graph: DependencyGraph
) -> Set[int]:
    """
    Identify constraints that are implied by others.

    Args:
        constraints: Constraint list
        graph: Dependency graph

    Returns:
        Set of redundant constraint IDs
    """
    redundant = set()

    for c in constraints:
        # Find all nodes that imply this node
        predecessors = set(graph.graph.predecessors(c.id))

        # If any predecessor implies node, it's redundant
        for pred in predecessors:
            if nx.has_path(graph.graph, pred, c.id):
                redundant.add(c.id)
                break

    return redundant


def find_independent_components(
    graph: DependencyGraph
) -> List[Set[int]]:
    """
    Find independent components (no implications between them).

    Returns:
        List of component node sets
    """
    # Find weakly connected components
    components = [
        set(comp)
        for comp in nx.weakly_connected_components(graph.graph)
    ]

    return components


def compute_closure(
    constraint_id: int,
    graph: DependencyGraph
) -> Set[int]:
    """
    Compute closure of constraint (all constraints it implies).

    Args:
        constraint_id: Starting constraint
        graph: Dependency graph

    Returns:
        Set of reachable constraint IDs
    """
    closure = graph.compute_transitive_closure()
    return closure.get(constraint_id, set())
