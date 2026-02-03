"""
Dependency Builder for Sovereign Decomposition System

This module provides production-ready dependency graph construction and analysis
for sub-problem decomposition. It implements graph algorithms from scratch without
external dependencies like networkx.

Core capabilities:
- Build DAG from sub-problem dependencies
- Detect circular dependencies
- Calculate optimal execution order (topological sort)
- Identify critical path
- Find parallelizable tasks

Author: OpenEvolve Frontend Team
Date: 2026-01-21
License: MIT
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from enum import Enum

# **ACTUAL INTEGRATION**: Alerting and knowledge for Dependency Builder
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# Import from sovereign_data_models
try:
    from sovereign_data_models import SubProblem, ProblemStatus
except ImportError:
    # Fallback definitions
    from dataclasses import dataclass
    from typing import Optional, Any
    from datetime import datetime
    from enum import Enum

    class ProblemStatus(Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        BLOCKED = "blocked"
        FAILED = "failed"

    @dataclass
    class SubProblem:
        sub_problem_id: str
        parent_id: Optional[str]
        title: str
        description: str
        status: ProblemStatus
        confidence: float
        assigned_agent: Optional[str]
        created_at: datetime
        completed_at: Optional[datetime]
        result: Optional[Any] = None


logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Base exception for dependency-related errors."""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependencies are detected."""
    pass


class InvalidGraphError(DependencyError):
    """Raised when the graph structure is invalid."""
    pass


@dataclass
class DependencyNode:
    """
    Represents a node in the dependency graph.

    Attributes:
        node_id: Unique identifier for the node
        sub_problem: Reference to the SubProblem
        dependencies: List of node IDs this node depends on
        dependents: List of node IDs that depend on this node
        depth: Depth in the dependency hierarchy
        complexity: Complexity score for critical path calculation
    """
    node_id: str
    sub_problem: SubProblem
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    depth: int = 0
    complexity: float = 1.0

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DependencyNode):
            return False
        return self.node_id == other.node_id


@dataclass
class DependencyGraph:
    """
    Represents a directed acyclic graph (DAG) of dependencies.

    Attributes:
        nodes: Dictionary mapping node IDs to DependencyNode objects
        edges: Dictionary mapping node IDs to their dependency lists
        execution_order: Topologically sorted list of node IDs
        critical_path: List of node IDs representing the critical path
        parallel_groups: List of lists, where each inner list contains
                       node IDs that can be executed in parallel
    """
    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def is_empty(self) -> bool:
        """Check if the graph is empty."""
        return len(self.nodes) == 0


class DependencyBuilder:
    """
    Builds and analyzes dependency graphs for sub-problem decomposition.

    This class provides methods to:
    - Build a dependency graph from sub-problems
    - Detect circular dependencies
    - Calculate optimal execution order
    - Identify the critical path
    - Find parallelizable tasks
    """

    def __init__(self, validate_on_build: bool = True):
        """
        Initialize the DependencyBuilder.

        Args:
            validate_on_build: If True, validate the graph after building
        """
        self.validate_on_build = validate_on_build
        self._graph: Optional[DependencyGraph] = None

    def build_dependency_graph(
        self,
        sub_problems: List[SubProblem]
    ) -> DependencyGraph:
        """
        Build a dependency graph from a list of sub-problems.

        Args:
            sub_problems: List of SubProblem objects

        Returns:
            DependencyGraph: Constructed dependency graph

        Raises:
            InvalidGraphError: If the graph structure is invalid
            CircularDependencyError: If circular dependencies are detected
        """
        if not sub_problems:
            logger.warning("No sub-problems provided, creating empty graph")
            return DependencyGraph()

        logger.info(f"Building dependency graph from {len(sub_problems)} sub-problems")

        # Initialize graph structure
        graph = DependencyGraph()
        graph.nodes = {}
        graph.edges = defaultdict(list)

        # Create nodes for each sub-problem
        for sp in sub_problems:
            node = DependencyNode(
                node_id=sp.sub_problem_id,
                sub_problem=sp,
                dependencies=list(sp.dependencies) if hasattr(sp, 'dependencies') else [],
                complexity=getattr(sp, 'complexity_score', 1.0)
            )
            graph.nodes[node.node_id] = node
            graph.edges[node.node_id] = list(node.dependencies)

        # Build reverse dependencies (dependents) and validate edges
        for node_id, node in graph.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in graph.nodes:
                    logger.warning(
                        f"Dependency {dep_id} not found in graph for node {node_id}. "
                        f"Treating as external dependency."
                    )
                    continue

                # Add reverse edge
                if dep_id in graph.nodes:
                    graph.nodes[dep_id].dependents.append(node_id)

        # Calculate depths
        self._calculate_node_depths(graph)

        # Validate if requested
        if self.validate_on_build:
            self._validate_graph(graph)

        self._graph = graph
        logger.info(f"Successfully built dependency graph with {len(graph)} nodes")

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful build
        _extract_dependency_knowledge("build_dependency_graph", graph)
        _track_dependency_performance("build_dependency_graph", True, len(graph))

        return graph

    def detect_circular_dependencies(
        self,
        graph: DependencyGraph
    ) -> List[List[str]]:
        """
        Detect circular dependencies in the graph using DFS.

        Args:
            graph: DependencyGraph to analyze

        Returns:
            List of cycles, where each cycle is a list of node IDs

        Raises:
            InvalidGraphError: If the graph is invalid
        """
        if graph.is_empty():
            return []

        logger.info("Detecting circular dependencies")

        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_id: str) -> bool:
            """DFS helper to detect cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in graph.edges.get(node_id, []):
                if neighbor not in graph.nodes:
                    continue  # Skip external dependencies

                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node_id)
            return False

        for node_id in graph.nodes:
            if node_id not in visited:
                dfs(node_id)

        if cycles:
            logger.warning(f"Detected {len(cycles)} circular dependencies")
            for i, cycle in enumerate(cycles, 1):
                logger.warning(f"Cycle {i}: {' -> '.join(cycle)}")

            # **ACTUAL INTEGRATION**: Trigger alert for circular dependencies
            _trigger_dependency_alerts(
                None,
                "detect_circular_dependencies",
                True,  # Detection succeeded
                len(graph),
                len(cycles),
                None,
                {"cycles": cycles[:5]}  # Include first 5 cycles
            )
        else:
            logger.info("No circular dependencies detected")

        return cycles

    def calculate_execution_order(
        self,
        graph: DependencyGraph
    ) -> List[str]:
        """
        Calculate optimal execution order using topological sort (Kahn's algorithm).

        Args:
            graph: DependencyGraph to analyze

        Returns:
            List of node IDs in topological order

        Raises:
            CircularDependencyError: If circular dependencies exist
            InvalidGraphError: If the graph is invalid
        """
        if graph.is_empty():
            return []

        logger.info("Calculating execution order using topological sort")

        # Check for cycles first
        cycles = self.detect_circular_dependencies(graph)
        if cycles:
            raise CircularDependencyError(
                f"Cannot calculate execution order: {len(cycles)} circular "
                f"dependencies detected. First cycle: {' -> '.join(cycles[0])}"
            )

        # Kahn's algorithm
        in_degree: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}

        # Calculate in-degrees
        for node_id in graph.nodes:
            for dep_id in graph.edges.get(node_id, []):
                if dep_id in graph.nodes:
                    in_degree[node_id] += 1

        # Initialize queue with nodes having zero in-degree
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        execution_order: List[str] = []

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)

            # Reduce in-degree for all dependents
            for dependent_id in graph.nodes[node_id].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # Verify all nodes were processed
        if len(execution_order) != len(graph.nodes):
            raise InvalidGraphError(
                f"Graph has circular dependencies or is invalid. "
                f"Processed {len(execution_order)} of {len(graph.nodes)} nodes."
            )

        # Update graph
        graph.execution_order = execution_order

        logger.info(f"Calculated execution order for {len(execution_order)} nodes")
        return execution_order

    def identify_critical_path(
        self,
        graph: DependencyGraph
    ) -> List[str]:
        """
        Identify the critical path (longest weighted path) in the DAG.

        The critical path is determined using dynamic programming:
        - For each node, calculate the maximum distance from any source
        - The path with the highest total complexity is the critical path

        Args:
            graph: DependencyGraph to analyze

        Returns:
            List of node IDs representing the critical path

        Raises:
            InvalidGraphError: If the graph is invalid
        """
        if graph.is_empty():
            return []

        logger.info("Identifying critical path")

        # Calculate execution order if not already done
        if not graph.execution_order:
            self.calculate_execution_order(graph)

        # Dynamic programming: calculate max distance to each node
        dist: Dict[str, float] = {node_id: 0.0 for node_id in graph.nodes}
        prev: Dict[str, Optional[str]] = {node_id: None for node_id in graph.nodes}

        # Process nodes in topological order
        for node_id in graph.execution_order:
            node = graph.nodes[node_id]

            # Update distances for dependents
            for dependent_id in node.dependents:
                new_dist = dist[node_id] + graph.nodes[dependent_id].complexity
                if new_dist > dist[dependent_id]:
                    dist[dependent_id] = new_dist
                    prev[dependent_id] = node_id

        # Find the node with maximum distance
        end_node = max(dist.keys(), key=lambda k: dist[k])

        # Reconstruct the path
        path: List[str] = []
        current = end_node
        while current is not None:
            path.append(current)
            current = prev[current]

        path.reverse()
        graph.critical_path = path

        total_complexity = sum(graph.nodes[nid].complexity for nid in path)
        logger.info(
            f"Critical path identified with {len(path)} nodes, "
            f"total complexity: {total_complexity:.2f}"
        )

        return path

    def find_parallelizable_tasks(
        self,
        graph: DependencyGraph
    ) -> List[List[str]]:
        """
        Find groups of tasks that can be executed in parallel.

        Uses a level-based approach where each level represents tasks
        that can be executed simultaneously.

        Args:
            graph: DependencyGraph to analyze

        Returns:
            List of lists, where each inner list contains node IDs
            that can be executed in parallel

        Raises:
            InvalidGraphError: If the graph is invalid
        """
        if graph.is_empty():
            return []

        logger.info("Finding parallelizable tasks")

        # Calculate execution order if not already done
        if not graph.execution_order:
            self.calculate_execution_order(graph)

        # Group by depth (already calculated in build_dependency_graph)
        depth_groups: Dict[int, List[str]] = defaultdict(list)

        for node_id in graph.execution_order:
            node = graph.nodes[node_id]
            depth_groups[node.depth].append(node_id)

        # Sort groups by depth and convert to list
        parallel_groups = [
            depth_groups[depth]
            for depth in sorted(depth_groups.keys())
        ]

        graph.parallel_groups = parallel_groups

        total_parallelism = sum(len(group) for group in parallel_groups)
        logger.info(
            f"Found {len(parallel_groups)} parallelization levels with "
            f"{total_parallelism} total tasks"
        )

        return parallel_groups

    def _calculate_node_depths(self, graph: DependencyGraph) -> None:
        """
        Calculate the depth of each node in the dependency graph.

        Depth is defined as the longest path from any source (node with no dependencies)
        to the current node.

        Args:
            graph: DependencyGraph to update with depths
        """
        # Initialize all depths to 0
        depth: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}

        # Process nodes in topological order
        # We need to compute order first, but without depths
        temp_order = self._topological_sort_kahn(graph)

        for node_id in temp_order:
            node = graph.nodes[node_id]
            if node.dependencies:
                # Depth is max depth of dependencies + 1
                dep_depths = [
                    depth[dep_id]
                    for dep_id in node.dependencies
                    if dep_id in graph.nodes
                ]
                if dep_depths:
                    depth[node_id] = max(dep_depths) + 1
                else:
                    depth[node_id] = 0
            else:
                depth[node_id] = 0

            # Update node object
            node.depth = depth[node_id]

    def _topological_sort_kahn(self, graph: DependencyGraph) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm (without validation).

        Args:
            graph: DependencyGraph to sort

        Returns:
            List of node IDs in topological order
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}

        for node_id in graph.nodes:
            for dep_id in graph.edges.get(node_id, []):
                if dep_id in graph.nodes:
                    in_degree[node_id] += 1

        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        order: List[str] = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)

            for dependent_id in graph.nodes[node_id].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        return order

    def _validate_graph(self, graph: DependencyGraph) -> None:
        """
        Validate the dependency graph structure.

        Args:
            graph: DependencyGraph to validate

        Raises:
            InvalidGraphError: If the graph is invalid
        """
        if not graph.nodes:
            return

        # Check for orphan nodes (no dependencies and no dependents)
        orphans = []
        for node_id, node in graph.nodes.items():
            if not node.dependencies and not node.dependents:
                orphans.append(node_id)

        if orphans:
            logger.info(f"Found {len(orphans)} orphan nodes: {orphans[:5]}...")

        # Check for self-dependencies
        self_deps = []
        for node_id, node in graph.nodes.items():
            if node_id in node.dependencies:
                self_deps.append(node_id)

        if self_deps:
            raise InvalidGraphError(
                f"Nodes with self-dependencies detected: {self_deps}"
            )

        # Validate edge consistency
        for node_id, deps in graph.edges.items():
            for dep_id in deps:
                if dep_id not in graph.nodes:
                    logger.warning(
                        f"Edge {node_id} -> {dep_id} references non-existent node"
                    )

    def analyze_graph_statistics(self, graph: DependencyGraph) -> Dict[str, Any]:
        """
        Compute statistics about the dependency graph.

        Args:
            graph: DependencyGraph to analyze

        Returns:
            Dictionary containing graph statistics
        """
        if graph.is_empty():
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "avg_dependencies": 0.0,
                "max_depth": 0,
                "sources": 0,
                "sinks": 0,
                "is_dag": True
            }

        total_nodes = len(graph.nodes)
        total_edges = sum(len(deps) for deps in graph.edges.values())

        # Count sources (no dependencies) and sinks (no dependents)
        sources = sum(1 for node in graph.nodes.values() if not node.dependencies)
        sinks = sum(1 for node in graph.nodes.values() if not node.dependents)

        # Calculate average dependencies
        avg_dependencies = total_edges / total_nodes if total_nodes > 0 else 0.0

        # Find max depth
        max_depth = max((node.depth for node in graph.nodes.values()), default=0)

        # Check if DAG
        is_dag = len(self.detect_circular_dependencies(graph)) == 0

        stats = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "avg_dependencies": round(avg_dependencies, 2),
            "max_depth": max_depth,
            "sources": sources,
            "sinks": sinks,
            "is_dag": is_dag,
            "critical_path_length": len(graph.critical_path),
            "parallelization_levels": len(graph.parallel_groups),
        }

        logger.info(f"Graph statistics: {stats}")
        return stats

    def export_graphviz(self, graph: DependencyGraph) -> str:
        """
        Export the graph as Graphviz DOT format for visualization.

        Args:
            graph: DependencyGraph to export

        Returns:
            Graphviz DOT format string
        """
        if graph.is_empty():
            return "digraph G {}"

        lines = ["digraph DependencyGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=rounded];")

        # Add nodes
        for node_id, node in graph.nodes.items():
            label = f"{node_id}\\nDepth: {node.depth}"
            if node_id in graph.critical_path:
                lines.append(f'  "{node_id}" [color=red, fontcolor=red, label="{label}"];')
            else:
                lines.append(f'  "{node_id}" [label="{label}"];')

        # Add edges
        for node_id, deps in graph.edges.items():
            for dep_id in deps:
                if dep_id in graph.nodes:
                    lines.append(f'  "{dep_id}" -> "{node_id}";')

        lines.append("}")
        return "\n".join(lines)


def build_dependency_graph(sub_problems: List[SubProblem]) -> DependencyGraph:
    """
    Convenience function to build a dependency graph.

    Args:
        sub_problems: List of SubProblem objects

    Returns:
        DependencyGraph: Constructed dependency graph

    Example:
        >>> from sovereign_data_models import SubProblem, ProblemStatus
        >>> from datetime import datetime
        >>> sp1 = SubProblem(
        ...     sub_problem_id="task1",
        ...     parent_id=None,
        ...     title="Task 1",
        ...     description="First task",
        ...     status=ProblemStatus.PENDING,
        ...     confidence=0.9,
        ...     assigned_agent=None,
        ...     created_at=datetime.now(),
        ...     completed_at=None
        ... )
        >>> sp2 = SubProblem(
        ...     sub_problem_id="task2",
        ...     parent_id=None,
        ...     title="Task 2",
        ...     description="Second task",
        ...     status=ProblemStatus.PENDING,
        ...     confidence=0.8,
        ...     assigned_agent=None,
        ...     created_at=datetime.now(),
        ...     completed_at=None
        ... )
        >>> # Add dependencies
        >>> sp1.dependencies = []
        >>> sp2.dependencies = ["task1"]
        >>> graph = build_dependency_graph([sp1, sp2])
    """
    builder = DependencyBuilder()
    return builder.build_dependency_graph(sub_problems)


def detect_circular_dependencies(graph: DependencyGraph) -> List[List[str]]:
    """
    Convenience function to detect circular dependencies.

    Args:
        graph: DependencyGraph to analyze

    Returns:
        List of cycles
    """
    builder = DependencyBuilder()
    return builder.detect_circular_dependencies(graph)


def calculate_execution_order(graph: DependencyGraph) -> List[str]:
    """
    Convenience function to calculate execution order.

    Args:
        graph: DependencyGraph to analyze

    Returns:
        List of node IDs in topological order
    """
    builder = DependencyBuilder()
    return builder.calculate_execution_order(graph)


def identify_critical_path(graph: DependencyGraph) -> List[str]:
    """
    Convenience function to identify the critical path.

    Args:
        graph: DependencyGraph to analyze

    Returns:
        List of node IDs representing the critical path
    """
    builder = DependencyBuilder()
    return builder.identify_critical_path(graph)


def find_parallelizable_tasks(graph: DependencyGraph) -> List[List[str]]:
    """
    Convenience function to find parallelizable tasks.

    Args:
        graph: DependencyGraph to analyze

    Returns:
        List of lists with parallelizable node IDs
    """
    builder = DependencyBuilder()
    return builder.find_parallelizable_tasks(graph)


# ============================================================================
# ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Dependency Builder
# ============================================================================

def _trigger_dependency_alerts(
    builder: DependencyBuilder,
    operation: str,
    success: bool,
    num_nodes: int = 0,
    num_cycles: int = 0,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """**ACTUAL INTEGRATION**: Trigger alerts for dependency builder failures or circular dependencies."""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_manager = get_alert_manager()

        # Alert on failures or circular dependencies
        if not success or num_cycles > 0:
            severity = AlertSeverity.HIGH if not success or num_cycles > 0 else AlertSeverity.MEDIUM

            alert_manager.create_alert(
                title=f"Dependency Builder Alert: {operation}",
                description=f"Dependency builder operation '{operation}' " +
                             ("failed" if not success else f"detected {num_cycles} circular dependencies") +
                             (f" in graph with {num_nodes} nodes" if num_nodes > 0 else "") +
                             ". " + (f"Error: {error}" if error else ""),
                severity=severity.value,
                source="dependency_builder",
                component="dependency_graph",
                metadata=metadata or {}
            )

    except Exception as e:
        logger.error(f"Failed to trigger Dependency Builder alert: {e}")


def _extract_dependency_knowledge(
    operation: str,
    graph: DependencyGraph,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """**ACTUAL INTEGRATION**: Extract dependency graph knowledge to knowledge engine."""
    if not KNOWLEDGE_AVAILABLE:
        return False

    try:
        knowledge_engine = get_knowledge_engine()

        artifact = KnowledgeArtifact(
            artifact_id=f"dependency_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            artifact_type="dependency_graph",
            source_component="dependency_builder",
            title=f"Dependency Graph: {operation} ({len(graph)} nodes)",
            content={
                "operation": operation,
                "num_nodes": len(graph),
                "num_edges": sum(len(edges) for edges in graph.edges.values()),
                "max_depth": max([node.depth for node in graph.nodes.values()]) if graph.nodes else 0,
                "timestamp": datetime.now().isoformat()
            },
            metadata={
                "node_ids": list(graph.nodes.keys())[:20],  # Store first 20 node IDs
                **(metadata or {})
            },
            tags=["dependency", "graph", operation]
        )

        knowledge_engine.store_artifact(artifact)
        logger.debug(f"Extracted Dependency knowledge for {operation}")
        return True

    except Exception as e:
        logger.error(f"Failed to extract Dependency knowledge: {e}")
        return False


def _track_dependency_performance(
    operation: str,
    success: bool,
    num_nodes: int = 0,
    build_time: float = 0.0
):
    """**ACTUAL INTEGRATION**: Track dependency builder performance in adaptive selector."""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker()

        # Quality based on success and graph size
        quality = 1.0 if success else 0.0
        if success:
            # Normalize for graph size (100 nodes is ideal baseline)
            size_factor = min(num_nodes / 100.0, 2.0)
            quality = min(quality * size_factor, 1.0)
        quality = max(quality, 0.0)

        performance_data = StrategyPerformanceData(
            strategy_name=f"dependency_builder_{operation}",
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            average_quality=quality,
            last_used=datetime.now(),
            total_attempts=1,
            metadata={
                "operation": operation,
                "num_nodes": num_nodes,
                "build_time": build_time
            }
        )

        if hasattr(tracker, 'performance_history'):
            tracker.performance_history.append(performance_data)
            logger.debug(f"Tracked Dependency Builder performance for {operation}")

    except Exception as e:
        logger.error(f"Failed to track Dependency Builder performance: {e}")


# ============================================================================
# UNIT TESTS
# ============================================================================

import unittest
from datetime import datetime


class TestDependencyBuilder(unittest.TestCase):
    """Unit tests for DependencyBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = DependencyBuilder()

    def create_test_sub_problem(
        self,
        sub_problem_id: str,
        dependencies: Optional[List[str]] = None,
        complexity: float = 1.0
    ) -> SubProblem:
        """Create a test SubProblem."""
        sp = SubProblem(
            sub_problem_id=sub_problem_id,
            parent_id=None,
            title=f"Task {sub_problem_id}",
            description=f"Description for {sub_problem_id}",
            status=ProblemStatus.PENDING,
            confidence=0.8,
            assigned_agent=None,
            created_at=datetime.now(),
            completed_at=None
        )
        sp.dependencies = dependencies or []
        sp.complexity_score = complexity
        return sp

    def test_empty_graph(self):
        """Test building an empty graph."""
        graph = self.builder.build_dependency_graph([])
        self.assertTrue(graph.is_empty())
        self.assertEqual(len(graph), 0)

    def test_simple_graph(self):
        """Test building a simple dependency graph."""
        sp1 = self.create_test_sub_problem("task1", [])
        sp2 = self.create_test_sub_problem("task2", ["task1"])
        sp3 = self.create_test_sub_problem("task3", ["task1"])

        graph = self.builder.build_dependency_graph([sp1, sp2, sp3])

        self.assertEqual(len(graph), 3)
        self.assertIn("task1", graph.nodes)
        self.assertIn("task2", graph.nodes)
        self.assertIn("task3", graph.nodes)

        # Check edges
        self.assertEqual(graph.nodes["task1"].dependents, ["task2", "task3"])
        self.assertEqual(graph.nodes["task2"].dependencies, ["task1"])

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        sp1 = self.create_test_sub_problem("task1", ["task2"])
        sp2 = self.create_test_sub_problem("task2", ["task3"])
        sp3 = self.create_test_sub_problem("task3", ["task1"])

        graph = self.builder.build_dependency_graph([sp1, sp2, sp3])
        cycles = self.builder.detect_circular_dependencies(graph)

        self.assertEqual(len(cycles), 1)
        self.assertIn("task1", cycles[0])
        self.assertIn("task2", cycles[0])
        self.assertIn("task3", cycles[0])

    def test_topological_sort(self):
        """Test topological sort execution order."""
        sp1 = self.create_test_sub_problem("task1", [])
        sp2 = self.create_test_sub_problem("task2", ["task1"])
        sp3 = self.create_test_sub_problem("task3", ["task2"])
        sp4 = self.create_test_sub_problem("task4", ["task1"])

        graph = self.builder.build_dependency_graph([sp1, sp2, sp3, sp4])
        order = self.builder.calculate_execution_order(graph)

        # task1 must come before task2 and task4
        self.assertLess(order.index("task1"), order.index("task2"))
        self.assertLess(order.index("task1"), order.index("task4"))

        # task2 must come before task3
        self.assertLess(order.index("task2"), order.index("task3"))

    def test_critical_path(self):
        """Test critical path identification."""
        sp1 = self.create_test_sub_problem("task1", [], complexity=1.0)
        sp2 = self.create_test_sub_problem("task2", ["task1"], complexity=2.0)
        sp3 = self.create_test_sub_problem("task3", ["task1"], complexity=1.0)
        sp4 = self.create_test_sub_problem("task4", ["task2"], complexity=3.0)

        graph = self.builder.build_dependency_graph([sp1, sp2, sp3, sp4])
        critical_path = self.builder.identify_critical_path(graph)

        # Critical path should be task1 -> task2 -> task4 (highest complexity)
        self.assertIn("task1", critical_path)
        self.assertIn("task2", critical_path)
        self.assertIn("task4", critical_path)
        self.assertNotIn("task3", critical_path)

    def test_parallelizable_tasks(self):
        """Test identification of parallelizable tasks."""
        sp1 = self.create_test_sub_problem("task1", [])
        sp2 = self.create_test_sub_problem("task2", ["task1"])
        sp3 = self.create_test_sub_problem("task3", ["task1"])
        sp4 = self.create_test_sub_problem("task4", ["task2", "task3"])

        graph = self.builder.build_dependency_graph([sp1, sp2, sp3, sp4])
        parallel_groups = self.builder.find_parallelizable_tasks(graph)

        # Level 0: task1 (no dependencies)
        # Level 1: task2, task3 (both depend on task1)
        # Level 2: task4 (depends on task2 and task3)

        self.assertEqual(len(parallel_groups), 3)

        # First level: only task1
        self.assertIn("task1", parallel_groups[0])
        self.assertEqual(len(parallel_groups[0]), 1)

        # Second level: task2 and task3 can run in parallel
        self.assertIn("task2", parallel_groups[1])
        self.assertIn("task3", parallel_groups[1])
        self.assertEqual(len(parallel_groups[1]), 2)

        # Third level: task4
        self.assertIn("task4", parallel_groups[2])
        self.assertEqual(len(parallel_groups[2]), 1)

    def test_self_dependency_error(self):
        """Test that self-dependencies are detected."""
        sp1 = self.create_test_sub_problem("task1", ["task1"])

        with self.assertRaises(InvalidGraphError):
            self.builder.build_dependency_graph([sp1])

    def test_orphan_nodes(self):
        """Test handling of orphan nodes."""
        sp1 = self.create_test_sub_problem("task1", [])

        graph = self.builder.build_dependency_graph([sp1])

        # task1 has no dependencies and no dependents
        self.assertEqual(len(graph.nodes["task1"].dependencies), 0)
        self.assertEqual(len(graph.nodes["task1"].dependents), 0)
        self.assertEqual(graph.nodes["task1"].depth, 0)

    def test_graph_statistics(self):
        """Test graph statistics calculation."""
        sp1 = self.create_test_sub_problem("task1", [])
        sp2 = self.create_test_sub_problem("task2", ["task1"])
        sp3 = self.create_test_sub_problem("task3", ["task1"])

        graph = self.builder.build_dependency_graph([sp1, sp2, sp3])
        stats = self.builder.analyze_graph_statistics(graph)

        self.assertEqual(stats["total_nodes"], 3)
        self.assertEqual(stats["total_edges"], 2)
        self.assertEqual(stats["sources"], 1)  # task1
        self.assertEqual(stats["sinks"], 2)  # task2, task3
        self.assertTrue(stats["is_dag"])

    def test_export_graphviz(self):
        """Test Graphviz export."""
        sp1 = self.create_test_sub_problem("task1", [])
        sp2 = self.create_test_sub_problem("task2", ["task1"])

        graph = self.builder.build_dependency_graph([sp1, sp2])
        dot = self.builder.export_graphviz(graph)

        self.assertIn("digraph DependencyGraph", dot)
        self.assertIn("task1", dot)
        self.assertIn("task2", dot)
        self.assertIn('"task1" -> "task2"', dot)

    def test_execution_order_with_cycle(self):
        """Test that execution order fails with circular dependencies."""
        sp1 = self.create_test_sub_problem("task1", ["task2"])
        sp2 = self.create_test_sub_problem("task2", ["task1"])

        graph = self.builder.build_dependency_graph([sp1, sp2])

        with self.assertRaises(CircularDependencyError):
            self.builder.calculate_execution_order(graph)

    def test_complex_dependency_chain(self):
        """Test a complex dependency chain."""
        # A -> B -> C -> D
        #      \-> E
        sp_a = self.create_test_sub_problem("A", [])
        sp_b = self.create_test_sub_problem("B", ["A"])
        sp_c = self.create_test_sub_problem("C", ["B"])
        sp_d = self.create_test_sub_problem("D", ["C"])
        sp_e = self.create_test_sub_problem("E", ["B"])

        graph = self.builder.build_dependency_graph([sp_a, sp_b, sp_c, sp_d, sp_e])
        order = self.builder.calculate_execution_order(graph)

        # Verify order constraints
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("B"), order.index("C"))
        self.assertLess(order.index("B"), order.index("E"))
        self.assertLess(order.index("C"), order.index("D"))

        # Verify depths
        self.assertEqual(graph.nodes["A"].depth, 0)
        self.assertEqual(graph.nodes["B"].depth, 1)
        self.assertEqual(graph.nodes["C"].depth, 2)
        self.assertEqual(graph.nodes["D"].depth, 3)
        self.assertEqual(graph.nodes["E"].depth, 2)

    def test_external_dependencies(self):
        """Test handling of external dependencies not in graph."""
        sp1 = self.create_test_sub_problem("task1", ["external_task"])
        sp2 = self.create_test_sub_problem("task2", [])

        graph = self.builder.build_dependency_graph([sp1, sp2])

        # external_task is not in nodes
        self.assertNotIn("external_task", graph.nodes)

        # task1 should still be in the graph
        self.assertIn("task1", graph.nodes)

        # Should be able to calculate execution order
        # (external dependencies are ignored for ordering)
        order = self.builder.calculate_execution_order(graph)
        self.assertIn("task1", order)
        self.assertIn("task2", order)


class TestDependencyBuilderIntegration(unittest.TestCase):
    """Integration tests for dependency builder with realistic scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = DependencyBuilder()

    def create_sub_problem(
        self,
        sub_problem_id: str,
        dependencies: Optional[List[str]] = None,
        complexity: float = 1.0
    ) -> SubProblem:
        """Create a test SubProblem."""
        sp = SubProblem(
            sub_problem_id=sub_problem_id,
            parent_id=None,
            title=f"Problem {sub_problem_id}",
            description=f"Solve {sub_problem_id}",
            status=ProblemStatus.PENDING,
            confidence=0.8,
            assigned_agent=None,
            created_at=datetime.now(),
            completed_at=None
        )
        sp.dependencies = dependencies or []
        sp.complexity_score = complexity
        return sp

    def test_software_deployment_scenario(self):
        """
        Test a realistic software deployment scenario.

        Structure:
        - design (no deps)
        - backend_dev (depends on: design)
        - frontend_dev (depends on: design)
        - api_integration (depends on: backend_dev, frontend_dev)
        - testing (depends on: api_integration)
        - deployment (depends on: testing)
        """
        design = self.create_sub_problem("design", [])
        backend = self.create_sub_problem("backend_dev", ["design"], complexity=3.0)
        frontend = self.create_sub_problem("frontend_dev", ["design"], complexity=2.0)
        api = self.create_sub_problem("api_integration", ["backend_dev", "frontend_dev"], complexity=2.5)
        testing = self.create_sub_problem("testing", ["api_integration"], complexity=1.5)
        deployment = self.create_sub_problem("deployment", ["testing"], complexity=1.0)

        graph = self.builder.build_dependency_graph([
            design, backend, frontend, api, testing, deployment
        ])

        # Test execution order
        order = self.builder.calculate_execution_order(graph)
        self.assertLess(order.index("design"), order.index("backend_dev"))
        self.assertLess(order.index("design"), order.index("frontend_dev"))
        self.assertLess(order.index("backend_dev"), order.index("api_integration"))
        self.assertLess(order.index("frontend_dev"), order.index("api_integration"))

        # Test parallelization
        parallel_groups = self.builder.find_parallelizable_tasks(graph)
        # Level 0: design
        self.assertIn("design", parallel_groups[0])
        # Level 1: backend_dev, frontend_dev (can run in parallel)
        self.assertIn("backend_dev", parallel_groups[1])
        self.assertIn("frontend_dev", parallel_groups[1])

        # Test critical path (should be design -> backend_dev -> api_integration -> testing -> deployment)
        critical_path = self.builder.identify_critical_path(graph)
        self.assertIn("design", critical_path)
        self.assertIn("backend_dev", critical_path)
        self.assertIn("deployment", critical_path)

        # Test statistics
        stats = self.builder.analyze_graph_statistics(graph)
        self.assertEqual(stats["total_nodes"], 6)
        self.assertTrue(stats["is_dag"])

    def test_microservices_scenario(self):
        """
        Test a microservices architecture scenario.

        Structure:
        - auth_service (no deps)
        - user_service (depends on: auth_service)
        - payment_service (depends on: auth_service)
        - order_service (depends on: user_service, payment_service)
        - notification_service (depends on: order_service)
        """
        auth = self.create_sub_problem("auth_service", [])
        user = self.create_sub_problem("user_service", ["auth_service"])
        payment = self.create_sub_problem("payment_service", ["auth_service"], complexity=2.5)
        order = self.create_sub_problem("order_service", ["user_service", "payment_service"])
        notification = self.create_sub_problem("notification_service", ["order_service"])

        graph = self.builder.build_dependency_graph([
            auth, user, payment, order, notification
        ])

        # Test that user and payment can run in parallel
        parallel_groups = self.builder.find_parallelizable_tasks(graph)
        level_1 = parallel_groups[1]
        self.assertIn("user_service", level_1)
        self.assertIn("payment_service", level_1)

        # Test critical path includes payment (higher complexity)
        critical_path = self.builder.identify_critical_path(graph)
        self.assertIn("payment_service", critical_path)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """
    Example: Basic usage of DependencyBuilder.

    This example demonstrates how to build a dependency graph
    from sub-problems and analyze it.
    """
    from sovereign_data_models import SubProblem, ProblemStatus
    from datetime import datetime

    # Create sub-problems
    design = SubProblem(
        sub_problem_id="design",
        parent_id=None,
        title="Design System",
        description="Design the core system architecture",
        status=ProblemStatus.PENDING,
        confidence=0.9,
        assigned_agent="architect",
        created_at=datetime.now(),
        completed_at=None
    )
    design.dependencies = []
    design.complexity_score = 2.0

    implementation = SubProblem(
        sub_problem_id="implementation",
        parent_id=None,
        title="Implement Core",
        description="Implement core functionality",
        status=ProblemStatus.PENDING,
        confidence=0.8,
        assigned_agent="developer",
        created_at=datetime.now(),
        completed_at=None
    )
    implementation.dependencies = ["design"]
    implementation.complexity_score = 3.0

    testing = SubProblem(
        sub_problem_id="testing",
        parent_id=None,
        title="Test System",
        description="Test the implementation",
        status=ProblemStatus.PENDING,
        confidence=0.7,
        assigned_agent="tester",
        created_at=datetime.now(),
        completed_at=None
    )
    testing.dependencies = ["implementation"]
    testing.complexity_score = 1.5

    # Build dependency graph
    builder = DependencyBuilder()
    graph = builder.build_dependency_graph([design, implementation, testing])

    # Analyze graph
    execution_order = builder.calculate_execution_order(graph)
    print(f"Execution order: {' -> '.join(execution_order)}")

    critical_path = builder.identify_critical_path(graph)
    print(f"Critical path: {' -> '.join(critical_path)}")

    parallel_tasks = builder.find_parallelizable_tasks(graph)
    print(f"Parallelization levels: {len(parallel_tasks)}")

    stats = builder.analyze_graph_statistics(graph)
    print(f"Graph statistics: {stats}")


def example_detecting_cycles():
    """
    Example: Detecting circular dependencies.

    This example shows how to detect and handle circular dependencies.
    """
    from sovereign_data_models import SubProblem, ProblemStatus
    from datetime import datetime

    # Create sub-problems with circular dependency
    task_a = SubProblem(
        sub_problem_id="task_a",
        parent_id=None,
        title="Task A",
        description="Task A depends on B",
        status=ProblemStatus.PENDING,
        confidence=0.8,
        assigned_agent=None,
        created_at=datetime.now(),
        completed_at=None
    )
    task_a.dependencies = ["task_b"]

    task_b = SubProblem(
        sub_problem_id="task_b",
        parent_id=None,
        title="Task B",
        description="Task B depends on A",
        status=ProblemStatus.PENDING,
        confidence=0.8,
        assigned_agent=None,
        created_at=datetime.now(),
        completed_at=None
    )
    task_b.dependencies = ["task_a"]

    # Build graph and detect cycles
    builder = DependencyBuilder()
    graph = builder.build_dependency_graph([task_a, task_b])

    cycles = builder.detect_circular_dependencies(graph)
    if cycles:
        print(f"WARNING: Detected {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles, 1):
            print(f"  Cycle {i}: {' -> '.join(cycle)}")
    else:
        print("OK: No circular dependencies detected")


def example_parallel_execution():
    """
    Example: Finding parallelizable tasks for execution.

    This example demonstrates how to identify tasks that can run
    in parallel to optimize execution time.
    """
    from sovereign_data_models import SubProblem, ProblemStatus
    from datetime import datetime

    # Create a complex dependency structure
    tasks = [
        ("setup", [], 1.0),
        ("backend", ["setup"], 3.0),
        ("frontend", ["setup"], 2.0),
        ("database", ["setup"], 2.5),
        ("integration", ["backend", "frontend", "database"], 2.0),
        ("testing", ["integration"], 1.5),
        ("deployment", ["testing"], 1.0),
    ]

    sub_problems = []
    for task_id, deps, complexity in tasks:
        sp = SubProblem(
            sub_problem_id=task_id,
            parent_id=None,
            title=task_id.title(),
            description=f"Execute {task_id}",
            status=ProblemStatus.PENDING,
            confidence=0.8,
            assigned_agent=None,
            created_at=datetime.now(),
            completed_at=None
        )
        sp.dependencies = deps
        sp.complexity_score = complexity
        sub_problems.append(sp)

    # Build and analyze
    builder = DependencyBuilder()
    graph = builder.build_dependency_graph(sub_problems)

    parallel_groups = builder.find_parallelizable_tasks(graph)

    print("Parallel Execution Plan:")
    print("=" * 50)
    for level, tasks in enumerate(parallel_groups):
        print(f"Level {level}: {len(tasks)} task(s) can run in parallel")
        for task_id in tasks:
            complexity = graph.nodes[task_id].complexity
            print(f"  - {task_id} (complexity: {complexity})")
        print()

    print(f"Total parallelization levels: {len(parallel_groups)}")


def example_visualization():
    """
    Example: Exporting graph for visualization.

    This example shows how to export the dependency graph in
    Graphviz DOT format for visualization.
    """
    from sovereign_data_models import SubProblem, ProblemStatus
    from datetime import datetime

    # Create sample workflow
    tasks = []
    for i, (task_id, deps) in enumerate([
        ("plan", []),
        ("design", ["plan"]),
        ("implement", ["design"]),
        ("test", ["implement"]),
        ("deploy", ["test"]),
    ]):
        sp = SubProblem(
            sub_problem_id=task_id,
            parent_id=None,
            title=task_id.title(),
            description=f"Execute {task_id}",
            status=ProblemStatus.PENDING,
            confidence=0.8,
            assigned_agent=None,
            created_at=datetime.now(),
            completed_at=None
        )
        sp.dependencies = deps
        sp.complexity_score = 1.0
        tasks.append(sp)

    # Build graph
    builder = DependencyBuilder()
    graph = builder.build_dependency_graph(tasks)

    # Identify critical path for visualization
    builder.identify_critical_path(graph)

    # Export to Graphviz
    dot = builder.export_graphviz(graph)

    print("Graphviz DOT Format:")
    print("=" * 50)
    print(dot)
    print()
    print("To visualize, save the output to a .dot file and use:")
    print("  dot -Tpng workflow.dot -o workflow.png")


if __name__ == "__main__":
    # Run examples
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    example_basic_usage()

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Detecting Cycles")
    print("=" * 60)
    example_detecting_cycles()

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Parallel Execution")
    print("=" * 60)
    example_parallel_execution()

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Visualization")
    print("=" * 60)
    example_visualization()

    # Run unit tests
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    unittest.main(argv=[''], verbosity=2, exit=False)
