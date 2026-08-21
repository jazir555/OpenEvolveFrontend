"""
Advanced Dependency Analyzer for Decomposition Engine

This module provides sophisticated dependency analysis capabilities including:
- Circular dependency detection using DFS with 3-color algorithm
- Critical path calculation using topological sort and longest path
- Parallelization opportunity detection using BFS level traversal
- Success dependency validation

PRODUCTION-GRADE IMPLEMENTATION:
- Robust graph algorithms with proper error handling
- Comprehensive edge case handling
- Integration with quality assessment system
- Full backward compatibility
"""
from __future__ import annotations


import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque, defaultdict

from sovereign_data_models import SubProblem, ComplexityScore
from utils.symbolic_analyzer import SymbolicAnalyzer

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """
    Advanced dependency analysis for sub-problem decomposition.

    Provides sophisticated graph-theoretic analysis of dependency relationships
    including cycle detection, critical path analysis, and parallelization opportunities.
    """

    def __init__(self):
        """Initialize the DependencyAnalyzer."""
        self.logger = logging.getLogger(__name__)
        self.symbolic_analyzer = SymbolicAnalyzer()

    def detect_cycles(self, sub_problems: List[SubProblem]) -> List[List[str]]:
        """
        Detect circular dependencies using DFS with 3-color algorithm.

        Uses depth-first search with node coloring (white=unvisited, gray=visiting,
        black=visited) to efficiently detect cycles in the dependency graph.

        Args:
            sub_problems: List of sub-problems to analyze

        Returns:
            List of cycles, where each cycle is a list of sub-problem IDs in order.
            Empty list if no cycles found.

        Algorithm:
            1. Build adjacency list from dependencies
            2. Perform DFS with 3-color marking
            3. When we encounter a gray node, we've found a cycle
            4. Backtrack to extract the cycle path

        Complexity:
            O(V + E) where V = number of sub-problems, E = number of dependencies
        """
        if not sub_problems:
            return []

        # Build adjacency list
        adj_list = self._build_adjacency_list(sub_problems)
        nodes = list(adj_list.keys())

        # Three colors for DFS: 0=white (unvisited), 1=gray (visiting), 2=black (visited)
        color = {node: 0 for node in nodes}
        parent = {node: None for node in nodes}
        cycles = []

        def dfs(node: str, path: List[str]) -> None:
            """Depth-first search with cycle detection."""
            color[node] = 1  # Mark as gray (visiting)
            path.append(node)

            for neighbor in adj_list.get(node, []):
                if neighbor not in color:
                    continue  # Skip nodes not in our graph

                if color[neighbor] == 1:
                    # Found a cycle! Extract it from the path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif color[neighbor] == 0:
                    # Continue DFS
                    parent[neighbor] = node
                    dfs(neighbor, path)

            path.pop()
            color[node] = 2  # Mark as black (visited)

        # Run DFS from each unvisited node
        for node in nodes:
            if color[node] == 0:
                dfs(node, [])

        # Remove duplicate cycles (may appear in reverse order)
        unique_cycles = self._deduplicate_cycles(cycles)

        self.logger.info(f"Detected {len(unique_cycles)} circular dependencies")
        return unique_cycles

    def calculate_critical_path(self, sub_problems: List[SubProblem]) -> Dict[str, Any]:
        """
        Calculate the critical path using topological sort and longest path algorithm.

        The critical path is the longest path through the dependency graph in terms
        of estimated effort. It represents the minimum time required to complete all
        sub-problems, assuming no parallelization.

        Args:
            sub_problems: List of sub-problems to analyze

        Returns:
            Dictionary containing:
                - critical_path: List of sub-problem IDs in order
                - critical_path_length: Number of nodes in critical path
                - estimated_duration: Total effort (hours) on critical path
                - slack_time_per_node: Dict mapping node IDs to slack time
                - all_paths: All topological paths with their durations

        Algorithm:
            1. Perform topological sort using Kahn's algorithm
            2. Calculate longest distances from source nodes
            3. Backtrack to find the critical path
            4. Calculate slack time for each node (earliest_latest - earliest_earliest)

        Complexity:
            O(V + E) where V = number of sub-problems, E = number of dependencies

        Raises:
            ValueError: If the graph contains cycles (use detect_cycles first)
        """
        if not sub_problems:
            return {
                "critical_path": [],
                "critical_path_length": 0,
                "estimated_duration": 0.0,
                "slack_time_per_node": {},
                "all_paths": []
            }

        # Check for cycles first
        cycles = self.detect_cycles(sub_problems)
        if cycles:
            raise ValueError(
                f"Cannot calculate critical path: graph contains {len(cycles)} cycle(s). "
                f"Resolve cycles before calculating critical path."
            )

        # Build adjacency list and get effort for each node
        adj_list = self._build_adjacency_list(sub_problems)
        node_effort = {sp.id: float(sp.estimated_effort) for sp in sub_problems}

        # Kahn's algorithm for topological sort
        in_degree = {sp.id: len(sp.dependencies) for sp in sub_problems}
        queue = deque([sp_id for sp_id, degree in in_degree.items() if degree == 0])
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)

            for neighbor in adj_list.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != len(sub_problems):
            # Graph has a cycle (shouldn't happen after detect_cycles check)
            raise ValueError("Graph contains a cycle")

        # Calculate longest distance (earliest start time) for each node
        dist = {sp_id: 0.0 for sp_id in node_effort}

        for node in topo_order:
            for neighbor in adj_list.get(node, []):
                if dist[neighbor] < dist[node] + node_effort[node]:
                    dist[neighbor] = dist[node] + node_effort[node]

        # Find the node with maximum distance (end of critical path)
        max_dist_node = max(dist.keys(), key=lambda x: dist[x])
        max_dist = dist[max_dist_node]

        # Backtrack to find critical path
        critical_path = []
        current = max_dist_node

        while current is not None:
            critical_path.append(current)
            # Find predecessor on critical path
            predecessors = [
                pred for pred in topo_order
                if current in adj_list.get(pred, []) and
                abs(dist[current] - (dist[pred] + node_effort[pred])) < 0.01
            ]
            current = predecessors[0] if predecessors else None

        critical_path.reverse()

        # Calculate latest start time for each node (reverse topological order)
        latest_start = {sp_id: float('inf') for sp_id in node_effort}
        latest_start[max_dist_node] = max_dist

        for node in reversed(topo_order):
            if adj_list.get(node, []):
                latest_start[node] = min(
                    latest_start[neighbor] - node_effort[node]
                    for neighbor in adj_list.get(node, [])
                )
            else:
                # Sink node
                latest_start[node] = dist[node]

        # Calculate slack time (latest - earliest)
        slack_time = {
            sp_id: latest_start[sp_id] - dist[sp_id]
            for sp_id in node_effort
        }

        # Calculate all paths for detailed analysis
        all_paths = self._find_all_paths(adj_list, node_effort)

        return {
            "critical_path": critical_path,
            "critical_path_length": len(critical_path),
            "estimated_duration": max_dist + node_effort[max_dist_node],
            "slack_time_per_node": slack_time,
            "all_paths": all_paths
        }

    def find_parallelization_opportunities(
        self,
        sub_problems: List[SubProblem]
    ) -> Dict[str, Any]:
        """
        Find parallelization opportunities using BFS level-by-level traversal.

        Analyzes the dependency graph to identify groups of sub-problems that can
        be executed in parallel. Each level represents a set of tasks that can
        run simultaneously.

        Args:
            sub_problems: List of sub-problems to analyze

        Returns:
            Dictionary containing:
                - parallelizable_groups: List of lists, where each inner list
                  contains IDs of sub-problems that can run in parallel
                - estimated_parallel_speedup: Theoretical maximum speedup
                  (sequential_time / parallel_time)
                - total_groups: Number of sequential groups (levels)
                - group_sizes: List of group sizes
                - parallelization_efficiency: Ratio of actual speedup to ideal

        Algorithm:
            1. Build dependency graph
            2. Use BFS to traverse level by level
            3. Each level represents tasks that can run in parallel
            4. Calculate theoretical speedup

        Complexity:
            O(V + E) where V = number of sub-problems, E = number of dependencies

        Note:
            This assumes perfect parallelization with no overhead. Real-world
            speedup will be lower due to communication, synchronization, and
            resource constraints.
        """
        if not sub_problems:
            return {
                "parallelizable_groups": [],
                "estimated_parallel_speedup": 1.0,
                "total_groups": 0,
                "group_sizes": [],
                "parallelization_efficiency": 1.0
            }

        # Build adjacency list
        adj_list = self._build_adjacency_list(sub_problems)
        in_degree = {sp.id: len(sp.dependencies) for sp in sub_problems}
        node_effort = {sp.id: float(sp.estimated_effort) for sp in sub_problems}

        # Check for cycles
        cycles = self.detect_cycles(sub_problems)
        if cycles:
            self.logger.warning(
                f"Graph contains {len(cycles)} cycle(s). "
                f"Parallelization analysis may be inaccurate."
            )

        # BFS level-by-level traversal
        levels = []
        current_level = [sp_id for sp_id, degree in in_degree.items() if degree == 0]
        visited = set(current_level)

        while current_level:
            levels.append(current_level.copy())
            next_level = []

            # Find all nodes that depend on current level nodes
            for node in current_level:
                for neighbor in adj_list.get(node, []):
                    if neighbor not in visited:
                        # Check if all dependencies are satisfied
                        neighbor_deps = self._get_dependencies_for_node(neighbor, sub_problems)
                        deps_satisfied = all(dep in visited for dep in neighbor_deps)
                        if deps_satisfied:
                            next_level.append(neighbor)
                            visited.add(neighbor)

            current_level = next_level

        # Calculate sequential and parallel execution time
        total_effort = sum(node_effort.values())
        sequential_time = total_effort

        # Parallel time = sum of max effort in each level
        parallel_time = sum(
            max(node_effort.get(node, 0) for node in level)
            for level in levels
        )

        # Calculate speedup
        estimated_speedup = sequential_time / max(parallel_time, 0.01)

        # Calculate ideal speedup (all tasks perfectly parallel)
        max_parallel_tasks = max(len(level) for level in levels) if levels else 1
        ideal_speedup = min(max_parallel_tasks, len(sub_problems))

        # Calculate efficiency
        efficiency = estimated_speedup / max(ideal_speedup, 1.0)

        group_sizes = [len(level) for level in levels]

        return {
            "parallelizable_groups": levels,
            "estimated_parallel_speedup": min(estimated_speedup, ideal_speedup),
            "total_groups": len(levels),
            "group_sizes": group_sizes,
            "parallelization_efficiency": min(efficiency, 1.0)
        }

    def validate_success_dependencies(
        self,
        sub_problems: List[SubProblem]
    ) -> Dict[str, Any]:
        """
        Validate that all success_dependencies reference valid sub-problems.

        Checks for:
        - Invalid references (non-existent sub-problem IDs)
        - Missing dependencies (referenced but not in list)
        - Self-dependencies (sub-problem depends on itself)
        - Circular dependencies in success_dependencies

        Args:
            sub_problems: List of sub-problems to validate

        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating overall validity
                - errors: List of error messages
                - warnings: List of warning messages
                - invalid_references: Dict mapping invalid refs to sub-problems
                - self_dependencies: List of sub-problems with self-deps
                - missing_dependencies: Dict mapping missing deps to sub-problems
                - referenced_subproblems: Dict of what each sub-problem references
        """
        if not sub_problems:
            return {
                "is_valid": True,
                "errors": [],
                "warnings": ["No sub-problems to validate"],
                "invalid_references": {},
                "self_dependencies": [],
                "missing_dependencies": {},
                "referenced_subproblems": {}
            }

        sub_problem_ids = {sp.id for sp in sub_problems}
        errors = []
        warnings = []
        invalid_references = {}
        self_dependencies = []
        missing_dependencies = {}
        referenced_subproblems = {}

        for sp in sub_problems:
            # Get success_dependencies from metadata
            success_deps = sp.metadata.get('success_dependencies', '')
            referenced = []

            if not success_deps or not isinstance(success_deps, str):
                referenced_subproblems[sp.id] = []
                continue

            # Parse success_dependencies (may contain sub-problem numbers or IDs)
            # Try to extract references
            import re
            # Look for patterns like "sub-problem 1", "task 2", ID references, etc.
            refs = re.findall(r'(?:sub-problem|task|problem)\s*(\d+)', success_deps, re.IGNORECASE)
            referenced.extend([f"subproblem_{r}" for r in refs])

            # Also look for direct ID references
            id_refs = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', success_deps, re.IGNORECASE)
            referenced.extend(id_refs)

            referenced_subproblems[sp.id] = referenced

            # Check each reference
            for ref in referenced:
                if ref == sp.id:
                    self_dependencies.append(sp.id)
                    errors.append(f"Sub-problem '{sp.title}' ({sp.id}) has self-dependency in success_dependencies")

                elif ref not in sub_problem_ids:
                    # Try to find if it's a reference by number
                    is_number_ref = any(ref == f"subproblem_{i+1}" for i in range(len(sub_problems)))

                    if not is_number_ref:
                        invalid_references[ref] = sp.id
                        errors.append(
                            f"Sub-problem '{sp.title}' ({sp.id}) references "
                            f"non-existent sub-problem '{ref}' in success_dependencies"
                        )

        # Check for circular dependencies
        cycles = self.detect_cycles(sub_problems)
        if cycles:
            for cycle in cycles:
                cycle_str = " -> ".join(cycle)
                warnings.append(f"Circular dependency detected: {cycle_str}")

        # Check for missing regular dependencies
        for sp in sub_problems:
            for dep in sp.dependencies:
                if dep not in sub_problem_ids:
                    missing_dependencies[dep] = sp.id
                    errors.append(
                        f"Sub-problem '{sp.title}' ({sp.id}) has missing "
                        f"dependency '{dep}' (not in sub-problem list)"
                    )

        is_valid = len(errors) == 0

        self.logger.info(
            f"Validation complete: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}"
        )

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "invalid_references": invalid_references,
            "self_dependencies": self_dependencies,
            "missing_dependencies": missing_dependencies,
            "referenced_subproblems": referenced_subproblems
        }

    def _build_adjacency_list(self, sub_problems: List[SubProblem]) -> Dict[str, List[str]]:
        """Build adjacency list representation of dependency graph."""
        adj_list = {sp.id: [] for sp in sub_problems}

        for sp in sub_problems:
            # Dependencies: sp depends on these, so edges go from deps -> sp
            for dep in sp.dependencies:
                if dep in adj_list:
                    adj_list[dep].append(sp.id)

        return adj_list

    def _deduplicate_cycles(self, cycles: List[List[str]]) -> List[List[str]]:
        """Remove duplicate cycles (may appear in reverse order)."""
        unique_cycles = []
        seen = set()

        for cycle in cycles:
            # Create normalized representation (start from smallest ID)
            cycle_str = " -> ".join(cycle)
            cycle_str_reversed = " -> ".join(reversed(cycle))

            if cycle_str not in seen and cycle_str_reversed not in seen:
                unique_cycles.append(cycle)
                seen.add(cycle_str)

        return unique_cycles

    def _find_all_paths(
        self,
        adj_list: Dict[str, List[str]],
        node_effort: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Find all topological paths with their durations."""
        # Find source nodes (no incoming edges)
        all_targets = set()
        for targets in adj_list.values():
            all_targets.update(targets)

        source_nodes = [node for node in adj_list.keys() if node not in all_targets]

        # Find all paths from each source
        all_paths = []

        for source in source_nodes:
            paths = self._find_paths_from_source(adj_list, source)
            for path in paths:
                duration = sum(node_effort.get(node, 0) for node in path)
                all_paths.append({
                    "path": path,
                    "duration": duration,
                    "length": len(path)
                })

        # Sort by duration (longest first)
        all_paths.sort(key=lambda x: x["duration"], reverse=True)

        return all_paths

    def _find_paths_from_source(
        self,
        adj_list: Dict[str, List[str]],
        source: str
    ) -> List[List[str]]:
        """Find all paths starting from a source node."""
        paths = []

        def dfs(node: str, current_path: List[str]) -> None:
            current_path.append(node)

            successors = adj_list.get(node, [])
            if not successors:
                # Sink node - save path
                paths.append(current_path.copy())
            else:
                for successor in successors:
                    dfs(successor, current_path)

            current_path.pop()

        dfs(source, [])
        return paths

    def _get_dependencies_for_node(
        self,
        node_id: str,
        sub_problems: List[SubProblem]
    ) -> List[str]:
        """Get dependency list for a specific node."""
        for sp in sub_problems:
            if sp.id == node_id:
                return sp.dependencies
        return []

    def build_entanglement_matrix(
        self,
        sub_problems: List[SubProblem]
    ) -> Dict[str, Set[str]]:
        """
        Build a symbolic entanglement matrix based on shared interface symbols.

        Returns:
            Dict mapping sub-problem id -> set of entangled sub-problem ids.
        """
        matrix: Dict[str, Set[str]] = {sp.id: set() for sp in sub_problems}
        symbol_map: Dict[str, Set[str]] = {}

        for sp in sub_problems:
            analysis = self.symbolic_analyzer.analyze(sp.description or "")
            for sym in analysis.symbols:
                symbol_map.setdefault(sym, set()).add(sp.id)

        for sym, components in symbol_map.items():
            if len(components) < 2:
                continue
            for comp in components:
                matrix[comp].update({c for c in components if c != comp})

        return matrix


def analyze_dependency_graph(
    sub_problems: List[SubProblem],
    analyzer: Optional[DependencyAnalyzer] = None
) -> Dict[str, Any]:
    """
    Convenience function to perform complete dependency analysis.

    Args:
        sub_problems: List of sub-problems to analyze
        analyzer: Optional DependencyAnalyzer instance (created if not provided)

    Returns:
        Comprehensive analysis dictionary containing:
            - cycles: List of circular dependencies
            - critical_path: Critical path analysis results
            - parallelization: Parallelization opportunity analysis
            - validation: Success dependency validation results
            - summary: High-level summary statistics
    """
    if analyzer is None:
        analyzer = DependencyAnalyzer()

    logger.info(f"Performing complete dependency analysis on {len(sub_problems)} sub-problems")

    # Detect cycles
    cycles = analyzer.detect_cycles(sub_problems)

    # Calculate critical path (skip if cycles found)
    try:
        critical_path = analyzer.calculate_critical_path(sub_problems)
    except ValueError as e:
        logger.warning(f"Critical path calculation failed: {e}")
        critical_path = {
            "error": str(e),
            "critical_path": [],
            "critical_path_length": 0,
            "estimated_duration": 0.0,
            "slack_time_per_node": {}
        }

    # Find parallelization opportunities
    parallelization = analyzer.find_parallelization_opportunities(sub_problems)

    # Validate success dependencies
    validation = analyzer.validate_success_dependencies(sub_problems)

    # Calculate summary statistics
    total_dependencies = sum(len(sp.dependencies) for sp in sub_problems)
    avg_dependencies = total_dependencies / len(sub_problems) if sub_problems else 0

    summary = {
        "total_subproblems": len(sub_problems),
        "total_dependencies": total_dependencies,
        "average_dependencies": round(avg_dependencies, 2),
        "has_cycles": len(cycles) > 0,
        "num_cycles": len(cycles),
        "is_valid": validation["is_valid"],
        "parallelizable": parallelization["estimated_parallel_speedup"] > 1.0,
        "estimated_speedup": round(parallelization["estimated_parallel_speedup"], 2)
    }

    return {
        "cycles": cycles,
        "critical_path": critical_path,
        "parallelization": parallelization,
        "validation": validation,
        "summary": summary
    }
