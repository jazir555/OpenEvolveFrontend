"""
Sovereign-Grade Problem Decomposition System - Dependency Manager

Manages dependencies between sub-problems, validates graphs, and optimizes execution.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict

from sovereign_data_models import SubProblem, DependencyGraph, ValidationResult, generate_id


logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages dependencies between sub-problems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_graph(self, sub_problems: List[SubProblem]) -> DependencyGraph:
        """
        Constructs dependency graph from sub-problems.
        
        Args:
            sub_problems: List of sub-problems
            
        Returns:
            DependencyGraph with nodes, edges, and analysis
        """
        self.logger.info(f"Building dependency graph for {len(sub_problems)} sub-problems")
        
        nodes = {sp.id: sp for sp in sub_problems}
        edges = {sp.id: sp.dependencies for sp in sub_problems}
        
        # Detect cycles
        cycles = self.detect_cycles(DependencyGraph(nodes=nodes, edges=edges))
        if cycles:
            self.logger.warning(f"Detected {len(cycles)} cycles in dependency graph")
        
        # Calculate critical path
        critical_path = self.find_critical_path(DependencyGraph(nodes=nodes, edges=edges))
        
        # Identify parallel opportunities
        parallel_groups = self.identify_parallel_opportunities(DependencyGraph(nodes=nodes, edges=edges))
        
        # Calculate execution order
        execution_order = self.calculate_execution_order(DependencyGraph(nodes=nodes, edges=edges))
        
        graph = DependencyGraph(
            nodes=nodes,
            edges=edges,
            critical_path=critical_path,
            parallel_groups=parallel_groups,
            execution_order=execution_order
        )
        
        self.logger.info(f"Dependency graph built: {len(critical_path)} nodes in critical path, "
                        f"{len(parallel_groups)} parallel groups")
        
        return graph
    
    def detect_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """
        Identifies circular dependencies using DFS.
        
        Args:
            graph: The dependency graph
            
        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph.nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def find_critical_path(self, graph: DependencyGraph) -> List[str]:
        """
        Identifies critical path through dependencies.
        
        The critical path is the longest path through the dependency graph,
        representing the minimum time to complete all sub-problems.
        
        Args:
            graph: The dependency graph
            
        Returns:
            List of node IDs in critical path
        """
        # Calculate earliest start times
        earliest_start = {}
        in_degree = {node: 0 for node in graph.nodes}
        
        # Build reverse edges for traversal
        reverse_edges = defaultdict(list)
        for node, deps in graph.edges.items():
            for dep in deps:
                reverse_edges[dep].append(node)
                in_degree[node] += 1
        
        # Find nodes with no dependencies (start nodes)
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        
        # Initialize earliest start times
        for node in queue:
            earliest_start[node] = 0
        
        # Calculate earliest start times using topological sort
        while queue:
            current = queue.popleft()
            current_time = earliest_start[current]
            current_effort = graph.nodes[current].estimated_effort
            
            for successor in reverse_edges[current]:
                # Update earliest start time for successor
                new_time = current_time + current_effort
                if successor not in earliest_start or new_time > earliest_start[successor]:
                    earliest_start[successor] = new_time
                
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Find the node with maximum earliest start time (end of critical path)
        if not earliest_start:
            return []
        
        end_node = max(earliest_start, key=earliest_start.get)
        
        # Backtrack to find critical path
        critical_path = [end_node]
        current = end_node
        
        while earliest_start[current] > 0:
            # Find predecessor that contributed to earliest start time
            current_start = earliest_start[current]
            found_predecessor = False
            
            for dep in graph.edges.get(current, []):
                dep_end_time = earliest_start[dep] + graph.nodes[dep].estimated_effort
                if dep_end_time == current_start:
                    critical_path.insert(0, dep)
                    current = dep
                    found_predecessor = True
                    break
            
            if not found_predecessor:
                break
        
        return critical_path
    
    def identify_parallel_opportunities(self, graph: DependencyGraph) -> List[List[str]]:
        """
        Finds sub-problems that can be solved concurrently.
        
        Args:
            graph: The dependency graph
            
        Returns:
            List of parallel groups (each group is a list of node IDs that can run in parallel)
        """
        # Group nodes by their level in the dependency graph
        levels = self._calculate_levels(graph)
        
        # Group nodes at the same level (they can run in parallel)
        parallel_groups = []
        level_groups = defaultdict(list)
        
        for node, level in levels.items():
            level_groups[level].append(node)
        
        # Convert to list of groups (only include groups with multiple nodes)
        for level in sorted(level_groups.keys()):
            group = level_groups[level]
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def calculate_execution_order(self, graph: DependencyGraph) -> List[str]:
        """
        Determines optimal execution sequence using topological sort.
        
        Args:
            graph: The dependency graph
            
        Returns:
            List of node IDs in execution order
        """
        # Calculate in-degree for each node
        in_degree = {node: len(deps) for node, deps in graph.edges.items()}
        
        # Ensure all nodes are in in_degree dict
        for node in graph.nodes:
            if node not in in_degree:
                in_degree[node] = 0
        
        # Find nodes with no dependencies
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        execution_order = []
        
        # Build reverse edges
        reverse_edges = defaultdict(list)
        for node, deps in graph.edges.items():
            for dep in deps:
                reverse_edges[dep].append(node)
        
        # Process nodes in topological order
        while queue:
            # Sort queue by priority (higher priority first)
            queue = deque(sorted(queue, key=lambda n: graph.nodes[n].priority, reverse=True))
            current = queue.popleft()
            execution_order.append(current)
            
            # Update in-degree for successors
            for successor in reverse_edges[current]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        return execution_order
    
    def validate_dependencies(self, graph: DependencyGraph) -> ValidationResult:
        """
        Ensures dependency graph is valid and acyclic.
        
        Args:
            graph: The dependency graph to validate
            
        Returns:
            ValidationResult with validation status and feedback
        """
        from datetime import datetime
        
        errors = []
        warnings = []
        
        # Check for cycles
        cycles = self.detect_cycles(graph)
        if cycles:
            errors.append(f"Found {len(cycles)} circular dependencies")
            for i, cycle in enumerate(cycles[:3]):  # Show first 3 cycles
                errors.append(f"Cycle {i+1}: {' -> '.join(cycle)}")
        
        # Check for invalid dependencies (references to non-existent nodes)
        for node, deps in graph.edges.items():
            for dep in deps:
                if dep not in graph.nodes:
                    errors.append(f"Node {node} depends on non-existent node {dep}")
        
        # Check for isolated nodes (no dependencies and no dependents)
        reverse_edges = defaultdict(list)
        for node, deps in graph.edges.items():
            for dep in deps:
                reverse_edges[dep].append(node)
        
        for node in graph.nodes:
            if not graph.edges.get(node, []) and not reverse_edges.get(node, []):
                warnings.append(f"Node {node} is isolated (no dependencies or dependents)")
        
        # Calculate validation score
        score = 1.0
        if errors:
            score = 0.0
        elif warnings:
            score = 0.8
        
        passed = len(errors) == 0
        
        feedback = "Dependency graph validation: "
        if passed:
            feedback += "PASSED"
            if warnings:
                feedback += f" with {len(warnings)} warnings"
        else:
            feedback += f"FAILED with {len(errors)} errors"
        
        return ValidationResult(
            validator="dependency_manager",
            passed=passed,
            score=score,
            feedback=feedback,
            improvements=errors + warnings,
            timestamp=datetime.now()
        )
    
    def _calculate_levels(self, graph: DependencyGraph) -> Dict[str, int]:
        """Calculate the level of each node in the dependency graph."""
        levels = {}
        in_degree = {node: len(deps) for node, deps in graph.edges.items()}
        
        # Ensure all nodes are in in_degree dict
        for node in graph.nodes:
            if node not in in_degree:
                in_degree[node] = 0
        
        # Build reverse edges
        reverse_edges = defaultdict(list)
        for node, deps in graph.edges.items():
            for dep in deps:
                reverse_edges[dep].append(node)
        
        # Start with nodes that have no dependencies (level 0)
        queue = deque([(node, 0) for node, degree in in_degree.items() if degree == 0])
        
        while queue:
            current, level = queue.popleft()
            levels[current] = level
            
            # Process successors
            for successor in reverse_edges[current]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append((successor, level + 1))
        
        return levels
    
    def optimize_execution_order(self, graph: DependencyGraph) -> List[str]:
        """
        Optimizes execution order considering priorities and parallelization.
        
        Args:
            graph: The dependency graph
            
        Returns:
            Optimized execution order
        """
        # Start with basic topological sort
        execution_order = self.calculate_execution_order(graph)
        
        # Within each level, prioritize by:
        # 1. Priority (higher first)
        # 2. Estimated effort (shorter first for quick wins)
        levels = self._calculate_levels(graph)
        level_groups = defaultdict(list)
        
        for node in execution_order:
            level = levels.get(node, 0)
            level_groups[level].append(node)
        
        # Re-sort within each level
        optimized_order = []
        for level in sorted(level_groups.keys()):
            group = level_groups[level]
            # Sort by priority (desc) then effort (asc)
            sorted_group = sorted(
                group,
                key=lambda n: (-graph.nodes[n].priority, graph.nodes[n].estimated_effort)
            )
            optimized_order.extend(sorted_group)
        
        return optimized_order
