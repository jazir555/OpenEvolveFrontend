"""
Sovereign-Grade Problem Decomposition System - Dependency Manager

Manages dependencies between sub-problems, validates graphs, and optimizes execution.
"""
from __future__ import annotations


import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime

from sovereign_data_models import SubProblem, DependencyGraph, ValidationResult, generate_id
from sovereign_reliability import with_error_handling, ErrorSeverity


logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Dependency Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# **ACTUAL INTEGRATION HELPER METHODS**: Dependency Manager
def _trigger_dependency_alerts(operation, success, graph_id=None, error=None, metadata=None):
    """Trigger alerts for dependency manager operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.HIGH if operation == "build_graph" else AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"Dependency {operation} Failed",
            message=f"Dependency manager operation '{operation}' failed: {error}",
            severity=severity,
            source="DependencyManager",
            metadata=metadata or {"graph_id": graph_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger dependency alert: {e}")


def _extract_dependency_knowledge(operation, graph_id, result):
    """Extract knowledge from dependency operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"dependency_{operation}_{graph_id}",
            artifact_type="dependency_execution",
            source_component="DependencyManager",
            content={
                "operation": operation,
                "graph_id": graph_id,
                "num_nodes": len(result.get("nodes", {})) if result else 0,
                "num_edges": sum(len(deps) for deps in result.get("edges", {}).values()) if result else 0,
                "has_cycles": len(result.get("cycles", [])) > 0 if result else False,
                "critical_path_length": len(result.get("critical_path", [])) if result else 0,
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract dependency knowledge: {e}")


def _track_dependency_performance(operation, success, duration_seconds, num_nodes, num_edges=0, has_cycles=False):
    """Track performance of dependency operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name="dependency_graph_analysis",
            component_name="DependencyManager",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "has_cycles": has_cycles
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track dependency performance: {e}")


class DependencyManager:
    """Manages dependencies between sub-problems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling(severity=ErrorSeverity.CRITICAL, fallback=lambda sub_problems: DependencyGraph(nodes={}, edges={}))
    def build_graph(self, sub_problems: List[SubProblem]) -> DependencyGraph:
        """
        Constructs dependency graph from sub-problems.

        Args:
            sub_problems: List of sub-problems

        Returns:
            DependencyGraph with nodes, edges, and analysis
        """
        import time
        start_time = time.time()
        success = False
        graph_id = f"dep_{hash(str(sub_problems)) % 10000:04d}"

        if not sub_problems:
            self.logger.warning("No sub-problems provided to build_graph. Returning an empty DependencyGraph.")
            return DependencyGraph(nodes={}, edges={})

        self.logger.info(f"Building dependency graph for {len(sub_problems)} sub-problems")

        try:
            nodes = {sp.id: sp for sp in sub_problems}
            edges = {sp.id: sp.dependencies for sp in sub_problems}
        except AttributeError as e:
            self.logger.error(f"Invalid sub-problem object found during graph building: {e}", exc_info=True)
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_dependency_alerts("build_graph", False, graph_id, str(e))
            _track_dependency_performance("build_graph", False, time.time() - start_time, 0)
            return DependencyGraph(nodes={}, edges={})
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while creating nodes and edges for the dependency graph: {e}", exc_info=True)
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_dependency_alerts("build_graph", False, graph_id, str(e))
            _track_dependency_performance("build_graph", False, time.time() - start_time, 0)
            return DependencyGraph(nodes={}, edges={})

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

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance
        success = True
        duration = time.time() - start_time
        result_dict = {
            "nodes": nodes,
            "edges": edges,
            "cycles": cycles,
            "critical_path": critical_path
        }
        _extract_dependency_knowledge("build_graph", graph_id, result_dict)
        _track_dependency_performance("build_graph", True, duration, len(nodes),
                                     sum(len(deps) for deps in edges.values()), len(cycles) > 0)

        return graph
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda graph: [])
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
        
        try:
            for node in graph.nodes:
                if node not in visited:
                    dfs(node)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during cycle detection: {e}", exc_info=True)
            return []
        
        return cycles
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda graph: [])
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
        try:
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
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during critical path calculation: {e}", exc_info=True)
            return []
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda graph: [])
    def identify_parallel_opportunities(self, graph: DependencyGraph) -> List[List[str]]:
        """
        Finds sub-problems that can be solved concurrently.
        
        Args:
            graph: The dependency graph
            
        Returns:
            List of parallel groups (each group is a list of node IDs that can run in parallel)
        """
        try:
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
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during parallel opportunities identification: {e}", exc_info=True)
            return []
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda graph: [])
    def calculate_execution_order(self, graph: DependencyGraph) -> List[str]:
        """
        Determines optimal execution sequence using topological sort.
        
        Args:
            graph: The dependency graph
            
        Returns:
            List of node IDs in execution order
        """
        try:
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
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during execution order calculation: {e}", exc_info=True)
            return []
    
    @with_error_handling(severity=ErrorSeverity.CRITICAL, fallback=lambda graph: ValidationResult(
        validator="dependency_manager", passed=False, score=0.0, feedback="Dependency validation failed due to unexpected error",
        improvements=["Review dependency graph for structural issues"], timestamp=datetime.now()
    ))
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
        try:
            for node, deps in graph.edges.items():
                for dep in deps:
                    if dep not in graph.nodes:
                        errors.append(f"Node {node} depends on non-existent node {dep}")
        except Exception as e:
            errors.append(f"Error checking for invalid dependencies: {e}")
            self.logger.error(f"Error checking for invalid dependencies: {e}", exc_info=True)
        
        # Check for isolated nodes (no dependencies and no dependents)
        try:
            reverse_edges = defaultdict(list)
            for node, deps in graph.edges.items():
                for dep in deps:
                    reverse_edges[dep].append(node)
            
            for node in graph.nodes:
                if not graph.edges.get(node, []) and not reverse_edges.get(node, []):
                    warnings.append(f"Node {node} is isolated (no dependencies or dependents)")
        except Exception as e:
            errors.append(f"Error checking for isolated nodes: {e}")
            self.logger.error(f"Error checking for isolated nodes: {e}", exc_info=True)
        
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
        try:
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
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during level calculation: {e}", exc_info=True)
            return {}
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda graph: [])
    def optimize_execution_order(self, graph: DependencyGraph) -> List[str]:
        """
        Optimizes execution order considering priorities and parallelization.
        
        Args:
            graph: The dependency graph
            
        Returns:
            Optimized execution order
        """
        try:
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
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during execution order optimization: {e}", exc_info=True)
            return []
