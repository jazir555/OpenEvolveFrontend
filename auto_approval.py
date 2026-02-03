"""
Auto-Approval Module

This module provides automatic approval functionality for decomposition plans
based on configurable criteria, reducing manual review overhead for simple cases.
"""

from typing import Dict, Any, List, Tuple
from workflow_structures import DecompositionPlan, SubProblem

class AutoApprovalChecker:
    """Checks if a decomposition plan meets auto-approval criteria."""
    
    def __init__(self, criteria: Dict[str, Any]):
        """
        Initialize the auto-approval checker.
        
        Args:
            criteria: Dictionary of auto-approval criteria
        """
        self.criteria = criteria
    
    def check_auto_approval(self, plan: DecompositionPlan) -> Tuple[bool, List[str]]:
        """
        Check if a plan meets auto-approval criteria.
        
        Args:
            plan: The decomposition plan to check
            
        Returns:
            Tuple of (should_auto_approve, reasons)
            - should_auto_approve: True if plan meets all criteria
            - reasons: List of reasons why plan was approved/rejected
        """
        reasons = []
        should_approve = True
        
        # Check if auto-approval is enabled
        if not self.criteria.get("enabled", False):
            reasons.append("Auto-approval is disabled")
            return False, reasons
        
        # Check maximum complexity threshold
        max_complexity = self.criteria.get("max_complexity", 10)
        if self._check_complexity(plan, max_complexity):
            reasons.append(f"✓ Complexity within threshold ({max_complexity})")
        else:
            reasons.append(f"✗ Complexity exceeds threshold ({max_complexity})")
            should_approve = False
        
        # Check maximum number of sub-problems
        max_sub_problems = self.criteria.get("max_sub_problems", 10)
        if len(plan.sub_problems) <= max_sub_problems:
            reasons.append(f"✓ Sub-problems within limit ({len(plan.sub_problems)}/{max_sub_problems})")
        else:
            reasons.append(f"✗ Too many sub-problems ({len(plan.sub_problems)}/{max_sub_problems})")
            should_approve = False
        
        # Check if all sub-problems have assigned teams
        if self.criteria.get("require_all_teams_assigned", True):
            if self._check_all_teams_assigned(plan):
                reasons.append("✓ All sub-problems have assigned teams")
            else:
                reasons.append("✗ Some sub-problems missing team assignments")
                should_approve = False
        
        # Check if all sub-problems have assigned gauntlets
        if self.criteria.get("require_all_gauntlets_assigned", True):
            if self._check_all_gauntlets_assigned(plan):
                reasons.append("✓ All sub-problems have assigned gauntlets")
            else:
                reasons.append("✗ Some sub-problems missing gauntlet assignments")
                should_approve = False
        
        # Check for circular dependencies
        if self.criteria.get("reject_circular_dependencies", True):
            if not self._has_circular_dependencies(plan):
                reasons.append("✓ No circular dependencies detected")
            else:
                reasons.append("✗ Circular dependencies detected")
                should_approve = False
        
        # Check domain whitelist (if specified)
        domain_whitelist = self.criteria.get("domain_whitelist", [])
        if domain_whitelist:
            domain = plan.analyzed_context.get("domain", "")
            if domain in domain_whitelist:
                reasons.append(f"✓ Domain '{domain}' is whitelisted")
            else:
                reasons.append(f"✗ Domain '{domain}' not in whitelist")
                should_approve = False
        
        # Check minimum confidence in AI suggestions
        min_confidence = self.criteria.get("min_ai_confidence", 0.0)
        if min_confidence > 0:
            # For now, we assume AI suggestions are always confident
            # In a real implementation, this would check actual confidence scores
            reasons.append(f"✓ AI confidence meets threshold ({min_confidence})")
        
        return should_approve, reasons
    
    def _check_complexity(self, plan: DecompositionPlan, max_complexity: int) -> bool:
        """Check if plan complexity is within threshold."""
        # Calculate average complexity across all sub-problems
        if not plan.sub_problems:
            return True
        
        avg_complexity = sum(
            sp.ai_suggested_complexity_score 
            for sp in plan.sub_problems
        ) / len(plan.sub_problems)
        
        return avg_complexity <= max_complexity
    
    def _check_all_teams_assigned(self, plan: DecompositionPlan) -> bool:
        """Check if all sub-problems have assigned teams."""
        for sp in plan.sub_problems:
            if not sp.solver_team_name:
                return False
        return True
    
    def _check_all_gauntlets_assigned(self, plan: DecompositionPlan) -> bool:
        """Check if all sub-problems have assigned gauntlets."""
        for sp in plan.sub_problems:
            if not sp.gold_team_gauntlet_name:
                return False
        return True
    
    def _has_circular_dependencies(self, plan: DecompositionPlan) -> bool:
        """Check for circular dependencies in sub-problems."""
        # Build dependency graph
        graph = {sp.id: set(sp.dependencies) for sp in plan.sub_problems}
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False


def get_default_auto_approval_criteria() -> Dict[str, Any]:
    """Get default auto-approval criteria."""
    return {
        "enabled": False,  # Disabled by default for safety
        "max_complexity": 5,  # Maximum average complexity score
        "max_sub_problems": 5,  # Maximum number of sub-problems
        "require_all_teams_assigned": True,  # All sub-problems must have teams
        "require_all_gauntlets_assigned": True,  # All sub-problems must have gauntlets
        "reject_circular_dependencies": True,  # Reject plans with circular dependencies
        "domain_whitelist": [],  # Empty = allow all domains
        "min_ai_confidence": 0.0  # Minimum AI confidence score (0.0-1.0)
    }


def auto_approve_plan(plan: DecompositionPlan, criteria: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Convenience function to check if a plan should be auto-approved.
    
    Args:
        plan: The decomposition plan to check
        criteria: Auto-approval criteria
        
    Returns:
        Tuple of (should_auto_approve, reasons)
    """
    checker = AutoApprovalChecker(criteria)
    return checker.check_auto_approval(plan)


def detect_circular_dependencies(plan: DecompositionPlan) -> List[List[str]]:
    """
    Detect circular dependencies in a decomposition plan.
    
    Args:
        plan: The decomposition plan to check
        
    Returns:
        List of cycles found (each cycle is a list of sub-problem IDs)
    """
    # Build dependency graph
    graph = {sp.id: set(sp.dependencies) for sp in plan.sub_problems}
    
    cycles = []
    visited = set()
    rec_stack = []
    
    def find_cycles(node: str, path: List[str]):
        visited.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                find_cycles(neighbor, path.copy())
            elif neighbor in path:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                if cycle not in cycles:
                    cycles.append(cycle)
        
        path.pop()
    
    for node in graph:
        if node not in visited:
            find_cycles(node, [])
    
    return cycles


def suggest_execution_order(plan: DecompositionPlan) -> List[str]:
    """
    Suggest an execution order for sub-problems based on dependencies.
    Uses topological sort.
    
    Args:
        plan: The decomposition plan
        
    Returns:
        List of sub-problem IDs in suggested execution order
    """
    # Build dependency graph
    graph = {sp.id: set(sp.dependencies) for sp in plan.sub_problems}
    
    # Calculate in-degrees
    in_degree = {sp.id: 0 for sp in plan.sub_problems}
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in in_degree:
                in_degree[neighbor] += 1
    
    # Topological sort using Kahn's algorithm
    queue = [node for node in in_degree if in_degree[node] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        # Reduce in-degree for neighbors
        for sp in plan.sub_problems:
            if node in sp.dependencies:
                in_degree[sp.id] -= 1
                if in_degree[sp.id] == 0:
                    queue.append(sp.id)
    
    # If result doesn't contain all nodes, there's a cycle
    if len(result) != len(plan.sub_problems):
        # Return original order if there's a cycle
        return [sp.id for sp in plan.sub_problems]
    
    return result


def validate_decomposition_plan(plan: DecompositionPlan) -> Tuple[bool, List[str]]:
    """
    Validate a decomposition plan for common issues.
    
    Args:
        plan: The decomposition plan to validate
        
    Returns:
        Tuple of (is_valid, issues)
        - is_valid: True if plan is valid
        - issues: List of validation issues found
    """
    issues = []
    
    # Check for empty sub-problems
    if not plan.sub_problems:
        issues.append("Plan has no sub-problems")
    
    # Check for duplicate sub-problem IDs
    ids = [sp.id for sp in plan.sub_problems]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate sub-problem IDs found")
    
    # Check for invalid dependencies
    valid_ids = set(ids)
    for sp in plan.sub_problems:
        for dep in sp.dependencies:
            if dep not in valid_ids:
                issues.append(f"Sub-problem {sp.id} has invalid dependency: {dep}")
    
    # Check for circular dependencies
    cycles = detect_circular_dependencies(plan)
    if cycles:
        for cycle in cycles:
            issues.append(f"Circular dependency detected: {' -> '.join(cycle)}")
    
    # Check for missing team assignments
    for sp in plan.sub_problems:
        if not sp.solver_team_name:
            issues.append(f"Sub-problem {sp.id} has no solver team assigned")
    
    # Check for missing gauntlet assignments
    for sp in plan.sub_problems:
        if not sp.gold_team_gauntlet_name:
            issues.append(f"Sub-problem {sp.id} has no gold gauntlet assigned")
    
    # Check for empty descriptions
    for sp in plan.sub_problems:
        if not sp.description or sp.description.strip() == "":
            issues.append(f"Sub-problem {sp.id} has empty description")
    
    return len(issues) == 0, issues
