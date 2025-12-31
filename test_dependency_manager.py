"""Tests for Dependency Manager"""

import pytest
from dependency_manager import DependencyManager
from sovereign_data_models import SubProblem, SubProblemType, ComplexityScore, DependencyGraph, generate_id


@pytest.fixture
def manager():
    return DependencyManager()


@pytest.fixture
def linear_subproblems():
    """Create sub-problems with linear dependencies."""
    sp1 = SubProblem(
        id="sp1", parent_id="p1", title="Task 1", description="First",
        type=SubProblemType.ANALYSIS, complexity_score=ComplexityScore(3,3,3,3,3,""),
        dependencies=[], success_criteria=[], validation_gauntlet="test", estimated_effort=8, priority=5
    )
    sp2 = SubProblem(
        id="sp2", parent_id="p1", title="Task 2", description="Second",
        type=SubProblemType.IMPLEMENTATION, complexity_score=ComplexityScore(3,3,3,3,3,""),
        dependencies=["sp1"], success_criteria=[], validation_gauntlet="test", estimated_effort=16, priority=5
    )
    sp3 = SubProblem(
        id="sp3", parent_id="p1", title="Task 3", description="Third",
        type=SubProblemType.VALIDATION, complexity_score=ComplexityScore(3,3,3,3,3,""),
        dependencies=["sp2"], success_criteria=[], validation_gauntlet="test", estimated_effort=8, priority=5
    )
    return [sp1, sp2, sp3]


@pytest.fixture
def parallel_subproblems():
    """Create sub-problems with parallel opportunities."""
    sp1 = SubProblem(
        id="sp1", parent_id="p1", title="Task 1", description="First",
        type=SubProblemType.ANALYSIS, complexity_score=ComplexityScore(3,3,3,3,3,""),
        dependencies=[], success_criteria=[], validation_gauntlet="test", estimated_effort=8, priority=5
    )
    sp2 = SubProblem(
        id="sp2", parent_id="p1", title="Task 2", description="Second",
        type=SubProblemType.IMPLEMENTATION, complexity_score=ComplexityScore(3,3,3,3,3,""),
        dependencies=["sp1"], success_criteria=[], validation_gauntlet="test", estimated_effort=16, priority=5
    )
    sp3 = SubProblem(
        id="sp3", parent_id="p1", title="Task 3", description="Third",
        type=SubProblemType.IMPLEMENTATION, complexity_score=ComplexityScore(3,3,3,3,3,""),
        dependencies=["sp1"], success_criteria=[], validation_gauntlet="test", estimated_effort=12, priority=5
    )
    return [sp1, sp2, sp3]


class TestDependencyManager:
    def test_build_graph(self, manager, linear_subproblems):
        graph = manager.build_graph(linear_subproblems)
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 3
        assert len(graph.execution_order) == 3
    
    def test_detect_no_cycles(self, manager, linear_subproblems):
        graph = manager.build_graph(linear_subproblems)
        cycles = manager.detect_cycles(graph)
        
        assert len(cycles) == 0
    
    def test_detect_cycles(self, manager):
        # Create circular dependency
        sp1 = SubProblem(
            id="sp1", parent_id="p1", title="Task 1", description="First",
            type=SubProblemType.ANALYSIS, complexity_score=ComplexityScore(3,3,3,3,3,""),
            dependencies=["sp2"], success_criteria=[], validation_gauntlet="test"
        )
        sp2 = SubProblem(
            id="sp2", parent_id="p1", title="Task 2", description="Second",
            type=SubProblemType.IMPLEMENTATION, complexity_score=ComplexityScore(3,3,3,3,3,""),
            dependencies=["sp1"], success_criteria=[], validation_gauntlet="test"
        )
        
        graph = DependencyGraph(
            nodes={"sp1": sp1, "sp2": sp2},
            edges={"sp1": ["sp2"], "sp2": ["sp1"]}
        )
        
        cycles = manager.detect_cycles(graph)
        assert len(cycles) > 0
    
    def test_find_critical_path(self, manager, linear_subproblems):
        graph = manager.build_graph(linear_subproblems)
        critical_path = manager.find_critical_path(graph)
        
        assert len(critical_path) > 0
        assert critical_path[0] == "sp1"
    
    def test_identify_parallel_opportunities(self, manager, parallel_subproblems):
        graph = manager.build_graph(parallel_subproblems)
        parallel_groups = manager.identify_parallel_opportunities(graph)
        
        # sp2 and sp3 both depend only on sp1, so they can run in parallel
        assert len(parallel_groups) > 0
        assert any(len(group) >= 2 for group in parallel_groups)
    
    def test_calculate_execution_order(self, manager, linear_subproblems):
        graph = manager.build_graph(linear_subproblems)
        order = manager.calculate_execution_order(graph)
        
        assert len(order) == 3
        assert order[0] == "sp1"
        assert order.index("sp1") < order.index("sp2")
        assert order.index("sp2") < order.index("sp3")
    
    def test_validate_valid_graph(self, manager, linear_subproblems):
        graph = manager.build_graph(linear_subproblems)
        result = manager.validate_dependencies(graph)
        
        assert result.passed is True
        assert result.score > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
