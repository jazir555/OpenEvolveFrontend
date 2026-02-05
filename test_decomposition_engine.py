"""
Unit tests for Decomposition Engine
"""

import pytest
from decomposition_engine import (
    DecompositionEngine, SemanticDecomposition,
    DependencyDecomposition, ComplexityDecomposition
)
from problem_analyzer import ProblemAnalyzer
from sovereign_data_models import ProblemType, DomainContext, ComplexityScore


@pytest.fixture
def analyzer():
    return ProblemAnalyzer()


@pytest.fixture
def engine():
    return DecompositionEngine()


@pytest.fixture
def sample_problem(analyzer):
    return analyzer.analyze_problem(
        "Build a scalable web application with user authentication and data analytics",
        title="Web Application"
    )


class TestSemanticDecomposition:
    def test_research_decomposition(self, analyzer):
        problem = analyzer.analyze_problem(
            "Research machine learning approaches for natural language processing",
            title="ML Research"
        )
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(problem)
        
        # Verify we get actual sub-problems (not empty)
        assert len(sub_problems) >= 3
        assert len(sub_problems) <= 6
        # Check for research-related content
        titles = [sp.title.lower() for sp in sub_problems]
        assert any("research" in t or "literature" in t or "analysis" in t for t in titles)
    
    def test_implementation_decomposition(self, sample_problem):
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(sample_problem)
        
        # Verify we get actual sub-problems (not empty)
        assert len(sub_problems) >= 3
        assert len(sub_problems) <= 6
        # Check for implementation-related content
        titles = [sp.title.lower() for sp in sub_problems]
        assert any("requirements" in t or "design" in t or "implementation" in t or "analysis" in t for t in titles)


class TestDependencyDecomposition:
    def test_creates_dependencies(self, sample_problem):
        strategy = DependencyDecomposition()
        sub_problems = strategy.decompose(sample_problem)
        
        # Verify we get actual sub-problems (not empty)
        assert len(sub_problems) >= 2
        # Check that at least one sub-problem has dependencies set
        has_dependencies = any(len(sp.dependencies) > 0 for sp in sub_problems)
        assert has_dependencies, "Expected at least one sub-problem to have dependencies"


class TestComplexityDecomposition:
    def test_splits_complex_problems(self, analyzer):
        problem = analyzer.analyze_problem(
            "Design and implement a highly scalable distributed system with real-time processing",
            title="Complex System"
        )
        # Set high complexity to trigger splitting
        problem.complexity_score.overall_complexity = 9.0
        problem.complexity_score.cognitive_complexity = 9.0
        problem.complexity_score.computational_complexity = 9.0
        
        strategy = ComplexityDecomposition()
        sub_problems = strategy.decompose(problem)
        
        # Verify we get actual sub-problems (not empty)
        assert len(sub_problems) >= 3
        # All sub-problems should have complexity scores
        for sp in sub_problems:
            assert sp.complexity_score is not None
            assert sp.complexity_score.overall_complexity > 0


class TestDecompositionEngine:
    def test_decompose_with_auto_strategy(self, engine, sample_problem):
        plan = engine.decompose(sample_problem)
        
        assert plan is not None
        assert len(plan.sub_problems) > 0, "Decomposition should return sub-problems, not empty list"
        assert plan.dependency_graph is not None
    
    def test_strategy_selection(self, engine, analyzer):
        # High complexity should select hybrid or complexity strategy
        complex_problem = analyzer.analyze_problem(
            "Build a distributed real-time system with fault tolerance",
            title="Complex"
        )
        complex_problem.complexity_score.overall_complexity = 8.5
        strategy = engine.select_strategy(complex_problem)
        # Should return a valid strategy name
        assert strategy in ['semantic', 'dependency', 'complexity', 'hybrid', 'research']
        
        # Lower complexity should select appropriate strategy
        constrained_problem = analyzer.analyze_problem(
            "Build system within budget and timeline with quality requirements",
            title="Constrained"
        )
        strategy = engine.select_strategy(constrained_problem)
        assert strategy in ['semantic', 'dependency', 'complexity', 'hybrid', 'research']
    
    def test_execution_order(self, engine, sample_problem):
        plan = engine.decompose(sample_problem)
        
        # Verify execution_order attribute exists on dependency_graph
        assert hasattr(plan.dependency_graph, 'execution_order')
        assert len(plan.dependency_graph.execution_order) == len(plan.sub_problems)
    
    def test_subproblem_has_execution_order_attribute(self, engine, sample_problem):
        """Test that SubProblem has execution_order attribute."""
        plan = engine.decompose(sample_problem)
        
        assert len(plan.sub_problems) > 0
        for sp in plan.sub_problems:
            # Verify execution_order attribute exists
            assert hasattr(sp, 'execution_order')
            # Verify it's an integer
            assert isinstance(sp.execution_order, int)
    
    def test_decomposition_returns_non_empty_subproblems(self, engine, analyzer):
        """Critical test: Decomposition must return actual sub-problems, not empty list."""
        problem = analyzer.analyze_problem(
            "Build a web application with user authentication",
            title="Web App"
        )
        plan = engine.decompose(problem)
        
        # CRITICAL: Should NOT return empty list
        assert len(plan.sub_problems) > 0, "CRITICAL: Decomposition returned empty sub-problems list"
        # Should return reasonable number of sub-problems
        assert len(plan.sub_problems) >= 2, "Should create at least 2 sub-problems for decomposition"
        assert len(plan.sub_problems) <= 10, "Should not create excessive sub-problems"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
