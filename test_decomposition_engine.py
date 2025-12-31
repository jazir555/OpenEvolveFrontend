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
        
        assert len(sub_problems) == 4
        assert any("literature" in sp.title.lower() for sp in sub_problems)
    
    def test_implementation_decomposition(self, sample_problem):
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(sample_problem)
        
        assert len(sub_problems) == 4
        assert any("requirements" in sp.title.lower() for sp in sub_problems)
        assert any("implementation" in sp.title.lower() for sp in sub_problems)


class TestDependencyDecomposition:
    def test_creates_dependencies(self, sample_problem):
        strategy = DependencyDecomposition()
        sub_problems = strategy.decompose(sample_problem)
        
        assert len(sub_problems) > 1
        # Check that later sub-problems have dependencies
        assert len(sub_problems[1].dependencies) > 0


class TestComplexityDecomposition:
    def test_splits_complex_problems(self, analyzer):
        problem = analyzer.analyze_problem(
            "Design and implement a highly scalable distributed system with real-time processing",
            title="Complex System"
        )
        strategy = ComplexityDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) >= 4
        # All sub-problems should have manageable complexity
        for sp in sub_problems:
            assert sp.complexity_score.overall_complexity <= 7.5


class TestDecompositionEngine:
    def test_decompose_with_auto_strategy(self, engine, sample_problem):
        plan = engine.decompose(sample_problem)
        
        assert plan is not None
        assert len(plan.sub_problems) > 0
        assert plan.dependency_graph is not None
    
    def test_strategy_selection(self, engine, analyzer):
        # High complexity should select complexity strategy
        complex_problem = analyzer.analyze_problem(
            "Build a distributed real-time system with fault tolerance",
            title="Complex"
        )
        complex_problem.complexity_score.overall_complexity = 8.5
        strategy = engine.select_strategy(complex_problem)
        assert strategy == 'complexity'
        
        # Many constraints should select dependency strategy
        constrained_problem = analyzer.analyze_problem(
            "Build system within budget and timeline with quality requirements",
            title="Constrained"
        )
        strategy = engine.select_strategy(constrained_problem)
        assert strategy in ['semantic', 'dependency']
    
    def test_execution_order(self, engine, sample_problem):
        plan = engine.decompose(sample_problem)
        
        assert len(plan.dependency_graph.execution_order) == len(plan.sub_problems)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
