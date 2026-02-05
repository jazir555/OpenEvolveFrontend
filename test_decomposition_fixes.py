"""
Quick verification tests for Decomposition fixes.
Tests run without external LLM dependencies for speed.
"""

import pytest
from datetime import datetime

# Import directly from data models to avoid heavy imports
from sovereign_data_models import (
    ProblemDefinition, SubProblem, DomainContext, ComplexityScore,
    ProblemType, SubProblemType, generate_id
)
from decomposition_engine import (
    SemanticDecomposition, DependencyDecomposition, 
    ComplexityDecomposition, DecompositionEngine
)


def create_test_problem(problem_type=ProblemType.IMPLEMENTATION, description="Test problem"):
    """Create a simple test problem without heavy analysis."""
    return ProblemDefinition(
        id=generate_id("problem"),
        title="Test Problem",
        description=description,
        problem_type=problem_type,
        domain_context=DomainContext(domain="software", subdomain="backend"),
        complexity_score=ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=4.0,
            domain_complexity=5.0,
            integration_complexity=3.0,
            overall_complexity=5.0,
            explanation="Test complexity"
        )
    )


class TestSemanticDecompositionFixes:
    """Test that SemanticDecomposition returns actual sub-problems."""
    
    def test_returns_non_empty_subproblems(self):
        """CRITICAL: Decomposition must return actual sub-problems, not empty list."""
        problem = create_test_problem(
            ProblemType.RESEARCH,
            "Research machine learning approaches for NLP"
        )
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(problem)
        
        # CRITICAL FIX: Should NOT return empty list
        assert len(sub_problems) > 0, "CRITICAL: SemanticDecomposition returned empty list!"
        assert len(sub_problems) >= 3, f"Expected at least 3 sub-problems, got {len(sub_problems)}"
        print(f"[OK] SemanticDecomposition returned {len(sub_problems)} sub-problems")
    
    def test_research_problem_decomposition(self):
        """Test research problem creates appropriate sub-problems."""
        problem = create_test_problem(
            ProblemType.RESEARCH,
            "Research approaches to climate change mitigation"
        )
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) >= 3
        titles = [sp.title.lower() for sp in sub_problems]
        # Should have research-related titles
        assert any("research" in t or "literature" in t or "analysis" in t for t in titles)
        print(f"[OK] Research decomposition works: {titles}")
    
    def test_implementation_problem_decomposition(self):
        """Test implementation problem creates appropriate sub-problems."""
        problem = create_test_problem(
            ProblemType.IMPLEMENTATION,
            "Build a web application with authentication"
        )
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) >= 3
        titles = [sp.title.lower() for sp in sub_problems]
        # Should have implementation-related titles
        assert any("requirements" in t or "design" in t or "implementation" in t for t in titles)
        print(f"[OK] Implementation decomposition works: {titles}")


class TestDependencyDecompositionFixes:
    """Test that DependencyDecomposition creates proper dependencies."""
    
    def test_returns_non_empty_subproblems(self):
        """CRITICAL: DependencyDecomposition must return actual sub-problems."""
        problem = create_test_problem(
            ProblemType.IMPLEMENTATION,
            "Build system with components A, B, and C"
        )
        strategy = DependencyDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) > 0, "CRITICAL: DependencyDecomposition returned empty list!"
        print(f"[OK] DependencyDecomposition returned {len(sub_problems)} sub-problems")
    
    def test_creates_dependencies_between_subproblems(self):
        """Test that sub-problems have dependencies set."""
        problem = create_test_problem(
            ProblemType.IMPLEMENTATION,
            "Build system with multiple components"
        )
        strategy = DependencyDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) >= 2
        # At least one sub-problem should have dependencies
        has_dependencies = any(len(sp.dependencies) > 0 for sp in sub_problems)
        assert has_dependencies, "Expected at least one sub-problem to have dependencies"
        print(f"[OK] Dependencies created correctly")


class TestComplexityDecompositionFixes:
    """Test that ComplexityDecomposition handles complexity properly."""
    
    def test_returns_non_empty_subproblems(self):
        """CRITICAL: ComplexityDecomposition must return actual sub-problems."""
        problem = create_test_problem(
            ProblemType.IMPLEMENTATION,
            "Design distributed system"
        )
        # Set high complexity to trigger potential splitting
        problem.complexity_score.overall_complexity = 9.0
        
        strategy = ComplexityDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) > 0, "CRITICAL: ComplexityDecomposition returned empty list!"
        print(f"[OK] ComplexityDecomposition returned {len(sub_problems)} sub-problems")
    
    def test_complexity_scores_are_set(self):
        """Test that all sub-problems have valid complexity scores."""
        problem = create_test_problem(
            ProblemType.IMPLEMENTATION,
            "Complex system design"
        )
        problem.complexity_score.overall_complexity = 8.0
        
        strategy = ComplexityDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) > 0
        for sp in sub_problems:
            assert sp.complexity_score is not None
            assert sp.complexity_score.overall_complexity > 0
        print(f"[OK] All {len(sub_problems)} sub-problems have complexity scores")


class TestExecutionOrderAttribute:
    """Test that SubProblem has execution_order attribute."""
    
    def test_subproblem_has_execution_order(self):
        """CRITICAL: SubProblem must have execution_order attribute."""
        problem = create_test_problem()
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(problem)
        
        assert len(sub_problems) > 0
        for sp in sub_problems:
            # CRITICAL FIX: Verify execution_order attribute exists
            assert hasattr(sp, 'execution_order'), f"SubProblem missing execution_order attribute!"
            assert isinstance(sp.execution_order, int), f"execution_order should be int, got {type(sp.execution_order)}"
        print(f"[OK] All {len(sub_problems)} sub-problems have execution_order attribute")
    
    def test_set_execution_order_method(self):
        """Test that set_execution_order method works."""
        problem = create_test_problem()
        strategy = SemanticDecomposition()
        sub_problems = strategy.decompose(problem)
        
        # Test set_execution_order method
        for i, sp in enumerate(sub_problems):
            sp.set_execution_order(i + 1)
            assert sp.execution_order == i + 1
        print(f"[OK] set_execution_order method works correctly")


class TestDecompositionEngineIntegration:
    """Test DecompositionEngine integration."""
    
    def test_engine_returns_non_empty_plan(self):
        """CRITICAL: DecompositionEngine must return plan with sub-problems."""
        problem = create_test_problem(
            ProblemType.IMPLEMENTATION,
            "Build web app with auth and database"
        )
        engine = DecompositionEngine()
        plan = engine.decompose(problem)
        
        assert plan is not None
        assert len(plan.sub_problems) > 0, "CRITICAL: DecompositionEngine returned empty sub-problems!"
        assert len(plan.sub_problems) >= 2, f"Expected at least 2 sub-problems, got {len(plan.sub_problems)}"
        print(f"[OK] DecompositionEngine returned plan with {len(plan.sub_problems)} sub-problems")
    
    def test_strategy_selection_returns_valid_strategy(self):
        """Test strategy selection returns valid strategy name."""
        problem = create_test_problem()
        problem.complexity_score.overall_complexity = 8.5  # High complexity
        
        engine = DecompositionEngine()
        strategy = engine.select_strategy(problem)
        
        # Should return a valid strategy name (string)
        assert strategy in ['semantic', 'dependency', 'complexity', 'hybrid', 'research']
        assert isinstance(strategy, str)
        print(f"[OK] Strategy selection returned: {strategy}")
    
    def test_dependency_graph_has_execution_order(self):
        """Test that dependency graph has execution_order list."""
        problem = create_test_problem()
        engine = DecompositionEngine()
        plan = engine.decompose(problem)
        
        assert plan.dependency_graph is not None
        assert hasattr(plan.dependency_graph, 'execution_order')
        assert len(plan.dependency_graph.execution_order) == len(plan.sub_problems)
        print(f"[OK] Dependency graph has execution_order with {len(plan.dependency_graph.execution_order)} items")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("DECOMPOSITION FIXES VERIFICATION")
    print("=" * 60)
    
    # Run all test classes
    test_classes = [
        TestSemanticDecompositionFixes(),
        TestDependencyDecompositionFixes(),
        TestComplexityDecompositionFixes(),
        TestExecutionOrderAttribute(),
        TestDecompositionEngineIntegration(),
    ]
    
    all_passed = True
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        print("-" * 40)
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    getattr(test_class, method_name)()
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: FAILED - {e}")
                    all_passed = False
                except Exception as e:
                    print(f"  [ERROR] {method_name}: ERROR - {e}")
                    all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED [SUCCESS]")
        print("Decomposition is now at 100%!")
    else:
        print("SOME TESTS FAILED [FAIL]")
    print("=" * 60)
