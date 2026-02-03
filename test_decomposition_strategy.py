"""
Unit Tests for decomposition_strategy.py

Comprehensive test suite covering:
- Strategy selection logic
- HYBRID decomposition
- ROMA decomposition
- SEMANTIC decomposition
- Edge cases
- Error handling
- Integration with sovereign_data_models
"""

import unittest
from datetime import datetime, timezone
from typing import List, Optional

# Import the module to test
from decomposition_strategy import (
    SovereignDecompositionStrategy,
    HybridDecompositionStrategy,
    RomadecompositionStrategy,
    SemanticDecompositionStrategy,
    StrategySelector,
    DecompositionStrategyExecutor,
    ComplexityScore,
    DependencyGraph,
    decompose_hybrid,
    decompose_roma,
    decompose_semantic,
    select_strategy,
    execute_strategy,
)

# Import data models
try:
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        ProblemStatus,
        generate_id
    )
except ImportError:
    # Use fallback definitions from decomposition_strategy
    from decomposition_strategy import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        ProblemStatus,
        generate_id
    )


class TestDataModels(unittest.TestCase):
    """Test data model initialization and validation."""

    def test_complexity_score_validation(self):
        """Test ComplexityScore validation."""
        # Valid scores
        score = ComplexityScore(
            explanation="Test",
            cognitive_complexity=5.0,
            computational_complexity=6.0,
            domain_complexity=7.0,
            integration_complexity=8.0,
            overall_complexity=6.5
        )
        self.assertEqual(score.overall_complexity, 6.5)

        # Invalid scores should raise ValueError
        with self.assertRaises(ValueError):
            ComplexityScore(
                explanation="Test",
                cognitive_complexity=15.0,  # Invalid: > 10.0
                computational_complexity=6.0,
                domain_complexity=7.0,
                integration_complexity=8.0,
                overall_complexity=6.5
            )

    def test_dependency_graph(self):
        """Test DependencyGraph functionality."""
        graph = DependencyGraph()
        graph.nodes = {'a': None, 'b': None, 'c': None}

        # Add edges
        graph.add_edge('a', 'b')
        graph.add_edge('a', 'c')
        graph.add_edge('b', 'c')

        # Check edges
        self.assertIn('b', graph.edges['a'])
        self.assertIn('c', graph.edges['a'])
        self.assertIn('c', graph.edges['b'])

        # Check execution order
        order = graph.get_execution_order()
        self.assertIn('a', order)
        self.assertIn('b', order)
        self.assertIn('c', order)

        # 'a' should come before 'b' and 'c'
        self.assertLess(order.index('a'), order.index('b'))
        self.assertLess(order.index('a'), order.index('c'))

    def test_problem_definition_creation(self):
        """Test ProblemDefinition creation."""
        problem = ProblemDefinition(
            problem_id="test_001",
            title="Test Problem",
            description="Test description",
            domain="software_engineering",
            complexity="moderate",
            priority="high",
            estimated_effort="medium",
            requirements=["req1", "req2"],
            constraints=["constraint1"],
            created_at=datetime.now(timezone.utc)
        )

        self.assertEqual(problem.title, "Test Problem")
        self.assertEqual(len(problem.requirements), 2)
        self.assertEqual(problem.domain, "software_engineering")


class TestHybridStrategy(unittest.TestCase):
    """Test HYBRID decomposition strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = HybridDecompositionStrategy()
        self.sample_problem = ProblemDefinition(
            problem_id="hybrid_test_001",
            title="Build Web Application",
            description="Design and implement a web application with user authentication, "
                       "database integration, and REST API. The application should be "
                       "scalable and secure.",
            domain="software_engineering",
            complexity="complex",
            priority="high",
            estimated_effort="large",
            requirements=[
                "User authentication",
                "Database integration",
                "REST API",
                "Scalability",
                "Security"
            ],
            constraints=[
                "Must use Python",
                "Budget: $5000",
                "Timeline: 3 months"
            ],
            created_at=datetime.now(timezone.utc)
        )

    def test_strategy_name(self):
        """Test strategy name."""
        self.assertEqual(
            self.strategy.get_strategy_name(),
            SovereignDecompositionStrategy.HYBRID.value
        )

    def test_decompose_basic(self):
        """Test basic decomposition."""
        plan = self.strategy.decompose(self.sample_problem, depth=2)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertIsNotNone(plan.plan_id)
        self.assertGreater(len(plan.sub_problems), 0)
        self.assertEqual(plan.problem.problem_id, self.sample_problem.problem_id)

    def test_decompose_creates_sub_problems(self):
        """Test that decomposition creates sub-problems."""
        plan = self.strategy.decompose(self.sample_problem, depth=2)

        self.assertGreater(len(plan.sub_problems), 1)
        self.assertLess(len(plan.sub_problems), 15)  # Max subproblems limit

        # Check sub-problem structure
        for sp in plan.sub_problems:
            self.assertIsInstance(sp, SubProblem)
            self.assertIsNotNone(sp.sub_problem_id)
            self.assertIsNotNone(sp.title)
            self.assertIsNotNone(sp.description)
            self.assertEqual(sp.status, ProblemStatus.PENDING)

    def test_decompose_identifies_phases(self):
        """Test phase identification."""
        phases = self.strategy._identify_phases(self.sample_problem)

        self.assertIsInstance(phases, list)
        self.assertGreater(len(phases), 0)

    def test_decompose_identifies_components(self):
        """Test component identification."""
        components = self.strategy._identify_components(self.sample_problem)

        self.assertIsInstance(components, list)

    def test_decompose_identifies_aspects(self):
        """Test aspect identification."""
        aspects = self.strategy._identify_aspects(self.sample_problem)

        self.assertIsInstance(aspects, list)

    def test_invalid_problem_raises_error(self):
        """Test that invalid problem raises ValueError."""
        invalid_problem = ProblemDefinition(
            problem_id="invalid",
            title="",  # Empty title
            description="",  # Empty description
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=[],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        with self.assertRaises(ValueError):
            self.strategy.decompose(invalid_problem)


class TestRomaStrategy(unittest.TestCase):
    """Test ROMA decomposition strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = RomadecompositionStrategy()
        self.sample_problem = ProblemDefinition(
            problem_id="roma_test_001",
            title="Design System Architecture",
            description="Design a comprehensive system architecture with multiple layers. "
                       "Include data layer, business logic layer, API layer, and presentation layer. "
                       "Each layer should be independent and communicate through interfaces.",
            domain="software_engineering",
            complexity="moderate",
            priority="medium",
            estimated_effort="medium",
            requirements=[
                "Data layer",
                "Business logic layer",
                "API layer",
                "Presentation layer"
            ],
            constraints=["Use layered architecture"],
            created_at=datetime.now(timezone.utc)
        )

    def test_strategy_name(self):
        """Test strategy name."""
        self.assertEqual(
            self.strategy.get_strategy_name(),
            SovereignDecompositionStrategy.ROMA.value
        )

    def test_decompose_basic(self):
        """Test basic decomposition."""
        plan = self.strategy.decompose(self.sample_problem, max_depth=3)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertIsNotNone(plan.plan_id)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_decompose_hierarchical_structure(self):
        """Test hierarchical decomposition structure."""
        plan = self.strategy.decompose(self.sample_problem, max_depth=2)

        # Should have parent-child relationships
        root_problems = [sp for sp in plan.sub_problems if sp.parent_id is None]
        child_problems = [sp for sp in plan.sub_problems if sp.parent_id is not None]

        self.assertGreater(len(root_problems), 0)
        # At depth 2, should have children
        self.assertGreater(len(plan.sub_problems), len(root_problems))

    def test_atomic_detection(self):
        """Test atomic problem detection."""
        atomic_problem = ProblemDefinition(
            problem_id="atomic",
            title="Simple Function",
            description="Create a simple function.",
            domain="software_engineering",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=["Write function"],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        is_atomic = self.strategy._is_atomic(atomic_problem)
        self.assertTrue(is_atomic)

    def test_breadth_first_order(self):
        """Test breadth-first execution order."""
        plan = self.strategy.decompose(self.sample_problem, max_depth=2)

        # Should have execution order
        self.assertGreater(len(plan.execution_order), 0)

        # All sub-problems should be in execution order
        sp_ids = {sp.sub_problem_id for sp in plan.sub_problems}
        order_ids = set(plan.execution_order)
        self.assertEqual(sp_ids, order_ids)


class TestSemanticStrategy(unittest.TestCase):
    """Test SEMANTIC decomposition strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SemanticDecompositionStrategy()
        self.sample_problem = ProblemDefinition(
            problem_id="semantic_test_001",
            title="Machine Learning Model Development",
            description="Develop a machine learning model for predictive analytics. "
                       "Include data preprocessing, feature engineering, model training, "
                       "evaluation, and deployment. Focus on accuracy and performance.",
            domain="data_science",
            complexity="complex",
            priority="high",
            estimated_effort="large",
            requirements=[
                "Data preprocessing",
                "Feature engineering",
                "Model training",
                "Model evaluation",
                "Model deployment"
            ],
            constraints=["Use Python", "Accuracy > 90%"],
            created_at=datetime.now(timezone.utc)
        )

    def test_strategy_name(self):
        """Test strategy name."""
        self.assertEqual(
            self.strategy.get_strategy_name(),
            SovereignDecompositionStrategy.SEMANTIC.value
        )

    def test_decompose_basic(self):
        """Test basic decomposition."""
        plan = self.strategy.decompose(self.sample_problem, clusters=5)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertIsNotNone(plan.plan_id)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_concept_extraction(self):
        """Test concept extraction."""
        concepts = self.strategy._extract_concepts(self.sample_problem)

        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)

        # Each concept should have required fields
        for concept in concepts[:5]:
            self.assertIn('id', concept)
            self.assertIn('word', concept)
            self.assertIn('frequency', concept)
            self.assertIn('relevance', concept)

    def test_concept_clustering(self):
        """Test concept clustering."""
        concepts = self.strategy._extract_concepts(self.sample_problem)
        clusters = self.strategy._cluster_concepts(concepts, num_clusters=3)

        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0)
        self.assertLessEqual(len(clusters), 3)

    def test_semantic_dependencies(self):
        """Test semantic dependency identification."""
        plan = self.strategy.decompose(self.sample_problem, clusters=3)

        # Should have dependencies dictionary
        self.assertIsInstance(plan.dependencies, dict)


class TestStrategySelector(unittest.TestCase):
    """Test strategy selection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = StrategySelector()

    def test_select_simple_problem(self):
        """Test selection for simple problem."""
        simple_problem = ProblemDefinition(
            problem_id="simple",
            title="Simple Task",
            description="A simple task",
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=["one requirement"],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        strategy = self.selector.select_strategy(simple_problem)
        self.assertIsInstance(strategy, SovereignDecompositionStrategy)

    def test_select_complex_problem(self):
        """Test selection for complex problem."""
        complex_problem = ProblemDefinition(
            problem_id="complex",
            title="Complex System",
            description="A very complex system with many components and requirements. " * 10,
            domain="software_engineering",
            complexity="complex",
            priority="high",
            estimated_effort="large",
            requirements=[f"Requirement {i}" for i in range(10)],
            constraints=[f"Constraint {i}" for i in range(5)],
            created_at=datetime.now(timezone.utc)
        )

        strategy = self.selector.select_strategy(complex_problem)
        self.assertIsInstance(strategy, SovereignDecompositionStrategy)
        # Complex problems often get HYBRID
        # (This is implementation-dependent, so we just check it returns a valid strategy)

    def test_score_strategy(self):
        """Test strategy scoring."""
        problem = ProblemDefinition(
            problem_id="score_test",
            title="Test",
            description="Test description",
            domain="software_engineering",
            complexity="moderate",
            priority="medium",
            estimated_effort="medium",
            requirements=["req1"],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        for strategy in SovereignDecompositionStrategy:
            score = self.selector._score_strategy(problem, strategy)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestDecompositionExecutor(unittest.TestCase):
    """Test the main decomposition executor."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = DecompositionStrategyExecutor()
        self.sample_problem = ProblemDefinition(
            problem_id="executor_test_001",
            title="Integration Test Problem",
            description="A problem for testing the executor.",
            domain="general",
            complexity="moderate",
            priority="medium",
            estimated_effort="medium",
            requirements=["req1", "req2"],
            constraints=["constraint1"],
            created_at=datetime.now(timezone.utc)
        )

    def test_execute_hybrid_strategy(self):
        """Test executing HYBRID strategy."""
        plan = self.executor.execute_strategy(
            "HYBRID",
            self.sample_problem,
            depth=2
        )

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_execute_roma_strategy(self):
        """Test executing ROMA strategy."""
        plan = self.executor.execute_strategy(
            "ROMA",
            self.sample_problem,
            max_depth=2
        )

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_execute_semantic_strategy(self):
        """Test executing SEMANTIC strategy."""
        plan = self.executor.execute_strategy(
            "SEMANTIC",
            self.sample_problem,
            clusters=3
        )

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_execute_invalid_strategy(self):
        """Test that invalid strategy name raises ValueError."""
        with self.assertRaises(ValueError):
            self.executor.execute_strategy(
                "INVALID_STRATEGY",
                self.sample_problem
            )

    def test_execute_with_auto_selection(self):
        """Test automatic strategy selection and execution."""
        plan = self.executor.execute_with_auto_selection(self.sample_problem)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_plan_validation(self):
        """Test plan validation."""
        # Valid plan
        plan = self.executor.execute_strategy(
            "HYBRID",
            self.sample_problem,
            depth=2
        )
        self.assertTrue(self.executor._validate_plan(plan))

        # Invalid plan (no sub-problems)
        invalid_plan = DecompositionPlan(
            plan_id="invalid",
            problem=self.sample_problem,
            sub_problems=[],
            dependencies={},
            execution_order=[],
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
            status=ProblemStatus.PENDING
        )
        self.assertFalse(self.executor._validate_plan(invalid_plan))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_problem = ProblemDefinition(
            problem_id="conv_test_001",
            title="Convenience Test",
            description="Test convenience functions.",
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=["req1"],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

    def test_decompose_hybrid(self):
        """Test decompose_hybrid convenience function."""
        plan = decompose_hybrid(self.sample_problem, depth=2)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_decompose_roma(self):
        """Test decompose_roma convenience function."""
        plan = decompose_roma(self.sample_problem, max_depth=2)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_decompose_semantic(self):
        """Test decompose_semantic convenience function."""
        plan = decompose_semantic(self.sample_problem, clusters=3)

        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)

    def test_select_strategy(self):
        """Test select_strategy convenience function."""
        strategy = select_strategy(self.sample_problem)

        self.assertIsInstance(strategy, SovereignDecompositionStrategy)

    def test_execute_strategy(self):
        """Test execute_strategy convenience function."""
        plan = execute_strategy(
            "HYBRID",
            self.sample_problem,
            depth=2
        )

        self.assertIsInstance(plan, DecompositionPlan)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = DecompositionStrategyExecutor()

    def test_empty_description(self):
        """Test problem with minimal description."""
        problem = ProblemDefinition(
            problem_id="empty_desc",
            title="Minimal Problem",
            description="",
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=[],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        # Should handle gracefully (might raise error or return minimal plan)
        try:
            plan = decompose_hybrid(problem, depth=1)
            # If it doesn't raise, should still have valid structure
            self.assertIsInstance(plan, DecompositionPlan)
        except (ValueError, RuntimeError):
            # Expected behavior for invalid input
            pass

    def test_very_long_description(self):
        """Test problem with very long description."""
        long_desc = "This is a very long description. " * 100

        problem = ProblemDefinition(
            problem_id="long_desc",
            title="Long Description Problem",
            description=long_desc,
            domain="software_engineering",
            complexity="complex",
            priority="high",
            estimated_effort="large",
            requirements=[f"Requirement {i}" for i in range(20)],
            constraints=[f"Constraint {i}" for i in range(10)],
            created_at=datetime.now(timezone.utc)
        )

        plan = decompose_hybrid(problem, depth=2)
        self.assertIsInstance(plan, DecompositionPlan)

    def test_no_requirements(self):
        """Test problem with no requirements."""
        problem = ProblemDefinition(
            problem_id="no_reqs",
            title="No Requirements",
            description="A problem without requirements.",
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=[],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        plan = decompose_semantic(problem, clusters=2)
        self.assertIsInstance(plan, DecompositionPlan)

    def test_many_constraints(self):
        """Test problem with many constraints."""
        problem = ProblemDefinition(
            problem_id="many_constraints",
            title="Many Constraints",
            description="A problem with many constraints.",
            domain="software_engineering",
            complexity="complex",
            priority="high",
            estimated_effort="large",
            requirements=["req1"],
            constraints=[f"Constraint {i}" for i in range(15)],
            created_at=datetime.now(timezone.utc)
        )

        plan = decompose_hybrid(problem, depth=2)
        self.assertIsInstance(plan, DecompositionPlan)


class TestIntegrationWithSovereignModels(unittest.TestCase):
    """Test integration with sovereign_data_models."""

    def test_sub_problem_structure(self):
        """Test that SubProblems match sovereign_data_models structure."""
        problem = ProblemDefinition(
            problem_id="integration_test",
            title="Integration Test",
            description="Test integration with sovereign models.",
            domain="general",
            complexity="moderate",
            priority="medium",
            estimated_effort="medium",
            requirements=["req1"],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        plan = decompose_hybrid(problem, depth=2)

        for sp in plan.sub_problems:
            # Check required fields exist
            self.assertTrue(hasattr(sp, 'sub_problem_id'))
            self.assertTrue(hasattr(sp, 'parent_id'))
            self.assertTrue(hasattr(sp, 'title'))
            self.assertTrue(hasattr(sp, 'description'))
            self.assertTrue(hasattr(sp, 'status'))
            self.assertTrue(hasattr(sp, 'confidence'))
            self.assertTrue(hasattr(sp, 'created_at'))

    def test_decomposition_plan_structure(self):
        """Test that DecompositionPlan matches sovereign_data_models structure."""
        problem = ProblemDefinition(
            problem_id="plan_test",
            title="Plan Structure Test",
            description="Test plan structure.",
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=["req1"],
            constraints=[],
            created_at=datetime.now(timezone.utc)
        )

        plan = decompose_roma(problem, max_depth=2)

        # Check required fields
        self.assertTrue(hasattr(plan, 'plan_id'))
        self.assertTrue(hasattr(plan, 'problem'))
        self.assertTrue(hasattr(plan, 'sub_problems'))
        self.assertTrue(hasattr(plan, 'dependencies'))
        self.assertTrue(hasattr(plan, 'execution_order'))
        self.assertTrue(hasattr(plan, 'created_at'))
        self.assertTrue(hasattr(plan, 'modified_at'))
        self.assertTrue(hasattr(plan, 'status'))


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataModels))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestRomaStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategySelector))
    suite.addTests(loader.loadTestsFromTestCase(TestDecompositionExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithSovereignModels))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
