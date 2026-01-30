#!/usr/bin/env python3
"""
Test suite to verify that all decomposition engine fixes are working correctly.
This tests the specific gaps that were identified and fixed.
"""

import unittest
import sys
import logging
from unittest.mock import Mock, patch

# Set up logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports
try:
    from decomposition_engine import DecompositionEngine, TEAM_ASSIGNMENT_AVAILABLE
    from sovereign_data_models import ProblemDefinition, SubProblem, DecompositionPlan
    from team_assignment_engine import TeamAssignmentEngine
    TEAM_ASSIGNMENT_IMPORT_SUCCESS = True
except ImportError as e:
    logger.error(f"Import failed: {e}")
    TEAM_ASSIGNMENT_IMPORT_SUCCESS = False

class TestDecompositionEngineFixes(unittest.TestCase):
    """Test that all decomposition engine fixes are working correctly."""

    def test_team_assignment_engine_import(self):
        """Test that TeamAssignmentEngine import is working."""
        self.assertTrue(TEAM_ASSIGNMENT_IMPORT_SUCCESS, "TeamAssignmentEngine should be importable")
        self.assertTrue(TEAM_ASSIGNMENT_AVAILABLE, "TEAM_ASSIGNMENT_AVAILABLE should be True")

    def test_decomposition_engine_initialization(self):
        """Test that DecompositionEngine can be initialized without errors."""
        try:
            engine = DecompositionEngine()
            self.assertIsNotNone(engine, "DecompositionEngine should initialize successfully")
            self.assertIsNotNone(engine.strategies, "Engine should have strategies")
            self.assertGreater(len(engine.strategies), 0, "Engine should have at least one strategy")
        except (RuntimeError, ValueError, ImportError) as e:
            self.fail(f"DecompositionEngine initialization failed: {e}")

    def test_team_assignment_without_engine(self):
        """Test team assignment when no TeamAssignmentEngine is provided."""
        engine = DecompositionEngine()
        
        # Create a mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_001"
        problem.title = "Test Problem"
        problem.description = "Test description"
        
        # Test decomposition without team assignment
        try:
            plan = engine.decompose(problem, assign_teams=False)
            self.assertIsNotNone(plan, "Decomposition should work without team assignment")
            self.assertIsInstance(plan, DecompositionPlan, "Result should be a DecompositionPlan")
        except (RuntimeError, ValueError, TypeError) as e:
            self.fail(f"Decomposition without team assignment failed: {e}")

    def test_team_assignment_without_teams(self):
        """Test team assignment when no teams are provided."""
        engine = DecompositionEngine()
        
        # Create a mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_002"
        problem.title = "Test Problem"
        problem.description = "Test description"
        
        # Test decomposition with team assignment but no teams
        try:
            plan = engine.decompose(problem, assign_teams=True, teams=None)
            self.assertIsNotNone(plan, "Decomposition should work even when teams are None")
            self.assertIsInstance(plan, DecompositionPlan, "Result should be a DecompositionPlan")
        except (RuntimeError, ValueError, TypeError) as e:
            self.fail(f"Decomposition with team assignment but no teams failed: {e}")

    def test_custom_strategy_error_handling(self):
        """Test that custom strategy method has proper error handling."""
        engine = DecompositionEngine()
        
        # Create a mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_003"
        problem.title = "Test Problem"
        problem.description = "Test description"
        
        # Test with invalid strategy config
        invalid_config = {"invalid": "config"}
        
        with patch('decomposition_engine.CustomStrategyBuilder') as mock_builder:
            mock_instance = mock_builder.return_value
            mock_instance.create_strategy.side_effect = ValueError("Invalid config")
            
            with self.assertRaises(RuntimeError):
                engine.use_custom_strategy(problem, invalid_config)

    def test_strategy_selection_fallback(self):
        """Test that strategy selection has proper fallback mechanisms."""
        engine = DecompositionEngine()
        
        # Create a mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_004"
        problem.title = "Test Problem"
        problem.description = "Test description"
        
        # Test strategy selection
        try:
            strategy = engine.select_strategy(problem)
            self.assertIsNotNone(strategy, "Strategy selection should return a strategy")
            self.assertIn(strategy, engine.strategies, "Selected strategy should be in available strategies")
        except (RuntimeError, ValueError, TypeError) as e:
            self.fail(f"Strategy selection failed: {e}")

    def test_error_handling_decorators(self):
        """Test that error handling decorators are properly applied."""
        engine = DecompositionEngine()
        
        # Check that key methods have error handling
        self.assertTrue(hasattr(engine.decompose, '__wrapped__'), "decompose method should have error handling")
        self.assertTrue(hasattr(engine.use_custom_strategy, '__wrapped__'), "use_custom_strategy method should have error handling")

    def test_resource_estimation_fallback(self):
        """Test that resource estimation has proper fallback."""
        engine = DecompositionEngine()
        
        # Create a mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_005"
        problem.title = "Test Problem"
        problem.description = "Test description"
        
        # Test decomposition (resource estimation should not fail even if engine is None)
        try:
            plan = engine.decompose(problem)
            self.assertIsNotNone(plan, "Decomposition should work even without resource estimation")
        except (RuntimeError, ValueError, TypeError) as e:
            self.fail(f"Decomposition with resource estimation fallback failed: {e}")

    def test_dependency_analysis_fallback(self):
        """Test that dependency analysis has proper fallback."""
        engine = DecompositionEngine()
        
        # Create a mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_006"
        problem.title = "Test Problem"
        problem.description = "Test description"
        
        # Test decomposition (dependency analysis should not fail even if analyzer is None)
        try:
            plan = engine.decompose(problem)
            self.assertIsNotNone(plan, "Decomposition should work even without dependency analysis")
            self.assertIsNotNone(plan.dependency_graph, "Plan should have a dependency graph")
        except (RuntimeError, ValueError, TypeError) as e:
            self.fail(f"Decomposition with dependency analysis fallback failed: {e}")

    def test_comprehensive_integration(self):
        """Test comprehensive integration of all fixes."""
        engine = DecompositionEngine()
        
        # Create a realistic mock problem
        problem = Mock(spec=ProblemDefinition)
        problem.id = "test_problem_integration"
        problem.title = "Integration Test Problem"
        problem.description = "A comprehensive problem for testing all decomposition features"
        problem.domain_context = Mock()
        problem.domain_context.domain = "Testing"
        problem.complexity_score = Mock()
        problem.complexity_score.overall_complexity = 7.5
        
        # Test all features together
        try:
            # Test basic decomposition
            plan = engine.decompose(problem)
            self.assertIsNotNone(plan, "Basic decomposition should work")
            
            # Test with team assignment (should not fail even without teams)
            plan_with_teams = engine.decompose(problem, assign_teams=True, teams=[])
            self.assertIsNotNone(plan_with_teams, "Decomposition with team assignment should work")
            
            # Test strategy selection
            strategy = engine.select_strategy(problem)
            self.assertIsNotNone(strategy, "Strategy selection should work")
            
            # Test custom strategy (should fail gracefully)
            with self.assertRaises((ImportError, RuntimeError, ValueError)):
                engine.use_custom_strategy(problem, {"test": "config"})
            
        except (RuntimeError, ValueError, TypeError) as e:
            self.fail(f"Comprehensive integration test failed: {e}")


def run_comprehensive_tests():
    """Run all decomposition engine fix tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDecompositionEngineFixes)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DECOMPOSITION ENGINE FIXES TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - All decomposition engine fixes are working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED - Please review the failures above.")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)