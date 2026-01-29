"""
Tests for OpenEvolve Integration with Enhanced Decomposition/Recomposition

This test suite covers:
1. OpenEvolveSolutionSolver
2. ParallelEvolutionManager
3. OpenEvolveIntegratedPipeline
4. OpenEvolveDecompositionAdapter
5. Integration with existing OpenEvolve infrastructure
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import time

# Import enhanced systems
from enhanced_decomposition_engine import (
    EnhancedDecompositionEngine,
    ProblemDefinition,
    SubProblem,
    DecompositionStrategy,
    ProblemDomain,
    SubProblemType,
    ComplexityScore,
    create_problem_definition
)

from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    SubProblemSolution,
    create_subproblem_solution
)

# Import OpenEvolve integration
from openevolve_enhanced_decomposition_integration import (
    OpenEvolveSolutionSolver,
    ParallelEvolutionManager,
    OpenEvolveIntegratedPipeline,
    EvolutionConfig,
    SubProblemEvolutionResult,
    quick_solve_with_openevolve,
    compare_strategies_with_openevolve
)

from openevolve_decomposition_adapter import (
    OpenEvolveDecompositionAdapter,
    OpenEvolveDecompositionAPI,
    integrate_with_existing_openevolve,
    create_decomposition_aware_config,
    convert_openevolve_result_to_solution,
    DecompositionMetricsCollector
)


# ============================================================================
# MOCK OPENEVOLVE CLIENT
# ============================================================================

class MockOpenEvolveClient:
    """Mock OpenEvolve client for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def evolve(self, content, **kwargs):
        """Mock evolve method."""
        self.call_count += 1
        
        # Create mock result
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = f"# Evolved solution\n{content[:100]}..."
        mock_result.best_score = 0.8 + (self.call_count * 0.02)
        mock_result.iterations_completed = kwargs.get('max_iterations', 50)
        
        return mock_result


# ============================================================================
# OPENEVOLVE SOLUTION SOLVER TESTS
# ============================================================================

class TestOpenEvolveSolutionSolver(unittest.TestCase):
    """Test cases for OpenEvolveSolutionSolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockOpenEvolveClient()
        self.config = EvolutionConfig(max_iterations=30)
        self.solver = OpenEvolveSolutionSolver(
            openevolve_client=self.mock_client,
            evolution_config=self.config
        )
        
        self.sample_subproblem = SubProblem(
            id="sub_1",
            parent_id="prob_1",
            title="Implement Authentication",
            description="Create user authentication system with JWT tokens",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=4.0,
                integration_complexity=5.0,
                coordination_complexity=3.0,
                technical_complexity=6.0,
                overall_complexity=5.5
            ),
            priority=8,
            estimated_effort_hours=16
        )
    
    def test_can_solve_implementation(self):
        """Test solver can handle implementation problems."""
        can_solve, confidence = self.solver.can_solve(self.sample_subproblem)
        
        self.assertTrue(can_solve)
        self.assertGreater(confidence, 0.8)
    
    def test_can_solve_research(self):
        """Test solver can handle research problems."""
        research_sp = SubProblem(
            id="sub_2",
            parent_id="prob_1",
            title="Research Approach",
            description="Research different approaches",
            type=SubProblemType.RESEARCH,
            complexity_score=ComplexityScore(5.0, 4.0, 4.0, 3.0, 2.0, 4.0, 4.0),
            priority=5,
            estimated_effort_hours=8
        )
        
        can_solve, confidence = self.solver.can_solve(research_sp)
        
        self.assertTrue(can_solve)
        self.assertGreater(confidence, 0.6)
    
    def test_solve_success(self):
        """Test successful solution generation."""
        solution = self.solver.solve(self.sample_subproblem)
        
        self.assertIsInstance(solution, SubProblemSolution)
        self.assertEqual(solution.sub_problem_id, self.sample_subproblem.id)
        self.assertGreater(solution.quality_score, 0.5)
        self.assertGreater(len(solution.solution_content), 0)
    
    def test_solve_without_client(self):
        """Test solver without OpenEvolve client."""
        solver_no_client = OpenEvolveSolutionSolver(openevolve_client=None)
        
        solution = solver_no_client.solve(self.sample_subproblem)
        
        self.assertIsInstance(solution, SubProblemSolution)
        self.assertEqual(solution.verification_status, "fallback")
    
    def test_evolution_prompt_creation(self):
        """Test evolution prompt creation."""
        prompt = self.solver._create_evolution_prompt(self.sample_subproblem)
        
        self.assertIn(self.sample_subproblem.title, prompt)
        self.assertIn(self.sample_subproblem.description, prompt)
        self.assertIn(self.sample_subproblem.type.value, prompt)
    
    def test_default_evaluator(self):
        """Test default evaluator function."""
        evaluator = self.solver._create_default_evaluator(self.sample_subproblem)
        
        # Test with good content
        good_content = """
# Overview
This is a comprehensive solution.

## Approach
The approach is well-designed.

## Implementation
```python
def authenticate():
    pass
```
"""
        score = evaluator(good_content)
        self.assertGreater(score, 0.5)
        
        # Test with poor content
        poor_content = "Short"
        score = evaluator(poor_content)
        self.assertLess(score, 0.5)
    
    def test_evolution_history_tracking(self):
        """Test evolution history is tracked."""
        initial_count = len(self.solver.evolution_history)
        
        self.solver.solve(self.sample_subproblem)
        
        self.assertEqual(len(self.solver.evolution_history), initial_count + 1)
        
        entry = self.solver.evolution_history[-1]
        self.assertEqual(entry['sub_problem_id'], self.sample_subproblem.id)
        self.assertIn('fitness', entry)


# ============================================================================
# PARALLEL EVOLUTION MANAGER TESTS
# ============================================================================

class TestParallelEvolutionManager(unittest.TestCase):
    """Test cases for ParallelEvolutionManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockOpenEvolveClient()
        self.solver = OpenEvolveSolutionSolver(openevolve_client=self.mock_client)
        self.manager = ParallelEvolutionManager(
            solver=self.solver,
            max_workers=2
        )
        
        self.sub_problems = [
            SubProblem(
                id=f"sub_{i}",
                parent_id="prob_1",
                title=f"Task {i}",
                description=f"Description {i}",
                type=SubProblemType.IMPLEMENTATION,
                complexity_score=ComplexityScore(5.0, 4.0, 4.0, 3.0, 2.0, 4.0, 4.0),
                priority=5,
                estimated_effort_hours=8,
                dependencies=[] if i == 0 else [f"sub_{i-1}"]
            )
            for i in range(3)
        ]
    
    def test_group_by_dependency_level(self):
        """Test grouping sub-problems by dependency level."""
        dependency_graph = {
            "sub_0": [],
            "sub_1": ["sub_0"],
            "sub_2": ["sub_1"]
        }
        
        levels = self.manager._group_by_dependency_level(
            self.sub_problems,
            dependency_graph
        )
        
        self.assertEqual(len(levels), 3)  # Three levels for linear dependencies
        self.assertEqual(len(levels[0]), 1)  # First level has 1
        self.assertEqual(levels[0][0].id, "sub_0")
    
    def test_evolve_all_sequential(self):
        """Test evolving sub-problems sequentially due to dependencies."""
        dependency_graph = {
            "sub_0": [],
            "sub_1": ["sub_0"],
            "sub_2": ["sub_1"]
        }
        
        solutions = self.manager.evolve_all(
            self.sub_problems,
            dependency_graph
        )
        
        self.assertEqual(len(solutions), 3)
        for sp_id in ["sub_0", "sub_1", "sub_2"]:
            self.assertIn(sp_id, solutions)
    
    def test_evolve_all_parallel(self):
        """Test evolving independent sub-problems in parallel."""
        # Make all independent
        for sp in self.sub_problems:
            sp.dependencies = []
        
        dependency_graph = {"sub_0": [], "sub_1": [], "sub_2": []}
        
        solutions = self.manager.evolve_all(
            self.sub_problems,
            dependency_graph
        )
        
        self.assertEqual(len(solutions), 3)


# ============================================================================
# OPENEVOLVE INTEGRATED PIPELINE TESTS
# ============================================================================

class TestOpenEvolveIntegratedPipeline(unittest.TestCase):
    """Test cases for OpenEvolveIntegratedPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockOpenEvolveClient()
        
        self.pipeline = OpenEvolveIntegratedPipeline(
            openevolve_client=self.mock_client,
            evolution_config=EvolutionConfig(max_iterations=20)
        )
        
        self.problem = create_problem_definition(
            title="Build API Service",
            description="Create RESTful API with authentication and database",
            domain=ProblemDomain.SOFTWARE,
            complexity=6.0
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.decomposition_engine)
        self.assertIsNotNone(self.pipeline.recomposition_engine)
        self.assertIsNotNone(self.pipeline.solver)
        self.assertIsNotNone(self.pipeline.parallel_manager)
    
    def test_execute_full_pipeline(self):
        """Test full pipeline execution."""
        result = self.pipeline.execute(self.problem)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.decomposition_plan)
        self.assertIsNotNone(result.integrated_solution)
        self.assertGreater(len(result.sub_solutions), 0)
        
        self.assertGreaterEqual(result.overall_quality, 0.0)
        self.assertLessEqual(result.overall_quality, 1.0)
    
    def test_metrics_collection(self):
        """Test that metrics are collected."""
        initial_count = len(self.pipeline.metrics_history)
        
        self.pipeline.execute(self.problem)
        
        self.assertEqual(len(self.pipeline.metrics_history), initial_count + 1)
        
        metrics = self.pipeline.metrics_history[-1]
        self.assertGreater(metrics.decomposition_time, 0)
        self.assertGreater(metrics.evolution_time, 0)
        self.assertGreater(metrics.total_time, 0)
    
    def test_sequential_vs_parallel(self):
        """Test both sequential and parallel execution."""
        # Sequential
        result_seq = self.pipeline.execute(self.problem, use_parallel_evolution=False)
        
        # Parallel
        result_par = self.pipeline.execute(self.problem, use_parallel_evolution=True)
        
        # Both should complete successfully
        self.assertTrue(result_seq.is_successful() or result_seq.overall_quality > 0)
        self.assertTrue(result_par.is_successful() or result_par.overall_quality > 0)


# ============================================================================
# ADAPTER TESTS
# ============================================================================

class TestOpenEvolveDecompositionAdapter(unittest.TestCase):
    """Test cases for OpenEvolveDecompositionAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = OpenEvolveDecompositionAdapter()
        
        # Replace with mock
        self.mock_client = MockOpenEvolveClient()
        self.adapter.pipeline.openevolve_client = self.mock_client
        self.adapter.pipeline.solver.openevolve_client = self.mock_client
        self.adapter.solver = self.adapter.pipeline.solver
    
    def test_decompose_and_evolve(self):
        """Test full decompose and evolve flow."""
        result = self.adapter.decompose_and_evolve(
            problem_description="Build a web application with authentication",
            problem_title="Web App",
            domain="software",
            complexity=6.0
        )
        
        self.assertIn('success', result)
        self.assertIn('decomposition', result)
        self.assertIn('solutions', result)
        self.assertIn('integrated_solution', result)
        
        if result['success']:
            self.assertGreater(result['overall_quality'], 0)
    
    def test_parse_domain(self):
        """Test domain parsing."""
        domains = [
            ('software', ProblemDomain.SOFTWARE),
            ('finance', ProblemDomain.FINANCE),
            ('healthcare', ProblemDomain.HEALTHCARE),
            ('unknown', ProblemDomain.GENERIC),
        ]
        
        for domain_str, expected in domains:
            result = self.adapter._parse_domain(domain_str)
            self.assertEqual(result, expected)
    
    def test_evolve_sub_problem(self):
        """Test evolving single sub-problem."""
        sub_problem = SubProblem(
            id="test_sub",
            parent_id="test_prob",
            title="Test Implementation",
            description="Test description",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(5.0, 4.0, 4.0, 3.0, 2.0, 4.0, 4.0),
            priority=5,
            estimated_effort_hours=8
        )
        
        solution = self.adapter.evolve_sub_problem(sub_problem)
        
        self.assertIsInstance(solution, SubProblemSolution)
        self.assertEqual(solution.sub_problem_id, sub_problem.id)


class TestOpenEvolveDecompositionAPI(unittest.TestCase):
    """Test cases for OpenEvolveDecompositionAPI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = OpenEvolveDecompositionAPI(
            base_url="http://localhost:8000",
            api_key="test_key",
            enable_decomposition=True
        )
    
    def test_parse_strategy(self):
        """Test strategy parsing."""
        strategies = [
            ('hierarchical', DecompositionStrategy.HIERARCHICAL),
            ('functional', DecompositionStrategy.FUNCTIONAL),
            ('semantic', DecompositionStrategy.SEMANTIC),
            ('hybrid', DecompositionStrategy.HYBRID),
            ('unknown', DecompositionStrategy.HYBRID),  # Default
        ]
        
        for strategy_str, expected in strategies:
            result = self.api._parse_strategy(strategy_str)
            self.assertEqual(result, expected)
    
    def test_get_decomposition_status(self):
        """Test getting decomposition status."""
        status = self.api.get_decomposition_status("test_evolution_123")
        
        self.assertIsNotNone(status)
        self.assertIn('evolution_id', status)
        self.assertIn('status', status)
    
    def test_get_decomposed_solution(self):
        """Test getting decomposed solution."""
        solution = self.api.get_decomposed_solution("test_evolution_123")
        
        self.assertIsNotNone(solution)
        self.assertIn('evolution_id', solution)
        self.assertIn('quality_score', solution)


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_decomposition_aware_config(self):
        """Test creating decomposition-aware config."""
        base_config = {'max_iterations': 100}
        
        result = create_decomposition_aware_config(
            base_config=base_config,
            decomposition_strategy='semantic',
            enable_parallel_evolution=True,
            max_subproblems=8
        )
        
        self.assertIn('decomposition', result)
        self.assertIn('recomposition', result)
        self.assertEqual(result['decomposition']['strategy'], 'semantic')
        self.assertEqual(result['decomposition']['max_subproblems'], 8)
    
    def test_convert_openevolve_result(self):
        """Test converting OpenEvolve result."""
        mock_result = Mock()
        mock_result.best_code = "def solution(): pass"
        mock_result.best_score = 0.85
        mock_result.iterations_completed = 25
        
        solution = convert_openevolve_result_to_solution(
            mock_result,
            "sub_123"
        )
        
        self.assertIsInstance(solution, SubProblemSolution)
        self.assertEqual(solution.sub_problem_id, "sub_123")
        self.assertEqual(solution.quality_score, 0.85)
    
    def test_quick_solve_with_openevolve(self):
        """Test quick solve function."""
        from decomposition_recomposition_integration import PipelineResult
        
        result = quick_solve_with_openevolve(
            title="Quick Test",
            description="Test problem",
            domain=ProblemDomain.SOFTWARE,
            complexity=5.0
        )
        
        self.assertIsInstance(result, PipelineResult)


# ============================================================================
# METRICS COLLECTOR TESTS
# ============================================================================

class TestDecompositionMetricsCollector(unittest.TestCase):
    """Test cases for DecompositionMetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = DecompositionMetricsCollector()
    
    def test_collect_decomposition_metrics(self):
        """Test collecting decomposition metrics."""
        plan = DecompositionPlan(
            id="plan_123",
            original_problem=create_problem_definition("Test", "Test"),
            sub_problems=[],
            strategy_used=DecompositionStrategy.HYBRID,
            overall_quality=0.85
        )
        
        metrics = self.collector.collect_decomposition_metrics(plan, 1.5)
        
        self.assertEqual(metrics['operation'], 'decomposition')
        self.assertEqual(metrics['duration'], 1.5)
        self.assertEqual(metrics['quality'], 0.85)
        self.assertEqual(len(self.collector.metrics), 1)
    
    def test_collect_evolution_metrics(self):
        """Test collecting evolution metrics."""
        metrics = self.collector.collect_evolution_metrics(
            "sub_1",
            0.82,
            25,
            3.2
        )
        
        self.assertEqual(metrics['operation'], 'evolution')
        self.assertEqual(metrics['sub_problem_id'], 'sub_1')
        self.assertEqual(metrics['fitness'], 0.82)
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        # Add some metrics
        plan = DecompositionPlan(
            id="plan_1",
            original_problem=create_problem_definition("Test", "Test"),
            sub_problems=[],
            strategy_used=DecompositionStrategy.HYBRID,
            overall_quality=0.8
        )
        
        self.collector.collect_decomposition_metrics(plan, 1.0)
        self.collector.collect_evolution_metrics("sub_1", 0.8, 20, 2.0)
        self.collector.collect_evolution_metrics("sub_2", 0.85, 25, 2.5)
        
        summary = self.collector.get_summary()
        
        self.assertEqual(summary['total_operations'], 3)
        self.assertEqual(summary['decompositions'], 1)
        self.assertEqual(summary['evolutions'], 2)
        self.assertGreater(summary['avg_fitness'], 0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullIntegration(unittest.TestCase):
    """Full integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create pipeline
        mock_client = MockOpenEvolveClient()
        pipeline = OpenEvolveIntegratedPipeline(openevolve_client=mock_client)
        
        # Define problem
        problem = create_problem_definition(
            title="E-Commerce Platform",
            description="""
            Build an e-commerce platform with product catalog,
            shopping cart, payment processing, and order management.
            """,
            domain=ProblemDomain.SOFTWARE,
            complexity=7.5
        )
        
        # Execute
        result = pipeline.execute(problem)
        
        # Verify
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.decomposition_plan)
        self.assertIsNotNone(result.integrated_solution)
        self.assertGreater(len(result.sub_solutions), 0)
        
        # Quality checks
        self.assertGreaterEqual(result.decomposition_quality, 0.0)
        self.assertGreaterEqual(result.solution_quality, 0.0)
        self.assertGreaterEqual(result.overall_quality, 0.0)
    
    def test_compare_strategies(self):
        """Test strategy comparison."""
        problem = create_problem_definition(
            title="Test Problem",
            description="Test description",
            complexity=5.0
        )
        
        result = compare_strategies_with_openevolve(
            problem,
            strategies=[
                DecompositionStrategy.HIERARCHICAL,
                DecompositionStrategy.FUNCTIONAL
            ]
        )
        
        self.assertIn('problem', result)
        self.assertIn('results', result)
        self.assertIn('best_strategy', result)
        self.assertEqual(len(result['results']), 2)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
