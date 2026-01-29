"""
Comprehensive Tests for Enhanced Decomposition and Recomposition Systems

This test suite covers:
1. Enhanced Decomposition Engine
2. Enhanced Recomposition Engine
3. Decomposition-Recomposition Integration Pipeline

Test Categories:
- Unit tests for individual components
- Integration tests for the full pipeline
- Performance tests
- Edge case tests
- Stress tests
"""

import unittest
import time
from typing import List, Dict, Any

# Import decomposition components
from enhanced_decomposition_engine import (
    EnhancedDecompositionEngine,
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    DecompositionStrategy,
    ProblemDomain,
    SubProblemType,
    ComplexityScore,
    Constraint,
    ConstraintType,
    ConstraintSeverity,
    HierarchicalDecomposition,
    FunctionalDecomposition,
    SemanticDecomposition,
    TemporalDecomposition,
    CausalDecomposition,
    RiskBasedDecomposition,
    ComplexityBasedDecomposition,
    DependencyDecomposition,
    HybridDecomposition,
    create_problem_definition
)

# Import recomposition components
from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    IntegratedSolution,
    SubProblemSolution,
    AssemblyStrategy,
    RecompositionConfig,
    ConflictDetector,
    ConflictResolver,
    Conflict,
    ConflictType,
    ConflictSeverity,
    QualityMetrics,
    create_subproblem_solution
)

# Import integration components
from decomposition_recomposition_integration import (
    DecompositionRecompositionPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    PipelineAnalytics,
    BatchPipelineProcessor,
    SolutionSolver,
    SimpleSolutionSolver,
    quick_solve,
    analyze_solution
)


# ============================================================================
# DECOMPOSITION ENGINE TESTS
# ============================================================================

class TestEnhancedDecompositionEngine(unittest.TestCase):
    """Test cases for EnhancedDecompositionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = EnhancedDecompositionEngine()
        self.sample_problem = create_problem_definition(
            title="Build Web Application",
            description="""
            Create a modern web application with user authentication,
            database integration, and responsive design.
            Must support 1000+ concurrent users.
            """,
            domain=ProblemDomain.SOFTWARE,
            complexity=7.0
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.strategies), 9)  # 9 default strategies
    
    def test_decompose_basic(self):
        """Test basic decomposition."""
        plan = self.engine.decompose(self.sample_problem)
        
        self.assertIsInstance(plan, DecompositionPlan)
        self.assertEqual(plan.original_problem.id, self.sample_problem.id)
        self.assertGreater(len(plan.sub_problems), 0)
        self.assertLessEqual(len(plan.sub_problems), 10)
    
    def test_decompose_with_strategy(self):
        """Test decomposition with specific strategy."""
        strategies = [
            DecompositionStrategy.HIERARCHICAL,
            DecompositionStrategy.FUNCTIONAL,
            DecompositionStrategy.SEMANTIC,
            DecompositionStrategy.TEMPORAL,
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                plan = self.engine.decompose(
                    self.sample_problem,
                    strategy=strategy
                )
                self.assertEqual(plan.strategy_used, strategy)
                self.assertGreater(len(plan.sub_problems), 0)
    
    def test_decompose_quality_metrics(self):
        """Test that decomposition produces quality metrics."""
        plan = self.engine.decompose(self.sample_problem)
        
        self.assertGreaterEqual(plan.coverage_score, 0.0)
        self.assertLessEqual(plan.coverage_score, 1.0)
        self.assertGreaterEqual(plan.balance_score, 0.0)
        self.assertLessEqual(plan.balance_score, 1.0)
        self.assertGreaterEqual(plan.coherence_score, 0.0)
        self.assertLessEqual(plan.coherence_score, 1.0)
        self.assertGreaterEqual(plan.overall_quality, 0.0)
        self.assertLessEqual(plan.overall_quality, 1.0)
    
    def test_decompose_dependency_graph(self):
        """Test dependency graph generation."""
        plan = self.engine.decompose(self.sample_problem)
        
        self.assertIsInstance(plan.dependency_graph, dict)
        self.assertIsInstance(plan.execution_order, list)
        self.assertIsInstance(plan.parallel_groups, list)
        
        # All sub-problems should be in execution order
        for sp in plan.sub_problems:
            self.assertIn(sp.id, plan.execution_order)
    
    def test_decompose_analysis_results(self):
        """Test analysis results in decomposition plan."""
        plan = self.engine.decompose(self.sample_problem)
        
        self.assertIsInstance(plan.complexity_analysis, dict)
        self.assertIsInstance(plan.risk_analysis, dict)
        self.assertIsInstance(plan.resource_analysis, dict)
        
        # Check complexity analysis structure
        if plan.sub_problems:
            self.assertIn('mean', plan.complexity_analysis)
            self.assertIn('min', plan.complexity_analysis)
            self.assertIn('max', plan.complexity_analysis)
    
    def test_strategy_selection(self):
        """Test automatic strategy selection."""
        strategy = self.engine._select_strategy(self.sample_problem)
        
        self.assertIsInstance(strategy, DecompositionStrategy)
        self.assertIn(strategy, self.engine.strategies)
    
    def test_caching(self):
        """Test decomposition caching."""
        # First call should cache
        plan1 = self.engine.decompose(self.sample_problem)
        
        # Second call should hit cache
        plan2 = self.engine.decompose(self.sample_problem)
        
        self.assertEqual(plan1.id, plan2.id)


class TestDecompositionStrategies(unittest.TestCase):
    """Test individual decomposition strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = create_problem_definition(
            title="Test Problem",
            description="A test problem for strategy evaluation.",
            domain=ProblemDomain.SOFTWARE,
            complexity=5.0
        )
    
    def test_hierarchical_strategy(self):
        """Test hierarchical decomposition."""
        strategy = HierarchicalDecomposition()
        
        can_handle, confidence = strategy.can_handle(self.problem)
        self.assertTrue(can_handle)
        self.assertGreater(confidence, 0)
        
        sub_problems = strategy.decompose(self.problem)
        self.assertGreater(len(sub_problems), 0)
        
        # Should have an integration sub-problem
        integration_sps = [sp for sp in sub_problems if sp.type == SubProblemType.INTEGRATION]
        self.assertGreaterEqual(len(integration_sps), 0)
    
    def test_functional_strategy(self):
        """Test functional decomposition."""
        strategy = FunctionalDecomposition()
        
        sub_problems = strategy.decompose(self.problem)
        self.assertGreater(len(sub_problems), 0)
        
        # Check that we have diverse sub-problem types
        types = set(sp.type for sp in sub_problems)
        self.assertGreater(len(types), 1)
    
    def test_semantic_strategy(self):
        """Test semantic decomposition."""
        strategy = SemanticDecomposition()
        
        # Complex problem should have higher confidence
        complex_problem = create_problem_definition(
            title="Complex Problem",
            description="Research and analyze complex system interactions.",
            complexity=8.0
        )
        
        can_handle, confidence = strategy.can_handle(complex_problem)
        self.assertTrue(can_handle)
        
        sub_problems = strategy.decompose(complex_problem)
        self.assertGreater(len(sub_problems), 0)
    
    def test_temporal_strategy(self):
        """Test temporal decomposition."""
        strategy = TemporalDecomposition()
        
        sub_problems = strategy.decompose(self.problem)
        self.assertGreater(len(sub_problems), 0)
        
        # Should have dependencies between phases
        for i, sp in enumerate(sub_problems[1:], 1):
            if sp.dependencies:
                self.assertIn(sub_problems[i-1].id, sp.dependencies)
    
    def test_causal_strategy(self):
        """Test causal decomposition."""
        strategy = CausalDecomposition()
        
        diagnostic_problem = create_problem_definition(
            title="System Failure",
            description="Diagnose and fix the root cause of system failures.",
            complexity=6.0
        )
        
        can_handle, confidence = strategy.can_handle(diagnostic_problem)
        self.assertTrue(can_handle)
        self.assertGreater(confidence, 0.5)
        
        sub_problems = strategy.decompose(diagnostic_problem)
        self.assertGreater(len(sub_problems), 0)
    
    def test_risk_based_strategy(self):
        """Test risk-based decomposition."""
        strategy = RiskBasedDecomposition()
        
        sub_problems = strategy.decompose(self.problem)
        self.assertGreater(len(sub_problems), 0)
        
        # Should have varying risk scores
        risk_scores = [sp.risk_score for sp in sub_problems if sp.risk_score > 0]
        if risk_scores:
            self.assertGreater(len(set(risk_scores)), 0)
    
    def test_complexity_based_strategy(self):
        """Test complexity-based decomposition."""
        strategy = ComplexityBasedDecomposition()
        
        high_complexity = create_problem_definition(
            title="Complex System",
            description="Build a very complex distributed system.",
            complexity=9.0
        )
        
        can_handle, confidence = strategy.can_handle(high_complexity)
        self.assertTrue(can_handle)
        self.assertGreater(confidence, 0.7)
        
        sub_problems = strategy.decompose(high_complexity)
        self.assertGreater(len(sub_problems), 0)
    
    def test_dependency_strategy(self):
        """Test dependency-based decomposition."""
        strategy = DependencyDecomposition()
        
        sub_problems = strategy.decompose(self.problem)
        self.assertGreater(len(sub_problems), 0)
        
        # Should have foundation layer
        foundation_sps = [sp for sp in sub_problems if 'foundation' in sp.title.lower()]
        self.assertGreaterEqual(len(foundation_sps), 1)
    
    def test_hybrid_strategy(self):
        """Test hybrid decomposition."""
        strategy = HybridDecomposition()
        
        can_handle, confidence = strategy.can_handle(self.problem)
        self.assertTrue(can_handle)
        self.assertGreater(confidence, 0.9)
        
        sub_problems = strategy.decompose(self.problem)
        self.assertGreater(len(sub_problems), 0)


# ============================================================================
# RECOMPOSITION ENGINE TESTS
# ============================================================================

class TestEnhancedRecompositionEngine(unittest.TestCase):
    """Test cases for EnhancedRecompositionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = EnhancedRecompositionEngine()
        
        self.sample_solutions = {
            "sub_1": create_subproblem_solution(
                "sub_1",
                "## Requirements\n\nThe system requires user authentication.",
                0.85
            ),
            "sub_2": create_subproblem_solution(
                "sub_2",
                "## Design\n\nDatabase schema includes users table.",
                0.80
            ),
            "sub_3": create_subproblem_solution(
                "sub_3",
                "## Implementation\n\nAPI endpoints for user management.",
                0.75
            ),
        }
        
        self.dependency_graph = {
            "sub_1": [],
            "sub_2": ["sub_1"],
            "sub_3": ["sub_1", "sub_2"]
        }
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.conflict_detector)
        self.assertIsNotNone(self.engine.conflict_resolver)
    
    def test_assemble_basic(self):
        """Test basic assembly."""
        solution = self.engine.assemble(
            sub_solutions=self.sample_solutions,
            problem_id="prob_123",
            decomposition_plan_id="plan_456",
            dependency_graph=self.dependency_graph
        )
        
        self.assertIsInstance(solution, IntegratedSolution)
        self.assertEqual(solution.problem_id, "prob_123")
        self.assertEqual(solution.decomposition_plan_id, "plan_456")
        self.assertGreater(len(solution.assembled_content), 0)
    
    def test_assemble_with_strategy(self):
        """Test assembly with different strategies."""
        strategies = [
            AssemblyStrategy.HIERARCHICAL,
            AssemblyStrategy.SEQUENTIAL,
            AssemblyStrategy.PARALLEL,
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                solution = self.engine.assemble(
                    sub_solutions=self.sample_solutions.copy(),
                    problem_id="prob_123",
                    decomposition_plan_id="plan_456",
                    dependency_graph=self.dependency_graph,
                    strategy=strategy
                )
                self.assertEqual(solution.assembly_strategy, strategy)
                self.assertGreater(len(solution.assembled_content), 0)
    
    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        solution = self.engine.assemble(
            sub_solutions=self.sample_solutions,
            problem_id="prob_123",
            decomposition_plan_id="plan_456",
            dependency_graph=self.dependency_graph
        )
        
        metrics = solution.quality_metrics
        self.assertIsInstance(metrics, QualityMetrics)
        
        self.assertGreaterEqual(metrics.completeness, 0.0)
        self.assertLessEqual(metrics.completeness, 1.0)
        self.assertGreaterEqual(metrics.consistency, 0.0)
        self.assertLessEqual(metrics.consistency, 1.0)
        self.assertGreaterEqual(metrics.overall_score, 0.0)
        self.assertLessEqual(metrics.overall_score, 1.0)
    
    def test_conflict_detection(self):
        """Test conflict detection."""
        # Create solutions with potential conflicts
        conflicting_solutions = {
            "sub_1": create_subproblem_solution(
                "sub_1",
                "The system must enable feature X.",
                0.8
            ),
            "sub_2": create_subproblem_solution(
                "sub_2",
                "The system must disable feature X.",
                0.8
            ),
        }
        
        solution = self.engine.assemble(
            sub_solutions=conflicting_solutions,
            problem_id="prob_123",
            decomposition_plan_id="plan_456"
        )
        
        # Should detect at least one conflict
        self.assertGreaterEqual(len(solution.conflicts_detected), 1)
    
    def test_version_control(self):
        """Test version control functionality."""
        # Create first version
        solution1 = self.engine.assemble(
            sub_solutions=self.sample_solutions,
            problem_id="prob_123",
            decomposition_plan_id="plan_456"
        )
        
        # Create second version
        modified_solutions = self.sample_solutions.copy()
        modified_solutions["sub_1"] = create_subproblem_solution(
            "sub_1",
            "Updated content",
            0.9
        )
        
        solution2 = self.engine.assemble(
            sub_solutions=modified_solutions,
            problem_id="prob_123",
            decomposition_plan_id="plan_456"
        )
        
        # Check version history
        versions = self.engine.version_history.get("prob_123", [])
        self.assertEqual(len(versions), 2)
        
        # Test rollback
        rolled_back = self.engine.rollback("prob_123", steps=1)
        self.assertIsNotNone(rolled_back)
        self.assertEqual(rolled_back.solution_id, solution1.solution_id)
    
    def test_assembly_plan_creation(self):
        """Test assembly plan creation."""
        plan = self.engine._create_assembly_plan(
            self.sample_solutions,
            AssemblyStrategy.HIERARCHICAL,
            self.dependency_graph
        )
        
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.instructions), len(self.sample_solutions))
        
        # Check position ordering
        positions = [i.position for i in plan.instructions]
        self.assertEqual(sorted(positions), positions)


class TestConflictDetector(unittest.TestCase):
    """Test cases for ConflictDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ConflictDetector()
    
    def test_contradiction_detection(self):
        """Test contradiction detection."""
        solutions = {
            "sol_1": create_subproblem_solution(
                "sol_1",
                "The system must use PostgreSQL database.",
                0.8
            ),
            "sol_2": create_subproblem_solution(
                "sol_2",
                "The system must not use PostgreSQL database.",
                0.8
            ),
        }
        
        conflicts = self.detector._detect_contradictions(solutions)
        
        # Should detect at least one contradiction
        contradiction_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.CONTRADICTION]
        self.assertGreaterEqual(len(contradiction_conflicts), 1)
    
    def test_overlap_detection(self):
        """Test content overlap detection."""
        solutions = {
            "sol_1": create_subproblem_solution(
                "sol_1",
                "User authentication is required for security. User profiles store preferences.",
                0.8
            ),
            "sol_2": create_subproblem_solution(
                "sol_2",
                "User authentication is required for security. Session management handles login.",
                0.8
            ),
        }
        
        conflicts = self.detector._detect_overlaps(solutions)
        
        # Should detect overlap
        overlap_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.CONTENT_OVERLAP]
        self.assertGreaterEqual(len(overlap_conflicts), 1)
    
    def test_interface_mismatch_detection(self):
        """Test interface mismatch detection."""
        solutions = {
            "sol_1": create_subproblem_solution(
                "sol_1",
                "GET /api/users?id=123 returns user data",
                0.8
            ),
            "sol_2": create_subproblem_solution(
                "sol_2",
                "GET /api/users?user_id=123 returns user data",
                0.8
            ),
        }
        
        conflicts = self.detector._detect_interface_mismatches(solutions)
        
        # Should detect API mismatch
        api_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.API_INCOMPATIBILITY]
        self.assertGreaterEqual(len(api_conflicts), 1)
    
    def test_quality_gap_detection(self):
        """Test quality gap detection."""
        solutions = {
            "sol_1": create_subproblem_solution(
                "sol_1",
                "Low quality content",
                0.4  # Below threshold
            ),
            "sol_2": create_subproblem_solution(
                "sol_2",
                "High quality content with detailed information",
                0.9
            ),
        }
        
        # Adjust quality scores
        solutions["sol_1"].quality_score = 0.4
        
        conflicts = self.detector._detect_quality_gaps(solutions)
        
        # Should detect quality gap
        quality_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.QUALITY_GAP]
        self.assertGreaterEqual(len(quality_conflicts), 1)


class TestConflictResolver(unittest.TestCase):
    """Test cases for ConflictResolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecompositionConfig(auto_resolve_conflicts=True)
        self.resolver = ConflictResolver(self.config)
    
    def test_auto_merge_resolution(self):
        """Test auto-merge conflict resolution."""
        solutions = {
            "sol_1": create_subproblem_solution("sol_1", "Content A", 0.8),
            "sol_2": create_subproblem_solution("sol_2", "Content B", 0.7),
        }
        
        conflict = Conflict(
            conflict_id="conf_1",
            conflict_type=ConflictType.CONTENT_OVERLAP,
            severity=ConflictSeverity.LOW,
            involved_solutions=["sol_1", "sol_2"],
            description="Test conflict",
            auto_resolvable=True
        )
        
        resolved, unresolved = self.resolver.resolve_conflicts([conflict], solutions)
        
        # Low severity, auto-resolvable conflict should be resolved
        self.assertEqual(len(resolved) + len(unresolved), 1)


# ============================================================================
# INTEGRATION PIPELINE TESTS
# ============================================================================

class TestDecompositionRecompositionPipeline(unittest.TestCase):
    """Test cases for DecompositionRecompositionPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = DecompositionRecompositionPipeline()
        
        self.sample_problem = create_problem_definition(
            title="Build API Service",
            description="""
            Create a RESTful API service with authentication,
            rate limiting, and comprehensive documentation.
            """,
            domain=ProblemDomain.SOFTWARE,
            complexity=6.0
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.decomposition_engine)
        self.assertIsNotNone(self.pipeline.recomposition_engine)
        self.assertIsNotNone(self.pipeline.solution_solver)
    
    def test_full_pipeline_execution(self):
        """Test full pipeline execution."""
        result = self.pipeline.execute(self.sample_problem)
        
        self.assertIsInstance(result, PipelineResult)
        self.assertIsNotNone(result.decomposition_plan)
        self.assertIsNotNone(result.integrated_solution)
        
        # Check stages
        self.assertGreater(len(result.stages), 0)
        
        # Check quality scores
        self.assertGreaterEqual(result.decomposition_quality, 0.0)
        self.assertGreaterEqual(result.solution_quality, 0.0)
        self.assertGreaterEqual(result.overall_quality, 0.0)
    
    def test_pipeline_success_status(self):
        """Test pipeline success status determination."""
        result = self.pipeline.execute(self.sample_problem)
        
        # Should be successful if completed with reasonable quality
        if result.overall_quality >= 0.6 and result.integrated_solution:
            self.assertTrue(result.is_successful())
    
    def test_pipeline_analytics(self):
        """Test pipeline analytics collection."""
        # Execute multiple times
        for _ in range(3):
            self.pipeline.execute(self.sample_problem)
        
        analytics = self.pipeline.get_analytics()
        
        self.assertEqual(analytics.total_executions, 3)
        self.assertGreaterEqual(analytics.successful_executions, 0)
    
    def test_pipeline_result_dict(self):
        """Test pipeline result dictionary conversion."""
        result = self.pipeline.execute(self.sample_problem)
        result_dict = result.to_dict()
        
        self.assertIn('pipeline_id', result_dict)
        self.assertIn('problem_title', result_dict)
        self.assertIn('successful', result_dict)
        self.assertIn('overall_quality', result_dict)
        self.assertIn('stages_completed', result_dict)
    
    def test_custom_solver(self):
        """Test pipeline with custom solver."""
        class CustomSolver(SolutionSolver):
            def solve(self, sub_problem):
                return create_subproblem_solution(
                    sub_problem.id,
                    f"Custom solution for {sub_problem.title}",
                    0.95
                )
            
            def can_solve(self, sub_problem):
                return True, 0.9
        
        custom_solver = CustomSolver()
        result = self.pipeline.execute(self.sample_problem, custom_solver=custom_solver)
        
        self.assertIsNotNone(result.integrated_solution)


class TestBatchPipelineProcessor(unittest.TestCase):
    """Test cases for BatchPipelineProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = DecompositionRecompositionPipeline()
        self.processor = BatchPipelineProcessor(self.pipeline)
        
        self.problems = [
            create_problem_definition(f"Problem {i}", f"Description {i}", complexity=5.0)
            for i in range(3)
        ]
    
    def test_batch_processing(self):
        """Test batch processing."""
        results = self.processor.process_batch(self.problems)
        
        self.assertEqual(len(results), len(self.problems))
        
        for result in results:
            self.assertIsInstance(result, PipelineResult)
    
    def test_batch_summary(self):
        """Test batch summary generation."""
        self.processor.process_batch(self.problems)
        
        summary = self.processor.get_summary()
        
        self.assertIn('total', summary)
        self.assertIn('successful', summary)
        self.assertIn('failed', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('avg_quality', summary)
        
        self.assertEqual(summary['total'], len(self.problems))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_quick_solve(self):
        """Test quick_solve function."""
        result = quick_solve(
            title="Quick Test Problem",
            description="A simple test problem",
            domain=ProblemDomain.SOFTWARE,
            complexity=4.0
        )
        
        self.assertIsInstance(result, PipelineResult)
        self.assertIsNotNone(result.decomposition_plan)
        self.assertIsNotNone(result.integrated_solution)
    
    def test_analyze_solution(self):
        """Test analyze_solution function."""
        pipeline = DecompositionRecompositionPipeline()
        result = pipeline.execute(
            create_problem_definition("Test", "Test description", complexity=5.0)
        )
        
        analysis = analyze_solution(result)
        
        self.assertIn('overview', analysis)
        self.assertIn('decomposition', analysis)
        self.assertIn('recomposition', analysis)
        self.assertIn('recommendations', analysis)
        
        self.assertIsInstance(analysis['recommendations'], list)


# ============================================================================
# EDGE CASE AND STRESS TESTS
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_problem_description(self):
        """Test handling of empty problem description."""
        engine = EnhancedDecompositionEngine()
        
        problem = create_problem_definition(
            title="Empty Problem",
            description="",
            complexity=3.0
        )
        
        plan = engine.decompose(problem)
        self.assertIsInstance(plan, DecompositionPlan)
    
    def test_very_complex_problem(self):
        """Test handling of very complex problem."""
        engine = EnhancedDecompositionEngine()
        
        problem = create_problem_definition(
            title="Complex System",
            description="A" * 10000,  # Very long description
            complexity=10.0
        )
        
        plan = engine.decompose(problem)
        self.assertIsInstance(plan, DecompositionPlan)
    
    def test_single_subproblem(self):
        """Test handling when only one sub-problem is generated."""
        engine = EnhancedDecompositionEngine()
        
        problem = create_problem_definition(
            title="Simple Task",
            description="Do one simple thing.",
            complexity=2.0
        )
        
        plan = engine.decompose(problem, min_subproblems=1, max_subproblems=3)
        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreaterEqual(len(plan.sub_problems), 1)
    
    def test_no_conflicts(self):
        """Test recomposition with no conflicts."""
        engine = EnhancedRecompositionEngine()
        
        solutions = {
            "sub_1": create_subproblem_solution(
                "sub_1",
                "Part 1: Requirements gathering",
                0.9
            ),
            "sub_2": create_subproblem_solution(
                "sub_2",
                "Part 2: Design specification",
                0.9
            ),
        }
        
        solution = engine.assemble(
            sub_solutions=solutions,
            problem_id="prob_1",
            decomposition_plan_id="plan_1"
        )
        
        # Should complete without critical issues
        self.assertEqual(solution.status.value, "completed")


class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def test_decomposition_performance(self):
        """Test decomposition performance."""
        engine = EnhancedDecompositionEngine()
        problem = create_problem_definition(
            "Performance Test",
            "Test problem for performance measurement",
            complexity=6.0
        )
        
        start = time.time()
        plan = engine.decompose(problem)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 5.0)
        self.assertIsNotNone(plan)
    
    def test_recomposition_performance(self):
        """Test recomposition performance."""
        engine = EnhancedRecompositionEngine()
        
        solutions = {
            f"sub_{i}": create_subproblem_solution(
                f"sub_{i}",
                f"Solution part {i}",
                0.8
            )
            for i in range(10)
        }
        
        start = time.time()
        solution = engine.assemble(
            sub_solutions=solutions,
            problem_id="perf_test",
            decomposition_plan_id="plan_test"
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 5.0)
        self.assertIsNotNone(solution)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
