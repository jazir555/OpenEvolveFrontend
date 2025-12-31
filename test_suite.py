"""
Sovereign-Grade Problem Decomposition System - Testing Framework
Implements comprehensive unit tests, integration tests, and end-to-end tests.
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, Any, List
import json
import tempfile
import os
from datetime import datetime
import logging

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
    Constraint, SuccessCriterion, DomainContext, ComplexityScore, generate_id
)
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from dependency_manager import DependencyManager
from sovereign_team_coordination import TeamCoordinator, DecompositionWorkflow
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_gauntlets import GauntletSystem
from sovereign_persistence import SovereignDatabase
from sovereign_reliability import (
    with_error_handling, with_retry, ErrorSeverity, 
    ValidationError, PersistenceError, DecompositionError
)


class TestProblemAnalyzer(unittest.TestCase):
    """Unit tests for ProblemAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ProblemAnalyzer()
    
    def test_analyze_problem_basic(self):
        """Test basic problem analysis."""
        problem_text = "How can we optimize database queries in a web application?"
        result = self.analyzer.analyze_problem(problem_text, "Query Optimization")
        
        self.assertIsInstance(result, ProblemDefinition)
        self.assertEqual(result.title, "Query Optimization")
        self.assertIn("database", result.description.lower())
        self.assertIsNotNone(result.domain_context.domain)
        self.assertIsNotNone(result.complexity_score)
    
    def test_extract_domain_context(self):
        """Test domain context extraction."""
        problem_text = "Machine learning model training optimization"
        result = self.analyzer.extract_domain_context(problem_text)
        
        self.assertIsInstance(result, DomainContext)
        # Either machine_learning or a related domain should be identified
        self.assertTrue(result.domain in ["machine_learning", "software_engineering", "optimization"])
    
    def test_classify_problem_type(self):
        """Test problem type classification."""
        analysis = ProblemAnalyzer()
        
        research_problem = "Analyze the impact of quantum computing on cryptography"
        research_type = analysis.classify_problem_type(research_problem)
        self.assertEqual(research_type.value, "RESEARCH")
        
        implementation_problem = "Build a REST API for user management"
        implementation_type = analysis.classify_problem_type(implementation_problem)
        self.assertEqual(implementation_type.value, "IMPLEMENTATION")
    
    def test_assess_complexity(self):
        """Test complexity assessment."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="This is a test problem with moderate complexity",
            problem_type="ANALYSIS",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Initial"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        complexity = self.analyzer.assess_complexity(problem)
        self.assertIsInstance(complexity, ComplexityScore)
        self.assertGreaterEqual(complexity.overall_complexity, 0)
        self.assertLessEqual(complexity.overall_complexity, 10)
    
    def test_identify_constraints(self):
        """Test constraint identification."""
        problem_text = "Solve this by Friday with budget under $10,000 and high quality"
        constraints = self.analyzer.identify_constraints(problem_text)
        
        self.assertIsInstance(constraints, list)
        # Should identify time and resource constraints
        constraint_types = [c.type for c in constraints]
        self.assertTrue(any(ct in constraint_types for ct in ["time", "resource"]))
    
    def test_generate_success_criteria(self):
        """Test success criteria generation."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="Solve this test problem",
            problem_type="ANALYSIS",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Initial"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        criteria = self.analyzer.generate_success_criteria(problem)
        self.assertIsInstance(criteria, list)
        self.assertGreater(len(criteria), 0)
        self.assertIsInstance(criteria[0], SuccessCriterion)


class TestDecompositionEngine(unittest.TestCase):
    """Unit tests for DecompositionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ProblemAnalyzer()
        self.engine = DecompositionEngine(problem_analyzer=self.analyzer)
    
    def test_decompose_problem(self):
        """Test problem decomposition."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Web Application Optimization",
            description="Optimize a web application for better performance",
            problem_type="OPTIMIZATION",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=7.0,
                computational_complexity=6.0,
                domain_complexity=7.0,
                integration_complexity=6.0,
                overall_complexity=6.5,
                explanation="Complex optimization problem"
            ),
            constraints=[
                Constraint(
                    id=generate_id("constraint"),
                    description="Must be completed in 2 weeks",
                    type="time",
                    severity="hard"
                )
            ],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description="Performance improved by 50%",
                    metric="performance_improvement",
                    threshold=0.5,
                    validation_method="measurement"
                )
            ]
        )
        
        plan = self.engine.decompose(problem, strategy="hybrid")
        
        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)
        self.assertEqual(plan.problem_id, problem.id)
        self.assertIsNotNone(plan.dependency_graph)
        self.assertIsNotNone(plan.quality_scores)
    
    def test_select_strategy(self):
        """Test strategy selection."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Simple Analysis",
            description="Perform a simple analysis of user data",
            problem_type="ANALYSIS",
            domain_context=DomainContext(domain="data_science"),
            complexity_score=ComplexityScore(
                cognitive_complexity=3.0,
                computational_complexity=2.0,
                domain_complexity=3.0,
                integration_complexity=2.0,
                overall_complexity=2.5,
                explanation="Simple problem"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        strategy = self.engine.select_strategy(problem)
        self.assertIn(strategy, ["semantic", "dependency", "complexity", "hybrid", "research"])
    
    def test_decompose_with_dependencies(self):
        """Test decomposition that respects dependencies."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Multi-step Process",
            description="A process that requires multiple dependent steps",
            problem_type="IMPLEMENTATION",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=6.0,
                integration_complexity=7.0,
                overall_complexity=6.0,
                explanation="Integration-heavy problem"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        plan = self.engine.decompose(problem, strategy="dependency")
        
        self.assertIsInstance(plan, DecompositionPlan)
        self.assertGreater(len(plan.sub_problems), 0)
        
        # Check that dependency graph has execution order
        if plan.dependency_graph and plan.dependency_graph.execution_order:
            self.assertGreater(len(plan.dependency_graph.execution_order), 0)
    
    def test_error_handling_in_decomposition(self):
        """Test error handling in decomposition."""
        # Create a problem with invalid data to trigger error handling
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="",  # Empty title should cause validation to fail
            description="",
            problem_type="ANALYSIS",
            domain_context=DomainContext(domain=""),
            complexity_score=ComplexityScore(
                cognitive_complexity=0.0,
                computational_complexity=0.0,
                domain_complexity=0.0,
                integration_complexity=0.0,
                overall_complexity=0.0,
                explanation="Invalid problem"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        # This should not crash due to error handling
        plan = self.engine.decompose(problem, strategy="semantic")
        self.assertIsInstance(plan, DecompositionPlan)  # Should return a plan even if error occurred


class TestDependencyManager(unittest.TestCase):
    """Unit tests for DependencyManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DependencyManager()
    
    def create_test_subproblems(self) -> List[SubProblem]:
        """Create test sub-problems with dependencies."""
        sp1 = SubProblem(
            id="sp1",
            parent_id="p1",
            title="Data Collection",
            description="Collect necessary data",
            type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=4.0,
                computational_complexity=3.0,
                domain_complexity=4.0,
                integration_complexity=2.0,
                overall_complexity=3.25,
                explanation="Moderate complexity"
            ),
            dependencies=[],
            success_criteria=[]
        )
        
        sp2 = SubProblem(
            id="sp2",
            parent_id="p1",
            title="Data Analysis",
            description="Analyze collected data",
            type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=3.0,
                overall_complexity=4.25,
                explanation="Moderate complexity"
            ),
            dependencies=["sp1"],  # Depends on sp1
            success_criteria=[]
        )
        
        sp3 = SubProblem(
            id="sp3",
            parent_id="p1",
            title="Implementation",
            description="Implement solution",
            type="IMPLEMENTATION",
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=7.0,
                domain_complexity=6.0,
                integration_complexity=5.0,
                overall_complexity=6.0,
                explanation="High complexity"
            ),
            dependencies=["sp1", "sp2"],  # Depends on both sp1 and sp2
            success_criteria=[]
        )
        
        return [sp1, sp2, sp3]
    
    def test_build_dependency_graph(self):
        """Test dependency graph construction."""
        sub_problems = self.create_test_subproblems()
        graph = self.manager.build_graph(sub_problems)
        
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 3)  # Each sub-problem in edges dict
        
        # Check dependencies
        self.assertEqual(graph.edges["sp2"], ["sp1"])  # sp2 depends on sp1
        self.assertEqual(set(graph.edges["sp3"]), {"sp1", "sp2"})  # sp3 depends on sp1 and sp2
    
    def test_detect_cycles(self):
        """Test cycle detection."""
        # Create a cycle: sp1 -> sp2 -> sp3 -> sp1
        sp1 = SubProblem(
            id="sp1", parent_id="p1", title="A", description="A", type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0, computational_complexity=5.0,
                domain_complexity=5.0, integration_complexity=5.0,
                overall_complexity=5.0, explanation="Test"
            ),
            dependencies=["sp3"], success_criteria=[]
        )
        
        sp2 = SubProblem(
            id="sp2", parent_id="p1", title="B", description="B", type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0, computational_complexity=5.0,
                domain_complexity=5.0, integration_complexity=5.0,
                overall_complexity=5.0, explanation="Test"
            ),
            dependencies=["sp1"], success_criteria=[]
        )
        
        sp3 = SubProblem(
            id="sp3", parent_id="p1", title="C", description="C", type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0, computational_complexity=5.0,
                domain_complexity=5.0, integration_complexity=5.0,
                overall_complexity=5.0, explanation="Test"
            ),
            dependencies=["sp2"], success_criteria=[]
        )
        
        sub_problems = [sp1, sp2, sp3]
        graph = self.manager.build_graph(sub_problems)
        
        cycles = self.manager.detect_cycles(graph)
        self.assertGreater(len(cycles), 0)  # Should detect the cycle
    
    def test_find_critical_path(self):
        """Test critical path identification."""
        sub_problems = self.create_test_subproblems()
        graph = self.manager.build_graph(sub_problems)
        
        critical_path = self.manager.find_critical_path(graph)
        
        # Critical path should include all three nodes in dependency order
        self.assertIn("sp1", critical_path)
        self.assertIn("sp2", critical_path)
        self.assertIn("sp3", critical_path)
        
        # sp1 should come before sp2, sp2 before sp3
        sp1_idx = critical_path.index("sp1")
        sp2_idx = critical_path.index("sp2")
        sp3_idx = critical_path.index("sp3")
        
        self.assertLess(sp1_idx, sp2_idx)
        self.assertLess(sp2_idx, sp3_idx)
    
    def test_calculate_execution_order(self):
        """Test execution order calculation."""
        sub_problems = self.create_test_subproblems()
        graph = self.manager.build_graph(sub_problems)
        
        execution_order = self.manager.calculate_execution_order(graph)
        
        # sp1 should be first (no dependencies), then sp2, then sp3
        self.assertEqual(execution_order[0], "sp1")
        # sp2 and sp3 can be in any order after sp1, but let's ensure sp2 comes before sp3 based on dependency
        sp2_index = execution_order.index("sp2")
        sp3_index = execution_order.index("sp3")
        # Actually, sp2 must come before sp3 since sp3 depends on sp2
        self.assertLess(sp2_index, sp3_index)


class TestSolutionOrchestrator(unittest.TestCase):
    """Unit tests for SolutionOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SolutionOrchestrator()
    
    def create_test_subproblem(self) -> SubProblem:
        """Create a test sub-problem."""
        return SubProblem(
            id="test_sp1",
            parent_id="test_p1",
            title="Test Sub-Problem",
            description="A test sub-problem to solve",
            type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5,
                explanation="Test complexity"
            ),
            dependencies=[],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description="Solution is correct and efficient",
                    metric="accuracy",
                    threshold=0.9,
                    validation_method="test"
                )
            ]
        )
    
    def test_track_solution_attempt(self):
        """Test tracking solution attempts."""
        subproblem_id = "test_sp1"
        approach = "statistical_analysis"
        solution_content = "Implemented statistical analysis approach..."
        team_id = "test_team"
        
        attempt = self.orchestrator.track_solution_attempt(
            subproblem_id, approach, solution_content, team_id, confidence_score=0.85
        )
        
        self.assertIsInstance(attempt, SolutionAttempt)
        self.assertEqual(attempt.sub_problem_id, subproblem_id)
        self.assertEqual(attempt.approach, approach)
        self.assertEqual(attempt.team_id, team_id)
        self.assertEqual(attempt.confidence_score, 0.85)
    
    def test_validate_solution(self):
        """Test solution validation."""
        subproblem = self.create_test_subproblem()
        
        attempt = SolutionAttempt(
            id=generate_id("solution"),
            sub_problem_id=subproblem.id,
            approach="test_approach",
            solution_content="Test solution content",
            team_id="test_team",
            confidence_score=0.8,
            validation_results=[],
            feedback=[],
            status="pending"
        )
        
        # Since this requires LLM, we'll test the structure
        # In a real test, we might mock the LLM call
        with patch.object(self.orchestrator, '_validate_solution_with_llm') as mock_llm:
            from sovereign_reliability import ValidationResult
            mock_llm.return_value = ValidationResult(
                validator="test_validator",
                passed=True,
                score=0.85,
                feedback="Solution looks good",
                improvements=[],
                timestamp=datetime.now()
            )
            
            result = self.orchestrator.validate_solution(attempt, subproblem)
            self.assertIsInstance(result, ValidationResult)
            self.assertTrue(result.passed)
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        attempt1 = SolutionAttempt(
            id=generate_id("solution1"),
            sub_problem_id="sp1",
            approach="approach1",
            solution_content="content1",
            team_id="team1",
            confidence_score=0.8,
            status="validated"
        )
        
        attempt2 = SolutionAttempt(
            id=generate_id("solution2"),
            sub_problem_id="sp1",
            approach="approach2",
            solution_content="content2",
            team_id="team2",
            confidence_score=0.9,
            status="validated"
        )
        
        confidence = self.orchestrator.calculate_confidence([attempt1, attempt2])
        
        # Should be between 0 and 1
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
        # Should be reasonably high since both attempts are validated with high confidence
        self.assertGreater(confidence, 0.7)


class TestGauntletSystem(unittest.TestCase):
    """Unit tests for GauntletSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gauntlet_system = GauntletSystem()
    
    def create_test_plan(self) -> DecompositionPlan:
        """Create a test decomposition plan."""
        sub_problem = SubProblem(
            id="sp1",
            parent_id="p1",
            title="Test Sub-Problem",
            description="A test sub-problem",
            type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5,
                explanation="Test"
            ),
            dependencies=[],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description="Test criterion",
                    metric="test_metric",
                    threshold=0.8,
                    validation_method="test"
                )
            ]
        )
        
        return DecompositionPlan(
            id=generate_id("plan"),
            problem_id="test_problem",
            strategy="SEMANTIC",
            sub_problems=[sub_problem]
        )
    
    def test_run_decomposition_gauntlets(self):
        """Test running decomposition gauntlets."""
        plan = self.create_test_plan()
        
        # Test running all gauntlets
        results = self.gauntlet_system.run_decomposition_gauntlets(plan)
        
        # Should have multiple results (for different gauntlets)
        self.assertGreater(len(results), 0)
        
        # Each result should be a ValidationResult
        for name, result in results.items():
            from sovereign_data_models import ValidationResult
            self.assertIsInstance(result, ValidationResult)
    
    def test_get_overall_quality(self):
        """Test overall quality calculation."""
        from sovereign_data_models import ValidationResult
        
        results = {
            "coherence": ValidationResult(
                validator="coherence",
                passed=True,
                score=0.8,
                feedback="Good coherence",
                improvements=[],
                timestamp=datetime.now()
            ),
            "completeness": ValidationResult(
                validator="completeness", 
                passed=True,
                score=0.75,
                feedback="Mostly complete",
                improvements=[],
                timestamp=datetime.now()
            ),
            "feasibility": ValidationResult(
                validator="feasibility",
                passed=False,
                score=0.6,
                feedback="Some feasibility issues",
                improvements=[],
                timestamp=datetime.now()
            )
        }
        
        overall_quality = self.gauntlet_system.get_overall_quality(results)
        self.assertGreaterEqual(overall_quality, 0)
        self.assertLessEqual(overall_quality, 1)
        
        # Should be the average of the scores
        expected = (0.8 + 0.75 + 0.6) / 3
        self.assertAlmostEqual(overall_quality, expected, places=2)
    
    def test_all_passed(self):
        """Test all passed check."""
        from sovereign_data_models import ValidationResult
        
        # All passed
        all_passed_results = {
            "g1": ValidationResult(validator="v1", passed=True, score=0.9, feedback="", improvements=[], timestamp=datetime.now()),
            "g2": ValidationResult(validator="v2", passed=True, score=0.8, feedback="", improvements=[], timestamp=datetime.now())
        }
        
        self.assertTrue(self.gauntlet_system.all_passed(all_passed_results))
        
        # Not all passed
        not_all_passed_results = {
            "g1": ValidationResult(validator="v1", passed=True, score=0.9, feedback="", improvements=[], timestamp=datetime.now()),
            "g2": ValidationResult(validator="v2", passed=False, score=0.4, feedback="", improvements=[], timestamp=datetime.now())
        }
        
        self.assertFalse(self.gauntlet_system.all_passed(not_all_passed_results))


class TestSovereignDatabase(unittest.TestCase):
    """Unit tests for SovereignDatabase."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary database file
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
        self.db = SovereignDatabase(db_path=self.db_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.db_file)
    
    def test_create_and_get_problem(self):
        """Test creating and retrieving a problem."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="A test problem for database operations",
            problem_type="ANALYSIS",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        # Create the problem
        result = self.db.create_problem(problem)
        self.assertTrue(result)
        
        # Retrieve the problem
        retrieved = self.db.get_problem(problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, problem.id)
        self.assertEqual(retrieved.title, problem.title)
        self.assertEqual(retrieved.description, problem.description)
    
    def test_create_and_get_subproblem(self):
        """Test creating and retrieving a sub-problem."""
        sub_problem = SubProblem(
            id=generate_id("subproblem"),
            parent_id="parent123",
            title="Test Sub-Problem",
            description="A test sub-problem",
            type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=4.0,
                computational_complexity=3.0,
                domain_complexity=4.0,
                integration_complexity=3.0,
                overall_complexity=3.5,
                explanation="Test"
            ),
            dependencies=[],
            success_criteria=[]
        )
        
        # Create the sub-problem
        result = self.db.create_subproblem(sub_problem)
        self.assertTrue(result)
        
        # Retrieve the sub-problem
        retrieved = self.db.get_subproblem(sub_problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, sub_problem.id)
        self.assertEqual(retrieved.title, sub_problem.title)
    
    def test_create_and_get_solution_attempt(self):
        """Test creating and retrieving a solution attempt."""
        attempt = SolutionAttempt(
            id=generate_id("solution"),
            sub_problem_id="sp123",
            approach="test_approach",
            solution_content="Test solution content",
            team_id="test_team",
            confidence_score=0.8
        )
        
        # Create the attempt
        result = self.db.create_solution_attempt(attempt)
        self.assertTrue(result)
        
        # Retrieve the attempt
        retrieved = self.db.get_solution_attempt(attempt.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, attempt.id)
        self.assertEqual(retrieved.team_id, attempt.team_id)
    
    def test_list_problems(self):
        """Test listing problems."""
        # Create several problems
        for i in range(3):
            problem = ProblemDefinition(
                id=generate_id("problem"),
                title=f"Test Problem {i}",
                description=f"A test problem {i}",
                problem_type="ANALYSIS",
                domain_context=DomainContext(domain="software_engineering"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0, computational_complexity=5.0,
                    domain_complexity=5.0, integration_complexity=5.0,
                    overall_complexity=5.0, explanation="Test"
                ),
                constraints=[],
                success_criteria=[]
            )
            self.db.create_problem(problem)
        
        # List all problems
        problems = self.db.list_problems()
        self.assertGreaterEqual(len(problems), 3)
    
    def test_list_subproblems(self):
        """Test listing sub-problems for a parent."""
        parent_id = "parent123"
        
        # Create several sub-problems with the same parent
        for i in range(2):
            sub_problem = SubProblem(
                id=generate_id("subproblem"),
                parent_id=parent_id,
                title=f"Test Sub-Problem {i}",
                description=f"A test sub-problem {i}",
                type="ANALYSIS",
                complexity_score=ComplexityScore(
                    cognitive_complexity=4.0, computational_complexity=3.0,
                    domain_complexity=4.0, integration_complexity=3.0,
                    overall_complexity=3.5, explanation="Test"
                ),
                dependencies=[],
                success_criteria=[]
            )
            self.db.create_subproblem(sub_problem)
        
        # List sub-problems for the parent
        sub_problems = self.db.list_subproblems(parent_id)
        self.assertEqual(len(sub_problems), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
        self.db = SovereignDatabase(db_path=self.db_file)
        self.analyzer = ProblemAnalyzer()
        self.engine = DecompositionEngine(problem_analyzer=self.analyzer)
        self.orchestrator = SolutionOrchestrator()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.db_file)
    
    def test_complete_workflow(self):
        """Test a complete workflow from problem analysis to solution integration."""
        # Step 1: Analyze a problem
        problem_text = "Develop a recommendation engine for e-commerce"
        problem = self.analyzer.analyze_problem(problem_text, "Recommendation Engine")
        
        # Save problem to database
        self.db.create_problem(problem)
        
        # Step 2: Decompose the problem
        plan = self.engine.decompose(problem)
        
        # Save plan to database
        self.db.create_plan(plan)
        
        # Step 3: Process sub-problems
        for sub_problem in plan.sub_problems:
            # Create a solution attempt
            attempt = self.orchestrator.track_solution_attempt(
                sub_problem.id,
                "machine_learning_approach",
                "Implemented ML-based solution",
                "ml_team",
                confidence_score=0.85
            )
            
            # Save solution attempt
            self.db.create_solution_attempt(attempt)
        
        # Step 4: Validate and integrate solutions
        integrated_solution = self.orchestrator.integrate_solutions(plan)
        
        # Assertions
        self.assertIsNotNone(integrated_solution)
        self.assertGreater(len(integrated_solution.sub_solutions), 0)
        self.assertGreater(len(integrated_solution.final_content), 0)
        self.assertGreater(integrated_solution.confidence_score, 0.5)


class TestPerformance(unittest.TestCase):
    """Performance tests for the system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ProblemAnalyzer()
        self.engine = DecompositionEngine(problem_analyzer=self.analyzer)
    
    def test_problem_analysis_performance(self):
        """Test performance of problem analysis."""
        import time
        
        problem_text = "Optimize database queries for a large-scale application with complex relationships and high throughput requirements"
        
        start_time = time.time()
        problem = self.analyzer.analyze_problem(problem_text, "Performance Optimization")
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Analysis should complete in a reasonable time (less than 10 seconds)
        self.assertLess(analysis_time, 10.0)
        self.assertIsNotNone(problem)
    
    def test_decomposition_performance(self):
        """Test performance of problem decomposition."""
        import time
        
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Complex System Integration",
            description="Integrate multiple complex systems with various dependencies and requirements",
            problem_type="IMPLEMENTATION",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=7.0,
                computational_complexity=6.0,
                domain_complexity=7.0,
                integration_complexity=8.0,
                overall_complexity=7.0,
                explanation="Complex integration problem"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        start_time = time.time()
        plan = self.engine.decompose(problem)
        end_time = time.time()
        
        decomposition_time = end_time - start_time
        
        # Decomposition should complete in a reasonable time (less than 15 seconds for complex problems)
        self.assertLess(decomposition_time, 15.0)
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.sub_problems), 0)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and resilience."""
    
    def test_with_error_handling_decorator(self):
        """Test the with_error_handling decorator."""
        
        @with_error_handling(fallback=lambda x: f"fallback_result_{x}", severity=ErrorSeverity.MEDIUM)
        def test_function(value):
            if value == "error":
                raise ValueError("Test error")
            return f"result_{value}"
        
        # Test normal execution
        result1 = test_function("ok")
        self.assertEqual(result1, "result_ok")
        
        # Test error handling
        result2 = test_function("error")
        self.assertEqual(result2, "fallback_result_error")
    
    def test_with_retry_decorator(self):
        """Test the with_retry decorator."""
        attempt_count = 0
        max_attempts = 3
        
        @with_retry(max_attempts=max_attempts, retry_on=(ValueError,))
        def test_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < max_attempts:
                raise ValueError("Simulated error")
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, max_attempts)


# Additional specialized tests
class TestGauntletIntegration(unittest.TestCase):
    """Tests for gauntlet integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gauntlet_system = GauntletSystem()
        self.analyzer = ProblemAnalyzer()
        self.engine = DecompositionEngine(problem_analyzer=self.analyzer)
    
    def test_gauntlet_validation_of_decomposition(self):
        """Test that gauntlets properly validate decompositions."""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="A test problem for gauntlet validation",
            problem_type="ANALYSIS",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        plan = self.engine.decompose(problem)
        
        # Run gauntlets
        gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(plan)
        
        # Check that results are returned
        self.assertGreater(len(gauntlet_results), 0)
        
        # Check that overall quality can be calculated
        overall_quality = self.gauntlet_system.get_overall_quality(gauntlet_results)
        self.assertIsInstance(overall_quality, (int, float))
        self.assertGreaterEqual(overall_quality, 0)
        self.assertLessEqual(overall_quality, 1)


# Test suite for running all tests
def create_test_suite():
    """Create a test suite with all tests."""
    suite = unittest.TestSuite()
    
    # Add all test cases
    loader = unittest.TestLoader()
    
    suite.addTests(loader.loadTestsFromTestCase(TestProblemAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDecompositionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDependencyManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSolutionOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestGauntletSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSovereignDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestGauntletIntegration))
    
    return suite


def run_tests():
    """Run all tests."""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    
    return result


if __name__ == "__main__":
    run_tests()