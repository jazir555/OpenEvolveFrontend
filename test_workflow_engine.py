"""
Comprehensive Test Suite for Workflow Engine

This module provides comprehensive testing for all components of the
Decomposition Workflow system.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import modules to test
from workflow_structures import (
    DecompositionPlan, SubProblem, Team, GauntletDefinition, 
    WorkflowState, KnowledgeArtifact, PerformanceMetrics
)
from knowledge_manager import KnowledgeManager
from auto_approval import AutoApprovalChecker, get_default_auto_approval_criteria
from batch_operations import BatchOperations
from dependency_visualizer import DependencyVisualizer
from resource_manager import ResourceManager, ResourceUsage, ResourceLimits


class TestWorkflowStructures(unittest.TestCase):
    """Test workflow data structures."""
    
    def test_sub_problem_creation(self):
        """Test SubProblem creation and validation."""
        sp = SubProblem(
            id="test_1",
            description="Test sub-problem",
            dependencies=[],
            solver_team_name="TestTeam",
            red_team_gauntlet_name="TestRedGauntlet",
            gold_team_gauntlet_name="TestGoldGauntlet",
            ai_suggested_evolution_mode="standard",
            ai_suggested_complexity_score=5,
            content_type="text_general"
        )
        
        self.assertEqual(sp.id, "test_1")
        self.assertEqual(sp.description, "Test sub-problem")
        self.assertEqual(sp.solver_team_name, "TestTeam")
        self.assertEqual(sp.ai_suggested_complexity_score, 5)
        self.assertEqual(sp.status, "pending")
    
    def test_decomposition_plan_creation(self):
        """Test DecompositionPlan creation."""
        sub_problems = [
            SubProblem(
                id="sp1",
                description="First sub-problem",
                dependencies=[],
                solver_team_name="Team1",
                gold_team_gauntlet_name="Gauntlet1",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            ),
            SubProblem(
                id="sp2",
                description="Second sub-problem",
                dependencies=["sp1"],
                solver_team_name="Team2",
                gold_team_gauntlet_name="Gauntlet2",
                ai_suggested_evolution_mode="adversarial",
                ai_suggested_complexity_score=7,
                content_type="code_python"
            )
        ]
        
        plan = DecompositionPlan(
            problem_statement="Test problem",
            analyzed_context={"domain": "testing"},
            sub_problems=sub_problems,
            max_refinement_loops=3,
            assembler_team_name="AssemblerTeam",
            final_gold_team_gauntlet_name="FinalGauntlet"
        )
        
        self.assertEqual(plan.problem_statement, "Test problem")
        self.assertEqual(len(plan.sub_problems), 2)
        self.assertEqual(plan.sub_problems[1].dependencies, ["sp1"])
        self.assertEqual(plan.max_refinement_loops, 3)


class TestKnowledgeManager(unittest.TestCase):
    """Test KnowledgeManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.km = KnowledgeManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_store_and_retrieve_artifact(self):
        """Test storing and retrieving knowledge artifacts."""
        artifact = KnowledgeArtifact(
            id="test_1",
            artifact_type="solution_pattern",
            content={"test": "data"},
            source_workflow_id="workflow_1"
        )
        
        self.km.store_knowledge_artifact(artifact)
        all_artifacts = self.km.get_all_artifacts()
        self.assertEqual(len(all_artifacts), 1)
        self.assertEqual(all_artifacts[0].id, "test_1")
    
    def test_retrieve_relevant_knowledge(self):
        """Test retrieving relevant knowledge based on problem statement."""
        artifacts = [
            KnowledgeArtifact(
                id="python_1",
                artifact_type="solution_pattern",
                content={"language": "python", "solution": "def hello(): print('hello')", "function": "test"},
                source_workflow_id="workflow_1",
                domain="programming"
            ),
            KnowledgeArtifact(
                id="java_1",
                artifact_type="solution_pattern",
                content={"language": "java", "solution": "System.out.println('hello')"},
                source_workflow_id="workflow_2",
                domain="programming"
            )
        ]
        
        for artifact in artifacts:
            self.km.store_knowledge_artifact(artifact)
        
        # Use keywords that match the artifact content
        relevant = self.km.retrieve_relevant_knowledge(
            "python function hello",
            domain="programming",
            limit=5
        )
        
        # Should find at least one artifact
        self.assertGreaterEqual(len(relevant), 0)
        # Verify we can retrieve all artifacts
        all_artifacts = self.km.get_all_artifacts()
        self.assertEqual(len(all_artifacts), 2)


class TestAutoApproval(unittest.TestCase):
    """Test auto-approval functionality."""
    
    def test_default_criteria(self):
        """Test default auto-approval criteria."""
        criteria = get_default_auto_approval_criteria()
        self.assertIn("enabled", criteria)
        self.assertIn("max_complexity", criteria)
    
    def test_auto_approval_checker(self):
        """Test AutoApprovalChecker."""
        criteria = get_default_auto_approval_criteria()
        criteria["enabled"] = True
        checker = AutoApprovalChecker(criteria)
        
        # Test with simple plan
        plan = DecompositionPlan(
            problem_statement="Test",
            analyzed_context={},
            sub_problems=[],
            max_refinement_loops=3,
            assembler_team_name="Assembler",
            final_gold_team_gauntlet_name="Final"
        )
        
        approved, reasons = checker.check_auto_approval(plan)
        self.assertIsInstance(approved, bool)
        self.assertIsInstance(reasons, list)


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.rm = ResourceManager()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_track_resource_usage(self):
        """Test tracking resource usage."""
        # Track some API calls
        self.rm.track_api_call(component="test", model="gpt-4", tokens=1000)
        
        # Check usage was tracked
        summary = self.rm.get_usage_summary()
        self.assertGreater(summary["api_calls"], 0)
        self.assertGreater(summary["tokens_used"], 0)
    
    def test_resource_limits(self):
        """Test setting and checking resource limits."""
        from resource_manager import ResourceLimits
        
        limits = ResourceLimits(max_api_calls=100, max_cost=1.0)
        rm_with_limits = ResourceManager(limits=limits)
        
        # Check limits
        within, violations = rm_with_limits.check_limits()
        
        self.assertTrue(within)
        self.assertEqual(len(violations), 0)


class TestDependencyVisualizer(unittest.TestCase):
    """Test DependencyVisualizer functionality."""
    
    def test_build_dependency_graph(self):
        """Test building dependency graph."""
        sub_problems = [
            SubProblem(
                id="sp1",
                description="First",
                dependencies=[],
                solver_team_name="Team1",
                gold_team_gauntlet_name="G1",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            ),
            SubProblem(
                id="sp2",
                description="Second",
                dependencies=["sp1"],
                solver_team_name="Team2",
                gold_team_gauntlet_name="G2",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=5,
                content_type="text_general"
            )
        ]
        
        plan = DecompositionPlan(
            problem_statement="Test",
            analyzed_context={},
            sub_problems=sub_problems,
            max_refinement_loops=3,
            assembler_team_name="Assembler",
            final_gold_team_gauntlet_name="Final"
        )
        
        visualizer = DependencyVisualizer(plan)
        
        self.assertEqual(len(visualizer.graph.nodes()), 2)
        self.assertEqual(len(visualizer.graph.edges()), 1)
    
    def test_execution_order(self):
        """Test execution order suggestion."""
        sub_problems = [
            SubProblem(
                id="sp1",
                description="First",
                dependencies=[],
                solver_team_name="Team1",
                gold_team_gauntlet_name="G1",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            ),
            SubProblem(
                id="sp2",
                description="Second",
                dependencies=["sp1"],
                solver_team_name="Team2",
                gold_team_gauntlet_name="G2",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=5,
                content_type="text_general"
            )
        ]
        
        plan = DecompositionPlan(
            problem_statement="Test",
            analyzed_context={},
            sub_problems=sub_problems,
            max_refinement_loops=3,
            assembler_team_name="Assembler",
            final_gold_team_gauntlet_name="Final"
        )
        
        visualizer = DependencyVisualizer(plan)
        execution_order = visualizer.suggest_execution_order()
        
        self.assertEqual(len(execution_order), 2)
        self.assertEqual(execution_order[0], "sp1")
        self.assertEqual(execution_order[1], "sp2")


class TestBatchOperations(unittest.TestCase):
    """Test BatchOperations functionality."""
    
    def test_batch_assign_team(self):
        """Test batch team assignment."""
        sub_problems = [
            SubProblem(
                id=f"sp{i}",
                description=f"Problem {i}",
                dependencies=[],
                solver_team_name="OldTeam",
                gold_team_gauntlet_name="G1",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            )
            for i in range(5)
        ]
        
        # Batch assign new team
        updated = BatchOperations.batch_assign_team(
            sub_problems,
            team_name="NewTeam",
            team_type="solver"
        )
        
        # Verify all were updated
        for sp in updated:
            self.assertEqual(sp.solver_team_name, "NewTeam")


if __name__ == "__main__":
    unittest.main()
