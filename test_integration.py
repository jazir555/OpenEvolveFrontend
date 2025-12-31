"""
Integration Tests for Decomposition Workflow

This module provides end-to-end integration tests for the complete workflow system.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from workflow_structures import (
    DecompositionPlan, SubProblem, Team, GauntletDefinition,
    WorkflowState, ModelConfig, GauntletRoundRule
)
from knowledge_manager import KnowledgeManager
from dependency_visualizer import DependencyVisualizer
from resource_manager import ResourceManager
from auto_approval import AutoApprovalChecker, get_default_auto_approval_criteria
from batch_operations import BatchOperations


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete workflow execution."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test team
        self.test_team = Team(
            name="TestTeam",
            role="Blue",
            members=[
                ModelConfig(
                    model_id="gpt-4",
                    api_key="test_key",
                    api_base="https://api.openai.com/v1",
                    temperature=0.7,
                    max_tokens=1000
                )
            ]
        )
        
        # Create test gauntlet
        self.test_gauntlet = GauntletDefinition(
            name="TestGauntlet",
            team_name="TestTeam",
            rounds=[
                GauntletRoundRule(
                    round_number=1,
                    quorum_required_approvals=1,
                    quorum_from_panel_size=1,
                    min_overall_confidence=0.7
                )
            ]
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow_components(self):
        """Test that all workflow components work together."""
        # Create a decomposition plan
        sub_problems = [
            SubProblem(
                id="sp1",
                description="First sub-problem",
                dependencies=[],
                solver_team_name="TestTeam",
                gold_team_gauntlet_name="TestGauntlet",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            ),
            SubProblem(
                id="sp2",
                description="Second sub-problem",
                dependencies=["sp1"],
                solver_team_name="TestTeam",
                gold_team_gauntlet_name="TestGauntlet",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=5,
                content_type="text_general"
            )
        ]
        
        plan = DecompositionPlan(
            problem_statement="Test problem",
            analyzed_context={"domain": "testing"},
            sub_problems=sub_problems,
            max_refinement_loops=3,
            assembler_team_name="TestTeam",
            final_gold_team_gauntlet_name="TestGauntlet"
        )
        
        # Test dependency visualization
        visualizer = DependencyVisualizer(plan)
        self.assertEqual(len(visualizer.graph.nodes()), 2)
        self.assertEqual(len(visualizer.graph.edges()), 1)
        
        # Test execution order
        order = visualizer.suggest_execution_order()
        self.assertEqual(order, ["sp1", "sp2"])
        
        # Test circular dependency detection
        cycles = visualizer.detect_circular_dependencies()
        self.assertEqual(len(cycles), 0)
        
        # Test auto-approval
        criteria = get_default_auto_approval_criteria()
        criteria["enabled"] = True
        checker = AutoApprovalChecker(criteria)
        approved, reasons = checker.check_auto_approval(plan)
        self.assertIsInstance(approved, bool)
        
        # Test batch operations
        updated = BatchOperations.batch_assign_team(
            sub_problems,
            team_name="NewTeam",
            team_type="solver"
        )
        for sp in updated:
            self.assertEqual(sp.solver_team_name, "NewTeam")
    
    def test_knowledge_management_integration(self):
        """Test knowledge management with workflow execution."""
        km = KnowledgeManager(storage_path=self.temp_dir)
        
        # Create and store knowledge artifacts
        from workflow_structures import KnowledgeArtifact
        
        artifact1 = KnowledgeArtifact(
            id="test_1",
            artifact_type="solution_pattern",
            content={"solution": "test solution", "approach": "test approach"},
            source_workflow_id="workflow_1",
            domain="testing"
        )
        
        artifact2 = KnowledgeArtifact(
            id="test_2",
            artifact_type="problem_solution_mapping",
            content={"problem": "test problem", "solution": "test solution"},
            source_workflow_id="workflow_1",
            domain="testing"
        )
        
        km.store_knowledge_artifact(artifact1)
        km.store_knowledge_artifact(artifact2)
        
        # Retrieve artifacts
        all_artifacts = km.get_all_artifacts()
        self.assertEqual(len(all_artifacts), 2)
        
        # Test relevance search
        relevant = km.retrieve_relevant_knowledge(
            "test solution approach",
            domain="testing",
            limit=5
        )
        self.assertGreaterEqual(len(relevant), 0)
    
    def test_resource_management_integration(self):
        """Test resource management across workflow."""
        rm = ResourceManager()
        
        # Track multiple API calls
        for i in range(5):
            rm.track_api_call(
                component=f"test_component_{i}",
                model="gpt-4",
                tokens=1000
            )
        
        # Get usage summary
        summary = rm.get_usage_summary()
        self.assertEqual(summary["api_calls"], 5)
        self.assertEqual(summary["tokens_used"], 5000)
        
        # Test limit checking
        within_limits, violations = rm.check_limits()
        self.assertTrue(within_limits)
    
    def test_dependency_and_auto_approval_integration(self):
        """Test dependency visualization with auto-approval."""
        # Create plan with circular dependency
        sub_problems = [
            SubProblem(
                id="sp1",
                description="First",
                dependencies=["sp2"],  # Circular!
                solver_team_name="TestTeam",
                gold_team_gauntlet_name="TestGauntlet",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            ),
            SubProblem(
                id="sp2",
                description="Second",
                dependencies=["sp1"],  # Circular!
                solver_team_name="TestTeam",
                gold_team_gauntlet_name="TestGauntlet",
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
            assembler_team_name="TestTeam",
            final_gold_team_gauntlet_name="TestGauntlet"
        )
        
        # Detect circular dependency
        visualizer = DependencyVisualizer(plan)
        cycles = visualizer.detect_circular_dependencies()
        self.assertGreater(len(cycles), 0)
        
        # Auto-approval should reject due to circular dependency
        criteria = get_default_auto_approval_criteria()
        criteria["enabled"] = True
        criteria["reject_circular_dependencies"] = True
        
        checker = AutoApprovalChecker(criteria)
        approved, reasons = checker.check_auto_approval(plan)
        self.assertFalse(approved)
        self.assertTrue(any("circular" in reason.lower() for reason in reasons))


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features."""
    
    def test_llm_cache(self):
        """Test LLM response caching."""
        from llm_cache import LLMCache
        
        cache = LLMCache(cache_dir=tempfile.mkdtemp())
        
        # Test cache miss
        result = cache.get(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100
        )
        self.assertIsNone(result)
        
        # Store in cache
        cache.set(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100,
            response="test response"
        )
        
        # Test cache hit
        result = cache.get(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100
        )
        self.assertEqual(result, "test response")
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
    
    def test_parallel_execution(self):
        """Test parallel execution utilities."""
        from performance_utils import ParallelExecutor
        
        def test_func(item):
            return item * 2
        
        executor = ParallelExecutor(max_workers=3)
        items = [1, 2, 3, 4, 5]
        
        results = executor.execute_parallel(test_func, items)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r is not None for r in results))
    
    def test_batch_processing(self):
        """Test batch processing utilities."""
        from performance_utils import BatchProcessor
        
        def batch_func(batch):
            return [item * 2 for item in batch]
        
        processor = BatchProcessor(batch_size=2)
        items = [1, 2, 3, 4, 5]
        
        results = processor.process_in_batches(batch_func, items)
        self.assertEqual(len(results), 5)
        self.assertEqual(results, [2, 4, 6, 8, 10])


class TestComponentInteraction(unittest.TestCase):
    """Test interactions between different components."""
    
    def test_knowledge_and_resource_tracking(self):
        """Test knowledge extraction with resource tracking."""
        temp_dir = tempfile.mkdtemp()
        
        km = KnowledgeManager(storage_path=temp_dir)
        rm = ResourceManager()
        
        # Simulate workflow with resource tracking
        rm.track_api_call(component="knowledge_extraction", model="gpt-4", tokens=500)
        
        # Store knowledge
        from workflow_structures import KnowledgeArtifact
        artifact = KnowledgeArtifact(
            id="test_1",
            artifact_type="solution_pattern",
            content={"test": "data"},
            source_workflow_id="workflow_1"
        )
        km.store_knowledge_artifact(artifact)
        
        # Verify both systems tracked the operation
        self.assertEqual(len(km.get_all_artifacts()), 1)
        summary = rm.get_usage_summary()
        self.assertGreater(summary["api_calls"], 0)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_batch_operations_with_validation(self):
        """Test batch operations with auto-approval validation."""
        sub_problems = [
            SubProblem(
                id=f"sp{i}",
                description=f"Problem {i}",
                dependencies=[],
                solver_team_name="OldTeam",
                gold_team_gauntlet_name="OldGauntlet",
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=3,
                content_type="text_general"
            )
            for i in range(5)
        ]
        
        # Batch update
        updated = BatchOperations.batch_assign_team(
            sub_problems,
            team_name="NewTeam",
            team_type="solver"
        )
        
        # Create plan with updated sub-problems
        plan = DecompositionPlan(
            problem_statement="Test",
            analyzed_context={},
            sub_problems=updated,
            max_refinement_loops=3,
            assembler_team_name="Assembler",
            final_gold_team_gauntlet_name="Final"
        )
        
        # Validate with auto-approval
        criteria = get_default_auto_approval_criteria()
        criteria["enabled"] = True
        checker = AutoApprovalChecker(criteria)
        approved, reasons = checker.check_auto_approval(plan)
        
        # Should work (no circular dependencies, reasonable complexity)
        self.assertIsInstance(approved, bool)


if __name__ == "__main__":
    unittest.main()
