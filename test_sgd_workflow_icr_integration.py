"""
Test SGDWorkflowOrchestrator ICR Integration

Tests for the ICR (Iterative Contextual Refinements) integration in SGDWorkflowOrchestrator.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import time
import json

# Import the SGD workflow orchestrator
from sgd_workflow_orchestrator import (
    SGDWorkflowOrchestrator,
    SGDWorkflowStatus
)


class TestSGDWorkflowICRIntegration(unittest.TestCase):
    """Test ICR integration in SGDWorkflowOrchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = SGDWorkflowOrchestrator(
            CREWAI_api_base="http://localhost:8002",
            openevolve_api_base="http://localhost:8000",
            enable_icr=True
        )
        
        # Sample configurations
        self.sample_team_config = {
            'content_analyzer_team': 'content_analysis_team',
            'planner_team': 'planning_team',
            'solver_team': 'solver_team',
            'patcher_team': 'patcher_team',
            'assembler_team': 'assembler_team'
        }
        
        self.sample_gauntlet_config = {
            'sub_problem_red_gauntlet': 'coherence',
            'sub_problem_gold_gauntlet': 'completeness',
            'final_red_gauntlet': 'feasibility',
            'final_gold_gauntlet': 'dependency'
        }
    
    def test_icr_enabled_by_default(self):
        """Test that ICR is enabled by default"""
        orchestrator = SGDWorkflowOrchestrator()
        self.assertTrue(orchestrator.enable_icr)
        self.assertIsNotNone(orchestrator.icr_patterns)
    
    def test_icr_can_be_disabled(self):
        """Test that ICR can be disabled"""
        orchestrator = SGDWorkflowOrchestrator(enable_icr=False)
        self.assertFalse(orchestrator.enable_icr)
    
    def test_analyze_problem_complexity_simple(self):
        """Test problem complexity analysis for simple problems"""
        problem = "Fix the login bug"
        problem_type, complexity = self.orchestrator._analyze_problem_complexity(problem)
        
        self.assertEqual(problem_type, "debugging")
        self.assertLessEqual(complexity, 10)
        self.assertGreaterEqual(complexity, 1)
    
    def test_analyze_problem_complexity_implementation(self):
        """Test problem complexity analysis for implementation problems"""
        problem = "Implement a secure user authentication system with OAuth2"
        problem_type, complexity = self.orchestrator._analyze_problem_complexity(problem)
        
        self.assertEqual(problem_type, "implementation")
        # Should have higher complexity due to keywords
        self.assertGreater(complexity, 4)
    
    def test_analyze_problem_complexity_design(self):
        """Test problem complexity analysis for design problems"""
        problem = "Design a scalable microservices architecture for a banking application"
        problem_type, complexity = self.orchestrator._analyze_problem_complexity(problem)
        
        self.assertEqual(problem_type, "design")
        # Should have higher complexity due to keywords
        self.assertGreater(complexity, 5)
    
    def test_analyze_problem_complexity_ml(self):
        """Test problem complexity analysis for ML problems"""
        problem = "Build a machine learning system using neural networks for natural language processing"
        problem_type, complexity = self.orchestrator._analyze_problem_complexity(problem)
        
        self.assertIn(problem_type, ["implementation", "general"])
        # Should have higher complexity due to ML keywords
        self.assertGreater(complexity, 5)
    
    def test_analyze_problem_complexity_length_factor(self):
        """Test that problem length affects complexity"""
        short_problem = "Fix bug"
        long_problem = "Fix bug " * 100  # Long problem statement
        
        _, short_complexity = self.orchestrator._analyze_problem_complexity(short_problem)
        _, long_complexity = self.orchestrator._analyze_problem_complexity(long_problem)
        
        self.assertLessEqual(long_complexity, 10)
        # Long problem should have higher or equal complexity
        self.assertGreaterEqual(long_complexity, short_complexity)
    
    def test_predict_workflow_success_no_patterns(self):
        """Test prediction when no patterns exist"""
        prediction = self.orchestrator.predict_workflow_success(
            problem_statement="Implement a simple feature",
            team_config=self.sample_team_config,
            gauntlet_config=self.sample_gauntlet_config
        )
        
        # Should return moderate probability with low confidence
        self.assertIn('success_probability', prediction)
        self.assertIn('confidence', prediction)
        self.assertGreaterEqual(prediction['success_probability'], 0.0)
        self.assertLessEqual(prediction['success_probability'], 1.0)
        self.assertLess(prediction['confidence'], 0.5)  # Low confidence without patterns
    
    def test_predict_workflow_success_with_patterns(self):
        """Test prediction with stored patterns"""
        # Store some successful patterns
        for i in range(5):
            self.orchestrator.store_workflow_pattern(
                workflow_id=f"wf_{i}",
                problem_statement="Implement a feature for user management",
                team_config=self.sample_team_config,
                gauntlet_config=self.sample_gauntlet_config,
                success=True,
                duration_seconds=300,
                stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']
            )
        
        # Store a failed pattern
        self.orchestrator.store_workflow_pattern(
            workflow_id="wf_failed",
            problem_statement="Fix a complex security issue",
            team_config=self.sample_team_config,
            gauntlet_config=self.sample_gauntlet_config,
            success=False,
            duration_seconds=600,
            stages_completed=['content_analysis', 'planning']
        )
        
        # Get prediction
        prediction = self.orchestrator.predict_workflow_success(
            problem_statement="Implement user feature",
            team_config=self.sample_team_config,
            gauntlet_config=self.sample_gauntlet_config
        )
        
        # Should have higher confidence with more patterns
        self.assertGreater(prediction['confidence'], 0.25)
        self.assertIn('problem_type', prediction)
        self.assertIn('estimated_complexity', prediction)
        self.assertIn('risk_factors', prediction)
        self.assertIn('recommendations', prediction)
    
    def test_store_workflow_pattern(self):
        """Test storing workflow patterns"""
        initial_stats = self.orchestrator.get_icr_statistics()
        initial_count = initial_stats['total_workflows']
        
        # Store a pattern
        self.orchestrator.store_workflow_pattern(
            workflow_id="wf_test_1",
            problem_statement="Implement a new feature",
            team_config=self.sample_team_config,
            gauntlet_config=self.sample_gauntlet_config,
            success=True,
            duration_seconds=450,
            stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']
        )
        
        # Verify pattern was stored
        stats = self.orchestrator.get_icr_statistics()
        self.assertEqual(stats['total_workflows'], initial_count + 1)
        self.assertIn('success_rates_by_type', stats)
    
    def test_recommend_optimal_config_no_patterns(self):
        """Test configuration recommendation when no patterns exist"""
        recommendation = self.orchestrator.recommend_optimal_config(
            problem_statement="Implement a simple feature"
        )
        
        # Should return default configuration
        self.assertIn('content_analyzer_team', recommendation)
        self.assertIn('planner_team', recommendation)
        self.assertIn('solver_team', recommendation)
        self.assertIn('mdap_enabled', recommendation)
        self.assertIn('reason', recommendation)
    
    def test_recommend_optimal_config_with_patterns(self):
        """Test configuration recommendation with stored patterns"""
        # Store successful patterns
        for i in range(5):
            self.orchestrator.store_workflow_pattern(
                workflow_id=f"wf_opt_{i}",
                problem_statement="Design a scalable system",
                team_config={\n            'content_analyzer_team': 'expert_analysis_team',\n            'planner_team': 'expert_planning_team',\n            'solver_team': 'expert_solver_team',\n            'patcher_team': 'expert_patcher_team',\n            'assembler_team': 'expert_assembler_team'\n        },\n        gauntlet_config={\n            'sub_problem_red_gauntlet': 'adaptive',\n            'sub_problem_gold_gauntlet': 'hierarchical',\n            'final_red_gauntlet': 'adaptive',\n            'final_gold_gauntlet': 'hierarchical'\n        },\n        success=True,\n        duration_seconds=500,\n        stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']\n    )\n\n    # Get recommendation\n    recommendation = self.orchestrator.recommend_optimal_config(\n        problem_statement=\"Design a scalable system for e-commerce\"\n    )\n\n    # Should recommend the successful configuration\n    self.assertEqual(recommendation['content_analyzer_team'], 'expert_analysis_team')\n    self.assertEqual(recommendation['planner_team'], 'expert_planning_team')\n    self.assertGreater(recommendation['confidence'], 0.5)\n    self.assertIn('estimated_success_rate', recommendation)\n\ndef test_get_icr_statistics(self):\n    \"\"\"Test ICR statistics retrieval\"\"\"\n    # Store some patterns\n    for i in range(3):\n        self.orchestrator.store_workflow_pattern(\n            workflow_id=f\"wf_stat_{i}\",\n            problem_statement=\"Implement feature\",\n            team_config=self.sample_team_config,\n            gauntlet_config=self.sample_gauntlet_config,\n            success=True,\n            duration_seconds=300,\n            stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']\n        )\n\n    # Store a failed pattern\n    self.orchestrator.store_workflow_pattern(\n        workflow_id=\"wf_stat_failed\",\n        problem_statement=\"Fix complex bug\",\n        team_config=self.sample_team_config,\n        gauntlet_config=self.sample_gauntlet_config,\n        success=False,\n        duration_seconds=600,\n        stages_completed=['content_analysis']\n    )\n\n    # Get statistics\n    stats = self.orchestrator.get_icr_statistics()\n\n    # Verify statistics structure\n    self.assertTrue(stats['icr_enabled'])\n    self.assertEqual(stats['total_workflows'], 4)\n    self.assertIn('overall_success_rate', stats)\n    self.assertIn('success_rates_by_type', stats)\n    self.assertIn('patterns_by_problem_type', stats)\n    self.assertIn('patterns_by_complexity', stats)\n    self.assertIn('average_duration_seconds', stats)\n    self.assertGreater(stats['overall_success_rate'], 0.5)  # 3/4 = 0.75\n\ndef test_clear_icr_patterns(self):\n    \"\"\"Test clearing ICR patterns\"\"\"\n    # Store some patterns\n    for i in range(5):\n        self.orchestrator.store_workflow_pattern(\n            workflow_id=f\"wf_clear_{i}\",\n            problem_statement=\"Implement feature\",\n            team_config=self.sample_team_config,\n            gauntlet_config=self.sample_gauntlet_config,\n            success=True,\n            duration_seconds=300,\n            stages_completed=['content_analysis', 'planning', 'decomposition']\n        )\n\n    # Verify patterns exist\n    stats_before = self.orchestrator.get_icr_statistics()\n    self.assertGreater(stats_before['total_workflows'], 0)\n\n    # Clear patterns\n    self.orchestrator.clear_icr_patterns()\n\n    # Verify patterns are cleared\n    stats_after = self.orchestrator.get_icr_statistics()\n    self.assertEqual(stats_after['total_workflows'], 0)\n\ndef test_predict_with_disabled_icr(self):\n    \"\"\"Test prediction when ICR is disabled\"\"\"\n    orchestrator = SGDWorkflowOrchestrator(enable_icr=False)\n\n    prediction = orchestrator.predict_workflow_success(\n        problem_statement=\"Implement feature\",\n        team_config=self.sample_team_config,\n        gauntlet_config=self.sample_gauntlet_config\n    )\n\n    # Should return default prediction\n    self.assertEqual(prediction['success_probability'], 0.5)\n    self.assertEqual(prediction['confidence'], 0.0)\n    self.assertEqual(prediction['reason'], 'ICR disabled')\n\n\nclass TestSGDWorkflowICRE2EWorkflow(unittest.TestCase):\n    \"\"\"End-to-end workflow tests for SGD Workflow ICR integration\"\"\"\n\n    def setUp(self):\n        \"\"\"Set up test fixtures\"\"\"\n        self.orchestrator = SGDWorkflowOrchestrator(\n            CREWAI_api_base=\"http://localhost:8002\",\n            openevolve_api_base=\"http://localhost:8000\",\n            enable_icr=True\n        )\n\n    def test_workflow_recommendation_improves_over_time(self):\n        \"\"\"Test that recommendations improve as patterns accumulate\"\"\"\n        # Initial recommendation\n        initial_rec = self.orchestrator.recommend_optimal_config(\n            problem_statement=\"Implement a machine learning system\"\n        )\n        initial_confidence = initial_rec.get('confidence', 0)\n\n        # Store successful patterns\n        for i in range(10):\n            self.orchestrator.store_workflow_pattern(\n                workflow_id=f\"wf_ml_{i}\",\n                problem_statement=\"Implement machine learning system for image recognition\",\n                team_config={\n                    'content_analyzer_team': 'ml_team',\n                    'planner_team': 'ml_planners',\n                    'solver_team': 'ml_solvers',\n                    'patcher_team': 'ml_patchers',\n                    'assembler_team': 'ml_assemblers'\n                },\n                gauntlet_config={\n                    'sub_problem_red_gauntlet': 'adaptive',\n                    'sub_problem_gold_gauntlet': 'hierarchical',\n                    'final_red_gauntlet': 'adaptive',\n                    'final_gold_gauntlet': 'hierarchical'\n                },\n                success=True,\n                duration_seconds=600,\n                stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']\n            )\n\n        # Updated recommendation\n        updated_rec = self.orchestrator.recommend_optimal_config(\n            problem_statement=\"Implement machine learning system\"\n        )\n        updated_confidence = updated_rec.get('confidence', 0)\n\n        # Confidence should improve with more patterns\n        self.assertGreater(updated_confidence, initial_confidence)\n\n        # Should recommend the ML team\n        self.assertEqual(updated_rec['content_analyzer_team'], 'ml_team')\n\n    def test_prediction_accuracy(self):\n        \"\"\"Test prediction accuracy over multiple workflows\"\"\"\n        problems = [\n            (\"Implement feature A\", True),\n            (\"Implement feature B\", True),\n            (\"Implement feature C\", True),\n            (\"Debug issue X\", True),\n            (\"Debug issue Y\", False),\n            (\"Design architecture\", True),\n            (\"Optimize performance\", True),\n        ]\n\n        for problem, success in problems:\n            self.orchestrator.store_workflow_pattern(\n                workflow_id=f\"wf_acc_{hash(problem)}\",\n                problem_statement=problem,\n                team_config={\n                    'content_analyzer_team': 'default_team',\n                    'planner_team': 'default_team',\n                    'solver_team': 'default_team',\n                    'patcher_team': 'default_team',\n                    'assembler_team': 'default_team'\n                },\n                gauntlet_config={\n                    'sub_problem_red_gauntlet': 'coherence',\n                    'sub_problem_gold_gauntlet': 'completeness',\n                    'final_red_gauntlet': 'feasibility',\n                    'final_gold_gauntlet': 'dependency'\n                },\n                success=success,\n                duration_seconds=300,\n                stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']\n            )\n\n        # Get statistics\n        stats = self.orchestrator.get_icr_statistics()\n\n        # Verify success rates\n        self.assertGreater(stats['total_workflows'], 0)\n        self.assertGreater(stats['overall_success_rate'], 0.5)\n\n    def test_complexity_patterns(self):\n        \"\"\"Test that complexity patterns are stored correctly\"\"\"\n        # Store patterns for different complexity levels\n        for i in range(3):\n            self.orchestrator.store_workflow_pattern(\n                workflow_id=f\"wf_complex_{i}\",\n                problem_statement=\"Fix bug\",  # Low complexity\n                team_config={\n                    'content_analyzer_team': 'team_a',\n                    'planner_team': 'team_a',\n                    'solver_team': 'team_a',\n                    'patcher_team': 'team_a',\n                    'assembler_team': 'team_a'\n                },\n                gauntlet_config={\n                    'sub_problem_red_gauntlet': 'coherence',\n                    'sub_problem_gold_gauntlet': 'completeness',\n                    'final_red_gauntlet': 'coherence',\n                    'final_gold_gauntlet': 'completeness'\n                },\n                success=True,\n                duration_seconds=100,\n                stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']\n            )\n\n        for i in range(3):\n            self.orchestrator.store_workflow_pattern(\n                workflow_id=f\"wf_complex_high_{i}\",\n                problem_statement=\"Design scalable distributed system with machine learning for real-time processing\",  # High complexity\n                team_config={\n                    'content_analyzer_team': 'team_b',\n                    'planner_team': 'team_b',\n                    'solver_team': 'team_b',\n                    'patcher_team': 'team_b',\n                    'assembler_team': 'team_b'\n                },\n                gauntlet_config={\n                    'sub_problem_red_gauntlet': 'adaptive',\n                    'sub_problem_gold_gauntlet': 'hierarchical',\n                    'final_red_gauntlet': 'adaptive',\n                    'final_gold_gauntlet': 'hierarchical'\n                },\n                success=True,\n                duration_seconds=900,\n                stages_completed=['content_analysis', 'planning', 'decomposition', 'solving', 'reassembly', 'verification']\n            )\n\n        # Get statistics\n        stats = self.orchestrator.get_icr_statistics()\n\n        # Verify patterns exist for different complexity levels\n        self.assertGreater(stats['total_workflows'], 0)\n        self.assertIn('patterns_by_complexity', stats)\n\n\nif __name__ == '__main__':\n    unittest.main()\n