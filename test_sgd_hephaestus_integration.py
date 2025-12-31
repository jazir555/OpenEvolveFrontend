"""
Test Suite for Sovereign-Grade Decomposition Workflow - Hephaestus Integration

This module tests the complete integration between OpenEvolve's Sovereign-Grade Decomposition
workflow and the Hephaestus agentic framework as specified in @Decomposition_Workflow.md.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the frontend directory to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from workflow_structures import (
    WorkflowState, SubProblem, SolutionAttempt, CritiqueReport, 
    VerificationReport, Team, GauntletDefinition, ModelConfig, DecompositionPlan
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from workflow_engine import run_sovereign_workflow
from sovereign_decomposition_hephaestus_integration import (
    SovereignDecompositionHephaestusIntegration,
    initialize_sgd_hephaestus_integration,
    get_sgd_hephaestus_integration,
    SGDStage
)


class TestSgdHephaestusIntegration(unittest.TestCase):
    """Test cases for the Sovereign-Grade Decomposition - Hephaestus integration"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_base = "http://localhost:8080"
        self.api_key = "test_key"
        self.project_id = "test_project"
        
        # Initialize the integration
        success = initialize_sgd_hephaestus_integration(self.api_base, self.api_key, self.project_id)
        self.assertTrue(success, "Failed to initialize SGD-Hephaestus integration")
        
        self.integration = get_sgd_hephaestus_integration()
        self.assertIsNotNone(self.integration, "Integration instance should be created")
        
        # Create a sample workflow state
        self.workflow_state = self._create_sample_workflow_state()
        
    def _create_sample_workflow_state(self) -> WorkflowState:
        """Create a sample workflow state for testing"""
        team_manager = TeamManager()
        gauntlet_manager = GauntletManager()
        
        # Create sample teams
        content_analyzer_team = Team(
            name="ContentAnalyzer",
            role="Blue",
            members=[ModelConfig(model_id="gpt-4o", api_key="test", api_base="https://api.openai.com/v1")]
        )
        planner_team = Team(
            name="Planner",
            role="Blue",
            members=[ModelConfig(model_id="gpt-4o", api_key="test", api_base="https://api.openai.com/v1")]
        )
        solver_team = Team(
            name="Solver",
            role="Blue",
            members=[ModelConfig(model_id="gpt-4o", api_key="test", api_base="https://api.openai.com/v1")]
        )
        patcher_team = Team(
            name="Patcher",
            role="Blue",
            members=[ModelConfig(model_id="gpt-4o", api_key="test", api_base="https://api.openai.com/v1")]
        )
        assembler_team = Team(
            name="Assembler",
            role="Blue",
            members=[ModelConfig(model_id="gpt-4o", api_key="test", api_base="https://api.openai.com/v1")]
        )
        
        # Create sample gauntlets
        red_gauntlet = GauntletDefinition(
            name="RedTeam",
            team_name="RedTeam",
            rounds=[]
        )
        gold_gauntlet = GauntletDefinition(
            name="GoldTeam",
            team_name="GoldTeam",
            rounds=[]
        )
        
        # Create sample decomposition plan
        sub_problems = [
            SubProblem(
                id="sub_1",
                description="Solve the first part of the complex problem",
                dependencies=[],
                ai_suggested_evolution_mode="standard",
                ai_suggested_complexity_score=5,
                ai_suggested_evaluation_prompt="Evaluate the solution quality",
                solver_team_name="Solver",
                red_team_gauntlet_name="RedTeam",
                gold_team_gauntlet_name="GoldTeam"
            ),
            SubProblem(
                id="sub_2",
                description="Solve the second part of the complex problem",
                dependencies=["sub_1"],
                ai_suggested_evolution_mode="adversarial",
                ai_suggested_complexity_score=7,
                ai_suggested_evaluation_prompt="Test the solution security",
                solver_team_name="Solver",
                red_team_gauntlet_name="RedTeam",
                gold_team_gauntlet_name="GoldTeam"
            )
        ]
        
        decomposition_plan = DecompositionPlan(
            problem_statement="Complex problem requiring decomposition",
            analyzed_context={
                "domain": "Software Development",
                "keywords": ["decomposition", "problem solving"],
                "estimated_complexity": 8
            },
            sub_problems=sub_problems
        )

        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id="test_workflow_123",
            workflow_type="sovereign_decomposition",  # This matches the enum value
            problem_statement="Test complex problem",
            current_stage="INITIALIZING",
            decomposition_plan=decomposition_plan,
            content_analyzer_team=content_analyzer_team,
            planner_team=planner_team,
            solver_team=solver_team,
            patcher_team=patcher_team,
            assembler_team=assembler_team,
            sub_problem_red_gauntlet=red_gauntlet,
            sub_problem_gold_gauntlet=gold_gauntlet,
            final_red_gauntlet=red_gauntlet,
            final_gold_gauntlet=gold_gauntlet,
            max_refinement_loops=3
        )

        return workflow_state
    
    def test_initialization(self):
        """Test that the integration initializes correctly"""
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.integration_manager)
    
    def test_workflow_initialization(self):
        """Test that a workflow can be initialized in Hephaestus"""
        with patch.object(self.integration.integration_manager.client, 'create_ticket', 
                         return_value="ticket_123"):
            result = self.integration.initialize_sovereign_workflow(self.workflow_state)
            self.assertTrue(result, "Workflow initialization should succeed")
    
    def test_create_subproblem_ticket(self):
        """Test that sub-problem tickets can be created"""
        sub_problem = self.workflow_state.decomposition_plan.sub_problems[0]
        
        with patch.object(self.integration.integration_manager.client, 'create_ticket', 
                         return_value="ticket_456"):
            ticket_id = self.integration._create_subproblem_ticket(self.workflow_state, sub_problem)
            self.assertEqual(ticket_id, "ticket_456", "Ticket ID should match expected value")
    
    def test_solution_synchronization(self):
        """Test that solutions are synchronized to Hephaestus tickets"""
        # Mock the ticket ID mapping
        self.workflow_state.id_to_ticket_id_map["sub_1"] = "ticket_789"
        
        solution = SolutionAttempt(
            sub_problem_id="sub_1",
            content="Sample solution content",
            generated_by_model="gpt-4o",
            timestamp=1234567890.0,
            history=[]
        )
        
        with patch.object(self.integration.integration_manager.client, 'update_ticket', 
                         return_value=True):
            result = self.integration.sync_solution_to_hephaestus_ticket(
                self.workflow_state, "sub_1", solution
            )
            self.assertTrue(result, "Solution synchronization should succeed")
    
    def test_critique_synchronization(self):
        """Test that critiques are synchronized to Hephaestus tickets"""
        # Mock the ticket ID mapping
        self.workflow_state.id_to_ticket_id_map["sub_1"] = "ticket_789"
        
        critique = CritiqueReport(
            solution_attempt_id="solution_123",
            gauntlet_name="RedTeam",
            is_approved=False,
            reports_by_judge=[
                {
                    "model_id": "gpt-4o",
                    "score": 0.3,
                    "justification": "This solution has significant flaws",
                    "targeted_feedback": []
                }
            ],
            summary="Solution has critical security flaws"
        )
        
        with patch.object(self.integration.integration_manager.client, 'update_ticket', 
                         return_value=True):
            result = self.integration.sync_critique_to_hephaestus_ticket(
                self.workflow_state, "sub_1", critique
            )
            self.assertTrue(result, "Critique synchronization should succeed")
    
    def test_verification_synchronization(self):
        """Test that verifications are synchronized to Hephaestus tickets"""
        # Mock the ticket ID mapping
        self.workflow_state.id_to_ticket_id_map["sub_1"] = "ticket_789"
        
        verification = VerificationReport(
            solution_attempt_id="solution_123",
            gauntlet_name="GoldTeam",
            is_approved=True,
            reports_by_judge=[
                {
                    "model_id": "gpt-4o",
                    "score": 0.9,
                    "justification": "Solution meets all requirements",
                    "targeted_feedback": []
                }
            ],
            average_score=0.9,
            score_variance=0.1,
            summary="Solution is high quality and complete"
        )
        
        with patch.object(self.integration.integration_manager.client, 'update_ticket', 
                         return_value=True):
            result = self.integration.sync_verification_to_hephaestus_ticket(
                self.workflow_state, "sub_1", verification
            )
            self.assertTrue(result, "Verification synchronization should succeed")
    
    def test_status_synchronization(self):
        """Test that solution statuses are synchronized to Hephaestus tickets"""
        # Mock the ticket ID mapping
        self.workflow_state.id_to_ticket_id_map["sub_1"] = "ticket_789"
        
        with patch.object(self.integration.integration_manager.client, 'update_ticket', 
                         return_value=True):
            result = self.integration.sync_solution_status_to_hephaestus_ticket(
                self.workflow_state, "sub_1", "solved", "Sample solution content"
            )
            self.assertTrue(result, "Status synchronization should succeed")
    
    def test_team_to_agent_mapping(self):
        """Test that OpenEvolve teams are correctly mapped to Hephaestus agents"""
        # Create a test team
        test_team = Team(
            name="Security Red Team",
            role="Red",
            members=[ModelConfig(model_id="gpt-4o", api_key="test", api_base="https://api.openai.com/v1")]
        )
        
        sub_problem = self.workflow_state.decomposition_plan.sub_problems[0]
        
        agent_name = self.integration.map_openevolve_team_to_hephaestus_agent(test_team, sub_problem)
        self.assertIsInstance(agent_name, str, "Agent name should be a string")
        self.assertGreater(len(agent_name), 0, "Agent name should not be empty")
    
    def test_openevolve_metrics_extraction(self):
        """Test that OpenEvolve metrics can be extracted from Hephaestus"""
        with patch.object(self.integration.integration_manager.client, 'get_workflow_tickets', 
                         return_value=[
                             {
                                 "id": "ticket_1",
                                 "status": "done",
                                 "assigned_agent_id": "ImplementationAgent",
                                 "created_at": "2023-01-01T00:00:00",
                                 "updated_at": "2023-01-01T01:00:00"
                             },
                             {
                                 "id": "ticket_2", 
                                 "status": "in_progress",
                                 "assigned_agent_id": "ValidationAgent",
                                 "created_at": "2023-01-01T00:00:00",
                                 "updated_at": "2023-01-01T00:30:00"
                             }
                         ]):
            metrics = self.integration.get_openevolve_metrics_from_hephaestus_agents("test_workflow_123")
            
            self.assertIn("agent_performance", metrics)
            self.assertIn("task_completion_rate", metrics)
            self.assertIn("average_resolution_time", metrics)
    
    def test_self_healing_trigger(self):
        """Test that self-healing is triggered based on agent discoveries"""
        self.workflow_state.hephaestus_workflow_id = "hephaestus_workflow_456"
        
        with patch.object(self.integration.integration_manager.client, 'get_workflow_tickets',
                         return_value=[
                             {
                                 "id": "ticket_1",
                                 "status": "blocked",
                                 "description": "Issue discovered by agent"
                             }
                         ]):
            result = self.integration.trigger_self_healing_from_agent_discoveries(self.workflow_state)
            self.assertTrue(result, "Self-healing should be triggered when issues are found")
    
    def test_close_workflow(self):
        """Test that workflows can be closed in Hephaestus"""
        self.workflow_state.hephaestus_workflow_id = "hephaestus_workflow_789"
        self.workflow_state.status = "completed"
        
        with patch.object(self.integration.integration_manager.client, 'update_ticket', 
                         return_value=True):
            result = self.integration.close_workflow_in_hephaestus(self.workflow_state)
            self.assertTrue(result, "Workflow closing should succeed")


class TestIntegrationInitialization(unittest.TestCase):
    """Test cases for integration initialization"""
    
    def test_initialize_integration(self):
        """Test that the integration can be initialized successfully"""
        api_base = "http://localhost:8080"
        api_key = "test_key" 
        project_id = "test_project"
        
        success = initialize_sgd_hephaestus_integration(api_base, api_key, project_id)
        self.assertTrue(success, "Integration should initialize successfully")
        
        integration = get_sgd_hephaestus_integration()
        self.assertIsNotNone(integration, "Integration should be available after initialization")


def run_tests():
    """Run all tests in the suite"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    print("Starting Sovereign-Grade Decomposition - Hephaestus Integration Tests...")
    print("=" * 60)
    
    # Run tests
    run_tests()
    
    print("=" * 60)
    print("Integration tests completed!")