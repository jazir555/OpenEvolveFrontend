import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from workflow_structures import (
    ModelConfig, Team, GauntletRoundRule, GauntletDefinition,
    SubProblem, DecompositionPlan, SolutionAttempt, CritiqueReport,
    VerificationReport, WorkflowState
)
from workflow_engine import run_sovereign_workflow
from team_manager import TeamManager
from gauntlet_manager import GauntletManager

class TestSovereignWorkflow(unittest.TestCase):

    def setUp(self):
        """Set up mock data for teams and gauntlets."""
        self.mock_model_config = ModelConfig(model_id="mock-model", api_key="mock-key")

        # Create mock teams
        self.content_analyzer_team = Team(name="Content-Analyzers", role="Blue", members=[self.mock_model_config])
        self.planner_team = Team(name="Planners", role="Blue", members=[self.mock_model_config])
        self.solver_team = Team(name="Solvers", role="Blue", members=[self.mock_model_config])
        self.patcher_team = Team(name="Patchers", role="Blue", members=[self.mock_model_config])
        self.assembler_team = Team(name="Assemblers", role="Blue", members=[self.mock_model_config])
        self.red_team = Team(name="Red-Team", role="Red", members=[self.mock_model_config])
        self.gold_team = Team(name="Gold-Team", role="Gold", members=[self.mock_model_config])

        # Create mock gauntlets
        mock_round = GauntletRoundRule(round_number=1, quorum_required_approvals=1, quorum_from_panel_size=1)
        self.blue_gauntlet = GauntletDefinition(name="Blue-Gen-Gauntlet", team_name="Solvers", rounds=[mock_round], generation_mode="single_candidate")
        self.red_gauntlet = GauntletDefinition(name="Red-Gauntlet", team_name="Red-Team", rounds=[mock_round])
        self.gold_gauntlet = GauntletDefinition(name="Gold-Gauntlet", team_name="Gold-Team", rounds=[mock_round])

        # Mock TeamManager and GauntletManager
        self.team_manager = TeamManager()
        self.gauntlet_manager = GauntletManager()
        
        self.team_manager.get_team = MagicMock(side_effect=self._get_mock_team)
        self.gauntlet_manager.get_gauntlet = MagicMock(side_effect=self._get_mock_gauntlet)

    def _get_mock_team(self, name):
        teams = {
            "Content-Analyzers": self.content_analyzer_team,
            "Planners": self.planner_team,
            "Solvers": self.solver_team,
            "Patchers": self.patcher_team,
            "Assemblers": self.assembler_team,
            "Red-Team": self.red_team,
            "Gold-Team": self.gold_team,
        }
        return teams.get(name)

    def _get_mock_gauntlet(self, name):
        gauntlets = {
            "Blue-Gen-Gauntlet": self.blue_gauntlet,
            "Red-Gauntlet": self.red_gauntlet,
            "Gold-Gauntlet": self.gold_gauntlet,
        }
        return gauntlets.get(name)

    @patch('workflow_engine._request_openai_compatible_chat')
    @patch('streamlit.info')
    @patch('streamlit.success')
    @patch('streamlit.warning')
    @patch('streamlit.error')
    def test_full_workflow_happy_path(self, mock_st_error, mock_st_warning, mock_st_success, mock_st_info, mock_llm_call):
        """Test the full Sovereign-Grade Decomposition Workflow on a happy path."""
        
        # --- Mock LLM Responses ---
        # Stage 0: Content Analysis
        mock_content_analysis_response = json.dumps({
            "domain": "Software Development",
            "keywords": ["python", "cli", "file system"],
            "summary": "A simple file listing tool."
        })
        
        # Stage 1: AI Decomposition
        mock_decomposition_response = json.dumps([
            {"id": "sub_1.0", "description": "List files in a directory", "dependencies": []}
        ])

        # Stage 3: Solution Generation
        mock_solution_response = "import os; print(os.listdir('.'))"

        # Stage 3: Red Team & Gold Team Gauntlets (approving)
        mock_gauntlet_approve_response = json.dumps({"score": 0.9, "justification": "Looks good."})

        # Stage 4: Reassembly (using a simple pass-through for the test)
        # In the real workflow, this is a more complex OpenEvolve call.
        # We mock the result of that call.
        mock_reassembly_content = f"```python\n{mock_solution_response}\n```"
        mock_reassembly_result = {
            "success": True,
            "best_solution": mock_reassembly_content
        }

        mock_llm_call.side_effect = [
            mock_content_analysis_response,
            mock_decomposition_response,
            mock_solution_response,
            mock_gauntlet_approve_response, # Sub-problem Red Team
            mock_gauntlet_approve_response, # Sub-problem Gold Team
            mock_gauntlet_approve_response, # Final Red Team
            mock_gauntlet_approve_response, # Final Gold Team
        ]

        # --- Workflow Setup ---
        workflow_state = WorkflowState(
            workflow_id="test_workflow_123",
            current_stage="INITIALIZING",
            problem_statement="Create a python script to list files in the current directory.",
            content_analyzer_team=self.content_analyzer_team,
            planner_team=self.planner_team,
            solver_team=self.solver_team,
            patcher_team=self.patcher_team,
            assembler_team=self.assembler_team,
            solver_generation_gauntlet=self.blue_gauntlet,
            sub_problem_red_gauntlet=self.red_gauntlet,
            sub_problem_gold_gauntlet=self.gold_gauntlet,
            final_red_gauntlet=self.red_gauntlet,
            final_gold_gauntlet=self.gold_gauntlet,
            max_refinement_loops=1
        )
        
        # Mock the managers and the evolution call used inside the workflow
        with patch('workflow_engine.TeamManager', return_value=self.team_manager), \
             patch('workflow_engine.GauntletManager', return_value=self.gauntlet_manager), \
             patch('workflow_engine.run_unified_evolution', return_value=mock_reassembly_result):

            # --- Execute Workflow Stages ---
            # The workflow is designed to be called repeatedly. We simulate this by looping.
            max_loops = 20 # Safety break
            loop_count = 0
            while loop_count < max_loops:
                # Stop looping if workflow is finished
                if workflow_state.status in ["completed", "failed"]:
                    break

                # Simulate the Streamlit UI interaction for manual review
                if workflow_state.current_stage == "Manual Review & Override" and workflow_state.status == "awaiting_user_input":
                    # This block simulates the user clicking "Approve" in the UI
                    workflow_state.status = "running" # Set status back to running to continue the loop
                    workflow_state.current_stage = "Sub-Problem Solving Loop"
                    
                    # We also need to populate the team/gauntlet names that the UI would normally handle during review.
                    for sp in workflow_state.decomposition_plan.sub_problems:
                        sp.solver_team_name = self.solver_team.name
                        sp.red_team_gauntlet_name = self.red_gauntlet.name
                        sp.gold_team_gauntlet_name = self.gold_gauntlet.name
                    
                    loop_count += 1
                    continue

                run_sovereign_workflow(
                    workflow_state=workflow_state,
                    content_analyzer_team=self.content_analyzer_team,
                    planner_team=self.planner_team,
                    solver_team=self.solver_team,
                    patcher_team=self.patcher_team,
                    assembler_team=self.assembler_team,
                    sub_problem_red_gauntlet=self.red_gauntlet,
                    sub_problem_gold_gauntlet=self.gold_gauntlet,
                    final_red_gauntlet=self.red_gauntlet,
                    final_gold_gauntlet=self.gold_gauntlet,
                    max_refinement_loops=1
                )
                loop_count += 1

        # --- Assertions ---
        self.assertEqual(workflow_state.status, "completed", f"Workflow failed to complete. Final status: {workflow_state.status}, Stage: {workflow_state.current_stage}")
        self.assertIsNotNone(workflow_state.final_solution)
        self.assertIn("os.listdir", workflow_state.final_solution.content)
        self.assertEqual(len(workflow_state.sub_problem_solutions), 1)
        mock_st_error.assert_not_called()


    @patch('workflow_engine._request_openai_compatible_chat')
    @patch('streamlit.info')
    @patch('streamlit.success')
    @patch('streamlit.warning')
    @patch('streamlit.error')
    def test_self_healing_loop(self, mock_st_error, mock_st_warning, mock_st_success, mock_st_info, mock_llm_call):
        """Test the self-healing loop when a sub-problem is initially rejected."""
        
        log_file = "test_trace.log"
        if os.path.exists(log_file):
            os.remove(log_file)

        # --- Mock LLM Responses ---
        mock_responses = [
            json.dumps({"summary": "A tool to write to a file."}), # 1. analysis
            json.dumps([{"id": "sub_1.0", "description": "Write content to a file", "dependencies": []}]), # 2. decomp
            "with open('file.txt', 'r') as f: f.write('content')", # 3. solve
            json.dumps({"score": 0.9, "justification": "Looks plausible."}), # 4. red
            json.dumps({"score": 0.2, "justification": "Incorrect file mode."}), # 5. gold (rejects)
            "with open('file.txt', 'w') as f: f.write('content')", # 6. patch
            json.dumps({"score": 0.9, "justification": "Looks plausible."}), # 7. red
            json.dumps({"score": 0.9, "justification": "Looks good."}), # 8. gold
            json.dumps({"score": 0.9, "justification": "Final check good."}), # 9. final red
            json.dumps({"score": 0.9, "justification": "Final check good."}), # 10. final gold
        ]

        call_counter = {"count": 0}
        def llm_mock_side_effect(*args, **kwargs):
            call_num = call_counter["count"]
            with open(log_file, "a") as f:
                f.write(f"LLM Call {call_num + 1}:\n")
                f.write(f"  - Model: {kwargs.get('model')}\n")
                #f.write(f"  - Messages: {kwargs.get('messages')}\n\n") # This can be too verbose
            
            response = mock_responses[call_num]
            call_counter["count"] += 1
            return response

        mock_llm_call.side_effect = llm_mock_side_effect

        # Reassembly mock result
        mock_reassembly_result = {"success": True, "best_solution": "final solution content"}

        # --- Workflow Setup ---
        workflow_state = WorkflowState(
            workflow_id="test_workflow_healing_123",
            current_stage="INITIALIZING",
            problem_statement="Create a python script to write to a file.",
            content_analyzer_team=self.content_analyzer_team,
            planner_team=self.planner_team,
            solver_team=self.solver_team,
            patcher_team=self.patcher_team,
            assembler_team=self.assembler_team,
            solver_generation_gauntlet=self.blue_gauntlet,
            sub_problem_red_gauntlet=self.red_gauntlet,
            sub_problem_gold_gauntlet=self.gold_gauntlet,
            final_red_gauntlet=self.red_gauntlet,
            final_gold_gauntlet=self.gold_gauntlet,
            max_refinement_loops=1
        )
        
        with patch('workflow_engine.TeamManager', return_value=self.team_manager), \
             patch('workflow_engine.GauntletManager', return_value=self.gauntlet_manager), \
             patch('workflow_engine.run_unified_evolution', return_value=mock_reassembly_result):

            max_loops = 20
            loop_count = 0
            while loop_count < max_loops:
                if workflow_state.status in ["completed", "failed"]:
                    break

                if workflow_state.current_stage == "Manual Review & Override" and workflow_state.status == "awaiting_user_input":
                    workflow_state.status = "running"
                    workflow_state.current_stage = "Sub-Problem Solving Loop"
                    for sp in workflow_state.decomposition_plan.sub_problems:
                        sp.solver_team_name = self.solver_team.name
                        sp.red_team_gauntlet_name = self.red_gauntlet.name
                        sp.gold_team_gauntlet_name = self.gold_gauntlet.name
                    loop_count += 1
                    continue

                run_sovereign_workflow(
                    workflow_state=workflow_state,
                    content_analyzer_team=self.content_analyzer_team,
                    planner_team=self.planner_team,
                    solver_team=self.solver_team,
                    patcher_team=self.patcher_team,
                    assembler_team=self.assembler_team,
                    sub_problem_red_gauntlet=self.red_gauntlet,
                    sub_problem_gold_gauntlet=self.gold_gauntlet,
                    final_red_gauntlet=self.red_gauntlet,
                    final_gold_gauntlet=self.gold_gauntlet,
                    max_refinement_loops=1
                )
                loop_count += 1

        # --- Assertions ---
        # We still assert the final state, even though the main goal is the log file
        self.assertEqual(call_counter["count"], 10)
        self.assertEqual(workflow_state.status, "completed", f"Workflow failed to complete. Final status: {workflow_state.status}, Stage: {workflow_state.current_stage}")
        mock_st_error.assert_not_called()

if __name__ == '__main__':
    unittest.main()
