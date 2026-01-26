
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import dataclasses
from typing import Dict, Any, List, Optional

# Mocking external modules that might require API keys or heavy dependencies
mock_llm = MagicMock()
# Return a valid JSON string that satisfies MDAP and MAKER parsing
mock_llm._request_openai_compatible_chat.return_value = '{"action": "test_action", "is_final": true, "rationale": "mock rationale"}'
sys.modules['llm_utils'] = mock_llm
sys.modules['ace_steer_integration'] = MagicMock()
sys.modules['hephaestus'] = MagicMock()

# Mock openevolve_imports to force fallback to direct imports (bypass broken mdap_maker_complete detection)
mock_oe_imports = MagicMock()
mock_oe_imports.MDAP_ENGINE_AVAILABLE = False
mock_oe_imports.MAKER_ENGINE_AVAILABLE = False
sys.modules['openevolve_imports'] = mock_oe_imports

# We don't mock workflow_engine, mdap_engine, maker_engine to avoid inheritance issues with Mocks
# But we mock dependencies they use

# Define mocks for engine classes we will patch later
class MockMDAPOrchestrator:
    def __init__(self, team, config):
        pass
    def execute_task(self, task):
        # Return a mock result
        mock_result = MagicMock()
        mock_result.metric = {"steps_completed": 1}
        mock_result.step_results = {}
        return mock_result

class MockMAKEREngine:
    def __init__(self, team, config):
        pass
    def solve(self, initial_state, step_builder, apply_action, stop_condition):
        mock_result = MagicMock()
        mock_result.state.history = []
        mock_result.state.current_state = {"actions": ["Action 1"]}
        mock_result.state.last_action = "Action 1"
        return mock_result

# Setup mocks for imports in the target file
with patch.dict(sys.modules, {
    'hephaestus_integration': MagicMock(),
    'bubblelabs_hephaestus_bridge': MagicMock(),
}):
    # Now import the class to test
    sys.path.append('c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend')
    from openevolve_workflow_manager_integrated import OpenEvolveWorkflowManager
    from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, Team


# ... (imports remain)

# Plain script execution
if __name__ == '__main__':
    print("Initializing Manager...")
    try:
        manager = OpenEvolveWorkflowManager(
            # config_path="dummy_config.yaml", # Removed
            enable_hephaestus=True
        )
        
        # Mock the hephaestus manager instance
        manager.hephaestus_manager = MagicMock()
        
        # Setup basic workflow state
        workflow_state = MagicMock(spec=WorkflowState)
        workflow_state.hephaestus_workflow_id = "epic-123"
        workflow_state.solver_team = MagicMock() # Remove spec to avoid attribute issues
        workflow_state.solver_team.name = "TestSolverTeam"
        # Add a dummy member mock
        dummy_member = MagicMock()
        dummy_member.model_id = "gpt-4"
        workflow_state.solver_team.members = [dummy_member]
        workflow_state.solved_sub_problem_ids = set()
        workflow_state.mdap_config = {}
        workflow_state.maker_config = {}
        
        # Sub-problem
        sub_problem = SubProblem(
            id="sp-1",
            description="Test Subproblem"
        )
        
        decomposition_plan = DecompositionPlan(
            problem_statement="Test Problem",
            analyzed_context={},
            sub_problems=[sub_problem]
        )
        
        print("\n--- Testing MDAP ---")
        workflow_state.mdap_enabled = True
        workflow_state.maker_enabled = False
        
        with patch('openevolve_workflow_manager_integrated.MDAPOrchestrator', side_effect=MockMDAPOrchestrator) as MockOrch:
            with patch('openevolve_workflow_manager_integrated.MDAPTask') as MockTask:
                 with patch('openevolve_workflow_manager_integrated.MDAP_MAKER_AVAILABLE', True):
                    solutions = manager._solve_sub_problems(workflow_state, decomposition_plan)
        
        print("MDAP Done. Verifying...")
        manager.hephaestus_manager.sync_mdap_task.assert_called()
        print("MDAP Success.")

        print("\n--- Testing MAKER ---")
        workflow_state.mdap_enabled = False
        workflow_state.maker_enabled = True
        
        with patch('openevolve_workflow_manager_integrated.MAKEREngine', side_effect=MockMAKEREngine) as MockEng:
            with patch('openevolve_workflow_manager_integrated.MakerConfig') as MockConfig:
                with patch('openevolve_workflow_manager_integrated.MDAP_MAKER_AVAILABLE', True):
                    solutions = manager._solve_sub_problems(workflow_state, decomposition_plan)

        print("MAKER Done. Verifying...")
        manager.hephaestus_manager.sync_maker_run.assert_called()
        print("MAKER Success.")
        
    except Exception as e:
        import traceback
        with open("error.log", "w") as f:
            f.write(f"CRITICAL FAILURE: {e}\n")
            traceback.print_exc(file=f)
        sys.exit(1)
