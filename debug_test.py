#!/usr/bin/env python3
"""Debug test for generate_solution_for_sub_problem"""
import sys
sys.path.insert(0, 'tests')

from unittest.mock import patch, MagicMock
from ui_shim import ui as st

# Mock UI shim
st.session_state = MagicMock()
st.session_state.edited_sub_problems = {}
st.info = MagicMock()
st.warning = MagicMock()
st.error = MagicMock()
st.success = MagicMock()

# Import after mocking
from workflow_engine import generate_solution_for_sub_problem
from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, Team, ModelConfig, GauntletDefinition, GauntletRoundRule

# Create test data
solver_team = Team(
    name='Solver',
    role='Blue',
    members=[ModelConfig(model_id='test-model', api_key='test-key')]
)
sub_problem = SubProblem(id='sub_1.1', description='Solve X')
workflow_state = WorkflowState(
    workflow_id='test',
    workflow_type='test_type',
    problem_statement='test',
    current_stage='test',
    decomposition_plan=DecompositionPlan(
        problem_statement='test',
        analyzed_context={},
        sub_problems=[]
    ),
    mdap_enabled=False,
    maker_enabled=False
)

rounds = [GauntletRoundRule(
    round_number=1,
    quorum_required_approvals=1,
    quorum_from_panel_size=1,
    min_overall_confidence=0.5
)]
solver_gauntlet = GauntletDefinition(
    name='SolverGauntlet',
    team_name='Solver',
    rounds=rounds,
    generation_mode='single_candidate'
)

print(f'SubProblem evolution_mode: {sub_problem.ai_suggested_evolution_mode}')
print(f'Gauntlet generation_mode: {solver_gauntlet.generation_mode}')
print(f'ADAPTIVE_MDAP_AVAILABLE: {getattr(sys.modules["workflow_engine"], "ADAPTIVE_MDAP_AVAILABLE", "NOT_FOUND")}')

with patch('workflow_engine._request_openai_compatible_chat') as mock_chat:
    mock_chat.return_value = 'Generated solution content.'

    # Patch also at llm_utils level
    with patch('llm_utils._request_openai_compatible_chat', return_value='Generated solution content.'):
        result = generate_solution_for_sub_problem(sub_problem, solver_team, {}, workflow_state, solver_gauntlet, emit_ui=False)

        print(f'Result: {repr(result)}')
        print(f'Mock called: {mock_chat.called}')
        print(f'Mock call count: {mock_chat.call_count}')
        print(f'Mock return value: {repr(mock_chat.return_value)}')

        if mock_chat.called:
            print(f'Mock calls: {mock_chat.call_args_list}')

        # Test passes if result is not None
        if result is None:
            print('\nFAILED: Result is None')
            sys.exit(1)
        elif result == 'Generated solution content.':
            print('\nPASSED: Got expected result')
            sys.exit(0)
        else:
            print(f'\nFAILED: Got unexpected result: {repr(result)}')
            sys.exit(1)
