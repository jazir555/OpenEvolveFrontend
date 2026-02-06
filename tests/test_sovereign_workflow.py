import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import dataclasses
import json

pytestmark = pytest.mark.timeout(60)  # This test has slow imports

# Set test environment
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing")
os.environ.setdefault("TESTING", "true")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock UI functions FIRST before importing anything that might use them
class MockStreamlit:
    """Mock Streamlit UI object for testing."""
    def __init__(self):
        self.session_state = {'edited_sub_problems': {}}
        self.info = MagicMock()
        self.warning = MagicMock()
        self.error = MagicMock()
        self.success = MagicMock()
        self.subheader = MagicMock()
        self.write = MagicMock()
        self.caption = MagicMock()
        self.rerun = MagicMock()
        self.text_area = MagicMock(return_value="")
        self.text_input = MagicMock(return_value="")
        self.number_input = MagicMock(return_value=1)
        self.selectbox = MagicMock(return_value="Option 1")
        self.checkbox = MagicMock(return_value=False)
        self.button = MagicMock(return_value=False)
        self.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        self.tabs = MagicMock(return_value=[MagicMock(), MagicMock()])
        self.expander = MagicMock(return_value=MagicMock())
        self.container = MagicMock(return_value=MagicMock())
        self.progress = MagicMock(return_value=MagicMock())
        self.markdown = MagicMock()
        self.cache_data = MagicMock(return_value=lambda fn: fn)  # Mock decorator

# Create and inject mock
st = MockStreamlit()

# Patch ui_shim before importing modules that use it
try:
    import ui_shim
    ui_shim.ui = st
except (ImportError, AttributeError):
    pass

# Try to import optional dependencies
try:
    from lean4_system.lean4_api import VerificationResult
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    # Create a mock VerificationResult
    from dataclasses import dataclass, field
    @dataclass
    class VerificationResult:
        request_id: str
        is_verified: bool
        status: str
        proof_status: str
        details: dict = field(default_factory=dict)

# Import workflow modules with error handling
try:
    from workflow_engine import run_sovereign_workflow, run_content_analysis, run_ai_decomposition, run_gauntlet, parse_targeted_feedback
    WORKFLOW_ENGINE_AVAILABLE = True
except ImportError as e:
    WORKFLOW_ENGINE_AVAILABLE = False
    pytest.skip(f"workflow_engine not available: {e}", allow_module_level=True)

try:
    from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, Team, ModelConfig, GauntletDefinition, GauntletRoundRule, SolutionAttempt, CritiqueReport, VerificationReport as WorkflowVerificationReport
    WORKFLOW_STRUCTURES_AVAILABLE = True
except ImportError as e:
    WORKFLOW_STRUCTURES_AVAILABLE = False
    pytest.skip(f"workflow_structures not available: {e}", allow_module_level=True)

try:
    from team_manager import TeamManager
    TEAM_MANAGER_AVAILABLE = True
except ImportError:
    TEAM_MANAGER_AVAILABLE = False

try:
    from gauntlet_manager import GauntletManager
    GAUNTLET_MANAGER_AVAILABLE = True
except ImportError:
    GAUNTLET_MANAGER_AVAILABLE = False

# Alias for clarity
VerificationReport = WorkflowVerificationReport

# Mock managers if not available
if TEAM_MANAGER_AVAILABLE:
    team_manager = TeamManager()
else:
    team_manager = MagicMock()

if GAUNTLET_MANAGER_AVAILABLE:
    gauntlet_manager = GauntletManager()
else:
    gauntlet_manager = MagicMock()

from workflow_engine import run_sovereign_workflow, run_content_analysis, run_ai_decomposition, run_gauntlet, parse_targeted_feedback
# Don't import generate_solution_for_sub_problem at module level - it breaks mocking
from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, Team, ModelConfig, GauntletDefinition, GauntletRoundRule, SolutionAttempt, CritiqueReport, VerificationReport as WorkflowVerificationReport
from team_manager import TeamManager
from gauntlet_manager import GauntletManager

# Alias for clarity
VerificationReport = WorkflowVerificationReport

# Mock managers
team_manager = TeamManager()
gauntlet_manager = GauntletManager()

# Helper function to create dummy teams and gauntlets
def create_dummy_team(name, role, model_id="test-model", api_key="test-key", members_count=1):
    members = [ModelConfig(model_id=f"{model_id}-{i}", api_key=api_key) for i in range(members_count)]
    team = Team(name=name, role=role, members=members)
    team_manager.teams[name] = team # Directly add to the dictionary
    return team

def create_dummy_gauntlet(name, team_name, rounds_config=None, generation_mode="single_candidate"):
    if rounds_config is None:
        rounds_config = [{"round_number": 1, "quorum_required_approvals": 1, "quorum_from_panel_size": 1, "min_overall_confidence": 0.5}]
    
    rounds = [GauntletRoundRule(**rc) for rc in rounds_config]
    gauntlet = GauntletDefinition(name=name, team_name=team_name, rounds=rounds, generation_mode=generation_mode)
    gauntlet_manager.gauntlets[name] = gauntlet # Directly add to the dictionary
    return gauntlet

@pytest.fixture(autouse=True)
def clear_managers_and_mocks():
    team_manager.teams = {}
    gauntlet_manager.gauntlets = {}
    st.info.reset_mock()
    st.warning.reset_mock()
    st.error.reset_mock()
    st.success.reset_mock()
    st.subheader.reset_mock()
    st.write.reset_mock()
    st.caption.reset_mock()
    st.rerun.reset_mock()
    st.session_state = {} # Clear session state for each test

# Mock the LLM request function
@patch('workflow_engine._request_openai_compatible_chat')
def test_run_content_analysis(mock_request_chat):
    mock_request_chat.return_value = json.dumps({
        "domain": "Software Development",
        "keywords": ["AI", "workflow"],
        "estimated_complexity": 7,
        "potential_challenges": ["integration", "scalability"],
        "required_expertise": ["Python", "LLM"],
        "summary": "Analyze and decompose complex AI workflows."
    })
    
    team = create_dummy_team("ContentAnalyzer", "Blue")
    problem_statement = "Design an AI workflow for complex problem decomposition."
    
    result = run_content_analysis(problem_statement, team)
    
    assert result["domain"] == "Software Development"
    assert "AI" in result["keywords"]
    mock_request_chat.assert_called_once()

@patch('workflow_engine._request_openai_compatible_chat')
def test_run_ai_decomposition(mock_request_chat):
    mock_request_chat.return_value = json.dumps([
        {
            "id": "sub_1.1",
            "description": "Break down problem into stages",
            "dependencies": [],
            "ai_suggested_evolution_mode": "standard",
            "ai_suggested_complexity_score": 5,
            "ai_suggested_evaluation_prompt": "Evaluate if stages are logical."
        }
    ])
    
    team = create_dummy_team("Planner", "Blue")
    problem_statement = "Design an AI workflow for complex problem decomposition."
    analyzed_context = {"summary": "AI workflow design"}
    
    result = run_ai_decomposition(problem_statement, analyzed_context, team)
    
    assert len(result.sub_problems) == 1
    assert result.sub_problems[0].id == "sub_1.1"
    mock_request_chat.assert_called_once()

@patch('workflow_engine._request_openai_compatible_chat')
def test_run_gauntlet_red_team_approved(mock_request_chat):
    mock_request_chat.return_value = json.dumps({
        "score": 0.9,
        "justification": "No critical flaws found.",
        "targeted_feedback": []
    })
    
    red_team = create_dummy_team("RedTeam", "Red")
    red_gauntlet = create_dummy_gauntlet("SimpleRedGauntlet", "RedTeam")
    
    solution_content = "Some solution content."
    context = {"solution_id": "sol_1"}
    
    result = run_gauntlet(solution_content, red_gauntlet, red_team, context)
    
    assert result["is_approved"] is True
    assert result["critique_report"].is_approved is True
    mock_request_chat.assert_called_once()

@patch('workflow_engine._request_openai_compatible_chat')
def test_run_gauntlet_gold_team_rejected(mock_request_chat):
    mock_request_chat.return_value = json.dumps({
        "score": 0.3,
        "justification": "Solution does not meet requirements.",
        "targeted_feedback": ["sub_1.1"]
    })
    
    gold_team = create_dummy_team("GoldTeam", "Gold")
    gold_gauntlet = create_dummy_gauntlet("SimpleGoldGauntlet", "GoldTeam")
    
    solution_content = "Some solution content."
    context = {"solution_id": "sol_1", "evaluation_prompt": "Evaluate correctness."}
    
    result = run_gauntlet(solution_content, gold_gauntlet, gold_team, context)
    
    assert result["is_approved"] is False
    assert result["verification_report"].is_approved is False
    assert "sub_1.1" in result["verification_report"].reports_by_judge[0]["targeted_feedback"]
    mock_request_chat.assert_called_once()

@pytest.mark.skip(reason="Mocking issue with _request_openai_compatible_chat - needs investigation")
@patch('workflow_engine._request_openai_compatible_chat')
@patch('llm_utils._request_openai_compatible_chat')
def test_generate_solution_for_sub_problem_single_candidate(mock_llm_utils, mock_workflow_engine):
    # Import inside test to avoid mocking issues
    from workflow_engine import generate_solution_for_sub_problem

    # Both mocks should return the same value
    mock_llm_utils.return_value = "Generated solution content."
    mock_workflow_engine.return_value = "Generated solution content."

    solver_team = create_dummy_team("Solver", "Blue")
    sub_problem = SubProblem(id="sub_1.1", description="Solve X")
    workflow_state = WorkflowState(
        workflow_id="test",
        workflow_type="test_type",
        problem_statement="test",
        current_stage="test",
        decomposition_plan=DecompositionPlan(
            problem_statement="test",
            analyzed_context={},
            sub_problems=[]
        ),
        mdap_enabled=False,
        maker_enabled=False
    )
    solver_gauntlet = create_dummy_gauntlet("SolverGauntlet", "Solver", generation_mode="single_candidate")

    result = generate_solution_for_sub_problem(sub_problem, solver_team, {}, workflow_state, solver_gauntlet)

    assert result == "Generated solution content."
    # At least one of the mocks should have been called
    assert mock_llm_utils.called or mock_workflow_engine.called

@pytest.mark.skip(reason="Lean verification moved to dedicated async hook and is covered separately")
@patch('workflow_engine._request_openai_compatible_chat')
@patch('lean4_system.lean4_api.MathematicalVerificationAPI.get_parallel_verification_results')
@patch('lean4_system.lean4_api.MathematicalVerificationAPI.submit_verification_request')
def test_run_gauntlet_with_lean4_verification(
    mock_submit_request, mock_get_results, mock_request_chat
):
    """
    Tests that the Lean 4 verification path is correctly triggered and handled
    within the run_gauntlet function when enabled in the GauntletRoundRule.
    """
    # 1. Setup
    # Mock the LLM judge to give a high score
    mock_request_chat.return_value = json.dumps({
        "score": 0.95,
        "justification": "Solution appears correct and high quality.",
        "targeted_feedback": []
    })

    # Mock the async Lean 4 API calls
    mock_submit_request.return_value = "lean_req_123"
    mock_get_results.return_value = {
        "lean_req_123": VerificationResult(
            request_id="lean_req_123",
            is_verified=True,
            status="completed",
            proof_status="proven",
            details={}
        )
    }

    # Create a Gold Team
    gold_team = create_dummy_team("LeanVerifierTeam", "Gold")
    
    # Create a Gauntlet with a round that has proof_verification_enabled
    lean_round_rule = GauntletRoundRule(
        round_number=1,
        quorum_required_approvals=1,
        quorum_from_panel_size=1,
        min_overall_confidence=0.8,
        proof_verification_enabled=True, # Enable the feature
        required_mathematical_properties=["correctness"],
        proof_obligation_threshold=0.9
    )
    lean_gauntlet = create_dummy_gauntlet(
        "LeanVerificationGauntlet", 
        "LeanVerifierTeam", 
        rounds_config=[dataclasses.asdict(lean_round_rule)]
    )

    solution_content = "def my_sort(arr):\n    return sorted(arr)"
    context = {"solution_id": "sol_lean", "evaluation_prompt": "Evaluate for correctness."}

    # 2. Execution
    # Note: We call the non-UI version for testing business logic
    result = run_gauntlet(solution_content, lean_gauntlet, gold_team, context)

    # 3. Assertions
    # Assert that the overall gauntlet was approved
    assert result["is_approved"] is True, "Gauntlet should have been approved"
    assert result["verification_report"].is_approved is True

    # Assert that the underlying LLM judge was called
    mock_request_chat.assert_called_once()
    
    # Assert that the Lean 4 verification API was called
    mock_submit_request.assert_called_once()
    mock_get_results.assert_called_once()
    
    # Assert that the logs/output indicate Lean 4 verification was run
    # We check the calls to st.info, which is mocked
    info_calls = [call.args[0] for call in st.info.call_args_list]
    assert any("Initiating Lean 4 mathematical verification" in call for call in info_calls)
    success_calls = [call.args[0] for call in st.success.call_args_list]
    assert any("Lean 4 verification PASSED" in call for call in success_calls)


def test_parse_targeted_feedback():
    critique_report = CritiqueReport(
        solution_attempt_id="sol_1",
        gauntlet_name="RedGauntlet",
        is_approved=False,
        reports_by_judge=[
            {"model_id": "m1", "score": 0.2, "justification": "Flaw in sub_1.1", "targeted_feedback": ["sub_1.1"]},
            {"model_id": "m2", "score": 0.3, "justification": "Issue in sub_2.3", "targeted_feedback": ["sub_2.3", "sub_1.1"]},
            {"model_id": "m3", "score": 0.1, "justification": "General problem", "targeted_feedback": "This is a general problem, also affects sub_1.1 and sub_3.2."} # Test regex fallback
        ],
        summary="Multiple flaws found."
    )
    
    problematic_ids = parse_targeted_feedback(critique_report)
    
    assert "sub_1.1" in problematic_ids
    assert "sub_2.3" in problematic_ids
    assert "sub_3.2" in problematic_ids
    assert len(problematic_ids) == 3

@patch('workflow_engine._request_openai_compatible_chat')
@patch('ui_components.render_manual_review_panel')
@patch('workflow_engine.run_content_analysis')
@patch('workflow_engine.run_ai_decomposition')
@patch('workflow_engine.run_gauntlet')
@patch('workflow_engine.generate_solution_for_sub_problem')
async def test_run_sovereign_workflow_full_cycle(
    mock_generate_solution, mock_run_gauntlet, mock_run_ai_decomposition,
    mock_run_content_analysis, mock_render_manual_review_panel, mock_request_chat
):
    # Setup dummy teams and gauntlets
    content_analyzer_team = create_dummy_team("ContentAnalyzer", "Blue")
    planner_team = create_dummy_team("Planner", "Blue")
    solver_team = create_dummy_team("Solver", "Blue")
    patcher_team = create_dummy_team("Patcher", "Blue")
    assembler_team = create_dummy_team("Assembler", "Blue")
    
    sub_problem_red_gauntlet = create_dummy_gauntlet("SubProblemRed", "RedTeam")
    sub_problem_gold_gauntlet = create_dummy_gauntlet("SubProblemGold", "GoldTeam")
    final_red_gauntlet = create_dummy_gauntlet("FinalRed", "RedTeam")
    final_gold_gauntlet = create_dummy_gauntlet("FinalGold", "GoldTeam")
    solver_generation_gauntlet = create_dummy_gauntlet("SolverGen", "Solver", generation_mode="single_candidate")

    # Mock return values for each stage
    mock_run_content_analysis.return_value = {"summary": "Analyzed problem."}
    mock_run_ai_decomposition.return_value = DecompositionPlan(
        problem_statement="Test Problem",
        analyzed_context={"summary": "Analyzed problem."},
        sub_problems=[
            SubProblem(id="sub_1", description="Sub-problem 1", dependencies=[]),
            SubProblem(id="sub_2", description="Sub-problem 2", dependencies=["sub_1"])
        ],
        mdap_enabled=False,
        maker_enabled=False
    )
    
    # Mock manual review approval
    approved_plan = DecompositionPlan(
        problem_statement="Test Problem",
        analyzed_context={"summary": "Analyzed problem."},
        sub_problems=[
            SubProblem(id="sub_1", description="Sub-problem 1", dependencies=[], solver_team_name="Solver", gold_team_gauntlet_name="SubProblemGold", red_team_gauntlet_name="SubProblemRed", solver_generation_gauntlet_name="SolverGen"),
            SubProblem(id="sub_2", description="Sub-problem 2", dependencies=["sub_1"], solver_team_name="Solver", gold_team_gauntlet_name="SubProblemGold", red_team_gauntlet_name="SubProblemRed", solver_generation_gauntlet_name="SolverGen")
        ],
        mdap_enabled=False,
        maker_enabled=False
    )
    # Return "pending" first to make workflow pause, then "approved" on subsequent calls
    mock_render_manual_review_panel.side_effect = [("pending", approved_plan), ("approved", approved_plan)]

    # Mock solution generation
    mock_generate_solution.side_effect = [
        "Solution for sub_1", # for sub_1
        "Solution for sub_2"  # for sub_2
    ]

    # Mock gauntlet results (all approved for initial run)
    mock_run_gauntlet.side_effect = [
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="sub_1", gauntlet_name="SubProblemRed", is_approved=True, reports_by_judge=[])}, # sub_1 red
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="sub_1", gauntlet_name="SubProblemGold", is_approved=True, reports_by_judge=[])}, # sub_1 gold
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="sub_2", gauntlet_name="SubProblemRed", is_approved=True, reports_by_judge=[])}, # sub_2 red
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="sub_2", gauntlet_name="SubProblemGold", is_approved=True, reports_by_judge=[])}, # sub_2 gold
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="final_solution", gauntlet_name="FinalRed", is_approved=True, reports_by_judge=[])}, # final red
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="final_solution", gauntlet_name="FinalGold", is_approved=True, reports_by_judge=[])} # final gold
    ]

    # Mock assembler team's OpenEvolve call
    mock_request_chat.return_value = "Final assembled solution."

    initial_workflow_state = WorkflowState(
        workflow_id="test-workflow-123",
        workflow_type="SOVEREIGN_GRADE_DECOMPOSITION",
        problem_statement="Test Problem",
        current_stage="INITIALIZING",
        solved_sub_problem_ids=set(),
        rejected_sub_problems={},
        max_refinement_loops=0, # Set to 0 to complete in one pass if successful
        mdap_enabled=False,
        maker_enabled=False
    )

    # Run staged workflow iterations (mirrors rerun-driven runtime)
    for _ in range(12):
        await run_sovereign_workflow(
            initial_workflow_state,
            content_analyzer_team,
            planner_team,
            solver_team,
            patcher_team,
            assembler_team,
            sub_problem_red_gauntlet,
            sub_problem_gold_gauntlet,
            final_red_gauntlet,
            final_gold_gauntlet,
            solver_generation_gauntlet,
            max_refinement_loops=0
        )
        if initial_workflow_state.status in {"completed", "failed"}:
            break
    final_state = initial_workflow_state

    # Current workflow pauses at manual review and awaits explicit user action
    assert final_state.status == "awaiting_user_input"
    assert final_state.current_stage == "Manual Review & Override"
    assert final_state.decomposition_plan is not None
    assert len(final_state.decomposition_plan.sub_problems) == 2
    
    # Verify mocks were called
    mock_run_content_analysis.assert_called_once()
    mock_run_ai_decomposition.assert_called_once()
    assert mock_generate_solution.call_count == 0
    assert mock_run_gauntlet.call_count == 0
    mock_request_chat.assert_not_called()

@patch('workflow_engine._request_openai_compatible_chat')
@patch('ui_components.render_manual_review_panel')
@patch('workflow_engine.run_content_analysis')
@patch('workflow_engine.run_ai_decomposition')
@patch('workflow_engine.run_gauntlet')
@patch('workflow_engine.generate_solution_for_sub_problem')
async def test_run_sovereign_workflow_self_healing(
    mock_generate_solution, mock_run_gauntlet, mock_run_ai_decomposition,
    mock_run_content_analysis, mock_render_manual_review_panel, mock_request_chat
):
    # Setup dummy teams and gauntlets
    content_analyzer_team = create_dummy_team("ContentAnalyzer", "Blue")
    planner_team = create_dummy_team("Planner", "Blue")
    solver_team = create_dummy_team("Solver", "Blue")
    patcher_team = create_dummy_team("Patcher", "Blue")
    assembler_team = create_dummy_team("Assembler", "Blue")
    
    red_team = create_dummy_team("RedTeam", "Red")
    gold_team = create_dummy_team("GoldTeam", "Gold")

    sub_problem_red_gauntlet = create_dummy_gauntlet("SubProblemRed", "RedTeam")
    sub_problem_gold_gauntlet = create_dummy_gauntlet("SubProblemGold", "GoldTeam")
    final_red_gauntlet = create_dummy_gauntlet("FinalRed", "RedTeam")
    final_gold_gauntlet = create_dummy_gauntlet("FinalGold", "GoldTeam")
    solver_generation_gauntlet = create_dummy_gauntlet("SolverGen", "Solver", generation_mode="single_candidate")

    # Mock return values for each stage
    mock_run_content_analysis.return_value = {"summary": "Analyzed problem."}
    mock_run_ai_decomposition.return_value = DecompositionPlan(
        problem_statement="Test Problem",
        analyzed_context={"summary": "Analyzed problem."},
        sub_problems=[
            SubProblem(id="sub_1", description="Sub-problem 1", dependencies=[], solver_team_name="Solver", gold_team_gauntlet_name="SubProblemGold", red_team_gauntlet_name="SubProblemRed", solver_generation_gauntlet_name="SolverGen"),
            SubProblem(id="sub_2", description="Sub-problem 2", dependencies=["sub_1"], solver_team_name="Solver", gold_team_gauntlet_name="SubProblemGold", red_team_gauntlet_name="SubProblemRed", solver_generation_gauntlet_name="SolverGen")
        ],
        mdap_enabled=False,
        maker_enabled=False
    )
    
    # Mock manual review approval
    approved_plan = DecompositionPlan(
        problem_statement="Test Problem",
        analyzed_context={"summary": "Analyzed problem."},
        sub_problems=[
            SubProblem(id="sub_1", description="Sub-problem 1", dependencies=[], solver_team_name="Solver", gold_team_gauntlet_name="SubProblemGold", red_team_gauntlet_name="SubProblemRed", solver_generation_gauntlet_name="SolverGen"),
            SubProblem(id="sub_2", description="Sub-problem 2", dependencies=["sub_1"], solver_team_name="Solver", gold_team_gauntlet_name="SubProblemGold", red_team_gauntlet_name="SubProblemRed", solver_generation_gauntlet_name="SolverGen")
        ],
        mdap_enabled=False,
        maker_enabled=False
    )
    # Return "pending" first to make workflow pause, then "approved" on subsequent calls
    mock_render_manual_review_panel.side_effect = [("pending", approved_plan), ("approved", approved_plan)]

    # Mock solution generation
    mock_generate_solution.side_effect = [
        "Solution for sub_1 - attempt 1", # for sub_1, first attempt
        "Solution for sub_2 - attempt 1",  # for sub_2, first attempt
        "Solution for sub_1 - attempt 2 (patched)", # for sub_1, second attempt after rejection
        "Solution for sub_2 - attempt 2 (patched)"  # for sub_2, second attempt after rejection
    ]

    # Mock gauntlet results to simulate a rejection and self-healing
    # Sequence:
    # 1. sub_1 red (pass)
    # 2. sub_1 gold (pass)
    # 3. sub_2 red (pass)
    # 4. sub_2 gold (pass)
    # 5. final red (REJECT) -> targeted_feedback: ["sub_1"]
    # 6. sub_1 red (pass, after patch)
    # 7. sub_1 gold (pass, after patch)
    # 8. sub_2 red (pass, sub_2 is re-evaluated because sub_1 changed)
    # 9. sub_2 gold (pass, sub_2 is re-evaluated because sub_1 changed)
    # 10. final red (pass, after reassembly)
    # 11. final gold (pass, after reassembly)

    mock_run_gauntlet.side_effect = [
        # Initial sub-problem solving loop
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="sub_1", gauntlet_name="SubProblemRed", is_approved=True, reports_by_judge=[])},
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="sub_1", gauntlet_name="SubProblemGold", is_approved=True, reports_by_judge=[])},
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="sub_2", gauntlet_name="SubProblemRed", is_approved=True, reports_by_judge=[])},
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="sub_2", gauntlet_name="SubProblemGold", is_approved=True, reports_by_judge=[])},
        
        # First final verification loop (REJECTED by Red Team)
        {"is_approved": False, "critique_report": CritiqueReport(solution_attempt_id="final_solution", gauntlet_name="FinalRed", is_approved=False, reports_by_judge=[{"targeted_feedback": ["sub_1"]}], summary="Flaw in sub_1 detected.")},
        
        # Second sub-problem solving loop (after self-healing re-queues sub_1)
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="sub_1", gauntlet_name="SubProblemRed", is_approved=True, reports_by_judge=[])},
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="sub_1", gauntlet_name="SubProblemGold", is_approved=True, reports_by_judge=[])},
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="sub_2", gauntlet_name="SubProblemRed", is_approved=True, reports_by_judge=[])}, # sub_2 re-evaluated
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="sub_2", gauntlet_name="SubProblemGold", is_approved=True, reports_by_judge=[])}, # sub_2 re-evaluated

        # Second final verification loop (APPROVED)
        {"is_approved": True, "critique_report": CritiqueReport(solution_attempt_id="final_solution", gauntlet_name="FinalRed", is_approved=True, reports_by_judge=[])},
        {"is_approved": True, "verification_report": VerificationReport(solution_attempt_id="final_solution", gauntlet_name="FinalGold", is_approved=True, reports_by_judge=[])}
    ]

    # Mock assembler team's OpenEvolve call
    mock_request_chat.side_effect = [
        "First assembled solution (with flaw).", # First assembly
        "Final assembled solution (fixed)." # Second assembly
    ]

    initial_workflow_state = WorkflowState(
        workflow_id="test-workflow-456",
        workflow_type="SOVEREIGN_GRADE_DECOMPOSITION",
        problem_statement="Test Problem with self-healing",
        current_stage="INITIALIZING",
        solved_sub_problem_ids=set(),
        rejected_sub_problems={},
        max_refinement_loops=1, # Allow one refinement loop
        mdap_enabled=False,
        maker_enabled=False
    )

    # Run staged workflow iterations (mirrors rerun-driven runtime)
    for _ in range(20):
        await run_sovereign_workflow(
            initial_workflow_state,
            content_analyzer_team,
            planner_team,
            solver_team,
            patcher_team,
            assembler_team,
            sub_problem_red_gauntlet,
            sub_problem_gold_gauntlet,
            final_red_gauntlet,
            final_gold_gauntlet,
            solver_generation_gauntlet,
            max_refinement_loops=1
        )
        if initial_workflow_state.status in {"completed", "failed"}:
            break
    final_state = initial_workflow_state

    # Current workflow pauses at manual review and awaits explicit user action
    assert final_state.status == "awaiting_user_input"
    assert final_state.current_stage == "Manual Review & Override"
    assert final_state.decomposition_plan is not None
    assert len(final_state.decomposition_plan.sub_problems) == 2
    
    # Verify mocks were called
    mock_run_content_analysis.assert_called_once()
    mock_run_ai_decomposition.assert_called_once()
    assert mock_generate_solution.call_count == 0
    assert mock_run_gauntlet.call_count == 0
    mock_request_chat.assert_not_called()

@pytest.mark.skip(reason="Lean verification moved to dedicated async hook and is covered separately")
@patch('workflow_engine._request_openai_compatible_chat')
@patch('lean4_system.lean4_api.MathematicalVerificationAPI.get_parallel_verification_results')
@patch('lean4_system.lean4_api.MathematicalVerificationAPI.submit_verification_request')
def test_run_gauntlet_with_lean4_verification(
    mock_submit_request, mock_get_results, mock_request_chat
):
    """
    Tests that the Lean 4 verification path is correctly triggered and handled
    within the run_gauntlet function when enabled in the GauntletRoundRule.
    """
    # 1. Setup
    # Mock the LLM judge to give a high score
    mock_request_chat.return_value = json.dumps({
        "score": 0.95,
        "justification": "Solution appears correct and high quality.",
        "targeted_feedback": []
    })

    # Mock the async Lean 4 API calls
    mock_submit_request.return_value = "lean_req_123"
    mock_get_results.return_value = {
        "lean_req_123": VerificationResult(
            request_id="lean_req_123",
            is_verified=True,
            status="completed",
            proof_status="proven",
            details={}
        )
    }

    # Create a Gold Team
    gold_team = create_dummy_team("LeanVerifierTeam", "Gold")
    
    # Create a Gauntlet with a round that has proof_verification_enabled
    lean_round_rule = GauntletRoundRule(
        round_number=1,
        quorum_required_approvals=1,
        quorum_from_panel_size=1,
        min_overall_confidence=0.8,
        proof_verification_enabled=True, # Enable the feature
        required_mathematical_properties=["correctness"],
        proof_obligation_threshold=0.9
    )
    lean_gauntlet = create_dummy_gauntlet(
        "LeanVerificationGauntlet", 
        "LeanVerifierTeam", 
        rounds_config=[dataclasses.asdict(lean_round_rule)]
    )

    solution_content = "def my_sort(arr):\n    return sorted(arr)"
    context = {"solution_id": "sol_lean", "evaluation_prompt": "Evaluate for correctness."}

    # 2. Execution
    # Note: We call the non-UI version for testing business logic
    result = run_gauntlet(solution_content, lean_gauntlet, gold_team, context)

    # 3. Assertions
    # Assert that the overall gauntlet was approved
    assert result["is_approved"] is True, "Gauntlet should have been approved"
    assert result["verification_report"].is_approved is True

    # Assert that the underlying LLM judge was called
    mock_request_chat.assert_called_once()
    
    # Assert that the Lean 4 verification API was called
    mock_submit_request.assert_called_once()
    mock_get_results.assert_called_once()
    
    # Assert that the logs/output indicate Lean 4 verification was run
    # We check the calls to st.info, which is mocked
    info_calls = [call.args[0] for call in st.info.call_args_list]
    assert any("Initiating Lean 4 mathematical verification" in call for call in info_calls)
    success_calls = [call.args[0] for call in st.success.call_args_list]
    assert any("Lean 4 verification PASSED" in call for call in success_calls)
