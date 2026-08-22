"""
Offline core tests for the Sovereign-Grade Decomposition Workflow engine.

These tests exercise the CORE engine logic in ``workflow_engine.py`` WITHOUT any
API keys or network access. The OpenAI-compatible chat function is monkeypatched
at the module level with a deterministic mock.

Run with:
    python -m pytest engines/other/test_workflow_engine_core.py -q -p no:pytest_ethereum
"""

import os
import sys
import types
from unittest.mock import patch

# --- Make the flat ``engines/`` scripts importable (no package, no __init__.py) ---
_HERE = os.path.dirname(os.path.abspath(__file__))          # engines/other
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # repo root

# Ensure BOTH directories are present. The repo root MUST come *before*
# ``engines/other`` in sys.path so that ``from utils.entanglement_utils import ...``
# inside ``workflow_engine`` resolves to the repo-root ``utils`` *package* and not
# to the unrelated ``engines/other/utils.py`` module that shadows it.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Force the repo root to the very front (it may already be present elsewhere due
# to pytest's rootdir handling; move it to the front unconditionally).
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import workflow_engine as we  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic offline chat mock.
#
# It returns a JSON string that is valid BOTH as a "solution" (used verbatim by
# the Blue Team generation path) and as a critique/verification report (parsed
# for ``score`` / ``targeted_feedback`` by the gauntlet runners).
# ---------------------------------------------------------------------------
def _mock_chat(**kwargs):
    return (
        '{"solution": "def solve():\n    return 42", '
        '"score": 0.9, '
        '"justification": "The solution is correct and complete.", '
        '"targeted_feedback": []}'
    )


# ---------------------------------------------------------------------------
# Fake object builders
# ---------------------------------------------------------------------------
def _make_member(model_id="mock-model"):
    """Build a fake ModelConfig-like object with every attribute the engine reads."""
    attrs = dict(
        model_id=model_id,
        api_key="",
        api_base="",
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=None,
        n=1,
        logit_bias=None,
        reasoning_effort=None,
        stop_sequences=None,
        logprobs=None,
        top_logprobs=None,
        response_format=None,
        stream=False,
        user=None,
        max_retries=1,
        timeout=10,
        organization=None,
        response_model=None,
        tools=None,
        tool_choice=None,
        system_fingerprint=None,
        deployment_id=None,
        encoding_format=None,
        max_input_tokens=None,
        stop_token=None,
        best_of=None,
        logprobs_offset=None,
        suffix=None,
        presence_penalty_range=None,
        frequency_penalty_range=None,
        stop_token_id=None,
        response_json_format=None,
        max_output_tokens=None,
        stream_options=None,
        logprobs_type=None,
        top_k=None,
        repetition_penalty=None,
        length_penalty=None,
        early_stopping=None,
        num_beams=None,
        do_sample=None,
        temperature_fallback=None,
        top_p_fallback=None,
        max_time=None,
        return_full_text=None,
        tokenizer_config=None,
        model_kwargs=None,
    )
    return types.SimpleNamespace(**attrs)


def _make_team(role="Blue", name="test-team"):
    return types.SimpleNamespace(
        name=name,
        role=role,
        members=[_make_member()],
        solver_system_prompt=None,
        solver_user_prompt_template=None,
        gold_team_system_prompt=None,
        gold_team_user_prompt_template=None,
    )


def _make_sub_problem(sub_id="sub_1.1", deps=None):
    return types.SimpleNamespace(
        id=sub_id,
        description=f"Solve sub-problem {sub_id}.",
        metadata={},
        evolution_params={},
        ai_suggested_evolution_mode="standard",
        ai_suggested_complexity_score=0.0,
        estimated_effort=None,
        content_type="text",
        dependencies=deps or [],
        solver_team_name=None,
        solver_generation_gauntlet_name=None,
    )


def _make_workflow_state():
    decomposition_plan = types.SimpleNamespace(
        analyzed_context={},
        maker_enabled=False,
        mdap_enabled=False,
    )
    return types.SimpleNamespace(
        maker_enabled=False,
        mdap_enabled=False,
        enable_adaptive_mdap=False,
        decomposition_plan=decomposition_plan,
    )


def _make_gauntlet(name="test-gauntlet", gauntlet_type="standard",
                   generation_mode="single_candidate"):
    return types.SimpleNamespace(
        name=name,
        team_name="test-team",
        rounds=[],
        gauntlet_type=gauntlet_type,
        generation_mode=generation_mode,
        attack_modes=[],
        description=None,
        gauntlet_config=None,
        red_flags={},
        metadata={},
    )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture
def chat_mock():
    with patch.object(we, "_request_openai_compatible_chat", _mock_chat):
        yield _mock_chat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_generate_solution_single_candidate(chat_mock):
    sp = _make_sub_problem()
    team = _make_team(role="Blue")
    ws = _make_workflow_state()
    gauntlet = _make_gauntlet(generation_mode="single_candidate")

    result = we.generate_solution_for_sub_problem(
        sub_problem=sp,
        team=team,
        context={"current_solution": ""},
        workflow_state=ws,
        solver_generation_gauntlet=gauntlet,
        emit_ui=False,
    )

    assert hasattr(result, "content"), "expected a SolutionAttempt-like result"
    assert isinstance(result.content, str) and result.content.strip()
    assert result.metadata.get("generation_mode") == "single_candidate"
    assert result.status == "generated"


def test_generate_solution_multi_candidate_peer_review(chat_mock):
    sp = _make_sub_problem()
    team = _make_team(role="Blue")
    # multi-candidate needs at least one member (we only have one, that's fine)
    ws = _make_workflow_state()
    gauntlet = _make_gauntlet(generation_mode="multi_candidate_peer_review")

    result = we.generate_solution_for_sub_problem(
        sub_problem=sp,
        team=team,
        context={"current_solution": ""},
        workflow_state=ws,
        solver_generation_gauntlet=gauntlet,
        emit_ui=False,
    )

    assert hasattr(result, "content")
    assert isinstance(result.content, str) and result.content.strip()
    assert result.metadata.get("generation_mode") == "multi_candidate_peer_review"
    # vote_counts should record that we generated candidates and synthesized one
    assert result.metadata.get("vote_counts", {}).get("candidates_generated", 0) >= 1


def test_generate_solution_injectable_chat_fn():
    sp = _make_sub_problem()
    team = _make_team(role="Blue")
    ws = _make_workflow_state()
    gauntlet = _make_gauntlet(generation_mode="single_candidate")

    captured = {}

    def my_chat(**kwargs):
        captured["called"] = True
        return "INJECTED_SOLUTION"

    result = we.generate_solution_for_sub_problem(
        sub_problem=sp,
        team=team,
        context={"current_solution": ""},
        workflow_state=ws,
        solver_generation_gauntlet=gauntlet,
        emit_ui=False,
        chat_fn=my_chat,
    )
    assert captured.get("called") is True
    assert result.content == "INJECTED_SOLUTION"


def test_parse_targeted_feedback_json(chat_mock):
    import json
    report = {
        "reports_by_judge": [
            {"model_id": "m1", "targeted_feedback": json.dumps(["sub_1.1", "sub_2.3"])},
        ],
        "problematic_sub_problems": ["sub_3.4"],
    }
    ids = we.parse_targeted_feedback(report)
    assert "sub_1.1" in ids
    assert "sub_2.3" in ids
    assert "sub_3.4" in ids


def test_parse_targeted_feedback_plain_text_fallback(chat_mock):
    report = {
        "reports_by_judge": [
            {"model_id": "m1", "targeted_feedback": "The flaw originates in sub_1.2 and propagates to sub_4.5."},
        ],
    }
    ids = we.parse_targeted_feedback(report)
    assert "sub_1.2" in ids
    assert "sub_4.5" in ids


def test_parse_targeted_feedback_raw_string_json(chat_mock):
    import json
    raw = json.dumps({"affected_components": ["sub_2.1", "sub_2.2"]})
    ids = we.parse_targeted_feedback(raw)
    assert "sub_2.1" in ids
    assert "sub_2.2" in ids


@pytest.mark.parametrize("gauntlet_type", [
    "adaptive", "hierarchical", "competitive", "collaborative",
])
def test_run_gauntlet_advanced_types(chat_mock, gauntlet_type):
    team = _make_team(role="Red")
    gauntlet = _make_gauntlet(name=f"{gauntlet_type}-g", gauntlet_type=gauntlet_type)
    result = we.run_gauntlet(
        solution_content="def solve():\n    return 42",
        gauntlet_def=gauntlet,
        team=team,
        context={},
    )
    assert isinstance(result, dict)
    assert "is_approved" in result
    assert "report_summary" in result


@pytest.mark.parametrize("gen_mode", ["single_candidate", "multi_candidate_peer_review"])
def test_run_gauntlet_blue_team_generation(chat_mock, gen_mode):
    team = _make_team(role="Blue")
    gauntlet = _make_gauntlet(name="blue-gen", gauntlet_type="standard", generation_mode=gen_mode)
    ws = _make_workflow_state()
    sp = _make_sub_problem(sub_id="sub_7.7")
    context = {"workflow_state": ws, "sub_problem": sp, "sub_problem_id": "sub_7.7"}

    result = we.run_gauntlet(
        solution_content="Generate a solution for sub_7.7.",
        gauntlet_def=gauntlet,
        team=team,
        context=context,
    )
    assert isinstance(result, dict)
    assert result.get("is_approved") is True
    assert "solution_attempt" in result
    assert result["solution_attempt"].content.strip()


@pytest.mark.parametrize("gen_mode", ["single_candidate", "multi_candidate_peer_review"])
def test_run_gauntlet_headless_blue_team_generation(chat_mock, gen_mode):
    team = _make_team(role="Blue")
    gauntlet = _make_gauntlet(name="blue-gen-h", gauntlet_type="standard", generation_mode=gen_mode)
    ws = _make_workflow_state()
    sp = _make_sub_problem(sub_id="sub_8.8")
    context = {"workflow_state": ws, "sub_problem": sp, "sub_problem_id": "sub_8.8"}

    result = we.run_gauntlet_headless(
        solution_content="Generate a solution for sub_8.8.",
        gauntlet_def=gauntlet,
        team=team,
        context=context,
    )
    assert isinstance(result, dict)
    assert result.get("is_approved") is True
    assert "report_object" in result
    assert result["report_object"]["generated_solution"].strip()


def test_detect_circular_dependencies_acyclic():
    subs = [
        _make_sub_problem("sub_1.1", deps=["sub_1.2"]),
        _make_sub_problem("sub_1.2", deps=[]),
        _make_sub_problem("sub_1.3", deps=["sub_1.1"]),
    ]
    assert we.detect_circular_dependencies(subs) == []


def test_detect_circular_dependencies_cycle():
    subs = [
        _make_sub_problem("sub_1.1", deps=["sub_1.2"]),
        _make_sub_problem("sub_1.2", deps=["sub_1.3"]),
        _make_sub_problem("sub_1.3", deps=["sub_1.1"]),
    ]
    cycles = we.detect_circular_dependencies(subs)
    assert set(cycles) == {"sub_1.1", "sub_1.2", "sub_1.3"}


def test_run_gauntlet_no_team_or_gauntlet(chat_mock):
    # No team and no gauntlet -> graceful error, must not crash.
    result = we.run_gauntlet("content", None, None, {})
    assert isinstance(result, dict)
    assert result.get("is_approved") is False
    assert result.get("error") == "missing_team_or_gauntlet"

    result_h = we.run_gauntlet_headless("content", None, None, {})
    assert isinstance(result_h, dict)
    assert result_h.get("is_approved") is False
    assert result_h.get("error") == "missing_team_or_gauntlet"
