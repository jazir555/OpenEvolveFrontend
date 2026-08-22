"""
End-to-end OFFLINE integration test for the Sovereign-Grade Decomposition Workflow.

This composes the real, already-unit-tested stage functions into the documented
data flow WITHOUT any API keys or network access:

    decompose -> solve each sub-problem -> verify via gauntlet -> extract knowledge
    -> solve a sub-problem dependency DAG via the distributed executor

The OpenAI-compatible chat function is monkeypatched at the module level with a
deterministic mock. Everything runs locally.

Run with:
    python -m pytest engines/other/test_decomposition_workflow_e2e.py -q -p no:pytest_ethereum
"""

import os
import sys
import types
from unittest.mock import patch

# --- Make the flat ``engines/`` scripts importable (no package, no __init__.py) ---
_HERE = os.path.dirname(os.path.abspath(__file__))            # engines/other
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # repo root

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import workflow_engine as we                                   # noqa: E402
import workflow_knowledge as wk                                # noqa: E402
import workflow_distributed as wd                              # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic offline chat mock (valid both as a solution and a report).
# ---------------------------------------------------------------------------
def _mock_chat(**kwargs):
    return (
        '{"solution": "def solve():\n    return 42", '
        '"score": 0.9, '
        '"justification": "The solution is correct and complete.", '
        '"targeted_feedback": []}'
    )


def _make_member(model_id="mock-model"):
    attrs = dict(
        model_id=model_id, api_key="", api_base="", temperature=0.0, top_p=1.0,
        max_tokens=512, frequency_penalty=0.0, presence_penalty=0.0, seed=None,
        n=1, logit_bias=None, reasoning_effort=None, stop_sequences=None,
        logprobs=None, top_logprobs=None, response_format=None, stream=False,
        user=None, max_retries=1, timeout=10, organization=None,
        response_model=None, tools=None, tool_choice=None,
        system_fingerprint=None, deployment_id=None, encoding_format=None,
        max_input_tokens=None, stop_token=None, best_of=None,
        logprobs_offset=None, suffix=None, presence_penalty_range=None,
        frequency_penalty_range=None, stop_token_id=None,
        response_json_format=None, max_output_tokens=None, stream_options=None,
        logprobs_type=None, top_k=None, repetition_penalty=None,
        length_penalty=None, early_stopping=None, num_beams=None,
        do_sample=None, temperature_fallback=None, top_p_fallback=None,
        max_time=None, return_full_text=None, tokenizer_config=None,
        model_kwargs=None,
    )
    return types.SimpleNamespace(**attrs)


def _make_team(role="Blue", name="test-team"):
    return types.SimpleNamespace(
        name=name, role=role, members=[_make_member()],
        solver_system_prompt=None, solver_user_prompt_template=None,
        gold_team_system_prompt=None, gold_team_user_prompt_template=None,
    )


def _make_sub_problem(sub_id, deps=None):
    return types.SimpleNamespace(
        id=sub_id, description=f"Solve sub-problem {sub_id}.", metadata={},
        evolution_params={}, ai_suggested_evolution_mode="standard",
        ai_suggested_complexity_score=0.0, estimated_effort=None,
        content_type="text", dependencies=deps or [],
        solver_team_name=None, solver_generation_gauntlet_name=None,
    )


def _make_workflow_state():
    decomposition_plan = types.SimpleNamespace(
        analyzed_context={}, maker_enabled=False, mdap_enabled=False,
    )
    return types.SimpleNamespace(
        workflow_id="e2e-wf", maker_enabled=False, mdap_enabled=False,
        enable_adaptive_mdap=False, decomposition_plan=decomposition_plan,
    )


def _make_gauntlet(name="test-gauntlet", gauntlet_type="standard",
                   generation_mode="single_candidate"):
    return types.SimpleNamespace(
        name=name, team_name="test-team", rounds=[], gauntlet_type=gauntlet_type,
        generation_mode=generation_mode, attack_modes=[], description=None,
        gauntlet_config=None, red_flags={}, metadata={},
    )


# ---------------------------------------------------------------------------
# End-to-end composition
# ---------------------------------------------------------------------------
def test_end_to_end_decomposition_solve_verify_extract():
    with patch.object(we, "_request_openai_compatible_chat", _mock_chat):
        ws = _make_workflow_state()
        solver_team = _make_team(role="Blue")
        gold_gauntlet = _make_gauntlet(name="gold", gauntlet_type="standard")

        sub_problems = [
            _make_sub_problem("sub_1.1"),
            _make_sub_problem("sub_1.2"),
        ]
        decomposition_plan = types.SimpleNamespace(
            sub_problems=sub_problems,
            analyzed_context={"auto_approval": True},
            maker_enabled=False, mdap_enabled=False,
        )
        ws.decomposition_plan = decomposition_plan

        # Stage 3/4: solve each sub-problem (real generation logic).
        attempts = []
        for sp in sub_problems:
            result = we.generate_solution_for_sub_problem(
                sub_problem=sp, team=solver_team,
                context={"current_solution": ""}, workflow_state=ws,
                solver_generation_gauntlet=gold_gauntlet, emit_ui=False,
            )
            assert hasattr(result, "content") and result.content.strip()
            attempts.append(result)

        # Stage 5/6: verify each solution via a Gold gauntlet (real runner).
        for attempt in attempts:
            report = we.run_gauntlet(
                solution_content=attempt.content, gauntlet_def=gold_gauntlet,
                team=solver_team, context={},
            )
            assert isinstance(report, dict)
            assert "is_approved" in report

        # Stage 7: extract knowledge from the run (real extraction).
        artifacts = wk.extract_workflow_knowledge(
            workflow_state=ws,
            decomposition_plan=decomposition_plan,
            solution_attempts=attempts,
            reports=[{"reports_by_judge": [], "is_approved": True}],
        )
        assert isinstance(artifacts, list) and len(artifacts) >= 1

        # Distributed: solve a sub-problem dependency DAG (real executor).
        dag = [
            _make_sub_problem("A"),
            _make_sub_problem("B", deps=["A"]),
            _make_sub_problem("C", deps=["A"]),
            _make_sub_problem("D", deps=["B", "C"]),
        ]

        def _solve(sp, voter=None):
            return types.SimpleNamespace(content=f"solution-for-{sp.id}", status="generated")

        run = wd.SubProblemExecutor(default_backend="local").execute(dag, _solve)
        assert run.backend_used in ("local", "multiprocessing")
        assert set(run.solutions.keys()) == {"A", "B", "C", "D"}


def test_end_to_end_self_healing_feedback_parsing():
    # The self-healing loop relies on parse_targeted_feedback to extract the
    # sub-problem IDs a critique flagged; verify it end-to-end with a JSON report.
    import json
    report = {
        "reports_by_judge": [
            {"model_id": "m1", "targeted_feedback": json.dumps(["sub_1.1", "sub_2.3"])},
        ],
        "problematic_sub_problems": ["sub_3.4"],
    }
    ids = we.parse_targeted_feedback(report)
    assert {"sub_1.1", "sub_2.3", "sub_3.4"}.issubset(set(ids))


def test_end_to_end_cycle_guard():
    # A cyclic decomposition must be detected before any solving happens.
    cyclic = [
        _make_sub_problem("sub_1.1", deps=["sub_1.2"]),
        _make_sub_problem("sub_1.2", deps=["sub_1.1"]),
    ]
    assert we.detect_circular_dependencies(cyclic)
