from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "engines" / "other"))

from psv_selfplay import (  # noqa: E402
    PSVConfig,
    PSVManager,
    FormalVerificationResult,
    RFTPreferencePair,
    MathematicalProblem,
    SolutionAttempt,
    VerificationResult,
    PSVEpisode,
)


def _make_episode(problem_id, statement, solution, verified):
    problem = MathematicalProblem(
        id=problem_id,
        statement=statement,
        domain="number_theory",
        difficulty=0.5,
    )
    solution_attempt = SolutionAttempt(
        problem_id=problem_id,
        solution=solution,
        solver_id="solver-1",
    )
    verification = VerificationResult(
        problem_id=problem_id,
        solution_id="solver-1",
        is_correct=verified,
        confidence=0.9 if verified else 0.3,
        feedback="ok",
    )
    return PSVEpisode(
        episode_id=f"ep-{problem_id}",
        problem=problem,
        solution=solution_attempt,
        verification=verification,
    )


def test_verify_with_verus_absent_degrades_gracefully():
    """When the verus CLI is not on PATH, verify_with_verus must not crash and
    must report available=False (degraded) so the caller can fall back."""
    # Force the "absent" branch regardless of the host's PATH.
    import psv_selfplay as m
    original = m.PSVManager.is_verus_available
    m.PSVManager.is_verus_available = staticmethod(lambda: False)
    try:
        cfg = PSVConfig(selfplay_formal_verification_backend="verus")
        mgr = PSVManager(cfg)
        result = mgr.verify_with_verus("some solution text", spec_id="p1")
    finally:
        m.PSVManager.is_verus_available = staticmethod(original)

    assert isinstance(result, FormalVerificationResult)
    assert result.backend == "verus"
    assert result.available is False
    assert result.degraded is True
    assert result.verified is False
    assert result.error == "verus binary not found on PATH"


def test_verify_with_verus_not_feasible_without_rust():
    """A pure-prose solution with no Rust/Verus code block is not expressible as
    a Verus program; verify_with_verus should return feasible=False safely."""
    import psv_selfplay as m
    original = m.PSVManager.is_verus_available
    m.PSVManager.is_verus_available = staticmethod(lambda: True)
    try:
        cfg = PSVConfig(selfplay_formal_verification_backend="verus")
        mgr = PSVManager(cfg)
        result = mgr.verify_with_verus("We solve by substitution. x=3, y=4.", spec_id="p2")
    finally:
        m.PSVManager.is_verus_available = staticmethod(original)

    assert result.available is True
    assert result.feasible is False
    assert result.verified is False
    assert "not expressible" in (result.error or "")


def test_emit_verus_program_extracts_code_block():
    cfg = PSVConfig()
    mgr = PSVManager(cfg)
    solution = (
        "Here is a Rust/Verus solution:\n"
        "```rust\n"
        "fn add(a: u32, b: u32) -> u32 { a + b }\n"
        "```"
    )
    program = mgr.emit_verus_program(solution, spec_id="p3")
    assert program is not None
    assert "fn add" in program
    assert "a + b" in program


def test_emit_verus_program_returns_none_for_prose():
    cfg = PSVConfig()
    mgr = PSVManager(cfg)
    assert mgr.emit_verus_program("just words, no code") is None


def test_rft_update_assembles_preference_pairs_and_writes_jsonl(tmp_path):
    cfg = PSVConfig()
    mgr = PSVManager(cfg)

    # One problem, two episodes: one verified (chosen), one rejected (rejected).
    mgr.episode_history.append(
        _make_episode("prob1", "Find integers x,y with x^2+y^2=25", "x=3,y=4", verified=True)
    )
    mgr.episode_history.append(
        _make_episode("prob1", "Find integers x,y with x^2+y^2=25", "x=1,y=1", verified=False)
    )

    out_file = tmp_path / "rft.jsonl"
    status = mgr.rft_update(output_path=str(out_file))

    assert status["status"] == "recorded"
    assert status["trained"] is False
    assert status["recorded"] is True
    assert status["num_pairs"] == 1
    assert out_file.exists()

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    pair = json.loads(lines[0])
    assert set(pair.keys()) >= {"prompt", "chosen", "rejected"}
    assert pair["chosen"] == "x=3,y=4"
    assert pair["rejected"] == "x=1,y=1"


def test_rft_update_calls_trainer_hook_when_configured():
    cfg = PSVConfig(selfplay_rft_trainer_hook=lambda dataset: {"trained": len(dataset)})
    mgr = PSVManager(cfg)
    mgr.episode_history.append(
        _make_episode("prob2", "Solve x+1=2", "x=1", verified=True)
    )
    mgr.episode_history.append(
        _make_episode("prob2", "Solve x+1=2", "x=99", verified=False)
    )

    status = mgr.rft_update()
    assert status["status"] == "trained"
    assert status["trained"] is True
    assert status["recorded"] is False
    assert status["hook_result"] == {"trained": 1}
    assert status["num_pairs"] == 1


def test_rft_update_skips_when_no_pairs():
    cfg = PSVConfig()
    mgr = PSVManager(cfg)
    status = mgr.rft_update()
    assert status["status"] == "skipped"
    assert status["num_pairs"] == 0


def test_collect_preference_pairs_structure():
    cfg = PSVConfig()
    mgr = PSVManager(cfg)
    mgr.episode_history.append(
        _make_episode("prob3", "problem", "good sol", verified=True)
    )
    mgr.episode_history.append(
        _make_episode("prob3", "problem", "bad sol", verified=False)
    )
    pairs = mgr.collect_preference_pairs()
    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, RFTPreferencePair)
    assert pair.chosen == "good sol"
    assert pair.rejected == "bad sol"
    assert pair.metadata["problem_id"] == "prob3"
