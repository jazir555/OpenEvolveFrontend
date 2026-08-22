"""
Offline (no API keys) end-to-end tests for the generic MAKER workflow.

These tests exercise the REAL MakerEngine voting + red-flagging machinery with an
injectable mock voter, demonstrating the paper's central claim GENERICALLY (no
Hanoi-specific code):

  * A single agent (k=1, no red-flagging) fails within a long sequence.
  * MAKER with first-to-ahead-by-k voting + red-flagging completes a long
    sequence (1000+ steps) with ZERO errors.
  * Red-flagging filters SYSTEMATIC (correlated) malformed outputs so the run
    still succeeds, while the same correlated noise breaks the unflagged run.

The mock voter models a stochastic but consistent per-step "correct" answer, plus
random unique wrong answers (which can never accumulate to a k-ahead lead) and
occasional malformed outputs (which red-flagging discards before voting).
"""

import os
import re
import sys
import json
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from maker_engine import run_generic_maker, MakerConfig, RedFlagRules


class MockVoter:
    """Backend-agnostic mock voter implementing the paper's error model.

    Per sample (independent draws):
      * with probability ``format_rate`` -> a malformed, structurally inconsistent
        output (the paper's correlated-error proxy). When red-flagging is enabled
        and a schema is supplied, this is discarded BEFORE voting.
      * otherwise with probability ``correct_rate`` -> the (consistent) correct
        action for this step; else a fresh, globally-unique wrong action that can
        never accumulate enough votes to win a k-ahead race on its own.
    """

    def __init__(self, correct_rate=0.85, format_rate=0.0, seed=1234):
        self.correct_rate = correct_rate
        self.format_rate = format_rate
        self._rng = random.Random(seed)
        self._counter = 0
        self.format_token = "###MALFORMED_OUTPUT###"

    def __call__(self, prompt, system_prompt, expected_schema, step):
        idx = 0
        m = re.search(r"(\d+)", step.step_id or "")
        if m:
            idx = int(m.group(1))

        if self._rng.random() < self.format_rate:
            # Systematic malformed output (correlated across samples).
            return (self.format_token, self.format_token)

        if self._rng.random() < self.correct_rate:
            cand = {
                "action": {"move": "correct", "step": idx},
                "next_state": {"step": idx},
            }
        else:
            self._counter += 1
            cand = {
                "action": {"move": f"wrong{self._counter}"},
                "next_state": {"step": idx},
            }
        return (json.dumps(cand), cand)


def _count_errors(actions, num_steps):
    """Compare produced actions against the known-correct oracle."""
    errors = 0
    for i, action in enumerate(actions):
        expected = {"move": "correct", "step": i + 1}
        if not isinstance(action, dict) or action.get("move") != "correct" or action.get("step") != i + 1:
            errors += 1
    # Any missing steps are also failures.
    errors += max(0, num_steps - len(actions))
    return errors


def test_single_agent_fails_within_long_sequence():
    """k=1, no red-flag schema -> the first sampled action decides; errors accrue."""
    num_steps = 100
    config = MakerConfig(k_min=1, k_max=1, max_votes_per_step=50)
    voter = MockVoter(correct_rate=0.8, format_rate=0.0, seed=7)

    result = run_generic_maker(
        initial_state={},
        num_steps=num_steps,
        step_prompt_template="State: {state}. History: {history}.",
        expected_schema=None,  # no schema -> malformed outputs not caught by red-flag
        config=config,
        voter=voter,
    )
    actions = result["actions"]
    errors = _count_errors(actions, num_steps)
    # Single agent with a ~20% per-step error rate MUST fail over 100 steps.
    assert errors > 0, f"single-agent unexpectedly had zero errors (errors={errors})"


def test_maker_zero_errors_long_sequence():
    """First-to-ahead-by-k voting + red-flagging -> ZERO errors over 1000 steps."""
    num_steps = 1000
    schema = {
        "type": "object",
        "required": ["action", "next_state"],
        "properties": {
            "action": {"type": "object"},
            "next_state": {"type": "object"},
        },
    }
    config = MakerConfig(k_min=5, k_max=5, max_votes_per_step=300)
    voter = MockVoter(correct_rate=0.85, format_rate=0.1, seed=99)

    result = run_generic_maker(
        initial_state={},
        num_steps=num_steps,
        step_prompt_template="State: {state}. History: {history}.",
        expected_schema=schema,  # enables red-flagging of malformed outputs
        config=config,
        voter=voter,
    )
    actions = result["actions"]
    errors = _count_errors(actions, num_steps)
    # MAKER must complete the long sequence with ZERO errors.
    assert errors == 0, f"MAKER produced {errors} errors over {num_steps} steps"
    assert result["metrics"]["red_flags"] > 0, "expected malformed outputs to be red-flagged"
    assert len(actions) == num_steps


def test_redflagging_removes_correlated_errors():
    """Systematic malformed output is filtered (red-flag ON) -> success;
    the same correlated noise, unflagged, breaks the run (red-flag OFF)."""
    num_steps = 1000
    schema = {
        "type": "object",
        "required": ["action", "next_state"],
        "properties": {
            "action": {"type": "object"},
            "next_state": {"type": "object"},
        },
    }

    # --- Red-flag ON: malformed outputs discarded before voting. ---
    config_on = MakerConfig(k_min=5, k_max=5, max_votes_per_step=300)
    voter_on = MockVoter(correct_rate=0.6, format_rate=0.4, seed=2024)
    result_on = run_generic_maker(
        initial_state={},
        num_steps=num_steps,
        step_prompt_template="State: {state}. History: {history}.",
        expected_schema=schema,
        config=config_on,
        voter=voter_on,
    )
    errors_on = _count_errors(result_on["actions"], num_steps)
    assert errors_on == 0, f"red-flagged run had {errors_on} errors"
    assert result_on["metrics"]["red_flags"] > 0, "malformed outputs should have been red-flagged"

    # --- Red-flag OFF: same voter; malformed output is a competing candidate. ---
    config_off = MakerConfig(
        k_min=5,
        k_max=5,
        max_votes_per_step=300,
        red_flag_rules=RedFlagRules(require_schema_match=False, max_characters=1_000_000, max_tokens=1_000_000),
    )
    voter_off = MockVoter(correct_rate=0.6, format_rate=0.4, seed=2024)
    result_off = run_generic_maker(
        initial_state={},
        num_steps=num_steps,
        step_prompt_template="State: {state}. History: {history}.",
        expected_schema=None,  # no schema -> malformed output is NOT red-flagged
        config=config_off,
        voter=voter_off,
    )
    errors_off = _count_errors(result_off["actions"], num_steps)
    # The correlated malformed output now competes and causes failures.
    assert errors_off > 0, "unflagged correlated errors should have caused failures"


def test_scaling_laws_returned():
    """The generic runner surfaces scaling-law predictions."""
    config = MakerConfig(k_min=3, k_max=3, max_votes_per_step=50)
    result = run_generic_maker(
        initial_state={},
        num_steps=500,
        step_prompt_template="State: {state}.",
        config=config,
        voter=MockVoter(seed=1),
        estimated_p=0.9,
        target_reliability=0.95,
    )
    sl = result["scaling_laws"]
    assert sl, "scaling_laws should be populated"
    assert 0.0 < sl["step_success_probability"] <= 1.0
    assert 0.0 <= sl["full_task_success_probability"] <= 1.0
    assert sl["required_k_for_reliability"] >= 1
    assert sl["parallelization_factor"] == sl["required_k_for_reliability"]
