"""
End-to-end smoke test for the offline mock LLM backend.

These tests exercise the real evolution engine (controller + process-parallel
workers + database + evaluator) with :class:`openevolve.llm.mock.MockLLM`, so
they require no API keys and make no network calls.
"""

import asyncio
import os
import shutil
import tempfile
import unittest

from openevolve.api import EvolutionResult, run_evolution
from openevolve.config import Config, LLMModelConfig
from openevolve.llm import MockLLM
from openevolve.llm.ensemble import LLMEnsemble, _create_model
from openevolve.llm.mock import total_calls
from openevolve.utils.code_utils import apply_diff, extract_diffs

# ---------------------------------------------------------------------------
# Fixtures: a trivial program plus a pure-Python evaluator
# ---------------------------------------------------------------------------

INITIAL_PROGRAM = '''"""Trivial program used for the mock evolution smoke test."""

# EVOLVE-BLOCK-START
def add(a, b):
    """Return the sum of two numbers."""
    total = a + b
    return total
# EVOLVE-BLOCK-END


def run():
    return add(2, 3)
'''

# Passed as source (not a callable) so it also works with spawned worker
# processes, which cannot see the parent process's in-memory functions.
EVALUATOR_SOURCE = '''
"""Pure-Python evaluator: score `add` against a local test table."""

import importlib.util


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return {"correctness": 0.0, "combined_score": 0.0}

    add = getattr(module, "add", None)
    if add is None:
        return {"correctness": 0.0, "combined_score": 0.0}

    cases = [((0, 0), 0), ((1, 2), 3), ((-4, 4), 0), ((10, 32), 42)]
    passed = 0
    for args, expected in cases:
        try:
            if add(*args) == expected:
                passed += 1
        except Exception:
            pass

    correctness = passed / len(cases)
    return {
        "correctness": correctness,
        "combined_score": correctness,
    }
'''


def make_mock_config(iterations: int = 3) -> Config:
    """Build a tiny, fully offline Config backed by the mock LLM."""
    config = Config()

    # No API key / base URL needed: name "mock" selects the mock backend.
    config.llm.models = [LLMModelConfig(name="mock", weight=1.0)]
    config.llm.evaluator_models = [LLMModelConfig(name="mock", weight=1.0)]
    config.llm.api_key = "not-needed"
    config.llm.timeout = 10
    config.llm.retries = 1

    config.max_iterations = iterations
    config.checkpoint_interval = max(1, iterations)
    config.random_seed = 42
    config.language = "python"
    config.diff_based_evolution = True
    config.log_level = "WARNING"

    # Keep the run tiny. A single parallel evaluation forces the robust
    # in-process execution path (no fragile worker process pool).
    config.evaluator.parallel_evaluations = 1

    # Keep the run tiny.
    config.database.population_size = 6
    config.database.archive_size = 3
    config.database.num_islands = 1
    config.database.in_memory = True
    config.database.log_prompts = False

    config.evaluator.timeout = 30
    config.evaluator.cascade_evaluation = False
    config.evaluator.use_llm_feedback = False
    config.evaluator.max_retries = 1

    return config


# ---------------------------------------------------------------------------
# Unit-level tests for the mock backend
# ---------------------------------------------------------------------------


class TestMockLLMBackend(unittest.TestCase):
    """The mock backend must satisfy the LLMInterface contract offline."""

    FENCE = "```"

    def _diff_prompt(self, code: str) -> str:
        return (
            "# Current Program\n"
            f"{self.FENCE}python\n{code}\n{self.FENCE}\n\n"
            "# Task\n"
            "You MUST use the exact SEARCH/REPLACE diff format shown below:\n"
            "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
        )

    def test_selected_by_model_name(self):
        model = _create_model(LLMModelConfig(name="mock"))
        self.assertIsInstance(model, MockLLM)

        model = _create_model(LLMModelConfig(name="mock-gpt"))
        self.assertIsInstance(model, MockLLM)

    def test_selected_by_provider(self):
        model = _create_model(LLMModelConfig(name="whatever", provider="mock"))
        self.assertIsInstance(model, MockLLM)

    def test_ensemble_uses_mock(self):
        ensemble = LLMEnsemble([LLMModelConfig(name="mock", weight=1.0)])
        self.assertEqual(len(ensemble.models), 1)
        self.assertIsInstance(ensemble.models[0], MockLLM)

    def test_generate_returns_applicable_diff(self):
        code = "# EVOLVE-BLOCK-START\ndef add(a, b):\n    return a + b\n# EVOLVE-BLOCK-END"
        llm = MockLLM(LLMModelConfig(name="mock"))

        response = asyncio.run(llm.generate(self._diff_prompt(code)))

        diffs = extract_diffs(response)
        self.assertTrue(diffs, f"mock produced no SEARCH/REPLACE blocks: {response}")

        child = apply_diff(code, response)
        self.assertNotEqual(child, code, "mock diff did not change the program")
        compile(child, "<child>", "exec")  # must stay valid Python

    def test_generate_is_deterministic(self):
        code = "# EVOLVE-BLOCK-START\ndef add(a, b):\n    return a + b\n# EVOLVE-BLOCK-END"
        prompt = self._diff_prompt(code)

        first = asyncio.run(MockLLM(LLMModelConfig(name="mock")).generate(prompt))
        second = asyncio.run(MockLLM(LLMModelConfig(name="mock")).generate(prompt))

        self.assertEqual(first, second)

    def test_successive_generations_progress(self):
        code = "# EVOLVE-BLOCK-START\ndef add(a, b):\n    return a + b\n# EVOLVE-BLOCK-END"

        llm = MockLLM(LLMModelConfig(name="mock"))
        child = apply_diff(code, asyncio.run(llm.generate(self._diff_prompt(code))))
        grandchild = apply_diff(
            child, asyncio.run(llm.generate(self._diff_prompt(child)))
        )

        self.assertNotEqual(grandchild, child)
        compile(grandchild, "<grandchild>", "exec")

    def test_full_rewrite_mode(self):
        code = "# EVOLVE-BLOCK-START\ndef add(a, b):\n    return a + b\n# EVOLVE-BLOCK-END"
        prompt = (
            "# Current Program\n"
            f"{self.FENCE}python\n{code}\n{self.FENCE}\n\n"
            "# Task\nRewrite the program. Provide the complete new program code.\n"
        )

        response = asyncio.run(MockLLM(LLMModelConfig(name="mock")).generate(prompt))

        self.assertIn(self.FENCE, response)
        from openevolve.utils.code_utils import parse_full_rewrite

        rewritten = parse_full_rewrite(response, "python")
        self.assertIsNotNone(rewritten)
        compile(rewritten, "<rewrite>", "exec")

    def test_generate_with_context_no_program(self):
        llm = MockLLM(LLMModelConfig(name="mock"))
        response = asyncio.run(
            llm.generate_with_context("system", [{"role": "user", "content": "hello"}])
        )
        self.assertIsInstance(response, str)
        self.assertTrue(response)


# ---------------------------------------------------------------------------
# End-to-end evolution smoke test
# ---------------------------------------------------------------------------


class TestMockEvolutionEndToEnd(unittest.TestCase):
    """Run the real engine end-to-end with the mock LLM."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="openevolve_mock_test_")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _run(self, iterations):
        # Reset the global mock call counter so the assertion is independent.
        import openevolve.llm.mock as mock_mod

        before = mock_mod.total_calls()
        result = run_evolution(
            initial_program=INITIAL_PROGRAM,
            evaluator=EVALUATOR_SOURCE,
            config=make_mock_config(iterations=iterations),
            iterations=iterations,
            output_dir=self.output_dir,
            cleanup=False,
        )
        after = mock_mod.total_calls()
        return result, after - before

    def test_run_evolution_with_mock_llm(self):
        result, mock_calls = self._run(3)

        self.assertIsInstance(result, EvolutionResult)
        self.assertIsNotNone(result.best_program)
        self.assertTrue(result.best_code.strip())
        self.assertIsInstance(result.metrics, dict)

        # The evaluator is deterministic and the initial program is correct,
        # so the best program must remain fully correct.
        self.assertAlmostEqual(result.best_score, 1.0, places=6)
        self.assertIn("correctness", result.metrics)

        # Best code must still be valid, working Python.
        compile(result.best_code, "<best>", "exec")
        self.assertIn("def add", result.best_code)

        # The offline mock LLM must have actually driven the generation steps.
        self.assertGreater(mock_calls, 0)

    def test_run_evolution_produces_output_dir(self):
        result, mock_calls = self._run(2)

        self.assertIsInstance(result, EvolutionResult)
        self.assertTrue(os.path.isdir(self.output_dir))
        self.assertGreater(mock_calls, 0)


if __name__ == "__main__":
    unittest.main()
