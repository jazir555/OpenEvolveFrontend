"""
Offline, deterministic tests for stable OpenEvolve core modules.

Covers:
  * ``openevolve.config``  - Config construction and field defaults
  * ``openevolve.database`` - Program storage, sampling and best-program tracking
  * ``openevolve.evaluator`` - a trivial custom (file-backed) evaluator

None of these require API keys, network access or an external database.
"""

import asyncio
import os
import random
import tempfile
import unittest
from dataclasses import fields

from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMModelConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfigDefaults(unittest.TestCase):
    """The default ``Config`` must carry the documented defaults."""

    def test_default_construction(self):
        config = Config()
        self.assertEqual(config.max_iterations, 10000)
        self.assertEqual(config.checkpoint_interval, 100)
        self.assertEqual(config.log_level, "INFO")
        self.assertTrue(config.diff_based_evolution)
        self.assertEqual(config.language, None)
        self.assertEqual(config.file_suffix, ".py")

    def test_nested_component_defaults(self):
        config = Config()

        # Database defaults
        self.assertTrue(config.database.in_memory)
        self.assertEqual(config.database.population_size, 1000)
        self.assertEqual(config.database.archive_size, 100)
        self.assertEqual(config.database.num_islands, 5)
        self.assertEqual(config.database.feature_dimensions, ["complexity", "diversity"])

        # Evaluator defaults
        self.assertEqual(config.evaluator.timeout, 300)
        self.assertEqual(config.evaluator.max_retries, 3)
        self.assertEqual(config.evaluator.parallel_evaluations, 1)
        self.assertFalse(config.evaluator.use_llm_feedback)

        # LLM defaults
        self.assertEqual(config.llm.api_base, "https://api.openai.com/v1")
        self.assertEqual(config.llm.temperature, 0.7)
        self.assertEqual(config.llm.max_tokens, 4096)

    def test_llm_model_config_defaults(self):
        model = LLMModelConfig(name="mock")
        self.assertEqual(model.weight, 1.0)
        self.assertEqual(model.name, "mock")
        self.assertIsNone(model.api_key)

    def test_top_level_seed_propagates_to_database(self):
        # A freshly built DatabaseConfig carries its own default seed.
        db_config = DatabaseConfig()
        self.assertEqual(db_config.random_seed, 42)
        # When the database seed is left unset (None), from_dict wires the
        # top-level random_seed into the database component.
        config2 = Config.from_dict(
            {"random_seed": 99, "database": {"random_seed": None}}
        )
        self.assertEqual(config2.database.random_seed, 99)


class TestConfigFromDict(unittest.TestCase):
    """``Config.from_dict`` parses nested overrides and validates regex."""

    def test_from_dict_overrides(self):
        config = Config.from_dict(
            {
                "max_iterations": 5,
                "language": "python",
                "database": {"population_size": 12, "num_islands": 2},
                "evaluator": {"parallel_evaluations": 4},
            }
        )
        self.assertEqual(config.max_iterations, 5)
        self.assertEqual(config.language, "python")
        self.assertEqual(config.database.population_size, 12)
        self.assertEqual(config.database.num_islands, 2)
        self.assertEqual(config.evaluator.parallel_evaluations, 4)

    def test_from_dict_rejects_bad_regex(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"diff_pattern": "([invalid"})

    def test_llm_models_from_dict(self):
        config = Config.from_dict(
            {
                "llm": {
                    "models": [
                        {"name": "mock", "weight": 1.0},
                        {"name": "mock-gpt", "weight": 0.5},
                    ]
                }
            }
        )
        self.assertEqual(len(config.llm.models), 2)
        self.assertEqual(config.llm.models[0].name, "mock")
        self.assertEqual(config.llm.models[1].weight, 0.5)
        # Evaluator models default to a copy of the evolution models.
        self.assertEqual(len(config.llm.evaluator_models), 2)


class TestConfigEnvVar(unittest.TestCase):
    """``${VAR}`` references in ``api_key`` are resolved from the environment."""

    def setUp(self):
        self._prev = os.environ.get("OPENEVOLVE_TEST_KEY")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("OPENEVOLVE_TEST_KEY", None)
        else:
            os.environ["OPENEVOLVE_TEST_KEY"] = self._prev

    def test_env_var_resolution(self):
        os.environ["OPENEVOLVE_TEST_KEY"] = "secret-value"
        model = LLMModelConfig(name="mock", api_key="${OPENEVOLVE_TEST_KEY}")
        self.assertEqual(model.api_key, "secret-value")

    def test_missing_env_var_raises(self):
        os.environ.pop("OPENEVOLVE_TEST_KEY", None)
        with self.assertRaises(ValueError):
            LLMModelConfig(name="mock", api_key="${OPENEVOLVE_TEST_KEY}")

    def test_plain_api_key_untouched(self):
        model = LLMModelConfig(name="mock", api_key="literal")
        self.assertEqual(model.api_key, "literal")


class TestConfigRoundTrip(unittest.TestCase):
    """``to_dict`` / ``from_dict`` (and YAML) preserve configuration."""

    def test_to_dict_from_dict_round_trip(self):
        config = Config()
        config.max_iterations = 17
        config.database.population_size = 9
        restored = Config.from_dict(config.to_dict())
        self.assertEqual(restored.max_iterations, 17)
        self.assertEqual(restored.database.population_size, 9)
        self.assertEqual(restored.llm.temperature, config.llm.temperature)

    def test_yaml_round_trip(self):
        config = Config()
        config.max_iterations = 23
        config.database.num_islands = 3
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.yaml")
            config.to_yaml(path)
            restored = Config.from_yaml(path)
        self.assertEqual(restored.max_iterations, 23)
        self.assertEqual(restored.database.num_islands, 3)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _make_program(code_id, score, island=None):
    prog = Program(code=f"x = {code_id}")
    prog.score = float(score)
    prog.metrics = {"combined_score": float(score), "correctness": float(score)}
    if island is not None:
        prog.metadata = {"island": island}
    return prog


class TestProgramDatabase(unittest.TestCase):
    """Offline mechanics of the in-memory program database."""

    def setUp(self):
        self.config = DatabaseConfig(
            in_memory=True,
            population_size=10,
            archive_size=5,
            num_islands=3,
            random_seed=42,
            log_prompts=False,
        )
        self.db = ProgramDatabase(self.config)

    def test_add_and_count(self):
        pid = self.db.add(_make_program(1, 0.5))
        self.assertIn(pid, self.db.programs)
        self.assertEqual(len(self.db.programs), 1)

    def test_best_program_tracking(self):
        self.db.add(_make_program(1, 0.3))
        self.db.add(_make_program(2, 0.9))
        self.db.add(_make_program(3, 0.6))
        best = self.db.get_best_program()
        self.assertIsNotNone(best)
        self.assertEqual(best.score, 0.9)

    def test_best_program_by_metric(self):
        self.db.add(_make_program(1, 0.3))
        self.db.add(_make_program(2, 0.9))
        best = self.db.get_best_program(metric="correctness")
        self.assertEqual(best.metrics["correctness"], 0.9)

    def test_best_program_empty(self):
        self.assertIsNone(self.db.get_best_program())

    def test_child_inherits_parent_island(self):
        parent = _make_program(1, 0.5)
        parent_id = self.db.add(parent, target_island=2)
        child = _make_program(2, 0.7)
        child.parent_id = parent_id
        self.db.add(child)
        # Child with no explicit island but a parent on island 2 should land there.
        self.assertIn(child.id, self.db.islands[2])
        self.assertEqual(child.metadata["island"], 2)

    def test_sample_returns_valid_parent_and_inspirations(self):
        for i in range(5):
            self.db.add(_make_program(i, 0.2 + i * 0.1))
        parent, inspirations = self.db.sample(num_inspirations=2)
        self.assertIn(parent.id, self.db.programs)
        self.assertLessEqual(len(inspirations), 2)
        for insp in inspirations:
            self.assertIn(insp.id, self.db.programs)

    def test_sample_from_island_is_deterministic(self):
        # All programs in island 0 for a focused, seed-controlled test.
        for i in range(5):
            self.db.add(_make_program(i, 0.2 + i * 0.1, island=0))
        # Re-seed explicitly so the comparison is independent of any global
        # random state left behind by other tests in the session.
        random.seed(self.config.random_seed)
        first_parent, first_insp = self.db.sample_from_island(0, num_inspirations=2)
        random.seed(self.config.random_seed)
        second_parent, second_insp = self.db.sample_from_island(0, num_inspirations=2)
        self.assertEqual(first_parent.id, second_parent.id)
        self.assertEqual([p.id for p in first_insp], [p.id for p in second_insp])

    def test_duplicate_code_does_not_inflate_population(self):
        # With a single "complexity" feature dimension, two programs with
        # identical code share the same MAP-Elites cell: the better one wins and
        # the weaker is displaced (and orphan-removed).
        db = ProgramDatabase(
            DatabaseConfig(
                in_memory=True,
                population_size=10,
                num_islands=1,
                random_seed=42,
                log_prompts=False,
                feature_dimensions=["complexity"],
            )
        )
        weak = _make_program(1, 0.2)
        strong = _make_program(1, 0.8)
        db.add(weak)
        count_before = len(db.programs)
        db.add(strong)
        # The weaker program should be evicted; population stays at one cell.
        self.assertEqual(len(db.programs), count_before)
        self.assertIn(strong.id, db.programs)
        self.assertNotIn(weak.id, db.programs)

    def test_orphaned_program_is_removed(self):
        # Directly exercise the (formerly missing) _remove_program_if_orphaned
        # helper: a non-best program that owns no cell and belongs to no island
        # is purged, while the tracked best program is always protected.
        self.db.add(_make_program(0, 0.9))  # becomes the best program
        pid = self.db.add(_make_program(1, 0.1))  # not the best
        self.db.islands[0].discard(pid)
        for feature_map in self.db.island_feature_maps:
            for key in list(feature_map.keys()):
                if feature_map[key] == pid:
                    del feature_map[key]
        self.db._remove_program_if_orphaned(pid)
        self.assertNotIn(pid, self.db.programs)

        # The absolute best program must never be orphan-removed.
        best_id = self.db.best_program_id
        self.db.islands[0].discard(best_id)
        for feature_map in self.db.island_feature_maps:
            for key in list(feature_map.keys()):
                if feature_map[key] == best_id:
                    del feature_map[key]
        self.db._remove_program_if_orphaned(best_id)
        self.assertIn(best_id, self.db.programs)


# ---------------------------------------------------------------------------
# Evaluator scaffolding (trivial custom evaluator)
# ---------------------------------------------------------------------------


class TestCustomEvaluator(unittest.TestCase):
    """A minimal file-backed evaluator must run offline and return scores."""

    def test_trivial_custom_evaluator(self):
        eval_source = (
            "def evaluate(program_path):\n"
            "    with open(program_path) as f:\n"
            "        code = f.read()\n"
            "    # Score 1.0 when the program defines a working adder.\n"
            "    correct = 'return a + b' in code\n"
            "    score = 1.0 if correct else 0.0\n"
            "    return {'correctness': score, 'combined_score': score}\n"
        )

        good_program = (
            "def add(a, b):\n"
            "    return a + b\n"
        )
        bad_program = "def add(a, b):\n    return 0\n"

        with tempfile.TemporaryDirectory() as tmp:
            eval_path = os.path.join(tmp, "eval.py")
            with open(eval_path, "w") as f:
                f.write(eval_source)

            config = EvaluatorConfig(parallel_evaluations=1, timeout=30, cascade_evaluation=False)
            evaluator = Evaluator(config, eval_path)

            good = asyncio.run(evaluator.evaluate_program(good_program, "good"))
            bad = asyncio.run(evaluator.evaluate_program(bad_program, "bad"))

        self.assertAlmostEqual(good["correctness"], 1.0)
        self.assertAlmostEqual(bad["correctness"], 0.0)

    def test_missing_eval_file_raises(self):
        config = EvaluatorConfig()
        with self.assertRaises(ValueError):
            Evaluator(config, "/nonexistent/path/eval.py")


if __name__ == "__main__":
    unittest.main()
