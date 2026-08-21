"""
Offline, deterministic tests for additional OpenEvolve core modules (wave 9).

Covers:
  * ``openevolve.evaluation_result`` - EvaluationResult dataclass + helpers
  * ``openevolve.utils.metrics_utils`` - safe numeric aggregation / fitness
  * ``openevolve.utils.format_utils`` - safe metric logging formatters
  * ``openevolve.prompt.templates`` - TemplateManager (default prompt loading)

None of these require API keys, network access or an external database.
"""

import unittest

from openevolve.evaluation_result import EvaluationResult
from openevolve.utils import metrics_utils
from openevolve.utils import format_utils
from openevolve.prompt.templates import TemplateManager


# ---------------------------------------------------------------------------
# evaluation_result
# ---------------------------------------------------------------------------


class TestEvaluationResult(unittest.TestCase):
    def test_default_construction_has_no_artifacts(self):
        result = EvaluationResult(metrics={"score": 0.9})
        self.assertEqual(result.metrics, {"score": 0.9})
        self.assertEqual(result.artifacts, {})
        self.assertFalse(result.has_artifacts())

    def test_from_dict_wraps_metrics(self):
        result = EvaluationResult.from_dict({"score": 0.5})
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.to_dict(), {"score": 0.5})

    def test_artifact_keys_and_existence(self):
        result = EvaluationResult(
            metrics={"score": 1.0},
            artifacts={"log": "hello", "data": b"\x00\x01"},
        )
        self.assertTrue(result.has_artifacts())
        self.assertEqual(set(result.get_artifact_keys()), {"log", "data"})

    def test_artifact_size_str_and_bytes(self):
        result = EvaluationResult(
            metrics={},
            artifacts={"log": "abc", "data": b"xy"},
        )
        self.assertEqual(result.get_artifact_size("log"), 3)
        self.assertEqual(result.get_artifact_size("data"), 2)
        self.assertEqual(result.get_artifact_size("missing"), 0)

    def test_total_artifact_size(self):
        result = EvaluationResult(
            metrics={},
            artifacts={"a": "abc", "b": b"xyzw"},
        )
        self.assertEqual(result.get_total_artifact_size(), 7)


# ---------------------------------------------------------------------------
# metrics_utils
# ---------------------------------------------------------------------------


class TestMetricsUtils(unittest.TestCase):
    def test_safe_numeric_average_empty(self):
        self.assertEqual(metrics_utils.safe_numeric_average({}), 0.0)

    def test_safe_numeric_average_excludes_strings_and_bools(self):
        metrics = {"score": 0.8, "flag": True, "label": "ok", "n": 0}
        # Only 0.8 and 0.0 are countable numerics -> average 0.4
        self.assertAlmostEqual(
            metrics_utils.safe_numeric_average(metrics), 0.4
        )

    def test_safe_numeric_average_ignores_nan(self):
        metrics = {"a": float("nan"), "b": 1.0}
        self.assertAlmostEqual(metrics_utils.safe_numeric_average(metrics), 1.0)

    def test_safe_numeric_sum(self):
        metrics = {"score": 2, "flag": True, "label": "x", "n": 3.0}
        # bool ("flag") is excluded, matching safe_numeric_average
        self.assertAlmostEqual(metrics_utils.safe_numeric_sum(metrics), 5.0)

    def test_get_fitness_score_prefers_combined_score(self):
        metrics = {"combined_score": 0.77, "score": 0.1}
        self.assertAlmostEqual(
            metrics_utils.get_fitness_score(metrics), 0.77
        )

    def test_get_fitness_score_excludes_feature_dimensions(self):
        metrics = {
            "combined_score": 0.9,
            "score": 0.5,
            "complexity": 100,
            "diversity": 200,
        }
        # complexity/diversity excluded when listed as features
        score = metrics_utils.get_fitness_score(
            metrics, feature_dimensions=["complexity", "diversity"]
        )
        self.assertAlmostEqual(score, 0.9)

    def test_get_fitness_score_falls_back_to_average(self):
        metrics = {"score": 0.4, "speed": 0.6}
        self.assertAlmostEqual(
            metrics_utils.get_fitness_score(metrics), 0.5
        )

    def test_format_feature_coordinates(self):
        metrics = {"complexity": 1.23456, "diversity": 5, "score": 0.9}
        out = metrics_utils.format_feature_coordinates(
            metrics, ["complexity", "diversity"]
        )
        self.assertEqual(out, "complexity=1.23, diversity=5.00")

    def test_format_feature_coordinates_empty(self):
        self.assertEqual(
            metrics_utils.format_feature_coordinates({"score": 0.9}, ["complexity"]),
            "",
        )


# ---------------------------------------------------------------------------
# format_utils
# ---------------------------------------------------------------------------


class TestFormatUtils(unittest.TestCase):
    def test_format_metrics_safe_empty(self):
        self.assertEqual(format_utils.format_metrics_safe({}), "")

    def test_format_metrics_safe_mixed_types(self):
        out = format_utils.format_metrics_safe(
            {"score": 0.5, "count": 3, "flag": True, "label": "ok"}
        )
        self.assertIn("score=0.5000", out)
        self.assertIn("count=3.0000", out)
        self.assertIn("flag=True", out)
        self.assertIn("label=ok", out)

    def test_format_improvement_safe_empty_inputs(self):
        self.assertEqual(format_utils.format_improvement_safe({}, {"a": 1}), "")
        self.assertEqual(format_utils.format_improvement_safe({"a": 1}, {}), "")

    def test_format_improvement_safe_numeric_diff(self):
        out = format_utils.format_improvement_safe(
            {"score": 0.2, "other": 1.0}, {"score": 0.5, "other": 1.0}
        )
        self.assertEqual(out, "score=+0.3000, other=+0.0000")

    def test_format_improvement_safe_ignores_non_numeric(self):
        out = format_utils.format_improvement_safe(
            {"label": "a"}, {"label": "b"}
        )
        self.assertEqual(out, "")


# ---------------------------------------------------------------------------
# prompt.templates
# ---------------------------------------------------------------------------


class TestTemplateManager(unittest.TestCase):
    def test_loads_default_templates(self):
        tm = TemplateManager()
        self.assertIn("diff_user", tm.templates)
        self.assertIn("system_message", tm.templates)
        self.assertTrue(tm.get_template("diff_user").strip())

    def test_get_template_missing_raises(self):
        tm = TemplateManager()
        with self.assertRaises(ValueError):
            tm.get_template("does_not_exist")

    def test_add_and_get_template(self):
        tm = TemplateManager()
        tm.add_template("custom", "hello {x}")
        self.assertEqual(tm.get_template("custom"), "hello {x}")

    def test_get_fragment_formats(self):
        tm = TemplateManager()
        out = tm.get_fragment("metrics_label", metrics="score=0.9")
        self.assertEqual(out, "Metrics: score=0.9")

    def test_get_fragment_missing(self):
        tm = TemplateManager()
        out = tm.get_fragment("no_such_fragment")
        self.assertTrue(out.startswith("[Missing fragment:"))

    def test_get_fragment_formatting_error(self):
        tm = TemplateManager()
        # fitness_improved expects {prev} and {current}; missing -> error string
        out = tm.get_fragment("fitness_improved")
        self.assertTrue(out.startswith("[Fragment formatting error:"))


if __name__ == "__main__":
    unittest.main()
