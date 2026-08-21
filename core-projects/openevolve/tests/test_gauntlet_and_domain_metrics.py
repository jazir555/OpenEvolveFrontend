"""
Offline tests for the gauntlet Red/Gold rounds and the domain optimizers.

The gauntlet rounds are exercised through :class:`openevolve.llm.mock.MockLLM`
(selected automatically when no judge model is configured), so these tests need
no API keys and make no network calls. The domain metrics are pure deterministic
heuristics, so they are asserted directly.
"""

import asyncio
import unittest

from openevolve.domain import get_optimizer
from openevolve.gauntlets.llm_judge import (
    GauntletJudge,
    parse_verdict,
    probe_solution,
    robustness_from_probes,
    verify_solution,
)
from openevolve.gauntlets.multi_round_orchestrator import (
    MultiRoundConfig,
    MultiRoundGauntletOrchestrator,
)
from openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundConfig,
    ThreeRoundGauntletOrchestrator,
)
from openevolve.llm.mock import total_calls

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROBUST_SOLUTION = '''
def mean(values):
    """Return the arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    assert all(isinstance(value, (int, float)) for value in values)
    try:
        return sum(values) / len(values)
    except TypeError as exc:
        raise ValueError("values must be numeric") from exc
'''

FRAGILE_SOLUTION = """
def mean(values):
    api_key = "sk-super-secret-value"
    total = 0
    for value in values:
        total = total + value
    return total / len(values)
"""

PROBLEM = "Compute the mean of a list of numbers"


class TestGauntletJudge(unittest.TestCase):
    """The Red/Gold judges must produce real verdicts offline"""

    def test_offline_judge_uses_mock_backend(self):
        judge = GauntletJudge()
        self.assertTrue(any("mock" in name for name in judge.model_names))

    def test_red_team_verdict_is_parsed_and_deterministic(self):
        judge = GauntletJudge()
        calls_before = total_calls()

        first = asyncio.run(
            judge.red_team(solution=ROBUST_SOLUTION, problem=PROBLEM, domain="general")
        )
        second = asyncio.run(
            judge.red_team(solution=ROBUST_SOLUTION, problem=PROBLEM, domain="general")
        )

        self.assertGreater(total_calls(), calls_before)  # the LLM was really called
        self.assertTrue(first.parsed)
        self.assertIsInstance(first.score, float)
        self.assertGreater(first.score, 0.0)
        self.assertLessEqual(first.score, 1.0)
        self.assertEqual(first.score, second.score)
        self.assertTrue(first.feedback)

    def test_gold_team_returns_one_vote_per_model(self):
        judge = GauntletJudge()
        votes = asyncio.run(
            judge.gold_team(solution=ROBUST_SOLUTION, problem=PROBLEM, domain="general")
        )
        self.assertEqual(len(votes), len(judge.model_names))
        self.assertTrue(all(vote.parsed for vote in votes))
        self.assertTrue(all(0.0 < vote.score <= 1.0 for vote in votes))

    def test_parse_verdict_prefers_explicit_score_and_rescales(self):
        verdict = parse_verdict(
            '```json\n{"overall_score": 8.5, "vulnerabilities": ["off-by-one"], '
            '"reasoning": "mostly fine"}\n```',
            model="test",
        )
        self.assertTrue(verdict.parsed)
        self.assertAlmostEqual(verdict.score, 0.85)
        self.assertEqual(verdict.findings, ["off-by-one"])
        self.assertIn("mostly fine", verdict.feedback)

    def test_parse_verdict_marks_unusable_responses(self):
        verdict = parse_verdict("I refuse to answer.", model="test")
        self.assertFalse(verdict.parsed)
        self.assertEqual(verdict.score, 0.0)


class TestStaticProbes(unittest.TestCase):
    """The deterministic attack probes must discriminate real weaknesses"""

    def test_fragile_solution_loses_more_attacks(self):
        robust = probe_solution(ROBUST_SOLUTION)
        fragile = probe_solution(FRAGILE_SOLUTION)

        self.assertGreater(len(robust), 0)
        self.assertGreater(
            robustness_from_probes(robust), robustness_from_probes(fragile)
        )
        names = {probe["name"] for probe in fragile if probe["successful"]}
        self.assertIn("credential_exposure", names)

    def test_markup_probes_used_for_html(self):
        probes = probe_solution('<div><img src="a.png"><form></form></div>')
        names = {probe["name"] for probe in probes}
        self.assertIn("form_validation_bypass", names)
        self.assertIn("accessibility_regression", names)

    def test_verification_rejects_broken_code(self):
        self.assertTrue(verify_solution(ROBUST_SOLUTION)["passed"])
        self.assertFalse(verify_solution("def broken(:\n    pass")["passed"])


class TestThreeRoundGauntlet(unittest.TestCase):
    """Rounds 2 and 3 must produce real scores, not placeholders"""

    def _run(self, solution):
        config = ThreeRoundConfig(
            round1_enabled=False,
            round2_threshold=0.6,
            round3_threshold=0.6,
            enable_early_termination=False,
        )
        orchestrator = ThreeRoundGauntletOrchestrator(config)
        return asyncio.run(orchestrator.run_full_gauntlet(solution, PROBLEM, "general"))

    def test_rounds_report_real_attacks_and_votes(self):
        result = self._run(ROBUST_SOLUTION)

        round2 = result.round2_result
        self.assertIsNotNone(round2)
        self.assertGreater(round2.attacks_attempted, 0)
        self.assertGreater(round2.robustness_score, 0.0)
        self.assertNotEqual(round2.score, 0.75)  # the old placeholder value
        self.assertTrue(round2.attack_details)

        round3 = result.round3_result
        self.assertIsNotNone(round3)
        self.assertNotEqual(round3.score, 0.85)  # the old placeholder value
        self.assertTrue(round3.evaluator_votes)
        self.assertTrue(round3.formal_verification_passed)
        self.assertGreater(result.final_score, 0.0)

    def test_fragile_solution_scores_lower(self):
        robust = self._run(ROBUST_SOLUTION)
        fragile = self._run(FRAGILE_SOLUTION)

        # Deterministic probe outcomes must separate the two candidates
        self.assertGreater(
            robust.round2_result.robustness_score,
            fragile.round2_result.robustness_score,
        )
        self.assertGreater(
            fragile.round2_result.attacks_successful,
            robust.round2_result.attacks_successful,
        )

        # Static verification must certify only the robust candidate
        self.assertTrue(robust.round3_result.formal_verification_passed)
        self.assertFalse(fragile.round3_result.formal_verification_passed)
        self.assertTrue(fragile.round3_result.evaluator_votes)

    def test_default_config_reaches_rounds_2_and_3(self):
        """A degraded Round 1 screen must not gate the Red/Gold rounds"""
        orchestrator = ThreeRoundGauntletOrchestrator(ThreeRoundConfig())
        result = asyncio.run(
            orchestrator.run_full_gauntlet(ROBUST_SOLUTION, PROBLEM, "general")
        )

        self.assertEqual(result.rounds_completed, 3)
        self.assertIsNotNone(result.round2_result)
        self.assertIsNotNone(result.round3_result)
        self.assertIsNone(result.termination_reason)


class TestMultiRoundGauntlet(unittest.TestCase):
    """The multi-round orchestrator must run Rounds 2 and 3 for real"""

    def test_rounds_2_and_3_are_evaluated(self):
        orchestrator = MultiRoundGauntletOrchestrator(
            MultiRoundConfig(enable_early_termination=False, fail_fast=False)
        )
        state = asyncio.run(
            orchestrator.execute_full_gauntlet(ROBUST_SOLUTION, PROBLEM, "general")
        )

        self.assertIn(2, state.rounds_completed)
        self.assertIn(3, state.rounds_completed)

        round2 = state.round2_result
        self.assertGreater(round2.attacks_attempted, 0)
        self.assertTrue(round2.edge_cases_tested)
        self.assertNotEqual(round2.score, 50.0)  # the old neutral fallback

        round3 = state.round3_result
        self.assertTrue(round3.judge_scores)
        self.assertNotEqual(round3.score, 5.0)  # the old neutral fallback
        self.assertGreater(orchestrator.calculate_final_score(state), 0.0)


class TestDomainMetrics(unittest.TestCase):
    """Domain optimizers must return real, discriminating metrics"""

    RICH_PAGE = """
    <html lang="en"><head><title>Ship faster</title>
    <meta name="description" content="Automated experiments" />
    <link rel="canonical" href="https://example.com/" /></head>
    <body><main><h1>Ship faster</h1><h2>Trusted by 10,000 customers</h2>
    <section><p>Verified reviews, secure checkout and a money-back guarantee.</p>
    <img src="hero.png" alt="Product screenshot" /><video src="demo.mp4"></video></section>
    <form aria-label="signup"><label for="email">Email</label>
    <input id="email" type="email" required /><button type="submit">Start free trial</button>
    </form></main></body></html>
    """

    POOR_PAGE = '<div><img src="a.png"><input><input><input><input><script>x()</script></div>'

    GOOD_STRATEGY = """
    def strategy(bars):
        rsi_period = 14
        stop_loss = 0.02
        take_profit = 0.06
        position_size = 0.1
        if regime_filter(bars) and macd(bars) > 0:
            entry = True
        if atr(bars) > 3 * stop_loss:
            exit = True
        return entry, exit
    """

    OVERFIT_STRATEGY = """
    def strategy(bars):
        a = 1.1
        b = 2.2
        c = 3.3
        d = 4.4
        e = 5.5
        f = 6.6
        g = 7.7
        h = 8.8
        i = 9.9
        return a + b + c + d + e + f + g + h + i
    """

    def test_web_design_metrics_are_computed_from_markup(self):
        optimizer = get_optimizer("web_design")
        rich = optimizer.evaluate_solution(self.RICH_PAGE, "Optimize for conversion")
        poor = optimizer.evaluate_solution(self.POOR_PAGE, "Optimize for conversion")

        for metrics in (rich, poor):
            self.assertEqual(set(metrics), set(optimizer.get_domain_metrics()))
            self.assertTrue(all(isinstance(value, float) for value in metrics.values()))

        self.assertGreater(rich["accessibility_score"], poor["accessibility_score"])
        self.assertGreater(rich["seo_score"], poor["seo_score"])
        self.assertGreater(rich["conversion_rate"], poor["conversion_rate"])
        self.assertLess(rich["bounce_rate"], poor["bounce_rate"])
        # The old placeholder always returned exactly these constants
        self.assertNotEqual(rich["conversion_rate"], 0.05)
        self.assertNotEqual(rich["seo_score"], 0.78)

    def test_trading_metrics_reward_risk_management(self):
        optimizer = get_optimizer("trading")
        good = optimizer.evaluate_solution(
            self.GOOD_STRATEGY, "Momentum strategy", {"max_drawdown": 0.2}
        )
        overfit = optimizer.evaluate_solution(
            self.OVERFIT_STRATEGY, "Momentum strategy", {"max_drawdown": 0.2}
        )

        self.assertGreater(good["sharpe_ratio"], overfit["sharpe_ratio"])
        self.assertGreater(good["total_return"], overfit["total_return"])
        self.assertLess(good["max_drawdown"], overfit["max_drawdown"])
        self.assertNotEqual(good["sharpe_ratio"], 1.8)  # old placeholder

        # Deterministic across calls
        self.assertEqual(
            good,
            optimizer.evaluate_solution(
                self.GOOD_STRATEGY, "Momentum strategy", {"max_drawdown": 0.2}
            ),
        )

    def test_finance_metrics_reward_diversification(self):
        optimizer = get_optimizer("finance")
        diversified = optimizer.evaluate_solution(
            '{"AAPL": 0.25, "MSFT": 0.25, "TLT": 0.25, "GLD": 0.25}',
            "Maximize risk-adjusted return",
        )
        concentrated = optimizer.evaluate_solution(
            '{"AAPL": 1.0}', "Maximize risk-adjusted return"
        )
        empty = optimizer.evaluate_solution("no portfolio here", "Maximize return")

        self.assertLess(diversified["volatility"], concentrated["volatility"])
        self.assertLess(diversified["max_drawdown"], concentrated["max_drawdown"])
        self.assertNotEqual(diversified["sharpe_ratio"], 1.5)  # old placeholder
        self.assertEqual(empty["sharpe_ratio"], 0.0)

    def test_remaining_domains_return_real_scores(self):
        cases = {
            "science": (
                "n = 40\nreplicates = 3\nseed = 7\nprotocol randomized control blinding\n"
                "temp = [20, 40]\nanova p_value",
                {"max_experiments": 20},
            ),
            "engineering": (
                "safety_factor = 2.6\nweight = 800\ncost = 4000\n"
                "fea mesh stress fatigue tolerance thermal aluminum",
                {"max_weight": 1000, "max_cost": 5000},
            ),
            "pharma": ("SMILES: CC(=O)Oc1ccccc1C(=O)O", {"max_toxicity": 0.3}),
        }

        for domain, (solution, constraints) in cases.items():
            optimizer = get_optimizer(domain)
            metrics = optimizer.evaluate_solution(solution, f"Optimize {domain}", constraints)
            self.assertTrue(metrics, f"{domain} returned no metrics")
            self.assertTrue(
                all(isinstance(value, float) for value in metrics.values()),
                f"{domain} returned non-float metrics: {metrics}",
            )
            self.assertTrue(
                any(value > 0.0 for value in metrics.values()),
                f"{domain} returned only zeros: {metrics}",
            )

    def test_pharma_penalizes_non_drug_like_molecules(self):
        optimizer = get_optimizer("pharma")
        aspirin = optimizer.evaluate_solution("CC(=O)Oc1ccccc1C(=O)O", "Optimize lead")
        greasy = optimizer.evaluate_solution(
            "CCCCCCCCCCCCCCCCCC(=O)N(C)Cc1ccc(Cl)cc1[N+](=O)[O-]", "Optimize lead"
        )

        self.assertGreater(aspirin["drug_likeness"], greasy["drug_likeness"])
        self.assertGreater(aspirin["solubility"], greasy["solubility"])
        self.assertLess(aspirin["toxicity"], greasy["toxicity"])


if __name__ == "__main__":
    unittest.main()
