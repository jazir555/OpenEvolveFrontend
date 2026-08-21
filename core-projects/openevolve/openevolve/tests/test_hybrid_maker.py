"""
Offline unit tests for the generic Hybrid MAKER strategies.

These run with a fake verification oracle and the built-in dependency-free
candidate generator, so no LeanAide / Z3 / LLM is required.
"""

import asyncio

import pytest

from openevolve.hybrid_maker import (
    AdaptiveMakerHybrid,
    CandidateGenerator,
    DefaultCandidateGenerator,
    MakerHybridStrategy,
    FullMakerHybrid,
    MakerAdversarialHybrid,
    MakerHybridConfig,
    MakerHybridMode,
    MakerMDAPParallel,
    MakerThenEvolution,
    MCTSThenMaker,
    VerificationOracle,
    VerificationResult,
    create_maker_hybrid,
    get_maker_hybrid_capabilities,
)


class FakeOracle(VerificationOracle):
    """A fake prover: accepts candidates containing the word 'induction'."""

    name = "fake"
    available = True

    def __init__(self):
        self.calls = 0

    def verify(self, problem, candidate, **kwargs):
        self.calls += 1
        ok = "induction" in candidate
        return VerificationResult(
            success=ok,
            score=1.0 if ok else 0.0,
            details="fake accept" if ok else "fake reject",
        )


class CountingGenerator(DefaultCandidateGenerator):
    """Default generator that records how many candidates it produced."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_generated = 0

    def generate(self, problem, n=1, population=None, **kwargs):
        out = super().generate(problem, n=n, population=population, **kwargs)
        self.total_generated += len(out)
        return out


def _make_strategy(cls, oracle, config=None, generator=None):
    return cls(oracle, config=config, candidate_generator=generator)


# --------------------------------------------------------------------------- #


def test_requires_oracle():
    with pytest.raises(ValueError):
        MCTSThenMaker(None)


def test_capabilities_reports_generic_status():
    caps = get_maker_hybrid_capabilities()
    assert caps["maker_hybrid_enabled"] is True
    assert "generic" in caps["integration_status"]
    assert set(caps["modes"]) == {m.value for m in MakerHybridMode}
    assert "FullMakerHybrid" in caps["strategies"]


def test_factory_builds_each_mode():
    oracle = FakeOracle()
    for mode in MakerHybridMode:
        strat = create_maker_hybrid(mode, oracle, config=MakerHybridConfig(random_seed=1))
        assert isinstance(strat, MakerHybridStrategy if mode != MakerHybridMode.FULL_MAKER_HYBRID else FullMakerHybrid)


def test_mcts_then_maker_produces_and_verifies():
    oracle = FakeOracle()
    strat = _make_strategy(MCTSThenMaker, oracle, config=MakerHybridConfig(random_seed=1))
    result = strat.run("n + 0 = n")
    assert len(result.candidates) > 0
    assert oracle.calls > 0
    assert result.oracle_calls == oracle.calls
    assert result.generations_completed >= 1
    # At least one candidate uses induction, so the run should succeed.
    assert result.success is True
    assert "induction" in result.best_candidate


def test_maker_then_evolution_runs_multiple_generations():
    oracle = FakeOracle()
    gen = CountingGenerator(seed=2)
    strat = _make_strategy(
        MakerThenEvolution,
        oracle,
        config=MakerHybridConfig(random_seed=2, evolution_generations=3, population_size=4),
        generator=gen,
    )
    result = strat.run("a + b = b + a")
    assert len(result.candidates) > 1
    assert result.generations_completed >= 2  # seed + generations
    assert oracle.calls > 0
    assert len(result.convergence_history) == result.generations_completed
    # Evolution must have generated offspring beyond the seed.
    assert gen.total_generated > 1


def test_adversarial_hybrid_consults_oracle_each_round():
    oracle = FakeOracle()
    strat = _make_strategy(
        MakerAdversarialHybrid,
        oracle,
        config=MakerHybridConfig(random_seed=3, adversarial_rounds=3, red_team_size=2, blue_team_size=2),
    )
    result = strat.run("forall n, n * 1 = n")
    assert result.generations_completed == 3
    assert result.oracle_calls >= 3 * (2 + 2)
    assert len(result.candidates) > 0


def test_adaptive_hybrid_terminates_on_convergence():
    oracle = FakeOracle()
    strat = _make_strategy(
        AdaptiveMakerHybrid,
        oracle,
        config=MakerHybridConfig(random_seed=4, convergence_threshold=0.95, population_size=4),
    )
    result = strat.run("x + 0 = x")
    assert result.success is True
    assert len(result.convergence_history) >= 1
    assert result.generations_completed <= (strat.config.evolution_generations + 2 + 1)


def test_mdap_parallel_runs_both_passes():
    oracle = FakeOracle()
    strat = _make_strategy(
        MakerMDAPParallel,
        oracle,
        config=MakerHybridConfig(random_seed=5, population_size=3),
    )
    result = strat.run("p ^ p = p")
    # Two parallel passes worth of candidates.
    assert len(result.candidates) >= 2 * 3
    assert result.oracle_calls >= 2 * 3


def test_full_hybrid_aggregates_all_variants():
    oracle = FakeOracle()
    strat = _make_strategy(
        FullMakerHybrid,
        oracle,
        config=MakerHybridConfig(random_seed=6, evolution_generations=2, adversarial_rounds=2),
    )
    result = strat.run("assoc add")
    assert result.success is True
    assert result.generations_completed > 0
    # Full pipeline exercises every variant, so the oracle is consulted heavily.
    assert oracle.calls > 10
    assert len(result.candidates) > 0


def test_oracle_budget_is_respected():
    class NeverOracle(FakeOracle):
        def verify(self, problem, candidate, **kwargs):
            return VerificationResult(success=False, score=0.0, details="never")

    oracle = NeverOracle()
    strat = _make_strategy(
        MCTSThenMaker,
        oracle,
        config=MakerHybridConfig(random_seed=7, mcts_simulations=50, max_oracle_calls=5),
    )
    result = strat.run("noop")
    assert oracle.calls <= 5
    assert result.oracle_calls <= 5
    assert result.success is False
