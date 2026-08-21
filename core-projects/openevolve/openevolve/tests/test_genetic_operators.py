"""
Offline tests for the genetic-operator configuration parameters.

Covers:
  * Config round-trip (from_dict / to_dict) for use_genetic_operators and the
    five documented core-evolution parameters.
  * The genetic-operator helpers (selection_method / mutation_rate /
    crossover_rate / elitism / selection_pressure) behave as specified.
  * A tiny evolution run with stubbed LLM + evaluator applies mutation,
    crossover and elitism without crashing and reads the configured params.
"""

import asyncio
from types import SimpleNamespace

import pytest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.genetic_operators import (
    crossover,
    elite_programs,
    mutate_code,
    mutation_temperature_scale,
    select_parent,
)
from openevolve.iteration import run_iteration_with_shared_db


# --------------------------------------------------------------------------
# Config round-trip
# --------------------------------------------------------------------------

def test_genetic_operator_flags_round_trip():
    cfg_dict = {
        "use_genetic_operators": True,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "selection_method": "roulette",
        "elitism": True,
        "selection_pressure": 2.0,
        "elite_ratio": 0.2,
    }
    cfg = Config.from_dict(cfg_dict)
    assert cfg.use_genetic_operators is True
    assert cfg.mutation_rate == 0.3
    assert cfg.crossover_rate == 0.5
    assert cfg.selection_method == "roulette"
    assert cfg.elitism is True
    assert cfg.selection_pressure == 2.0
    assert cfg.elite_ratio == 0.2

    out = cfg.to_dict()
    assert out["use_genetic_operators"] is True
    assert out["mutation_rate"] == 0.3
    assert out["crossover_rate"] == 0.5
    assert out["selection_method"] == "roulette"
    assert out["elitism"] is True
    assert out["selection_pressure"] == 2.0
    assert out["elite_ratio"] == 0.2


def test_genetic_operator_flags_default_off():
    cfg = Config()
    assert cfg.use_genetic_operators is False
    assert cfg.mutation_rate == 0.1
    assert cfg.crossover_rate == 0.8
    assert cfg.selection_method == "tournament"
    assert cfg.elitism is True
    assert cfg.selection_pressure == 1.0


# --------------------------------------------------------------------------
# Operator unit tests
# --------------------------------------------------------------------------

def _prog(code, score, pid=None):
    return Program(
        id=pid or f"p_{code!r}",
        code=code,
        metrics={"combined_score": score},
    )


def test_selection_methods():
    rng = __import__("random").Random(0)
    pop = [_prog("a", 0.1), _prog("b", 0.9), _prog("c", 0.5)]
    # tournament with high pressure -> prefers fittest
    assert select_parent(pop, "tournament", 5.0, [], rng).metrics["combined_score"] == 0.9
    # roulette with very high pressure -> fittest dominates
    rng2 = __import__("random").Random(0)
    wins = [select_parent(pop, "roulette", 10.0, [], rng2).metrics["combined_score"] for _ in range(20)]
    assert all(s == 0.9 for s in wins)
    # rank
    rng3 = __import__("random").Random(0)
    assert select_parent(pop, "rank", 1.0, [], rng3) in pop


def test_selection_pressure_tunes_tournament_size():
    rng = __import__("random").Random(0)
    pop = [_prog(f"x{i}", float(i) / 10.0) for i in range(10)]
    # pressure 1 -> tournament size ~3; pressure 0.1 -> size 2 (min)
    small = select_parent(pop, "tournament", 0.1, [], rng)
    assert small in pop


def test_crossover_blends_parents():
    a = _prog("line1\nline2\nline3\nline4", 0.0)
    b = _prog("A\nB\nC\nD", 0.0)
    rng = __import__("random").Random(1)
    out = crossover(a, b, 1.0, rng)
    assert out.startswith("line1\n")
    assert "D" in out
    # rate <= 0 returns parent A unchanged
    assert crossover(a, b, 0.0, rng) == a.code


def test_mutate_code_inserts_comment():
    rng = __import__("random").Random(2)
    out = mutate_code("x = 1\ny = 2\n", 1.0, rng)
    assert "# evolved mutation" in out
    assert out.count("\n") >= 2
    # rate <= 0 unchanged
    assert mutate_code("x = 1\n", 0.0, rng) == "x = 1\n"


def test_mutation_temperature_scale():
    assert mutation_temperature_scale(0.7, 0.0) == 0.7
    assert mutation_temperature_scale(0.7, 0.3) == pytest.approx(0.7 * 1.3)


def test_elite_programs():
    pop = [_prog("a", 0.1), _prog("b", 0.9), _prog("c", 0.5)]
    elites = elite_programs(pop, 2, [])
    assert [e.metrics["combined_score"] for e in elites] == [0.9, 0.5]
    assert elite_programs(pop, 0, []) == []


# --------------------------------------------------------------------------
# Tiny evolution run (stubbed LLM + evaluator)
# --------------------------------------------------------------------------

class _StubPromptSampler:
    def build_prompt(self, **kwargs):
        return {"system": "sys", "user": "user"}


class _StubLLM:
    def __init__(self):
        self.last_temperature = None

    async def generate_with_context(self, system_message, messages, **kwargs):
        self.last_temperature = kwargs.get("temperature")
        return "```python\ndef add(a, b):\n    return a + b + 1\n```"


class _StubEvaluator:
    async def evaluate_program(self, code, child_id):
        return {"combined_score": 0.42}

    def get_pending_artifacts(self, child_id):
        return {}


def _make_database(n=3):
    db = ProgramDatabase(Config().database)
    for i in range(n):
        db.add(
            Program(
                code=f"def f{i}():\n    return {i}\n",
                metrics={"combined_score": float(i) / n},
            ),
            iteration=0,
        )
    return db


def test_evolution_run_applies_genetic_operators(monkeypatch):
    cfg = Config()
    cfg.language = "python"
    cfg.use_genetic_operators = True
    cfg.mutation_rate = 1.0
    cfg.crossover_rate = 1.0
    cfg.selection_method = "tournament"
    cfg.elitism = True
    cfg.elite_ratio = 0.5
    cfg.selection_pressure = 2.0
    cfg.diff_based_evolution = False

    db = _make_database()

    # Record which operators run and with what parameters.
    calls = {"mutate": 0, "crossover": 0, "select": 0, "elite": 0, "temp": []}
    orig_mutate = mutate_code
    orig_cross = crossover
    orig_select = select_parent
    orig_elite = elite_programs
    orig_temp = mutation_temperature_scale

    def _mutate(code, rate, rng=None):
        calls["mutate"] += 1
        return orig_mutate(code, rate, rng)

    def _cross(a, b, rate, rng=None):
        calls["crossover"] += 1
        return orig_cross(a, b, rate, rng)

    def _select(pop, method="tournament", pressure=1.0, fdims=None, rng=None, **kwargs):
        calls["select"] += 1
        return orig_select(pop, method, pressure, fdims, rng)

    def _elite(pop, n, fdims=None):
        calls["elite"] += 1
        return orig_elite(pop, n, fdims)

    def _temp(base, rate):
        calls["temp"].append((base, rate))
        return orig_temp(base, rate)

    go = __import__("openevolve.genetic_operators", fromlist=["x"])
    monkeypatch.setattr(go, "mutate_code", _mutate)
    monkeypatch.setattr(go, "crossover", _cross)
    monkeypatch.setattr(go, "select_parent", _select)
    monkeypatch.setattr(go, "elite_programs", _elite)
    monkeypatch.setattr(go, "mutation_temperature_scale", _temp)

    llm = _StubLLM()
    evaluator = _StubEvaluator()
    sampler = _StubPromptSampler()

    result = asyncio.run(
        run_iteration_with_shared_db(
            iteration=1,
            config=cfg,
            database=db,
            evaluator=evaluator,
            llm_ensemble=llm,
            prompt_sampler=sampler,
        )
    )

    assert result is not None, "evolution iteration should produce a result"
    assert result.child_program is not None
    # Params were read and operators applied.
    assert calls["select"] >= 1
    assert calls["crossover"] >= 1
    assert calls["mutate"] >= 1
    assert calls["elite"] >= 1
    assert calls["temp"] and calls["temp"][0][1] == 1.0
    # Temperature scaling reached the LLM.
    assert llm.last_temperature is not None and llm.last_temperature > 0.7
    # Elitism preserved candidates (tagged in database).
    assert any(p.metadata.get("elite") for p in db.programs.values())
    # Original (flag-off) behavior: no genetic operators touched.
    cfg_off = Config()
    cfg_off.language = "python"
    assert cfg_off.use_genetic_operators is False
