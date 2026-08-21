"""
Tests for the LoongFlow adapter.

These exercise the previously-broken default path: when the LoongFlow agent
lacks a ``run``/``run_sync`` callable, ``evolve()`` must NOT raise
``NotImplementedError``. Instead it dispatches through a generic callable
detection and, failing that, drives a real evolution through OpenEvolve's
engine (offline mock LLM). All tests run without network access or API keys.
"""

import asyncio

import pytest

from openevolve.integrations.loongflow_adapter import LoongFlowAdapter


def _force_loongflow(adapter: LoongFlowAdapter, agent) -> None:
    """Make an adapter behave as if LoongFlow was selected (no real import)."""
    adapter.using_loongflow = True
    adapter.mode = "loongflow"
    adapter.pes_agent = agent


def test_default_openevolve_fallback_returns_result():
    """Default path (LoongFlow disabled/unavailable) returns a real result."""
    config = {
        "max_iterations": 3,
        "enable_loongflow": False,
        "mode": "standard",
    }
    adapter = LoongFlowAdapter(config)
    assert adapter.using_loongflow is False

    result = asyncio.run(
        adapter.evolve(problem="Find the best sorting algorithm", domain="code")
    )

    assert isinstance(result, dict)
    assert result.get("system_used") == "openevolve"
    assert "best_fitness" in result
    assert result.get("error") is None or "best_solution" in result


def test_no_agent_callable_falls_back_to_engine():
    """Agent with no run/run_sync/etc. callable -> engine fallback (no error)."""

    class DuckAgent:
        pass

    config = {"max_iterations": 2}
    adapter = LoongFlowAdapter(config)
    _force_loongflow(adapter, DuckAgent())

    result = asyncio.run(
        adapter.evolve(
            problem="Optimize f(x)=x^2",
            domain="math",
            initial_code="# EVOLVE-BLOCK-START\ndef solve(x):\n    return x\n# EVOLVE-BLOCK-END\n",
        )
    )

    assert isinstance(result, dict)
    # Engine fallback drives real OpenEvolve evolution.
    assert result.get("system_used") == "openevolve"
    assert "best_fitness" in result
    assert "best_solution" in result


def test_agent_run_async_path():
    """Documented async ``run`` entrypoint is used when present."""

    class RunAgent:
        async def run(self, problem_data):
            return {
                "best_solution": "x = 1",
                "best_fitness": 0.91,
                "total_evaluations": 2,
                "iterations_performed": 1,
                "convergence_curve": [0.1, 0.91],
                "planning_strategies": ["p"],
                "execution_patterns": [],
                "summaries": [],
            }

    config = {"max_iterations": 2}
    adapter = LoongFlowAdapter(config)
    _force_loongflow(adapter, RunAgent())

    result = asyncio.run(adapter.evolve(problem="solve", domain="code"))
    assert result["best_fitness"] == 0.91
    assert result["best_solution"] == "x = 1"


def test_agent_run_sync_path():
    """Documented sync ``run_sync`` entrypoint is used when present."""

    class SyncAgent:
        def run_sync(self, problem_data):
            return {
                "best_solution": "y = 2",
                "best_fitness": 0.42,
                "total_evaluations": 1,
                "iterations_performed": 1,
                "convergence_curve": [0.42],
                "planning_strategies": [],
                "execution_patterns": [],
                "summaries": [],
            }

    config = {"max_iterations": 2}
    adapter = LoongFlowAdapter(config)
    _force_loongflow(adapter, SyncAgent())

    result = asyncio.run(adapter.evolve(problem="solve", domain="math"))
    assert result["best_fitness"] == 0.42


@pytest.mark.parametrize(
    "attr", ["invoke", "predict", "generate", "chat", "__call__"]
)
def test_generic_callable_dispatch(attr):
    """Other common agent callables are discovered and invoked."""
    captured = {}

    class GenericAgent:
        def invoke(self, pd):
            captured["pd"] = pd
            return {"best_solution": "i", "best_fitness": 0.5}

        def predict(self, pd):
            captured["pd"] = pd
            return {"best_solution": "p", "best_fitness": 0.5}

        def generate(self, pd):
            captured["pd"] = pd
            return {"best_solution": "g", "best_fitness": 0.5}

        def chat(self, pd):
            captured["pd"] = pd
            return {"best_solution": "c", "best_fitness": 0.5}

        def __call__(self, pd):
            captured["pd"] = pd
            return {"best_solution": "call", "best_fitness": 0.5}

    config = {"max_iterations": 2}
    adapter = LoongFlowAdapter(config)
    _force_loongflow(adapter, GenericAgent())

    result = asyncio.run(adapter.evolve(problem="dispatch", domain="code"))
    assert captured.get("pd") is not None
    assert result["best_fitness"] == 0.5


def test_resolve_agent_callable_prefers_run():
    """_resolve_agent_callable prefers run over generic callables."""
    adapter = LoongFlowAdapter({"max_iterations": 1})

    class Agent:
        def run(self, pd):
            return None

        def predict(self, pd):
            return None

    fn, is_async = adapter._resolve_agent_callable(Agent())
    assert fn.__name__ == "run"
    assert is_async is False


def test_resolve_agent_callable_returns_none_for_empty():
    """A plain object with no callables resolves to (None, False)."""
    adapter = LoongFlowAdapter({"max_iterations": 1})

    class Empty:
        pass

    fn, is_async = adapter._resolve_agent_callable(Empty())
    assert fn is None
    assert is_async is False
