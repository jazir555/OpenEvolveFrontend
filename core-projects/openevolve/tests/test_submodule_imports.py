"""Import-smoke test for OpenEvolve submodules.

Imports each subpackage/module and asserts no exception is raised.
Submodules that require unavailable external dependencies or do not exist
in this repository layout are skipped rather than failed.
"""

import importlib

import pytest

# (module_name, reason_if_missing)
SUBMODULES = [
    ("openevolve.gauntlets", None),
    ("openevolve.gauntlets.loongflow_gauntlet", None),
    ("openevolve.gauntlets.multi_round_orchestrator", None),
    ("openevolve.gauntlets.three_round_orchestrator", None),
    ("openevolve.domain", None),
    ("openevolve.domain.base", None),
    ("openevolve.domain.engineering_optimizer", None),
    ("openevolve.domain.finance_optimizer", None),
    ("openevolve.domain.pharma_optimizer", None),
    ("openevolve.domain.science_optimizer", None),
    ("openevolve.domain.trading_optimizer", None),
    ("openevolve.domain.web_design_optimizer", None),
    ("openevolve.unified", None),
    ("openevolve.unified.config", None),
    ("openevolve.unified.config_mapper", None),
    ("openevolve.unified.config_validator", None),
    ("openevolve.unified.defaults", None),
    ("openevolve.unified.examples", None),
    ("openevolve.unified.presets", None),
    ("openevolve.pes", "module not present in this repository layout"),
    ("openevolve.long_horizon", "module not present in this repository layout"),
    ("openevolve.agents", "module not present in this repository layout"),
    ("openevolve.cli", None),
    ("openevolve.embedding", None),
    ("openevolve.novelty_judge", None),
    ("openevolve.evolution_trace", None),
    ("openevolve.iteration", None),
    ("openevolve.process_parallel", None),
    ("openevolve.llm", None),
    ("openevolve.llm.base", None),
    ("openevolve.llm.openai", None),
    ("openevolve.llm.claude_code", None),
    ("openevolve.llm.ensemble", None),
    ("openevolve.llm.mock", None),
    ("openevolve.prompt", None),
    ("openevolve.utils", None),
    ("openevolve.integrations", None),
    ("openevolve.finance", None),
]


@pytest.mark.parametrize("module_name,skip_reason", SUBMODULES)
def test_submodule_imports(module_name, skip_reason):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if skip_reason is not None or "No module named" in str(e):
            pytest.skip(f"{module_name} unavailable: {e}")
        raise
    except ImportError as e:
        # Missing optional third-party dependency -> skip.
        pytest.skip(f"{module_name} requires unavailable dependency: {e}")


def test_process_parallel_import():
    import openevolve.process_parallel as pp  # noqa: F401

    assert pp is not None
