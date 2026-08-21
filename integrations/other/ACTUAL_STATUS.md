# ACTUAL_STATUS: integrations/other

**Verdict: MIXED DUMPING GROUND - not a coherent package; some files import, many do not**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

31 unrelated flat modules: steer_* context engine, datapizza_*, claudiomiro_*, rese_z3_*, mcp_* gateway/server/bridge, ragbits_*, dspy/dts/causal_learn/neuralkg/valkey integrations. No __init__.py, no README, no tests.

## Measured facts

- Python files: **31**
- `python -m py_compile` on every file: **all 31 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **PARTIAL - `steer_context_engine` imports (guarded); several modules need absent repo modules**
- `__init__.py` present: **NO (flat directory, namespace package only)**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports (break when imported as `integrations.other.*`): **5**
  - `claudiomiro_crewai_bridge.py` -> `claudiomiro_mcp_tools`
  - `datapizza_crewai_bridge.py` -> `datapizza_mcp_tools`
  - `dts_integration.py` -> `dspy_integration`
  - `steer_context_engine.py` -> `steer_crewai_bridge`, `steer_mcp_tools`
  - `steer_crewai_bridge.py` -> `steer_mcp_tools`
- Referenced modules that are NOT importable here: **20** - `ace_steer_config`, `adaptive_strategy_selector`, `alerting_system`, `backend`, `bubblelabs_crewai_bridge`, `bubblelabs_integration`, `bubblelabs_leanaide_integration`, `bubblelabs_node_completion`, `crewai_state_management`, `crewai_zero_error_workflow`, `datapizza`, `leanaide_autoformalization_mdap_maker`, `leanaide_client`, `leanaide_crewai_bridge`, `lmql`, `roma_crewai_bridge`, `roma_mdap_maker_engine`, `roma_openevolve_integration` ...

## Honest notes

- The steer_* files asked about specifically: `steer_context_engine.py` DOES import (its dependencies are guarded), but it needs the absent `steer` and `ace_steer_config` modules plus repo-level `knowledge_engine` for real work; `steer_crewai_bridge.py` and `steer_mcp_tools.py` need the absent `steer` package and `crewai_state_management`.
- Other absent third-party deps: `datapizza`, `valkey`, `lmql`.
- This directory should be split up or pruned; as-is it has no coherent public surface.

## Bottom line

MIXED DUMPING GROUND. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
