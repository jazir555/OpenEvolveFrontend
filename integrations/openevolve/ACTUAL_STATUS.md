# ACTUAL_STATUS: integrations/openevolve

**Verdict: NOT PRODUCTION-READY - import-fragile, depends on ~45 absent modules**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

25 flat modules wiring OpenEvolve to CrewAI, BubbleLabs, LeanAide, decomposition, workflow management and MCP. No __init__.py, no README, no tests.

## Measured facts

- Python files: **25**
- `python -m py_compile` on every file: **all 25 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **PARTIAL - `openevolve_structures` imports; most other modules need absent repo modules**
- `__init__.py` present: **NO (flat directory, namespace package only)**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports (break when imported as `integrations.openevolve.*`): **4**
  - `openevolve_api.py` -> `openevolve_structures`
  - `openevolve_bubblelabs_ui.py` -> `openevolve_workflow_manager_integrated`
  - `openevolve_crewai_adapter.py` -> `openevolve_crewai_delegation`
  - `openevolve_decomposition_adapter.py` -> `openevolve_enhanced_decomposition_integration`
- Referenced modules that are NOT importable here: **44** - `adversarial`, `analytics_manager`, `blue_team`, `bubblelabs_analytics`, `bubblelabs_crewai_bridge`, `bubblelabs_integration`, `bubblelabs_plugin_system`, `bubblelabs_ui_component`, `crewai_integration`, `crewai_state_management`, `crewai_zero_error_workflow`, `decomposition_engine`, `decomposition_recomposition_integration`, `end_to_end_invention_planner`, `enhanced_decomposition_engine`, `enhanced_recomposition_engine`, `evaluator_team`, `evolution` ...

## Honest notes

- This is NOT the BubbleLab OpenEvolve adapter under core-projects/ - it is a separate, much rougher flat directory.
- It references roughly 45 modules that are not importable from the repo root, including `openevolve_client`, `workflow_structures`, `workflow_engine`, `gauntlet_manager`, `team_manager`, `parameter_manager`, `leanaide_client`, `crewai_state_management`, `mdap_engine`, `maker_engine`, `red_team`, `blue_team`.
- `openevolve_imports.py` alone is a compatibility shim over ~15 absent modules.
- Only leaf modules such as `openevolve_structures` import successfully.

## Bottom line

NOT PRODUCTION-READY. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
