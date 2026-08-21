# ACTUAL_STATUS: integrations/crewai

**Verdict: NOT PRODUCTION-READY - import-fragile, most modules need absent repo modules**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

17 flat modules: research core/tools/templates/external, MDAP maker engine, hub, API routes, ICR glue. No __init__.py, no README, no tests.

## Measured facts

- Python files: **17**
- `python -m py_compile` on every file: **all 17 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **PARTIAL - `crewai_research_core` imports; other modules need absent repo modules**
- `__init__.py` present: **NO (flat directory, namespace package only)**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports (break when imported as `integrations.crewai.*`): **3**
  - `crewai_api_routes.py` -> `crewai_hub`
  - `crewai_hub.py` -> `crewai_integration_complete`
  - `demo_crewai_research_features.py` -> `crewai_research_core`, `crewai_research_external`, `crewai_research_templates`, `crewai_research_tools`
- Referenced modules that are NOT importable here: **19** - `Bio`, `ace_crewai_bridge`, `adaptive_strategy_selector`, `alerting_system`, `arxiv`, `bubblelabs_maker_integration`, `crewai_client`, `crewai_state_management`, `crewai_unified_bridge`, `crewai_unified_flow`, `crewai_zero_error_workflow`, `decomposition_crewai_bridge`, `enhanced_decomposition_engine`, `enhanced_recomposition_engine`, `leanaide_client`, `openevolve_enhanced_decomposition_integration`, `scholarly`, `sovereign_data_models` ...

## Honest notes

- Missing importable modules referenced here include `crewai_state_management`, `crewai_client`, `crewai_unified_bridge`, `crewai_zero_error_workflow`, `ace_crewai_bridge`, `enhanced_decomposition_engine`, `leanaide_client`, `sovereign_data_models`, `sovereign_refinement`.
- Optional third-party research deps also absent: `arxiv`, `scholarly`, `Bio` (biopython).
- 3 files use flat sibling imports (`import crewai_hub`, etc.) that only resolve if integrations/crewai is itself on sys.path.

## Bottom line

NOT PRODUCTION-READY. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
