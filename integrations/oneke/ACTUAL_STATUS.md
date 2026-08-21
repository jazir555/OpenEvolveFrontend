# ACTUAL_STATUS: integrations/oneke

**Verdict: MOST COMPLETE PACKAGE HERE - real test suite, mostly passing, but core extraction needs an uninstalled dependency**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

OneKE knowledge extraction with reflection, quality enhancement and case-based learning (Phase 4 'enhanced' layer).

## Measured facts

- Python files: **11**
- `python -m py_compile` on every file: **all 11 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.oneke` succeeds**
- `__init__.py` present: **yes**
- Test files: **test_enhanced.py, verify_installation.py**
- Docs: **ENHANCED_README.md, FILE_STRUCTURE.md, INTEGRATION_COMPLETE.md, PHASE4_IMPLEMENTATION_SUMMARY.md, QUICKSTART.md**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **2** - `src`, `workflow_structures`

## Honest notes

- This is the only subpackage with a genuine, runnable test suite. Measured result (existing tests, nothing added):
-   `pytest integrations/oneke/test_enhanced.py --asyncio-mode=auto` -> **13 passed, 2 skipped, 2 failed**.
-   The 2 failures (`test_full_enhancement_pipeline`, `test_learning_loop`) fail with `KeyError: 'total_cases'` because the bridge could not initialise: logs show `OneKE is NOT installed` and `OPENAI_API_KEY not set`, so `get_repository_statistics()` returns `{'error': ...}` instead of stats.
-   Also logged: `SchemaDefinition.__init__() missing 1 required positional argument: 'entity_types'` - a real signature mismatch in the enhanced-extraction path, not just a missing key.
- `python -m integrations.oneke.verify_installation` -> **19 passed, 1 failed, 1 warning**. The failure is 'KnowledgeEngine Methods: All 4 OneKE methods present' (the repo-level `knowledge_engine` package does not expose them); the warning is that `sentence-transformers` is absent so similarity falls back to keyword matching.
- Invocation gotcha (not fixed, documented): running `python integrations/oneke/verify_installation.py` directly reports 9 spurious failures ('No module named integrations') because the repo root is not on sys.path. Run it as a module instead. `test_enhanced.py` likewise cannot be run as a script (relative imports).
- README claim check: ENHANCED_README lists Reflection Agent / Quality Enhancement / Case Repository / Enhanced Bridge - all four verified present as `OneKEReflectionAgent`, `OneKEQualityEnhancer`, `OneKECaseRepository`, `EnhancedOneKEBridge`.
- Caveat on the 5 markdown files claiming 'INTEGRATION_COMPLETE' / 'TRUE 100%': the extraction backend is not installed, so those claims are not substantiated here.

## Bottom line

MOST COMPLETE PACKAGE HERE. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
