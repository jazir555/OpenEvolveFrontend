# ACTUAL_STATUS: integrations/pygraphistry

**Verdict: ADAPTER SCAFFOLD - imports cleanly, all real backends missing**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

PyGraphistry visualisation adapter + bridge, with config.yaml.

## Measured facts

- Python files: **3**
- `python -m py_compile` on every file: **all 3 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.pygraphistry` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **4** - `cudf`, `cuml`, `graphistry`, `umap`

## Honest notes

- None of `graphistry`, `cudf`, `cuml`, `umap` are installed. `cudf`/`cuml` are GPU-only RAPIDS packages, so this cannot be fully exercised on this machine.
- Also needs a Graphistry server/API key to render anything.
- No tests, no README.

## Bottom line

ADAPTER SCAFFOLD. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
