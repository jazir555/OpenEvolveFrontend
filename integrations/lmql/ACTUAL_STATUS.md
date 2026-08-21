# ACTUAL_STATUS: integrations/lmql

**Verdict: ADAPTER SCAFFOLD - imports cleanly, real functionality needs `lmql`**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

LMQL adapter, constraint engine and query templates.

## Measured facts

- Python files: **4**
- `python -m py_compile` on every file: **all 4 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.lmql` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **0**

## Honest notes

- The package imports, but the `lmql` library itself is NOT installed (`integrations/other/lmql_adapter.py` imports it directly and would fail).
- No tests, no README; no constrained generation has been verified.

## Bottom line

ADAPTER SCAFFOLD. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
