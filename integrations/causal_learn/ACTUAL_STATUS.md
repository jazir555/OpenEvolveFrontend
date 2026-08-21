# ACTUAL_STATUS: integrations/causal_learn

**Verdict: ADAPTER SCAFFOLD - imports cleanly, real functionality needs `causallearn`**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

Adapter + bridge over the causal-learn library, with config.yaml.

## Measured facts

- Python files: **3**
- `python -m py_compile` on every file: **all 3 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.causal_learn` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **1** - `causallearn`

## Honest notes

- The `causallearn` package is NOT installed, so every real discovery call falls back or raises. Import succeeds only because the dependency is guarded.
- No tests, no README. Nothing verifies the causal-discovery output.

## Bottom line

ADAPTER SCAFFOLD. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
