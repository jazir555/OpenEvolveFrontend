# ACTUAL_STATUS: integrations/curie

**Verdict: ADAPTER SCAFFOLD - imports cleanly, no backend verification**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

Adapter + bridge for the Curie experimentation framework, plus biology/chemistry/physics YAML templates.

## Measured facts

- Python files: **3**
- `python -m py_compile` on every file: **all 3 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.curie` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **0**

## Honest notes

- Imports fine and the three templates exist on disk.
- No tests, no README; no evidence any experiment has actually been run through it.

## Bottom line

ADAPTER SCAFFOLD. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
