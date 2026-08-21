# ACTUAL_STATUS: integrations/uqtestfuns

**Verdict: ADAPTER SCAFFOLD - imports cleanly, one internal import is wrong**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

Adapter + bridge for UQTestFuns uncertainty-quantification test functions, with config.yaml.

## Measured facts

- Python files: **3**
- `python -m py_compile` on every file: **all 3 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.uqtestfuns` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **1** - `uq_interface`

## Honest notes

- The package imports, but `adapter.py` imports a bare top-level `uq_interface`, which is not importable; the real interface is `integrations.base.uq_interface`. The import is guarded, so this silently degrades rather than failing loudly.
- Left unfixed because it is inside a fallback chain and changing it alters which implementation is selected at runtime - that needs a behavioural decision, not a mechanical edit.
- No tests, no README.

## Bottom line

ADAPTER SCAFFOLD. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
