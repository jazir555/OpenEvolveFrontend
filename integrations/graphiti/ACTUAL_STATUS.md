# ACTUAL_STATUS: integrations/graphiti

**Verdict: ADAPTER SCAFFOLD - imports cleanly, real functionality needs `graphiti_core`**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

Adapter + bridge for Graphiti temporal knowledge graphs, with config.yaml.

## Measured facts

- Python files: **3**
- `python -m py_compile` on every file: **all 3 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.graphiti` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **1** - `graphiti_core`

## Honest notes

- `graphiti_core` is NOT installed, so all graph operations are unavailable; the import only succeeds because the dependency is guarded.
- Graphiti also normally needs a running Neo4j instance - nothing here verifies that.
- No tests, no README.

## Bottom line

ADAPTER SCAFFOLD. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
