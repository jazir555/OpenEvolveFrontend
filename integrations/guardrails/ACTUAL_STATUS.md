# ACTUAL_STATUS: integrations/guardrails

**Verdict: CODE PRESENT, BEHAVIOUR UNVERIFIED - imports cleanly, zero tests**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

Guardrails engine plus policies, rails, validators and actions.

## Measured facts

- Python files: **6**
- `python -m py_compile` on every file: **all 6 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK - `import integrations.guardrails` succeeds**
- `__init__.py` present: **yes**
- Test files: **NONE**
- Docs: **NONE**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **0**

## Honest notes

- All 6 modules import with no missing top-level dependency.
- No tests and no README, so none of the policy-enforcement behaviour is verified. For a safety component this is the gap that matters most.
- `integrations/other/guardrails_mcp_tools.py` is a separate file that depends on the repo-level `reliability` package.

## Bottom line

CODE PRESENT, BEHAVIOUR UNVERIFIED. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
