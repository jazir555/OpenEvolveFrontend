# ACTUAL_STATUS: integrations/bug_fixes

**Verdict: PARTIALLY WORKING - 2 of 4 documented adapters verified by its own tests**

_Verified 2026-08-20 on Python 3.11.0 (Windows), run from the repo root. No tests were written for this audit; only pre-existing tests were executed._

## What it is

Anti-corruption-layer adapters that patch bugs in core projects without editing them.

## Measured facts

- Python files: **5**
- `python -m py_compile` on every file: **all 5 compile** (no syntax errors anywhere in integrations/)
- Importable as a package: **OK after the fixes below (previously ImportError)**
- `__init__.py` present: **yes**
- Test files: **test_fixes.py**
- Docs: **README.md**
- Files using flat sibling imports: **0**
- Referenced modules that are NOT importable here: **4** - `adversarial_maker_integration`, `evolution`, `openevolve_imports`, `red_team`

## Honest notes

- Ran the existing `test_fixes.py` (no tests were added):
-   - EvolutionConfigurationWrapper: PASS
-   - AdversarialImportResolver: PASS (exercises its fallback path; the real `red_team` module is not importable, so the fallback is what got tested)
-   - crewaiConfigOverride: FAIL - `AssertionError: Config should have 'paths' section`. The adapter logs `CrewAI config not found at None`, so the external crewai config file it is supposed to fix is absent.
-   - ConfigProvider: FAIL - `config_provider.py` does not exist in this package. The only ConfigProvider in the repo is `engines/config/config_provider.py`, which is an empty stub (`class ConfigProvider: pass`) with no `get_env()` / `validate_config()`. NOT implemented; the README section describing it is aspirational.
- README claim check: README documents `CrewAIConfigOverride` - verified, the class is defined at crewai_config_fix.py:34.

## Bottom line

PARTIALLY WORKING. Compiling is not working code: outside of `oneke` (real test suite) and `bug_fixes` (partial self-tests), **nothing in this package is covered by any executable test**, so no functional claim about it is substantiated.
