# engines/ ACTUAL STATUS

Verification of the `engines/` Python package at the repo root (OpenEvolve engine facades/plugins).
Methodology and results are reproducible from the notes at the bottom.

## 1. Syntax (`python -m py_compile`, all 525 .py files)

| State | Count |
|-------|-------|
| Compiles (before fixes) | 523 |
| Syntax errors (before fixes) | 2 |
| Compiles (after fixes) | **525 / 525** |

The package is **100% syntactically valid** after two safe, obvious fixes:
- `engines/knowledge/ml_pattern_clustering.py:1841-1842` — removed a duplicated
  `enable_deepke: bool = True` parameter in `__init__`.
- `engines/other/.tmp_dump_env.py` — deleted a broken temp file (literal `\n`
  escapes, filename prefixed `.tmp_`). Not a real module.

No other syntax errors exist.

## 2. Import-time landscape (all engine subdirs on `sys.path`)

With every `engines/<sub>` on `sys.path` (which is how these scripts evidently expect
to be run — see `engines/orchestration/providercatalogue.py` debug dump), **369 / 525
modules import cleanly**. The remaining 156 fail for three distinct reasons:

| Category | Files | Cause |
|----------|-------|-------|
| Undefined-name / design-only | 34 | Reference symbols that are **never defined anywhere** (e.g. `LeanAideClient`, `Z3Constraint`, `ProofState`, `Tactic`, `ProofHint`, `SOPParameter`, `SubProblem`, `Z3Config`, `MathematicalDomain`, `EvaluationMetric`). These are stubs/sketches, not runnable. |
| External / sibling-repo deps | ~120 | Need packages outside `engines/` (see §4). |
| Internal forward-refs | 5 | Sibling modules referenced but never defined (`decomposition_engine`, `mdap_maker_complete`, `vision_language_monitor`). |

Importantly: **no `__init__.py` exists in any subpackage and there are zero
relative imports** — `engines/` is a flat collection of scripts loaded via sys.path
manipulation, NOT a real import package. Do not add `__init__.py`; that would
contradict the design.

## 3. Per-subsystem readiness (import-ok / total)

| Subsystem | OK / Total | Readiness |
|-----------|-----------|-----------|
| other | 160 / 229 | Partial — many proof-engine/integrations stubs fail; core helpers OK |
| orchestration | 22 / 34 | Partial — imports providercatalogue, rlm, etc. |
| knowledge | 42 / 48 | Mostly OK (guarded optional deps) |
| observability | 24 / 29 | Mostly OK |
| teams | 11 / 17 | Partial — `red_team_coordinator` etc. need `LeanAideClient` |
| gauntlets | 9 / 19 | Partial — analyzer/manager stubs need `LeanAideClient` |
| decomposition | 13 / 26 | **Weak** — heavy `Z3Constraint`/`LeanAideClient`/`MathematicalDomain` gaps |
| mcts_mdap | 4 / 12 | **Weak** — `ProofState`/`Tactic` undefined in evolved-policy modules |
| e2e_invention | 4 / 8 | Partial — planner stubs need `SOPParameter` |
| sop | 1 / 8 | **Weak** — `sop_generator_*` need `SOPParameter` |
| workflow | 7 / 11 | Mostly OK |
| config, deploy, domain, validation, quality, security, ui, plugins, adaptive, alerting, reliability, strategies, solutions | broadly OK | Smaller, largely import-clean facades |

"Import-OK" means the module loads without raising; it does **not** mean it runs a
real workload — most modules guard external services (`LEAN_AVAILABLE`,
`DEEPKE_AVAILABLE`, etc.) and silently fall back. The 369 import-clean modules are
structurally sound but their actual behavior depends on external services (LLM APIs,
Lean4 server, Z3, Chroma, etc.).

## 4. External / sibling dependency gaps

| Missing dependency | Files | Notes |
|--------------------|-------|-------|
| `openevolve.kernel` | 87 | Core OpenEvolve library. Installed editable `openevolve 0.3.2` does not expose this submodule path. Upstream wiring missing. |
| `openevolve_structures` | 10 | Expected sibling module not on path. |
| `rese`, `rese_pipeline` | 2 | `rese/` subsystem at repo root; not wired into engines path. |
| `bubblelabs_*` (evolution_integration, ragbits_bubbles) | 2 | BubbleLab subproject modules. |
| `datapizza_config` | 3 | `integrations/datapizza`. |
| `transformers` | 2 | HuggingFace (optional, not installed). |
| `aiosqlite` | 1 | Optional DB dep. |
| `integrations.graphiti_integration` | 1 | graphiti integration. |
| `continuous_math_detector`, `leanaide_mcts`, `quality_control`, `validation_manager`, `sovereign_performance_optimization` | 7 | Sibling modules not importable from `engines/`. |

## 5. Architecture-doc router spot-check

`docs/architecture/OPENEVOLVE_BUBBLELAB_STATUS.md` claims
`services/openevolve-api/main.py` mounts `workflows`, `teams`, `gauntlets`,
`decomposition` routers. Findings:

- `services/openevolve-api` does **not** exist at that path. The real API service is
  `core-projects/BubbleLab/services/openevolve-api/`, and its `routers/` directory
  **does** contain `teams.py`, `gauntlets.py`, `decomposition.py` (plus `workflows.py`,
  `execution.py`, `icr.py`, `determinism.py`, etc.). So the routers are **implemented**
  (in the BubbleLab subproject), but the doc path is wrong.
- Inside `engines/`, `decomposition/`, `teams/`, `gauntlets/` are **facades/engine
  modules, not FastAPI routers**. Several of those engine modules are design-only
  (see §2). The doc's claim that the routers "reimplement evolution/adversarial/sovereign
  logic and do NOT use the real engine" is consistent with what we found.

## 6. Honest verdict

- **Compiles:** 525/525. ✅
- **Structurally importable (given all subdirs on sys.path):** 369/525 (70%).
- **Would boot/run a real workload:** a minority. Most import-clean modules are
  guarded facades that degrade to stubs without external services. The
  decomposition / sop / mcts_mdap subsystems and several `other/` proof-engine modules
  are **design-only** (undefined names, never implemented).
- **Needs external services/deps:** `openevolve.kernel` (87 files), Lean4, web3,
  Z3, transformers, BubbleLab/rese/DataPizza siblings. These are documented, not fixed.
- **No fake tests were created.** Safe fixes were limited to the 2 syntax errors above.

## Reproduce

```powershell
# syntax sweep
python -m py_compile (Get-ChildItem engines -Recurse -Filter *.py).FullName

# import sweep (adds each engines/<sub> to sys.path)
# see scripts used during this audit: engines_import2.py / engines_per.py
```

Safe fixes applied: `engines/knowledge/ml_pattern_clustering.py` (dup arg),
`engines/other/.tmp_dump_env.py` (removed). No `__init__.py` added, no sys.path hacks
committed, no logic rewritten.
