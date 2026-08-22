---
name: crewai-openevolve-orchestration
description: CrewAI <> OpenEvolve orchestration, including unified workflow routing, bridge/client wiring, BubbleLab service bubble integration, and OpenEvolve LLM ensemble configuration. Use when updating or troubleshooting CrewAI execution methods, OpenEvolve evolutionary workflows, ensemble model configs, or BubbleLab orchestration endpoints.
---

# CrewAI <> OpenEvolve Orchestration

## System map
Use this high-level routing map to avoid mixing paths:
```
BubbleLab -> CrewAIBubble HTTP -> CrewAI Unified Flow (6 phases)
                          |
                          +-> CrewAI execution methods (Traditional/ROMA/ROMA_MDAP_MAKER/Claudiomiro/DataPizza)

OpenEvolve Evolutionary Workflow (6 phases) = separate path
  -> OpenEvolveClient.evolve -> OpenEvolve LLM Ensemble
```

## Quick rules
- Treat CrewAI Unified Flow as the main orchestration entry point for general problem solving.
- Treat OpenEvolve evolutionary workflows as a separate path that needs explicit calls.
- Preserve ensemble configs when routing into OpenEvolve (do not overwrite with a single-model fallback).
- Keep BubbleLab orchestration aligned with CrewAI API shapes.
- When a request explicitly says "evolution" or "optimize via OpenEvolve", use the OpenEvolve workflow path.

## Key files to inspect first
- `crewai_unified_flow.py` for phase routing and `ExecutionMethod`.
- `crewai_unified_bridge.py` for public APIs and status helpers.
- `crewai_client.py` for the client wrapper and method mapping.
- `decomposition_crewai_bridge.py`, `roma_crewai_bridge.py`, `roma_mdap_maker_crewai_bridge.py`, `datapizza_crewai_bridge.py`, `claudiomiro_crewai_bridge.py` for per-method phases.
- `roma_config.py`, `datapizza_config.py`, `claudiomiro_config.py` for method-specific configuration defaults.
- `openevolve_crewai_bridge.py` for the OpenEvolve workflow and ensemble pass-through.
- `openevolve_client.py` for LLM ensemble configuration handling.
- `BubbleLab/integrations/openevolve/service-bubbles/crewai-bubble.ts` for BubbleLab orchestration endpoints.
- `BubbleLab/integrations/openevolve/index.ts` for bubble registration and aliasing.
- `BubbleLab/integrations/openevolve/service-bubbles/crewai-bubble.ts` for legacy aliasing.

## CrewAI decorators behavior
- If CrewAI is installed, `@start`, `@listen`, and `@router` are real decorators.
- If CrewAI is missing, the decorators become no-ops and methods can be called directly.
- `CREWAI_AVAILABLE=False` does not block direct method calls but disables event-driven orchestration.

## Entry points (Python)
- `CrewAIUnifiedFlow.execute_full_workflow(problem_statement, execution_method=..., **kwargs)`
- `crewai_unified_bridge.execute_full_workflow(problem_statement, execution_method_phase2=..., **kwargs)`
- `CrewAIClient.execute_workflow(problem_statement, execution_method=..., **kwargs)`
- `execute_full_openevolve_workflow(problem_description, **kwargs)`

## Core methods (CrewAIUnifiedFlow)
- `phase_1_setup(...)`, `phase_2_solve(...)`, `phase_3_critique(...)`, `phase_4_verify(...)`
- `phase_5_reassemble(...)`, `phase_6_final_validation(...)`
- `_select_execution_method(problem_statement, use_roma_mdap_maker)`
- `execute_full_workflow(problem_statement, execution_method=..., **kwargs)`
- `get_status()`

## Return types (Python)
`CrewAIClient.execute_workflow` returns `ExecutionResult`:
- `workflow_id`, `status`, `final_solution`, `phase_results`, `metrics`, `error`
`ExecutionMetrics` includes:
- `phases_completed`, `phases_total`, `tokens_used`, `agents_deployed`, `voting_rounds`, `red_flags`, `errors`

`OpenEvolveClient.evolve` returns `EvolutionResult`:
- `success`, `best_code`, `best_score`, `iterations_completed`, `metrics`, `error`

## Result mapping (high level)
- Unified flow final output uses `final_solution` (from phase 5 reassembly).
- OpenEvolve full workflow uses `final_code` (from phase 6 selection).
- Phase results are nested under `phases.phase1...phase6` in unified flow results.
- `CrewAIClient` computes `phases_completed` by counting non-empty phase results.
- If `final_solution` is missing, check `phase5.reassembled_content`.

## Entry point selection (when to use which)
- Use `CrewAIUnifiedFlow` if you need direct flow control or integration inside Python.
- Use `CrewAIClient` if you want execution metrics and consolidated results.
- Use `crewai_unified_bridge` if you need the compatibility layer used by legacy integrations.
- Use `CrewAIBubble` for BubbleLab orchestration via HTTP.
- Use `execute_full_openevolve_workflow` only when the user explicitly requests evolutionary optimization.
- `create_unified_flow(..., enable_persistence=True)` and `CrewAIClient(enable_persistence=True)` turn on state persistence.

## Entry points (HTTP)
- `POST /api/crewai/workflows` -> run full unified workflow
- `POST /api/crewai/workflows/{id}/phases/{phase}` -> run a specific phase
- `GET /api/crewai/workflows/{id}/status` -> workflow state
- `GET /api/crewai/workflows/{id}/results` -> results summary

## Naming conventions (BubbleLab -> API)
- BubbleLab service bubble inputs use camelCase (`problemStatement`, `executionMethod`).
- CrewAI API expects snake_case (`problem_statement`, `execution_method`).
- The service bubble translates these for you; additional `parameters` are forwarded as-is.

## Decision tree (which workflow to use)
- If the request is general problem solving with decomposition and critique -> use CrewAI unified flow.
- If the request explicitly mentions evolution, optimization, or OpenEvolve -> use OpenEvolve evolutionary workflow.
- If the request emphasizes zero-error or high-reliability -> use `roma_mdap_maker`.
- If the request emphasizes hierarchical decomposition -> use `roma`.
- If the request emphasizes multi-agent coordination -> use `datapizza`.
- If the request emphasizes autonomous CLI development -> use `claudiomiro`.

## Integration patterns (recommended)
- Full unified flow (default): call `execute_full_workflow` and use `final_solution`.
- Phase-by-phase: call `execute_phase` sequentially if you need intermediate artifacts.
- OpenEvolve-only: call `execute_full_openevolve_workflow` for evolutionary optimization tasks.
- Custom hybrid: use unified flow Phase 1 for decomposition, then feed sub-problems into OpenEvolve, then reassemble via Phase 5.

## Bridge and method mapping
Unified flow method -> bridge module:
- `traditional` -> `decomposition_crewai_bridge.py`
- `roma` -> `roma_crewai_bridge.py`
- `roma_mdap_maker` -> `roma_mdap_maker_crewai_bridge.py`
- `datapizza` -> `datapizza_crewai_bridge.py`
- `claudiomiro` -> `claudiomiro_crewai_bridge.py`

Method coverage notes:
- `roma_mdap_maker` supports phases 1-6 (full ROMA-MDAP flow).
- `roma` supports phases 1-6.
- `datapizza` supports phases 1-4 directly; phases 5-6 fall back to decomposition.
- `claudiomiro` supports phases 1-6 when CLI bridge is available; else fallback.
- `traditional` supports phases 1-6 via decomposition workflow.
- If a method bridge is unavailable, unified flow logs a warning and falls back to `traditional`.

## Execution methods and routing
- Use `ExecutionMethod` values: `traditional`, `roma`, `roma_mdap_maker`, `datapizza`, `claudiomiro`, `hybrid`, `auto`.
- Remember `hybrid` currently routes to traditional.
- When `auto`, rely on the keyword router in `CrewAIUnifiedFlow._select_execution_method`.
- When `execution_method` is auto, always read back `phase1_result.execution_method` and normalize before phase 3+.
- `ExecutionMethod` is duplicated in `crewai_state_management.py`; keep values aligned.

## CrewAI status helpers
- `CrewAIUnifiedFlow.get_status()` returns engine version, execution methods, and availability flags.
- `crewai_unified_bridge.get_unified_bridge_status()` merges unified flow status with ROMA-MDAP bridge status.

Example status output (abbrev):
```json
{
  "engine": "CrewAI",
  "version": "1.0.0",
  "execution_methods": ["traditional", "roma", "roma_mdap_maker", "claudiomiro", "datapizza", "hybrid", "auto"],
  "availability": {
    "decomposition_bridge": true,
    "roma_bridge": true
  }
}
```

## Availability flags (common)
- `CREWAI_AVAILABLE`: CrewAI package import status.
- `DECOMPOSITION_BRIDGE_AVAILABLE`: decomposition bridge import status.
- `ROMA_BRIDGE_AVAILABLE`, `ROMA_MDAP_MAKER_BRIDGE_AVAILABLE`
- `DATAPIZZA_BRIDGE_AVAILABLE`, `CLAUDIOMIRO_BRIDGE_AVAILABLE`
- `CLAUDIOMIRO_AVAILABLE`: Claudiomiro CLI dependency flag.

## Execution method selection (auto routing)
Keyword groups used by `CrewAIUnifiedFlow._select_execution_method`:
- Zero-error critical: `critical`, `zero error`, `flawless`, `mission-critical`, `medical`, `financial`, `legal compliance` -> `roma_mdap_maker`
- Hierarchical decomposition: `decompose`, `break down`, `hierarchical`, `recursive` -> `roma`
- Multi-agent coordination: `multi-agent`, `coordination`, `swarm` -> `datapizza`
- CLI/development: `cli`, `command line`, `autonomous`, `development` -> `claudiomiro`
- Default -> `traditional`
- `use_roma_mdap_maker=True` raises priority for zero-error routing.

## CrewAI unified workflow (phases 1-6)
Use `CrewAIUnifiedFlow.execute_full_workflow` as the standard entry point.

Phase 1 (setup)
- Input: `problem_statement`, `execution_method`, plus method-specific params.
- Output: `workflow_id`, `execution_method`, `decomposition_plan` or method-specific metadata.
- State persistence: saves `WorkflowState` to `./crewai_states` when enabled.

Phase 2 (solve)
- Input: `phase_1_result`, `team_name`, `solve_subset`, and method-specific parameters.
- Output: list of `solutions` or method-specific solve results.
  - ROMA-MDAP: solves each sub-problem independently and returns `solutions` with a `raw` payload per item.
  - Solutions typically include `confidence` when available.

Phase 3 (critique)
- Input: `phase_2_result`, `problem_statement`, `execution_method`.
- Output: critique reports or heuristics depending on method.

Phase 4 (verify)
- Input: `phase_2_result`, `critiques`, `execution_method`.
- Output: validation results or heuristic verification.

Phase 5 (reassemble)
- Input: `phase_2_result`, `problem_statement`, `execution_method`.
- Output: `final_solution` string and assembly metadata.

Phase 6 (final validation)
- Input: `final_solution`, `problem_statement`, `execution_method`.
- Output: final validation report.

Routing note:
- `execute_full_workflow` runs Phase 1 with the requested method, then uses the selected method for Phases 3-6.

## Unified flow schema (approx)
Phase 1 input:
```json
{
  "problem_statement": "string",
  "execution_method": "traditional|roma|roma_mdap_maker|datapizza|claudiomiro|hybrid|auto"
}
```
Phase 1 output (core):
```json
{
  "workflow_id": "string",
  "status": "completed|failed",
  "execution_method": "string",
  "decomposition_plan": { "sub_problems": [] },
  "analysis": {}
}
```
Phase 2 input:
```json
{ "phase_1_result": { "execution_method": "string", "decomposition_plan": {} } }
```
Phase 2 output (core):
```json
{ "status": "completed|failed", "solutions": [ { "id": "string", "solution": "string" } ] }
```
Phase 3 input:
```json
{ "phase_2_result": { "solutions": [] }, "problem_statement": "string" }
```
Phase 4 input:
```json
{ "phase_2_result": { "solutions": [] }, "critiques": [] }
```
Phase 5 input:
```json
{ "phase_2_result": { "solutions": [] }, "problem_statement": "string" }
```
Phase 5 output (core):
```json
{ "final_solution": "string", "components_used": [] }
```
Phase 6 input:
```json
{ "final_solution": "string", "problem_statement": "string" }
```
Phase 6 output (core):
```json
{ "passed": true, "overall_score": 0.9 }
```

## Decomposition workflow stage mapping
When using `traditional` (decomposition), phases map to zero-error stages:
- Phase 1 -> Stage 0 (analysis) + Stage 1 (decomposition)
- Phase 2 -> Stage 3A (blue team solve)
- Phase 3 -> Stage 3B (red team gauntlet)
- Phase 4 -> Stage 3C (gold team validation)
- Phase 5 -> Stage 4 (reassembly)
- Phase 6 -> Stage 5/6 (final validation + knowledge extraction)

## Team roles (decomposition workflow)
- Blue team: Phase 2 solving.
- Red team: Phase 3 critique (adversarial gauntlet).
- Gold team: Phase 4 verify (validation gauntlet).
- Phase 5 reassembles sub-solutions; Phase 6 validates final output.

## Workflow state model (persistence)
Key fields from `WorkflowState` in `crewai_state_management.py`:
- `workflow_id`, `problem_statement`, `phase`, `status`, `execution_method`
- `decomposition_plan`, `sub_solutions`, `critique_reports`, `verification_results`
- `reassembly_result`, `final_validation`, `overall_score`

Workflow status values:
- `pending`, `in_progress`, `setup_complete`, `solving`, `critique`, `verifying`, `reassembling`, `final_validation`, `completed`, `failed`, `cancelled`

## Core data shapes (state models)
SubProblem (decomposition):
```json
{
  "id": "string",
  "title": "string",
  "description": "string",
  "dependencies": [],
  "complexity_score": 0.5,
  "estimated_effort": 5
}
```
SolutionAttempt (phase 2):
```json
{
  "id": "string",
  "sub_problem_id": "string",
  "content": "string",
  "status": "PENDING|IN_PROGRESS|COMPLETED|FAILED"
}
```
CritiqueReport (phase 3) and ValidationResult (phase 4/6) store `findings` and `passed` results with confidence scores.

## CrewAI client usage (Python)
- `CrewAIClient.execute_workflow(...)` returns a unified workflow result.
- `CrewAIClient.execute_phase(phase_number, phase_input, execution_method=...)` supports phases 1-6.
- `CrewAIClient` maps `ExecutionMethod` to `CrewAIUnifiedFlow.ExecutionMethod`.

## Execution-method specific parameters
Traditional (decomposition)
- `max_sub_problems`, `decomposition_strategy`, `use_evolution`, `evolution_iterations`
 - Defaults: `max_sub_problems=15`, `decomposition_strategy="semantic"`, `use_evolution=True`, `evolution_iterations=50`

ROMA
- `roma_max_depth`, `roma_execution_mode`, `roma_provider`, `roma_model`
 - Defaults: `roma_max_depth=3`, `roma_execution_mode="recursive"`

ROMA_MDAP_MAKER (zero-error)
- `reliability_preset` (standard, thorough, fast, validation)
- `reliability_overrides` (dict of maker/mdap parameters)
- `use_roma_mdap_maker=True` to force auto routing
 - Default preset: `standard`

DataPizza
- `provider`, `api_key`, `model`, `enable_web_search`, `planning_interval`, `max_steps`
 - Defaults: `planning_interval=3`, `max_steps=15` when not provided.

Claudiomiro
- `target_branch`, `create_pr`, plus claudiomiro-specific repo settings in `claudiomiro_config.py`
 - Requires Claudiomiro CLI availability (`CLAUDIOMIRO_AVAILABLE=True`).

Additional phase 2 options (common)
- `team_name`: defaults to `blue_team` for decomposition.
- `solve_subset`: list of sub-problem IDs to solve.

## Zero-error workflow dependencies
- Decomposition and ROMA-MDAP use `crewai_zero_error_workflow.create_zero_error_workflow`.
- These workflows emit team-based solving outputs and may include `metrics` in phase results.

## Method fallback behavior (summary)
- Missing ROMA-MDAP bridge -> fall back to `roma`.
- Missing ROMA bridge -> fall back to `traditional`.
- Missing DataPizza bridge -> fall back to `traditional`.
- Missing Claudiomiro bridge -> fall back to `traditional`.
- Missing decomposition bridge -> raise `NotImplementedError` in traditional path.

## Phase input/output shapes (core fields)
Phase 1 result (unified flow)
- `workflow_id`, `status`, `execution_method`, `decomposition_plan`, `sub_problems`, `analysis`

Phase 2 result (unified flow)
- `status`, `solutions` (list of `{id, solution, confidence}`), `execution_method`

Phase 3 result
- `critiques` list (method-specific), or heuristic critique output

Phase 4 result
- `verification_results` list (method-specific), or heuristic verify output

Phase 5 result
- `final_solution` (string), `components_used`, `metrics`

Phase 6 result
- `final_validation` (pass/fail with detail), `overall_score`

## Phase payload example (unified flow)
Execute full workflow:
```python
flow.execute_full_workflow(
    problem_statement="Build a task scheduler with retries",
    execution_method=ExecutionMethod.ROMA,
    roma_max_depth=3,
)
```

## Client example (CrewAIClient)
```python
client = CrewAIClient(enable_persistence=True)
result = client.execute_workflow(
    problem_statement="Design an event-driven pipeline",
    execution_method=ExecutionMethod.TRADITIONAL,
)
print(result.final_solution)
```

## Unified bridge example
```python
from crewai_unified_bridge import execute_full_workflow
result = execute_full_workflow(
    problem_statement="Design a zero-error database",
    execution_method_phase2="roma_mdap_maker",
    reliability_preset="thorough",
)
```

## Method-specific examples (short)
DataPizza:
```python
flow.execute_full_workflow(
    problem_statement="Coordinate multiple agents to build a UI",
    execution_method=ExecutionMethod.DATAPIZZA,
    provider="openai",
    model="gpt-4",
)
```
ROMA-MDAP:
```python
flow.execute_full_workflow(
    problem_statement="Build a safety-critical controller",
    execution_method=ExecutionMethod.ROMA_MDAP_MAKER,
    reliability_preset="validation",
)
```
Claudiomiro:
```python
flow.execute_full_workflow(
    problem_statement="Implement CLI scaffolding",
    execution_method=ExecutionMethod.CLAUDIOMIRO,
    target_branch="main",
)
```

## Normalization helpers (unified flow)
- `_normalize_execution_method` converts strings to enums; default is `auto`.
- `_normalize_decomposition_plan` accepts dicts or Pydantic models and returns a dict with `sub_problems`.
- `_extract_solutions` handles list or dict `solutions` formats.
- Results always set `execution_method` as a string (`ExecutionMethod.value`).

## Phase payload example (OpenEvolve)
```python
execute_full_openevolve_workflow(
    problem_description="Optimize a sorting algorithm",
    search_space="optimization",
    llm_models=[
        {"name": "gpt-4", "weight": 0.6},
        {"name": "gpt-4o-mini", "weight": 0.4},
    ],
    evaluator_models=[
        {"name": "gpt-4", "weight": 1.0},
    ],
)
```

## OpenEvolve workflow schema (approx)
Phase 1 output (core):
```json
{ "initial_code": "string", "decomposition_plan": { "sub_problems": [] } }
```
Phase 2 output (core):
```json
{ "evolved_code": "string" }
```
Phase 3 output (core):
```json
{ "diverse_variants": [ { "objective": "string", "code": "string" } ] }
```
Phase 4 output (core):
```json
{ "correct_code": "string" }
```
Phase 5 output (core):
```json
{ "pareto_front": [ { "objective": "string", "code": "string" } ] }
```
Phase 6 output (core):
```json
{ "selected_code": "string" }
```

## OpenEvolve evolution modes (internal)
`_map_goal_to_mode` mapping:
- `diversity:*` -> `quality_diversity`
- `multi_*` / `pareto` -> `multi_objective`
- `correctness` / `robust` -> `adversarial`
- default -> `standard`

## ROMA-MDAP reliability parameters (common)
Expect these keys inside `reliability_overrides`:
- `maker_k_ahead`, `mdap_k_min`, `mdap_k_max`
- `enable_red_flagging`, `enable_first_to_ahead`

ROMA-MDAP preset values (from `get_reliability_config`):
- `standard`: `maker_k_ahead=5`, `mdap_k_min=2`, `mdap_k_max=8`
- `thorough`: `maker_k_ahead=7`, `mdap_k_min=3`, `mdap_k_max=10`
- `fast`: `maker_k_ahead=3`, `mdap_k_min=2`, `mdap_k_max=5`
- `validation`: `maker_k_ahead=10`, `mdap_k_min=5`, `mdap_k_max=15`
- All presets set `enable_red_flagging=True`, `enable_first_to_ahead=True`
- Source of truth: `roma_mdap_maker_reliability_ssot.get_reliability_config`.

## ROMA-MDAP runtime behavior
- `reliability_config` is built in phase 1 and passed into ROMA-MDAP bridge calls.
- Phase 2 solves each sub-problem independently using ROMA-MDAP Maker/MDAP voting.

## Claudiomiro bridge defaults
`CrewAIUnifiedFlow._get_claudiomiro_bridge` defaults:
- `working_dir="."`, `ai_provider="claude"`, `enable_parallel=True`, `max_cycles=20`

## CrewAI unified bridge (function signature)
`crewai_unified_bridge.execute_full_workflow` key args:
- `execution_method_phase2` (string, mapped to ExecutionMethod)
- `use_evolution`, `use_roma_workflow`, `roma_max_depth_analysis`, `roma_max_depth_solving`
- `roma_execution_mode`, `roma_provider`, `roma_model`
- `reliability_preset`, `reliability_overrides`

Bridge notes:
- `use_roma_workflow=True` forces `execution_method_phase2="roma"` before mapping.
- `execution_method_phase2` is mapped via `_map_execution_method` and can accept enums with `.value`.

## Unified flow output shape (full)
```json
{
  "workflow": "unified_crewai",
  "status": "completed|failed",
  "phases": {
    "phase1": {},
    "phase2": {},
    "phase3": {},
    "phase4": {},
    "phase5": {},
    "phase6": {}
  },
  "final_solution": "string",
  "message": "Workflow completed"
}
```

## OpenEvolve evolutionary workflow (phases 1-6)
Use `execute_full_openevolve_workflow` when the user explicitly requests evolution or optimization.

OpenEvolve phase functions:
- `execute_phase_1_setup`
- `execute_phase_2_optimize`
- `execute_phase_3_diversity`
- `execute_phase_4_correctness`
- `execute_phase_5_multi_objective`
- `execute_phase_6_selection`

Phase 1 (setup)
- Input: `problem_description`, `search_space`
- Output: `initial_code`, `decomposition_plan` with a single sub-problem.

Phase 2 (optimize)
- Input: `initial_code`, `iterations`, `optimization_goal`
- Output: `evolved_code`

Phase 3 (diversity)
- Input: `evolved_code`, `diversity_objectives`
- Output: `diverse_variants` (multiple evolved programs)

Phase 4 (correctness)
- Input: `code`, `correctness_criteria`
- Output: `correct_code`

Phase 5 (multi-objective)
- Input: `code`, `objectives`
- Output: `pareto_front`

Phase 6 (selection)
- Input: candidate list (from Phase 5/3/4)
- Output: `selected_code`

## OpenEvolve workflow parameters (common)
- `search_space`: "optimization" (default), "search", or custom; used to seed initial code.
- `iterations`: per phase where evolution occurs (Phase 2-5).
- `optimization_goal`: affects mode mapping for Phase 2.
- `diversity_objectives`, `correctness_criteria`, `objectives`: used in phases 3-5.
- `phases` argument is accepted but currently not used to skip phases.

## OpenEvolve workflow output shape (full)
```json
{
  "workflow": "openevolve",
  "status": "completed|failed",
  "phases": {
    "phase1": {},
    "phase2": {},
    "phase3": {},
    "phase4": {},
    "phase5": {},
    "phase6": {}
  },
  "final_code": "string"
}
```

## OpenEvolve ensemble configuration
Pass ensemble configs through to `OpenEvolveClient.evolve` using these shapes:
- `llm_models`: list of model dicts for evolution LLMs
- `evaluator_models`: list of model dicts for evaluator LLMs
- `llm`: dict of shared overrides (`api_key`, `api_base`, `temperature`, `max_tokens`, etc.)
- `openevolve_config`: full OpenEvolve `Config` dict (advanced)

Model dict fields (subset of `LLMModelConfig`):
- `name`, `weight`, `api_key`, `api_base`, `temperature`, `top_p`, `max_tokens`, `timeout`, `retries`, `retry_delay`, `random_seed`, `reasoning_effort`
Optional:
- `init_client` (custom LLM client factory)
Notes:
- Use `api_base` when routing through proxies or local gateways.
- Keep weights proportional; they are normalized internally.

Do:
- Pass `llm_models` to keep ensembles intact.
- Pass `evaluator_models` when evaluator ensembles differ from evolution models.
- Pass `llm` to set shared defaults across models.

Do not:
- Overwrite `config.llm.models` with a single model if an ensemble was supplied.

Example (Python, short):
```python
execute_full_openevolve_workflow(
    problem_description="Optimize sorting algorithm",
    llm_models=[
        {"name": "gpt-4", "weight": 0.6},
        {"name": "gpt-4o-mini", "weight": 0.4},
    ],
)
```

Advanced example (pass full config dict):
```python
execute_full_openevolve_workflow(
    problem_description="Optimize sorting algorithm",
    openevolve_config={
        "max_iterations": 200,
        "llm": {
            "models": [
                {"name": "gpt-4", "weight": 0.6},
                {"name": "gpt-4o-mini", "weight": 0.4},
            ],
            "temperature": 0.7,
        },
        "database": {"population_size": 200},
    },
)
```

YAML config reference (from `openevolve/README.md`):
```yaml
llm:
  models:
    - name: "gemini-2.5-pro"
      weight: 0.6
    - name: "gemini-2.5-flash"
      weight: 0.4
  temperature: 0.7
```

## OpenEvolve LLM ensemble behavior
- Weights are normalized; if only one model is supplied, it is used directly.
- `random_seed` from the first model config seeds ensemble sampling.
- If `evaluator_models` is omitted, it defaults to `models`.
- Ensemble implementation lives in `openevolve/openevolve/llm/ensemble.py`.
- Config schema for models is in `openevolve/openevolve/config.py` under `LLMConfig`.
- `OpenEvolveClient._prepare_config` merges `config`, `llm` overrides, and top-level kwargs, then applies `llm_models` and `evaluator_models` if provided.
- If no API key is available, `_prepare_config` injects a fallback model (local base URL); this can mask missing keys.
- Avoid weight sets that sum to zero; weights are normalized at runtime.

## OpenEvolve evolve() supported kwargs
`OpenEvolveClient._filter_openevolve_parameters` passes only:
- `iterations`, `output_dir`, `verbose`, `log_level`, `save_intermediate`, `resume_from`
- `max_iterations` is mapped into `iterations` if provided.
Other OpenEvolve parameters are validated by `UnifiedConfiguration` when available, but not passed into `run_evolution` unless explicitly supported.
Validation notes:
- `UnifiedConfiguration.validate()` can return warnings for unused parameters.
- Validation failures return an `EvolutionResult` with `success=False` and `error`.

## OpenEvolve config precedence (summary)
`OpenEvolveClient._prepare_config` applies config in this order:
1) `config` argument (dict or Config) if provided
2) `llm` overrides (from client config, then kwargs)
3) top-level kwargs (`api_key`, `api_base`, `temperature`, `max_tokens`, etc.)
4) `llm_models` / `evaluator_models` replacement, if provided
5) Shared LLM defaults applied to all models via `update_model_params`

Common OpenEvolve kwargs:
- `model_name`, `api_key`, `api_base`, `temperature`, `max_tokens`
- `max_iterations` (mapped to `iterations` in `run_evolution`)
- `population_size`, `archive_size`, `feature_dimensions`
Defaults in `_prepare_config` (if not provided):
- `max_iterations=10`
- `population_size=20`

## BubbleLab orchestration notes
Use the CrewAI HTTP endpoints via `CrewAIBubble`.

Service bubble schema (key fields)
- `operation`: `health_check`, `get_capabilities`, `execute_workflow`, `execute_phase`, `get_status`, `get_results`, `list_workflows`, `delegate_task`
- `baseUrl`, `apiKey`, `timeout`
- `workflowId`, `problemStatement`, `executionMethod`
- `phaseNumber`, `phaseInput`
- `taskName`, `taskDescription`, `teamName`
- `parameters` (free-form for extra args)

Execution method enum (BubbleLab)
- `traditional`, `roma`, `roma_mdap_maker`, `claudiomiro`, `datapizza`, `hybrid`, `auto`

Endpoints
- `GET /api/crewai/health`
- `GET /api/crewai/capabilities`
- `GET /api/crewai/workflows`
- `POST /api/crewai/workflows`
- `POST /api/crewai/workflows/{id}/phases/{phase}`
- `GET /api/crewai/workflows/{id}/status`
- `GET /api/crewai/workflows/{id}/results`
- `POST /api/crewai/tasks`

Payloads (service bubble)
- Execute workflow:
  ```json
  {
    "problem_statement": "...",
    "execution_method": "auto",
    "parameters": { "key": "value" }
  }
  ```
- Execute phase:
  ```json
  {
    "phase_input": { "phase_specific": "data" },
    "parameters": { "key": "value" }
  }
  ```
- Delegate task:
  ```json
  {
    "task_name": "name",
    "task_description": "details",
    "team_name": "blue_team",
    "context": { "key": "value" }
  }
  ```

Bubble response shape
- `success`, `operation`, `workflowId`, `status`, `data`, `error`, `timing`

Status/Results patterns (typical)
- `get_status` -> `{ status: "in_progress|completed|failed", phase: <n>, execution_method: "..." }`
- `get_results` -> `{ status: "completed|failed", final_solution: "...", phases: {...} }`

Ensemble routing (BubbleLab)
- Include ensemble settings in `parameters.llm_models` or `parameters.openevolve_config.llm.models`.
- If the request is explicitly evolution-focused, call the OpenEvolve workflow path rather than the unified flow.

Back-compat alias
- The `crewai` bubble is an alias to `crewai` in BubbleLab integration. Use `crewai` in new code.

## BubbleLab payload example
```json
{
  "operation": "execute_workflow",
  "problemStatement": "Design a high-reliability payments pipeline",
  "executionMethod": "roma_mdap_maker",
  "parameters": {
    "reliability_preset": "thorough",
    "llm_models": [
      {"name": "gpt-4", "weight": 0.6},
      {"name": "gpt-4o-mini", "weight": 0.4}
    ]
  }
}
```

## BubbleLab execute_phase example
```json
{
  "operation": "execute_phase",
  "workflowId": "workflow_123",
  "phaseNumber": 3,
  "phaseInput": {
    "phase_2_result": { "solutions": [] },
    "problem_statement": "..."
  }
}
```

## BubbleLab status/result example
```json
{
  "operation": "get_status",
  "workflowId": "workflow_123"
}
```

## CrewAI parameters to pass through BubbleLab
Common `parameters` keys that map into CrewAI flow:
- `max_sub_problems`, `decomposition_strategy`, `use_evolution`, `evolution_iterations`
- `roma_max_depth`, `roma_execution_mode`, `roma_provider`, `roma_model`
- `reliability_preset`, `reliability_overrides`
- `provider`, `api_key`, `model`, `enable_web_search`, `planning_interval`, `max_steps`
- `working_dir`, `ai_provider`, `enable_parallel`, `max_cycles`, `target_branch`, `create_pr`
- `llm_models`, `evaluator_models`, `openevolve_config`, `llm` (for OpenEvolve calls)

## BubbleLab execute_phase usage
- Requires `workflowId` and `phaseNumber`.
- `phaseInput` should match the phase being executed:
  - Phase 1: `problem_statement`, `execution_method`, method-specific params
  - Phase 2: `phase_1_result` (or `decomposition_plan`) + method-specific params
  - Phase 3: `phase_2_result`, `problem_statement`
  - Phase 4: `phase_2_result`, `critiques`
  - Phase 5: `phase_2_result`, `problem_statement`
  - Phase 6: `final_solution`, `problem_statement`

## CrewAIClient.execute_phase input hints
- Phase 1: pass `problem_statement` directly in `phase_input`.
- Phase 2: pass `phase_1_result` (or `decomposition_plan`) in `phase_input`.
- Phase 3-6: pass the upstream phase outputs in `phase_input`.

## State persistence and workflow IDs
- Enable persistence by keeping `CrewAIUnifiedFlow(enable_persistence=True)`.
- Default storage path: `./crewai_states`.
- Expect a `workflow_id` in phase 1 results; use it for status/results/phase calls.
- `StateManager` supports snapshots, versioning, export/import, and rollback in `crewai_state_management.py`.
- State files are stored as `.json.gz` by default (`enable_compression=True`).
- Version index lives in `./crewai_states/.versions.json`.
- State file naming: `<workflow_id>.json.gz` (or `.json` when compression disabled).
- Disable persistence for ephemeral runs or tests by passing `enable_persistence=False`.
- Use `create_state_manager(storage_dir, enable_compression)` for explicit state manager control.
- If a workflow does not return an ID, the flow generates one (timestamp or UUID-based).

## Persistence behavior (CrewAIClient)
- `execute_workflow` saves an initial `WorkflowState` before running the unified flow.
- `execute_phase` updates state per phase when persistence is enabled.
- `active_workflows` is kept in memory for the current process only.

## Inspecting persisted state (quick check)
- List files: `Get-ChildItem .\crewai_states`
- Read compressed state (Python):
  ```python
  import gzip, json
  with gzip.open("./crewai_states/<workflow_id>.json.gz", "rt", encoding="utf-8") as f:
      data = json.load(f)
  print(data.get("status"), data.get("phase"))
  ```

## StateManager quick actions
- `save_state(workflow_id, state)` / `load_state(workflow_id)` for basic persistence.
- `save_state_with_versioning(...)`, `get_state_versions(...)`, `rollback_to_version(...)`.
- `create_snapshot(...)`, `list_snapshots(...)`, `restore_snapshot(...)`.
- `export_state(...)` and `import_state(...)` for debugging.

## Environment and config keys
- `OPENAI_API_KEY` is used by `OpenEvolveClient` when no API key is provided.
- Pass `state_storage_dir` when using non-default state paths.
- Prefer explicit `llm_models` over implicit `model_name` for ensemble reliability.
- `CrewAIBubble.baseUrl` defaults to `http://localhost:8000` and `timeout` defaults to 60000ms.
- For OpenEvolve ensemble behavior, ensure `config.llm.models` is populated with multiple models.

## OpenEvolve config blocks (high level)
From `openevolve/openevolve/config.py`:
- `llm`: ensemble models, model weights, temperature, max tokens, timeouts
- `database`: population size, archive size, number of islands
- `evaluator`: evaluation settings and optional LLM feedback
- `prompt`: template directory, artifact inclusion, example counts
LLM backward-compat fields:
- `primary_model`, `secondary_model` (and weights) populate `llm.models` if provided.

Useful prompt/evaluator knobs:
- `prompt.template_dir` to load custom templates
- `prompt.num_top_programs` and `prompt.num_diverse_programs` to balance exploitation/exploration
- `evaluator.use_llm_feedback` to enable LLM-based scoring

## State storage patterns
- Workflow state: `<workflow_id>.json.gz`
- Snapshot state: `<workflow_id>_snapshot_<name>.json.gz`
- Versioned state: `<workflow_id>_v<timestamp>.json.gz`

## OpenEvolve fallback behavior
- If `OPENEVOLVE_AVAILABLE` is false, `OpenEvolveClient.evolve` returns a fallback result without evolution.
- When no API key is present, `_prepare_config` inserts a local fallback model which can hide missing key issues.

## CrewAI compatibility notes
- CrewAI replaces CrewAI with local execution (no HTTP client needed).
- CrewAI bubble is aliased to CrewAI in BubbleLab for backward compatibility.
- Legacy scripts expecting a remote DB should use `crewai_state_management` instead.

## Performance tuning (quick knobs)
- Reduce `max_sub_problems` for smaller decomposition graphs.
- Lower `evolution_iterations` for faster solve cycles.
- Reduce `planning_interval` or `max_steps` for DataPizza tasks.
- Use `reliability_preset=fast` when latency matters more than rigor.

## Security notes
- Avoid logging raw API keys; prefer environment variables.
- Treat `parameters` passed through BubbleLab as untrusted input (validate if used downstream).

## Troubleshooting and known pitfalls
- Verify bridge availability: `ROMA_BRIDGE_AVAILABLE`, `ROMA_MDAP_MAKER_BRIDGE_AVAILABLE`, `DATAPIZZA_BRIDGE_AVAILABLE`, `CLAUDIOMIRO_BRIDGE_AVAILABLE`.
- Expect `hybrid` to fall back to `traditional` until a dedicated hybrid path exists.
- Avoid recursion by keeping critique/verify helpers non-recursive (ROMA tools are heuristic).
- Ensure `execution_method` is normalized from phase 1 output before phase 3+.
- Do not call `execute_full_openevolve_workflow` via CrewAI unified flow unless the user explicitly wants evolution.
- If OpenEvolve returns fallback code, check `OPENEVOLVE_AVAILABLE` and `OpenEvolveClient._prepare_config` ensemble inputs.
- If phase 2 results omit `execution_method`, set a default before phase 3+.
- If `CrewAIUnifiedFlow` raises "bridge not available", confirm the import path and optional dependencies.
- If `get_status` reports a bridge unavailable but the module exists, validate import-time exceptions.
- If `OpenEvolveClient` raises "No LLM models configured", check `llm_models`, `openevolve_config.llm.models`, or `OPENAI_API_KEY`.
- If BubbleLab calls fail, verify `baseUrl` and that the CrewAI API server is running.
- If ensemble usage looks single-model, check that `llm_models` is non-empty and not overwritten later in `_prepare_config`.

## Error reference (common)
- `NotImplementedError: Traditional decomposition bridge not available` -> missing `decomposition_crewai_bridge.py` or import error.
- `NotImplementedError: Phase 5/6 not implemented for <method>` -> missing bridge support, fallback path not available.
- `ValueError: No LLM models configured` -> missing `OPENAI_API_KEY` or `llm_models`.
- `Workflow failed: ...` in `crewai_unified_bridge.execute_full_workflow` -> check log context for the phase that raised.

## Runbook: triage a failed workflow
1) Check `get_unified_bridge_status()` and confirm required bridges are available.
2) Inspect phase results from `flow.execute_full_workflow(...)` or `GET /api/crewai/workflows/{id}/results`.
3) If Phase 1 fails, verify input parameters and `execution_method` mapping.
4) If Phase 2 fails, confirm `decomposition_plan` shape and method availability.
5) If Phase 3/4 fails, ensure critique/verify helpers are non-recursive and inputs are present.
6) If Phase 5/6 fails, verify `final_solution` exists and is non-empty.
7) Check `./crewai_states` for persisted state (compressed JSON by default).

## Runbook: confirm OpenEvolve ensemble usage
1) Pass `llm_models` (and optional `evaluator_models`) explicitly.
2) Verify `OpenEvolveClient._prepare_config` receives `llm_models`.
3) Look for ensemble initialization log: "Initialized LLM ensemble with models".
4) Ensure `OPENAI_API_KEY` is set or model configs include `api_key`.

## Runbook: phase-by-phase execution (manual)
1) Call Phase 1 and capture `workflow_id` and `decomposition_plan`.
2) Call Phase 2 with `phase_1_result` (or `decomposition_plan`) and method-specific parameters.
3) Call Phase 3 with `phase_2_result` and `problem_statement`.
4) Call Phase 4 with `phase_2_result` and `critiques`.
5) Call Phase 5 with `phase_2_result` and `problem_statement`.
6) Call Phase 6 with `final_solution` and `problem_statement`.

## Runbook: BubbleLab integration check
1) Call `health_check` and `get_capabilities` to verify service availability.
2) Submit a minimal `execute_workflow` call with `executionMethod="traditional"`.
3) Use `list_workflows` to confirm the workflow appears.
4) Fetch `get_status` and `get_results` using the returned `workflowId`.

## Common mistakes
- Passing camelCase keys directly to Python functions (use snake_case).
- Forgetting to propagate `execution_method` into phase 3+ calls.
- Using OpenEvolve for non-evolution tasks (unified flow should handle most).
- Sending `execution_method_phase2` as an enum instead of a string (bridge expects strings).
- Expecting `execute_full_openevolve_workflow(phases=[...])` to skip phases (it runs all phases).

## Extending with a new execution method
Checklist when adding a new execution method:
- Add enum value in `crewai_unified_flow.ExecutionMethod` and `crewai_state_management.ExecutionMethod`.
- Add bridge import + availability flag in `crewai_unified_flow.py`.
- Add mapping in `crewai_unified_bridge._map_execution_method`.
- Add mapping in `crewai_client._map_to_flow_execution_method`.
- Update `CrewAIExecutionMethodSchema` in `BubbleLab/integrations/openevolve/service-bubbles/crewai-bubble.ts`.
- Ensure phase 1 sets `execution_method` in results; phase 2 propagates it.
- Update any docs/README references if new method is user-facing.
- Update `CrewAIUnifiedFlow.get_status()` to include the new method and availability flag.

## Validation checklist
- Run `python -m py_compile` on touched Python modules.
- Smoke-test `get_unified_bridge_status()` and confirm reported availability.
- Test one full unified workflow run (auto or explicit method).
- Test one OpenEvolve run with a multi-model ensemble config.
- Verify BubbleLab endpoint payloads match the service bubble schema.

## Suggested test scenarios
- Traditional: `executionMethod="traditional"` with a small problem statement.
- ROMA: `executionMethod="roma"` with `roma_max_depth=2`.
- ROMA-MDAP: `executionMethod="roma_mdap_maker"` with `reliability_preset="standard"`.
- DataPizza: `executionMethod="datapizza"` with `max_steps=5`.
- Claudiomiro: `executionMethod="claudiomiro"` with `target_branch="main"`.
- OpenEvolve: `execute_full_openevolve_workflow` with `llm_models` ensemble.

## Status and verification
- Use `get_unified_bridge_status()` for availability checks.
- Confirm `OPENEVOLVE_AVAILABLE` and that ensemble configs reach `OpenEvolveClient._prepare_config`.
- Run `python -m py_compile` on any touched Python modules.
- Verify `final_solution` exists in unified flow results and `final_code` exists in OpenEvolve results.

## Glossary
- **Bridge**: a module that adapts a method workflow to CrewAI phases.
- **Execution method**: one of `traditional`, `roma`, `roma_mdap_maker`, `datapizza`, `claudiomiro`, `hybrid`, `auto`.
- **Unified flow**: the 6-phase CrewAI orchestration pipeline.
- **OpenEvolve workflow**: the 6-phase evolutionary coding pipeline.
- **Ensemble**: weighted list of LLM configurations used for generation or evaluation.
