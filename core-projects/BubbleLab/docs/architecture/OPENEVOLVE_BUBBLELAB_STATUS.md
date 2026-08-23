# OpenEvolve ↔ BubbleLab Integration Status

## OpenEvolve bubble consolidation (facade)

The OpenEvolve service bubbles in `integrations/openevolve` are now a thin
facade: `CrewAIBubble`, `LeanAideBubble`, `Z3ProverBubble`, and `ACEToolsBubble`
(re-exported as `OpenEvolveCrewAIBubble`, `OpenEvolveLeanAideBubble`,
`OpenEvolveZ3ProverBubble`, `OpenEvolveAceToolsBubble`) are re-exported from the
**single canonical source** — the built `@bubblelab/bubble-core` package —
instead of being duplicated locally. `QdrantBubble`, `ElasticsearchBubble`,
`PostgreSQLBubble` (as `PostgreSQLBubbleExtended`), and `RedisBubble` remain
local because bubble-core does not ship openevolve-prefixed versions of those.

> NOTE (resolved): `KnowledgeEngineBubble` and `WorkflowOrchestratorBubble` are
> re-exported from `@bubblelab/bubble-core`, and bubble-core's
> `OpenEvolveWorkflowOrchestratorBubble` /
> `OpenEvolveKnowledgeEngineBubble` were **aligned to the `/api/v1/*` server
> contract** exposed by `services/openevolve-api/api/openevolve_v1.py`, so the
> single-source consolidation is now functional. The bubbles use camelCase
> params (`baseUrl`, `problemStatement`, `generations`, `populationSize`,
> `workflowId`), `GET /api/v1/health` for `health_check` (real `response.ok`,
> never hardcoded), `POST /api/v1/workflows/orchestrate` for `start_workflow`
> (reading `workflowId`), and `GET /api/v1/runs/{workflowId}` for
> `get_status` / `get_results` (surfacing `status` and `result.best_code`);
> the base URL resolves from the `baseUrl` param, then
> `OPENEVOLVE_API_URL` / `OPENEVOLVE_BASE_URL`, then `http://localhost:8000`,
> and `action()` returns the flat `{ success, timing, ... }` result fields
> alongside the standard bubble envelope. `npm run test:bubbles` passes 9/9 and
> `npm run test:e2e` passes 8/8 against the live server (no Python-server or
> re-export changes were needed).

## Wave 3 — OpenEvolve SDK surface as chainable bubbles (COMPLETE)

All OpenEvolve SDK capabilities were surfaced as chainable BubbleLab bubbles and
registered in the four required places: `BubbleName` union
(`packages/bubble-shared-schemas/src/types.ts`), `BUBBLE_CREDENTIAL_OPTIONS`
(`packages/bubble-shared-schemas/src/credential-schema.ts`),
`bubble-factory.ts` (import + `this.register`), and **both**
`apps/bubble-studio/public/bubbles.json` and
`apps/bubblelab-api/src/services/ai/bubbles.json`.

### Bubbles added (58 total: 49 service + 9 workflow)
- **suggestions** (4): classification, content, improvement, security
- **knowledge-explorer** (5): status, query, extract, extract-file, history
- **icr** (10): status, submit, health, components, learnings, history,
  patterns, agents, metrics, benchmark
- **adaptive-mdap** (5): complexity, allocate, cost, profiles, dashboard
- **ragbits** (3): ingest, search, stats
- **dspy** (3): assess, fix, optimize
- **web3** (8): status, ingest, slither, foundry, invariants, exploit-witness,
  audit-exploit, mcp-inventory
- **maker** (7): status, execute, test, validate, tools, delegations,
  delegations-sync
- **bubblelabs** (7): status, initialize, ace, z3, roma, knowledge, analytics
- **integrated** (4): run, orchestrate, monitor, report
- **orchestration** (4): pipeline, multi-agent, coordinator, supervisor
- **team** (3): create, manage, assign
- **v1** (3): health, workflow, run
- **content** (4): generate, optimize, analyze, publish
- **gateway** (2): route, proxy
- **single**: version-control, validation, parameters, providers,
  audit-logs, auto-approval, prompts
- **workflow** (9): evolution-pipeline, continuous-evolution,
  adaptive-evolution, decomposition-to-evolution, gauntlet-red-blue-gold,
  adversarial-to-evolution, full-evolution-lifecycle,
  workflow-decomposition-pipeline, verified-deployment

### Verification (post-Wave-3)
- `tsc --noEmit`: **0 errors** across `bubble-shared-schemas`, `bubble-core`,
  `bubble-studio`, `bubblelab-api`.
- All 58 class files exist in `packages/bubble-core/src/bubbles/*/`
  (factory dynamic imports resolve; 0 missing).
- All 58 names present in **both** `bubbles.json` files (verified by union grep).
- `bubble-studio` production build: **success** (`✓ built in ~25s`; only a
  non-fatal chunk-size warning).
- Pre-existing gap fixed: 11 earlier OpenEvolve bubbles (oneke, gket,
  evolution-*, metrics-collector, knowledge-retrieval, knowledge-capture)
  were present in the API `bubbles.json` but missing from the studio
  `bubbles.json`; merged so the studio file now lists all 94 OpenEvolve union
  names (totalCount 146 -> 157).

### Backend routers (`:8000`) supporting Wave-3 bubbles
`web3.py` (`/web3`), `maker.py` (`/maker`), `suggestions.py` (`/suggestions`),
`adaptive_mdap.py` (`/adaptive-mdap`), `ragbits.py` (`/openevolve/ragbits`),
`dspy.py` (`/api/openevolve`), plus extended `bubblelabs_control.py` and
`icr.py`. All verified booting healthy and returning 200 on spot-checks.

### Deferred
- **Sovereign workflow bubble** (`openevolve-sovereign-*`): the backend has
  `engines/other/sovereign_persistence.py` + DB tables but no dedicated
  `:8000` router/subsystem module; flagged as a follow-up Wave 4 item.

## Wave 4 — Sovereign decomposition subsystem as bubbles (COMPLETE)

The Sovereign subsystem (engines/other: SovereignDatabase + ProblemAnalyzer +
SolutionOrchestrator + HealthMonitor) was previously only reachable via the
standalone `engines/other/api_server.py`. It is now surfaced through `:8000` and
as chainable BubbleLab bubbles.

### Backend router `:8000` (`/sovereign`)
New file `core-projects/BubbleLab/services/openevolve-api/api/sovereign.py`,
mounted in `main.py` (`app.include_router(sovereign_router, prefix="/sovereign")`).
Endpoints (all verified returning 200 against a live server):
- `GET  /sovereign/health`     -> HealthMonitor status
- `GET  /sovereign/problems[?problem_type=]`   -> list problems (real DB)
- `GET  /sovereign/problems/{id}`              -> one problem
- `GET  /sovereign/plans[?status=]`           -> list decomposition plans
- `GET  /sovereign/plans/{id}`                -> one plan
- `GET  /sovereign/subproblems/{parent_id}`   -> sub-problems
- `GET  /sovereign/solution-attempts/{sub_problem_id}` -> attempts
- `GET  /sovereign/stats`     -> DB row counts + size
- `POST /sovereign/run`       -> analyzes + persists a problem (202 + problem_id)
- `GET  /sovereign/runs`      -> in-memory run ledger
- `GET  /sovereign/runs/{id}` -> one run

Read endpoints query rows directly via `SovereignDatabase.get_connection()`
(JSON-decoding TEXT columns). This bypasses a pre-existing engine bug where
`ProblemDefinition.from_dict` does not exist (only `to_dict`), which made the
upstream `list_problems`/`get_problem` raise `AttributeError`.

### Engine bug fixes (engines/other/sovereign_persistence.py)
`create_problem` had two pre-existing defects that broke persistence:
1. A positional `INSERT INTO problems VALUES (...)` omitted `parent_id` and
   shifted every value, so the 15th column (`metadata`) received no value
   ("15 columns but 14 values"). Fixed with an explicit column list.
2. `problem_type` was stored via `str(enum)` -> `"ProblemType.ANALYSIS"`, which
   `from_dict`/round-trip could not parse. Fixed to store `enum.value`
   (`"analysis"`). `ProblemAnalyzer` + `SovereignDatabase` now persist and
   re-read correctly end-to-end.

### Frontend bubbles (BubbleLab)
7 new bubbles registered in all four places (BubbleName union,
BUBBLE_CREDENTIAL_OPTIONS, bubble-factory safeImport+register, and both
`bubbles.json`):
- `sovereign-status` (GET /sovereign/health)
- `sovereign-problems` (GET /sovereign/problems)
- `sovereign-plans` (GET /sovereign/plans)
- `sovereign-stats` (GET /sovereign/stats)
- `sovereign-run` (POST /sovereign/run, body {problem_statement,title?,strategy?})
- `sovereign-runs` (GET /sovereign/runs)
- `sovereign-pipeline` (workflow bubble chaining status -> run -> problems)

### Verification
- `tsc --noEmit`: **0 errors** across bubble-shared-schemas, bubble-core,
  bubble-studio, bubblelab-api.
- `bubble-core` build regenerates `dist/bubbles.json` (239 bubbles) including
  all 7 sovereign names; `bubble-studio` build copies it to
  `public/bubbles.json` (total 239, sovereign present).
- `bubble-studio` production build: **success** (~25s; non-fatal chunk warning).
- Live `:8000` smoke test: `/sovereign/health`, `/problems`, `/plans`,
  `/stats`, `POST /run` (persists, returns problem_id), `/runs` all 200/202.

### Maintenance note
`apps/bubble-studio/public/bubbles.json` is **generated** on every
`npm run build`/`dev` via `cp ../../packages/bubble-core/dist/bubbles.json
public/bubbles.json`. Do NOT hand-edit it — add bubbles to `bubble-core`
(class + factory + shared-schemas union) and rebuild `shared-schemas` then
`bubble-core`; the manifest regenerates automatically. The
`apps/bubblelab-api/src/services/ai/bubbles.json` is hand-maintained and was
updated directly (totalCount 234 -> 241).
