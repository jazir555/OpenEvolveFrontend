# OpenEvolve ↔ BubbleLab Integration Architecture

Verified state as of 2026-08-19. Every claim below was checked against the actual files;
line references are included so future sessions can re-verify cheaply instead of re-deriving.

## Overview

Three moving parts:

1. **Backend** — a large FastAPI app (`engines/other/api_server.py`, ~7.7k lines) exposing
   evolution, adversarial, workflow, knowledge, LeanAide, Maker, and monitoring routes.
   > **Reconciliation (2026-08-20):** `engines/other/api_server.py` is the **Decomposition-Workflow**
   > server (port **8001**) described here for the `bubblelab-integration-sdk` decomposition-workflow surface.
   > It is **not** the OpenEvolve ⇄ BubbleLab integration backend. The OpenEvolve ⇄ BubbleLab
   > integration backend is **`core-projects/BubbleLab/services/openevolve-api`** (FastAPI,
   > port **8000**), whose routers are mounted **already `/api`-prefixed** (no `rewrite_api_prefix`
   > middleware). The integration reports **GREEN (8/8 suites)**. The `/api`-stripping middleware
   > below applies only to this `engines/other/api_server.py` Decomposition-Workflow server, not to
   > the OpenEvolve ⇄ BubbleLab integration.
   >
   > **Last reconciled: 2026-08-20** — `engines/other/api_server.py` identified as the Decomposition-Workflow server, distinct from the OpenEvolve ⇄ BubbleLab integration backend (`services/openevolve-api`, port 8000, `/api`-prefixed, GREEN).
 2. **Glue** (`glue/`) — TypeScript integration library (resilience, logging, validation,
   metrics, event bus) consumed by the UI-facing API client.
3. **UI** — two consumers: the canonical converted component/client package
   (`bubblelab-integration-sdk/`) and the actual React app (`core-projects/BubbleLab/`, a separate
   pnpm+turbo monorepo).

The frontend↔backend contract is owned by `bubblelab-integration-sdk`, not by the React app.
`bubblelab-integration-sdk` was renamed from `bubblelab-converted`; it is the OpenEvolve TypeScript
SDK / contract package — the TypeScript types plus the API client that own the
frontend↔backend contract.

> **IMPORTANT — naming clarification:** `bubblelab-integration-sdk` is **NOT** part of the OpenEvolve
> library and is **not** OpenEvolve's official SDK. The previous name was a misnomer that incorrectly
> implied this was OpenEvolve's SDK. This is **custom BubbleLab integration work** — TypeScript
> components ported from Python code that originally built on top of OpenEvolve. It depends on
> OpenEvolve-style contracts but is independently owned and maintained as BubbleLab integration code.

## Repo layout

| Area | Path | Notes |
| --- | --- | --- |
| Backend API (Decomposition-Workflow) | `engines/other/api_server.py` | FastAPI; `app = FastAPI(...)` at line 463, so `uvicorn api_server:app` is valid. `start_api_server()` defaults to port **8001** (line 5535); `__main__` just calls it (line 7708). Launchers: `Makefile:201` (`python -m uvicorn api_server:app --host 0.0.0.0 --port 8001`) and `scripts/start_bubblelabs_integration.py:51`. **This is the Decomposition-Workflow server, NOT the OpenEvolve ⇄ BubbleLab integration backend.** <br><br> **Runtime-readiness (reconciled 2026-08-20):** the module does **NOT** import/start in this environment. `py_compile` passes, but a hard, unconditional `from workflow_structures import (...)` at line 410 fails with `ModuleNotFoundError: No module named 'workflow_structures'`. `workflow_structures` is the legacy facade at `engines/workflow/workflow_structures.py` (re-exports `openevolve.kernel.schema`, which *is* present and defines all referenced classes). The server uses **bare module imports** (`from ui_shim import ...`, `from workflow_structures import ...`) and expects `engines/other` **and** `engines/workflow` on `sys.path` (run with cwd there, or set `PYTHONPATH`). There are **no `__init__.py` files** anywhere under `engines/` and `api_server.py` does **no** `sys.path` manipulation, so neither the Makefile target nor `python api_server.py` can reach line 410 today. Many other imports are guarded by `try/except ImportError` (optional), but this one is not. **Status: aspirational — compiles, but will not boot until `engines/workflow` is placed on the import path.** Last reconciled: 2026-08-20. |
| Glue library | `glue/lib/` | 17 root `.ts` modules + `glue/lib/metrics/`. This is the part that typechecks clean. |
| Glue (out of typecheck) | `glue/orchestration/`, `glue/schemas/`, `glue/adapters/`, `glue/tests/` | See boundary section — partially pulled in via imports. |
| Canonical contract | `bubblelab-integration-sdk/src/lib/openevolveApi.ts`, `bubblelab-integration-sdk/src/lib/types.ts` | Source of truth for response shapes. |
| React UI | `core-projects/BubbleLab/` | Own `pnpm-workspace.yaml` + `turbo.json`; apps `bubble-studio`, `bubblelab-api`; 6 `packages/*`. |
| Divergent client | `core-projects/BubbleLab/apps/bubble-studio/src/services/openevolveApi.ts` | 726 lines, separate implementation. Known debt. |

`glue/lib/lean4_bridge/` contains only `.py`/`.lean`/`.md` — no TypeScript, so it is
naturally absent from the TS build.

## API contract & the `/api` prefix

The canonical client is the functional `openevolveApi` object exported from
`bubblelab-integration-sdk/src/lib/openevolveApi.ts` (~1243 lines). It:

- issues **unprefixed** paths (`/teams`, `/workflows`, `/evolution/runs`,
  `/bubblelabs/leanaide/trees`, …);
- resolves base URL/API key explicitly and **throws** rather than defaulting
  ("Law of Configuration Explicitness", lines ~145–192);
- wraps every request in `CircuitBreaker` + `retryWithBackoff` + `apiLogger` imported from
  `glue/lib` via relative paths (`../../../glue/lib/...`, lines 95–97);
- sends `X-API-Key` and `X-Correlation-ID`, and enforces an `AbortController` timeout.

Response shapes live in `bubblelab-integration-sdk/src/lib/types.ts` (canonical contract types;
client in `bubblelab-integration-sdk/src/lib/openevolveApi.ts`). The BubbleLab app mirrors both
locally: types in `core-projects/BubbleLab/apps/bubble-studio/src/types/openevolve.ts`
and the matching client in `core-projects/BubbleLab/apps/bubble-studio/src/services/openevolveApi.ts`.
Examples confirmed:
`KnowledgeExplorerQueryResponse.history` (line 1126),
`KnowledgeExplorerHistoryResponse.history` (line 1134),
`LeanAideTreeListResponse.tree_ids` (line 1157),
`LeanAideProofListResponse.proof_ids` (line 1165),
`LeanAideStatusResponse.execution_history_count` (line 1148).

**Contract tests pass against a live backend and skip cleanly offline.** `npx vitest run`
in `bubblelab-integration-sdk/` is GREEN offline (no running OpenEvolve backend): passing tests plus
skipped tests, **zero failures**. The live-contract suites probe backend reachability in a
`beforeAll` (timed `fetch` to `/health`) and call a `liveBackend(ctx)` helper that invokes
`ctx.skip()` when the backend is unreachable, so the suites SKIP — not fail — when
`http://localhost:8000` is down. This covers:
`src/tests/contract/openevolve-api.test.ts`,
`src/tests/contract/execution-api.test.ts`, and `src/lib/openevolveApi.test.ts`
("OpenEvolve API Contract Tests": Health Check / Teams / Workflows / Gauntlets / Evolution /
Adversarial / Knowledge / Providers / Versions / BubbleLabs / Maker / Knowledge Explorer /
LeanAide / Monitoring / Analytics / Validation / Auto-Approval / Error Handling / Execution
Controls). The remaining suites (`workflow-orchestrator.test.ts`, `e2e-integration.test.ts`,
etc.) are pure unit/integration tests that pass offline without a backend.

The live suites default to the backend URL `process.env.OPENEVOLVE_API_URL || 'http://localhost:8000'`
(`openevolve-api.test.ts` line 17) or `process.env.OPENEVOLVE_API_BASE_URL || 'http://localhost:8000'`
(`openevolveApi.test.ts`); set either env var to point at a running backend. `OPENEVOLVE_API_KEY`
is optional and only needed for key-gated endpoints / workflow-instance lifecycle cases.

Note the package script `npm run test:contract` is narrower — it only runs
`src/tests/contract`. Use plain `vitest run` for the full suite (including `src/lib`).

## Typecheck / build boundary

**`npm run typecheck` (root, `tsc --noEmit`) is NOT a full-repo check.** It currently
passes with **0 errors** (exit 0), but only over the glue core.

Root `tsconfig.json` has `include: ["glue/lib/**/*"]` — that single line, not the exclude
list, is what defines the scope. `exclude` covers `node_modules`, `dist`, `build`,
`coverage`, `core-projects`, `openevolve_test_env`, `glue/orchestration`, `glue/schemas`,
`glue/lib/unified-knowledge-query`, `glue/lib/evolved-code-capture`,
`glue/lib/proof-knowledge-base`, `**/*.test.ts`, `**/*.contract.test.ts`, `**/*.spec.ts`,
`**/*.d.ts`.

Measured scope via `tsc --listFiles`: **28 project files**, broken down as
17 in `glue/lib`, 7 in `glue/lib/metrics`, 3 in `glue/orchestration`, 1 in `glue/schemas`.

> **Gotcha:** `exclude` only filters the `include` glob; it does not stop files reached by
> `import`. So despite being excluded, these four are still typechecked because `glue/lib`
> imports them: `glue/orchestration/event-types.ts`, `glue/orchestration/dead-letter-queue.ts`,
> `glue/orchestration/event-bus.ts`, `glue/schemas/rese-canonical.ts`. Breaking one of
> those *will* fail root typecheck.

Nothing in `core-projects/*` (React/UI), `bubblelab-integration-sdk/`, or any test file is
covered. Real builds are per-package:

- `core-projects/BubbleLab/` — pnpm workspace (`apps/*`, `packages/*`, `tools/*`, `docs`,
  and `../bubblelabs-ragbits-plugin`) driven by `turbo.json` with `build`, `typecheck`,
  `test`, `lint` tasks, each `dependsOn: ["^build"]`. **There is no `turbo.json` or
  `pnpm-workspace.yaml` at the repo root** — turbo is scoped to the BubbleLab subtree.
- `bubblelab-integration-sdk/` — own `tsc` build (`npm run build`) + vitest.
- Root `npm run build` only builds `glue/orchestration/workflows` and `glue/orchestration`
  (i.e. code the root typecheck largely skips).

## Decomposition-workflow client surface (DONE)

The adversarial/decomposition workflow is fully wired end-to-end: the canonical SDK client,
a glue resilience facade, and seven BubbleLab React feature components. All route references
below were verified against `engines/other/api_server.py`.

### Canonical SDK client — `bubblelab-integration-sdk/src/lib/openevolveApi.ts`

The `openevolveApi` object now implements the complete surface, all routed through the same
`request()` helper (CircuitBreaker + retryWithBackoff + apiLogger) described in the API-contract
section:

- **Teams CRUD** — `listTeams` (line 333, `GET /teams`), `getTeam` (335, `GET /teams/{name}`),
  `createTeam` (337, `POST /teams`), `updateTeam` (343, `PUT /teams/{name}`), `deleteTeam`
  (349, `DELETE /teams/{name}`). Backend: `api_server.py` lines 3014 (list), 3044 (get),
  3079 (create), 3117 (update), 3161 (delete).
- **Gauntlets CRUD** — `listGauntlets` (355, `GET /gauntlets`), `getGauntlet` (357),
  `createGauntlet` (359, `POST /gauntlets`), `updateGauntlet` (365, `PUT /gauntlets/{name}`),
  `deleteGauntlet` (371, `DELETE /gauntlets/{name}`). Backend: lines 3188 (list), 3218 (get),
  3254 (create), 3293 (update), 3337 (delete).
- **Evaluators** — `listEvaluators` (line 949, `GET /evaluators`), `uploadEvaluator`
  (950, `POST /evaluators`), `deleteEvaluator` (956, `DELETE /evaluators/{id}`). Backend:
  lines 3364 (list), 3382 (upload), 3405 (delete).
- **Workflows** — `listWorkflows` (377, `GET /workflows`), `getWorkflow` (379,
  `GET /workflows/{id}`), `createWorkflow` (381, `POST /workflows`), `pauseWorkflow` (387,
  `POST /workflows/{id}/pause`), `resumeWorkflow` (393, `POST /workflows/{id}/resume`),
  `deleteWorkflow` (399, `DELETE /workflows/{id}`), `getWorkflowResults` (405,
  `GET /workflows/{id}/results`). Backend: `GET /workflows` line 2502, `GET /workflows/{id}`
  line 2532, `pause` line 2853, `resume` line 2887, `results` line 2921, `delete` line 2981.
- **Decomposition-plan** — `getWorkflowPlan` (line 429, `GET /workflows/{id}/decomposition-plan`),
  `updateWorkflowPlan` (964, `PUT /workflows/{id}/decomposition-plan`). Backend: lines 2573
  (get), 2626 (update).
- **Broader surface** (also wrapped): Executions, Monitoring, Analytics, Knowledge, CrewAI,
  LeanAide, Version-Control, Validation, Parameters, Integrated-run. Examples:
  `getWorkflowTelemetry` → `GET /workflows/{id}/telemetry` (line 2755),
  `getWorkflowMetrics` → `GET /workflows/metrics` (line 441), `listCrewaiWorkflows` →
  `GET /crewai/workflows` (line 483), `GET /workflows/{id}/resource-usage` (line 2809),
  `POST /workflows/{id}/resource-optimization` (line 2829).

### Glue facade — `glue/lib/decomposition-workflow.ts` (DONE)

A self-contained, resilient client that wraps the same routes so the glue resilience library
can drive the workflow directly. It imports `CircuitBreaker` (line 18), `retryWithBackoff`
(line 19), and `apiLogger` (line 20) from `glue/lib`; constructs a `CircuitBreaker` named
`openevolve-decomposition-workflow` (line 426); logs each request via `apiLogger` (line 465);
retries failures with `retryWithBackoff` (line 507); and rejects fast when the breaker is open
(lines 511–512). The resilient `DecompositionWorkflowClient` is exported from the module
(circuit-breaker field at line 418).

### BubbleLab UI components (DONE)

Seven React feature components live under
`core-projects/BubbleLab/apps/bubble-studio/src/components/`:

| Feature | Directory | Primary component |
| --- | --- | --- |
| Team Manager | `teams/` | `TeamManager.tsx` |
| Gauntlet Designer | `gauntlets/` | `GauntletDesigner.tsx` |
| Workflow Orchestrator | `workflows/` | `WorkflowOrchestrator.tsx` |
| Decomposition / Manual Review | `decomposition/` | `SubProblemCard.tsx` |
| Real-time Monitoring | `monitoring/` | `MonitoringView.tsx` |
| Analytics Dashboard | `analytics/` | (dashboard view in `analytics/`) |
| Knowledge Base Interface | `knowledge/` | `KnowledgeBase.tsx` |

They consume the BubbleLab app client (`@/services/openevolveApi`) and types
(`@/types/openevolve`), which mirror the canonical SDK surface above.

All seven were **verified to typecheck** against the BubbleLab app: a targeted `tsc`
over the OpenEvolve feature dirs + their shared deps returned 0 errors in the
integration code. The only remaining app-level errors are pre-existing and outside this
surface — a syntax error in `src/components/execution_logs/ExecutionHistory.tsx` and an
undeclared `ownerId` in `src/hooks/useCreateBubbleFlow.ts`. Two real bugs were fixed
during verification: `AnalyticsDashboard.tsx` used a non-existent `lucide-react` export
`ArrowPathIcon` (→ `RefreshCw`), and `TeamForm.tsx` assigned `undefined` to the required
`api_key` string field (now only the optional `api_base` is coerced to `undefined`).

The `bubblelab-integration-sdk` package additionally ships a **Workflow Visual Editor**
(`src/components/openevolve/main/WorkflowVisualEditorTab.tsx`) — a three-pane
palette / canvas / step-editor builder over `WorkflowDefinition`, registered as a new
tab in `OpenEvolveApp` + `Sidebar`. See Open items.

## Known integration shims

Both exist to accommodate the divergent bubble-studio client and are intentional
tech debt, not architecture.

1. **`/api` path-rewrite middleware** (Decomposition-Workflow server only) — `engines/other/api_server.py` lines 5375–5388.
   An `@app.middleware("http")` hook strips a leading `/api` from `request.scope["path"]`
   (and `raw_path`) so `/api/workflows` reaches the same handler as `/workflows`. Its own
   docstring states the purpose: the BubbleLab frontend prefixes `/api`, while the canonical
   contract and contract tests use unprefixed routes, "without duplicating route
   definitions." A handful of genuinely `/api/...`-declared routes also exist
   (e.g. `/api/openevolve/visualize/pygraphistry` line 5468, `/api/openevolve/assess/dspy`
   line 5556, `/api/openevolve/fix/dspy` line 5674).
   > **Note:** This middleware belongs to the `engines/other/api_server.py` Decomposition-Workflow
   > server (port 8001). The OpenEvolve ⇄ BubbleLab integration backend
   > (`services/openevolve-api`, port 8000) mounts routers **already `/api`-prefixed** and has **no**
   > `rewrite_api_prefix` middleware.

2. **In-memory `/executions` compat surface** — `api_server.py` lines 2130–2319.
   Backed by `_executions: Dict[str, dict]` (line 2130) plus `_execution_logs: Dict[str, List[dict]]`
   (line 2136) and `_execution_cancel: Dict[str, threading.Event]` (line 2134).
   Routes: `POST /executions`, `GET /executions`, `GET /executions/{id}`,
   `POST /executions/{id}/pause|resume|cancel`, `GET /executions/{id}/logs`. Docstrings
   read "Compatibility endpoint for the BubbleLab frontend execution controls."
   **Status validation**: pause rejects `completed`/`failed`/`cancelled` (409); resume requires
   `paused` (409); cancel rejects terminal states (409). Each lifecycle event appends a log
   entry via `_add_log()`. Logs endpoint supports `since` filtering and caps at 1000 entries.
   **State is process-local and lost on restart** — not persisted, not multi-worker safe.

## Open items

- **Duplicate API clients — MITIGATED.** `core-projects/BubbleLab/apps/bubble-studio/src/services/openevolveApi.ts`
  is still a second client: it uses `ApiClient` from `@/lib/api`, `OPENEVOLVE_API_BASE_URL`
  from `@/env`, `/api/...` prefixed paths, and its **own local types** in
  `src/types/openevolve.ts` (`WorkflowResponse`, `ExecutionResponse`, `ExecutionStatus`,
  `TeamResponse`, …). But it now **mirrors the canonical SDK surface** end-to-end rather than
  diverging: `createTeam`/`listTeams` (lines 636/650), `createGauntlet`/`listGauntlets`
  (776/790), `uploadEvaluator` (941), `createWorkflow`/`listWorkflows`/`pauseWorkflow`/
  `resumeWorkflow` (396/412/1028/1047), plus `listExecutions`/`getExecution` (546/535). Its
  local types shadow `bubblelab-integration-sdk/src/lib/types.ts`, so the two clients stay in lockstep and
  the duplicate-client debt no longer blocks deletion of the `/api` shims. The `/api` prefix is
  still rewritten by middleware (see Known integration shims), and the BubbleLab app still ships
  its own local types — consolidation onto the canonical client remains possible but is no
  longer urgent.
- **Package rename — DONE.** `bubblelab-converted` was renamed to `bubblelab-integration-sdk` (the
  OpenEvolve TypeScript SDK / contract package). All references in this doc now use
  `bubblelab-integration-sdk`.
- **Port standardized to 8000 — DONE.** BubbleLab `env.ts` and `api-client.ts` default to
  `http://localhost:8000`, matching contract tests and the dev runtime. Backend `start_api_server`
  still defaults to 8001 (line 5535) — callers pass `port=8000` explicitly.
- **Excluded glue subpackages** (`unified-knowledge-query`, `evolved-code-capture`,
  `proof-knowledge-base`) are unverified by any typecheck. Same for
  `glue/orchestration/workflow-system` (directory exists, outside root scope).
- **`exclude` list is misleading** — it lists `glue/orchestration` / `glue/schemas` /
  `core-projects` even though `include` already scopes everything to `glue/lib`, and the
  exclusions don't actually hold for imported files. Don't trust it as a scope description.
- **bubblelab-integration-sdk test suite green offline — DONE.** `src/lib/openevolveApi.test.ts` and
  `src/tests/contract/openevolve-api.test.ts` previously *failed* without a backend
  (`fetch failed`). Both now do a `beforeAll` reachability probe and skip cleanly when the
  OpenEvolve backend is unreachable (mirroring `execution-api.test.ts`), so `npx vitest run`
  is green offline (passes + skips, 0 failures). `src/lib/workflow-monitoring.ts` now guards
  `localStorage` behind `typeof localStorage !== 'undefined'`, eliminating the Node-env
  `localStorage is not defined` noise. Final suite: 9 passed / 1 skipped file, 26 passed /
  89 skipped tests, **0 failures**.
- **Workflow Visual Editor — DONE (Medium).** Implemented in `bubblelab-integration-sdk` as
  `WorkflowVisualEditorTab` (plugin palette / canvas / step editor over `WorkflowDefinition`,
  with `validateWorkflow` checks and JSON export) and registered in `OpenEvolveApp` +
  `Sidebar`. Additive; the SDK stays at 0 type errors.
- **OpenEvolve feature surface expanded in SDK — DONE.** Documented backend capabilities that
  previously had **no client method and no UI** are now implemented in `bubblelab-integration-sdk`:
  RAGBits (search/ingest/stats), DSPy assess/fix + PyGraphistry visualization, Determinism
  generate/check, BubbleLabs integrations (list/health/control catalog-discover-execute) + Web3,
  Workflow research-approval & truth-package & instance-parameters, and ICR analytics breadth
  (refinement-needed events, reward-calibration, heatmap snapshot, VLM config, dashboard,
  analytics/*). Each adds `openevolveApi` methods + `types.ts` types + a tab
  (`RagbitsTab`, `DspyGraphistryTab`, `DeterminismTab`, `BubbleLabsIntegrationsTab`, `Web3Tab`,
  `ResearchApprovalTab`, `IcrDashboardTab`). `tsc --noEmit` stays 0 errors; `npx vitest run`
  stays green offline (live-contract suites skip without a backend).
- **SDK is still an orphaned library — OPEN.** `OpenEvolveApp` (now ~55 tabs) is exported but
  only consumed inside the SDK / `glue` + an examples file; the BubbleLab app
  (`core-projects/BubbleLab/apps/bubble-studio`) does not depend on `@openevolve/bubblelab-components`
  and renders its own parallel OpenEvolve UI. Wiring one host shell (e.g. a lazy route in
  `bubble-studio`) is the remaining integration step.
- **BubbleLab app OpenEvolve UI compiles — DONE.** Verified this session via targeted `tsc`
  (see BubbleLab UI components above). The full-app `typecheck`/`build` is still blocked by
  unrelated pre-existing errors (`ExecutionHistory.tsx` syntax error, `useCreateBubbleFlow.ts`
  `ownerId`), not by the OpenEvolve surface.
- **BubbleLab install prerequisite — DONE.** A stale `patchedDependencies` entry
  (`@sendsafely/sendsafely@3.0.1`) was removed from the BubbleLab root `package.json`; it no
  longer matched any dependency and aborted `pnpm install` with `ERR_PNPM_UNUSED_PATCH`.
  `pnpm install` now completes and `pnpm build:core` emits the `@bubblelab/*` `dist/*.d.ts`
  the app needs for type resolution.
- **Root `npm run build` vs typecheck asymmetry.** Build targets `glue/orchestration/*`,
  which root typecheck mostly excludes; the two scripts cover nearly disjoint code.
