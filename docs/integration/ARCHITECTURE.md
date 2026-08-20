# OpenEvolve ↔ BubbleLab Integration Architecture

Verified state as of 2026-08-19. Every claim below was checked against the actual files;
line references are included so future sessions can re-verify cheaply instead of re-deriving.

## Overview

Three moving parts:

1. **Backend** — a large FastAPI app (`engines/other/api_server.py`, ~7.7k lines) exposing
   evolution, adversarial, workflow, knowledge, LeanAide, Maker, and monitoring routes.
2. **Glue** (`glue/`) — TypeScript integration library (resilience, logging, validation,
   metrics, event bus) consumed by the UI-facing API client.
3. **UI** — two consumers: the canonical converted component/client package
   (`bubblelab-converted/`) and the actual React app (`core-projects/BubbleLab/`, a separate
   pnpm+turbo monorepo).

The frontend↔backend contract is owned by `bubblelab-converted`, not by the React app.

## Repo layout

| Area | Path | Notes |
| --- | --- | --- |
| Backend API | `engines/other/api_server.py` | FastAPI; `start_api_server()` defaults to port **8001** (line 5497). `__main__` just calls it (line 7670). |
| Glue library | `glue/lib/` | 17 root `.ts` modules + `glue/lib/metrics/`. This is the part that typechecks clean. |
| Glue (out of typecheck) | `glue/orchestration/`, `glue/schemas/`, `glue/adapters/`, `glue/tests/` | See boundary section — partially pulled in via imports. |
| Canonical contract | `bubblelab-converted/src/lib/openevolveApi.ts`, `bubblelab-converted/src/lib/types.ts` | Source of truth for response shapes. |
| React UI | `core-projects/BubbleLab/` | Own `pnpm-workspace.yaml` + `turbo.json`; apps `bubble-studio`, `bubblelab-api`; 6 `packages/*`. |
| Divergent client | `core-projects/BubbleLab/apps/bubble-studio/src/services/openevolveApi.ts` | 726 lines, separate implementation. Known debt. |

`glue/lib/lean4_bridge/` contains only `.py`/`.lean`/`.md` — no TypeScript, so it is
naturally absent from the TS build.

## API contract & the `/api` prefix

The canonical client is the functional `openevolveApi` object exported from
`bubblelab-converted/src/lib/openevolveApi.ts` (~1243 lines). It:

- issues **unprefixed** paths (`/teams`, `/workflows`, `/evolution/runs`,
  `/bubblelabs/leanaide/trees`, …);
- resolves base URL/API key explicitly and **throws** rather than defaulting
  ("Law of Configuration Explicitness", lines ~145–192);
- wraps every request in `CircuitBreaker` + `retryWithBackoff` + `apiLogger` imported from
  `glue/lib` via relative paths (`../../../glue/lib/...`, lines 95–97);
- sends `X-API-Key` and `X-Correlation-ID`, and enforces an `AbortController` timeout.

Response shapes live in `bubblelab-converted/src/lib/types.ts`. Examples confirmed:
`KnowledgeExplorerQueryResponse.history` (line 1126),
`KnowledgeExplorerHistoryResponse.history` (line 1134),
`LeanAideTreeListResponse.tree_ids` (line 1157),
`LeanAideProofListResponse.proof_ids` (line 1165),
`LeanAideStatusResponse.execution_history_count` (line 1148).

**Contract tests: 82 passing, 9 skipped (backend-e2E).** `npx vitest run` in `bubblelab-converted/`
reports `Test Files 5 passed / Tests 82 passed` (plus 1 skipped file with 9 skipIf-guarded tests).
The 82 span five files:

- `src/tests/contract/openevolve-api.test.ts`
- `src/tests/contract/workflow-orchestrator.test.ts`
- `src/tests/contract/execution-api.test.ts`
- `src/tests/integration/e2e-integration.test.ts`
- `src/lib/openevolveApi.test.ts`

Note the package script `npm run test:contract` is narrower — it only runs
`src/tests/contract` (**29 tests, 3 files**). Use plain `vitest run` for the full 82.
These suites hit a **live backend**; `openevolve-api.test.ts` defaults to
`process.env.OPENEVOLVE_API_URL || 'http://localhost:8000'` (line 17) and skips
workflow-instance cases when `OPENEVOLVE_API_KEY` is unset.

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

Nothing in `core-projects/*` (React/UI), `bubblelab-converted/`, or any test file is
covered. Real builds are per-package:

- `core-projects/BubbleLab/` — pnpm workspace (`apps/*`, `packages/*`, `tools/*`, `docs`,
  and `../bubblelabs-ragbits-plugin`) driven by `turbo.json` with `build`, `typecheck`,
  `test`, `lint` tasks, each `dependsOn: ["^build"]`. **There is no `turbo.json` or
  `pnpm-workspace.yaml` at the repo root** — turbo is scoped to the BubbleLab subtree.
- `bubblelab-converted/` — own `tsc` build (`npm run build`) + vitest.
- Root `npm run build` only builds `glue/orchestration/workflows` and `glue/orchestration`
  (i.e. code the root typecheck largely skips).

## Known integration shims

Both exist to accommodate the divergent bubble-studio client and are intentional
tech debt, not architecture.

1. **`/api` path-rewrite middleware** — `engines/other/api_server.py` lines 5375–5388.
   An `@app.middleware("http")` hook strips a leading `/api` from `request.scope["path"]`
   (and `raw_path`) so `/api/workflows` reaches the same handler as `/workflows`. Its own
   docstring states the purpose: the BubbleLab frontend prefixes `/api`, while the canonical
   contract and contract tests use unprefixed routes, "without duplicating route
   definitions." A handful of genuinely `/api/...`-declared routes also exist
   (e.g. `/api/openevolve/visualize/pygraphistry` line 5468, `/api/openevolve/assess/dspy`
   line 5556, `/api/openevolve/fix/dspy` line 5674).

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

- **Duplicate API clients.** `core-projects/BubbleLab/apps/bubble-studio/src/services/openevolveApi.ts`
  is a second, divergent client: it uses `ApiClient` from `@/lib/api`,
  `OPENEVOLVE_API_BASE_URL` from `@/env`, `/api/...` prefixed paths, and its **own local
  types** (`WorkflowResponse`, `ExecutionResponse`, `ExecutionStatus`, `TeamResponse`, …)
  that duplicate `bubblelab-converted/src/lib/types.ts`. It targets `/api/executions`,
  which is exactly what keeps both shims alive. Consolidating on the canonical client
  would let both shims be deleted. **Mitigated**: BubbleLab client now has `listExecutions`
  + `getExecution` methods and aligned `ExecutionResponse` fields (`name`, `real_engine`,
  `real_engine_available`, `best_score`, `result_summary`).
- **Port standardized to 8000.** BubbleLab `env.ts` and `api-client.ts` now default to
  `http://localhost:8000`, matching contract tests and the dev runtime. Backend `start_api_server`
  still defaults to 8001 — callers should pass `port=8000` explicitly.
- **Excluded glue subpackages** (`unified-knowledge-query`, `evolved-code-capture`,
  `proof-knowledge-base`) are unverified by any typecheck. Same for
  `glue/orchestration/workflow-system` (directory exists, outside root scope).
- **`exclude` list is misleading** — it lists `glue/orchestration` / `glue/schemas` /
  `core-projects` even though `include` already scopes everything to `glue/lib`, and the
  exclusions don't actually hold for imported files. Don't trust it as a scope description.
- **Test-suite noise.** The vitest run passes but logs real `localStorage is not defined`
  errors from `workflow-monitoring.ts` (Node env, no jsdom) plus live 404/429 responses.
  Green, but noisy.
- **Root `npm run build` vs typecheck asymmetry.** Build targets `glue/orchestration/*`,
  which root typecheck mostly excludes; the two scripts cover nearly disjoint code.
