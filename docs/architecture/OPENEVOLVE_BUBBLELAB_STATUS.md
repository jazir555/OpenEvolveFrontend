# OpenEvolve + BubbleLab Integration — Gap Analysis & Status

**Author:** Research pass (read-only)
**Date:** 2026-08-19
**Last updated:** 2026-08-20, waves 1-4
**Scope:** Intended OpenEvolve ⇄ BubbleLab integration across three layers (BubbleLab UI → BubbleLab backend (Bun/Hono) → OpenEvolve backend (Python/FastAPI)), compared against the actual code in this repo.

---

## 1. Summary

The intended integration is a three-tier system in which BubbleLab's visual Flow Studio drives OpenEvolve's evolutionary/adversarial/sovereign workflows through a BubbleLab (Bun/Hono) API proxy that mediates a Python/FastAPI OpenEvolve backend (`docs/architecture/BUBBLELABS_SYSTEM_ARCHITECTURE.md`). In reality the wiring is fragmented and partly contradictory: there is **no Hono OpenEvolve proxy** (the documented `apps/bubblelab-api/src/routes/openevolve.ts` does not exist), the UI client talks **directly** to a FastAPI service (`services/openevolve-api`, port 8000) that is an **independent reimplementation** and does **not** import the real OpenEvolve core library (`core-projects/openevolve`), and a large set of BubbleLab "integration" adapter bubbles exists but is **orphaned** (not part of the pnpm workspace, no `package.json`, never built or type-checked). A substantial UI client and control panel exist but were written against a *different, larger* backend (`engines/other/api_server.py`) than the one that is actually present, so most of its routes 404 today. Net state: **pieces exist on every layer, but end-to-end functionality is broken / unverified.**

---

## 2. Intended Architecture (distilled from docs)

- **BubbleLab UI layer** — Flow Studio (React 19 + ReactFlow, `apps/bubble-studio`) for visual workflow design, parameter config, and real-time execution monitoring (`docs/architecture/BUBBLELABS_SYSTEM_ARCHITECTURE.md:42-71`).
- **API Gateway Integration** — a BubbleLab backend route `apps/bubblelab-api/src/routes/openevolve.ts` acting as an **OpenEvolve API Proxy** (request/response transform, auth forwarding, retry, caching) (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:74-81, :142-149`).
- **BubbleLab backend (Bun/Hono)** — `bubblelab-api` mediates between UI and OpenEvolve backend; Bubble Runtime executes workflows (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:22-28, :83-90`).
- **OpenEvolve Backend (Python/FastAPI)** — Evolution Engine, Team Manager, Workflow Orchestration services (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:101-128`).
- **Service / Tool "bubbles"** — adapters per external system (Qdrant, Elasticsearch, Knowledge Engine, Workflow Orchestrator, CrewAI, LeanAide, Z3, PostgreSQL, Redis, ACE Tools + LogParser/MetricsCollector tools) behind an **Anti-Corruption Layer** with circuit breakers and canonical schemas (`docs/integrations/openevolve/README.md`, `core-projects/BubbleLab/integrations/openevolve/README.md`).
- **Execution flow:** UI → Bubble Runtime → OpenEvolve API Proxy → OpenEvolve Backend → results; status flows back via WebSocket/SSE (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:165-173`).

---

## 3. Gap Checklist

| Component / Feature | Doc says | Code reality (file paths) | Status |
|---|---|---|---|
| **OpenEvolve REST API server / backend** | Python/FastAPI backend with Evolution/Team/Workflow services; architecture cites `core-projects/openevolve/openevolve/server.py` as the API surface | `server.py` is **MISSING** in `core-projects/openevolve/openevolve/` — but the library now ships `server_stdlib.py` (`python -m openevolve.server_stdlib`, port 8000, `/api/v1/*`). A separate FastAPI service also exists at `core-projects/BubbleLab/services/openevolve-api/` exposing `/api/workflows`, `/api/teams`, `/api/gauntlets`, `/api/executions`, `/api/decomposition`, `/api/settings`, `/icr`, `/determinism`, `/api/parameters`, `/api/monitoring`, `/api/validation`, `/api/analytics`, `/api/v1/*`, `/health`, SSE `/stream/workflow/{id}` | **Implemented** — library `server_stdlib.py` + `services/openevolve-api` (the latter also mounts a `/api/v1/*` router that mirrors `server_stdlib.py` and drives the real engine) |
| **Backend uses real OpenEvolve engine?** | Backend = OpenEvolve engine (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:103-110`) | `services/openevolve-api/api/openevolve_v1.py` is wired to the real engine through `core/openevolve_bridge.py` (env `OPENEVOLVE_BRIDGE_ENABLED=1`), which imports `core-projects/openevolve` and runs an offline **mock** LLM (no keys). The legacy `/api/workflows` etc. routers remain self-contained reimplementations. | **Implemented** — `/api/v1/*` orchestrate runs the REAL OpenEvolve engine via the bridge (mock LLM offline; real LLM needs keys). Legacy routers still divergent. |
| **BubbleLab backend proxy (Bun/Hono OpenEvolve API proxy)** | Route `apps/bubblelab-api/src/routes/openevolve.ts` (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:143`) | **Now exists.** `apps/bubblelab-api/src/routes/openevolve.ts` forwards verbatim to `OPENEVOLVE_API_URL` (default `http://localhost:8000`). Passive proxy — no auth/retry/caching added; documented 8001 BubbleLab port was corrected to 8000 in the integration bubbles. | **Implemented** — passive Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts` (forwards to `OPENEVOLVE_API_URL`, default `:8000`) |
| **UI → backend path** | UI → Hono proxy → OpenEvolve backend | UI client `apps/bubble-studio/src/services/openevolveApi.ts:29,116` targets `OPENEVOLVE_API_BASE_URL` **directly** (the FastAPI service, bypassing Hono) | **Partial / Contradicts docs** |
| **Service bubble: KnowledgeEngineBubble** | Adapter calls knowledge backend | `integrations/openevolve/service-bubbles/knowledge-engine-bubble.ts` — real `fetch` to Qdrant/ES/Bedrock/OpenAI (`knowledge-engine-bubble.ts:304,635,685`); `bubble-core` copy at `packages/bubble-core/src/bubbles/service-bubble/openevolve-knowledge-engine-bubble.ts` | **Implemented (orphaned)** |
| **Service bubble: WorkflowOrchestratorBubble** | Adapter calls workflow system | `service-bubbles/workflow-orchestrator-bubble.ts:79,145` real `fetch` to `localhost:8000`; `bubble-core` copy exists | **Implemented (orphaned)** |
| **Service bubble: CrewAIBubble** | Adapter for AI agent teams | `service-bubbles/crewai-bubble.ts:41,132` real `fetch`; `bubble-core` copy exists | **Implemented (orphaned)** |
| **Service bubble: LeanAideBubble** | Adapter for Lean proofs | `service-bubbles/leanaide-bubble.ts:42,180` real `fetch localhost:7654`; `bubble-core` copy exists | **Implemented (orphaned)** |
| **Service bubble: Z3ProverBubble** | Adapter for SMT solving | `service-bubbles/z3prover-bubble.ts:43,153` real `fetch localhost:7655`; `bubble-core` copy exists | **Implemented (orphaned)** |
| **Service bubble: QdrantBubble** | Vector DB adapter | `service-bubbles/qdrant-bubble.ts:54,186` real `fetch` (baseUrl required) | **Implemented (orphaned)** |
| **Service bubble: ElasticsearchBubble** | Search adapter | `service-bubbles/elasticsearch-bubble.ts:51,197` real `fetch` (baseUrl required) | **Implemented (orphaned)** |
| **Service bubble: PostgreSQLBubble** | Relational DB adapter | `service-bubbles/postgresql-bubble.ts` (exported as `PostgreSQLBubbleExtended`, `index.ts:27`) | **Implemented (orphaned)** |
| **Service bubble: RedisBubble** | Cache/pub-sub adapter | `service-bubbles/redis-bubble.ts` | **Implemented (orphaned)** |
| **Service bubble: ACEToolsBubble** | Analytics/verify adapter | `service-bubbles/ace-tools-bubble.ts:28,114` real `fetch localhost:8000` | **Implemented (orphaned)** |
| **Tool bubble: LogParserTool** | Parse/analyze logs | `tool-bubbles/log-parser-tool.ts` | **Implemented (orphaned)** |
| **Tool bubble: MetricsCollectorTool** | Collect/aggregate metrics | `tool-bubbles/metrics-collector-tool.ts` | **Implemented (orphaned)** |
| **Anti-Corruption Layer / schemas / circuit breaker** | Protocol adapters, transformers, circuit breaker, canonical models | `adapters/anti-corruption-layer.ts`, `adapters/resilience.ts`, `schemas/canonical-models.ts` all present | **Implemented (orphaned)** |
| **Integration build / typecheck harness** | "Production-ready, type-safe throughout" (`integrations/openevolve/README.md:314-319`) | `integrations/openevolve/` now has `package.json` + `tsconfig.json` + a type shim; `npm run typecheck` (`tsc --noEmit`) passes and `npm run test:e2e` (`tests/e2e_contract.mjs`) spawns the library server and asserts health→orchestrate→completed (PASS). Still **outside** the pnpm workspace (not auto-built by root `pnpm typecheck`), but now runnable on its own. | **Done** — package.json + tsconfig + shim; `tsc` passes, `test:e2e` PASS (run standalone) |
| **UI layer — Flow Studio / ReactFlow for OpenEvolve** | Visual workflow builder + OpenEvolve node components (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:53-70, :150-156`) | `apps/bubble-studio/src/services/openevolveApi.ts`, `src/components/settings/OpenEvolveControlPanel.tsx`, `src/types/openevolve.ts` exist; but client calls **many routes the backend lacks** | **Partial** |
| **OpenEvolve core engine importability / runnability** | Importable/runnable engine | `core-projects/openevolve/openevolve/__init__.py` exports `OpenEvolve`, `run_evolution` (library API, `api.py`); runs as a **library**, not a service. `server.py` absent | **Implemented as library / Partial as server** |
| **Route parity: UI client vs backend** | Backend supports the client's full surface | UI calls `/api/knowledge/*`, `/api/monitoring/*`, `/api/crewai/*`, `/api/bubblelabs/leanaide/*`, `/api/version-control/*`, `/api/validation/*`, `/api/parameters/*`, `/api/integrated/*`, `/api/analytics/*`, `/api/evaluators/*` (`openevolveApi.ts:956-2235`). Backend now also mounts `/api/parameters`, `/api/monitoring`, `/api/validation`, `/api/analytics` and `/api/v1/*`, closing several gaps; `/api/knowledge`, `/api/crewai`, `/api/integrated`, `/api/evaluators`, `/api/version-control`, `/api/bubblelabs/*` are still absent (client 404s there). | **Partial** — `/api/parameters`, `/api/monitoring`, `/api/validation`, `/api/analytics`, `/api/v1/*` now implemented; several UI groups still unimplemented |
| **PATH CONTRACT** | Backend `engines/other/api_server.py` exposes unprefixed `/workflows` + `rewrite_api_prefix` middleware that strips `/api` (`openevolveApi.ts:12-17`) | Actual `services/openevolve-api` mounts routers **already prefixed** `/api/workflows` (`main.py:59`, `api/workflows.py:190`). No `rewrite_api_prefix` middleware; `engines/other/api_server.py` not present here | **Contradicts docs** |

---

## 3b. Implementation Progress (waves 1-4)

This section summarizes what changed between the original gap analysis (wave 1, 2026-08-19) and the current state after waves 1-4 (last updated 2026-08-20). It is a status delta, not a rewrite of the gap checklist above.

### DONE

- **OpenEvolve REST API server (library).** `core-projects/openevolve/openevolve/server_stdlib.py` now exists and runs as a stdlib HTTP server (`python -m openevolve.server_stdlib`, port 8000) speaking `/api/v1/*`. The gap-checklist row "OpenEvolve REST API server" is **Implemented**.
- **Primary backend `/api/v1/*` router (REAL engine).** `services/openevolve-api/api/openevolve_v1.py` is mounted at `/api/v1` and drives the actual OpenEvolve engine through `core/openevolve_bridge.py` (set `OPENEVOLVE_BRIDGE_ENABLED=1`). It implements health / evolve / runs / workflows / orchestrate. The gap-checklist row "Backend uses real OpenEvolve engine?" is **Implemented** (for `/api/v1/*`; legacy routers still self-contained).
- **Backend feature routers.** Added `/api/parameters`, `/api/monitoring`, `/api/validation`, `/api/analytics` to fill UI route groups that previously 404'd. Health/parameters/monitoring are exercised by the new `scripts/launch_demo.py`.
- **BubbleLab backend proxy.** `apps/bubblelab-api/src/routes/openevolve.ts` now exists and forwards verbatim to `OPENEVOLVE_API_URL` (default `http://localhost:8000`). The gap-checklist row "BubbleLab backend proxy" is **Implemented** (passive proxy).
- **Port fix (8001 → 8000).** The integration bubbles previously defaulted to BubbleLab port `8001`; defaults now point at `8000` (the OpenEvolve backend). `OPENEVOLVE_BASE_URL` / `OPENEVOLVE_API_URL` are the canonical override (default `http://localhost:8000`).
- **Integration build / typecheck.** `core-projects/BubbleLab/integrations/openevolve` gained `package.json` + `tsconfig.json` + a type shim; `npm run typecheck` (`tsc --noEmit`) passes and `npm run test:e2e` (`tests/e2e_contract.mjs`) spawns the library `server_stdlib` and asserts health→orchestrate→completed (PASS). The gap-checklist row "Integration build / typecheck harness" is **Done** (run standalone; still outside the pnpm workspace, so root `pnpm typecheck` does not build it).
- **End-to-end boot smoke + demo scripts.** `services/openevolve-api/scripts/smoke_boot.py` runs a real evolution over HTTP (PASS). New `services/openevolve-api/scripts/launch_demo.py` proves the browser-free happy path (health → parameters → monitoring → orchestrate → completed) and exits non-zero on failure (PASS).

### STILL OPEN

- **Two parallel OpenEvolve backends.** The library `server_stdlib.py` and `services/openevolve-api` are separate servers that both speak `/api/v1/*` but are not the same process. They are intended to be protocol-compatible; this is WIP and not formally reconciled into a single source of truth.
- **Legacy routers still divergent.** `/api/workflows`, `/api/teams`, `/api/gauntlets`, `/api/executions`, `/api/decomposition` reimplement evolution/adversarial/sovereign logic and do NOT use the real engine (only `/api/v1/*` does).
- **UI route parity still partial.** `/api/knowledge`, `/api/crewai`, `/api/integrated`, `/api/evaluators`, `/api/version-control`, `/api/bubblelabs/*` are still absent on the backend; the BubbleLab UI client will 404 on those groups.
- **Proxy is passive.** `apps/bubblelab-api/src/routes/openevolve.ts` forwards verbatim with no auth, retry, caching, or response transform (contrasts with the documented "API Gateway" role).
- **Orphaned from pnpm workspace.** `integrations/openevolve` is still not part of the root workspace, so it must be type-checked/tested manually rather than via `pnpm typecheck`.
- **Duplicated bubbles.** OpenEvolve bubbles exist under both `integrations/openevolve/service-bubbles/*` and `packages/bubble-core/src/bubbles/service-bubble/openevolve-*-bubble.ts` — drift risk remains.
- **Mock LLM only offline.** The `/api/v1/*` engine path runs an offline **mock** LLM by default. Real LLM providers require API keys and are not exercised by the smoke/demo scripts.

---

## 4. Top Priorities (highest-value, concrete)

1. **Reconcile the two OpenEvolve backends.** Decide the single source of truth: either (a) make `services/openevolve-api` import and drive the real engine in `core-projects/openevolve` (`controller.OpenEvolve` / `api.run_evolution`), or (b) formally adopt `services/openevolve-api` as the backend and update all docs + `server.py` references. Today the integration never runs the real OpenEvolve engine (`services/openevolve-api/core/*.py` are standalone).
2. **Fix route parity between UI and backend.** The UI client (`apps/bubble-studio/src/services/openevolveApi.ts`) expects ~12 route groups the FastAPI service does not implement. Either implement the missing routers (`/api/knowledge`, `/api/monitoring`, `/api/crewai`, `/api/bubblelabs/leanaide`, `/api/version-control`, `/api/validation`, `/api/parameters`, `/api/integrated`, `/api/analytics`, `/api/evaluators`) in `services/openevolve-api/api/`, or trim the client to the implemented surface. Add a contract test.
3. **Resolve the proxy question.** Either create the documented Hono proxy `apps/bubblelab-api/src/routes/openevolve.ts` that forwards to `services/openevolve-api`, or update the architecture docs to state the UI talks to the FastAPI service directly (current reality). The "BubbleLab backend proxy" in `BUBBLELABS_SYSTEM_ARCHITECTURE.md` is currently fiction.
4. **Correct the PATH CONTRACT.** Remove/replace the stale `engines/other/api_server.py` + `rewrite_api_prefix` reference in `openevolveApi.ts:12-17`; align prefixes (`/api/...`) with what `services/openevolve-api/main.py` actually mounts.
5. **Un-orphan the integration adapter suite.** Add `package.json` + workspace entry (or fold `integrations/openevolve` into `packages/`) so the 10 service bubbles + 2 tool bubbles + ACL actually type-check and build via `pnpm typecheck`. Note there are **two parallel bubble sets** (`integrations/openevolve/service-bubbles/*` and `packages/bubble-core/src/bubbles/service-bubble/openevolve-*-bubble.ts`) — consolidate to one.
6. **Stand up and smoke-test the backend.** Run `services/openevolve-api` via `uvicorn openevolve_api.main:app --port 8000` (per its `Makefile:29`, `README.md:94`), verify CORS allows `bubble-studio` origins (`main.py:50-56`), and confirm the bubbles' default `baseUrl` (`http://localhost:8000`) matches a running service.
7. **Document `server.py` reality.** `core-projects/openevolve/openevolve/server.py` does not exist; update `docs/architecture/BUBBLELABS_SYSTEM_ARCHITECTURE.md` and any task references accordingly (the "OpenEvolve Backend" is `services/openevolve-api`, not a server module inside the library).
8. **End-to-end verification.** Wire a minimal path UI → (proxy or direct) → `services/openevolve-api` → a real evolution run using the actual OpenEvolve library, with an automated contract/integration test (the existing `services/openevolve-api/tests/test_api_integration.py` and `bubble-studio/src/services/__tests__/openevolveApi.test.ts` can seed this).

---

## 5. Notes (doc ↔ code contradictions)

- **Wrong backend path.** `BUBBLELABS_SYSTEM_ARCHITECTURE.md:143` cites `apps/bubblelab-api/src/routes/openevolve.ts` and `:133` cites `packages/bubble-core/src/bubbles/openevolve/`; actual paths are `apps/bubblelab-api/src/routes/` (no openevolve route) and `packages/bubble-core/src/bubbles/service-bubble/openevolve-*-bubble.ts`.
- **Two OpenEvolve codebases.** The "OpenEvolve Backend (Python/FastAPI)" in the architecture is described as the engine, but `services/openevolve-api` reimplements evolution/adversarial/sovereign logic and does **not** use `core-projects/openevolve`. The library there is only a client-side importable API (`api.py`, `controller.py`), with **no server**.
- **"Proxy" vs direct.** Docs describe a BubbleLab (Hono) proxy; the real UI client bypasses it and hits the FastAPI service directly (`OPENEVOLVE_API_BASE_URL`).
- **PATH CONTRACT references a non-existent file** (`engines/other/api_server.py`, `rewrite_api_prefix`) in `openevolveApi.ts:12-17`; the deployed service uses `/api`-prefixed routers with no rewrite middleware.
- **Massive route gap.** The UI client's surface (~12 groups) far exceeds `services/openevolve-api`'s mounted routers (5 groups). The integration "works" only for workflows/teams/gauntlets/executions/decomposition if a service is running.
- **Orphaned adapter suite.** `integrations/openevolve/README.md` claims "20+ production-ready, type-safe adapters" and "npm test", but the folder has **no `package.json`** and is excluded from the pnpm workspace, so those claims are unverifiable in-repo.
- **Duplicated bubbles.** OpenEvolve bubbles exist both under `integrations/openevolve/service-bubbles/` (orphaned) and `packages/bubble-core/src/bubbles/service-bubble/openevolve-*-bubble.ts` (built) — risk of drift.
- **`server.py` — not created by a parallel agent.** Explicitly verified `Test-Path` = `False` at `core-projects/openevolve/openevolve/server.py`; report it as **Missing**, not "Implemented (added)".
