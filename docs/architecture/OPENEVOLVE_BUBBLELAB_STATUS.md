# OpenEvolve + BubbleLab Integration — Gap Analysis & Status

**Author:** Research pass (read-only)
**Date:** 2026-08-19
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
| **OpenEvolve REST API server / backend** | Python/FastAPI backend with Evolution/Team/Workflow services; architecture cites `core-projects/openevolve/openevolve/server.py` as the API surface | `server.py` is **MISSING** in `core-projects/openevolve/openevolve/`. A separate FastAPI service **does** exist at `core-projects/BubbleLab/services/openevolve-api/` (`main.py:59-66`) exposing `/api/workflows`, `/api/teams`, `/api/gauntlets`, `/api/executions`, `/api/decomposition`, `/health`, SSE `/stream/workflow/{id}` | **Implemented (alternate location)** — but it is an independent reimplementation, not the real OpenEvolve library |
| **Backend uses real OpenEvolve engine?** | Backend = OpenEvolve engine (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:103-110`) | `services/openevolve-api` **never imports** `core-projects/openevolve` (`core/evolution.py`, `core/adversarial.py`, `core/sovereign.py` are self-contained). Only string matches "OpenEvolve" in comments/config (`credential_manager.py:73`, `team_assignment.py:74`) | **Broken / Divergent** — two parallel OpenEvolve codebases; integration does not exercise the real engine |
| **BubbleLab backend proxy (Bun/Hono OpenEvolve API proxy)** | Route `apps/bubblelab-api/src/routes/openevolve.ts` (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:143`) | **File does not exist.** `apps/bubblelab-api/src/routes/` has `evolution-graph.ts`, `evolution-judge.ts`, `evolution-mutate.ts`, `z3.ts`, `leanaide.ts`, etc., but **no `openevolve.ts`** and **no proxy** to `services/openevolve-api` (grep for `openevolve`/`8000`/`proxy` in `src/*.ts` returns nothing) | **Missing** |
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
| **Integration build / typecheck harness** | "Production-ready, type-safe throughout" (`integrations/openevolve/README.md:314-319`) | `integrations/openevolve/` has **no `package.json`** and is **not** in the pnpm workspace (root `package.json` workspaces = `packages/*`, `apps/*`; lint/typecheck globs exclude it). Never built or type-checked | **Missing (orphaned)** |
| **UI layer — Flow Studio / ReactFlow for OpenEvolve** | Visual workflow builder + OpenEvolve node components (`BUBBLELABS_SYSTEM_ARCHITECTURE.md:53-70, :150-156`) | `apps/bubble-studio/src/services/openevolveApi.ts`, `src/components/settings/OpenEvolveControlPanel.tsx`, `src/types/openevolve.ts` exist; but client calls **many routes the backend lacks** | **Partial** |
| **OpenEvolve core engine importability / runnability** | Importable/runnable engine | `core-projects/openevolve/openevolve/__init__.py` exports `OpenEvolve`, `run_evolution` (library API, `api.py`); runs as a **library**, not a service. `server.py` absent | **Implemented as library / Partial as server** |
| **Route parity: UI client vs backend** | Backend supports the client's full surface | UI calls `/api/knowledge/*`, `/api/monitoring/*`, `/api/crewai/*`, `/api/bubblelabs/leanaide/*`, `/api/version-control/*`, `/api/validation/*`, `/api/parameters/*`, `/api/integrated/*`, `/api/analytics/*`, `/api/evaluators/*` (`openevolveApi.ts:956-2235`) but `services/openevolve-api/main.py:59-66` only mounts workflows/teams/gauntlets/executions/decomposition/settings/icr/determinism | **Broken** — most UI routes 404 |
| **PATH CONTRACT** | Backend `engines/other/api_server.py` exposes unprefixed `/workflows` + `rewrite_api_prefix` middleware that strips `/api` (`openevolveApi.ts:12-17`) | Actual `services/openevolve-api` mounts routers **already prefixed** `/api/workflows` (`main.py:59`, `api/workflows.py:190`). No `rewrite_api_prefix` middleware; `engines/other/api_server.py` not present here | **Contradicts docs** |

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
