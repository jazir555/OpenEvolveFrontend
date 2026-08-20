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
