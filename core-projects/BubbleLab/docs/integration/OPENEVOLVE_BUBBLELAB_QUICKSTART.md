# OpenEvolve ↔ BubbleLab Integration — Quickstart (LLM wiring)

This note covers how to drive a **real LLM** through the OpenEvolve integration
instead of the offline mock backend.

## Verify everything

Run the unified harness to exercise all Python + TypeScript suites and print a
single green/red status: `pwsh core-projects/BubbleLab/scripts/verify_integration.ps1`
(from the `OpenEvolveFrontend` repo root). It runs each suite independently with
a 300s timeout, reports `SKIP` for missing tooling, and exits non-zero on `RED`.

## Default: offline mock (no credentials)

Out of the box, the integration runs fully offline. When no real LLM config is
supplied, `core/openevolve_bridge.run_openevolve_workflow` selects the
deterministic `LLMModelConfig(name="mock", provider="mock")` backend, so no API
key or network access is needed. All existing tests exercise this path.

## Real LLM: pass `llm` config through the bubble

LLM configuration is threaded across three layers:

1. **Bubble params → server body** —
   `packages/bubble-core/src/bubbles/service-bubble/openevolve-workflow-orchestrator-bubble.ts`
   adds optional params `llmApiKey`, `llmModel`, `llmApiBase`, `llmProvider`.
   When `llmModel` is set, `startWorkflow` includes an `llm` object in the
   `POST /api/v1/workflows/orchestrate` body.

2. **Server body → bridge** —
   `services/openevolve-api/api/openevolve_v1.py`
   `_orchestrate_request_to_bridge` now forwards an optional `llm` field (and an
   optional top-level `config`) into the bridge request dict passed to
   `run_openevolve_workflow`.

3. **Bridge** —
   `core/openevolve_bridge._build_llm_models` uses a **live** model only when the
   `llm` object supplies both a `name` AND an `api_key`; otherwise it falls back
   to the mock backend. The bridge also accepts `{"models": [...]}` for an
   ensemble.

### Example (TypeScript)

```ts
new OpenEvolveWorkflowOrchestratorBubble({
  operation: 'start_workflow',
  system: 'evolutionary',
  problemStatement: 'evolve a function that adds two numbers',
  llmModel: 'gpt-4o',
  llmApiKey: process.env.OPENAI_API_KEY,   // omit -> mock
  llmApiBase: 'https://api.openai.com/v1',  // optional
  llmProvider: 'openai',                    // optional
});
```

### Example (HTTP)

```bash
curl -X POST http://localhost:8000/api/v1/workflows/orchestrate \
  -H 'Content-Type: application/json' \
  -d '{
    "system": "evolutionary",
    "problemStatement": "evolve a function that adds two numbers",
    "llm": { "name": "gpt-4o", "api_key": "<KEY>", "api_base": "https://api.openai.com/v1" }
  }'
```

## Docker (launch the OpenEvolve backend via compose)

The root `docker-compose.yml` (at `core-projects/BubbleLab`) brings up the
OpenEvolve API service in one command. The service directory is named
`openevolve-api` (hyphen), so the compose entrypoint creates a thin
`openevolve_api` package stub and launches uvicorn with `PYTHONPATH` covering
the stub, the service, and the real OpenEvolve engine library (mounted
read-only from `../openevolve`). This is the same workaround used by
`scripts/launch_demo.py` and `scripts/proxy_path_test.py`.

```bash
# from core-projects/BubbleLab
cp .env.example .env        # if a sample exists; otherwise create one (see vars below)
docker compose build        # image build not yet executed in CI; run before first up
docker compose up -d        # openevolve-api on ${OPENEVOLVE_API_PORT:-8000}
docker compose ps           # verify "health" check passes
curl http://localhost:8000/health
```

### `.env` variables

The compose file reads `.env` and passes through these (with defaults shown):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENEVOLVE_API_PORT` | `8000` | Host port the service is published on (container always `8000`). |
| `OPENEVOLVE_LLM_PROVIDER` | `openai` | LLM provider for live runs. |
| `OPENEVOLVE_LLM_MODEL` | `gpt-4` | Model name. |
| `OPENEVOLVE_LLM_BASE_URL` | `https://api.openai.com/v1` | Provider base URL. |
| `OPENEVOLVE_LLM_API_KEY` | _(empty)_ | Provider key; empty => offline mock backend. |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key. |
| `ANTHROPIC_API_KEY` | _(empty)_ | Anthropic key. |
| `OPENEVOLVE_MAX_WORKERS` | `5` | Concurrent execution workers. |
| `OPENEVOLVE_EXECUTION_TIMEOUT` | `300` | Per-execution timeout (s). |
| `OPENEVOLVE_LOG_LEVEL` | `INFO` | Logging verbosity. |
| `LEANAIDE_API_URL` | `http://host.docker.internal:7654` | LeanAide sidecar URL. |

With no LLM key the service still boots and serves the offline mock path (this
is what `proxy_path_test.py` exercises).

### BubbleLab API proxy runs separately

The BubbleLab API proxy (`apps/bubblelab-api`, a bun/Hono app) is **intentionally
omitted** from the compose file — it needs a separate bun toolchain/build. Run
it on its own, pointing it at the compose-published backend:

```bash
# apps/bubblelab-api
OPENEVOLVE_API_URL=http://localhost:${OPENEVOLVE_API_PORT:-8000} bun run dev
```

The proxy (`src/routes/openevolve.ts`) is fully passive: it forwards every
`/api/v1/*` and `/api/*` request verbatim (method + body + path) to
`OPENEVOLVE_API_URL` and returns the upstream status/body unchanged. So the
"UI → proxy → backend" path succeeds whenever the backend contract holds — which
`services/openevolve-api/scripts/proxy_path_test.py` verifies
(`/api/v1/health` → 200, `/api/v1/workflows/orchestrate` → 202,
`/api/v1/runs/{id}` → completed).

## Limitations

- A **real** run requires a valid `api_key` **and** network access to the
  provider; an invalid key fails the run with an authentication error (the
  endpoint still accepts the request and returns a `workflowId`).
- The mock backend remains the default and is what all CI tests use.
