# OpenEvolve ↔ BubbleLab Integration — Quickstart (LLM wiring)

This note covers how to drive a **real LLM** through the OpenEvolve integration
instead of the offline mock backend.

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

## Limitations

- A **real** run requires a valid `api_key` **and** network access to the
  provider; an invalid key fails the run with an authentication error (the
  endpoint still accepts the request and returns a `workflowId`).
- The mock backend remains the default and is what all CI tests use.
