# Evolution Orchestrator Service

Minimal TypeScript orchestrator that ties together mutation, rendering, and judging services.

## Run locally

```bash
pnpm install
pnpm dev
```

## Environment

- `PORT` (default: 8003)
- `SCREENSHOT_RENDERER_URL` (default: http://localhost:8001)
- `MUTATION_ENGINE_URL` (default: http://localhost:8002)
- `JUDGE_URL` (default: http://localhost:3001/evolution-judge/judge)
- `JUDGE_API_TOKEN` (optional bearer token for judge endpoint)
