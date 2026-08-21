# OpenEvolve + BubbleLab — QUICKSTART

**Last updated:** 2026-08-20 (waves 1-4)
**Scope:** Run the OpenEvolve ⇄ BubbleLab integration end-to-end and verify it works **without a browser**.

> **Offline mode works with no keys.** The backend runs the REAL OpenEvolve engine
> via `core/openevolve_bridge.py` using an **offline mock LLM** by default
> (`OPENEVOLVE_BRIDGE_ENABLED=1`). A *real* LLM provider requires API keys and is
> **not** exercised by the smoke/demo scripts below.

---

## 1. Two OpenEvolve backends (know the difference)

Both speak the same `/api/v1/*` dialect. For the BubbleLab integration you normally
run **only one** of them — the FastAPI service is the primary target.

| Backend | How to run | Port | Notes |
|---|---|---|---|
| **Library stdlib server** (`core-projects/openevolve`) | `python -m openevolve.server_stdlib` | 8000 | Used by the TS integration e2e test to self-verify. |
| **Primary FastAPI service** (`core-projects/BubbleLab/services/openevolve-api`) | `uvicorn openevolve_api.main:app --port 8000` | 8000 | Implements `/api/v1/*` (real engine) **plus** the BubbleLab feature routers (`/api/parameters`, `/api/monitoring`, `/api/validation`, `/api/analytics`). This is the main backend. |

---

## 2. Run the primary backend (Python)

```bash
cd core-projects/BubbleLab/services/openevolve-api
pip install -r requirements.txt
uvicorn openevolve_api.main:app --port 8000
```

- The import trick: the service dir is named `openevolve-api` (hyphen) and is not a
  package, so the launch scripts drop a thin `openevolve_api` stub on `PYTHONPATH`
  (see `scripts/smoke_boot.py` / `scripts/launch_demo.py`). Running `uvicorn` directly
  from this directory works because `uvicorn` resolves `openevolve_api.main:app` via
  the stub + `PYTHONPATH=services/openevolve-api:core-projects/openevolve`.
- Default mock LLM: **no API keys required.**
- Requires `OPENEVOLVE_BRIDGE_ENABLED=1` (default) for the `/api/v1/*` routes to use
  the real engine.

### Verify the backend booted

Either of these proves the happy path over HTTP (no browser):

```bash
# Real evolution over HTTP (PASS)
python scripts/smoke_boot.py

# Browser-free happy path: health + parameters + monitoring + orchestrate -> completed (PASS)
python scripts/launch_demo.py
```

Both exit **non-zero** on failure. `launch_demo.py` also kills the server in its
`finally` block, so it is safe to run repeatedly.

Expected `launch_demo.py` output ends with:

```
[PASS] service boots (/health=200)
[PASS] GET /api/parameters/schema
[PASS] GET /api/monitoring/dashboard
[PASS] POST /api/v1/workflows/orchestrate
[PASS] evolution run completed with best_code  ... (best_code present, real evolution ran)
DEMO PASS: OpenEvolve + BubbleLab happy path verified (offline mock LLM).
```

---

## 3. Run the TypeScript integration (adapter bubbles)

The integration adapter suite is **standalone** (not part of the pnpm workspace).

```bash
cd core-projects/BubbleLab/integrations/openevolve
npm install
npm run typecheck     # tsc --noEmit (PASS)
npm run test:e2e      # spawns library server_stdlib, asserts health->orchestrate->completed (PASS)
```

- `npm run test:e2e` starts the library `server_stdlib` server itself and targets it
  via `OPENEVOLVE_BASE_URL` (default `http://127.0.0.1:8000`). It does **not** require
  the FastAPI service to be running.

---

## 4. (Optional) BubbleLab Hono proxy

`apps/bubblelab-api/src/routes/openevolve.ts` is a **passive** proxy that forwards
requests verbatim to the backend:

```bash
cd core-projects/BubbleLab/apps/bubblelab-api
OPENEVOLVE_API_URL=http://localhost:8000  # default
# bun run dev  (or your normal Hono dev command)
```

The UI can then talk to the Hono proxy instead of the FastAPI service directly.

---

## 5. Environment variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `OPENEVOLVE_API_URL` | Backend proxy (`apps/bubblelab-api/src/routes/openevolve.ts`) | `http://localhost:8000` | Upstream OpenEvolve backend the Hono proxy forwards to. |
| `OPENEVOLVE_BASE_URL` | TS integration bubbles (e.g. `knowledge-engine-bubble.ts`, `workflow-orchestrator-bubble.ts`, `openevolve-health.ts`) | `http://localhost:8000` | Canonical base URL for the bubbles' HTTP calls; falls back to `OPENEVOLVE_API_URL` then `http://localhost:8000`. |
| `OPENEVOLVE_BRIDGE_ENABLED` | FastAPI service | `1` | When set, `/api/v1/*` routes use the real OpenEvolve engine via `core/openevolve_bridge.py`. |
| `WORKFLOW_DB_PATH` | FastAPI service | (temp) | SQLite path used by the workflow DB. |

---

## 6. Limitations / not yet done (WIP)

- **Mock LLM offline only.** The engine path uses a mock LLM by default; real LLM
  providers need keys and are unverified here.
- **Legacy routers diverge.** `/api/workflows`, `/api/teams`, `/api/gauntlets`,
  `/api/executions`, `/api/decomposition` are self-contained reimplementations and do
  **not** use the real engine (only `/api/v1/*` does).
- **UI route parity COMPLETE.** `services/openevolve-api/main.py` mounts every route
  group the UI client calls — `/api/workflows`, `/api/teams`, `/api/gauntlets`,
  `/api/executions`, `/api/settings`, `/api/icr`, `/api/determinism`,
  `/api/decomposition`, `/api/parameters`, `/api/monitoring`, `/api/validation`,
  `/api/analytics`, `/api/crewai`, `/api/version-control`, `/api/evaluators`,
  `/api/integrated`, `/api/leanaide`, `/api/knowledge`, `/api/bubblelabs/*`, and
  `/api/v1/*` (real-engine dialect). No UI-expected group 404s.
- **Proxy is passive.** No auth/retry/caching/transform in `openevolve.ts`.
- **Integration IS in the pnpm workspace.** `integrations/openevolve` is now part of the
  workspace (`pnpm-workspace.yaml` → `integrations/*`) and type-checks / tests under it
  (no longer a manual-only step).
- **Two backends, not reconciled.** Library `server_stdlib` and `services/openevolve-api`
  are separate servers; protocol-compatible (`/api/v1/*`) but not a single process.

> **Last reconciled: 2026-08-20** — route-parity and workspace claims updated to match the
> GREEN integration (8/8 harness suites). See `docs/architecture/OPENEVOLVE_BUBBLELAB_STATUS.md`.
