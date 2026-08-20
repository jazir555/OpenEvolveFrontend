import { OpenAPIHono } from '@hono/zod-openapi';
import type { Context } from 'hono';
import {
  setupErrorHandler,
  validationErrorHook,
} from '../utils/error-handler.js';
import { env } from '../config/env.js';

const app = new OpenAPIHono({
  defaultHook: validationErrorHook,
});
setupErrorHandler(app);

// ============================================================================
// OpenEvolve API PROXY
// ============================================================================

/**
 * Proxy requests to the OpenEvolve backend.
 *
 * Primary backend: `services/openevolve-api` (FastAPI) mounts its routers
 * already prefixed as `/api/workflows`, `/api/teams`, `/api/gauntlets`,
 * `/api/executions`, `/api/monitoring`, etc. There is NO `rewrite_api_prefix`
 * middleware — the `/api/...` paths are the final, canonical routes.
 *
 * A separate library server (`core-projects/openevolve/openevolve/server_stdlib.py`)
 * also exists and exposes `/api/v1/...` routes that wrap the real engine.
 *
 * This proxy forwards requests verbatim to `OPENEVOLVE_API_URL` (default
 * http://localhost:8000). It is entirely passive: it does not require the
 * upstream to be running at import time. If the upstream is unreachable, each
 * request fails with a clear 502 response.
 */

const OPENEVOLVE_API_URL = env.OPENEVOLVE_API_URL;
const OPENEVOLVE_TIMEOUT = process.env.OPENEVOLVE_TIMEOUT
  ? Number(process.env.OPENEVOLVE_TIMEOUT)
  : 30000;

/**
 * Forward the incoming request to the OpenEvolve upstream, returning the
 * upstream's status code and body unchanged (only normalizing headers).
 */
async function proxyToOpenEvolve(
  c: Context,
  path: string,
  timeout: number = OPENEVOLVE_TIMEOUT
): Promise<Response> {
  const method = c.req.method;
  const url = `${OPENEVOLVE_API_URL}${path}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const init: RequestInit = {
      method,
      headers: { ...c.req.header() },
      signal: controller.signal,
    };

    if (method !== 'GET' && method !== 'HEAD') {
      init.body = await c.req.text();
    }

    const upstream = await fetch(url, init);
    clearTimeout(timeoutId);

    const body = await upstream.text();
    const headers = new Headers();
    const contentType = upstream.headers.get('content-type');
    if (contentType) {
      headers.set('content-type', contentType);
    }

    return new Response(body, {
      status: upstream.status,
      headers,
    });
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`OpenEvolve backend timeout after ${timeout}ms`);
    }
    throw error;
  }
}

/**
 * Catch-all handler that proxies a request to the upstream and converts any
 * transport failure (unreachable backend, DNS error, timeout) into a clear 502.
 */
async function handleProxy(c: Context, path: string): Promise<Response> {
  try {
    return await proxyToOpenEvolve(c, path);
  } catch (error: any) {
    console.error('[OpenEvolve] Proxy error:', error);
    return c.json(
      {
        error: 'OpenEvolve backend unreachable',
        detail: error?.message || 'Unknown error',
        upstream: OPENEVOLVE_API_URL,
      },
      502
    );
  }
}

// --- Explicitly documented OpenEvolve routes --------------------------------

// GET /api/v1/health  (library server_stdlib.py `/api/v1/...` contract)
app.get('/api/v1/health', (c) => handleProxy(c, '/api/v1/health'));

// GET /health  (FastAPI service control-plane health, served unprefixed)
app.get('/health', (c) => handleProxy(c, '/health'));

// POST /api/v1/evolve  (library server_stdlib.py contract)
app.post('/api/v1/evolve', (c) => handleProxy(c, '/api/v1/evolve'));

// GET /api/v1/runs/:id  (library server_stdlib.py contract)
app.get('/api/v1/runs/:id', (c) =>
  handleProxy(c, `/api/v1/runs/${c.req.param('id')}`)
);

// POST /api/v1/workflows/orchestrate  (library server_stdlib.py contract)
app.post('/api/v1/workflows/orchestrate', (c) =>
  handleProxy(c, '/api/v1/workflows/orchestrate')
);

// --- Catch-all: proxy any other `/api/*` path verbatim ----------------------
// Covers `/api/workflows`, `/api/teams`, `/api/executions`, `/api/monitoring`,
// `/api/gauntlets`, etc. served by `services/openevolve-api` (FastAPI).
// Registered last so the explicit routes above take precedence.
app.all('/api/*', (c) => handleProxy(c, c.req.path));

export default app;
