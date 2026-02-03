import { OpenAPIHono } from '@hono/zod-openapi';
import {
  solveRoute,
  optimizeRoute,
  simplifyRoute,
  tacticRoute,
  fixedpointRoute,
  getTacticsRoute,
  getLogicsRoute,
  getVersionRoute,
  healthRoute,
} from '../schemas/z3.js';
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
// Z3 API PROXY
// ============================================================================

/**
 * Proxy requests to the Z3 server (port 7655)
 *
 * Z3 is a Python library, so we run it as a simple HTTP server
 * (similar to LeanAide architecture). This provides:
 * - Process isolation
 * - Memory management
 * - Consistent architecture with other services
 * - Easy to scale independently
 */

const Z3_API_URL = env.Z3_API_URL || 'http://localhost:7655';
const Z3_TIMEOUT = env.Z3_TIMEOUT || 60000;

/**
 * Helper function to proxy requests to Z3 server
 */
async function proxyToZ3(
  path: string,
  body: any,
  timeout: number = Z3_TIMEOUT
): Promise<Response> {
  const url = `${Z3_API_URL}${path}`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Z3 server returned ${response.status}: ${errorText}`);
    }

    return response;
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw new Error(`Z3 server timeout after ${timeout}ms`);
    }
    throw error;
  }
}

// POST /z3/solve - Solve SMT problem
app.openapi(solveRoute, async (c) => {
  const request = c.req.valid('json');
  const startTime = Date.now();

  try {
    const response = await proxyToZ3('/solve', request);
    const data: any = await response.json();

    return c.json({
      ...data,
      timing: Date.now() - startTime,
    }, 200);
  } catch (error: any) {
    console.error('[Z3] Solve error:', error);
    return c.json(
      {
        result: 'unknown',
        error: error.message || 'Failed to solve SMT problem',
        timing: Date.now() - startTime,
      },
      500
    );
  }
});

// POST /z3/optimize - Solve optimization problem
app.openapi(optimizeRoute, async (c) => {
  const request = c.req.valid('json');
  const startTime = Date.now();

  try {
    const response = await proxyToZ3('/optimize', request);
    const data: any = await response.json();

    return c.json({
      ...data,
      timing: Date.now() - startTime,
    }, 200);
  } catch (error: any) {
    console.error('[Z3] Optimize error:', error);
    return c.json(
      {
        status: 'unknown',
        error: error.message || 'Failed to solve optimization problem',
        timing: Date.now() - startTime,
      },
      500
    );
  }
});

// POST /z3/simplify - Simplify expression
app.openapi(simplifyRoute, async (c) => {
  const request = c.req.valid('json');
  const startTime = Date.now();

  try {
    const response = await proxyToZ3('/simplify', request);
    const data: any = await response.json();

    return c.json({
      ...data,
      timing: Date.now() - startTime,
    }, 200);
  } catch (error: any) {
    console.error('[Z3] Simplify error:', error);
    return c.json(
      {
        result: '',
        error: error.message || 'Failed to simplify expression',
        timing: Date.now() - startTime,
      },
      500
    );
  }
});

// POST /z3/tactic - Apply tactic to goal
app.openapi(tacticRoute, async (c) => {
  const request = c.req.valid('json');
  const startTime = Date.now();

  try {
    const response = await proxyToZ3('/tactic', request);
    const data: any = await response.json();

    return c.json({
      ...data,
      timing: Date.now() - startTime,
    }, 200);
  } catch (error: any) {
    console.error('[Z3] Tactic error:', error);
    return c.json(
      {
        status: 'error',
        error: error.message || 'Failed to apply tactic',
        timing: Date.now() - startTime,
      },
      500
    );
  }
});

// POST /z3/fixedpoint - Fixedpoint query
app.openapi(fixedpointRoute, async (c) => {
  const request = c.req.valid('json');
  const startTime = Date.now();

  try {
    const response = await proxyToZ3('/fixedpoint', request);
    const data: any = await response.json();

    return c.json({
      ...data,
      timing: Date.now() - startTime,
    }, 200);
  } catch (error: any) {
    console.error('[Z3] Fixedpoint error:', error);
    return c.json(
      {
        result: 'error',
        error: error.message || 'Failed to execute fixedpoint query',
        timing: Date.now() - startTime,
      },
      500
    );
  }
});

// GET /z3/tactics - Get available tactics
app.openapi(getTacticsRoute, async (c) => {
  try {
    const response = await proxyToZ3('/tactics', {}, 5000);
    const data: any = await response.json();

    return c.json(data, 200);
  } catch (error: any) {
    console.error('[Z3] Get tactics error:', error);
    return c.json(
      {
        error: error.message || 'Failed to get tactics',
      },
      500
    );
  }
});

// GET /z3/logics - Get supported logics
app.openapi(getLogicsRoute, async (c) => {
  try {
    const response = await proxyToZ3('/logics', {}, 5000);
    const data: any = await response.json();

    return c.json(data, 200);
  } catch (error: any) {
    console.error('[Z3] Get logics error:', error);
    return c.json(
      {
        error: error.message || 'Failed to get logics',
      },
      500
    );
  }
});

// GET /z3/version - Get Z3 version
app.openapi(getVersionRoute, async (c) => {
  try {
    const response = await proxyToZ3('/version', {}, 5000);
    const data: any = await response.json();

    return c.json(data, 200);
  } catch (error: any) {
    console.error('[Z3] Get version error:', error);
    return c.json(
      {
        error: error.message || 'Failed to get version',
      },
      500
    );
  }
});

// Health check endpoint
app.openapi(healthRoute, async (c) => {
  try {
    const response = await proxyToZ3('/health', {}, 3000);
    const data: any = await response.json();

    return c.json({
      status: 'ok' as const,
      z3_available: data.z3_available as boolean,
      version: data.version as string | undefined,
    }, 200);
  } catch (error: any) {
    return c.json(
      {
        status: 'degraded' as const,
        z3_available: false as const,
        error: error.message as string | undefined,
      },
      503
    );
  }
});

export default app;
