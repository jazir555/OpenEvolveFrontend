import { OpenAPIHono } from '@hono/zod-openapi';
import {
  generateProofRoute,
  verifyProofRoute,
  getModelsRoute,
  runBenchmarkRoute,
  getBenchmarkResultsRoute,
} from '../schemas/leanaide.js';
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
// LEANAIDE API PROXY
// ============================================================================

/**
 * Proxy requests to the LeanAide server (port 7654)
 *
 * The LeanAide server is a standalone service that provides Lean 4 theorem
 * proving capabilities. This proxy layer integrates it into the BubbleLab API.
 *
 * LeanAide Server Endpoints:
 * - POST /         - Main task execution (translate, prove, verify, etc.)
 * - POST /run-sim-search - Similarity search
 *
 * This proxy maps:
 * - /leanaide/generate  -> LeanAide server (prove_for_formalization task)
 * - /leanaide/verify    -> LeanAide server (elaborate task)
 * - /leanaide/models    -> Mock response (LeanAide doesn't provide this endpoint)
 * - /leanaide/benchmark/* -> Mock response (benchmarking not implemented)
 */

const LEANAIDE_API_URL = env.LEANAIDE_API_URL;
const LEANAIDE_TIMEOUT = env.LEANAIDE_TIMEOUT;

/**
 * Helper function to proxy requests to LeanAide server
 */
async function proxyToLeanAide(
  path: string,
  body: any,
  timeout: number = LEANAIDE_TIMEOUT
): Promise<Response> {
  const url = `${LEANAIDE_API_URL}${path}`;

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
      throw new Error(`LeanAide server returned ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw new Error(`LeanAide server timeout after ${timeout}ms`);
    }
    throw error;
  }
}

// POST /leanaide/generate - Generate Lean 4 proof
app.openapi(generateProofRoute, async (c) => {
  const request = c.req.valid('json');

  try {
    // Map to LeanAide task type
    const leanaideRequest = {
      task: 'prove_for_formalization',
      theorem: request.theorem,
      proof_attempt: request.proof_attempt || '',
      model: request.model,
      temperature: request.temperature,
    };

    const result = await proxyToLeanAide('/', leanaideRequest);

    // Transform response to match frontend expectations
    const data: any = await result.json();
    return c.json({
      success: data.success !== false,
      lean_code: data.lean_code || data.code,
      proof: data.proof,
      error: data.error,
      logs: data.logs,
      response_time: data.response_time,
    }, 200);
  } catch (error: any) {
    console.error('[LeanAide] Generate proof error:', error);
    return c.json(
      {
        success: false,
        error: error.message || 'Failed to generate proof',
        lean_code: undefined,
        proof: undefined,
      },
      500
    );
  }
});

// POST /leanaide/verify - Verify Lean 4 proof
app.openapi(verifyProofRoute, async (c) => {
  const request = c.req.valid('json');

  try {
    // Map to LeanAide task type
    const leanaideRequest = {
      task: 'elaborate',
      code: request.code,
    };

    const result = await proxyToLeanAide('/', leanaideRequest);

    // Transform response to match frontend expectations
    const data: any = await result.json();
    return c.json({
      valid: data.success !== false && !data.error,
      errors: data.error ? [data.error] : data.errors,
      warnings: data.warnings,
      tactic_state: data.tactic_state,
      goals: data.goals,
    }, 200);
  } catch (error: any) {
    console.error('[LeanAide] Verify proof error:', error);
    return c.json(
      {
        error: error.message || 'Failed to verify proof',
      },
      500
    );
  }
});

// GET /leanaide/models - Get available models
app.openapi(getModelsRoute, async (c) => {
  try {
    // LeanAide doesn't provide a models endpoint, so we return a static list
    // This is based on the models LeanAide supports via configuration
    const models = [
      {
        provider: 'OpenAI',
        models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
      },
      {
        provider: 'Anthropic',
        models: ['claude-3-opus', 'claude-3-sonnet'],
      },
      {
        provider: 'Google',
        models: ['gemini-pro'],
      },
      {
        provider: 'OpenRouter',
        models: ['openai/gpt-4', 'anthropic/claude-3-opus'],
      },
    ];

    return c.json(models, 200);
  } catch (error: any) {
    console.error('[LeanAide] Get models error:', error);
    return c.json(
      {
        error: error.message || 'Failed to get models',
      },
      500
    );
  }
});

// POST /leanaide/benchmark/start - Start benchmark
app.openapi(runBenchmarkRoute, async (c) => {
  try {
    // LeanAide doesn't implement benchmarking yet, return mock response
    const benchmarkId = `bm_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

    return c.json(
      {
        benchmark_id: benchmarkId,
        status: 'pending',
        message: 'Benchmarking is not yet implemented in LeanAide server',
      },
      200
    );
  } catch (error: any) {
    console.error('[LeanAide] Run benchmark error:', error);
    return c.json(
      {
        error: error.message || 'Failed to start benchmark',
      },
      500
    );
  }
});

// GET /leanaide/benchmark/:benchmarkId/results - Get benchmark results
app.openapi(getBenchmarkResultsRoute, async (c) => {
  const { benchmarkId } = c.req.valid('param');

  try {
    // LeanAide doesn't implement benchmarking yet
    return c.json(
      {
        benchmark_id: benchmarkId,
        status: 'not_implemented',
        results: [],
        total: 0,
        successful: 0,
        failed: 0,
        avg_time: 0,
      },
      200
    );
  } catch (error: any) {
    console.error('[LeanAide] Get benchmark results error:', error);
    return c.json(
      {
        error: error.message || 'Failed to get benchmark results',
      },
      500
    );
  }
});

// Health check endpoint
app.get('/health', (c) => {
  return c.json({ status: 'ok', service: 'leanaide-proxy' });
});

export default app;
