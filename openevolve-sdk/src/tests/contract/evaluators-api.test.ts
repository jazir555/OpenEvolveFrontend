/**
 * Evaluators API Contract Tests (live backend)
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 *
 * Covers the decomposition-workflow Evaluators surface, which previously had no
 * contract coverage:
 *
 *   POST   /evaluators            upload an evaluator from `{ code }`
 *   GET    /evaluators            list evaluators (id -> code map)
 *   DELETE /evaluators/{id}       delete an evaluator
 *   DELETE /evaluators/{missing}  404 for unknown evaluator
 *
 * The uploaded snippet is intentionally trivial and side-effect free: the backend
 * only stores the file (it must define an `evaluate` function to pass upload
 * validation) and never executes it as part of these tests.
 *
 * Skip-on-unreachable pattern (same shape as `execution-api.test.ts` /
 * `src/integration/backend-e2e.test.ts`):
 *   - base URL comes from OPENEVOLVE_API_URL (default http://localhost:8000)
 *   - a `beforeAll` probe decides whether the backend is live; when the socket is
 *     refused (ECONNREFUSED), DNS fails, or the probe times out, every test is
 *     SKIPPED via `ctx.skip()` instead of failing
 *   - when the backend answers but rejects the API key (401/403/422) the suite is
 *     also skipped, with a warning explaining the missing provisioning
 *
 * Transport notes:
 *   - happy paths call the SDK client (`openevolveApi.*`)
 *   - negative paths (expected 404s, role-gated DELETE) use raw `fetch` to avoid
 *     the SDK retry + circuit-breaker wrapper amplifying expected failures
 *   - requests are paced because the backend rate-limits per API key
 *
 * Resource hygiene: the uploaded evaluator id is captured and deleted in
 * `afterAll`, even if an assertion fails mid-suite.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openevolveApi, type ApiConfig } from '../../lib/openevolveApi';

const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.OPENEVOLVE_API_KEY || 'test-key';
const TIMEOUT = 30000;

const PROBE_TIMEOUT_MS = Number(process.env.OPENEVOLVE_CONTRACT_PROBE_TIMEOUT_MS || 5000);
const PACING_MS = Number(process.env.OPENEVOLVE_CONTRACT_PACING_MS || 250);

const TEST_CONFIG: ApiConfig = {
  baseUrl: API_URL,
  apiKey: API_KEY,
  timeout: TIMEOUT,
};

const JSON_HEADERS: Record<string, string> = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

const LOG_PREFIX = '[evaluators-api contract]';
const AUTH_STATUSES = [401, 403, 422];

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const pace = () => sleep(PACING_MS);

const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
const MISSING_EVALUATOR_ID = `eval_missing_${Math.random().toString(36).slice(2, 10)}`;

/**
 * Trivial, safe evaluator. Upload validation requires a `def evaluate` function;
 * the body performs no I/O, imports nothing, and returns a constant score.
 */
const EVALUATOR_CODE = [
  '"""Contract-test evaluator (safe, constant score)."""',
  '',
  '',
  'def evaluate(program_path):',
  `    """Marker: ${uniqueSuffix}. Ignores the program and returns a fixed metric."""`,
  '    return {"score": 1.0}',
  '',
].join('\n');

/** fetch with an explicit abort timeout so a dead host cannot hang the suite. */
async function timedFetch(
  path: string,
  init: RequestInit = {},
  timeoutMs = TIMEOUT,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { ...JSON_HEADERS, ...((init.headers as Record<string, string>) || {}) },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

/** Raw-fetch helper for status-code assertions, with rate-limiter pacing. */
async function api(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await timedFetch(path, init);
  await pace();
  return response;
}

const readBody = async (response: Response): Promise<any> => {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
};

const errorMessage = (body: any): string | undefined => {
  if (!body || typeof body !== 'object') {
    return typeof body === 'string' ? body : undefined;
  }
  if (typeof body.detail === 'string') return body.detail;
  if (typeof body.error === 'string') return body.error;
  return undefined;
};

/** The route returns `{ evaluators: { id: code } }`; tolerate an id array too. */
const evaluatorIds = (evaluators: unknown): string[] => {
  if (Array.isArray(evaluators)) {
    return evaluators.map((entry) =>
      typeof entry === 'string' ? entry : String((entry as { id?: unknown })?.id ?? ''),
    );
  }
  if (evaluators && typeof evaluators === 'object') {
    return Object.keys(evaluators as Record<string, unknown>);
  }
  return [];
};

let backendLive = false;
let skipReason = 'backend not probed';
let evaluatorId: string | undefined;

beforeAll(async () => {
  try {
    const probe = await timedFetch('/evaluators', {}, PROBE_TIMEOUT_MS);

    if (AUTH_STATUSES.includes(probe.status)) {
      skipReason =
        `backend at ${API_URL} rejected the API key on GET /evaluators (HTTP ${probe.status}). ` +
        'Set OPENEVOLVE_API_KEY to a key the backend accepts (dev backends use API_KEY_TEST=test-key:admin).';
    } else {
      backendLive = true;
    }
  } catch (error) {
    skipReason =
      `backend unreachable at ${API_URL} ` +
      `(${error instanceof Error ? error.message : String(error)}). ` +
      'Start the API server or set OPENEVOLVE_API_URL to run Evaluators contract tests.';
  }

  if (!backendLive) {
    console.warn(`${LOG_PREFIX} skipping suite: ${skipReason}`);
  }
  await pace();
});

afterAll(async () => {
  if (!backendLive || !evaluatorId) {
    return;
  }
  // Best-effort cleanup in case the delete test skipped or failed.
  try {
    await api(`/evaluators/${encodeURIComponent(evaluatorId)}`, { method: 'DELETE' });
  } catch (error) {
    console.warn(
      `${LOG_PREFIX} cleanup of ${evaluatorId} failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
});

/** Guard replacing `skipIf` for a value only known after the async probe. */
const liveBackend = (ctx: { skip: () => void }): boolean => {
  if (!backendLive) {
    console.warn(`${LOG_PREFIX} ${skipReason}`);
    ctx.skip();
    return false;
  }
  return true;
};

/** Role-gated routes (DELETE requires ADMIN) skip rather than fail on 401/403. */
const authorized = (
  ctx: { skip: () => void },
  response: Response,
  label: string,
): boolean => {
  if (AUTH_STATUSES.includes(response.status)) {
    console.warn(
      `${LOG_PREFIX} ${label} returned HTTP ${response.status}; the configured API key lacks the ` +
        'required role (this route needs ADMIN). Skipping.',
    );
    ctx.skip();
    return false;
  }
  return true;
};

describe('Evaluators API Contract', () => {
  describe('POST /evaluators', () => {
    it('should upload an evaluator and return its id', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await openevolveApi.uploadEvaluator({ code: EVALUATOR_CODE }, TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(typeof response.evaluator_id).toBe('string');
      expect(response.evaluator_id.length).toBeGreaterThan(0);
      // Observed backend format is `eval_<hex>`; only the non-empty string id is
      // part of the contract the SDK relies on.
      evaluatorId = response.evaluator_id;
    });
  });

  describe('GET /evaluators', () => {
    it('should list evaluators and include the uploaded one', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(evaluatorId, 'POST /evaluators must succeed first').toBeTruthy();

      const response = await openevolveApi.listEvaluators(TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(response.evaluators, 'response must expose an `evaluators` map').toBeDefined();
      expect(Array.isArray(response.evaluators)).toBe(false);
      expect(typeof response.evaluators).toBe('object');

      const ids = evaluatorIds(response.evaluators);
      expect(ids).toContain(evaluatorId as string);

      const stored = (response.evaluators as Record<string, string>)[evaluatorId as string];
      expect(typeof stored).toBe('string');
      expect(stored).toContain('def evaluate');
    });
  });

  describe('DELETE /evaluators/{id}', () => {
    it('should delete the uploaded evaluator and report success', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(evaluatorId, 'POST /evaluators must succeed first').toBeTruthy();

      const response = await api(`/evaluators/${encodeURIComponent(evaluatorId as string)}`, {
        method: 'DELETE',
      });
      if (!authorized(ctx, response, 'DELETE /evaluators/{id}')) return;

      expect(response.status).toBe(200);
      const body = await readBody(response);
      expect(body, 'DELETE must return a JSON body').toBeTruthy();
      expect(body?.success).toBe(true);
      expect(body?.evaluator_id).toBe(evaluatorId);

      // The evaluator must be gone from the listing.
      const listed = await openevolveApi.listEvaluators(TEST_CONFIG);
      await pace();
      expect(evaluatorIds(listed.evaluators)).not.toContain(evaluatorId as string);

      evaluatorId = undefined; // nothing left for afterAll to clean up
    });

    it('should answer 404 when deleting an evaluator that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      // Raw fetch: the SDK client would retry this expected failure and feed its
      // shared circuit breaker.
      const response = await api(`/evaluators/${encodeURIComponent(MISSING_EVALUATOR_ID)}`, {
        method: 'DELETE',
      });

      if (!authorized(ctx, response, 'DELETE /evaluators/{id}')) return;

      expect(response.status).toBe(404);
      const body = await readBody(response);
      expect(errorMessage(body), `404 body must explain the error: ${JSON.stringify(body)}`).toBeTruthy();
    });
  });
});
