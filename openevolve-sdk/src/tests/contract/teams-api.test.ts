/**
 * Teams API Contract Tests (live backend)
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 *
 * Covers the decomposition-workflow Teams surface, which previously had no
 * contract coverage:
 *
 *   POST   /teams            create a team
 *   GET    /teams            list teams
 *   GET    /teams/{name}     get a single team
 *   PUT    /teams/{name}     update a team
 *   DELETE /teams/{name}     delete a team
 *   GET    /teams/{missing}  404 for unknown team
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
 *   - happy paths call the SDK client (`openevolveApi.*`) so the client's own
 *     method + JSON contract is what gets verified
 *   - negative paths (expected 404s, role-gated DELETE) use raw `fetch`, because
 *     the SDK `request()` helper wraps every call in retry + circuit-breaker
 *     logic that would turn one expected 404 into several retried failures
 *   - requests are paced because the backend runs a token-bucket rate limiter
 *
 * Resource hygiene: every team name is unique per run and is deleted in
 * `afterAll`, so repeated runs never collide.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openevolveApi, type ApiConfig } from '../../lib/openevolveApi';
import type { Team } from '../../lib/types';

const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.OPENEVOLVE_API_KEY || 'test-key';
const TIMEOUT = 30000;

/** Short probe timeout so an unreachable/black-holed host skips fast. */
const PROBE_TIMEOUT_MS = Number(process.env.OPENEVOLVE_CONTRACT_PROBE_TIMEOUT_MS || 5000);
/** The backend rate-limits per API key; pace requests so a 429 is not misread. */
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

const LOG_PREFIX = '[teams-api contract]';
const AUTH_STATUSES = [401, 403, 422];

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const pace = () => sleep(PACING_MS);

const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
const TEAM_NAME = `contract-team-${uniqueSuffix}`;
const MISSING_TEAM_NAME = `contract-team-missing-${uniqueSuffix}`;

const teamBody = (description: string): Team => ({
  name: TEAM_NAME,
  role: 'Blue',
  description,
  members: [
    {
      model_id: 'contract-test-model',
      api_key: 'contract-test-key',
      temperature: 0.2,
      max_tokens: 256,
    },
  ],
});

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

/** FastAPI reports HTTPException as `detail`; the global handler adds `error`. */
const errorMessage = (body: any): string | undefined => {
  if (!body || typeof body !== 'object') {
    return typeof body === 'string' ? body : undefined;
  }
  if (typeof body.detail === 'string') return body.detail;
  if (typeof body.error === 'string') return body.error;
  return undefined;
};

let backendLive = false;
let skipReason = 'backend not probed';

beforeAll(async () => {
  try {
    const probe = await timedFetch('/teams', {}, PROBE_TIMEOUT_MS);

    if (AUTH_STATUSES.includes(probe.status)) {
      skipReason =
        `backend at ${API_URL} rejected the API key on GET /teams (HTTP ${probe.status}). ` +
        'Set OPENEVOLVE_API_KEY to a key the backend accepts (dev backends use API_KEY_TEST=test-key:admin).';
    } else {
      // Any other HTTP answer means the service is live; real contract problems
      // (5xx, wrong payload shape) must fail the assertions below, not skip.
      backendLive = true;
    }
  } catch (error) {
    skipReason =
      `backend unreachable at ${API_URL} ` +
      `(${error instanceof Error ? error.message : String(error)}). ` +
      'Start the API server or set OPENEVOLVE_API_URL to run Teams contract tests.';
  }

  if (!backendLive) {
    console.warn(`${LOG_PREFIX} skipping suite: ${skipReason}`);
  }
  await pace();
});

afterAll(async () => {
  if (!backendLive) {
    return;
  }
  // Best-effort cleanup: never let teardown fail the run.
  try {
    await api(`/teams/${encodeURIComponent(TEAM_NAME)}`, { method: 'DELETE' });
  } catch (error) {
    console.warn(
      `${LOG_PREFIX} cleanup of ${TEAM_NAME} failed: ${error instanceof Error ? error.message : String(error)}`,
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

describe('Teams API Contract', () => {
  let created = false;

  describe('POST /teams', () => {
    it('should create a team and echo its name', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await openevolveApi.createTeam(
        teamBody('Teams contract test team'),
        TEST_CONFIG,
      );
      await pace();

      expect(response).toBeDefined();
      expect(typeof response.message).toBe('string');
      expect(response.team_name).toBe(TEAM_NAME);
      created = true;
    });
  });

  describe('GET /teams', () => {
    it('should list teams with total count and include the created team', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /teams must succeed first').toBe(true);

      const response = await openevolveApi.listTeams(TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(Array.isArray(response.teams)).toBe(true);
      expect(typeof response.total).toBe('number');

      const names = response.teams.map((team) => team.name);
      expect(names).toContain(TEAM_NAME);

      const summary = response.teams.find((team) => team.name === TEAM_NAME);
      expect(summary).toBeDefined();
      expect(typeof summary?.name).toBe('string');
      expect(typeof summary?.role).toBe('string');
      expect(typeof summary?.member_count).toBe('number');
      expect(summary?.member_count).toBeGreaterThanOrEqual(1);
    });
  });

  describe('GET /teams/{name}', () => {
    it('should retrieve the created team with members', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /teams must succeed first').toBe(true);

      const response = await openevolveApi.getTeam(TEAM_NAME, TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(response.name).toBe(TEAM_NAME);
      expect(typeof response.role).toBe('string');
      expect(Array.isArray(response.members)).toBe(true);
      expect(response.members.length).toBeGreaterThanOrEqual(1);
      expect(typeof response.members[0].model_id).toBe('string');
    });

    it('should answer 404 for a team that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      // Raw fetch: the SDK client would retry this expected failure 3 times and
      // feed its shared circuit breaker.
      const response = await api(`/teams/${encodeURIComponent(MISSING_TEAM_NAME)}`);

      expect(response.status).toBe(404);
      const body = await readBody(response);
      expect(errorMessage(body), `404 body must explain the error: ${JSON.stringify(body)}`).toBeTruthy();
    });
  });

  describe('PUT /teams/{name}', () => {
    it('should update the team and persist the change', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /teams must succeed first').toBe(true);

      const updatedDescription = `Teams contract test team (updated ${uniqueSuffix})`;
      const response = await openevolveApi.updateTeam(
        TEAM_NAME,
        teamBody(updatedDescription),
        TEST_CONFIG,
      );
      await pace();

      expect(response).toBeDefined();
      expect(typeof response.message).toBe('string');
      expect(response.team_name).toBe(TEAM_NAME);

      const reread = await openevolveApi.getTeam(TEAM_NAME, TEST_CONFIG);
      await pace();
      expect(reread.name).toBe(TEAM_NAME);
      expect(reread.description).toBe(updatedDescription);
    });

    it('should answer 404 when updating a team that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await api(`/teams/${encodeURIComponent(MISSING_TEAM_NAME)}`, {
        method: 'PUT',
        body: JSON.stringify({
          name: MISSING_TEAM_NAME,
          role: 'Blue',
          description: 'should not be created',
          members: [{ model_id: 'contract-test-model', api_key: 'contract-test-key' }],
        }),
      });

      if (!authorized(ctx, response, 'PUT /teams/{name}')) return;
      expect(response.status).toBe(404);
    });
  });

  describe('DELETE /teams/{name}', () => {
    it('should delete the created team and then 404 on lookup', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /teams must succeed first').toBe(true);

      const response = await api(`/teams/${encodeURIComponent(TEAM_NAME)}`, { method: 'DELETE' });
      if (!authorized(ctx, response, 'DELETE /teams/{name}')) return;

      expect(response.status).toBe(200);
      const body = await readBody(response);
      expect(body, 'DELETE must return a JSON body').toBeTruthy();
      // The SDK types this as `{ success: boolean }` while the backend answers
      // `{ message, team_name }`; accept either so the test pins the behaviour
      // that matters (the delete was acknowledged) without encoding one wording.
      expect(
        body?.success === true || typeof body?.message === 'string',
        `unexpected DELETE payload: ${JSON.stringify(body)}`,
      ).toBe(true);

      const lookup = await api(`/teams/${encodeURIComponent(TEAM_NAME)}`);
      expect(lookup.status).toBe(404);
    });

    it('should answer 404 when deleting a team that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await api(`/teams/${encodeURIComponent(MISSING_TEAM_NAME)}`, {
        method: 'DELETE',
      });

      if (!authorized(ctx, response, 'DELETE /teams/{name}')) return;
      expect(response.status).toBe(404);
    });
  });
});
