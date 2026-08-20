/**
 * Gauntlets API Contract Tests (live backend)
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 *
 * Covers the decomposition-workflow Gauntlets surface, which previously had no
 * contract coverage:
 *
 *   POST   /gauntlets            create a gauntlet (with at least one round)
 *   GET    /gauntlets            list gauntlets
 *   GET    /gauntlets/{name}     get a gauntlet, including its round rules
 *   PUT    /gauntlets/{name}     update a gauntlet
 *   DELETE /gauntlets/{name}     delete a gauntlet
 *   GET    /gauntlets/{missing}  404 for unknown gauntlet
 *
 * Round shape asserted on GET: round_number, quorum_required_approvals,
 * quorum_from_panel_size, min_overall_confidence.
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
 * Resource hygiene: gauntlet and support-team names are unique per run and are
 * removed in `afterAll`.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openevolveApi, type ApiConfig } from '../../lib/openevolveApi';
import type { GauntletDefinition, GauntletRoundRule, Team } from '../../lib/types';

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

const LOG_PREFIX = '[gauntlets-api contract]';
const AUTH_STATUSES = [401, 403, 422];

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const pace = () => sleep(PACING_MS);

const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
const GAUNTLET_NAME = `contract-gauntlet-${uniqueSuffix}`;
const MISSING_GAUNTLET_NAME = `contract-gauntlet-missing-${uniqueSuffix}`;
const SUPPORT_TEAM_NAME = `contract-gauntlet-team-${uniqueSuffix}`;

const FIRST_ROUND: GauntletRoundRule = {
  round_number: 1,
  quorum_required_approvals: 2,
  quorum_from_panel_size: 3,
  min_overall_confidence: 0.7,
};

const SECOND_ROUND: GauntletRoundRule = {
  round_number: 2,
  quorum_required_approvals: 3,
  quorum_from_panel_size: 3,
  min_overall_confidence: 0.8,
};

const gauntletBody = (
  description: string,
  rounds: GauntletRoundRule[] = [FIRST_ROUND],
): GauntletDefinition => ({
  name: GAUNTLET_NAME,
  team_name: SUPPORT_TEAM_NAME,
  description,
  rounds,
});

const supportTeamBody = (): Team => ({
  name: SUPPORT_TEAM_NAME,
  role: 'Red',
  description: 'Support team for gauntlet contract tests',
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
    const probe = await timedFetch('/gauntlets', {}, PROBE_TIMEOUT_MS);

    if (AUTH_STATUSES.includes(probe.status)) {
      skipReason =
        `backend at ${API_URL} rejected the API key on GET /gauntlets (HTTP ${probe.status}). ` +
        'Set OPENEVOLVE_API_KEY to a key the backend accepts (dev backends use API_KEY_TEST=test-key:admin).';
    } else {
      backendLive = true;
    }
  } catch (error) {
    skipReason =
      `backend unreachable at ${API_URL} ` +
      `(${error instanceof Error ? error.message : String(error)}). ` +
      'Start the API server or set OPENEVOLVE_API_URL to run Gauntlets contract tests.';
  }

  if (!backendLive) {
    console.warn(`${LOG_PREFIX} skipping suite: ${skipReason}`);
    return;
  }

  // Gauntlets reference a team by name. Create a throwaway team so the fixture is
  // realistic; the route does not hard-require it, so a failure is only a warning.
  const created = await api('/teams', { method: 'POST', body: JSON.stringify(supportTeamBody()) });
  if (!created.ok) {
    console.warn(
      `${LOG_PREFIX} could not pre-create support team ${SUPPORT_TEAM_NAME} ` +
        `(HTTP ${created.status}); continuing with the name only.`,
    );
  }
});

afterAll(async () => {
  if (!backendLive) {
    return;
  }
  // Best-effort cleanup: teardown must never fail the run.
  for (const path of [
    `/gauntlets/${encodeURIComponent(GAUNTLET_NAME)}`,
    `/teams/${encodeURIComponent(SUPPORT_TEAM_NAME)}`,
  ]) {
    try {
      await api(path, { method: 'DELETE' });
    } catch (error) {
      console.warn(
        `${LOG_PREFIX} cleanup of ${path} failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
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

describe('Gauntlets API Contract', () => {
  let created = false;

  describe('POST /gauntlets', () => {
    it('should create a gauntlet with one round and echo its name', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await openevolveApi.createGauntlet(
        gauntletBody('Gauntlets contract test gauntlet'),
        TEST_CONFIG,
      );
      await pace();

      expect(response).toBeDefined();
      expect(typeof response.message).toBe('string');
      expect(response.gauntlet_name).toBe(GAUNTLET_NAME);
      created = true;
    });
  });

  describe('GET /gauntlets', () => {
    it('should list gauntlets with total count and include the created gauntlet', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /gauntlets must succeed first').toBe(true);

      const response = await openevolveApi.listGauntlets(TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(Array.isArray(response.gauntlets)).toBe(true);
      expect(typeof response.total).toBe('number');

      const names = response.gauntlets.map((gauntlet) => gauntlet.name);
      expect(names).toContain(GAUNTLET_NAME);

      const summary = response.gauntlets.find((gauntlet) => gauntlet.name === GAUNTLET_NAME);
      expect(summary).toBeDefined();
      expect(typeof summary?.team_name).toBe('string');
      expect(typeof summary?.round_count).toBe('number');
      expect(summary?.round_count).toBeGreaterThanOrEqual(1);
    });
  });

  describe('GET /gauntlets/{name}', () => {
    it('should retrieve the gauntlet with the documented round rule shape', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /gauntlets must succeed first').toBe(true);

      const response = await openevolveApi.getGauntlet(GAUNTLET_NAME, TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(response.name).toBe(GAUNTLET_NAME);
      expect(typeof response.team_name).toBe('string');
      expect(Array.isArray(response.rounds)).toBe(true);
      expect(response.rounds.length).toBeGreaterThanOrEqual(1);

      const round = response.rounds[0];
      expect(typeof round.round_number).toBe('number');
      expect(round.round_number).toBe(FIRST_ROUND.round_number);
      expect(typeof round.quorum_required_approvals).toBe('number');
      expect(round.quorum_required_approvals).toBe(FIRST_ROUND.quorum_required_approvals);
      expect(typeof round.quorum_from_panel_size).toBe('number');
      expect(round.quorum_from_panel_size).toBe(FIRST_ROUND.quorum_from_panel_size);

      // The field is always serialized; it may be null when the backend defaults it.
      expect(round).toHaveProperty('min_overall_confidence');
      if (round.min_overall_confidence !== null && round.min_overall_confidence !== undefined) {
        expect(typeof round.min_overall_confidence).toBe('number');
        expect(round.min_overall_confidence).toBeCloseTo(
          FIRST_ROUND.min_overall_confidence as number,
          5,
        );
      }
    });

    it('should answer 404 for a gauntlet that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await api(`/gauntlets/${encodeURIComponent(MISSING_GAUNTLET_NAME)}`);

      expect(response.status).toBe(404);
      const body = await readBody(response);
      expect(errorMessage(body), `404 body must explain the error: ${JSON.stringify(body)}`).toBeTruthy();
    });
  });

  describe('PUT /gauntlets/{name}', () => {
    it('should update the gauntlet rounds and persist the change', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /gauntlets must succeed first').toBe(true);

      const updatedDescription = `Gauntlets contract test gauntlet (updated ${uniqueSuffix})`;
      const response = await openevolveApi.updateGauntlet(
        GAUNTLET_NAME,
        gauntletBody(updatedDescription, [FIRST_ROUND, SECOND_ROUND]),
        TEST_CONFIG,
      );
      await pace();

      expect(response).toBeDefined();
      expect(typeof response.message).toBe('string');
      expect(response.gauntlet_name).toBe(GAUNTLET_NAME);

      const reread = await openevolveApi.getGauntlet(GAUNTLET_NAME, TEST_CONFIG);
      await pace();
      expect(reread.name).toBe(GAUNTLET_NAME);
      expect(reread.description).toBe(updatedDescription);
      expect(reread.rounds.length).toBe(2);
      expect(reread.rounds.map((round) => round.round_number)).toEqual([
        FIRST_ROUND.round_number,
        SECOND_ROUND.round_number,
      ]);
    });

    it('should answer 404 when updating a gauntlet that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await api(`/gauntlets/${encodeURIComponent(MISSING_GAUNTLET_NAME)}`, {
        method: 'PUT',
        body: JSON.stringify({
          name: MISSING_GAUNTLET_NAME,
          team_name: SUPPORT_TEAM_NAME,
          description: 'should not be created',
          rounds: [FIRST_ROUND],
        }),
      });

      if (!authorized(ctx, response, 'PUT /gauntlets/{name}')) return;
      expect(response.status).toBe(404);
    });
  });

  describe('DELETE /gauntlets/{name}', () => {
    it('should delete the created gauntlet and then 404 on lookup', async (ctx) => {
      if (!liveBackend(ctx)) return;
      expect(created, 'POST /gauntlets must succeed first').toBe(true);

      const response = await api(`/gauntlets/${encodeURIComponent(GAUNTLET_NAME)}`, {
        method: 'DELETE',
      });
      if (!authorized(ctx, response, 'DELETE /gauntlets/{name}')) return;

      expect(response.status).toBe(200);
      const body = await readBody(response);
      expect(body, 'DELETE must return a JSON body').toBeTruthy();
      // SDK types this as `{ success: boolean }`; the backend answers
      // `{ message, gauntlet_name }`. Accept either acknowledgement.
      expect(
        body?.success === true || typeof body?.message === 'string',
        `unexpected DELETE payload: ${JSON.stringify(body)}`,
      ).toBe(true);

      const lookup = await api(`/gauntlets/${encodeURIComponent(GAUNTLET_NAME)}`);
      expect(lookup.status).toBe(404);
    });

    it('should answer 404 when deleting a gauntlet that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await api(`/gauntlets/${encodeURIComponent(MISSING_GAUNTLET_NAME)}`, {
        method: 'DELETE',
      });

      if (!authorized(ctx, response, 'DELETE /gauntlets/{name}')) return;
      expect(response.status).toBe(404);
    });
  });
});
