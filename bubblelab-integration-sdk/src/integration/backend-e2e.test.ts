/**
 * Live backend end-to-end integration tests.
 *
 * These tests do NOT run by default. They are opt-in via `RUN_BACKEND_E2E=1`, so
 * `npm test` stays hermetic:
 *
 *   RUN_BACKEND_E2E=1 npm test -- backend-e2e
 *
 * Unlike `src/lib/openevolveApi.test.ts` (which talks to the unprefixed routes),
 * this suite drives the canonical client `src/lib/openevolveApi.ts` against the
 * `/api`-prefixed surface, exercising the FastAPI `rewrite_api_prefix` middleware
 * in `engines/other/api_server.py` that maps `/api/<path>` -> `/<path>`.
 *
 * Prerequisites:
 *   - Backend listening on http://127.0.0.1:8000 (see repo README / start_api_server)
 *   - `API_KEY_TEST=test-key:admin` in the backend environment
 *
 * Configuration (all optional, defaults target a local dev backend):
 *   RUN_BACKEND_E2E         must be set to enable this suite
 *   OPENEVOLVE_API_BASE_URL default http://127.0.0.1:8000/api
 *   OPENEVOLVE_API_KEY      default test-key
 *
 * Note on status codes: the canonical client throws on any non-2xx response
 * (see `request()` -> `if (!response.ok) throw`). A resolved promise therefore
 * proves the backend answered 2xx; the assertions below then pin the payload shape.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openevolveApi, ApiConfig } from '../lib/openevolveApi';

const RUN_BACKEND_E2E = Boolean(process.env.RUN_BACKEND_E2E);

const BASE_URL = process.env.OPENEVOLVE_API_BASE_URL || 'http://127.0.0.1:8000/api';
const API_KEY = process.env.OPENEVOLVE_API_KEY || 'test-key';

const TEST_CONFIG: ApiConfig = {
  baseUrl: BASE_URL,
  apiKey: API_KEY,
  timeout: 20000,
};

/**
 * The backend enables a token-bucket rate limiter (burst 10, ~100 req/min per
 * API key). Pace requests so a fast suite cannot trip a 429 that would be
 * misreported as a contract failure.
 */
const PACING_MS = 750;
const pace = () => new Promise<void>((resolve) => setTimeout(resolve, PACING_MS));

/** Mismatches are collected so a single run reports every shape problem at once. */
const mismatches: string[] = [];

const preview = (value: unknown, max = 600): string => {
  let text: string;
  try {
    text = JSON.stringify(value);
  } catch {
    text = String(value);
  }
  if (text === undefined) {
    text = String(value);
  }
  return text.length > max ? `${text.slice(0, max)}...[truncated]` : text;
};

/**
 * Assert a client-expected field exists with the right runtime type, and record a
 * human-readable mismatch (with the raw server response) when it does not.
 */
const expectField = (
  endpoint: string,
  expectedType: 'array' | 'object' | 'string' | 'number',
  fieldPath: string,
  actual: unknown,
  response: unknown,
): void => {
  const ok =
    expectedType === 'array'
      ? Array.isArray(actual)
      : expectedType === 'object'
        ? typeof actual === 'object' && actual !== null && !Array.isArray(actual)
        : typeof actual === expectedType;

  if (!ok) {
    const actualType = Array.isArray(actual) ? 'array' : actual === null ? 'null' : typeof actual;
    mismatches.push(
      `${endpoint}: client expects \`${fieldPath}\` to be ${expectedType}, got ${actualType}. ` +
        `Server response: ${preview(response)}`,
    );
  }

  expect(
    ok,
    `${endpoint}: expected \`${fieldPath}\` to be ${expectedType}. Server response: ${preview(response)}`,
  ).toBe(true);
};

describe.skipIf(!RUN_BACKEND_E2E)('Live backend E2E via /api prefix', () => {
  beforeAll(async () => {
    // Fail fast and loudly (rather than 6 confusing timeouts) if nothing is listening.
    const health = await fetch(`${BASE_URL}/health`, { headers: { 'X-API-Key': API_KEY } }).catch(
      (error: unknown) => {
        throw new Error(
          `Backend unreachable at ${BASE_URL}/health: ${error instanceof Error ? error.message : String(error)}. ` +
            'Start the API server on 127.0.0.1:8000 before running RUN_BACKEND_E2E=1.',
        );
      },
    );
    expect(
      health.ok,
      `Backend health probe via ${BASE_URL}/health returned ${health.status}`,
    ).toBe(true);
    await pace();
  });

  afterAll(() => {
    if (mismatches.length > 0) {
      console.error(
        `\n=== BACKEND CONTRACT MISMATCHES (${mismatches.length}) ===\n${mismatches.join('\n')}\n`,
      );
    } else {
      console.log('\n=== BACKEND CONTRACT: no mismatches detected ===\n');
    }
  });

  describe('/api/workflows', () => {
    it('listWorkflows returns { workflows: [], total: number }', async () => {
      const response = await openevolveApi.listWorkflows(TEST_CONFIG);

      expect(response).toBeDefined();
      expectField('/api/workflows', 'array', 'workflows', response?.workflows, response);
      expectField('/api/workflows', 'number', 'total', response?.total, response);
      await pace();
    });
  });

  describe('/api/teams', () => {
    it('listTeams returns { teams: [], total: number }', async () => {
      const response = await openevolveApi.listTeams(TEST_CONFIG);

      expect(response).toBeDefined();
      expectField('/api/teams', 'array', 'teams', response?.teams, response);
      expectField('/api/teams', 'number', 'total', response?.total, response);
      await pace();
    });
  });

  describe('/api/gauntlets', () => {
    it('listGauntlets returns { gauntlets: [], total: number }', async () => {
      const response = await openevolveApi.listGauntlets(TEST_CONFIG);

      expect(response).toBeDefined();
      expectField('/api/gauntlets', 'array', 'gauntlets', response?.gauntlets, response);
      expectField('/api/gauntlets', 'number', 'total', response?.total, response);
      await pace();
    });
  });

  describe('/api/executions', () => {
    let createdExecutionId: string | undefined;

    it('createExecution returns a 2xx execution record', async () => {
      const created = await openevolveApi.createExecution(
        { name: `e2e-${Date.now()}`, workflow_id: 'e2e-workflow' },
        TEST_CONFIG,
      );

      expect(created).toBeDefined();
      expectField('/api/executions (POST)', 'string', 'id', created?.id, created);
      expectField('/api/executions (POST)', 'string', 'status', created?.status, created);
      expectField('/api/executions (POST)', 'string', 'created_at', created?.created_at, created);

      createdExecutionId = created?.id;
      await pace();
    });

    it('listExecutions reflects the created execution', async () => {
      expect(createdExecutionId, 'createExecution must run first').toBeTruthy();

      const response = await openevolveApi.listExecutions(undefined, TEST_CONFIG);

      expect(response).toBeDefined();
      expectField('/api/executions (GET)', 'array', 'executions', response?.executions, response);
      expectField('/api/executions (GET)', 'number', 'total', response?.total, response);

      const ids = (response?.executions ?? []).map((execution) => execution.id);
      expect(
        ids,
        `/api/executions (GET) did not include created execution ${createdExecutionId}. ` +
          `Server response: ${preview(response)}`,
      ).toContain(createdExecutionId);
      await pace();
    });

    it('cancelExecution transitions the record status', async () => {
      expect(createdExecutionId, 'createExecution must run first').toBeTruthy();

      const cancelled = await openevolveApi.cancelExecution(createdExecutionId as string, TEST_CONFIG);

      expect(cancelled).toBeDefined();
      expectField(
        '/api/executions/{id}/cancel',
        'string',
        'status',
        cancelled?.status,
        cancelled,
      );
      expect(cancelled.id).toBe(createdExecutionId);
      expect(cancelled.status).toBe('cancelled');
      await pace();
    });
  });

  describe('/api/bubblelabs/knowledge (explorer)', () => {
    it('query-advanced (exploreKnowledge) returns { results, history }', async () => {
      const response = await openevolveApi.bubblelabsKnowledgeQueryAdvanced(
        { query: 'openevolve backend e2e probe' },
        TEST_CONFIG,
      );

      expect(response).toBeDefined();
      expectField(
        '/api/bubblelabs/knowledge/query-advanced',
        'object',
        'results',
        response?.results,
        response,
      );
      expectField(
        '/api/bubblelabs/knowledge/query-advanced',
        'array',
        'history',
        response?.history,
        response,
      );
      await pace();
    });

    it('query-history (listKnowledgeQueries) returns { history }', async () => {
      const response = await openevolveApi.bubblelabsKnowledgeQueryHistory(TEST_CONFIG);

      expect(response).toBeDefined();
      expectField(
        '/api/bubblelabs/knowledge/query-history',
        'array',
        'history',
        response?.history,
        response,
      );
      await pace();
    });
  });

  describe('/api/bubblelabs/leanaide/trees', () => {
    it('listLeanAideTrees returns { tree_ids: string[] }', async () => {
      const response = await openevolveApi.bubblelabsLeanAideTrees(TEST_CONFIG);

      expect(response).toBeDefined();
      expectField(
        '/api/bubblelabs/leanaide/trees',
        'array',
        'tree_ids',
        response?.tree_ids,
        response,
      );
      for (const treeId of response?.tree_ids ?? []) {
        expect(
          typeof treeId,
          `/api/bubblelabs/leanaide/trees: tree_ids entries must be strings. Server response: ${preview(response)}`,
        ).toBe('string');
      }
      await pace();
    });
  });
});
