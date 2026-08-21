/**
 * Route-Contract Test — BubbleLab (bubble-studio) OpenEvolve client
 *
 * Purpose: lock the client<->backend path + HTTP-method contract so that a
 * backend route change (or an accidental client-side drift) is caught by CI
 * instead of surfacing as a runtime 404/405 in the UI.
 *
 * This test is OFFLINE: it mocks the `ApiClient` transport (reusing the same
 * mock infrastructure as `openevolveApi.test.ts`) so no server or network is
 * required. For every route in the manifest that has a representative client
 * method, we drive that method and assert the exact path + HTTP verb that the
 * client issues. Routes the client does not yet call are still kept in the
 * manifest (and asserted to be present) so the contract stays documented.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Capture the module-level `ApiClient` instance so we can inspect the
// path + verb each client method ultimately issues. `vi.hoisted` keeps the
// reference reachable inside the hoisted `vi.mock` factory.
const hoisted = vi.hoisted(() => ({
  capturedClient: null as {
    get: ReturnType<typeof vi.fn>;
    post: ReturnType<typeof vi.fn>;
    put: ReturnType<typeof vi.fn>;
    delete: ReturnType<typeof vi.fn>;
  } | null,
}));

vi.mock('@/lib/api', () => ({
  ApiClient: vi.fn().mockImplementation(() => {
    hoisted.capturedClient = {
      get: vi.fn().mockResolvedValue({}),
      post: vi.fn().mockResolvedValue({}),
      put: vi.fn().mockResolvedValue({}),
      delete: vi.fn().mockResolvedValue({}),
    };
    return hoisted.capturedClient;
  }),
}));

vi.mock('@/utils/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import { openevolveApi } from '../openevolveApi';

type Verb = 'GET' | 'POST' | 'PUT' | 'DELETE';

interface ContractEntry {
  /** The canonical backend route (or route group) the client must target. */
  route: string;
  /** The HTTP method the client must use. */
  method: Verb;
  /** The exact path the client method must issue as the first request arg. */
  expectedPath: string;
  /**
   * A representative client method that exercises this route. `null` means the
   * client does not yet call this route (documented gap, kept for the contract).
   */
  call: (() => Promise<unknown>) | null;
}

/**
 * The expected backend route manifest. This is the source of truth for the
 * client<->backend contract and is kept in sync with `openevolveApi.ts`.
 */
const ROUTE_MANIFEST: ContractEntry[] = [
  {
    route: '/api/workflows',
    method: 'POST',
    expectedPath: '/api/workflows',
    call: () =>
      openevolveApi.createWorkflow({
        name: 'contract-wf',
        description: 'route contract',
        workflow_type: 'evolution',
      } as never),
  },
  {
    route: '/api/teams',
    method: 'POST',
    expectedPath: '/api/teams',
    call: () =>
      openevolveApi.createTeam({
        name: 't',
        description: 'd',
        members: [],
      } as never),
  },
  {
    route: '/api/gauntlets',
    method: 'POST',
    expectedPath: '/api/gauntlets',
    call: () =>
      openevolveApi.createGauntlet({
        name: 'g',
        description: 'd',
        rounds: [],
      } as never),
  },
  {
    route: '/api/executions',
    method: 'POST',
    expectedPath: '/api/executions',
    call: () =>
      openevolveApi.executeWorkflow('wf-1', { problem_statement: 'p' } as never),
  },
  // The client does not currently call these routes; kept in the manifest so
  // the contract stays documented and a future wiring is caught.
  { route: '/api/settings', method: 'GET', expectedPath: '/api/settings', call: null },
  { route: '/icr', method: 'GET', expectedPath: '/icr', call: null },
  { route: '/determinism', method: 'GET', expectedPath: '/determinism', call: null },
  {
    route: '/api/decomposition',
    method: 'GET',
    expectedPath: '/api/decomposition',
    call: null,
  },
  { route: '/api/v1/*', method: 'GET', expectedPath: '/api/v1/', call: null },
  {
    route: '/api/parameters',
    method: 'GET',
    expectedPath: '/api/parameters/schema',
    call: () => openevolveApi.getParameterSchema(),
  },
  {
    route: '/api/monitoring',
    method: 'GET',
    expectedPath: '/api/monitoring/dashboard',
    call: () => openevolveApi.getMonitoringDashboard(),
  },
  {
    route: '/api/validation',
    method: 'GET',
    expectedPath: '/api/validation/rules',
    call: () => openevolveApi.listValidationRules(),
  },
  {
    route: '/api/analytics',
    method: 'GET',
    expectedPath: '/api/analytics/workflow-metrics',
    call: () => openevolveApi.getWorkflowMetrics(),
  },
  {
    route: '/api/crewai',
    method: 'GET',
    expectedPath: '/api/crewai/workflows',
    call: () => openevolveApi.listCrewaiWorkflows(),
  },
  {
    route: '/api/version-control',
    method: 'GET',
    expectedPath: '/api/version-control/versions',
    call: () => openevolveApi.listVersions(),
  },
  {
    route: '/api/evaluators',
    method: 'GET',
    expectedPath: '/api/evaluators',
    call: () => openevolveApi.listEvaluators(),
  },
  {
    route: '/api/integrated',
    method: 'POST',
    expectedPath: '/api/integrated/run',
    call: () =>
      openevolveApi.runIntegratedWorkflow({
        content_type: 'x',
        max_iterations: 1,
      } as never),
  },
  {
    route: '/api/bubblelabs/leanaide',
    method: 'GET',
    expectedPath: '/api/bubblelabs/leanaide/status',
    call: () => openevolveApi.bubblelabsLeanAideStatus(),
  },
  {
    route: '/api/knowledge',
    method: 'GET',
    expectedPath: '/api/knowledge/artifacts',
    call: () => openevolveApi.listKnowledgeArtifacts(),
  },
  {
    route: '/bubblelabs/*',
    method: 'GET',
    expectedPath: '/bubblelabs/control/catalog',
    call: () => openevolveApi.getControlCatalog(),
  },
  { route: '/health', method: 'GET', expectedPath: '/health', call: () => openevolveApi.health() },
  {
    route: '/stream/workflow/{id}',
    method: 'GET',
    expectedPath: '/stream/workflow/',
    call: null,
  },
];

/** Every route the manifest is contractually required to declare. */
const REQUIRED_ROUTES = [
  '/api/workflows',
  '/api/teams',
  '/api/gauntlets',
  '/api/executions',
  '/api/settings',
  '/icr',
  '/determinism',
  '/api/decomposition',
  '/api/v1/*',
  '/api/parameters',
  '/api/monitoring',
  '/api/validation',
  '/api/analytics',
  '/api/crewai',
  '/api/version-control',
  '/api/evaluators',
  '/api/integrated',
  '/api/bubblelabs/leanaide',
  '/api/knowledge',
  '/bubblelabs/*',
  '/health',
  '/stream/workflow/{id}',
];

describe('OpenEvolve backend route contract', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('declares the full expected backend route manifest', () => {
    const declared = ROUTE_MANIFEST.map((e) => e.route);
    for (const route of REQUIRED_ROUTES) {
      expect(declared, `manifest is missing required route "${route}"`).toContain(route);
    }
  });

  it('routes every representative client method to the correct path + HTTP method', async () => {
    const covered = ROUTE_MANIFEST.filter((e) => e.call !== null);
    expect(covered.length).toBeGreaterThan(0);

    for (const entry of covered) {
      const client = hoisted.capturedClient;
      expect(client, 'ApiClient transport was not captured').not.toBeNull();

      // The client normalizes backend responses; our mock transport may not
      // satisfy every normalization path, so we tolerate a thrown result and
      // still assert the path + HTTP method that were issued (the contract).
      try {
        await entry.call!();
      } catch {
        /* contract assertion below does not depend on the resolved value */
      }

      const verbMock = {
        GET: client!.get,
        POST: client!.post,
        PUT: client!.put,
        DELETE: client!.delete,
      }[entry.method];

      // Correct verb was used.
      expect(
        verbMock.mock.calls.length,
        `${entry.route}: expected ${entry.method} to be called once`
      ).toBe(1);

      // Correct path was issued.
      expect(
        verbMock.mock.calls[0][0],
        `${entry.route}: wrong path for ${entry.method}`
      ).toBe(entry.expectedPath);

      // No other HTTP method was used (locks the verb).
      const otherVerbs = (['GET', 'POST', 'PUT', 'DELETE'] as Verb[]).filter(
        (v) => v !== entry.method
      );
      for (const other of otherVerbs) {
        const otherMock = {
          GET: client!.get,
          POST: client!.post,
          PUT: client!.put,
          DELETE: client!.delete,
        }[other];
        expect(
          otherMock.mock.calls.length,
          `${entry.route}: ${other} should NOT have been called`
        ).toBe(0);
      }

      vi.clearAllMocks();
    }
  });
});
