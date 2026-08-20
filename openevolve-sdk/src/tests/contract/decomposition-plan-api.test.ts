/**
 * Workflow Decomposition Plan API Contract Tests (live backend)
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 *
 * Covers the decomposition-plan surface of the workflow API, which previously had
 * no contract coverage:
 *
 *   GET /workflows                                  discover workflows to inspect
 *   GET /workflows/{id}/decomposition-plan          read the plan + dependency graph
 *   PUT /workflows/{id}/decomposition-plan          edit sub-problems, re-read to confirm
 *   GET /workflows/{missing}/decomposition-plan     404 for unknown workflow
 *   PUT /workflows/{missing}/decomposition-plan     404 for unknown workflow
 *
 * Fixture discovery (a plan only exists once a workflow has been decomposed):
 *   1. list workflows and adopt the first one whose plan endpoint answers 200
 *   2. otherwise create a support team + gauntlet and a workflow via the SDK
 *      `createWorkflow`, then re-check for a plan
 *   3. if nothing exposes a plan, every test SKIPS with an explanatory warning
 *      instead of failing
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
 *   - happy paths call the SDK client (`openevolveApi.getWorkflowPlan` /
 *     `updateWorkflowPlan`)
 *   - negative paths use raw `fetch` to avoid the SDK retry + circuit-breaker
 *     wrapper amplifying expected failures
 *   - requests are paced because the backend rate-limits per API key
 *
 * Resource hygiene: when the suite adopts a pre-existing workflow it restores the
 * original sub-problem description in `afterAll`; anything the suite created
 * itself (workflow, team, gauntlet) is deleted there too.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openevolveApi, type ApiConfig } from '../../lib/openevolveApi';
import type {
  GauntletDefinition,
  Team,
  WorkflowPlanUpdateRequest,
  WorkflowSubProblem,
} from '../../lib/types';

const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.OPENEVOLVE_API_KEY || 'test-key';
const TIMEOUT = 30000;

const PROBE_TIMEOUT_MS = Number(process.env.OPENEVOLVE_CONTRACT_PROBE_TIMEOUT_MS || 5000);
const PACING_MS = Number(process.env.OPENEVOLVE_CONTRACT_PACING_MS || 250);
/** Cap the fixture scan so the rate limiter is never hammered. */
const MAX_WORKFLOW_SCAN = Number(process.env.OPENEVOLVE_CONTRACT_WORKFLOW_SCAN || 5);

const TEST_CONFIG: ApiConfig = {
  baseUrl: API_URL,
  apiKey: API_KEY,
  timeout: TIMEOUT,
};

const JSON_HEADERS: Record<string, string> = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

const LOG_PREFIX = '[decomposition-plan-api contract]';
const AUTH_STATUSES = [401, 403, 422];

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const pace = () => sleep(PACING_MS);

const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
const MISSING_WORKFLOW_ID = `contract-workflow-missing-${uniqueSuffix}`;
const SUPPORT_TEAM_NAME = `contract-plan-team-${uniqueSuffix}`;
const SUPPORT_GAUNTLET_NAME = `contract-plan-gauntlet-${uniqueSuffix}`;

const planPath = (workflowId: string) =>
  `/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`;

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

const supportTeamBody = (): Team => ({
  name: SUPPORT_TEAM_NAME,
  role: 'Blue',
  description: 'Support team for decomposition-plan contract tests',
  members: [
    {
      model_id: 'contract-test-model',
      api_key: 'contract-test-key',
      temperature: 0.2,
      max_tokens: 256,
    },
  ],
});

const supportGauntletBody = (): GauntletDefinition => ({
  name: SUPPORT_GAUNTLET_NAME,
  team_name: SUPPORT_TEAM_NAME,
  description: 'Support gauntlet for decomposition-plan contract tests',
  rounds: [
    {
      round_number: 1,
      quorum_required_approvals: 1,
      quorum_from_panel_size: 1,
      min_overall_confidence: 0.5,
    },
  ],
});

let backendLive = false;
let skipReason = 'backend not probed';

/** Workflow whose plan the suite exercises (may be pre-existing or created here). */
let planWorkflowId: string | undefined;
let planSkipReason = '';
/** Set when this suite created the workflow, so teardown can remove it. */
let createdWorkflowId: string | undefined;
let createdSupportFixtures = false;
/** Original description of the sub-problem mutated by the PUT test. */
let mutatedSubProblemId: string | undefined;
let originalSubProblemDescription: string | undefined;

/** Fetch a plan payload, or undefined when the workflow has no plan yet. */
async function fetchPlan(workflowId: string): Promise<any | undefined> {
  const response = await api(planPath(workflowId));
  if (!response.ok) {
    return undefined;
  }
  const body = await readBody(response);
  return Array.isArray(body?.plan?.sub_problems) && body.plan.sub_problems.length > 0
    ? body
    : undefined;
}

/** Adopt the first listed workflow that already exposes a decomposition plan. */
async function discoverExistingPlanWorkflow(): Promise<string | undefined> {
  const listResponse = await api('/workflows');
  if (!listResponse.ok) {
    return undefined;
  }
  const body = await readBody(listResponse);
  const workflows: Array<{ workflow_id?: string }> = Array.isArray(body?.workflows)
    ? body.workflows
    : [];

  for (const workflow of workflows.slice(0, MAX_WORKFLOW_SCAN)) {
    const workflowId = workflow?.workflow_id;
    if (!workflowId) continue;
    const plan = await fetchPlan(workflowId);
    if (plan) {
      return workflowId;
    }
  }
  return undefined;
}

/** Create team + gauntlet + workflow so a plan can exist at all. */
async function createWorkflowFixture(): Promise<string | undefined> {
  const teamResponse = await api('/teams', {
    method: 'POST',
    body: JSON.stringify(supportTeamBody()),
  });
  const gauntletResponse = await api('/gauntlets', {
    method: 'POST',
    body: JSON.stringify(supportGauntletBody()),
  });

  if (!teamResponse.ok || !gauntletResponse.ok) {
    console.warn(
      `${LOG_PREFIX} could not create support fixtures ` +
        `(team HTTP ${teamResponse.status}, gauntlet HTTP ${gauntletResponse.status}).`,
    );
    createdSupportFixtures = teamResponse.ok || gauntletResponse.ok;
    return undefined;
  }
  createdSupportFixtures = true;

  try {
    // One team/gauntlet can satisfy every role slot for a contract-only workflow.
    const created = await openevolveApi.createWorkflow(
      {
        problem_statement:
          'Contract test: verify the decomposition-plan API surface without running evolution.',
        content_analyzer_team: SUPPORT_TEAM_NAME,
        planner_team: SUPPORT_TEAM_NAME,
        solver_team: SUPPORT_TEAM_NAME,
        patcher_team: SUPPORT_TEAM_NAME,
        assembler_team: SUPPORT_TEAM_NAME,
        sub_problem_red_gauntlet: SUPPORT_GAUNTLET_NAME,
        sub_problem_gold_gauntlet: SUPPORT_GAUNTLET_NAME,
        final_red_gauntlet: SUPPORT_GAUNTLET_NAME,
        final_gold_gauntlet: SUPPORT_GAUNTLET_NAME,
        solver_generation_gauntlet: SUPPORT_GAUNTLET_NAME,
        max_refinement_loops: 1,
      },
      TEST_CONFIG,
    );
    await pace();
    createdWorkflowId = created?.workflow_id;
    return createdWorkflowId;
  } catch (error) {
    console.warn(
      `${LOG_PREFIX} createWorkflow failed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return undefined;
  }
}

beforeAll(async () => {
  try {
    const probe = await timedFetch('/workflows', {}, PROBE_TIMEOUT_MS);

    if (AUTH_STATUSES.includes(probe.status)) {
      skipReason =
        `backend at ${API_URL} rejected the API key on GET /workflows (HTTP ${probe.status}). ` +
        'Set OPENEVOLVE_API_KEY to a key the backend accepts (dev backends use API_KEY_TEST=test-key:admin).';
    } else {
      backendLive = true;
    }
  } catch (error) {
    skipReason =
      `backend unreachable at ${API_URL} ` +
      `(${error instanceof Error ? error.message : String(error)}). ` +
      'Start the API server or set OPENEVOLVE_API_URL to run decomposition-plan contract tests.';
  }

  if (!backendLive) {
    console.warn(`${LOG_PREFIX} skipping suite: ${skipReason}`);
    return;
  }
  await pace();

  planWorkflowId = await discoverExistingPlanWorkflow();

  if (!planWorkflowId) {
    const candidate = await createWorkflowFixture();
    if (candidate && (await fetchPlan(candidate))) {
      planWorkflowId = candidate;
    }
  }

  if (!planWorkflowId) {
    planSkipReason =
      'no workflow exposes a decomposition plan (a plan only exists after the planner stage has ' +
      'run). Run a decomposition workflow, or point OPENEVOLVE_API_URL at a backend that has one.';
    console.warn(`${LOG_PREFIX} skipping plan tests: ${planSkipReason}`);
  }
});

afterAll(async () => {
  if (!backendLive) {
    return;
  }

  // Restore the sub-problem description if the PUT test mutated a pre-existing plan.
  if (planWorkflowId && !createdWorkflowId && mutatedSubProblemId && originalSubProblemDescription) {
    try {
      await api(planPath(planWorkflowId), {
        method: 'PUT',
        body: JSON.stringify({
          sub_problems: [
            { id: mutatedSubProblemId, description: originalSubProblemDescription },
          ],
        }),
      });
    } catch (error) {
      console.warn(
        `${LOG_PREFIX} failed to restore sub-problem ${mutatedSubProblemId}: ` +
          `${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  // Best-effort teardown of anything this suite created.
  const cleanupPaths: string[] = [];
  if (createdWorkflowId) {
    cleanupPaths.push(`/workflows/${encodeURIComponent(createdWorkflowId)}`);
  }
  if (createdSupportFixtures) {
    cleanupPaths.push(`/gauntlets/${encodeURIComponent(SUPPORT_GAUNTLET_NAME)}`);
    cleanupPaths.push(`/teams/${encodeURIComponent(SUPPORT_TEAM_NAME)}`);
  }
  for (const path of cleanupPaths) {
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

/** Additional guard for tests that need a workflow with an existing plan. */
const livePlan = (ctx: { skip: () => void }): boolean => {
  if (!liveBackend(ctx)) return false;
  if (!planWorkflowId) {
    console.warn(`${LOG_PREFIX} ${planSkipReason}`);
    ctx.skip();
    return false;
  }
  return true;
};

describe('Workflow Decomposition Plan API Contract', () => {
  describe('GET /workflows/{id}/decomposition-plan', () => {
    it('should return the plan with a sub_problems array and dependency graph', async (ctx) => {
      if (!livePlan(ctx)) return;

      const response = await openevolveApi.getWorkflowPlan(planWorkflowId as string, TEST_CONFIG);
      await pace();

      expect(response).toBeDefined();
      expect(response.workflow_id).toBe(planWorkflowId);
      expect(response.plan).toBeDefined();
      expect(typeof response.plan.problem_statement).toBe('string');
      expect(Array.isArray(response.plan.sub_problems)).toBe(true);
      expect(response.plan.sub_problems.length).toBeGreaterThan(0);

      const subProblem: WorkflowSubProblem = response.plan.sub_problems[0];
      expect(typeof subProblem.id).toBe('string');
      expect(subProblem.id.length).toBeGreaterThan(0);
      expect(typeof subProblem.description).toBe('string');
      expect(Array.isArray(subProblem.dependencies)).toBe(true);

      // Every declared dependency must reference a sub-problem in the same plan.
      const ids = response.plan.sub_problems.map((entry) => entry.id);
      for (const entry of response.plan.sub_problems) {
        for (const dependency of entry.dependencies ?? []) {
          expect(ids).toContain(dependency);
        }
      }

      expect(response.dependency_graph).toBeDefined();
      expect(typeof response.dependency_graph.edges).toBe('object');
      expect(Array.isArray(response.dependency_graph.edges)).toBe(false);
      if (response.dependency_graph.execution_order !== undefined) {
        expect(Array.isArray(response.dependency_graph.execution_order)).toBe(true);
      }

      // Remember what the PUT test will mutate so afterAll can restore it.
      mutatedSubProblemId = subProblem.id;
      originalSubProblemDescription = subProblem.description;
    });

    it('should answer 404 for a workflow that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      // Raw fetch: the SDK client would retry this expected failure and feed its
      // shared circuit breaker.
      const response = await api(planPath(MISSING_WORKFLOW_ID));

      expect(response.status).toBe(404);
      const body = await readBody(response);
      expect(errorMessage(body), `404 body must explain the error: ${JSON.stringify(body)}`).toBeTruthy();
    });
  });

  describe('PUT /workflows/{id}/decomposition-plan', () => {
    it('should update a sub-problem and expose it on re-fetch', async (ctx) => {
      if (!livePlan(ctx)) return;

      const before = await openevolveApi.getWorkflowPlan(planWorkflowId as string, TEST_CONFIG);
      await pace();
      expect(before.plan.sub_problems.length).toBeGreaterThan(0);

      const target = before.plan.sub_problems[0];
      mutatedSubProblemId = target.id;
      if (originalSubProblemDescription === undefined) {
        originalSubProblemDescription = target.description;
      }

      const updatedDescription = `${target.description} [contract-test ${uniqueSuffix}]`;
      const payload: WorkflowPlanUpdateRequest = {
        sub_problems: [{ ...target, description: updatedDescription }],
      };

      const updated = await openevolveApi.updateWorkflowPlan(
        planWorkflowId as string,
        payload,
        TEST_CONFIG,
      );
      await pace();

      expect(updated).toBeDefined();
      expect(typeof updated.message).toBe('string');
      expect(Array.isArray(updated.execution_order)).toBe(true);
      // The returned topological order must cover every sub-problem in the plan.
      expect(updated.execution_order.length).toBe(before.plan.sub_problems.length);
      expect(updated.execution_order).toContain(target.id);

      const after = await openevolveApi.getWorkflowPlan(planWorkflowId as string, TEST_CONFIG);
      await pace();
      const reread = after.plan.sub_problems.find((entry) => entry.id === target.id);
      expect(reread, `sub-problem ${target.id} must survive the update`).toBeDefined();
      expect(reread?.description).toBe(updatedDescription);
      // Unlisted sub-problems must be left alone.
      expect(after.plan.sub_problems.length).toBe(before.plan.sub_problems.length);
    });

    it('should answer 404 when updating the plan of a workflow that does not exist', async (ctx) => {
      if (!liveBackend(ctx)) return;

      const response = await api(planPath(MISSING_WORKFLOW_ID), {
        method: 'PUT',
        body: JSON.stringify({
          sub_problems: [{ id: 'contract-test-sub-problem', description: 'should not persist' }],
        }),
      });

      if (AUTH_STATUSES.includes(response.status)) {
        console.warn(
          `${LOG_PREFIX} PUT /workflows/{id}/decomposition-plan returned HTTP ${response.status}; ` +
            'the configured API key lacks the required role. Skipping.',
        );
        ctx.skip();
        return;
      }

      expect(response.status).toBe(404);
    });
  });
});
