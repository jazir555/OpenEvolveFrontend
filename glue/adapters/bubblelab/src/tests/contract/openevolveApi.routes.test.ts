/**
 * Route-contract test for glue openevolveApi client.
 *
 * Mocks global fetch and asserts that key client methods resolve to the
 * canonical OpenEvolve backend routes (services/openevolve-api, port 8000),
 * which mounts routers ALREADY prefixed under `/api/...` (per the canonical
 * bubble-studio client). This locks in the route surface so the glue copy
 * cannot silently regress to the previously-unprefixed paths.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { openevolveApi, type ApiConfig } from '../../lib/openevolveApi';

const CONFIG: ApiConfig = {
  baseUrl: 'http://localhost:8000',
  apiKey: 'test-key',
  timeout: 5000,
};

type Call = { method: string; url: string };

function installFetchMock(): { calls: Call[] } {
  const calls: Call[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: any, init?: any) => {
      const url = typeof input === 'string' ? input : input?.url ?? String(input);
      calls.push({ method: (init?.method ?? 'GET').toUpperCase(), url });
      return new Response(JSON.stringify({}), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
  return { calls };
}

describe('openevolveApi route surface', () => {
  let calls: Call[];

  beforeEach(() => {
    calls = installFetchMock().calls;
  });

  it('prefixes /api on core resource routes', async () => {
    await openevolveApi.listTeams(CONFIG);
    await openevolveApi.getTeam('t', CONFIG);
    await openevolveApi.listWorkflows(CONFIG);
    await openevolveApi.getWorkflow('w', CONFIG);
    await openevolveApi.listGauntlets(CONFIG);
    await openevolveApi.listEvaluators(CONFIG);

    const urls = calls.map((c) => c.url);
    expect(urls).toContain('http://localhost:8000/api/teams');
    expect(urls).toContain('http://localhost:8000/api/teams/t');
    expect(urls).toContain('http://localhost:8000/api/workflows');
    expect(urls).toContain('http://localhost:8000/api/workflows/w');
    expect(urls).toContain('http://localhost:8000/api/gauntlets');
    expect(urls).toContain('http://localhost:8000/api/evaluators');
  });

  it('prefixes /api on monitoring, crewai, validation, version-control, knowledge, parameters, analytics', async () => {
    await openevolveApi.getMonitoringDashboard(CONFIG);
    await openevolveApi.listCrewaiWorkflows(CONFIG);
    await openevolveApi.listValidationRules(CONFIG);
    await openevolveApi.listVersions(CONFIG);
    await openevolveApi.listKnowledgeArtifacts(CONFIG);
    await openevolveApi.getParameterSchema(CONFIG);
    await openevolveApi.getPerformanceMetrics(undefined, 10, CONFIG);

    const urls = calls.map((c) => c.url);
    expect(urls).toContain('http://localhost:8000/api/monitoring/dashboard');
    expect(urls).toContain('http://localhost:8000/api/crewai/workflows');
    expect(urls).toContain('http://localhost:8000/api/validation/rules');
    expect(urls.some((u) => u.startsWith('http://localhost:8000/api/version-control'))).toBe(true);
    expect(urls).toContain('http://localhost:8000/api/knowledge/artifacts');
    expect(urls).toContain('http://localhost:8000/api/parameters/schema');
    expect(urls.some((u) => u.includes('/api/analytics/performance-metrics'))).toBe(true);
  });

  it('prefixes /api/bubblelabs/leanaide but keeps the control plane unprefixed', async () => {
    await openevolveApi.bubblelabsLeanAideTrees(CONFIG);
    await openevolveApi.getBubblelabsStatus(CONFIG);
    await openevolveApi.bubblelabsControlCatalog(CONFIG);

    const urls = calls.map((c) => c.url);
    expect(urls).toContain('http://localhost:8000/api/bubblelabs/leanaide/trees');
    expect(urls).toContain('http://localhost:8000/bubblelabs/status');
    expect(urls).toContain('http://localhost:8000/bubblelabs/control/catalog');
  });

  it('keeps health unprefixed (matches canonical backend /health)', async () => {
    await openevolveApi.getHealth(CONFIG);
    expect(calls[0].url).toBe('http://localhost:8000/health');
  });
});
