import { describe, it, expect, beforeAll } from 'vitest';
import { openevolveApi, ApiConfig } from '../../lib/openevolveApi';

const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.OPENEVOLVE_API_KEY || 'test-key';
const PROBE_TIMEOUT_MS = Number(process.env.OPENEVOLVE_CONTRACT_PROBE_TIMEOUT_MS || 5000);
const LOG_PREFIX = '[execution-api contract]';
const AUTH_STATUSES = [401, 403, 422];

let backendLive = false;
let skipReason = 'backend not probed';

async function timedFetch(path: string, init: RequestInit = {}, timeoutMs = PROBE_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

beforeAll(async () => {
   try {
    const probe = await timedFetch('/health', {}, PROBE_TIMEOUT_MS);
    if (AUTH_STATUSES.includes(probe.status)) {
      skipReason = `backend at ${API_URL} rejected the API key on GET /health (HTTP ${probe.status}). Set OPENEVOLVE_API_KEY.`;
    } else {
      backendLive = true;
    }
  } catch (error) {
    skipReason = `backend unreachable at ${API_URL} (${error instanceof Error ? error.message : String(error)}). Start the API server or set OPENEVOLVE_API_URL.`;
  }
  if (!backendLive) {
    console.warn(`${LOG_PREFIX} skipping suite: ${skipReason}`);
  }
});

const liveBackend = (ctx: { skip: () => void }): boolean => {
  if (!backendLive) {
    console.warn(`${LOG_PREFIX} ${skipReason}`);
    ctx.skip();
    return false;
  }
  return true;
};

const TEST_CONFIG: ApiConfig = {
  baseUrl: process.env.OPENEVOLVE_API_BASE_URL || 'http://localhost:8000',
  apiKey: process.env.OPENEVOLVE_API_KEY || 'test-key',
  timeout: 30000,
};

describe('Execution API Contract', () => {
  let createdExecutionId: string;

  describe('POST /executions', () => {
    it('should create an execution and return record with required fields', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.createExecution(
        { name: 'contract-test-exec', workflow_id: 'test-workflow' },
        TEST_CONFIG,
      );
      expect(response).toBeDefined();
      expect(typeof response.id).toBe('string');
      expect(response.id.startsWith('exec-')).toBe(true);
      expect(typeof response.status).toBe('string');
      expect(typeof response.created_at).toBe('string');
      expect(typeof response.real_engine_available).toBe('boolean');
      createdExecutionId = response.id;
    });
  });

  describe('GET /executions', () => {
    it('should list executions with total count', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.listExecutions(undefined, TEST_CONFIG);
      expect(response).toBeDefined();
      expect(Array.isArray(response.executions)).toBe(true);
      expect(typeof response.total).toBe('number');
    });

    it('should respect limit parameter', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.listExecutions({ limit: 2 }, TEST_CONFIG);
      expect(response.executions.length).toBeLessThanOrEqual(2);
    });
  });

  describe('GET /executions/{id}', () => {
    it('should retrieve the created execution by id', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.getExecution(createdExecutionId, TEST_CONFIG);
      expect(response).toBeDefined();
      expect(response.id).toBe(createdExecutionId);
      expect(typeof response.status).toBe('string');
    });

    it('should throw for non-existent execution', async (ctx) => {
      if (!liveBackend(ctx)) return;
      await expect(
        openevolveApi.getExecution('exec-nonexistent-id', TEST_CONFIG)
      ).rejects.toThrow();
    });
  });

  describe('POST /executions/{id}/pause', () => {
    it('should pause a running execution', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.pauseExecution(createdExecutionId, TEST_CONFIG);
      expect(response).toBeDefined();
      expect(response.id).toBe(createdExecutionId);
      expect(response.status).toBe('paused');
    });
  });

  describe('POST /executions/{id}/resume', () => {
    it('should resume a paused execution', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.resumeExecution(createdExecutionId, TEST_CONFIG);
      expect(response).toBeDefined();
      expect(response.id).toBe(createdExecutionId);
      expect(response.status).toBe('running');
    });
  });

  describe('POST /executions/{id}/cancel', () => {
    it('should cancel a running execution', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.cancelExecution(createdExecutionId, TEST_CONFIG);
      expect(response).toBeDefined();
      expect(response.id).toBe(createdExecutionId);
      expect(response.status).toBe('cancelled');
    });
  });

  describe('GET /executions/{id}/logs', () => {
    it('should return logs array for an execution', async (ctx) => {
      if (!liveBackend(ctx)) return;
      const response = await openevolveApi.getExecutionLogs(createdExecutionId, undefined, TEST_CONFIG);
      expect(response).toBeDefined();
      expect(typeof response.execution_id).toBe('string');
      expect(Array.isArray(response.logs)).toBe(true);
    });
  });
});
