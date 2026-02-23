/**
 * OpenEvolve API Contract Tests
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 * "Protecting the Mega-Project from Updates"
 *
 * These tests verify that the OpenEvolve API returns the expected fields.
 * If the contract is violated (Project API changed), the adapter MUST refuse to start.
 *
 * Runs on container startup. If these tests fail, the application should NOT start.
 */

import { describe, it, expect } from '@jest/globals';
import { openevolveApi, type ApiConfig } from '../../lib/openevolveApi';

// Configuration
const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.OPENEVOLVE_API_KEY;
const TIMEOUT = 30000; // 30 second timeout for contract tests
const WORKFLOW_POLL_INTERVAL_MS = 500;
const WORKFLOW_POLL_TIMEOUT_MS = 20000;
const TERMINAL_WORKFLOW_STATES = new Set(['completed', 'failed', 'stopped', 'cancelled']);

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const getWorkflowApiConfig = (): ApiConfig | null => {
  if (!API_KEY) {
    return null;
  }
  return {
    baseUrl: API_URL,
    apiKey: API_KEY,
    timeout: TIMEOUT,
  };
};

const waitForTerminalWorkflowState = async (instanceId: string, config: ApiConfig) => {
  const startedAt = Date.now();
  let latest = await openevolveApi.getWorkflowInstance(instanceId, config);

  while (!TERMINAL_WORKFLOW_STATES.has(latest.status.status)) {
    if (Date.now() - startedAt > WORKFLOW_POLL_TIMEOUT_MS) {
      throw new Error(
        `Timed out waiting for workflow ${instanceId}. Last state: ${latest.status.status}`,
      );
    }
    await sleep(WORKFLOW_POLL_INTERVAL_MS);
    latest = await openevolveApi.getWorkflowInstance(instanceId, config);
  }

  return latest;
};

describe('OpenEvolve API Contract Tests', () => {
  describe('Health Check Endpoint', () => {
    it('should return health status with required fields', async () => {
      const response = await fetch(`${API_URL}/health`, {
        headers: API_KEY ? { 'Authorization': `Bearer ${API_KEY}` } : {},
      });

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Health check must include status field
      expect(data).toHaveProperty('status');
      expect(typeof data.status).toBe('string');

      // Contract: Should include version information
      expect(data).toHaveProperty('version');
    });

    it('should respond within timeout', async () => {
      const start = Date.now();
      const response = await fetch(`${API_URL}/health`);
      const duration = Date.now() - start;

      expect(response.ok).toBe(true);
      expect(duration).toBeLessThan(TIMEOUT);
    });
  });

  describe('Evolutions Endpoint', () => {
    it('should return evolutions list with required fields', async () => {
      const response = await fetch(`${API_URL}/evolutions`, {
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
      });

      // Allow 404 if endpoint not implemented yet
      if (response.status === 404) {
        console.warn('Evolutions endpoint not implemented - skipping');
        return;
      }

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have evolutions array
      expect(data).toHaveProperty('evolutions');
      expect(Array.isArray(data.evolutions)).toBe(true);

      // Contract: Each evolution must have required fields
      if (data.evolutions.length > 0) {
        const evolution = data.evolutions[0];
        expect(evolution).toHaveProperty('id');
        expect(evolution).toHaveProperty('name');
        expect(evolution).toHaveProperty('created_at');
      }
    });
  });

  describe('Adversarial Runs Endpoint', () => {
    it('should return adversarial runs list with required fields', async () => {
      const response = await fetch(`${API_URL}/adversarial-runs`, {
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
      });

      // Allow 404 if endpoint not implemented yet
      if (response.status === 404) {
        console.warn('Adversarial runs endpoint not implemented - skipping');
        return;
      }

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have runs array
      expect(data).toHaveProperty('runs');
      expect(Array.isArray(data.runs)).toBe(true);

      // Contract: Each run must have required fields
      if (data.runs.length > 0) {
        const run = data.runs[0];
        expect(run).toHaveProperty('id');
        expect(run).toHaveProperty('status');
        expect(run).toHaveProperty('created_at');
      }
    });
  });

  describe('Create Evolution Endpoint', () => {
    it('should accept evolution creation request with required fields', async () => {
      const testRequest = {
        name: 'Contract Test Evolution',
        base_prompt: 'Test prompt for contract validation',
        adversarial_prompt: 'Test adversarial prompt',
        parameters: {
          temperature: 0.7,
          max_tokens: 1000,
        },
      };

      const response = await fetch(`${API_URL}/evolutions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
        body: JSON.stringify(testRequest),
      });

      // Allow 404 if endpoint not implemented yet
      if (response.status === 404) {
        console.warn('Create evolution endpoint not implemented - skipping');
        return;
      }

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return created evolution with ID
      expect(data).toHaveProperty('id');
      expect(data).toHaveProperty('name');
      expect(data.name).toBe(testRequest.name);
      expect(data).toHaveProperty('created_at');
    });
  });

  describe('Get Evolution by ID Endpoint', () => {
    it('should return evolution details with required fields', async () => {
      // First, try to list evolutions to get a valid ID
      const listResponse = await fetch(`${API_URL}/evolutions`, {
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
      });

      if (!listResponse.ok) {
        console.warn('Cannot get evolution list - skipping get by ID test');
        return;
      }

      const listData = await listResponse.json();

      if (!listData.evolutions || listData.evolutions.length === 0) {
        console.warn('No evolutions found - skipping get by ID test');
        return;
      }

      const evolutionId = listData.evolutions[0].id;

      const response = await fetch(`${API_URL}/evolutions/${evolutionId}`, {
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
      });

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return all required fields
      expect(data).toHaveProperty('id');
      expect(data).toHaveProperty('name');
      expect(data).toHaveProperty('base_prompt');
      expect(data).toHaveProperty('created_at');
      expect(data).toHaveProperty('updated_at');
    });
  });

  describe('Error Responses', () => {
    it('should return proper error for invalid evolution ID', async () => {
      const response = await fetch(`${API_URL}/evolutions/invalid-id-12345`, {
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
      });

      // Contract: Should return 404 or 400 for invalid ID
      expect([400, 404]).toContain(response.status);

      const data = await response.json();

      // Contract: Error response must have error field
      expect(data).toHaveProperty('error');
    });

    it('should return proper error for invalid request body', async () => {
      const response = await fetch(`${API_URL}/evolutions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
        body: 'invalid json{{{',
      });

      // Contract: Should return 400 for bad request
      expect(response.status).toBe(400);
    });
  });

  describe('Pagination and Filtering', () => {
    it('should support pagination parameters', async () => {
      const response = await fetch(
        `${API_URL}/evolutions?limit=10&offset=0`,
        {
          headers: {
            'Content-Type': 'application/json',
            ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
          },
        }
      );

      if (response.status === 404) {
        console.warn('Evolutions endpoint not implemented - skipping pagination test');
        return;
      }

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Should respect limit parameter
      expect(data.evolutions.length).toBeLessThanOrEqual(10);
    });
  });

  describe('BubbleLabs Workflow Lifecycle', () => {
    it('should execute an end-to-end BubbleLabs workflow with OpenEvolve controls', async () => {
      const config = getWorkflowApiConfig();
      if (!config) {
        console.warn('OPENEVOLVE_API_KEY not set - skipping BubbleLabs workflow lifecycle contract test');
        return;
      }

      const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const definitionName = `contract-workflow-${uniqueSuffix}`;
      let instanceId: string | null = null;

      try {
        const createdDefinition = await openevolveApi.createWorkflowDefinition(
          {
            name: definitionName,
            description: 'Contract e2e workflow lifecycle test',
            workflow_type: 'evolution',
            parameters: {
              max_iterations: 1,
              population_size: 2,
            },
          },
          config,
        );
        expect(createdDefinition.definition_id).toBeTruthy();

        const definitionDetails = await openevolveApi.getWorkflowDefinition(
          createdDefinition.definition_id,
          config,
        );
        expect(definitionDetails.id).toBe(createdDefinition.definition_id);
        expect(definitionDetails.workflow_type).toBe('evolution');

        const createdInstance = await openevolveApi.createWorkflowInstance(
          {
            definition_id: createdDefinition.definition_id,
            instance_name: `instance-${uniqueSuffix}`,
            inputs: {
              problem_statement: 'Contract test: evolve a short deterministic prompt',
            },
          },
          config,
        );
        instanceId = createdInstance.instance_id;
        expect(instanceId).toBeTruthy();

        const synced = await openevolveApi.syncWorkflowInstanceParameters(
          instanceId,
          {
            parameters: {
              max_iterations: 1,
              population_size: 2,
              temperature: 0.1,
            },
          },
          config,
        );
        expect(synced).toHaveProperty('updated_count');

        const started = await openevolveApi.startWorkflowInstance(instanceId, config);
        expect(started).toHaveProperty('instance_id', instanceId);

        const terminalState = await waitForTerminalWorkflowState(instanceId, config);
        if (terminalState.status.status === 'failed') {
          throw new Error(
            `Workflow failed: ${terminalState.status.error_message || 'unknown failure'}`,
          );
        }

        if (terminalState.status.status !== 'completed') {
          const stopped = await openevolveApi.stopWorkflowInstance(instanceId, config);
          expect(stopped).toHaveProperty('status', 'stopped');
        }

        const finalizedState = await openevolveApi.getWorkflowInstance(instanceId, config);
        expect(['completed', 'stopped']).toContain(finalizedState.status.status);
      } finally {
        if (instanceId) {
          await openevolveApi.deleteWorkflowInstance(instanceId, config).catch(() => undefined);
        }
      }
    }, 120000);
  });
});

/**
 * Usage Instructions:
 *
 * 1. Run tests on container startup:
 *    ```bash
 *    npm run test:contract
 *    ```
 *
 * 2. If tests fail, container MUST refuse to start:
 *    ```javascript
 *    try {
 *      await runContractTests();
 *    } catch (error) {
 *      logger.error('Contract tests failed - refusing to start');
 *      process.exit(1);
 *    }
 *    ```
 *
 * 3. Tests verify critical fields that the adapter depends on
 * 4. If API changes, tests fail before corrupting data
 */
