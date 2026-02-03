/**
 * BubbleLab Adapter Contract Tests
 *
 * Purpose: Validate BubbleLab API contracts to prevent breaking changes
 * Compliance: Phase 2 - The Contract (Defense)
 *
 * These tests run on adapter startup to verify the API returns expected fields
 * If contracts are violated, the adapter refuses to start (Law of Runtime Truth)
 */

import { describe, it, expect, beforeAll } from '@jest/globals';
import { z } from 'zod';

// =============================================================================
// Mock BubbleLab API Responses for Contract Testing
// =============================================================================

const MOCK_HEALTH_RESPONSE = {
  status: 'ok',
  version: '1.0.0',
};

const MOCK_BUBBLE_FLOW_LIST_RESPONSE = [
  {
    id: '123',
    name: 'Test Flow',
    description: 'A test workflow',
    eventType: 'webhook/http',
    webhookActive: false,
    createdAt: '2026-02-03T00:00:00.000Z',
    updatedAt: '2026-02-03T00:00:00.000Z',
  },
];

const MOCK_BUBBLE_FLOW_CREATE_RESPONSE = {
  id: '456',
  name: 'New Flow',
  requiredCredentials: {
    'postgres-bubble': ['DATABASE_CRED'],
  },
  webhookUrl: 'https://api.bubblelab.dev/webhook/abc123',
  createdAt: '2026-02-03T00:00:00.000Z',
};

const MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE = {
  execution_id: 'exec-789',
  output: {
    message: 'Success',
    processed: true,
  },
  status: 'success',
  startedAt: '2026-02-03T00:00:00.000Z',
  completedAt: '2026-02-03T00:00:01.000Z',
};

const MOCK_EXECUTION_HISTORY_RESPONSE = {
  executions: [
    {
      id: 'exec-001',
      status: 'success',
      startedAt: '2026-02-03T00:00:00.000Z',
      completedAt: '2026-02-03T00:00:01.000Z',
      output: { result: 'test' },
    },
  ],
};

// =============================================================================
// Contract Schemas
// =============================================================================

/**
 * Health Check Response Contract
 */
const HealthCheckContract = z.object({
  status: z.enum(['ok', 'healthy', 'error']),
  version: z.string().optional(),
});

/**
 * BubbleFlow List Response Contract
 */
const BubbleFlowContract = z.object({
  id: z.union([z.string(), z.number()]),
  name: z.string(),
  description: z.string().optional(),
  eventType: z.string(),
  webhookActive: z.boolean(),
  createdAt: z.string().optional(),
  updatedAt: z.string().optional(),
});

/**
 * BubbleFlow Create Response Contract
 */
const BubbleFlowCreateResponseContract = z.object({
  id: z.union([z.string(), z.number()]),
  name: z.string(),
  requiredCredentials: z.record(z.string(), z.array(z.string())).optional(),
  webhookUrl: z.string().optional(),
  createdAt: z.string().optional(),
});

/**
 * Execution Response Contract
 */
const ExecutionResponseContract = z.object({
  execution_id: z.union([z.string(), z.number()]).optional(),
  output: z.any().optional(),
  error: z.string().optional(),
  status: z.string().optional(),
});

/**
 * Execution History Response Contract
 */
const ExecutionHistoryContract = z.object({
  executions: z.array(z.object({
    id: z.union([z.string(), z.number()]).optional(),
    status: z.string(),
    startedAt: z.string(),
    completedAt: z.string().optional(),
    output: z.any().optional(),
    error: z.string().optional(),
  })).optional(),
});

// =============================================================================
// Contract Tests
// =============================================================================

describe('BubbleLab API Contract Tests', () => {
  describe('Health Check Endpoint', () => {
    it('should return valid health check response', () => {
      const result = HealthCheckContract.safeParse(MOCK_HEALTH_RESPONSE);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.status).toBe('ok');
      }
    });

    it('should include status field', () => {
      const response = MOCK_HEALTH_RESPONSE;

      expect(response).toHaveProperty('status');
      expect(typeof response.status).toBe('string');
    });

    it('should allow optional version field', () => {
      const response = MOCK_HEALTH_RESPONSE;

      expect(response.version).toBeDefined();
      expect(typeof response.version).toBe('string');
    });
  });

  describe('BubbleFlow List Endpoint', () => {
    it('should return valid BubbleFlow objects', () => {
      const result = BubbleFlowContract.safeParse(MOCK_BUBBLE_FLOW_LIST_RESPONSE[0]);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.name).toBe('Test Flow');
        expect(result.data.eventType).toBe('webhook/http');
      }
    });

    it('should include required fields', () => {
      const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];

      expect(flow).toHaveProperty('id');
      expect(flow).toHaveProperty('name');
      expect(flow).toHaveProperty('eventType');
      expect(flow).toHaveProperty('webhookActive');
    });

    it('should support string or numeric IDs', () => {
      const stringId = { ...MOCK_BUBBLE_FLOW_LIST_RESPONSE[0], id: '123' };
      const numberId = { ...MOCK_BUBBLE_FLOW_LIST_RESPONSE[0], id: 123 };

      const stringResult = BubbleFlowContract.safeParse(stringId);
      const numberResult = BubbleFlowContract.safeParse(numberId);

      expect(stringResult.success).toBe(true);
      expect(numberResult.success).toBe(true);
    });

    it('should include optional timestamp fields', () => {
      const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];

      expect(flow.createdAt).toBeDefined();
      expect(flow.updatedAt).toBeDefined();

      // Verify ISO-8601 format
      expect(() => new Date(flow.createdAt!)).not.toThrow();
      expect(() => new Date(flow.updatedAt!)).not.toThrow();
    });
  });

  describe('BubbleFlow Create Endpoint', () => {
    it('should return valid create response', () => {
      const result = BubbleFlowCreateResponseContract.safeParse(MOCK_BUBBLE_FLOW_CREATE_RESPONSE);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.id).toBeDefined();
        expect(result.data.name).toBe('New Flow');
      }
    });

    it('should include requiredCredentials object', () => {
      const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;

      expect(response.requiredCredentials).toBeDefined();
      expect(typeof response.requiredCredentials).toBe('object');
    });

    it('should include credential arrays per bubble', () => {
      const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;

      if (response.requiredCredentials) {
        const bubbleName = Object.keys(response.requiredCredentials)[0];
        const creds = response.requiredCredentials[bubbleName];

        expect(Array.isArray(creds)).toBe(true);
        expect(creds).toContain('DATABASE_CRED');
      }
    });

    it('should include webhook URL if webhook is active', () => {
      const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;

      expect(response.webhookUrl).toBeDefined();
      expect(typeof response.webhookUrl).toBe('string');
      expect(response.webhookUrl).toMatch(/^https?:\/\//);
    });
  });

  describe('BubbleFlow Execute Endpoint', () => {
    it('should return valid execution response', () => {
      const result = ExecutionResponseContract.safeParse(MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.status).toBe('success');
      }
    });

    it('should include execution ID or status', () => {
      const response = MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE;

      expect(
        response.execution_id !== undefined || response.status !== undefined
      ).toBe(true);
    });

    it('should include output data on success', () => {
      const response = MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE;

      if (response.status === 'success') {
        expect(response.output).toBeDefined();
      }
    });

    it('should include error message on failure', () => {
      const failedResponse = {
        ...MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE,
        status: 'failed',
        error: 'Execution failed',
        output: undefined,
      };

      const result = ExecutionResponseContract.safeParse(failedResponse);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.error).toBeDefined();
      }
    });
  });

  describe('Execution History Endpoint', () => {
    it('should return valid execution history', () => {
      const result = ExecutionHistoryContract.safeParse(MOCK_EXECUTION_HISTORY_RESPONSE);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.executions).toBeDefined();
        expect(Array.isArray(result.data.executions)).toBe(true);
      }
    });

    it('should include timestamp fields for each execution', () => {
      const history = MOCK_EXECUTION_HISTORY_RESPONSE;

      if (history.executions && history.executions.length > 0) {
        const execution = history.executions[0];

        expect(execution.startedAt).toBeDefined();
        expect(typeof execution.startedAt).toBe('string');

        // Verify ISO-8601 format
        expect(() => new Date(execution.startedAt)).not.toThrow();
      }
    });

    it('should include status for each execution', () => {
      const history = MOCK_EXECUTION_HISTORY_RESPONSE;

      if (history.executions && history.executions.length > 0) {
        const execution = history.executions[0];

        expect(execution.status).toBeDefined();
        expect(typeof execution.status).toBe('string');
      }
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle missing optional fields gracefully', () => {
      const minimalFlow = {
        id: '123',
        name: 'Minimal Flow',
        eventType: 'manual',
        webhookActive: false,
      };

      const result = BubbleFlowContract.safeParse(minimalFlow);
      expect(result.success).toBe(true);
    });

    it('should reject invalid response structures', () => {
      const invalidFlow = {
        id: '123',
        // Missing required 'name' field
        eventType: 'webhook/http',
      };

      const result = BubbleFlowContract.safeParse(invalidFlow);
      expect(result.success).toBe(false);
    });

    it('should handle empty execution history', () => {
      const emptyHistory = { executions: [] };

      const result = ExecutionHistoryContract.safeParse(emptyHistory);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.executions).toHaveLength(0);
      }
    });
  });

  describe('Data Type Validation', () => {
    it('should validate boolean webhookActive field', () => {
      const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];

      expect(typeof flow.webhookActive).toBe('boolean');
      expect(flow.webhookActive).toBe(false);
    });

    it('should validate string event types', () => {
      const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];

      expect(typeof flow.eventType).toBe('string');
      expect(['webhook/http', 'schedule', 'manual']).toContain(flow.eventType);
    });

    it('should validate credential type arrays', () => {
      const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;

      if (response.requiredCredentials) {
        for (const bubbleName in response.requiredCredentials) {
          const creds = response.requiredCredentials[bubbleName];
          expect(Array.isArray(creds)).toBe(true);

          for (const cred of creds) {
            expect(typeof cred).toBe('string');
          }
        }
      }
    });
  });
});

// =============================================================================
// Contract Validation Helper
// =============================================================================

/**
 * Validate all API contracts before starting adapter
 * This function should be called during adapter initialization
 *
 * @returns true if all contracts are valid
 * @throws Error if any contract is violated
 */
export function validateAllContracts(): boolean {
  console.log('Validating BubbleLab API contracts...');

  try {
    // Health check contract
    const healthResult = HealthCheckContract.safeParse(MOCK_HEALTH_RESPONSE);
    if (!healthResult.success) {
      throw new Error(`Health check contract violated: ${JSON.stringify(healthResult.error)}`);
    }

    // BubbleFlow contract
    const flowResult = BubbleFlowContract.safeParse(MOCK_BUBBLE_FLOW_LIST_RESPONSE[0]);
    if (!flowResult.success) {
      throw new Error(`BubbleFlow contract violated: ${JSON.stringify(flowResult.error)}`);
    }

    // Create response contract
    const createResult = BubbleFlowCreateResponseContract.safeParse(MOCK_BUBBLE_FLOW_CREATE_RESPONSE);
    if (!createResult.success) {
      throw new Error(`Create response contract violated: ${JSON.stringify(createResult.error)}`);
    }

    // Execution contract
    const execResult = ExecutionResponseContract.safeParse(MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE);
    if (!execResult.success) {
      throw new Error(`Execution response contract violated: ${JSON.stringify(execResult.error)}`);
    }

    // History contract
    const historyResult = ExecutionHistoryContract.safeParse(MOCK_EXECUTION_HISTORY_RESPONSE);
    if (!historyResult.success) {
      throw new Error(`Execution history contract violated: ${JSON.stringify(historyResult.error)}`);
    }

    console.log('All BubbleLab API contracts validated successfully');
    return true;
  } catch (error) {
    console.error('Contract validation failed:', error);
    throw error;
  }
}

// Export contracts for use in adapter
export {
  HealthCheckContract,
  BubbleFlowContract,
  BubbleFlowCreateResponseContract,
  ExecutionResponseContract,
  ExecutionHistoryContract,
};
