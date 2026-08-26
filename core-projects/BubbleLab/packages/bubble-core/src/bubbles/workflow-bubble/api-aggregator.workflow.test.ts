import { describe, it, expect, beforeEach, vi } from 'vitest';
import { APIAggregatorWorkflow } from './api-aggregator.workflow';

describe('api-aggregator.workflow', () => {
  const mockContext: any = {
    logger: {
      info: vi.fn(),
      error: vi.fn(),
      warn: vi.fn(),
      debug: vi.fn(),
      logBubbleExecution: vi.fn(),
      logBubbleExecutionComplete: vi.fn(),
    },
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new APIAggregatorWorkflow(
        { apis: [{ name: 'test', url: 'https://example.com/api' }] },
        mockContext
      );
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new APIAggregatorWorkflow(
        { apis: 'not-an-array' } as any,
        mockContext
      );
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing required fields', async () => {
      const instance = new APIAggregatorWorkflow({} as any, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });

    it('should return a controlled error from action() for invalid apis type', async () => {
      const instance = new APIAggregatorWorkflow(
        { apis: 'nope' } as any,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it.skipIf(!process.env.API_AGGREGATOR_TEST_URL)(
      'should aggregate a reachable API and succeed',
      async () => {
        const instance = new APIAggregatorWorkflow(
          {
            apis: [
              {
                name: 'test',
                url: process.env.API_AGGREGATOR_TEST_URL as string,
              },
            ],
          },
          mockContext
        );
        const result = await instance.action();
        expect(result.success).toBe(true);
        expect(result.data).toBeDefined();
        expect(Array.isArray(result.data.results)).toBe(true);
      }
    );
  });
});
