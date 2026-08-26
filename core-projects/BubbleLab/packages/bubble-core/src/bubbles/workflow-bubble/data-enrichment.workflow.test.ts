import { describe, it, expect, beforeEach, vi } from 'vitest';
import { DataEnrichmentWorkflow } from './data-enrichment.workflow';

describe('data-enrichment.workflow', () => {
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

  const disabledSources = {
    webSearch: false,
    vectorSearch: false,
    aiAnalysis: false,
    databaseLookup: false,
  };

  const validParams = {
    record: { id: 1, name: 'Acme Corp' },
    sources: disabledSources,
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new DataEnrichmentWorkflow(validParams, mockContext);
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new DataEnrichmentWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing record', async () => {
      const instance = new DataEnrichmentWorkflow(
        { sources: disabledSources } as any,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it('should complete without external sources enabled (no network)', async () => {
      const instance = new DataEnrichmentWorkflow(validParams, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data.enrichedRecord).toBeDefined();
      expect(result.data.metadata).toBeDefined();
    });
  });
});
