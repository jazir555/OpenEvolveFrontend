import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ETLPipelineWorkflow } from './etl-pipeline.workflow';

describe('etl-pipeline.workflow', () => {
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

  const validTransformParams = {
    phase: 'transform' as const,
    source: { type: 'file' as const, config: {} },
    destination: { type: 'file' as const, config: {} },
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new ETLPipelineWorkflow(
        validTransformParams,
        mockContext
      );
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new ETLPipelineWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing required fields', async () => {
      const instance = new ETLPipelineWorkflow({} as any, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it('should run the transform phase without network', async () => {
      const instance = new ETLPipelineWorkflow(
        validTransformParams,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data.phase).toBe('transform');
    });
  });
});
