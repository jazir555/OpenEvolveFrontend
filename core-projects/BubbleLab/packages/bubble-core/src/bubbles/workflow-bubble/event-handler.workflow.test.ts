import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EventHandlerWorkflow } from './event-handler.workflow';

describe('event-handler.workflow', () => {
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

  const validParams = {
    eventType: 'custom' as const,
    eventPayload: { type: 'signup' },
    routingRules: [
      {
        condition: "payload.type === 'signup'",
        handler: { type: 'function' as const, config: {} },
      },
    ],
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new EventHandlerWorkflow(validParams, mockContext);
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new EventHandlerWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing required fields', async () => {
      const instance = new EventHandlerWorkflow({} as any, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it('should match and execute a function handler without network', async () => {
      const instance = new EventHandlerWorkflow(validParams, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(Array.isArray(result.data.matchedHandlers)).toBe(true);
      expect(result.data.matchedHandlers.length).toBe(1);
    });
  });
});
