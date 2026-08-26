import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MonitoringAlertWorkflow } from './monitoring-alert.workflow';

describe('monitoring-alert.workflow', () => {
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
    metrics: [
      {
        name: 'cpu',
        type: 'gauge' as const,
        value: 50,
        threshold: { warning: 80, error: 90, critical: 95 },
      },
    ],
    alertConfig: {
      severity: 'info' as const,
      message: 'CPU nominal',
      source: 'monitoring',
    },
    notifications: {},
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new MonitoringAlertWorkflow(validParams, mockContext);
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new MonitoringAlertWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing required fields', async () => {
      const instance = new MonitoringAlertWorkflow({} as any, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it('should resolve when no thresholds are breached (no network)', async () => {
      const instance = new MonitoringAlertWorkflow(validParams, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data.alert?.status).toBe('resolved');
    });

    it('should activate alert when a threshold is breached (no notifications)', async () => {
      const breached = {
        ...validParams,
        metrics: [
          {
            name: 'cpu',
            type: 'gauge' as const,
            value: 99,
            threshold: { warning: 80, error: 90, critical: 95 },
          },
        ],
      };
      const instance = new MonitoringAlertWorkflow(breached, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data.alert?.status).toBe('active');
    });
  });
});
