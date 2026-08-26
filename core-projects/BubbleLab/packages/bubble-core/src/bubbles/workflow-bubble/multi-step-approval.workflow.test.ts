import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MultiStepApprovalWorkflow } from './multi-step-approval.workflow';

describe('multi-step-approval.workflow', () => {
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

  const validSubmitParams = {
    action: 'submit' as const,
    title: 'Test Approval',
    requester: 'user1',
    approvalSteps: [
      {
        stepName: 'step1',
        approvers: [{ userId: 'a', name: 'Alice' }],
      },
    ],
    notifyOnComplete: false,
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new MultiStepApprovalWorkflow(
        validSubmitParams,
        mockContext
      );
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new MultiStepApprovalWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing required fields', async () => {
      const instance = new MultiStepApprovalWorkflow({} as any, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it('should submit an approval workflow without network', async () => {
      const instance = new MultiStepApprovalWorkflow(
        validSubmitParams,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(typeof result.data.workflowId).toBe('string');
      expect(result.data.workflowId.length).toBeGreaterThan(0);
      expect(result.data.status).toBe('pending');
    });
  });
});
