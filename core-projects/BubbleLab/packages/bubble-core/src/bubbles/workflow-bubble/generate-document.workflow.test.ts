import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GenerateDocumentWorkflow } from './generate-document.workflow';

describe('generate-document.workflow', () => {
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
    documents: [
      {
        content: '# Expense Report\n| Vendor | Amount |\n| Office | 10 |\n',
        index: 0,
      },
    ],
    outputDescription: 'Extract vendor and amount as a table',
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new GenerateDocumentWorkflow(validParams, mockContext);
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new GenerateDocumentWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for empty documents', async () => {
      const instance = new GenerateDocumentWorkflow(
        { documents: [], outputDescription: 'short' } as any,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });

    it('should return a controlled error from action() for short output description', async () => {
      const instance = new GenerateDocumentWorkflow(
        { documents: [{ content: 'x', index: 0 }], outputDescription: 'short' } as any,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it.skipIf(!process.env.GOOGLE_GEMINI_CRED && !process.env.OPENAI_CRED)(
      'should generate structured document output via AI',
      async () => {
        const instance = new GenerateDocumentWorkflow(validParams, mockContext);
        const result = await instance.action();
        expect(result.success).toBe(true);
        expect(result.data).toBeDefined();
        expect(Array.isArray(result.data.columns)).toBe(true);
      }
    );
  });
});
