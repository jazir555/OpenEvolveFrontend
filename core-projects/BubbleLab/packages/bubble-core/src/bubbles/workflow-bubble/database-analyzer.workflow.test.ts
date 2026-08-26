import { describe, it, expect, beforeEach, vi } from 'vitest';
import { DatabaseAnalyzerWorkflowBubble } from './database-analyzer.workflow';

describe('database-analyzer.workflow', () => {
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
    dataSourceType: 'postgresql' as const,
    includeMetadata: true,
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new DatabaseAnalyzerWorkflowBubble(
        validParams,
        mockContext
      );
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new DatabaseAnalyzerWorkflowBubble(
        { dataSourceType: 'mysql' } as any,
        mockContext
      );
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for unsupported data source', async () => {
      const instance = new DatabaseAnalyzerWorkflowBubble(
        { dataSourceType: 'mysql' } as any,
        mockContext
      );
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it.skipIf(!process.env.DATABASE_CONNECTION_STRING)(
      'should analyze a live PostgreSQL database',
      async () => {
        const instance = new DatabaseAnalyzerWorkflowBubble(
          {
            dataSourceType: 'postgresql',
            credentials: {
              POSTGRES_CRED:
                process.env.DATABASE_CONNECTION_STRING as string,
            },
          },
          mockContext
        );
        const result = await instance.action();
        expect(result.success).toBe(true);
        expect(result.data.databaseSchema).toBeDefined();
      }
    );
  });
});
