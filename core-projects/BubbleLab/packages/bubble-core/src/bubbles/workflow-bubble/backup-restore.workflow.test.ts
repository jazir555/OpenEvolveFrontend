import { describe, it, expect, beforeEach, vi } from 'vitest';
import { BackupRestoreWorkflow } from './backup-restore.workflow';

describe('backup-restore.workflow', () => {
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

  const validListParams = {
    operation: 'list' as const,
    database: {
      type: 'postgresql' as const,
      connectionString: 'postgres://localhost:5432/test',
      databaseName: 'test',
    },
    storage: { backend: 'local' as const, config: {} },
  };

  describe('Construction', () => {
    it('should construct without throwing for valid params', () => {
      const instance = new BackupRestoreWorkflow(validListParams, mockContext);
      expect(instance).toBeDefined();
    });

    it('should not throw on invalid params (validation captured, not thrown)', () => {
      const instance = new BackupRestoreWorkflow({} as any, mockContext);
      expect(instance).toBeDefined();
    });
  });

  describe('Input validation (constructor does not throw)', () => {
    it('should return a controlled error from action() for missing required fields', async () => {
      const instance = new BackupRestoreWorkflow({} as any, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });
  });

  describe('Workflow execution', () => {
    it('should list backups (local storage) without network', async () => {
      const instance = new BackupRestoreWorkflow(validListParams, mockContext);
      const result = await instance.action();
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(Array.isArray(result.data.backups)).toBe(true);
    });
  });
});
