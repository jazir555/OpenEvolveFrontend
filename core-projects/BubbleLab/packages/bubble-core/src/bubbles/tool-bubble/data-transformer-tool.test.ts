/*
 * Test Suite for data-transformer-tool
 * Covers the real DataTransformerTool API (constructor + action()).
 */

import { describe, it, expect } from 'vitest';
import { DataTransformerTool } from './data-transformer-tool';

describe('data-transformer-tool', () => {
  describe('Construction', () => {
    it('should construct with valid params (no throw)', () => {
      const tool = new DataTransformerTool({
        inputData: [{ a: 1 }],
        operation: 'map',
      });
      expect(tool).toBeDefined();
      expect(tool.name).toBe('data-transformer-tool');
    });

    it('should not throw on invalid params (captures validation error)', () => {
      const tool = new DataTransformerTool({});
      expect(tool).toBeDefined();
    });
  });

  describe('action()', () => {
    it('should apply map (copy) transformation', async () => {
      const tool = new DataTransformerTool({
        inputData: [{ name: 'bob' }],
        operation: 'map',
        mapOperations: [
          { targetField: 'nameCopy', sourceField: 'name', transform: 'copy' },
        ],
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.outputData[0].nameCopy).toBe('bob');
      expect(result.data?.fieldsAdded).toContain('nameCopy');
    });

    it('should apply filter transformation', async () => {
      const tool = new DataTransformerTool({
        inputData: [{ a: 1 }, { a: 2 }, { a: 3 }],
        operation: 'filter',
        filterConditions: [{ field: 'a', operator: 'gt', value: 1 }],
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.outputData).toHaveLength(2);
      expect(result.data?.outputData[0].a).toBe(2);
    });

    it('should apply sort transformation', async () => {
      const tool = new DataTransformerTool({
        inputData: [{ a: 3 }, { a: 1 }, { a: 2 }],
        operation: 'sort',
        sortFields: [{ field: 'a', order: 'asc' }],
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.outputData.map((r) => r.a)).toEqual([1, 2, 3]);
    });

    it('should return a controlled error for groupBy missing aggregations', async () => {
      const tool = new DataTransformerTool({
        inputData: [{ a: 1 }],
        operation: 'groupBy',
        groupByFields: ['a'],
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('aggregations are required');
    });

    it('should return a controlled error for invalid params', async () => {
      const tool = new DataTransformerTool({});
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });
  });
});
