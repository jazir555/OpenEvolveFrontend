/*
 * Test Suite for csv-processor-tool
 * Covers the real CSVProcessorTool API (constructor + action()).
 */

import { describe, it, expect } from 'vitest';
import { CSVProcessorTool } from './csv-processor-tool';
import { CSVOperationType } from './csv-processor-tool';

describe('csv-processor-tool', () => {
  describe('Construction', () => {
    it('should construct with valid params (no throw)', () => {
      const tool = new CSVProcessorTool({
        operation: CSVOperationType.PARSE,
        csvData: 'a,b\n1,2',
      });
      expect(tool).toBeDefined();
      expect(tool.name).toBe('csv-processor-tool');
    });

    it('should not throw on invalid params (captures validation error)', () => {
      const tool = new CSVProcessorTool({});
      expect(tool).toBeDefined();
    });
  });

  describe('action()', () => {
    it('should parse CSV data into rows', async () => {
      const tool = new CSVProcessorTool({
        operation: CSVOperationType.PARSE,
        csvData: 'name,age\nbob,5\nalice,7',
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.rowCount).toBe(2);
      expect(result.data?.columnCount).toBe(2);
      expect(result.data?.headers).toEqual(['name', 'age']);
      expect(result.data?.data).toHaveLength(2);
    });

    it('should export data array to CSV string', async () => {
      const tool = new CSVProcessorTool({
        operation: CSVOperationType.EXPORT,
        exportData: [
          { a: 1, b: 2 },
          { a: 3, b: 4 },
        ],
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.csvOutput).toContain('a,b');
      expect(result.data?.csvOutput).toContain('1,2');
    });

    it('should filter rows based on conditions', async () => {
      const tool = new CSVProcessorTool({
        operation: CSVOperationType.FILTER,
        csvData: 'a\n1\n2\n3',
        filterRules: [
          { column: 'a', operator: 'gt', value: 1 },
        ],
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.rowCount).toBe(2);
    });

    it('should return a controlled error for parse without csvData', async () => {
      const tool = new CSVProcessorTool({
        operation: CSVOperationType.PARSE,
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('csvData is required');
    });

    it('should return a controlled error for invalid params', async () => {
      const tool = new CSVProcessorTool({});
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });
  });
});
