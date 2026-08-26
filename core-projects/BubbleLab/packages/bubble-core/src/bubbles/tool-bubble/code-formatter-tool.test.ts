/*
 * Test Suite for code-formatter-tool
 * Covers the real CodeFormatterTool API (constructor + action()).
 */

import { describe, it, expect } from 'vitest';
import { CodeFormatterTool } from './code-formatter-tool';
import { CodeLanguage } from './code-formatter-tool';

describe('code-formatter-tool', () => {
  describe('Construction', () => {
    it('should construct with valid params (no throw)', () => {
      const tool = new CodeFormatterTool({
        code: 'const x=1',
        language: CodeLanguage.TYPESCRIPT,
      });
      expect(tool).toBeDefined();
      expect(tool.name).toBe('code-formatter-tool');
    });

    it('should not throw on invalid params (captures validation error)', () => {
      const tool = new CodeFormatterTool({});
      expect(tool).toBeDefined();
    });
  });

  describe('action()', () => {
    it('should format JSON with indentation', async () => {
      const tool = new CodeFormatterTool({
        code: '{"a":1,"b":2}',
        language: CodeLanguage.JSON,
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.formattedCode).toContain('"a": 1');
      expect(result.data?.formattedCode).toContain('"b": 2');
      expect(result.data?.stats.originalLength).toBeGreaterThan(0);
    });

    it('should trim trailing whitespace for generic code', async () => {
      const tool = new CodeFormatterTool({
        code: 'const x = 1   \nconst y = 2   ',
        language: CodeLanguage.JAVASCRIPT,
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.formattedCode).not.toMatch(/ +$/m);
    });

    it('should expose change statistics', async () => {
      const tool = new CodeFormatterTool({
        code: 'const x=1',
        language: CodeLanguage.JAVASCRIPT,
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.changes).toBeDefined();
      expect(result.data?.stats).toBeDefined();
      expect(typeof result.data?.stats.processingTime).toBe('number');
    });

    it('should return a controlled error for invalid params', async () => {
      const tool = new CodeFormatterTool({});
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });

    it('should return a controlled error for an invalid language enum', async () => {
      const tool = new CodeFormatterTool({
        code: 'x',
        // @ts-expect-error intentionally invalid enum
        language: 'not-a-language',
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });
  });
});
