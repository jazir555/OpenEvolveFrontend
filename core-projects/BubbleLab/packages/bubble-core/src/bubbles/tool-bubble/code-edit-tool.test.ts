/*
 * Test Suite for code-edit-tool
 * Covers the real EditBubbleFlowTool API (constructor + action()).
 */

import { describe, it, expect } from 'vitest';
import { EditBubbleFlowTool } from './code-edit-tool';

describe('code-edit-tool', () => {
  describe('Construction', () => {
    it('should construct with valid params (no throw)', () => {
      const tool = new EditBubbleFlowTool({
        initialCode: 'hello world',
        old_string: 'world',
        new_string: 'there',
      });
      expect(tool).toBeDefined();
      expect(tool.name).toBe('code-edit-tool');
    });

    it('should not throw on invalid params (captures validation error)', () => {
      // Missing required fields -> constructor does NOT throw; action() returns a
      // controlled error instead.
      const tool = new EditBubbleFlowTool({});
      expect(tool).toBeDefined();
    });
  });

  describe('action()', () => {
    it('should apply a unique find-and-replace edit', async () => {
      const tool = new EditBubbleFlowTool({
        initialCode: 'hello world',
        old_string: 'world',
        new_string: 'there',
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.mergedCode).toBe('hello there');
      expect(result.data?.applied).toBe(true);
      expect(result.data?.error).toBe('');
    });

    it('should apply replace_all when requested', async () => {
      const tool = new EditBubbleFlowTool({
        initialCode: 'a a a',
        old_string: 'a',
        new_string: 'b',
        replace_all: true,
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.mergedCode).toBe('b b b');
      expect(result.data?.applied).toBe(true);
    });

    it('should return success:false when old_string is not found', async () => {
      const tool = new EditBubbleFlowTool({
        initialCode: 'hello',
        old_string: 'xyz',
        new_string: 'abc',
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.data?.applied).toBe(false);
      expect(result.data?.error).toContain('not found');
    });

    it('should return success:false when old_string is not unique', async () => {
      const tool = new EditBubbleFlowTool({
        initialCode: 'hello hello',
        old_string: 'hello',
        new_string: 'hi',
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.data?.applied).toBe(false);
      expect(result.data?.error).toContain('not unique');
    });

    it('should return success:false when initial code is empty', async () => {
      const tool = new EditBubbleFlowTool({
        initialCode: '',
        old_string: 'a',
        new_string: 'b',
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.data?.applied).toBe(false);
      expect(result.data?.error).toContain('empty');
    });

    it('should return a controlled error for invalid params', async () => {
      const tool = new EditBubbleFlowTool({});
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });
  });
});
