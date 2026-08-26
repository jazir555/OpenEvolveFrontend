/*
 * Test Suite for file-processor-tool
 * Covers the real FileProcessorTool API (constructor + action()).
 * Uses a relative temp directory inside the project (the security util
 * rejects absolute paths unless they are in an allow-list).
 */

import { describe, it, expect, afterEach } from 'vitest';
import { existsSync, writeFileSync, mkdirSync, rmSync } from 'fs';
import { join } from 'path';
import { FileProcessorTool } from './file-processor-tool';
import { FileOperationType } from './file-processor-tool';

const TMP = 'fp-test-tmp';

describe('file-processor-tool', () => {
  afterEach(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  describe('Construction', () => {
    it('should construct with valid params (no throw)', () => {
      const tool = new FileProcessorTool({
        operation: FileOperationType.READ,
        filePath: 'x.txt',
      });
      expect(tool).toBeDefined();
      expect(tool.name).toBe('file-processor-tool');
    });

    it('should not throw on invalid params (captures validation error)', () => {
      const tool = new FileProcessorTool({});
      expect(tool).toBeDefined();
    });
  });

  describe('action()', () => {
    it('should write, read, and delete a file', async () => {
      const file = join(TMP, 'out.txt');
      const write = new FileProcessorTool({
        operation: FileOperationType.WRITE,
        filePath: file,
        content: 'hello world',
        createDirectory: true,
      });
      const writeResult = await write.action();
      expect(writeResult.success).toBe(true);
      expect(existsSync(file)).toBe(true);

      const read = new FileProcessorTool({
        operation: FileOperationType.READ,
        filePath: file,
      });
      const readResult = await read.action();
      expect(readResult.success).toBe(true);
      expect(readResult.data?.content).toBe('hello world');

      const del = new FileProcessorTool({
        operation: FileOperationType.DELETE,
        filePath: file,
      });
      const delResult = await del.action();
      expect(delResult.success).toBe(true);
      expect(existsSync(file)).toBe(false);
    });

    it('should report metadata for an existing file', async () => {
      const file = join(TMP, 'meta.txt');
      mkdirSync(TMP, { recursive: true });
      writeFileSync(file, 'abc');
      const meta = new FileProcessorTool({
        operation: FileOperationType.METADATA,
        filePath: file,
      });
      const result = await meta.action();
      expect(result.success).toBe(true);
      expect(result.data?.metadata?.isFile).toBe(true);
      expect(result.data?.metadata?.size).toBe(3);
    });

    it('should return a controlled error for a missing file on read', async () => {
      const file = join(TMP, 'missing.txt');
      const tool = new FileProcessorTool({
        operation: FileOperationType.READ,
        filePath: file,
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('does not exist');
    });

    it('should return a controlled error for invalid params', async () => {
      const tool = new FileProcessorTool({});
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });
  });
});
