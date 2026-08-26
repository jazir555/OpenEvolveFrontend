/*
 * Test suite for pdf-form-operations.workflow
 *
 * These tests exercise the real Bubble API:
 *   - `new PDFFormOperationsWorkflow(params, context?)` does NOT throw on
 *     invalid params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - PDF form operations require an external Python/PyMuPDF runtime and are
 *     skipped here.
 */

import { describe, it, expect, vi } from 'vitest';
import { PDFFormOperationsWorkflow } from './pdf-form-operations.workflow';

function makeLogger() {
  return {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    log: vi.fn(),
    logBubbleExecution: vi.fn(),
    logBubbleExecutionComplete: vi.fn(),
  };
}

const VALID_PARAMS = { operation: 'discover' as const, pdfData: '' };

describe('pdf-form-operations.workflow', () => {
  it('does not throw when constructed with invalid params', () => {
    const instance = new PDFFormOperationsWorkflow({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new PDFFormOperationsWorkflow({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new PDFFormOperationsWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it.skip('executes a PDF form operation end-to-end (requires PyMuPDF runtime)', async () => {
    const instance = new PDFFormOperationsWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();
    expect(result.success).toBe(true);
    expect(result.data?.operation).toBe('discover');
  });
});
