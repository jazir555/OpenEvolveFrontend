/*
 * Test suite for parse-document.workflow
 *
 * These tests exercise the real Bubble API:
 *   - `new ParseDocumentWorkflow(params, context?)` does NOT throw on invalid
 *     params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - Full PDF/AI parsing requires external services (PyMuPDF + AI creds) and
 *     is skipped here.
 */

import { describe, it, expect, vi } from 'vitest';
import { ParseDocumentWorkflow } from './parse-document.workflow';

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

const VALID_PARAMS = { documentData: 'dGVzdA==' };

describe('parse-document.workflow', () => {
  it('does not throw when constructed with invalid params', () => {
    const instance = new ParseDocumentWorkflow({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new ParseDocumentWorkflow({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new ParseDocumentWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it.skip('executes the full parsing workflow end-to-end (requires PyMuPDF + AI credentials)', async () => {
    const instance = new ParseDocumentWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();
    expect(result.success).toBe(true);
    expect(result.data?.markdown).toBeDefined();
  });
});
