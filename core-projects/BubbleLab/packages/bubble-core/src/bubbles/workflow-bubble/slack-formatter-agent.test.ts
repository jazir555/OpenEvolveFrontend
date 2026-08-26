/*
 * Test suite for slack-formatter-agent
 *
 * These tests exercise the real Bubble API:
 *   - `new SlackFormatterAgentBubble(params, context?)` does NOT throw on
 *     invalid params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - Formatting requires an external LLM and is skipped here.
 */

import { describe, it, expect, vi } from 'vitest';
import { SlackFormatterAgentBubble } from './slack-formatter-agent';

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

const VALID_PARAMS = { message: 'hello world' };

describe('slack-formatter-agent', () => {
  it('does not throw when constructed with invalid params', () => {
    const instance = new SlackFormatterAgentBubble({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new SlackFormatterAgentBubble({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new SlackFormatterAgentBubble(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it.skip('formats a Slack message end-to-end (requires an LLM provider)', async () => {
    const instance = new SlackFormatterAgentBubble(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();
    expect(result.success).toBe(true);
    expect(result.data?.message).toBeDefined();
  });
});
