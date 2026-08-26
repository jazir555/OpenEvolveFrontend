/*
 * Test suite for slack-notifier.workflow
 *
 * These tests exercise the real Bubble API:
 *   - `new SlackNotifierWorkflowBubble(params, context?)` does NOT throw on
 *     invalid params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - Formatting + Slack delivery require external services and are skipped.
 */

import { describe, it, expect, vi } from 'vitest';
import { SlackNotifierWorkflowBubble } from './slack-notifier.workflow';

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

const VALID_PARAMS = {
  contentToFormat: 'Hello team, deployment succeeded.',
  targetChannel: 'general',
};

describe('slack-notifier.workflow', () => {
  it('does not throw when constructed with invalid params', () => {
    const instance = new SlackNotifierWorkflowBubble({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new SlackNotifierWorkflowBubble({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new SlackNotifierWorkflowBubble(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it.skip('formats and delivers a Slack message end-to-end (requires Slack + AI credentials)', async () => {
    const instance = new SlackNotifierWorkflowBubble(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();
    expect(result.success).toBe(true);
    expect(result.data?.formattedMessage).toBeDefined();
  });
});
