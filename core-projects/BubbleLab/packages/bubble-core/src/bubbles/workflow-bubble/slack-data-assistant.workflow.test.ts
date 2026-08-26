/*
 * Test suite for slack-data-assistant.workflow
 *
 * These tests exercise the real Bubble API:
 *   - `new SlackDataAssistantWorkflow(params, context?)` does NOT throw on
 *     invalid params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - Query generation + Slack delivery require external services and are skipped.
 */

import { describe, it, expect, vi } from 'vitest';
import { SlackDataAssistantWorkflow } from './slack-data-assistant.workflow';

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
  slackChannel: 'C1234567890',
  userQuestion: 'How many users signed up last week?',
};

describe('slack-data-assistant.workflow', () => {
  it('does not throw when constructed with invalid params', () => {
    const instance = new SlackDataAssistantWorkflow({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new SlackDataAssistantWorkflow({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new SlackDataAssistantWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it.skip('answers a data question end-to-end (requires database + AI + Slack credentials)', async () => {
    const instance = new SlackDataAssistantWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();
    expect(result.success).toBe(true);
    expect(result.data?.query).toBeDefined();
  });
});
