/*
 * Test suite for scheduled-task.workflow
 *
 * These tests exercise the real Bubble API:
 *   - `new ScheduledTaskWorkflow(params, context?)` does NOT throw on invalid
 *     params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - A `cron` schedule resolves fully offline (no external execution).
 */

import { describe, it, expect, vi } from 'vitest';
import { ScheduledTaskWorkflow } from './scheduled-task.workflow';

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
  taskName: 'test-task',
  schedule: { type: 'cron' as const, expression: '* * * * *' },
  action: { type: 'function' as const, config: {} },
};

describe('scheduled-task.workflow', () => {
  it('does not throw when constructed with invalid params', () => {
    const instance = new ScheduledTaskWorkflow({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new ScheduledTaskWorkflow({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new ScheduledTaskWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('schedules a cron task and returns success without external execution', async () => {
    const instance = new ScheduledTaskWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();
    expect(result.success).toBe(true);
    expect(result.data?.status).toBe('scheduled');
    expect(typeof result.data?.taskId).toBe('string');
  });
});
