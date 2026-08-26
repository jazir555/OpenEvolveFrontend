/*
 * Test suite for webhook-repeater.workflow
 *
 * These tests exercise the real Bubble API:
 *   - `new WebhookRepeaterWorkflow(params, context?)` does NOT throw on invalid
 *     params; instead it captures a validation error.
 *   - `action()` returns a controlled `{ success: false, error }` result for
 *     invalid input, and `{ success, data, error }` for valid input.
 *   - Delivery is exercised offline by mocking the global `fetch`.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { WebhookRepeaterWorkflow } from './webhook-repeater.workflow';

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
  webhookUrl: 'https://example.com/webhook',
  method: 'POST' as const,
  payload: '{"hello":"world"}',
};

describe('webhook-repeater.workflow', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does not throw when constructed with invalid params', () => {
    const instance = new WebhookRepeaterWorkflow({});
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('returns a controlled error from action() for invalid params', async () => {
    const instance = new WebhookRepeaterWorkflow({});
    const result = await instance.action();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('constructs a defined instance for valid params', () => {
    const instance = new WebhookRepeaterWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    expect(instance).toBeDefined();
    expect(typeof instance.action).toBe('function');
  });

  it('reports success when the endpoint responds with 2xx (mocked fetch)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response('ok', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const instance = new WebhookRepeaterWorkflow(VALID_PARAMS, {
      logger: makeLogger(),
    });
    const result = await instance.action();

    expect(result.success).toBe(true);
    expect(result.data?.deliveryStatus?.delivered).toBe(true);
    expect(fetchMock).toHaveBeenCalled();
  });

  it('reports failure when the endpoint responds with an error status (mocked fetch)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response('boom', { status: 500 }));
    vi.stubGlobal('fetch', fetchMock);

    const instance = new WebhookRepeaterWorkflow(
      { ...VALID_PARAMS, retryStrategy: { maxAttempts: 1 } },
      { logger: makeLogger() }
    );
    const result = await instance.action();

    expect(result.success).toBe(false);
    expect(typeof result.error).toBe('string');
    expect(result.data?.deliveryStatus?.delivered).toBe(false);
  });
});
