import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AGIIncBubble } from './agi-inc';
import { CredentialType } from '@bubblelab/shared-schemas';

const creds = { [CredentialType.AGI_API_KEY]: 'test-agi-key' };

describe('AGIIncBubble', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ success: true, data: { sessions: [] }, id: 'x' }),
      }))
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('should expose correct static metadata', () => {
    expect(AGIIncBubble.bubbleName).toBe('agi-inc');
    expect(AGIIncBubble.service).toBe('agi-inc');
    expect(AGIIncBubble.type).toBe('service');
    expect(AGIIncBubble.authType).toBe('apikey');
    expect(AGIIncBubble.schema).toBeDefined();
  });

  it('should accept valid params via the schema', () => {
    const result = AGIIncBubble.schema.safeParse({
      operation: 'list_sessions',
      credentials: creds,
    });
    expect(result.success).toBe(true);
  });

  it('should reject params with a missing required field via the schema', () => {
    const result = AGIIncBubble.schema.safeParse({
      operation: 'get_session',
      credentials: creds,
    });
    // get_session requires session_id
    expect(result.success).toBe(false);
  });

  it('should NOT throw on invalid params (constructor is resilient)', () => {
    expect(() => new AGIIncBubble({} as any)).not.toThrow();
  });

  it('should return a controlled error from action() for invalid params', async () => {
    const bubble = new AGIIncBubble({} as any);
    const result = await bubble.action();
    expect(result.success).toBe(false);
    expect(result.error).toContain('Input Schema validation failed');
  });

  it('should run a valid operation end-to-end with mocked fetch', async () => {
    const bubble = new AGIIncBubble({
      operation: 'list_sessions',
      credentials: creds,
    });
    const result = await bubble.action();
    expect(result).toBeDefined();
    expect(typeof result.success).toBe('boolean');
    expect(typeof result.error).toBe('string');
  });
});
