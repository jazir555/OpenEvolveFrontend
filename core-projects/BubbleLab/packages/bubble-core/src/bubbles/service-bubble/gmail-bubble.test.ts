import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GmailBubble } from './gmail-bubble';
import { CredentialType } from '@bubblelab/shared-schemas';

const creds = { [CredentialType.GMAIL_CRED]: 'test-gmail-token' };

describe('GmailBubble (gmail-bubble.ts)', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ messages: [], resultSizeEstimate: 0, success: true }),
      }))
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('should expose correct static metadata', () => {
    expect(GmailBubble.bubbleName).toBe('gmail');
    expect(GmailBubble.service).toBe('gmail');
    expect(GmailBubble.type).toBe('service');
    expect(GmailBubble.authType).toBe('oauth');
    expect(GmailBubble.schema).toBeDefined();
  });

  it('should accept valid params via the schema', () => {
    const result = GmailBubble.schema.safeParse({
      operation: 'searchEmails',
      query: 'from:user@example.com',
      credentials: creds,
    });
    expect(result.success).toBe(true);
  });

  it('should reject params missing required fields via the schema', () => {
    const result = GmailBubble.schema.safeParse({
      operation: 'searchEmails',
      credentials: creds,
    });
    expect(result.success).toBe(false);
  });

  it('should NOT throw on invalid params (constructor is resilient)', () => {
    expect(() => new GmailBubble({} as any)).not.toThrow();
  });

  it('should return a controlled error from action() for invalid params', async () => {
    const bubble = new GmailBubble({} as any);
    const result = await bubble.action();
    expect(result.success).toBe(false);
    expect(result.error).toContain('Input Schema validation failed');
  });

  it('should run a valid operation end-to-end with mocked fetch', async () => {
    const bubble = new GmailBubble({
      operation: 'searchEmails',
      query: 'from:user@example.com',
      credentials: creds,
    });
    const result = await bubble.action();
    expect(result).toBeDefined();
    expect(typeof result.success).toBe('boolean');
    expect(typeof result.error).toBe('string');
  });
});
