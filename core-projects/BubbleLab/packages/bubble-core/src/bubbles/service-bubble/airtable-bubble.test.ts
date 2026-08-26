import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AirtableBubble } from './airtable-bubble';
import { CredentialType } from '@bubblelab/shared-schemas';

const creds = { [CredentialType.AIRTABLE_CRED]: 'patTest123' };

describe('AirtableBubble', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ records: [], id: 'x' }),
      }))
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('should expose correct static metadata', () => {
    expect(AirtableBubble.bubbleName).toBe('airtable');
    expect(AirtableBubble.service).toBe('airtable');
    expect(AirtableBubble.type).toBe('service');
    expect(AirtableBubble.schema).toBeDefined();
  });

  it('should accept valid params via the schema', () => {
    const result = AirtableBubble.schema.safeParse({
      operation: 'listRecords',
      baseId: 'app123',
      tableId: 'tbl123',
      credentials: creds,
    });
    expect(result.success).toBe(true);
  });

  it('should reject params missing required fields via the schema', () => {
    const result = AirtableBubble.schema.safeParse({
      operation: 'listRecords',
      credentials: creds,
    });
    expect(result.success).toBe(false);
  });

  it('should NOT throw on invalid params (constructor is resilient)', () => {
    expect(() => new AirtableBubble({} as any)).not.toThrow();
  });

  it('should return a controlled error from action() for invalid params', async () => {
    const bubble = new AirtableBubble({} as any);
    const result = await bubble.action();
    expect(result.success).toBe(false);
    expect(result.error).toContain('Input Schema validation failed');
  });

  it('should run a valid operation end-to-end with mocked fetch', async () => {
    const bubble = new AirtableBubble({
      operation: 'listRecords',
      baseId: 'app123',
      tableId: 'tbl123',
      credentials: creds,
    });
    const result = await bubble.action();
    expect(result).toBeDefined();
    expect(typeof result.error).toBe('string');
  });
});
