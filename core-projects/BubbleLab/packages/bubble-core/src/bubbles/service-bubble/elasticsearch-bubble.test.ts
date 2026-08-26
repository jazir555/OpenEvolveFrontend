import { describe, it, expect, vi } from 'vitest';
import { ElasticsearchBubble } from './elasticsearch-bubble';
import { CredentialType } from '@bubblelab/shared-schemas';

const creds = { [CredentialType.ELASTICSEARCH_CRED]: 'http://localhost:9200' };

describe('ElasticsearchBubble', () => {
  it('should expose correct static metadata', () => {
    expect(ElasticsearchBubble.bubbleName).toBe('elasticsearch');
    expect(ElasticsearchBubble.service).toBe('elasticsearch');
    expect(ElasticsearchBubble.type).toBe('service');
    expect(ElasticsearchBubble.schema).toBeDefined();
  });

  it('should accept valid params via the schema', () => {
    const result = ElasticsearchBubble.schema.safeParse({
      operation: 'indexExists',
      indexName: 'my-index',
      credentials: creds,
    });
    expect(result.success).toBe(true);
  });

  it('should reject params missing required fields via the schema', () => {
    const result = ElasticsearchBubble.schema.safeParse({
      operation: 'search',
      credentials: creds,
    });
    // search requires indexName and query
    expect(result.success).toBe(false);
  });

  it('should NOT throw on invalid params (constructor is resilient)', () => {
    expect(() => new ElasticsearchBubble({} as any)).not.toThrow();
  });

  it('should return a controlled error from action() for invalid params', async () => {
    const bubble = new ElasticsearchBubble({} as any);
    const result = await bubble.action();
    expect(result.success).toBe(false);
    expect(result.error).toContain('Input Schema validation failed');
  });

  it('should construct valid params without throwing (client is lazy)', () => {
    expect(
      () =>
        new ElasticsearchBubble({
          operation: 'indexExists',
          indexName: 'my-index',
          credentials: creds,
        })
    ).not.toThrow();
  });
});
