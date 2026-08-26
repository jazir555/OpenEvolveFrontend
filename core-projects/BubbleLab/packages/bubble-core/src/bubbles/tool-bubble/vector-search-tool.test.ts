/**
 * Vector Search Tool - unit tests
 * Rewritten to exercise the real VectorSearchTool API (performAction).
 * Uses the in-memory similarity path (no external Qdrant required).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { VectorSearchTool, VectorDistanceMetric } from './vector-search-tool';

const baseVectors = [
  { id: 'a', vector: [1, 0, 0], payload: { label: 'A' } },
  { id: 'b', vector: [0, 1, 0], payload: { label: 'B' } },
  { id: 'c', vector: [0, 0, 1], payload: { label: 'C' } },
];

describe('VectorSearchTool', () => {
  const originalQdrantUrl = process.env.QDRANT_URL;

  beforeEach(() => {
    // Force the in-memory search path (avoids any network call to Qdrant).
    process.env.QDRANT_URL = 'http://127.0.0.1:6333';
  });

  afterEach(() => {
    if (originalQdrantUrl === undefined) {
      delete process.env.QDRANT_URL;
    } else {
      process.env.QDRANT_URL = originalQdrantUrl;
    }
  });

  it('constructs with required params', () => {
    const tool = new VectorSearchTool({
      vector: [1, 2, 3],
      collectionName: 'c',
      vectorDimension: 3,
    });
    expect(tool).toBeDefined();
  });

  it('performs in-memory cosine similarity search', async () => {
    const tool = new VectorSearchTool({
      vector: [1, 0, 0],
      collectionName: 'test',
      vectorDimension: 3,
      distanceMetric: VectorDistanceMetric.COSINE,
      filter: { vectors: baseVectors },
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].id).toBe('a');
    expect(result.totalResults).toBe(result.results.length);
    expect(result.queryMetadata.collectionName).toBe('test');
  });

  it('validates vector dimension mismatch', async () => {
    const tool = new VectorSearchTool({
      vector: [1, 0],
      collectionName: 'test',
      vectorDimension: 3,
      filter: { vectors: baseVectors },
    });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('dimension');
  });
});
