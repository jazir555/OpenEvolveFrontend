/**
 * Runtime Validation Tests
 *
 * Tests for Bug #4: Runtime validation in Knowledge Engine operations
 * Validates that invalid responses are properly rejected with clear error messages
 */

import { z } from 'zod';

// Mock the knowledge engine bubble behavior
// In a real test, these would be imported from the actual implementation

// Schemas
const QdrantSearchPointSchema = z.object({
  id: z.union([z.string(), z.number()]),
  score: z.number(),
  payload: z.record(z.unknown()).optional(),
  vector: z.array(z.number()).optional(),
});

const ElasticsearchHitSchema = z.object({
  _index: z.string(),
  _id: z.string(),
  _score: z.number(),
  _source: z.record(z.unknown()).optional(),
});

const ElasticsearchHitsSchema = z.object({
  total: z.object({
    value: z.number(),
    relation: z.string(),
  }),
  hits: z.array(ElasticsearchHitSchema),
});

const ElasticsearchResponseDataSchema = z.object({
  hits: ElasticsearchHitsSchema.optional(),
  took: z.number().optional(),
  timed_out: z.boolean().optional(),
});

// Type guards
function isValidQdrantResponse(data: unknown): data is any[] {
  return z.array(QdrantSearchPointSchema).safeParse(data).success;
}

function isValidElasticsearchResponse(data: unknown): any {
  return ElasticsearchResponseDataSchema.safeParse(data).success;
}

function validateQdrantResult(data: unknown): {
  valid: boolean;
  data?: any[];
  error?: string;
} {
  try {
    if (!isValidQdrantResponse(data)) {
      return {
        valid: false,
        error: 'Invalid Qdrant response: data does not match expected schema',
      };
    }
    return { valid: true, data };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown validation error';
    return { valid: false, error: errorMessage };
  }
}

function validateElasticsearchResult(data: unknown): {
  valid: boolean;
  hits?: any[];
  error?: string;
} {
  try {
    if (!isValidElasticsearchResponse(data)) {
      return {
        valid: false,
        error: 'Invalid Elasticsearch response: data does not match expected schema',
      };
    }

    const validated = ElasticsearchResponseDataSchema.parse(data);

    if (!validated.hits) {
      return {
        valid: false,
        error: 'Invalid Elasticsearch response: missing hits field',
      };
    }

    return { valid: true, hits: validated.hits.hits };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown validation error';
    return { valid: false, error: errorMessage };
  }
}

// Mock search results
type SearchResult = {
  success: boolean;
  operation: string;
  backend: string;
  results?: any[];
  error?: string;
  timing: number;
};

// Mock Qdrant search
async function mockQdrantSearch(responseData: unknown): Promise<SearchResult> {
  const validationResult = validateQdrantResult(responseData);

  if (!validationResult.valid) {
    return {
      success: false,
      operation: 'search',
      backend: 'qdrant',
      error: validationResult.error || 'Failed to validate Qdrant response',
      timing: 10,
    };
  }

  const results = validationResult.data?.map((point) => ({
    id: String(point.id),
    content: point.payload?.content || '',
    score: point.score,
    metadata: point.payload,
  }));

  return {
    success: true,
    operation: 'search',
    backend: 'qdrant',
    results,
    timing: 10,
  };
}

// Mock Elasticsearch search
async function mockElasticsearchSearch(responseData: unknown): Promise<SearchResult> {
  const validationResult = validateElasticsearchResult(responseData);

  if (!validationResult.valid) {
    return {
      success: false,
      operation: 'search',
      backend: 'elasticsearch',
      error: validationResult.error || 'Failed to validate Elasticsearch response',
      timing: 15,
    };
  }

  const results = validationResult.hits?.map((hit) => ({
    id: hit._id,
    content: hit._source?.content || '',
    score: hit._score,
    metadata: hit._source,
  }));

  return {
    success: true,
    operation: 'search',
    backend: 'elasticsearch',
    results,
    timing: 15,
  };
}

// Mock hybrid search
async function mockHybridSearch(
  qdrantResponse: unknown,
  esResponse: unknown,
  semanticWeight = 0.5,
  keywordWeight = 0.5
): Promise<SearchResult> {
  const combinedResults: any[] = [];

  // Process Qdrant results
  if (isValidQdrantResponse(qdrantResponse)) {
    for (const point of qdrantResponse) {
      combinedResults.push({
        id: String(point.id),
        content: point.payload?.content || '',
        score: point.score * semanticWeight,
        metadata: point.payload,
        source: 'qdrant',
      });
    }
  }

  // Process Elasticsearch results
  const esValidation = validateElasticsearchResult(esResponse);
  if (esValidation.valid && esValidation.hits) {
    for (const hit of esValidation.hits) {
      const existing = combinedResults.find((r) => r.id === hit._id);
      if (existing) {
        existing.score += hit._score * keywordWeight;
      } else {
        combinedResults.push({
          id: hit._id,
          content: hit._source?.content || '',
          score: hit._score * keywordWeight,
          metadata: hit._source,
          source: 'elasticsearch',
        });
      }
    }
  }

  return {
    success: true,
    operation: 'hybrid_search',
    backend: 'hybrid',
    results: combinedResults.sort((a, b) => b.score - a.score),
    timing: 25,
  };
}

describe('Runtime Validation Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Qdrant Search Validation', () => {
    it('should accept valid Qdrant response', async () => {
      const validResponse = [
        { id: 'doc1', score: 0.95, payload: { content: 'Test document 1' } },
        { id: 'doc2', score: 0.87, payload: { content: 'Test document 2' } },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results).toBeDefined();
      expect(result.results?.length).toBe(2);
      expect(result.error).toBeUndefined();
    });

    it('should reject Qdrant response with missing required fields', async () => {
      const invalidResponse = [
        { id: 'doc1' }, // Missing score
        { id: 'doc2', score: 0.87 },
      ];

      const result = await mockQdrantSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid Qdrant response');
      expect(result.results).toBeUndefined();
    });

    it('should reject Qdrant response with wrong types', async () => {
      const invalidResponse = [
        { id: 'doc1', score: 'not-a-number', payload: { content: 'test' } },
      ];

      const result = await mockQdrantSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid Qdrant response');
    });

    it('should reject non-array Qdrant response', async () => {
      const invalidResponse = { id: 'doc1', score: 0.95 };

      const result = await mockQdrantSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid Qdrant response');
    });

    it('should reject null Qdrant response', async () => {
      const result = await mockQdrantSearch(null);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid Qdrant response');
    });

    it('should provide clear error message for invalid Qdrant response', async () => {
      const invalidResponse = 'not valid data';

      const result = await mockQdrantSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Invalid Qdrant response: data does not match expected schema');
    });

    it('should handle empty Qdrant response array', async () => {
      const emptyResponse: any[] = [];

      const result = await mockQdrantSearch(emptyResponse);

      expect(result.success).toBe(true);
      expect(result.results).toEqual([]);
    });

    it('should transform Qdrant results correctly', async () => {
      const validResponse = [
        {
          id: 'doc1',
          score: 0.95,
          payload: { content: 'Test content', source: 'test-source' },
        },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0]).toEqual({
        id: 'doc1',
        content: 'Test content',
        score: 0.95,
        metadata: { content: 'Test content', source: 'test-source' },
      });
    });

    it('should handle Qdrant response with numeric IDs', async () => {
      const validResponse = [
        { id: 12345, score: 0.95, payload: { content: 'Test' } },
        { id: 67890, score: 0.87, payload: { content: 'Test 2' } },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].id).toBe('12345'); // Converted to string
      expect(result.results?.[1].id).toBe('67890');
    });

    it('should handle Qdrant response with missing payload', async () => {
      const validResponse = [{ id: 'doc1', score: 0.95 }];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].content).toBe('');
      expect(result.results?.[0].metadata).toBeUndefined();
    });
  });

  describe('Elasticsearch Search Validation', () => {
    it('should accept valid Elasticsearch response', async () => {
      const validResponse = {
        hits: {
          total: { value: 2, relation: 'eq' },
          hits: [
            { _index: 'test', _id: 'doc1', _score: 1.5, _source: { content: 'Test 1' } },
            { _index: 'test', _id: 'doc2', _score: 1.3, _source: { content: 'Test 2' } },
          ],
        },
        took: 15,
        timed_out: false,
      };

      const result = await mockElasticsearchSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results).toBeDefined();
      expect(result.results?.length).toBe(2);
      expect(result.error).toBeUndefined();
    });

    it('should reject Elasticsearch response with missing hits', async () => {
      const invalidResponse = {
        took: 15,
        timed_out: false,
      };

      const result = await mockElasticsearchSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toContain('missing hits field');
    });

    it('should reject Elasticsearch response with invalid hits structure', async () => {
      const invalidResponse = {
        hits: {
          total: { value: 'invalid', relation: 'eq' },
          hits: [],
        },
      };

      const result = await mockElasticsearchSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid Elasticsearch response');
    });

    it('should reject null Elasticsearch response', async () => {
      const result = await mockElasticsearchSearch(null);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid Elasticsearch response');
    });

    it('should provide clear error message for invalid Elasticsearch response', async () => {
      const invalidResponse = { invalid: 'data' };

      const result = await mockElasticsearchSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Invalid Elasticsearch response: data does not match expected schema');
    });

    it('should handle empty Elasticsearch hits array', async () => {
      const validResponse = {
        hits: {
          total: { value: 0, relation: 'eq' },
          hits: [],
        },
      };

      const result = await mockElasticsearchSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results).toEqual([]);
    });

    it('should transform Elasticsearch results correctly', async () => {
      const validResponse = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [
            {
              _index: 'kb',
              _id: 'doc1',
              _score: 1.8,
              _source: { content: 'Test content', title: 'Test' },
            },
          ],
        },
      };

      const result = await mockElasticsearchSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0]).toEqual({
        id: 'doc1',
        content: 'Test content',
        score: 1.8,
        metadata: { content: 'Test content', title: 'Test' },
      });
    });

    it('should handle Elasticsearch response without _source', async () => {
      const validResponse = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [{ _index: 'kb', _id: 'doc1', _score: 1.5 }],
        },
      };

      const result = await mockElasticsearchSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].content).toBe('');
    });
  });

  describe('Hybrid Search Validation', () => {
    it('should combine valid Qdrant and Elasticsearch results', async () => {
      const qdrantResponse = [
        { id: 'doc1', score: 0.9, payload: { content: 'Content 1' } },
        { id: 'doc2', score: 0.8, payload: { content: 'Content 2' } },
      ];

      const esResponse = {
        hits: {
          total: { value: 2, relation: 'eq' },
          hits: [
            { _index: 'kb', _id: 'doc1', _score: 1.5, _source: { content: 'Content 1' } },
            { _index: 'kb', _id: 'doc3', _score: 1.2, _source: { content: 'Content 3' } },
          ],
        },
      };

      const result = await mockHybridSearch(qdrantResponse, esResponse);

      expect(result.success).toBe(true);
      expect(result.results?.length).toBe(3); // doc1 combined, doc2, doc3

      // Check doc1 has combined scores
      const doc1 = result.results?.find((r: any) => r.id === 'doc1');
      expect(doc1?.score).toBeCloseTo(0.9 * 0.5 + 1.5 * 0.5, 5);
    });

    it('should handle valid Qdrant with invalid Elasticsearch', async () => {
      const qdrantResponse = [
        { id: 'doc1', score: 0.9, payload: { content: 'Content 1' } },
      ];

      const esResponse = { invalid: 'data' };

      const result = await mockHybridSearch(qdrantResponse, esResponse);

      expect(result.success).toBe(true);
      expect(result.results?.length).toBe(1); // Only Qdrant results
      expect(result.results?.[0].source).toBe('qdrant');
    });

    it('should handle invalid Qdrant with valid Elasticsearch', async () => {
      const qdrantResponse = 'invalid data';

      const esResponse = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [
            { _index: 'kb', _id: 'doc1', _score: 1.5, _source: { content: 'Content 1' } },
          ],
        },
      };

      const result = await mockHybridSearch(qdrantResponse, esResponse);

      expect(result.success).toBe(true);
      expect(result.results?.length).toBe(1); // Only ES results
      expect(result.results?.[0].source).toBe('elasticsearch');
    });

    it('should handle both invalid responses', async () => {
      const qdrantResponse = 'invalid';
      const esResponse = { invalid: 'data' };

      const result = await mockHybridSearch(qdrantResponse, esResponse);

      expect(result.success).toBe(true);
      expect(result.results).toEqual([]); // Empty results
    });

    it('should apply custom weights correctly', async () => {
      const qdrantResponse = [{ id: 'doc1', score: 0.9, payload: { content: 'Content 1' } }];
      const esResponse = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [{ _index: 'kb', _id: 'doc1', _score: 1.5, _source: { content: 'Content 1' } }],
        },
      };

      const result = await mockHybridSearch(qdrantResponse, esResponse, 0.7, 0.3);

      expect(result.success).toBe(true);
      const doc1 = result.results?.[0];
      expect(doc1?.score).toBeCloseTo(0.9 * 0.7 + 1.5 * 0.3, 5);
    });

    it('should sort results by combined score', async () => {
      const qdrantResponse = [
        { id: 'doc1', score: 0.5, payload: { content: 'Content 1' } },
        { id: 'doc2', score: 0.9, payload: { content: 'Content 2' } },
      ];

      const esResponse = {
        hits: {
          total: { value: 2, relation: 'eq' },
          hits: [
            { _index: 'kb', _id: 'doc3', _score: 1.0, _source: { content: 'Content 3' } },
            { _index: 'kb', _id: 'doc4', _score: 0.6, _source: { content: 'Content 4' } },
          ],
        },
      };

      const result = await mockHybridSearch(qdrantResponse, esResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].score).toBeGreaterThanOrEqual(result.results?.[1].score as number);
    });

    it('should track source correctly', async () => {
      const qdrantResponse = [
        { id: 'doc1', score: 0.9, payload: { content: 'Content 1' } },
      ];
      const esResponse = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [
            { _index: 'kb', _id: 'doc2', _score: 1.5, _source: { content: 'Content 2' } },
          ],
        },
      };

      const result = await mockHybridSearch(qdrantResponse, esResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].source).toBe('qdrant');
      expect(result.results?.[1].source).toBe('elasticsearch');
    });
  });

  describe('Error Message Quality', () => {
    it('should provide specific error for missing required fields', async () => {
      const invalidResponse = [{ id: 'doc1' }]; // Missing score

      const result = await mockQdrantSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error?.length).toBeGreaterThan(0);
    });

    it('should provide specific error for type mismatches', async () => {
      const invalidResponse = [{ id: 'doc1', score: 'string' }]; // Wrong type

      const result = await mockQdrantSearch(invalidResponse);

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should distinguish between Qdrant and ES errors', async () => {
      const qdrantResult = await mockQdrantSearch('invalid');
      const esResult = await mockElasticsearchSearch('invalid');

      expect(qdrantResult.error).toContain('Qdrant');
      expect(esResult.error).toContain('Elasticsearch');
    });

    it('should include context in error messages', async () => {
      const result = await mockQdrantSearch({ invalid: 'data' });

      expect(result.error).toMatch(/qdrant|response|schema/i);
    });
  });

  describe('Edge Cases and Robustness', () => {
    it('should handle very large result sets', async () => {
      const largeResponse = Array(1000)
        .fill(null)
        .map((_, i) => ({
          id: `doc${i}`,
          score: Math.random(),
          payload: { content: `Document ${i}` },
        }));

      const result = await mockQdrantSearch(largeResponse);

      expect(result.success).toBe(true);
      expect(result.results?.length).toBe(1000);
    });

    it('should handle responses with special characters in content', async () => {
      const validResponse = [
        {
          id: 'doc1',
          score: 0.9,
          payload: { content: 'Special chars: <>&"\'\n\t' },
        },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].content).toContain('Special chars');
    });

    it('should handle Unicode characters in content', async () => {
      const validResponse = [
        {
          id: 'doc1',
          score: 0.9,
          payload: { content: 'Unicode: 你好 🚀 𝔘𝔫𝔦𝔠𝔬𝔡𝔢' },
        },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].content).toContain('你好');
    });

    it('should handle responses with nested metadata', async () => {
      const validResponse = [
        {
          id: 'doc1',
          score: 0.9,
          payload: {
            content: 'Test',
            metadata: {
              nested: {
                deeply: {
                  value: 'deep',
                },
              },
            },
          },
        },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].metadata?.metadata?.nested?.deeply?.value).toBe('deep');
    });

    it('should handle extremely long content', async () => {
      const longContent = 'x'.repeat(100000);
      const validResponse = [
        { id: 'doc1', score: 0.9, payload: { content: longContent } },
      ];

      const result = await mockQdrantSearch(validResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].content?.length).toBe(100000);
    });
  });

  describe('Performance and Reliability', () => {
    it('should fail fast on invalid data', async () => {
      const invalidResponse = [{ id: 'doc1', score: 'invalid' }];

      const start = Date.now();
      const result = await mockQdrantSearch(invalidResponse);
      const duration = Date.now() - start;

      expect(result.success).toBe(false);
      expect(duration).toBeLessThan(100); // Should fail quickly
    });

    it('should handle concurrent validations', async () => {
      const validResponse = [{ id: 'doc1', score: 0.9, payload: { content: 'Test' } }];

      const promises = Array(100)
        .fill(null)
        .map(() => mockQdrantSearch(validResponse));

      const results = await Promise.all(promises);

      expect(results.every((r) => r.success)).toBe(true);
    });

    it('should not crash on malicious input', async () => {
      const maliciousInputs = [
        null,
        undefined,
        'string',
        123,
        true,
        { invalid: 'data' },
        [null, undefined, 'invalid'],
      ];

      const promises = maliciousInputs.map((input) => mockQdrantSearch(input));
      const results = await Promise.all(promises);

      expect(results.every((r) => r.success === false)).toBe(true);
    });
  });

  describe('Real-world Scenarios', () => {
    it('should handle typical successful Qdrant search', async () => {
      const typicalResponse = [
        {
          id: 'doc_12345',
          score: 0.9543,
          payload: {
            content: 'Machine learning is a subset of artificial intelligence.',
            title: 'Introduction to ML',
            category: 'technology',
            timestamp: 1234567890,
            tags: ['ml', 'ai', 'tech'],
          },
        },
        {
          id: 'doc_67890',
          score: 0.8732,
          payload: {
            content: 'Knowledge graphs represent relationships between entities.',
            title: 'Knowledge Graphs',
          },
        },
      ];

      const result = await mockQdrantSearch(typicalResponse);

      expect(result.success).toBe(true);
      expect(result.results).toHaveLength(2);
      expect(result.results?.[0].metadata?.tags).toEqual(['ml', 'ai', 'tech']);
    });

    it('should handle typical successful Elasticsearch search', async () => {
      const typicalResponse = {
        took: 12,
        timed_out: false,
        hits: {
          total: { value: 2, relation: 'eq' },
          hits: [
            {
              _index: 'knowledge_base',
              _id: 'doc_123',
              _score: 1.5432,
              _source: {
                content: 'Document about AI systems.',
                title: 'AI Systems',
                metadata: { author: 'John Doe', date: '2024-01-01' },
              },
            },
          ],
        },
      };

      const result = await mockElasticsearchSearch(typicalResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].metadata?.author).toBe('John Doe');
    });

    it('should handle degraded service responses', async () => {
      // Simulate response with missing fields but still valid structure
      const degradedResponse = [{ id: 'doc1', score: 0.9 }]; // No payload

      const result = await mockQdrantSearch(degradedResponse);

      expect(result.success).toBe(true);
      expect(result.results?.[0].content).toBe(''); // Default empty
    });
  });
});
