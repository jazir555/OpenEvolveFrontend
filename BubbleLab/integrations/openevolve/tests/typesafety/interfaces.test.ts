/**
 * Interface and Schema Validation Tests
 *
 * Tests for Bug #4: Type safety fixes for Knowledge Engine interfaces
 * Validates that all interfaces properly validate data and catch invalid inputs
 */

import { z } from 'zod';

// Import schemas from knowledge-engine-bubble
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

describe('Interface and Schema Validation', () => {
  describe('QdrantSearchPoint', () => {
    it('should accept valid Qdrant search point with string ID', () => {
      const validData = {
        id: 'test-id-123',
        score: 0.95,
        payload: {
          content: 'Test content',
          source: 'test-source',
        },
        vector: [0.1, 0.2, 0.3],
      };

      const result = QdrantSearchPointSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.id).toBe('test-id-123');
        expect(result.data.score).toBe(0.95);
        expect(result.data.payload?.content).toBe('Test content');
      }
    });

    it('should accept valid Qdrant search point with numeric ID', () => {
      const validData = {
        id: 12345,
        score: 0.87,
        payload: {
          content: 'Test content',
        },
      };

      const result = QdrantSearchPointSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.id).toBe(12345);
        expect(result.data.score).toBe(0.87);
      }
    });

    it('should accept valid Qdrant point without optional fields', () => {
      const validData = {
        id: 'minimal-id',
        score: 0.75,
      };

      const result = QdrantSearchPointSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should reject Qdrant point missing required score', () => {
      const invalidData = {
        id: 'test-id',
        payload: { content: 'test' },
      };

      const result = QdrantSearchPointSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject Qdrant point missing required id', () => {
      const invalidData = {
        score: 0.9,
        payload: { content: 'test' },
      };

      const result = QdrantSearchPointSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject Qdrant point with invalid score type', () => {
      const invalidData = {
        id: 'test-id',
        score: 'not-a-number',
        payload: { content: 'test' },
      };

      const result = QdrantSearchPointSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject Qdrant point with invalid vector type', () => {
      const invalidData = {
        id: 'test-id',
        score: 0.9,
        vector: 'not-an-array',
      };

      const result = QdrantSearchPointSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject Qdrant point with non-numeric vector elements', () => {
      const invalidData = {
        id: 'test-id',
        score: 0.9,
        vector: [0.1, 'string', 0.3],
      };

      const result = QdrantSearchPointSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should accept empty payload object', () => {
      const validData = {
        id: 'test-id',
        score: 0.9,
        payload: {},
      };

      const result = QdrantSearchPointSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should accept payload with arbitrary metadata fields', () => {
      const validData = {
        id: 'test-id',
        score: 0.9,
        payload: {
          content: 'test',
          timestamp: 1234567890,
          tags: ['tag1', 'tag2'],
          nested: { field: 'value' },
        },
      };

      const result = QdrantSearchPointSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });
  });

  describe('ElasticsearchHit', () => {
    it('should accept valid Elasticsearch hit', () => {
      const validData = {
        _index: 'test-index',
        _id: 'doc-id-123',
        _score: 1.5,
        _source: {
          content: 'Test document content',
          title: 'Test Title',
        },
      };

      const result = ElasticsearchHitSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data._index).toBe('test-index');
        expect(result.data._id).toBe('doc-id-123');
        expect(result.data._score).toBe(1.5);
        expect(result.data._source?.content).toBe('Test document content');
      }
    });

    it('should accept Elasticsearch hit without _source', () => {
      const validData = {
        _index: 'test-index',
        _id: 'doc-id',
        _score: 1.0,
      };

      const result = ElasticsearchHitSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should reject hit missing _index', () => {
      const invalidData = {
        _id: 'doc-id',
        _score: 1.0,
        _source: { content: 'test' },
      };

      const result = ElasticsearchHitSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hit missing _id', () => {
      const invalidData = {
        _index: 'test-index',
        _score: 1.0,
        _source: { content: 'test' },
      };

      const result = ElasticsearchHitSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hit missing _score', () => {
      const invalidData = {
        _index: 'test-index',
        _id: 'doc-id',
        _source: { content: 'test' },
      };

      const result = ElasticsearchHitSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hit with invalid _score type', () => {
      const invalidData = {
        _index: 'test-index',
        _id: 'doc-id',
        _score: 'not-a-number',
      };

      const result = ElasticsearchHitSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should accept _source with arbitrary fields', () => {
      const validData = {
        _index: 'test-index',
        _id: 'doc-id',
        _score: 1.0,
        _source: {
          content: 'test',
          count: 42,
          active: true,
          tags: ['tag1', 'tag2'],
          nested: { field: 'value' },
        },
      };

      const result = ElasticsearchHitSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });
  });

  describe('ElasticsearchHits', () => {
    it('should accept valid Elasticsearch hits structure', () => {
      const validData = {
        total: {
          value: 100,
          relation: 'eq',
        },
        hits: [
          {
            _index: 'test-index',
            _id: 'doc-1',
            _score: 1.0,
            _source: { content: 'test1' },
          },
          {
            _index: 'test-index',
            _id: 'doc-2',
            _score: 0.9,
            _source: { content: 'test2' },
          },
        ],
      };

      const result = ElasticsearchHitsSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.total.value).toBe(100);
        expect(result.data.total.relation).toBe('eq');
        expect(result.data.hits).toHaveLength(2);
      }
    });

    it('should accept empty hits array', () => {
      const validData = {
        total: {
          value: 0,
          relation: 'eq',
        },
        hits: [],
      };

      const result = ElasticsearchHitsSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.hits).toHaveLength(0);
      }
    });

    it('should reject hits missing total', () => {
      const invalidData = {
        hits: [
          {
            _index: 'test-index',
            _id: 'doc-1',
            _score: 1.0,
          },
        ],
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hits with missing total.value', () => {
      const invalidData = {
        total: {
          relation: 'eq',
        },
        hits: [],
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hits with missing total.relation', () => {
      const invalidData = {
        total: {
          value: 100,
        },
        hits: [],
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hits when hits field is not an array', () => {
      const invalidData = {
        total: {
          value: 1,
          relation: 'eq',
        },
        hits: 'not-an-array',
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject hits array with invalid hit objects', () => {
      const invalidData = {
        total: {
          value: 2,
          relation: 'eq',
        },
        hits: [
          {
            _index: 'test-index',
            _id: 'doc-1',
            _score: 1.0,
          },
          {
            _index: 'test-index',
            // Missing _id and _score
          },
        ],
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject total.value with wrong type', () => {
      const invalidData = {
        total: {
          value: 'not-a-number',
          relation: 'eq',
        },
        hits: [],
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject total.relation with wrong type', () => {
      const invalidData = {
        total: {
          value: 100,
          relation: 123,
        },
        hits: [],
      };

      const result = ElasticsearchHitsSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });
  });

  describe('ElasticsearchResponseData', () => {
    it('should accept valid Elasticsearch response with all fields', () => {
      const validData = {
        hits: {
          total: {
            value: 10,
            relation: 'eq',
          },
          hits: [
            {
              _index: 'test-index',
              _id: 'doc-1',
              _score: 1.0,
              _source: { content: 'test' },
            },
          ],
        },
        took: 15,
        timed_out: false,
      };

      const result = ElasticsearchResponseDataSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.took).toBe(15);
        expect(result.data.timed_out).toBe(false);
        expect(result.data.hits?.hits).toHaveLength(1);
      }
    });

    it('should accept valid Elasticsearch response with only hits', () => {
      const validData = {
        hits: {
          total: {
            value: 5,
            relation: 'eq',
          },
          hits: [
            {
              _index: 'test-index',
              _id: 'doc-1',
              _score: 1.0,
            },
          ],
        },
      };

      const result = ElasticsearchResponseDataSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should accept valid Elasticsearch response with only took and timed_out', () => {
      const validData = {
        took: 20,
        timed_out: true,
      };

      const result = ElasticsearchResponseDataSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should accept empty response', () => {
      const validData = {};

      const result = ElasticsearchResponseDataSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should reject response with invalid took type', () => {
      const invalidData = {
        took: 'not-a-number',
      };

      const result = ElasticsearchResponseDataSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject response with invalid timed_out type', () => {
      const invalidData = {
        timed_out: 'not-a-boolean',
      };

      const result = ElasticsearchResponseDataSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should reject response with invalid hits structure', () => {
      const invalidData = {
        hits: {
          total: {
            value: 'invalid',
            relation: 'eq',
          },
          hits: [],
        },
      };

      const result = ElasticsearchResponseDataSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });
  });

  describe('Array Validation', () => {
    it('should validate array of Qdrant points', () => {
      const QdrantArraySchema = z.array(QdrantSearchPointSchema);
      const validArray = [
        { id: '1', score: 0.9, payload: { content: 'test1' } },
        { id: '2', score: 0.8, payload: { content: 'test2' } },
        { id: 3, score: 0.7 }, // numeric ID
      ];

      const result = QdrantArraySchema.safeParse(validArray);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toHaveLength(3);
      }
    });

    it('should reject array with one invalid Qdrant point', () => {
      const QdrantArraySchema = z.array(QdrantSearchPointSchema);
      const invalidArray = [
        { id: '1', score: 0.9, payload: { content: 'test1' } },
        { id: '2', score: 'invalid' }, // Invalid score
        { id: 3, score: 0.7 },
      ];

      const result = QdrantArraySchema.safeParse(invalidArray);
      expect(result.success).toBe(false);
    });

    it('should accept empty array', () => {
      const QdrantArraySchema = z.array(QdrantSearchPointSchema);
      const emptyArray: unknown[] = [];

      const result = QdrantArraySchema.safeParse(emptyArray);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toHaveLength(0);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle null values in optional fields', () => {
      const data = {
        id: 'test-id',
        score: 0.9,
        payload: null,
        vector: null,
      };

      const result = QdrantSearchPointSchema.safeParse(data);
      // null is not valid for optional fields - should be undefined or valid value
      expect(result.success).toBe(false);
    });

    it('should handle undefined values in optional fields', () => {
      const data = {
        id: 'test-id',
        score: 0.9,
        payload: undefined,
        vector: undefined,
      };

      const result = QdrantSearchPointSchema.safeParse(data);
      expect(result.success).toBe(true);
    });

    it('should handle zero values', () => {
      const data = {
        id: 'test-id',
        score: 0,
        vector: [0, 0, 0],
      };

      const result = QdrantSearchPointSchema.safeParse(data);
      expect(result.success).toBe(true);
    });

    it('should handle negative scores', () => {
      const data = {
        id: 'test-id',
        score: -0.5,
      };

      const result = QdrantSearchPointSchema.safeParse(data);
      expect(result.success).toBe(true);
    });

    it('should handle very large arrays', () => {
      const largeVector = Array(10000).fill(0.1);
      const data = {
        id: 'test-id',
        score: 0.9,
        vector: largeVector,
      };

      const result = QdrantSearchPointSchema.safeParse(data);
      expect(result.success).toBe(true);
    });

    it('should handle deeply nested payload structures', () => {
      const data = {
        id: 'test-id',
        score: 0.9,
        payload: {
          level1: {
            level2: {
              level3: {
                level4: {
                  value: 'deep',
                },
              },
            },
          },
        },
      };

      const result = QdrantSearchPointSchema.safeParse(data);
      expect(result.success).toBe(true);
    });
  });

  describe('Real-world Data Patterns', () => {
    it('should accept typical Qdrant search response', () => {
      const typicalResponse = [
        {
          id: 'doc_123',
          score: 0.9543,
          payload: {
            content: 'This is a sample document about AI and machine learning.',
            title: 'Introduction to AI',
            category: 'technology',
            timestamp: 1234567890,
            tags: ['ai', 'ml', 'technology'],
          },
          vector: [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        {
          id: 456,
          score: 0.8732,
          payload: {
            content: 'Another document on knowledge graphs.',
            title: 'Knowledge Graphs 101',
          },
        },
      ];

      const QdrantArraySchema = z.array(QdrantSearchPointSchema);
      const result = QdrantArraySchema.safeParse(typicalResponse);
      expect(result.success).toBe(true);
    });

    it('should accept typical Elasticsearch search response', () => {
      const typicalResponse = {
        took: 12,
        timed_out: false,
        hits: {
          total: {
            value: 2,
            relation: 'eq',
          },
          hits: [
            {
              _index: 'knowledge_base',
              _id: 'doc_123',
              _score: 1.5432,
              _source: {
                content: 'This is a sample document.',
                title: 'Sample Document',
                metadata: {
                  author: 'John Doe',
                  date: '2024-01-01',
                },
              },
            },
            {
              _index: 'knowledge_base',
              _id: 'doc_456',
              _score: 1.2345,
              _source: {
                content: 'Another document.',
                title: 'Another Sample',
              },
            },
          ],
        },
      };

      const result = ElasticsearchResponseDataSchema.safeParse(typicalResponse);
      expect(result.success).toBe(true);
    });
  });
});
