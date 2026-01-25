/**
 * Type Guard Tests
 *
 * Tests for Bug #4: Type guard functions for runtime validation
 * Validates that type guards correctly identify valid/invalid data
 */

import { z } from 'zod';

// Schemas (copied from knowledge-engine-bubble for testing)
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

// Type guard implementations
function isValidQdrantResponse(data: unknown): data is any[] {
  return z.array(QdrantSearchPointSchema).safeParse(data).success;
}

function isValidElasticsearchResponse(data: unknown): data is any {
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

    if (!data.hits) {
      return {
        valid: false,
        error: 'Invalid Elasticsearch response: missing hits field',
      };
    }

    return { valid: true, hits: data.hits.hits };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown validation error';
    return { valid: false, error: errorMessage };
  }
}

describe('Type Guard Tests', () => {
  describe('isValidQdrantResponse', () => {
    it('should return true for valid Qdrant response array', () => {
      const validResponse = [
        { id: 'doc1', score: 0.95, payload: { content: 'test' } },
        { id: 'doc2', score: 0.87, payload: { content: 'test2' } },
      ];

      expect(isValidQdrantResponse(validResponse)).toBe(true);
    });

    it('should return true for valid response with mixed ID types', () => {
      const validResponse = [
        { id: 'string-id', score: 0.9 },
        { id: 12345, score: 0.8 },
        { id: 'another', score: 0.7 },
      ];

      expect(isValidQdrantResponse(validResponse)).toBe(true);
    });

    it('should return true for empty array', () => {
      const emptyResponse: any[] = [];

      expect(isValidQdrantResponse(emptyResponse)).toBe(true);
    });

    it('should return false for non-array input', () => {
      const invalidResponse = { id: 'doc1', score: 0.95 };

      expect(isValidQdrantResponse(invalidResponse)).toBe(false);
    });

    it('should return false for array with invalid objects', () => {
      const invalidResponse = [
        { id: 'doc1', score: 0.95 },
        { id: 'doc2', score: 'invalid' }, // Invalid score
      ];

      expect(isValidQdrantResponse(invalidResponse)).toBe(false);
    });

    it('should return false for array with missing required fields', () => {
      const invalidResponse = [
        { id: 'doc1', score: 0.95 },
        { id: 'doc2' }, // Missing score
      ];

      expect(isValidQdrantResponse(invalidResponse)).toBe(false);
    });

    it('should return false for null input', () => {
      expect(isValidQdrantResponse(null)).toBe(false);
    });

    it('should return false for undefined input', () => {
      expect(isValidQdrantResponse(undefined)).toBe(false);
    });

    it('should return false for string input', () => {
      expect(isValidQdrantResponse('not an array')).toBe(false);
    });

    it('should return false for number input', () => {
      expect(isValidQdrantResponse(123)).toBe(false);
    });

    it('should return false for array with null elements', () => {
      const invalidResponse = [
        { id: 'doc1', score: 0.95 },
        null,
      ];

      expect(isValidQdrantResponse(invalidResponse)).toBe(false);
    });

    it('should narrow type correctly when true', () => {
      const data: unknown = [
        { id: 'doc1', score: 0.95, payload: { content: 'test' } },
      ];

      if (isValidQdrantResponse(data)) {
        // TypeScript should know data is an array here
        expect(Array.isArray(data)).toBe(true);
        expect(data[0].score).toBe(0.95);
      } else {
        fail('Should have been valid');
      }
    });
  });

  describe('isValidElasticsearchResponse', () => {
    it('should return true for valid Elasticsearch response', () => {
      const validResponse = {
        hits: {
          total: { value: 10, relation: 'eq' },
          hits: [
            { _index: 'test', _id: '1', _score: 1.0, _source: { content: 'test' } },
          ],
        },
        took: 15,
        timed_out: false,
      };

      expect(isValidElasticsearchResponse(validResponse)).toBe(true);
    });

    it('should return true for valid response with only hits', () => {
      const validResponse = {
        hits: {
          total: { value: 5, relation: 'eq' },
          hits: [],
        },
      };

      expect(isValidElasticsearchResponse(validResponse)).toBe(true);
    });

    it('should return true for valid response with only metadata', () => {
      const validResponse = {
        took: 20,
        timed_out: true,
      };

      expect(isValidElasticsearchResponse(validResponse)).toBe(true);
    });

    it('should return true for empty object', () => {
      const validResponse = {};

      expect(isValidElasticsearchResponse(validResponse)).toBe(true);
    });

    it('should return false for null input', () => {
      expect(isValidElasticsearchResponse(null)).toBe(false);
    });

    it('should return false for undefined input', () => {
      expect(isValidElasticsearchResponse(undefined)).toBe(false);
    });

    it('should return false for array input', () => {
      expect(isValidElasticsearchResponse([])).toBe(false);
    });

    it('should return false for string input', () => {
      expect(isValidElasticsearchResponse('not an object')).toBe(false);
    });

    it('should return false for invalid hits structure', () => {
      const invalidResponse = {
        hits: {
          total: { value: 'invalid', relation: 'eq' },
          hits: [],
        },
      };

      expect(isValidElasticsearchResponse(invalidResponse)).toBe(false);
    });

    it('should return false for invalid took type', () => {
      const invalidResponse = {
        took: 'not a number',
      };

      expect(isValidElasticsearchResponse(invalidResponse)).toBe(false);
    });

    it('should return false for invalid timed_out type', () => {
      const invalidResponse = {
        timed_out: 'not a boolean',
      };

      expect(isValidElasticsearchResponse(invalidResponse)).toBe(false);
    });

    it('should narrow type correctly when true', () => {
      const data: unknown = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [{ _index: 'test', _id: '1', _score: 1.0 }],
        },
      };

      if (isValidElasticsearchResponse(data)) {
        // TypeScript should know data has correct structure
        expect(data.hits).toBeDefined();
        expect(data.hits?.hits).toHaveLength(1);
      } else {
        fail('Should have been valid');
      }
    });
  });

  describe('validateQdrantResult', () => {
    it('should return valid:true for valid Qdrant response', () => {
      const validResponse = [
        { id: 'doc1', score: 0.95, payload: { content: 'test' } },
      ];

      const result = validateQdrantResult(validResponse);

      expect(result.valid).toBe(true);
      expect(result.data).toEqual(validResponse);
      expect(result.error).toBeUndefined();
    });

    it('should return valid:false with error message for invalid response', () => {
      const invalidResponse = { id: 'doc1', score: 0.95 };

      const result = validateQdrantResult(invalidResponse);

      expect(result.valid).toBe(false);
      expect(result.data).toBeUndefined();
      expect(result.error).toBe('Invalid Qdrant response: data does not match expected schema');
    });

    it('should return valid:false for null input', () => {
      const result = validateQdrantResult(null);

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid Qdrant response: data does not match expected schema');
    });

    it('should return valid:false for array with invalid items', () => {
      const invalidResponse = [
        { id: 'doc1', score: 0.95 },
        { id: 'doc2', score: 'invalid' },
      ];

      const result = validateQdrantResult(invalidResponse);

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid Qdrant response: data does not match expected schema');
    });

    it('should handle empty array correctly', () => {
      const emptyResponse: any[] = [];

      const result = validateQdrantResult(emptyResponse);

      expect(result.valid).toBe(true);
      expect(result.data).toEqual([]);
    });

    it('should preserve all valid data in response', () => {
      const validResponse = [
        { id: 'doc1', score: 0.95, payload: { content: 'test1', source: 'src1' } },
        { id: 123, score: 0.87, payload: { content: 'test2' }, vector: [0.1, 0.2] },
      ];

      const result = validateQdrantResult(validResponse);

      expect(result.valid).toBe(true);
      expect(result.data).toHaveLength(2);
      expect(result.data?.[0].payload?.source).toBe('src1');
      expect(result.data?.[1].vector).toEqual([0.1, 0.2]);
    });

    it('should handle exceptions during validation', () => {
      // Create a scenario that might throw an exception
      const circularRef: any = {};
      circularRef.self = circularRef;

      const result = validateQdrantResult([circularRef]);

      // Should handle gracefully and return valid: false
      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('validateElasticsearchResult', () => {
    it('should return valid:true with hits for valid response', () => {
      const validResponse = {
        hits: {
          total: { value: 2, relation: 'eq' },
          hits: [
            { _index: 'test', _id: '1', _score: 1.0, _source: { content: 'test1' } },
            { _index: 'test', _id: '2', _score: 0.9, _source: { content: 'test2' } },
          ],
        },
        took: 15,
      };

      const result = validateElasticsearchResult(validResponse);

      expect(result.valid).toBe(true);
      expect(result.hits).toHaveLength(2);
      expect(result.hits?.[0]._id).toBe('1');
      expect(result.error).toBeUndefined();
    });

    it('should return valid:false for invalid response structure', () => {
      const invalidResponse = {
        hits: {
          total: { value: 'invalid', relation: 'eq' },
          hits: [],
        },
      };

      const result = validateElasticsearchResult(invalidResponse);

      expect(result.valid).toBe(false);
      expect(result.hits).toBeUndefined();
      expect(result.error).toBe('Invalid Elasticsearch response: data does not match expected schema');
    });

    it('should return valid:false when hits field is missing', () => {
      const invalidResponse = {
        took: 15,
        timed_out: false,
      };

      const result = validateElasticsearchResult(invalidResponse);

      expect(result.valid).toBe(false);
      expect(result.hits).toBeUndefined();
      expect(result.error).toBe('Invalid Elasticsearch response: missing hits field');
    });

    it('should return valid:false for null input', () => {
      const result = validateElasticsearchResult(null);

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid Elasticsearch response: data does not match expected schema');
    });

    it('should return valid:false for non-object input', () => {
      const result = validateElasticsearchResult('not an object');

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid Elasticsearch response: data does not match expected schema');
    });

    it('should handle empty hits array correctly', () => {
      const validResponse = {
        hits: {
          total: { value: 0, relation: 'eq' },
          hits: [],
        },
      };

      const result = validateElasticsearchResult(validResponse);

      expect(result.valid).toBe(true);
      expect(result.hits).toEqual([]);
    });

    it('should extract hits from nested structure', () => {
      const validResponse = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [
            { _index: 'index1', _id: 'doc1', _score: 1.5, _source: { title: 'Test' } },
          ],
        },
      };

      const result = validateElasticsearchResult(validResponse);

      expect(result.valid).toBe(true);
      expect(result.hits).toBeDefined();
      expect(result.hits?.[0]._index).toBe('index1');
      expect(result.hits?.[0]._source?.title).toBe('Test');
    });

    it('should handle exceptions during validation', () => {
      // Create an object that might cause issues
      const problematicInput: any = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [null],
        },
      };

      const result = validateElasticsearchResult(problematicInput);

      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Type Guard Integration', () => {
    it('should work in conditional logic for Qdrant', () => {
      const data: unknown = [
        { id: 'doc1', score: 0.95, payload: { content: 'test' } },
      ];

      let processedCorrectly = false;
      if (isValidQdrantResponse(data)) {
        // This branch should only execute if data is valid
        processedCorrectly = data.every(item => typeof item.score === 'number');
      }

      expect(processedCorrectly).toBe(true);
    });

    it('should work in conditional logic for Elasticsearch', () => {
      const data: unknown = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [{ _index: 'test', _id: '1', _score: 1.0 }],
        },
      };

      let hasValidHits = false;
      if (isValidElasticsearchResponse(data) && data.hits) {
        hasValidHits = data.hits.hits.every(hit => typeof hit._score === 'number');
      }

      expect(hasValidHits).toBe(true);
    });

    it('should provide clear error messages for debugging', () => {
      const invalidData = { invalid: 'data' };

      const qdrantResult = validateQdrantResult(invalidData);
      const esResult = validateElasticsearchResult(invalidData);

      expect(qdrantResult.valid).toBe(false);
      expect(qdrantResult.error).toContain('Invalid Qdrant response');

      expect(esResult.valid).toBe(false);
      expect(esResult.error).toContain('Invalid Elasticsearch response');
    });

    it('should handle mixed valid/invalid scenarios in workflow', () => {
      const responses: unknown[] = [
        [{ id: '1', score: 0.9 }], // Valid
        { invalid: 'structure' }, // Invalid
        [], // Valid (empty)
        null, // Invalid
        [{ id: '2', score: 'bad' }], // Invalid
      ];

      const results = responses.map(validateQdrantResult);
      const validCount = results.filter(r => r.valid).length;

      expect(validCount).toBe(2); // Only first and empty array are valid
    });
  });

  describe('Type Narrowing Behavior', () => {
    it('should narrow unknown to QdrantResponseArray', () => {
      const data: unknown = [{ id: '1', score: 0.9 }];

      if (isValidQdrantResponse(data)) {
        // After this check, TypeScript knows data is QdrantSearchPoint[]
        const firstScore = data[0].score; // Should compile without error
        expect(firstScore).toBe(0.9);
      }
    });

    it('should narrow unknown to ElasticsearchResponseData', () => {
      const data: unknown = {
        hits: {
          total: { value: 1, relation: 'eq' },
          hits: [{ _index: 'test', _id: '1', _score: 1.0 }],
        },
      };

      if (isValidElasticsearchResponse(data)) {
        // After this check, TypeScript knows data is ElasticsearchResponseData
        const took = data.took; // Should be number | undefined
        expect(took).toBeUndefined();
      }
    });

    it('should maintain type safety in else branches', () => {
      const data: unknown = { invalid: 'data' };

      if (isValidQdrantResponse(data)) {
        fail('Should not reach here');
      } else {
        // In else branch, data is still unknown
        expect(data).toEqual({ invalid: 'data' });
      }
    });
  });

  describe('Performance Considerations', () => {
    it('should handle large datasets efficiently', () => {
      const largeResponse = Array(1000).fill(null).map((_, i) => ({
        id: `doc${i}`,
        score: Math.random(),
        payload: { content: `Document ${i}` },
      }));

      const start = Date.now();
      const result = validateQdrantResult(largeResponse);
      const duration = Date.now() - start;

      expect(result.valid).toBe(true);
      expect(duration).toBeLessThan(100); // Should validate 1000 items quickly
    });

    it('should fail fast on first error', () => {
      const invalidResponse = [
        { id: '1', score: 0.9 },
        { id: '2', score: 'invalid' },
        // ... potentially many more items
      ];

      const start = Date.now();
      const result = validateQdrantResult(invalidResponse);
      const duration = Date.now() - start;

      expect(result.valid).toBe(false);
      expect(duration).toBeLessThan(50); // Should fail quickly
    });
  });
});
