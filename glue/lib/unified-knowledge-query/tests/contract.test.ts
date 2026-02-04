/**
 * Contract Tests for Unified Knowledge Query Interface
 *
 * Federation Constitution - Phase 2: The Contract (Defense)
 * "Protecting the Mega-Project from Updates"
 *
 * These tests verify that the API returns the specific fields we rely on.
 * This test runs on container startup. If the contract is violated,
 * the adapter refuses to start to prevent data corruption.
 */

import { describe, test, expect } from '@jest/globals';
import {
  validateQuery,
  validateResult,
  UnifiedKnowledgeQuerySchema,
  UnifiedQueryResultSchema,
  KnowledgeDomainSchema,
  KnowledgeTypeSchema,
  QueryTypeSchema,
  SystemSourceSchema,
  KnowledgeItemSchema,
} from '../src/canonical';

describe('Canonical Schema Contract Tests', () => {
  describe('UnifiedKnowledgeQuery Schema', () => {
    test('should validate valid query with all fields', () => {
      const validQuery = {
        query: 'test query',
        domains: ['all'] as const,
        queryType: 'hybrid' as const,
        temporalFilter: {
          startDate: '2024-01-01T00:00:00Z',
          endDate: '2024-12-31T23:59:59Z',
        },
        knowledgeTypes: ['document', 'entity'] as const,
        maxResults: 50,
        minConfidence: 0.5,
        maxDepth: 2,
        includeMetadata: true,
        correlationId: '550e8400-e29b-41d4-a716-446655440000',
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(validQuery);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.query).toBe('test query');
        expect(result.data.domains).toEqual(['all']);
        expect(result.data.maxResults).toBe(50);
      }
    });

    test('should validate query with minimal required fields', () => {
      const minimalQuery = {
        query: 'minimal query',
        domains: ['ragbits'] as const,
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(minimalQuery);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.queryType).toBe('hybrid'); // default
        expect(result.data.maxResults).toBe(50); // default
        expect(result.data.minConfidence).toBe(0.0); // default
      }
    });

    test('should reject query without query text', () => {
      const invalidQuery = {
        domains: ['all'] as const,
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
      expect(result.success).toBe(false);
    });

    test('should reject query with invalid maxResults', () => {
      const invalidQuery = {
        query: 'test',
        domains: ['all'] as const,
        maxResults: 2000, // exceeds max of 1000
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
      expect(result.success).toBe(false);
    });

    test('should reject query with invalid confidence', () => {
      const invalidQuery = {
        query: 'test',
        domains: ['all'] as const,
        minConfidence: 1.5, // exceeds max of 1.0
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
      expect(result.success).toBe(false);
    });

    test('should reject invalid domain', () => {
      const invalidQuery = {
        query: 'test',
        domains: ['invalid-domain'] as any,
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
      expect(result.success).toBe(false);
    });

    test('should reject invalid temporal filter (non-ISO date)', () => {
      const invalidQuery = {
        query: 'test',
        domains: ['all'] as const,
        temporalFilter: {
          startDate: 'not-a-date',
        },
      };

      const result = UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
      expect(result.success).toBe(false);
    });
  });

  describe('UnifiedQueryResult Schema', () => {
    test('should validate valid result with all fields', () => {
      const validResult = {
        query: 'test query',
        results: [
          {
            content: 'test content',
            source: 'ragbits' as const,
            id: 'test-id',
            type: 'document' as const,
            confidence: 0.8,
            relevance: 0.9,
            timestamp: '2024-01-01T00:00:00Z',
            metadata: { key: 'value' },
          },
        ],
        sources: [
          {
            system: 'ragbits' as const,
            queryTimeMs: 100,
            resultCount: 1,
            success: true,
          },
        ],
        confidence: 0.8,
        executionTimeMs: 200,
        correlationId: '550e8400-e29b-41d4-a716-446655440000',
      };

      const result = UnifiedQueryResultSchema.safeParse(validResult);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.results).toHaveLength(1);
        expect(result.data.sources).toHaveLength(1);
        expect(result.data.confidence).toBe(0.8);
      }
    });

    test('should reject result without query', () => {
      const invalidResult = {
        results: [],
        sources: [],
        confidence: 0,
        executionTimeMs: 0,
        correlationId: '550e8400-e29b-41d4-a716-446655440000',
      };

      const result = UnifiedQueryResultSchema.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });

    test('should reject result with invalid confidence range', () => {
      const invalidResult = {
        query: 'test',
        results: [],
        sources: [],
        confidence: 1.5, // exceeds 1.0
        executionTimeMs: 0,
        correlationId: '550e8400-e29b-41d4-a716-446655440000',
      };

      const result = UnifiedQueryResultSchema.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });
  });

  describe('KnowledgeItem Schema', () => {
    test('should validate valid knowledge item', () => {
      const validItem = {
        content: 'test content',
        source: 'graphiti' as const,
        id: 'item-123',
        type: 'entity' as const,
        confidence: 0.7,
        relevance: 0.8,
        timestamp: '2024-01-01T00:00:00Z',
      };

      const result = KnowledgeItemSchema.safeParse(validItem);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.source).toBe('graphiti');
        expect(result.data.type).toBe('entity');
      }
    });

    test('should reject item with confidence out of range', () => {
      const invalidItem = {
        content: 'test',
        source: 'vectordb' as const,
        id: 'item-123',
        type: 'document' as const,
        confidence: -0.1, // negative
        relevance: 0.8,
        timestamp: '2024-01-01T00:00:00Z',
      };

      const result = KnowledgeItemSchema.safeParse(invalidItem);
      expect(result.success).toBe(false);
    });
  });

  describe('Enum Validation', () => {
    test('should accept valid knowledge domains', () => {
      const domains = ['ragbits', 'graphiti', 'vectordb', 'all'] as const;

      for (const domain of domains) {
        const result = KnowledgeDomainSchema.safeParse(domain);
        expect(result.success).toBe(true);
      }
    });

    test('should reject invalid knowledge domain', () => {
      const result = KnowledgeDomainSchema.safeParse('invalid');
      expect(result.success).toBe(false);
    });

    test('should accept valid knowledge types', () => {
      const types = ['document', 'entity', 'proof', 'code', 'relationship', 'all'] as const;

      for (const type of types) {
        const result = KnowledgeTypeSchema.safeParse(type);
        expect(result.success).toBe(true);
      }
    });

    test('should reject invalid knowledge type', () => {
      const result = KnowledgeTypeSchema.safeParse('invalid');
      expect(result.success).toBe(false);
    });

    test('should accept valid query types', () => {
      const types = ['semantic-search', 'temporal-query', 'graph-traversal', 'hybrid', 'fallback'] as const;

      for (const type of types) {
        const result = QueryTypeSchema.safeParse(type);
        expect(result.success).toBe(true);
      }
    });

    test('should reject invalid query type', () => {
      const result = QueryTypeSchema.safeParse('invalid');
      expect(result.success).toBe(false);
    });

    test('should accept valid system sources', () => {
      const sources = ['ragbits', 'graphiti', 'vectordb', 'fused'] as const;

      for (const source of sources) {
        const result = SystemSourceSchema.safeParse(source);
        expect(result.success).toBe(true);
      }
    });

    test('should reject invalid system source', () => {
      const result = SystemSourceSchema.safeParse('invalid');
      expect(result.success).toBe(false);
    });
  });

  describe('Validation Helpers', () => {
    test('validateQuery should return parsed query', () => {
      const query = {
        query: 'test',
        domains: ['all'] as const,
      };

      const result = validateQuery(query);
      expect(result.query).toBe('test');
      expect(result.domains).toEqual(['all']);
    });

    test('validateQuery should throw on invalid input', () => {
      const invalidQuery = {
        domains: ['all'] as const,
        // missing query
      };

      expect(() => validateQuery(invalidQuery)).toThrow();
    });

    test('validateResult should return parsed result', () => {
      const result = {
        query: 'test',
        results: [],
        sources: [],
        confidence: 0,
        executionTimeMs: 0,
        correlationId: '550e8400-e29b-41d4-a716-446655440000',
      };

      const parsed = validateResult(result);
      expect(parsed.query).toBe('test');
    });

    test('validateResult should throw on invalid input', () => {
      const invalidResult = {
        // missing required fields
      };

      expect(() => validateResult(invalidResult)).toThrow();
    });
  });

  describe('Type Guards', () => {
    test('isValidQuery should return true for valid query', () => {
      const query = {
        query: 'test',
        domains: ['all'] as const,
      };

      expect(isValidQuery(query)).toBe(true);
    });

    test('isValidQuery should return false for invalid query', () => {
      const invalidQuery = {
        domains: ['all'] as const,
      };

      expect(isValidQuery(invalidQuery)).toBe(false);
    });

    test('isValidResult should return true for valid result', () => {
      const result = {
        query: 'test',
        results: [],
        sources: [],
        confidence: 0,
        executionTimeMs: 0,
        correlationId: '550e8400-e29b-41d4-a716-446655440000',
      };

      expect(isValidResult(result)).toBe(true);
    });

    test('isValidResult should return false for invalid result', () => {
      const invalidResult = {
        query: 'test',
        // missing required fields
      };

      expect(isValidResult(invalidResult)).toBe(false);
    });
  });

  describe('UTC Timestamp Compliance', () => {
    test('should require UTC ISO-8601 timestamps', () => {
      const validTimestamps = [
        '2024-01-01T00:00:00Z',
        '2024-12-31T23:59:59.999Z',
        '2024-06-15T10:30:45.123Z',
      ];

      for (const timestamp of validTimestamps) {
        const item = {
          content: 'test',
          source: 'ragbits' as const,
          id: 'test-id',
          type: 'document' as const,
          confidence: 0.5,
          relevance: 0.5,
          timestamp,
        };

        const result = KnowledgeItemSchema.safeParse(item);
        expect(result.success).toBe(true);
      }
    });

    test('should reject non-UTC timestamps', () => {
      const invalidTimestamps = [
        '2024-01-01T00:00:00+05:00', // timezone offset
        '2024-01-01 00:00:00', // missing T and Z
        '2024-01-01', // date only
        'not-a-date',
      ];

      for (const timestamp of invalidTimestamps) {
        const item = {
          content: 'test',
          source: 'ragbits' as const,
          id: 'test-id',
          type: 'document' as const,
          confidence: 0.5,
          relevance: 0.5,
          timestamp,
        };

        const result = KnowledgeItemSchema.safeParse(item);
        expect(result.success).toBe(false);
      }
    });
  });
});
