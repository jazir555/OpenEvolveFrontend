"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const canonical_1 = require("../src/canonical");
(0, globals_1.describe)('Canonical Schema Contract Tests', () => {
    (0, globals_1.describe)('UnifiedKnowledgeQuery Schema', () => {
        (0, globals_1.test)('should validate valid query with all fields', () => {
            const validQuery = {
                query: 'test query',
                domains: ['all'],
                queryType: 'hybrid',
                temporalFilter: {
                    startDate: '2024-01-01T00:00:00Z',
                    endDate: '2024-12-31T23:59:59Z',
                },
                knowledgeTypes: ['document', 'entity'],
                maxResults: 50,
                minConfidence: 0.5,
                maxDepth: 2,
                includeMetadata: true,
                correlationId: '550e8400-e29b-41d4-a716-446655440000',
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(validQuery);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.query).toBe('test query');
                (0, globals_1.expect)(result.data.domains).toEqual(['all']);
                (0, globals_1.expect)(result.data.maxResults).toBe(50);
            }
        });
        (0, globals_1.test)('should validate query with minimal required fields', () => {
            const minimalQuery = {
                query: 'minimal query',
                domains: ['ragbits'],
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(minimalQuery);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.queryType).toBe('hybrid'); // default
                (0, globals_1.expect)(result.data.maxResults).toBe(50); // default
                (0, globals_1.expect)(result.data.minConfidence).toBe(0.0); // default
            }
        });
        (0, globals_1.test)('should reject query without query text', () => {
            const invalidQuery = {
                domains: ['all'],
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should reject query with invalid maxResults', () => {
            const invalidQuery = {
                query: 'test',
                domains: ['all'],
                maxResults: 2000, // exceeds max of 1000
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should reject query with invalid confidence', () => {
            const invalidQuery = {
                query: 'test',
                domains: ['all'],
                minConfidence: 1.5, // exceeds max of 1.0
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should reject invalid domain', () => {
            const invalidQuery = {
                query: 'test',
                domains: ['invalid-domain'],
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should reject invalid temporal filter (non-ISO date)', () => {
            const invalidQuery = {
                query: 'test',
                domains: ['all'],
                temporalFilter: {
                    startDate: 'not-a-date',
                },
            };
            const result = canonical_1.UnifiedKnowledgeQuerySchema.safeParse(invalidQuery);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('UnifiedQueryResult Schema', () => {
        (0, globals_1.test)('should validate valid result with all fields', () => {
            const validResult = {
                query: 'test query',
                results: [
                    {
                        content: 'test content',
                        source: 'ragbits',
                        id: 'test-id',
                        type: 'document',
                        confidence: 0.8,
                        relevance: 0.9,
                        timestamp: '2024-01-01T00:00:00Z',
                        metadata: { key: 'value' },
                    },
                ],
                sources: [
                    {
                        system: 'ragbits',
                        queryTimeMs: 100,
                        resultCount: 1,
                        success: true,
                    },
                ],
                confidence: 0.8,
                executionTimeMs: 200,
                correlationId: '550e8400-e29b-41d4-a716-446655440000',
            };
            const result = canonical_1.UnifiedQueryResultSchema.safeParse(validResult);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.results).toHaveLength(1);
                (0, globals_1.expect)(result.data.sources).toHaveLength(1);
                (0, globals_1.expect)(result.data.confidence).toBe(0.8);
            }
        });
        (0, globals_1.test)('should reject result without query', () => {
            const invalidResult = {
                results: [],
                sources: [],
                confidence: 0,
                executionTimeMs: 0,
                correlationId: '550e8400-e29b-41d4-a716-446655440000',
            };
            const result = canonical_1.UnifiedQueryResultSchema.safeParse(invalidResult);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should reject result with invalid confidence range', () => {
            const invalidResult = {
                query: 'test',
                results: [],
                sources: [],
                confidence: 1.5, // exceeds 1.0
                executionTimeMs: 0,
                correlationId: '550e8400-e29b-41d4-a716-446655440000',
            };
            const result = canonical_1.UnifiedQueryResultSchema.safeParse(invalidResult);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('KnowledgeItem Schema', () => {
        (0, globals_1.test)('should validate valid knowledge item', () => {
            const validItem = {
                content: 'test content',
                source: 'graphiti',
                id: 'item-123',
                type: 'entity',
                confidence: 0.7,
                relevance: 0.8,
                timestamp: '2024-01-01T00:00:00Z',
            };
            const result = canonical_1.KnowledgeItemSchema.safeParse(validItem);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.source).toBe('graphiti');
                (0, globals_1.expect)(result.data.type).toBe('entity');
            }
        });
        (0, globals_1.test)('should reject item with confidence out of range', () => {
            const invalidItem = {
                content: 'test',
                source: 'vectordb',
                id: 'item-123',
                type: 'document',
                confidence: -0.1, // negative
                relevance: 0.8,
                timestamp: '2024-01-01T00:00:00Z',
            };
            const result = canonical_1.KnowledgeItemSchema.safeParse(invalidItem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Enum Validation', () => {
        (0, globals_1.test)('should accept valid knowledge domains', () => {
            const domains = ['ragbits', 'graphiti', 'vectordb', 'all'];
            for (const domain of domains) {
                const result = canonical_1.KnowledgeDomainSchema.safeParse(domain);
                (0, globals_1.expect)(result.success).toBe(true);
            }
        });
        (0, globals_1.test)('should reject invalid knowledge domain', () => {
            const result = canonical_1.KnowledgeDomainSchema.safeParse('invalid');
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should accept valid knowledge types', () => {
            const types = ['document', 'entity', 'proof', 'code', 'relationship', 'all'];
            for (const type of types) {
                const result = canonical_1.KnowledgeTypeSchema.safeParse(type);
                (0, globals_1.expect)(result.success).toBe(true);
            }
        });
        (0, globals_1.test)('should reject invalid knowledge type', () => {
            const result = canonical_1.KnowledgeTypeSchema.safeParse('invalid');
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should accept valid query types', () => {
            const types = ['semantic-search', 'temporal-query', 'graph-traversal', 'hybrid', 'fallback'];
            for (const type of types) {
                const result = canonical_1.QueryTypeSchema.safeParse(type);
                (0, globals_1.expect)(result.success).toBe(true);
            }
        });
        (0, globals_1.test)('should reject invalid query type', () => {
            const result = canonical_1.QueryTypeSchema.safeParse('invalid');
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('should accept valid system sources', () => {
            const sources = ['ragbits', 'graphiti', 'vectordb', 'fused'];
            for (const source of sources) {
                const result = canonical_1.SystemSourceSchema.safeParse(source);
                (0, globals_1.expect)(result.success).toBe(true);
            }
        });
        (0, globals_1.test)('should reject invalid system source', () => {
            const result = canonical_1.SystemSourceSchema.safeParse('invalid');
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Validation Helpers', () => {
        (0, globals_1.test)('validateQuery should return parsed query', () => {
            const query = {
                query: 'test',
                domains: ['all'],
            };
            const result = (0, canonical_1.validateQuery)(query);
            (0, globals_1.expect)(result.query).toBe('test');
            (0, globals_1.expect)(result.domains).toEqual(['all']);
        });
        (0, globals_1.test)('validateQuery should throw on invalid input', () => {
            const invalidQuery = {
                domains: ['all'],
                // missing query
            };
            (0, globals_1.expect)(() => (0, canonical_1.validateQuery)(invalidQuery)).toThrow();
        });
        (0, globals_1.test)('validateResult should return parsed result', () => {
            const result = {
                query: 'test',
                results: [],
                sources: [],
                confidence: 0,
                executionTimeMs: 0,
                correlationId: '550e8400-e29b-41d4-a716-446655440000',
            };
            const parsed = (0, canonical_1.validateResult)(result);
            (0, globals_1.expect)(parsed.query).toBe('test');
        });
        (0, globals_1.test)('validateResult should throw on invalid input', () => {
            const invalidResult = {
            // missing required fields
            };
            (0, globals_1.expect)(() => (0, canonical_1.validateResult)(invalidResult)).toThrow();
        });
    });
    (0, globals_1.describe)('Type Guards', () => {
        (0, globals_1.test)('isValidQuery should return true for valid query', () => {
            const query = {
                query: 'test',
                domains: ['all'],
            };
            (0, globals_1.expect)(isValidQuery(query)).toBe(true);
        });
        (0, globals_1.test)('isValidQuery should return false for invalid query', () => {
            const invalidQuery = {
                domains: ['all'],
            };
            (0, globals_1.expect)(isValidQuery(invalidQuery)).toBe(false);
        });
        (0, globals_1.test)('isValidResult should return true for valid result', () => {
            const result = {
                query: 'test',
                results: [],
                sources: [],
                confidence: 0,
                executionTimeMs: 0,
                correlationId: '550e8400-e29b-41d4-a716-446655440000',
            };
            (0, globals_1.expect)(isValidResult(result)).toBe(true);
        });
        (0, globals_1.test)('isValidResult should return false for invalid result', () => {
            const invalidResult = {
                query: 'test',
                // missing required fields
            };
            (0, globals_1.expect)(isValidResult(invalidResult)).toBe(false);
        });
    });
    (0, globals_1.describe)('UTC Timestamp Compliance', () => {
        (0, globals_1.test)('should require UTC ISO-8601 timestamps', () => {
            const validTimestamps = [
                '2024-01-01T00:00:00Z',
                '2024-12-31T23:59:59.999Z',
                '2024-06-15T10:30:45.123Z',
            ];
            for (const timestamp of validTimestamps) {
                const item = {
                    content: 'test',
                    source: 'ragbits',
                    id: 'test-id',
                    type: 'document',
                    confidence: 0.5,
                    relevance: 0.5,
                    timestamp,
                };
                const result = canonical_1.KnowledgeItemSchema.safeParse(item);
                (0, globals_1.expect)(result.success).toBe(true);
            }
        });
        (0, globals_1.test)('should reject non-UTC timestamps', () => {
            const invalidTimestamps = [
                '2024-01-01T00:00:00+05:00', // timezone offset
                '2024-01-01 00:00:00', // missing T and Z
                '2024-01-01', // date only
                'not-a-date',
            ];
            for (const timestamp of invalidTimestamps) {
                const item = {
                    content: 'test',
                    source: 'ragbits',
                    id: 'test-id',
                    type: 'document',
                    confidence: 0.5,
                    relevance: 0.5,
                    timestamp,
                };
                const result = canonical_1.KnowledgeItemSchema.safeParse(item);
                (0, globals_1.expect)(result.success).toBe(false);
            }
        });
    });
});
//# sourceMappingURL=contract.test.js.map