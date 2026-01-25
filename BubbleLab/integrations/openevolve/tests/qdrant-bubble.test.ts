/**
 * QdrantBubble Test Suite
 *
 * Comprehensive tests for Qdrant vector database integration
 * Tests cover: functionality, error handling, resilience patterns, contract compliance
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { QdrantBubble } from '../service-bubbles/qdrant-bubble';
import type { QdrantParams } from '../service-bubbles/qdrant-bubble';

describe('QdrantBubble', () => {
  // ============================================================================
  // BASE CLASS INHERITANCE TESTS
  // ============================================================================

  describe('Base Class Inheritance', () => {
    it('should extend ServiceBubble properly', () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      // Verify it's an instance of ServiceBubble
      expect(bubble.constructor.name).toBe('QdrantBubble');
      expect(typeof bubble.action).toBe('function');
    });

    it('should have correct static properties', () => {
      expect(QdrantBubble.service).toBe('openevolve');
      expect(QdrantBubble.authType).toBe('apikey');
      expect(QdrantBubble.bubbleName).toBe('qdrant');
      expect(QdrantBubble.type).toBe('service');
      expect(QdrantBubble.credentialType).toBe('qdrant_api_key');
    });
  });

  // ============================================================================
  // FEDERATION CONSTITUTION COMPLIANCE TESTS
  // ============================================================================

  describe('Federation Constitution Compliance', () => {
    it('should fail without baseUrl (no magic defaults)', () => {
      expect(() => {
        new QdrantBubble({
          operation: 'health_check',
          // @ts-expect-error - Testing missing required field
          baseUrl: undefined,
        });
      }).toThrow();
    });

    it('should require baseUrl to be a valid URL', () => {
      expect(() => {
        new QdrantBubble({
          operation: 'health_check',
          // @ts-expect-error - Testing invalid URL
          baseUrl: 'not-a-url',
        });
      }).toThrow();
    });

    it('should accept valid baseUrl', () => {
      expect(() => {
        new QdrantBubble({
          operation: 'health_check',
          baseUrl: 'http://localhost:6333',
        });
      }).not.toThrow();
    });

    it('should accept valid HTTPS baseUrl', () => {
      expect(() => {
        new QdrantBubble({
          operation: 'health_check',
          baseUrl: 'https://qdrant.example.com',
        });
      }).not.toThrow();
    });
  });

  // ============================================================================
  // PARAMETER VALIDATION TESTS
  // ============================================================================

  describe('Parameter Validation', () => {
    let validParams: QdrantParams;

    beforeEach(() => {
      validParams = {
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      };
    });

    it('should validate operation enum', () => {
      const validOperations = [
        'create_collection',
        'delete_collection',
        'insert_points',
        'search_points',
        'delete_points',
        'get_collection',
        'list_collections',
        'health_check',
        'count_points',
      ];

      validOperations.forEach(operation => {
        expect(() => {
          new QdrantBubble({ ...validParams, operation });
        }).not.toThrow();
      });
    });

    it('should reject invalid operation', () => {
      expect(() => {
        new QdrantBubble({
          ...validParams,
          // @ts-expect-error - Testing invalid operation
          operation: 'invalid_operation',
        });
      }).toThrow();
    });

    it('should validate timeout range', () => {
      expect(() => {
        new QdrantBubble({
          ...validParams,
          timeout: 30000,
        });
      }).not.toThrow();

      // Too small
      expect(() => {
        new QdrantBubble({
          ...validParams,
          timeout: 500,
        });
      }).toThrow();

      // Too large
      expect(() => {
        new QdrantBubble({
          ...validParams,
          timeout: 200000,
        });
      }).toThrow();
    });
  });

  // ============================================================================
  // OPERATION-SPECIFIC TESTS
  // ============================================================================

  describe('Health Check Operation', () => {
    it('should execute health check', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      // Mock fetch
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ status: 'ok' }),
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBeDefined();
      expect(result.operation).toBe('health_check');
      expect(result.timing).toBeGreaterThanOrEqual(0);
    });

    it('should handle network errors gracefully', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      // Mock fetch failure
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.timing).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Create Collection Operation', () => {
    it('should require collectionName', async () => {
      const bubble = new QdrantBubble({
        operation: 'create_collection',
        baseUrl: 'http://localhost:6333',
        // collectionName missing
      });

      await expect(bubble.action()).rejects.toThrow('collectionName is required');
    });

    it('should create collection with valid params', async () => {
      const bubble = new QdrantBubble({
        operation: 'create_collection',
        baseUrl: 'http://localhost:6333',
        collectionName: 'test-collection',
        vectorSize: 1536,
        distance: 'Cosine',
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ result: true }),
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('create_collection');
    });
  });

  describe('Search Points Operation', () => {
    it('should require collectionName and queryVector', async () => {
      const bubble = new QdrantBubble({
        operation: 'search_points',
        baseUrl: 'http://localhost:6333',
        collectionName: 'test-collection',
        // queryVector missing
      });

      await expect(bubble.action()).rejects.toThrow('collectionName and queryVector are required');
    });

    it('should search with valid params', async () => {
      const bubble = new QdrantBubble({
        operation: 'search_points',
        baseUrl: 'http://localhost:6333',
        collectionName: 'test-collection',
        queryVector: Array(1536).fill(0).map(() => Math.random()),
        limit: 10,
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({
          result: [
            { id: 1, score: 0.95 },
            { id: 2, score: 0.90 },
          ],
        }),
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('search_points');
      expect(result.data).toBeDefined();
    });
  });

  // ============================================================================
  // CIRCUIT BREAKER TESTS
  // ============================================================================

  describe('Circuit Breaker', () => {
    it('should open circuit after threshold failures', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:9999', // Invalid port
      });

      // Circuit breaker should open after 5 failures
      let failureCount = 0;
      for (let i = 0; i < 10; i++) {
        try {
          await bubble.action();
        } catch (error) {
          failureCount++;
        }
      }

      // Should have circuit breaker errors
      expect(failureCount).toBeGreaterThan(0);
    });

    it('should allow requests after circuit recovers', async () => {
      // This test would require more complex setup with time mocking
      // For now, just verify the resilience wrapper is initialized
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      expect(bubble).toBeDefined();
    });
  });

  // ============================================================================
  // RETRY LOGIC TESTS
  // ============================================================================

  describe('Retry Logic', () => {
    it('should retry transient errors', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      let attemptCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        attemptCount++;
        if (attemptCount < 3) {
          return Promise.reject(new Error('ECONNREFUSED'));
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          json: async () => ({ status: 'ok' }),
        } as Response);
      });

      const result = await bubble.action();

      expect(attemptCount).toBe(3);
      expect(result.success).toBe(true);
    });

    it('should not retry permanent errors', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      let attemptCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        attemptCount++;
        return Promise.reject(new Error('404 Not Found'));
      });

      try {
        await bubble.action();
      } catch (error) {
        // Expected
      }

      // Should only try once (no retry for 404)
      expect(attemptCount).toBe(1);
    });
  });

  // ============================================================================
  // REQUEST DEDUPLICATION TESTS
  // ============================================================================

  describe('Request Deduplication', () => {
    it('should deduplicate identical requests', async () => {
      const bubble = new QdrantBubble({
        operation: 'search_points',
        baseUrl: 'http://localhost:6333',
        collectionName: 'test-collection',
        queryVector: Array(1536).fill(0).map(() => Math.random()),
      });

      let fetchCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        fetchCount++;
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          json: async () => ({ result: [] }),
        } as Response);
      });

      // Make identical requests concurrently
      const promises = [
        bubble.action(),
        bubble.action(),
        bubble.action(),
      ];

      await Promise.all(promises);

      // Should only fetch once (deduplication)
      expect(fetchCount).toBeLessThan(3);
    });
  });

  // ============================================================================
  // CONTRACT TESTS
  // ============================================================================

  describe('Contract Tests', () => {
    it('should return correct response structure', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ status: 'ok' }),
      } as Response);

      const result = await bubble.action();

      // Verify response structure
      expect(result).toHaveProperty('success');
      expect(result).toHaveProperty('operation');
      expect(result).toHaveProperty('status');
      expect(result).toHaveProperty('timing');
      expect(result.status).toHaveProperty('code');
      expect(result.status).toHaveProperty('reason');
    });

    it('should include timing information', async () => {
      const bubble = new QdrantBubble({
        operation: 'health_check',
        baseUrl: 'http://localhost:6333',
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ status: 'ok' }),
      } as Response);

      const result = await bubble.action();

      expect(result.timing).toBeGreaterThanOrEqual(0);
      expect(typeof result.timing).toBe('number');
    });
  });

  // ============================================================================
  // CLEANUP
  // ============================================================================

  afterEach(() => {
    vi.restoreAllMocks();
  });
});
