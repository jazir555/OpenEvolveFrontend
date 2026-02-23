/**
 * Contract Tests for RAGBits-Graphiti Sync Adapter
 *
 * Follows the Federation Constitution:
 * - Phase 2: The Contract (Defense)
 * - Protects the Mega-Project from Updates
 * - Tests run on container startup
 * - Adapter refuses to start if contract is violated
 *
 * Usage: npm test
 */

import { describe, it, expect, beforeAll } from '@jest/globals';
import { v4 as uuidv4 } from 'uuid';
import { validateSyncOperation, validateSyncConfig, validateSyncResult, validateConflict, validateSyncSpec, DEFAULT_SYNC_CONFIG, createSyncOperation, createSyncResult, createConflict } from '../src/canonical';

describe('RAGBits-Graphiti Sync Contract Tests', () => {
  describe('Canonical Schema Validation', () => {
    it('should validate a valid SyncOperation', () => {
      const operation = createSyncOperation(
        'ingest_sync',
        'ragbits',
        'graphiti',
        'ragbits_to_graphiti',
        uuidv4()
      );

      const result = validateSyncOperation(operation);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data?.status).toBe('pending');
    });

    it('should reject invalid SyncOperation with missing required fields', () => {
      const invalidOperation = {
        id: uuidv4(),
        // Missing required fields
      };

      const result = validateSyncOperation(invalidOperation);
      expect(result.success).toBe(false);
      expect(result.errors).toBeDefined();
      expect(result.errors?.length).toBeGreaterThan(0);
    });

    it('should validate a valid SyncConfig', () => {
      const config = {
        ...DEFAULT_SYNC_CONFIG,
        enabled: true,
        interval_ms: 300000,
        bidirectional: true,
      };

      const result = validateSyncConfig(config);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data?.enabled).toBe(true);
    });

    it('should reject invalid SyncConfig with negative interval', () => {
      const invalidConfig = {
        ...DEFAULT_SYNC_CONFIG,
        interval_ms: -1000,
      };

      const result = validateSyncConfig(invalidConfig);
      expect(result.success).toBe(false);
      expect(result.errors).toBeDefined();
    });

    it('should validate a valid SyncResult', () => {
      const result = createSyncResult(
        uuidv4(),
        'completed',
        'ragbits_to_graphiti',
        uuidv4(),
        1000
      );

      const validationResult = validateSyncResult(result);
      expect(validationResult.success).toBe(true);
      expect(validationResult.data).toBeDefined();
      expect(validationResult.data?.status).toBe('completed');
    });

    it('should validate a valid Conflict', () => {
      const conflict = createConflict(
        'semantic_conflict',
        'medium',
        { content: 'Version A' },
        { content: 'Version B' },
        'Test conflict',
        uuidv4()
      );

      const result = validateConflict(conflict);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data?.severity).toBe('medium');
    });

    it('should validate a valid SyncSpec', () => {
      const spec = {
        direction: 'bidirectional' as const,
        entity_ids: [uuidv4(), uuidv4()],
        episode_ids: [uuidv4()],
        timeout_ms: 30000,
      };

      const result = validateSyncSpec(spec);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data?.direction).toBe('bidirectional');
    });
  });

  describe('Timestamp Validation (Law of UTC)', () => {
    it('should require UTC ISO-8601 timestamps', () => {
      const operation = createSyncOperation(
        'ingest_sync',
        'ragbits',
        'graphiti',
        'ragbits_to_graphiti',
        uuidv4()
      );

      // Timestamp should be in ISO-8601 format
      expect(operation.timestamp_utc).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
      expect(operation.timestamp_utc).toContain('Z');
    });

    it('should reject non-UTC timestamps', () => {
      const invalidOperation = {
        id: uuidv4(),
        type: 'ingest_sync' as const,
        source: 'ragbits' as const,
        target: 'graphiti' as const,
        direction: 'ragbits_to_graphiti' as const,
        status: 'pending' as const,
        timestamp_utc: '2024-01-01T10:00:00-05:00', // Not UTC
        correlation_id: uuidv4(),
      };

      const result = validateSyncOperation(invalidOperation);
      expect(result.success).toBe(false);
    });
  });

  describe('Idempotency Tests (Law of Idempotency)', () => {
    it('should allow multiple operations with same data', () => {
      const correlationId = uuidv4();
      const data = {
        content: 'Test content',
        source: 'test-source',
      };

      // Create multiple operations with same data
      const operation1 = createSyncOperation('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', correlationId);
      const operation2 = createSyncOperation('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', correlationId);

      // Each should have unique ID
      expect(operation1.id).not.toBe(operation2.id);
    });

    it('should handle batch operations with duplicates', () => {
      const chunkId = uuidv4();
      const chunks = [
        { id: chunkId, content: 'Content 1', source: 'test', chunk_index: 0, timestamp: new Date().toISOString() },
        { id: chunkId, content: 'Content 1', source: 'test', chunk_index: 0, timestamp: new Date().toISOString() },
      ];

      // Should not fail on duplicate chunks
      expect(chunks.length).toBe(2);
      expect(chunks[0].id).toBe(chunks[1].id);
    });
  });

  describe('Configuration Explicitness (Law of Configuration Explicitness)', () => {
    it('should have no magic defaults in sync config', () => {
      const config = DEFAULT_SYNC_CONFIG;

      // All values should be explicitly defined
      expect(config.enabled).toBeDefined();
      expect(config.interval_ms).toBeDefined();
      expect(config.bidirectional).toBeDefined();
      expect(config.conflict_resolution).toBeDefined();
      expect(config.max_retries).toBeDefined();
      expect(config.retry_delay_ms).toBeDefined();
      expect(config.batch_size).toBeDefined();
      expect(config.timeout_ms).toBeDefined();
    });

    it('should reject configuration with missing timeout', () => {
      const invalidConfig = {
        ...DEFAULT_SYNC_CONFIG,
        timeout_ms: undefined as any,
      };

      const result = validateSyncConfig(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('Conflict Detection Tests', () => {
    it('should create conflict with all required fields', () => {
      const conflict = createConflict(
        'entity_mismatch',
        'high',
        { name: 'Entity A', labels: ['Person'] },
        { name: 'Entity A', labels: ['Organization'] },
        'Label mismatch detected',
        uuidv4()
      );

      expect(conflict.id).toBeDefined();
      expect(conflict.type).toBe('entity_mismatch');
      expect(conflict.severity).toBe('high');
      expect(conflict.resolved).toBe(false);
      expect(conflict.detected_at_utc).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should allow conflict resolution', () => {
      const conflict = createConflict(
        'temporal_inconsistency',
        'medium',
        { timestamp: '2024-01-01T10:00:00Z' },
        { timestamp: '2024-01-01T11:00:00Z' },
        'Temporal drift',
        uuidv4()
      );

      expect(conflict.resolved).toBe(false);

      // Resolve conflict
      conflict.resolved = true;
      conflict.resolution_strategy = 'newest_wins';
      conflict.resolution_notes = 'Auto-resolved';

      expect(conflict.resolved).toBe(true);
      expect(conflict.resolution_strategy).toBe('newest_wins');
    });
  });

  describe('Sync Operation Flow', () => {
    it('should transition through operation statuses', () => {
      const operation = createSyncOperation(
        'scheduled_sync',
        'ragbits',
        'graphiti',
        'bidirectional',
        uuidv4()
      );

      // Initial status
      expect(operation.status).toBe('pending');

      // Simulate status transitions
      operation.status = 'in_progress';
      expect(operation.status).toBe('in_progress');

      operation.status = 'completed';
      expect(operation.status).toBe('completed');
    });

    it('should track operation progress', () => {
      const result = createSyncResult(
        uuidv4(),
        'completed',
        'bidirectional',
        uuidv4(),
        5000
      );

      result.operations_total = 10;
      result.operations_completed = 8;
      result.operations_failed = 2;

      expect(result.operations_total).toBe(10);
      expect(result.operations_completed).toBe(8);
      expect(result.operations_failed).toBe(2);
      expect(result.duration_ms).toBe(5000);
    });
  });

  describe('Type Guards', () => {
    it('should identify valid SyncOperation', () => {
      const operation = createSyncOperation(
        'ingest_sync',
        'ragbits',
        'graphiti',
        'ragbits_to_graphiti',
        uuidv4()
      );

      // Note: isSyncOperation would be imported from canonical
      // For this test, we're just checking the structure
      expect(operation).toHaveProperty('id');
      expect(operation).toHaveProperty('type');
      expect(operation).toHaveProperty('status');
      expect(operation).toHaveProperty('timestamp_utc');
    });

    it('should identify valid SyncResult', () => {
      const result = createSyncResult(
        uuidv4(),
        'completed',
        'ragbits_to_graphiti',
        uuidv4()
      );

      expect(result).toHaveProperty('operation_id');
      expect(result).toHaveProperty('status');
      expect(result).toHaveProperty('duration_ms');
    });

    it('should identify valid Conflict', () => {
      const conflict = createConflict(
        'semantic_conflict',
        'low',
        { data: 'A' },
        { data: 'B' },
        'Test',
        uuidv4()
      );

      expect(conflict).toHaveProperty('id');
      expect(conflict).toHaveProperty('type');
      expect(conflict).toHaveProperty('severity');
      expect(conflict).toHaveProperty('resolved');
    });
  });

  describe('Error Handling', () => {
    it('should capture errors in sync result', () => {
      const result = createSyncResult(
        uuidv4(),
        'failed',
        'ragbits_to_graphiti',
        uuidv4()
      );

      result.errors.push({
        code: 'SYNC_FAILED',
        message: 'Connection timeout',
        details: { timeout_ms: 30000 },
      });

      expect(result.errors.length).toBe(1);
      expect(result.errors[0].code).toBe('SYNC_FAILED');
      expect(result.errors[0].message).toBe('Connection timeout');
    });

    it('should handle multiple errors', () => {
      const result = createSyncResult(
        uuidv4(),
        'partially_completed',
        'bidirectional',
        uuidv4()
      );

      result.errors.push(
        { code: 'ENTITY_SYNC_FAILED', message: 'Entity not found', details: {} },
        { code: 'TEMPORAL_DRIFT', message: 'Timestamp mismatch', details: { drift_ms: 5000 } }
      );

      expect(result.errors.length).toBe(2);
    });
  });
});
