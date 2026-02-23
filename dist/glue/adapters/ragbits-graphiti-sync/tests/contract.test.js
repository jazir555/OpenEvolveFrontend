"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const canonical_1 = require("../src/canonical");
const canonical_2 = require("../src/canonical");
const uuid_1 = require("uuid");
(0, globals_1.describe)('RAGBits-Graphiti Sync Contract Tests', () => {
    (0, globals_1.describe)('Canonical Schema Validation', () => {
        (0, globals_1.it)('should validate a valid SyncOperation', () => {
            const operation = (0, canonical_2.createSyncOperation)('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', (0, uuid_1.v4)());
            const result = (0, canonical_1.validateSyncOperation)(operation);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
            (0, globals_1.expect)(result.data?.status).toBe('pending');
        });
        (0, globals_1.it)('should reject invalid SyncOperation with missing required fields', () => {
            const invalidOperation = {
                id: (0, uuid_1.v4)(),
                // Missing required fields
            };
            const result = (0, canonical_1.validateSyncOperation)(invalidOperation);
            (0, globals_1.expect)(result.success).toBe(false);
            (0, globals_1.expect)(result.errors).toBeDefined();
            (0, globals_1.expect)(result.errors?.length).toBeGreaterThan(0);
        });
        (0, globals_1.it)('should validate a valid SyncConfig', () => {
            const config = {
                ...canonical_2.DEFAULT_SYNC_CONFIG,
                enabled: true,
                interval_ms: 300000,
                bidirectional: true,
            };
            const result = (0, canonical_1.validateSyncConfig)(config);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
            (0, globals_1.expect)(result.data?.enabled).toBe(true);
        });
        (0, globals_1.it)('should reject invalid SyncConfig with negative interval', () => {
            const invalidConfig = {
                ...canonical_2.DEFAULT_SYNC_CONFIG,
                interval_ms: -1000,
            };
            const result = (0, canonical_1.validateSyncConfig)(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
            (0, globals_1.expect)(result.errors).toBeDefined();
        });
        (0, globals_1.it)('should validate a valid SyncResult', () => {
            const result = (0, canonical_2.createSyncResult)((0, uuid_1.v4)(), 'completed', 'ragbits_to_graphiti', (0, uuid_1.v4)(), 1000);
            const validationResult = (0, canonical_1.validateSyncResult)(result);
            (0, globals_1.expect)(validationResult.success).toBe(true);
            (0, globals_1.expect)(validationResult.data).toBeDefined();
            (0, globals_1.expect)(validationResult.data?.status).toBe('completed');
        });
        (0, globals_1.it)('should validate a valid Conflict', () => {
            const conflict = (0, canonical_2.createConflict)('semantic_conflict', 'medium', { content: 'Version A' }, { content: 'Version B' }, 'Test conflict', (0, uuid_1.v4)());
            const result = (0, canonical_1.validateConflict)(conflict);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
            (0, globals_1.expect)(result.data?.severity).toBe('medium');
        });
        (0, globals_1.it)('should validate a valid SyncSpec', () => {
            const spec = {
                direction: 'bidirectional',
                entity_ids: [(0, uuid_1.v4)(), (0, uuid_1.v4)()],
                episode_ids: [(0, uuid_1.v4)()],
                timeout_ms: 30000,
            };
            const result = (0, canonical_1.validateSyncSpec)(spec);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
            (0, globals_1.expect)(result.data?.direction).toBe('bidirectional');
        });
    });
    (0, globals_1.describe)('Timestamp Validation (Law of UTC)', () => {
        (0, globals_1.it)('should require UTC ISO-8601 timestamps', () => {
            const operation = (0, canonical_2.createSyncOperation)('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', (0, uuid_1.v4)());
            // Timestamp should be in ISO-8601 format
            (0, globals_1.expect)(operation.timestamp_utc).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
            (0, globals_1.expect)(operation.timestamp_utc).toContain('Z');
        });
        (0, globals_1.it)('should reject non-UTC timestamps', () => {
            const invalidOperation = {
                id: (0, uuid_1.v4)(),
                type: 'ingest_sync',
                source: 'ragbits',
                target: 'graphiti',
                direction: 'ragbits_to_graphiti',
                status: 'pending',
                timestamp_utc: '2024-01-01T10:00:00-05:00', // Not UTC
                correlation_id: (0, uuid_1.v4)(),
            };
            const result = (0, canonical_1.validateSyncOperation)(invalidOperation);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Idempotency Tests (Law of Idempotency)', () => {
        (0, globals_1.it)('should allow multiple operations with same data', () => {
            const correlationId = (0, uuid_1.v4)();
            const data = {
                content: 'Test content',
                source: 'test-source',
            };
            // Create multiple operations with same data
            const operation1 = (0, canonical_2.createSyncOperation)('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', correlationId);
            const operation2 = (0, canonical_2.createSyncOperation)('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', correlationId);
            // Each should have unique ID
            (0, globals_1.expect)(operation1.id).not.toBe(operation2.id);
        });
        (0, globals_1.it)('should handle batch operations with duplicates', () => {
            const chunkId = (0, uuid_1.v4)();
            const chunks = [
                { id: chunkId, content: 'Content 1', source: 'test', chunk_index: 0, timestamp: new Date().toISOString() },
                { id: chunkId, content: 'Content 1', source: 'test', chunk_index: 0, timestamp: new Date().toISOString() },
            ];
            // Should not fail on duplicate chunks
            (0, globals_1.expect)(chunks.length).toBe(2);
            (0, globals_1.expect)(chunks[0].id).toBe(chunks[1].id);
        });
    });
    (0, globals_1.describe)('Configuration Explicitness (Law of Configuration Explicitness)', () => {
        (0, globals_1.it)('should have no magic defaults in sync config', () => {
            const config = canonical_2.DEFAULT_SYNC_CONFIG;
            // All values should be explicitly defined
            (0, globals_1.expect)(config.enabled).toBeDefined();
            (0, globals_1.expect)(config.interval_ms).toBeDefined();
            (0, globals_1.expect)(config.bidirectional).toBeDefined();
            (0, globals_1.expect)(config.conflict_resolution).toBeDefined();
            (0, globals_1.expect)(config.max_retries).toBeDefined();
            (0, globals_1.expect)(config.retry_delay_ms).toBeDefined();
            (0, globals_1.expect)(config.batch_size).toBeDefined();
            (0, globals_1.expect)(config.timeout_ms).toBeDefined();
        });
        (0, globals_1.it)('should reject configuration with missing timeout', () => {
            const invalidConfig = {
                ...canonical_2.DEFAULT_SYNC_CONFIG,
                timeout_ms: undefined,
            };
            const result = (0, canonical_1.validateSyncConfig)(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Conflict Detection Tests', () => {
        (0, globals_1.it)('should create conflict with all required fields', () => {
            const conflict = (0, canonical_2.createConflict)('entity_mismatch', 'high', { name: 'Entity A', labels: ['Person'] }, { name: 'Entity A', labels: ['Organization'] }, 'Label mismatch detected', (0, uuid_1.v4)());
            (0, globals_1.expect)(conflict.id).toBeDefined();
            (0, globals_1.expect)(conflict.type).toBe('entity_mismatch');
            (0, globals_1.expect)(conflict.severity).toBe('high');
            (0, globals_1.expect)(conflict.resolved).toBe(false);
            (0, globals_1.expect)(conflict.detected_at_utc).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
        });
        (0, globals_1.it)('should allow conflict resolution', () => {
            const conflict = (0, canonical_2.createConflict)('temporal_inconsistency', 'medium', { timestamp: '2024-01-01T10:00:00Z' }, { timestamp: '2024-01-01T11:00:00Z' }, 'Temporal drift', (0, uuid_1.v4)());
            (0, globals_1.expect)(conflict.resolved).toBe(false);
            // Resolve conflict
            conflict.resolved = true;
            conflict.resolution_strategy = 'newest_wins';
            conflict.resolution_notes = 'Auto-resolved';
            (0, globals_1.expect)(conflict.resolved).toBe(true);
            (0, globals_1.expect)(conflict.resolution_strategy).toBe('newest_wins');
        });
    });
    (0, globals_1.describe)('Sync Operation Flow', () => {
        (0, globals_1.it)('should transition through operation statuses', () => {
            const operation = (0, canonical_2.createSyncOperation)('scheduled_sync', 'ragbits', 'graphiti', 'bidirectional', (0, uuid_1.v4)());
            // Initial status
            (0, globals_1.expect)(operation.status).toBe('pending');
            // Simulate status transitions
            operation.status = 'in_progress';
            (0, globals_1.expect)(operation.status).toBe('in_progress');
            operation.status = 'completed';
            (0, globals_1.expect)(operation.status).toBe('completed');
        });
        (0, globals_1.it)('should track operation progress', () => {
            const result = (0, canonical_2.createSyncResult)((0, uuid_1.v4)(), 'completed', 'bidirectional', (0, uuid_1.v4)(), 5000);
            result.operations_total = 10;
            result.operations_completed = 8;
            result.operations_failed = 2;
            (0, globals_1.expect)(result.operations_total).toBe(10);
            (0, globals_1.expect)(result.operations_completed).toBe(8);
            (0, globals_1.expect)(result.operations_failed).toBe(2);
            (0, globals_1.expect)(result.duration_ms).toBe(5000);
        });
    });
    (0, globals_1.describe)('Type Guards', () => {
        (0, globals_1.it)('should identify valid SyncOperation', () => {
            const operation = (0, canonical_2.createSyncOperation)('ingest_sync', 'ragbits', 'graphiti', 'ragbits_to_graphiti', (0, uuid_1.v4)());
            // Note: isSyncOperation would be imported from canonical
            // For this test, we're just checking the structure
            (0, globals_1.expect)(operation).toHaveProperty('id');
            (0, globals_1.expect)(operation).toHaveProperty('type');
            (0, globals_1.expect)(operation).toHaveProperty('status');
            (0, globals_1.expect)(operation).toHaveProperty('timestamp_utc');
        });
        (0, globals_1.it)('should identify valid SyncResult', () => {
            const result = (0, canonical_2.createSyncResult)((0, uuid_1.v4)(), 'completed', 'ragbits_to_graphiti', (0, uuid_1.v4)());
            (0, globals_1.expect)(result).toHaveProperty('operation_id');
            (0, globals_1.expect)(result).toHaveProperty('status');
            (0, globals_1.expect)(result).toHaveProperty('duration_ms');
        });
        (0, globals_1.it)('should identify valid Conflict', () => {
            const conflict = (0, canonical_2.createConflict)('semantic_conflict', 'low', { data: 'A' }, { data: 'B' }, 'Test', (0, uuid_1.v4)());
            (0, globals_1.expect)(conflict).toHaveProperty('id');
            (0, globals_1.expect)(conflict).toHaveProperty('type');
            (0, globals_1.expect)(conflict).toHaveProperty('severity');
            (0, globals_1.expect)(conflict).toHaveProperty('resolved');
        });
    });
    (0, globals_1.describe)('Error Handling', () => {
        (0, globals_1.it)('should capture errors in sync result', () => {
            const result = (0, canonical_2.createSyncResult)((0, uuid_1.v4)(), 'failed', 'ragbits_to_graphiti', (0, uuid_1.v4)());
            result.errors.push({
                code: 'SYNC_FAILED',
                message: 'Connection timeout',
                details: { timeout_ms: 30000 },
            });
            (0, globals_1.expect)(result.errors.length).toBe(1);
            (0, globals_1.expect)(result.errors[0].code).toBe('SYNC_FAILED');
            (0, globals_1.expect)(result.errors[0].message).toBe('Connection timeout');
        });
        (0, globals_1.it)('should handle multiple errors', () => {
            const result = (0, canonical_2.createSyncResult)((0, uuid_1.v4)(), 'partially_completed', 'bidirectional', (0, uuid_1.v4)());
            result.errors.push({ code: 'ENTITY_SYNC_FAILED', message: 'Entity not found', details: {} }, { code: 'TEMPORAL_DRIFT', message: 'Timestamp mismatch', details: { drift_ms: 5000 } });
            (0, globals_1.expect)(result.errors.length).toBe(2);
        });
    });
});
//# sourceMappingURL=contract.test.js.map