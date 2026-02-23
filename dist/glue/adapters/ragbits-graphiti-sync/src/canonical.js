"use strict";
/**
 * Canonical Schema for RAGBits-Graphiti Bidirectional Synchronization
 *
 * Follows the Federation Constitution:
 * - Anti-Corruption Layer: Normalizes data between RAGBits and Graphiti
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Configuration Explicitness: No magic defaults
 *
 * This schema defines the contract for synchronization operations between
 * RAGBits (RAG system) and Graphiti (Temporal Knowledge Graph).
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_SYNC_CONFIG = exports.BoostFactorSchema = exports.EnhancedQuerySchema = exports.SyncSpecSchema = exports.ConflictReportSchema = exports.ConflictSchema = exports.ConflictSeverityEnum = exports.ConflictTypeEnum = exports.SyncResultSchema = exports.SyncConfigSchema = exports.ConflictResolutionEnum = exports.SyncOperationSchema = exports.SyncStatusEnum = exports.SyncDirectionEnum = void 0;
exports.validateSyncOperation = validateSyncOperation;
exports.validateSyncConfig = validateSyncConfig;
exports.validateSyncResult = validateSyncResult;
exports.validateConflict = validateConflict;
exports.validateSyncSpec = validateSyncSpec;
exports.isSyncOperation = isSyncOperation;
exports.isSyncResult = isSyncResult;
exports.isConflict = isConflict;
exports.createSyncOperation = createSyncOperation;
exports.createSyncResult = createSyncResult;
exports.createConflict = createConflict;
const zod_1 = require("zod");
// ============================================================================
// SYNC OPERATION SCHEMAS
// ============================================================================
/**
 * Sync Direction Enumeration
 */
exports.SyncDirectionEnum = zod_1.z.enum([
    'ragbits_to_graphiti',
    'graphiti_to_ragbits',
    'bidirectional',
]);
/**
 * Sync Status Enumeration
 */
exports.SyncStatusEnum = zod_1.z.enum([
    'pending',
    'in_progress',
    'completed',
    'failed',
    'conflict_detected',
    'partially_completed',
]);
/**
 * Sync Operation Type
 *
 * Represents a single synchronization operation
 */
exports.SyncOperationSchema = zod_1.z.object({
    id: zod_1.z.string().uuid().describe('Unique identifier for the sync operation'),
    type: zod_1.z.enum([
        'ingest_sync',
        'entity_sync',
        'episode_sync',
        'enhancement_sync',
        'conflict_resolution',
        'scheduled_sync',
    ]).describe('Type of sync operation'),
    source: zod_1.z.enum(['ragbits', 'graphiti']).describe('Source system'),
    target: zod_1.z.enum(['ragbits', 'graphiti']).describe('Target system'),
    direction: exports.SyncDirectionEnum.describe('Direction of sync'),
    status: exports.SyncStatusEnum.describe('Current status of the operation'),
    timestamp_utc: zod_1.z.string().datetime().describe('UTC timestamp of operation (ISO-8601)'),
    correlation_id: zod_1.z.string().uuid().describe('Correlation ID for distributed tracing'),
    entity_ids: zod_1.z.array(zod_1.z.string().uuid()).optional().describe('IDs of entities being synced'),
    episode_ids: zod_1.z.array(zod_1.z.string().uuid()).optional().describe('IDs of episodes being synced'),
    chunk_ids: zod_1.z.array(zod_1.z.string().uuid()).optional().describe('IDs of document chunks being synced'),
    error_message: zod_1.z.string().optional().describe('Error message if status is failed'),
    metadata: zod_1.z.record(zod_1.z.any()).optional().describe('Additional metadata'),
});
// ============================================================================
// SYNC CONFIGURATION SCHEMAS
// ============================================================================
/**
 * Conflict Resolution Strategy
 */
exports.ConflictResolutionEnum = zod_1.z.enum([
    'source_wins',
    'target_wins',
    'newest_wins',
    'manual',
    'merge',
]);
/**
 * Sync Configuration
 *
 * Configuration for synchronization behavior
 */
exports.SyncConfigSchema = zod_1.z.object({
    enabled: zod_1.z.boolean().describe('Whether sync is enabled'),
    interval_ms: zod_1.z
        .number()
        .int()
        .positive()
        .describe('Interval for scheduled sync in milliseconds'),
    bidirectional: zod_1.z.boolean().describe('Whether to enable bidirectional sync'),
    conflict_resolution: exports.ConflictResolutionEnum.describe('Default conflict resolution strategy'),
    max_retries: zod_1.z
        .number()
        .int()
        .positive()
        .describe('Maximum number of retry attempts for failed operations'),
    retry_delay_ms: zod_1.z
        .number()
        .int()
        .positive()
        .describe('Delay between retries in milliseconds'),
    batch_size: zod_1.z
        .number()
        .int()
        .positive()
        .describe('Number of items to process in a single batch'),
    timeout_ms: zod_1.z
        .number()
        .int()
        .positive()
        .describe('Timeout for sync operations in milliseconds'),
    auto_resolve_conflicts: zod_1.z.boolean().describe('Whether to automatically resolve conflicts'),
    sync_on_ingest: zod_1.z.boolean().describe('Whether to trigger sync on document ingest'),
    enhance_retrieval: zod_1.z.boolean().describe('Whether to enhance retrieval with graph entities'),
});
// ============================================================================
// SYNC RESULT SCHEMAS
// ============================================================================
/**
 * Sync Result
 *
 * Represents the result of a synchronization operation
 */
exports.SyncResultSchema = zod_1.z.object({
    operation_id: zod_1.z.string().uuid().describe('ID of the sync operation'),
    status: exports.SyncStatusEnum.describe('Final status of the operation'),
    direction: exports.SyncDirectionEnum.describe('Direction of sync'),
    operations_completed: zod_1.z.number().int().nonnegative().describe('Number of operations completed'),
    operations_failed: zod_1.z.number().int().nonnegative().describe('Number of operations that failed'),
    operations_total: zod_1.z.number().int().nonnegative().describe('Total number of operations'),
    conflicts_detected: zod_1.z.number().int().nonnegative().describe('Number of conflicts detected'),
    conflicts_resolved: zod_1.z.number().int().nonnegative().describe('Number of conflicts resolved'),
    errors: zod_1.z
        .array(zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }))
        .describe('Array of errors that occurred during sync'),
    duration_ms: zod_1.z.number().nonnegative().describe('Duration of the sync operation in milliseconds'),
    timestamp_utc: zod_1.z.string().datetime().describe('UTC timestamp of result (ISO-8601)'),
    correlation_id: zod_1.z.string().uuid().describe('Correlation ID for tracing'),
    metadata: zod_1.z.record(zod_1.z.any()).optional().describe('Additional metadata'),
});
// ============================================================================
// CONFLICT DETECTION SCHEMAS
// ============================================================================
/**
 * Conflict Type Enumeration
 */
exports.ConflictTypeEnum = zod_1.z.enum([
    'entity_mismatch',
    'temporal_inconsistency',
    'semantic_conflict',
    'data_collision',
    'reference_missing',
    'version_conflict',
]);
/**
 * Conflict Severity Enumeration
 */
exports.ConflictSeverityEnum = zod_1.z.enum(['low', 'medium', 'high', 'critical']);
/**
 * Conflict
 *
 * Represents a detected conflict between systems
 */
exports.ConflictSchema = zod_1.z.object({
    id: zod_1.z.string().uuid().describe('Unique identifier for the conflict'),
    type: exports.ConflictTypeEnum.describe('Type of conflict'),
    severity: exports.ConflictSeverityEnum.describe('Severity level of the conflict'),
    ragbits_data: zod_1.z.record(zod_1.z.any()).describe('Data from RAGBits system'),
    graphiti_data: zod_1.z.record(zod_1.z.any()).describe('Data from Graphiti system'),
    entity_id: zod_1.z.string().uuid().optional().describe('ID of the entity in conflict'),
    episode_id: zod_1.z.string().uuid().optional().describe('ID of the episode in conflict'),
    chunk_id: zod_1.z.string().uuid().optional().describe('ID of the chunk in conflict'),
    description: zod_1.z.string().describe('Human-readable description of the conflict'),
    suggested_resolution: zod_1.z.string().optional().describe('Suggested resolution for the conflict'),
    detected_at_utc: zod_1.z.string().datetime().describe('UTC timestamp when conflict was detected'),
    resolved: zod_1.z.boolean().describe('Whether the conflict has been resolved'),
    resolution_strategy: exports.ConflictResolutionEnum.optional().describe('Strategy used for resolution'),
    resolution_notes: zod_1.z.string().optional().describe('Notes about the resolution'),
    metadata: zod_1.z.record(zod_1.z.any()).optional().describe('Additional metadata'),
});
/**
 * Conflict Report
 *
 * Report of detected conflicts and their resolutions
 */
exports.ConflictReportSchema = zod_1.z.object({
    sync_operation_id: zod_1.z.string().uuid().describe('ID of the sync operation'),
    conflicts: zod_1.z.array(exports.ConflictSchema).describe('Array of detected conflicts'),
    resolutions: zod_1.z.array(zod_1.z.object({
        conflict_id: zod_1.z.string().uuid(),
        strategy: exports.ConflictResolutionEnum,
        applied_at_utc: zod_1.z.string().datetime(),
        notes: zod_1.z.string().optional(),
    })).describe('Array of conflict resolutions'),
    unresolved: zod_1.z.array(zod_1.z.string().uuid()).describe('IDs of unresolved conflicts'),
    total_conflicts: zod_1.z.number().int().nonnegative().describe('Total number of conflicts'),
    resolved_count: zod_1.z.number().int().nonnegative().describe('Number of resolved conflicts'),
    unresolved_count: zod_1.z.number().int().nonnegative().describe('Number of unresolved conflicts'),
    timestamp_utc: zod_1.z.string().datetime().describe('UTC timestamp of report (ISO-8601)'),
    correlation_id: zod_1.z.string().uuid().describe('Correlation ID for tracing'),
});
// ============================================================================
// SYNC SPECIFICATION SCHEMAS
// ============================================================================
/**
 * Sync Specification
 *
 * Specifies what to sync and how
 */
exports.SyncSpecSchema = zod_1.z.object({
    direction: exports.SyncDirectionEnum.describe('Direction of sync'),
    entity_ids: zod_1.z.array(zod_1.z.string().uuid()).optional().describe('Specific entities to sync'),
    episode_ids: zod_1.z.array(zod_1.z.string().uuid()).optional().describe('Specific episodes to sync'),
    chunk_ids: zod_1.z.array(zod_1.z.string().uuid()).optional().describe('Specific chunks to sync'),
    date_range: zod_1.z
        .object({
        start: zod_1.z.string().datetime(),
        end: zod_1.z.string().datetime(),
    })
        .optional()
        .describe('Date range for sync'),
    filters: zod_1.z.record(zod_1.z.any()).optional().describe('Filters to apply'),
    conflict_resolution: exports.ConflictResolutionEnum.optional().describe('Conflict resolution strategy'),
    timeout_ms: zod_1.z.number().int().positive().optional().describe('Timeout for this sync'),
    metadata: zod_1.z.record(zod_1.z.any()).optional().describe('Additional metadata'),
});
// ============================================================================
// ENHANCED RETRIEVAL SCHEMAS
// ============================================================================
/**
 * Enhanced Query
 *
 * Represents a query enhanced with knowledge graph information
 */
exports.EnhancedQuerySchema = zod_1.z.object({
    original_query: zod_1.z.string().describe('Original user query'),
    enhanced_query: zod_1.z.string().describe('Query enhanced with entity context'),
    entities: zod_1.z
        .array(zod_1.z.object({
        id: zod_1.z.string().uuid(),
        name: zod_1.z.string(),
        labels: zod_1.z.array(zod_1.z.string()),
        boost_factor: zod_1.z.number().nonnegative(),
    }))
        .describe('Entities extracted from the graph'),
    boost_factors: zod_1.z.record(zod_1.z.number()).optional().describe('Boost factors for retrieval'),
    metadata: zod_1.z.record(zod_1.z.any()).optional().describe('Additional metadata'),
});
/**
 * Boost Factor
 *
 * Represents a boost factor for retrieval
 */
exports.BoostFactorSchema = zod_1.z.object({
    entity_id: zod_1.z.string().uuid().describe('ID of the entity to boost'),
    boost_value: zod_1.z.number().nonnegative().describe('Boost value to apply'),
    reason: zod_1.z.string().describe('Reason for the boost'),
    confidence: zod_1.z.number().min(0).max(1).describe('Confidence score (0-1)'),
});
// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================
/**
 * Validate sync operation
 */
function validateSyncOperation(data) {
    const result = exports.SyncOperationSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate sync config
 */
function validateSyncConfig(data) {
    const result = exports.SyncConfigSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate sync result
 */
function validateSyncResult(data) {
    const result = exports.SyncResultSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate conflict
 */
function validateConflict(data) {
    const result = exports.ConflictSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate sync spec
 */
function validateSyncSpec(data) {
    const result = exports.SyncSpecSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
// ============================================================================
// TYPE GUARDS
// ============================================================================
/**
 * Check if data is a valid SyncOperation
 */
function isSyncOperation(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'id' in data &&
        'type' in data &&
        'status' in data &&
        'timestamp_utc' in data);
}
/**
 * Check if data is a valid SyncResult
 */
function isSyncResult(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'operation_id' in data &&
        'status' in data &&
        'duration_ms' in data);
}
/**
 * Check if data is a valid Conflict
 */
function isConflict(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'id' in data &&
        'type' in data &&
        'severity' in data &&
        'resolved' in data);
}
// ============================================================================
// DEFAULT CONFIGURATIONS
// ============================================================================
/**
 * Default sync configuration
 */
exports.DEFAULT_SYNC_CONFIG = {
    enabled: true,
    interval_ms: 300000, // 5 minutes
    bidirectional: true,
    conflict_resolution: 'newest_wins',
    max_retries: 3,
    retry_delay_ms: 1000,
    batch_size: 10,
    timeout_ms: 30000,
    auto_resolve_conflicts: true,
    sync_on_ingest: true,
    enhance_retrieval: true,
};
/**
 * Create a sync operation
 */
function createSyncOperation(type, source, target, direction, correlationId) {
    return {
        id: crypto.randomUUID(),
        type,
        source,
        target,
        direction,
        status: 'pending',
        timestamp_utc: new Date().toISOString(),
        correlation_id: correlationId,
    };
}
/**
 * Create a sync result
 */
function createSyncResult(operationId, status, direction, correlationId, durationMs = 0) {
    return {
        operation_id: operationId,
        status,
        direction,
        operations_completed: 0,
        operations_failed: 0,
        operations_total: 0,
        conflicts_detected: 0,
        conflicts_resolved: 0,
        errors: [],
        duration_ms: durationMs,
        timestamp_utc: new Date().toISOString(),
        correlation_id: correlationId,
    };
}
/**
 * Create a conflict
 */
function createConflict(type, severity, ragbitsData, graphitiData, description, correlationId) {
    return {
        id: crypto.randomUUID(),
        type,
        severity,
        ragbits_data: ragbitsData,
        graphiti_data: graphitiData,
        description,
        detected_at_utc: new Date().toISOString(),
        resolved: false,
        correlation_id: correlationId,
    };
}
//# sourceMappingURL=canonical.js.map