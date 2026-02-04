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

import { z } from 'zod';

// ============================================================================
// SYNC OPERATION SCHEMAS
// ============================================================================

/**
 * Sync Direction Enumeration
 */
export const SyncDirectionEnum = z.enum([
  'ragbits_to_graphiti',
  'graphiti_to_ragbits',
  'bidirectional',
]);

export type SyncDirection = z.infer<typeof SyncDirectionEnum>;

/**
 * Sync Status Enumeration
 */
export const SyncStatusEnum = z.enum([
  'pending',
  'in_progress',
  'completed',
  'failed',
  'conflict_detected',
  'partially_completed',
]);

export type SyncStatus = z.infer<typeof SyncStatusEnum>;

/**
 * Sync Operation Type
 *
 * Represents a single synchronization operation
 */
export const SyncOperationSchema = z.object({
  id: z.string().uuid().describe('Unique identifier for the sync operation'),
  type: z.enum([
    'ingest_sync',
    'entity_sync',
    'episode_sync',
    'enhancement_sync',
    'conflict_resolution',
    'scheduled_sync',
  ]).describe('Type of sync operation'),
  source: z.enum(['ragbits', 'graphiti']).describe('Source system'),
  target: z.enum(['ragbits', 'graphiti']).describe('Target system'),
  direction: SyncDirectionEnum.describe('Direction of sync'),
  status: SyncStatusEnum.describe('Current status of the operation'),
  timestamp_utc: z.string().datetime().describe('UTC timestamp of operation (ISO-8601)'),
  correlation_id: z.string().uuid().describe('Correlation ID for distributed tracing'),
  entity_ids: z.array(z.string().uuid()).optional().describe('IDs of entities being synced'),
  episode_ids: z.array(z.string().uuid()).optional().describe('IDs of episodes being synced'),
  chunk_ids: z.array(z.string().uuid()).optional().describe('IDs of document chunks being synced'),
  error_message: z.string().optional().describe('Error message if status is failed'),
  metadata: z.record(z.any()).optional().describe('Additional metadata'),
});

export type SyncOperation = z.infer<typeof SyncOperationSchema>;

// ============================================================================
// SYNC CONFIGURATION SCHEMAS
// ============================================================================

/**
 * Conflict Resolution Strategy
 */
export const ConflictResolutionEnum = z.enum([
  'source_wins',
  'target_wins',
  'newest_wins',
  'manual',
  'merge',
]);

export type ConflictResolution = z.infer<typeof ConflictResolutionEnum>;

/**
 * Sync Configuration
 *
 * Configuration for synchronization behavior
 */
export const SyncConfigSchema = z.object({
  enabled: z.boolean().describe('Whether sync is enabled'),
  interval_ms: z
    .number()
    .int()
    .positive()
    .describe('Interval for scheduled sync in milliseconds'),
  bidirectional: z.boolean().describe('Whether to enable bidirectional sync'),
  conflict_resolution: ConflictResolutionEnum.describe('Default conflict resolution strategy'),
  max_retries: z
    .number()
    .int()
    .positive()
    .describe('Maximum number of retry attempts for failed operations'),
  retry_delay_ms: z
    .number()
    .int()
    .positive()
    .describe('Delay between retries in milliseconds'),
  batch_size: z
    .number()
    .int()
    .positive()
    .describe('Number of items to process in a single batch'),
  timeout_ms: z
    .number()
    .int()
    .positive()
    .describe('Timeout for sync operations in milliseconds'),
  auto_resolve_conflicts: z.boolean().describe('Whether to automatically resolve conflicts'),
  sync_on_ingest: z.boolean().describe('Whether to trigger sync on document ingest'),
  enhance_retrieval: z.boolean().describe('Whether to enhance retrieval with graph entities'),
});

export type SyncConfig = z.infer<typeof SyncConfigSchema>;

// ============================================================================
// SYNC RESULT SCHEMAS
// ============================================================================

/**
 * Sync Result
 *
 * Represents the result of a synchronization operation
 */
export const SyncResultSchema = z.object({
  operation_id: z.string().uuid().describe('ID of the sync operation'),
  status: SyncStatusEnum.describe('Final status of the operation'),
  direction: SyncDirectionEnum.describe('Direction of sync'),
  operations_completed: z.number().int().nonnegative().describe('Number of operations completed'),
  operations_failed: z.number().int().nonnegative().describe('Number of operations that failed'),
  operations_total: z.number().int().nonnegative().describe('Total number of operations'),
  conflicts_detected: z.number().int().nonnegative().describe('Number of conflicts detected'),
  conflicts_resolved: z.number().int().nonnegative().describe('Number of conflicts resolved'),
  errors: z
    .array(
      z.object({
        code: z.string(),
        message: z.string(),
        details: z.record(z.any()).optional(),
      })
    )
    .describe('Array of errors that occurred during sync'),
  duration_ms: z.number().nonnegative().describe('Duration of the sync operation in milliseconds'),
  timestamp_utc: z.string().datetime().describe('UTC timestamp of result (ISO-8601)'),
  correlation_id: z.string().uuid().describe('Correlation ID for tracing'),
  metadata: z.record(z.any()).optional().describe('Additional metadata'),
});

export type SyncResult = z.infer<typeof SyncResultSchema>;

// ============================================================================
// CONFLICT DETECTION SCHEMAS
// ============================================================================

/**
 * Conflict Type Enumeration
 */
export const ConflictTypeEnum = z.enum([
  'entity_mismatch',
  'temporal_inconsistency',
  'semantic_conflict',
  'data_collision',
  'reference_missing',
  'version_conflict',
]);

export type ConflictType = z.infer<typeof ConflictTypeEnum>;

/**
 * Conflict Severity Enumeration
 */
export const ConflictSeverityEnum = z.enum(['low', 'medium', 'high', 'critical']);

export type ConflictSeverity = z.infer<typeof ConflictSeverityEnum>;

/**
 * Conflict
 *
 * Represents a detected conflict between systems
 */
export const ConflictSchema = z.object({
  id: z.string().uuid().describe('Unique identifier for the conflict'),
  type: ConflictTypeEnum.describe('Type of conflict'),
  severity: ConflictSeverityEnum.describe('Severity level of the conflict'),
  ragbits_data: z.record(z.any()).describe('Data from RAGBits system'),
  graphiti_data: z.record(z.any()).describe('Data from Graphiti system'),
  entity_id: z.string().uuid().optional().describe('ID of the entity in conflict'),
  episode_id: z.string().uuid().optional().describe('ID of the episode in conflict'),
  chunk_id: z.string().uuid().optional().describe('ID of the chunk in conflict'),
  description: z.string().describe('Human-readable description of the conflict'),
  suggested_resolution: z.string().optional().describe('Suggested resolution for the conflict'),
  detected_at_utc: z.string().datetime().describe('UTC timestamp when conflict was detected'),
  resolved: z.boolean().describe('Whether the conflict has been resolved'),
  resolution_strategy: ConflictResolutionEnum.optional().describe('Strategy used for resolution'),
  resolution_notes: z.string().optional().describe('Notes about the resolution'),
  metadata: z.record(z.any()).optional().describe('Additional metadata'),
});

export type Conflict = z.infer<typeof ConflictSchema>;

/**
 * Conflict Report
 *
 * Report of detected conflicts and their resolutions
 */
export const ConflictReportSchema = z.object({
  sync_operation_id: z.string().uuid().describe('ID of the sync operation'),
  conflicts: z.array(ConflictSchema).describe('Array of detected conflicts'),
  resolutions: z.array(
    z.object({
      conflict_id: z.string().uuid(),
      strategy: ConflictResolutionEnum,
      applied_at_utc: z.string().datetime(),
      notes: z.string().optional(),
    })
  ).describe('Array of conflict resolutions'),
  unresolved: z.array(z.string().uuid()).describe('IDs of unresolved conflicts'),
  total_conflicts: z.number().int().nonnegative().describe('Total number of conflicts'),
  resolved_count: z.number().int().nonnegative().describe('Number of resolved conflicts'),
  unresolved_count: z.number().int().nonnegative().describe('Number of unresolved conflicts'),
  timestamp_utc: z.string().datetime().describe('UTC timestamp of report (ISO-8601)'),
  correlation_id: z.string().uuid().describe('Correlation ID for tracing'),
});

export type ConflictReport = z.infer<typeof ConflictReportSchema>;

// ============================================================================
// SYNC SPECIFICATION SCHEMAS
// ============================================================================

/**
 * Sync Specification
 *
 * Specifies what to sync and how
 */
export const SyncSpecSchema = z.object({
  direction: SyncDirectionEnum.describe('Direction of sync'),
  entity_ids: z.array(z.string().uuid()).optional().describe('Specific entities to sync'),
  episode_ids: z.array(z.string().uuid()).optional().describe('Specific episodes to sync'),
  chunk_ids: z.array(z.string().uuid()).optional().describe('Specific chunks to sync'),
  date_range: z
    .object({
      start: z.string().datetime(),
      end: z.string().datetime(),
    })
    .optional()
    .describe('Date range for sync'),
  filters: z.record(z.any()).optional().describe('Filters to apply'),
  conflict_resolution: ConflictResolutionEnum.optional().describe('Conflict resolution strategy'),
  timeout_ms: z.number().int().positive().optional().describe('Timeout for this sync'),
  metadata: z.record(z.any()).optional().describe('Additional metadata'),
});

export type SyncSpec = z.infer<typeof SyncSpecSchema>;

// ============================================================================
// ENHANCED RETRIEVAL SCHEMAS
// ============================================================================

/**
 * Enhanced Query
 *
 * Represents a query enhanced with knowledge graph information
 */
export const EnhancedQuerySchema = z.object({
  original_query: z.string().describe('Original user query'),
  enhanced_query: z.string().describe('Query enhanced with entity context'),
  entities: z
    .array(
      z.object({
        id: z.string().uuid(),
        name: z.string(),
        labels: z.array(z.string()),
        boost_factor: z.number().nonnegative(),
      })
    )
    .describe('Entities extracted from the graph'),
  boost_factors: z.record(z.number()).optional().describe('Boost factors for retrieval'),
  metadata: z.record(z.any()).optional().describe('Additional metadata'),
});

export type EnhancedQuery = z.infer<typeof EnhancedQuerySchema>;

/**
 * Boost Factor
 *
 * Represents a boost factor for retrieval
 */
export const BoostFactorSchema = z.object({
  entity_id: z.string().uuid().describe('ID of the entity to boost'),
  boost_value: z.number().nonnegative().describe('Boost value to apply'),
  reason: z.string().describe('Reason for the boost'),
  confidence: z.number().min(0).max(1).describe('Confidence score (0-1)'),
});

export type BoostFactor = z.infer<typeof BoostFactorSchema>;

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validate sync operation
 */
export function validateSyncOperation(data: unknown): {
  success: boolean;
  data?: SyncOperation;
  errors?: string[];
} {
  const result = SyncOperationSchema.safeParse(data);

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
export function validateSyncConfig(data: unknown): {
  success: boolean;
  data?: SyncConfig;
  errors?: string[];
} {
  const result = SyncConfigSchema.safeParse(data);

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
export function validateSyncResult(data: unknown): {
  success: boolean;
  data?: SyncResult;
  errors?: string[];
} {
  const result = SyncResultSchema.safeParse(data);

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
export function validateConflict(data: unknown): {
  success: boolean;
  data?: Conflict;
  errors?: string[];
} {
  const result = ConflictSchema.safeParse(data);

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
export function validateSyncSpec(data: unknown): {
  success: boolean;
  data?: SyncSpec;
  errors?: string[];
} {
  const result = SyncSpecSchema.safeParse(data);

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
export function isSyncOperation(data: unknown): data is SyncOperation {
  return (
    typeof data === 'object' &&
    data !== null &&
    'id' in data &&
    'type' in data &&
    'status' in data &&
    'timestamp_utc' in data
  );
}

/**
 * Check if data is a valid SyncResult
 */
export function isSyncResult(data: unknown): data is SyncResult {
  return (
    typeof data === 'object' &&
    data !== null &&
    'operation_id' in data &&
    'status' in data &&
    'duration_ms' in data
  );
}

/**
 * Check if data is a valid Conflict
 */
export function isConflict(data: unknown): data is Conflict {
  return (
    typeof data === 'object' &&
    data !== null &&
    'id' in data &&
    'type' in data &&
    'severity' in data &&
    'resolved' in data
  );
}

// ============================================================================
// DEFAULT CONFIGURATIONS
// ============================================================================

/**
 * Default sync configuration
 */
export const DEFAULT_SYNC_CONFIG: SyncConfig = {
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
export function createSyncOperation(
  type: SyncOperation['type'],
  source: SyncOperation['source'],
  target: SyncOperation['target'],
  direction: SyncDirection,
  correlationId: string
): SyncOperation {
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
export function createSyncResult(
  operationId: string,
  status: SyncStatus,
  direction: SyncDirection,
  correlationId: string,
  durationMs: number = 0
): SyncResult {
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
export function createConflict(
  type: ConflictType,
  severity: ConflictSeverity,
  ragbitsData: Record<string, any>,
  graphitiData: Record<string, any>,
  description: string,
  correlationId: string
): Conflict {
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
