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
/**
 * Sync Direction Enumeration
 */
export declare const SyncDirectionEnum: z.ZodEnum<["ragbits_to_graphiti", "graphiti_to_ragbits", "bidirectional"]>;
export type SyncDirection = z.infer<typeof SyncDirectionEnum>;
/**
 * Sync Status Enumeration
 */
export declare const SyncStatusEnum: z.ZodEnum<["pending", "in_progress", "completed", "failed", "conflict_detected", "partially_completed"]>;
export type SyncStatus = z.infer<typeof SyncStatusEnum>;
/**
 * Sync Operation Type
 *
 * Represents a single synchronization operation
 */
export declare const SyncOperationSchema: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["ingest_sync", "entity_sync", "episode_sync", "enhancement_sync", "conflict_resolution", "scheduled_sync"]>;
    source: z.ZodEnum<["ragbits", "graphiti"]>;
    target: z.ZodEnum<["ragbits", "graphiti"]>;
    direction: z.ZodEnum<["ragbits_to_graphiti", "graphiti_to_ragbits", "bidirectional"]>;
    status: z.ZodEnum<["pending", "in_progress", "completed", "failed", "conflict_detected", "partially_completed"]>;
    timestamp_utc: z.ZodString;
    correlation_id: z.ZodString;
    entity_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    episode_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    chunk_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    error_message: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    correlation_id: string;
    status: "completed" | "failed" | "pending" | "in_progress" | "conflict_detected" | "partially_completed";
    type: "ingest_sync" | "entity_sync" | "episode_sync" | "enhancement_sync" | "conflict_resolution" | "scheduled_sync";
    id: string;
    source: "ragbits" | "graphiti";
    timestamp_utc: string;
    target: "ragbits" | "graphiti";
    direction: "ragbits_to_graphiti" | "graphiti_to_ragbits" | "bidirectional";
    error_message?: string | undefined;
    metadata?: Record<string, any> | undefined;
    entity_ids?: string[] | undefined;
    episode_ids?: string[] | undefined;
    chunk_ids?: string[] | undefined;
}, {
    correlation_id: string;
    status: "completed" | "failed" | "pending" | "in_progress" | "conflict_detected" | "partially_completed";
    type: "ingest_sync" | "entity_sync" | "episode_sync" | "enhancement_sync" | "conflict_resolution" | "scheduled_sync";
    id: string;
    source: "ragbits" | "graphiti";
    timestamp_utc: string;
    target: "ragbits" | "graphiti";
    direction: "ragbits_to_graphiti" | "graphiti_to_ragbits" | "bidirectional";
    error_message?: string | undefined;
    metadata?: Record<string, any> | undefined;
    entity_ids?: string[] | undefined;
    episode_ids?: string[] | undefined;
    chunk_ids?: string[] | undefined;
}>;
export type SyncOperation = z.infer<typeof SyncOperationSchema>;
/**
 * Conflict Resolution Strategy
 */
export declare const ConflictResolutionEnum: z.ZodEnum<["source_wins", "target_wins", "newest_wins", "manual", "merge"]>;
export type ConflictResolution = z.infer<typeof ConflictResolutionEnum>;
/**
 * Sync Configuration
 *
 * Configuration for synchronization behavior
 */
export declare const SyncConfigSchema: z.ZodObject<{
    enabled: z.ZodBoolean;
    interval_ms: z.ZodNumber;
    bidirectional: z.ZodBoolean;
    conflict_resolution: z.ZodEnum<["source_wins", "target_wins", "newest_wins", "manual", "merge"]>;
    max_retries: z.ZodNumber;
    retry_delay_ms: z.ZodNumber;
    batch_size: z.ZodNumber;
    timeout_ms: z.ZodNumber;
    auto_resolve_conflicts: z.ZodBoolean;
    sync_on_ingest: z.ZodBoolean;
    enhance_retrieval: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    enabled: boolean;
    timeout_ms: number;
    interval_ms: number;
    max_retries: number;
    retry_delay_ms: number;
    bidirectional: boolean;
    conflict_resolution: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge";
    batch_size: number;
    auto_resolve_conflicts: boolean;
    sync_on_ingest: boolean;
    enhance_retrieval: boolean;
}, {
    enabled: boolean;
    timeout_ms: number;
    interval_ms: number;
    max_retries: number;
    retry_delay_ms: number;
    bidirectional: boolean;
    conflict_resolution: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge";
    batch_size: number;
    auto_resolve_conflicts: boolean;
    sync_on_ingest: boolean;
    enhance_retrieval: boolean;
}>;
export type SyncConfig = z.infer<typeof SyncConfigSchema>;
/**
 * Sync Result
 *
 * Represents the result of a synchronization operation
 */
export declare const SyncResultSchema: z.ZodObject<{
    operation_id: z.ZodString;
    status: z.ZodEnum<["pending", "in_progress", "completed", "failed", "conflict_detected", "partially_completed"]>;
    direction: z.ZodEnum<["ragbits_to_graphiti", "graphiti_to_ragbits", "bidirectional"]>;
    operations_completed: z.ZodNumber;
    operations_failed: z.ZodNumber;
    operations_total: z.ZodNumber;
    conflicts_detected: z.ZodNumber;
    conflicts_resolved: z.ZodNumber;
    errors: z.ZodArray<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>, "many">;
    duration_ms: z.ZodNumber;
    timestamp_utc: z.ZodString;
    correlation_id: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    correlation_id: string;
    duration_ms: number;
    status: "completed" | "failed" | "pending" | "in_progress" | "conflict_detected" | "partially_completed";
    errors: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }[];
    timestamp_utc: string;
    direction: "ragbits_to_graphiti" | "graphiti_to_ragbits" | "bidirectional";
    operation_id: string;
    operations_completed: number;
    operations_failed: number;
    operations_total: number;
    conflicts_detected: number;
    conflicts_resolved: number;
    metadata?: Record<string, any> | undefined;
}, {
    correlation_id: string;
    duration_ms: number;
    status: "completed" | "failed" | "pending" | "in_progress" | "conflict_detected" | "partially_completed";
    errors: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }[];
    timestamp_utc: string;
    direction: "ragbits_to_graphiti" | "graphiti_to_ragbits" | "bidirectional";
    operation_id: string;
    operations_completed: number;
    operations_failed: number;
    operations_total: number;
    conflicts_detected: number;
    conflicts_resolved: number;
    metadata?: Record<string, any> | undefined;
}>;
export type SyncResult = z.infer<typeof SyncResultSchema>;
/**
 * Conflict Type Enumeration
 */
export declare const ConflictTypeEnum: z.ZodEnum<["entity_mismatch", "temporal_inconsistency", "semantic_conflict", "data_collision", "reference_missing", "version_conflict"]>;
export type ConflictType = z.infer<typeof ConflictTypeEnum>;
/**
 * Conflict Severity Enumeration
 */
export declare const ConflictSeverityEnum: z.ZodEnum<["low", "medium", "high", "critical"]>;
export type ConflictSeverity = z.infer<typeof ConflictSeverityEnum>;
/**
 * Conflict
 *
 * Represents a detected conflict between systems
 */
export declare const ConflictSchema: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["entity_mismatch", "temporal_inconsistency", "semantic_conflict", "data_collision", "reference_missing", "version_conflict"]>;
    severity: z.ZodEnum<["low", "medium", "high", "critical"]>;
    ragbits_data: z.ZodRecord<z.ZodString, z.ZodAny>;
    graphiti_data: z.ZodRecord<z.ZodString, z.ZodAny>;
    entity_id: z.ZodOptional<z.ZodString>;
    episode_id: z.ZodOptional<z.ZodString>;
    chunk_id: z.ZodOptional<z.ZodString>;
    description: z.ZodString;
    suggested_resolution: z.ZodOptional<z.ZodString>;
    detected_at_utc: z.ZodString;
    resolved: z.ZodBoolean;
    resolution_strategy: z.ZodOptional<z.ZodEnum<["source_wins", "target_wins", "newest_wins", "manual", "merge"]>>;
    resolution_notes: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    type: "entity_mismatch" | "temporal_inconsistency" | "semantic_conflict" | "data_collision" | "reference_missing" | "version_conflict";
    id: string;
    severity: "high" | "medium" | "low" | "critical";
    description: string;
    ragbits_data: Record<string, any>;
    graphiti_data: Record<string, any>;
    detected_at_utc: string;
    resolved: boolean;
    metadata?: Record<string, any> | undefined;
    episode_id?: string | undefined;
    chunk_id?: string | undefined;
    entity_id?: string | undefined;
    suggested_resolution?: string | undefined;
    resolution_strategy?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
    resolution_notes?: string | undefined;
}, {
    type: "entity_mismatch" | "temporal_inconsistency" | "semantic_conflict" | "data_collision" | "reference_missing" | "version_conflict";
    id: string;
    severity: "high" | "medium" | "low" | "critical";
    description: string;
    ragbits_data: Record<string, any>;
    graphiti_data: Record<string, any>;
    detected_at_utc: string;
    resolved: boolean;
    metadata?: Record<string, any> | undefined;
    episode_id?: string | undefined;
    chunk_id?: string | undefined;
    entity_id?: string | undefined;
    suggested_resolution?: string | undefined;
    resolution_strategy?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
    resolution_notes?: string | undefined;
}>;
export type Conflict = z.infer<typeof ConflictSchema>;
/**
 * Conflict Report
 *
 * Report of detected conflicts and their resolutions
 */
export declare const ConflictReportSchema: z.ZodObject<{
    sync_operation_id: z.ZodString;
    conflicts: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["entity_mismatch", "temporal_inconsistency", "semantic_conflict", "data_collision", "reference_missing", "version_conflict"]>;
        severity: z.ZodEnum<["low", "medium", "high", "critical"]>;
        ragbits_data: z.ZodRecord<z.ZodString, z.ZodAny>;
        graphiti_data: z.ZodRecord<z.ZodString, z.ZodAny>;
        entity_id: z.ZodOptional<z.ZodString>;
        episode_id: z.ZodOptional<z.ZodString>;
        chunk_id: z.ZodOptional<z.ZodString>;
        description: z.ZodString;
        suggested_resolution: z.ZodOptional<z.ZodString>;
        detected_at_utc: z.ZodString;
        resolved: z.ZodBoolean;
        resolution_strategy: z.ZodOptional<z.ZodEnum<["source_wins", "target_wins", "newest_wins", "manual", "merge"]>>;
        resolution_notes: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        type: "entity_mismatch" | "temporal_inconsistency" | "semantic_conflict" | "data_collision" | "reference_missing" | "version_conflict";
        id: string;
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        ragbits_data: Record<string, any>;
        graphiti_data: Record<string, any>;
        detected_at_utc: string;
        resolved: boolean;
        metadata?: Record<string, any> | undefined;
        episode_id?: string | undefined;
        chunk_id?: string | undefined;
        entity_id?: string | undefined;
        suggested_resolution?: string | undefined;
        resolution_strategy?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
        resolution_notes?: string | undefined;
    }, {
        type: "entity_mismatch" | "temporal_inconsistency" | "semantic_conflict" | "data_collision" | "reference_missing" | "version_conflict";
        id: string;
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        ragbits_data: Record<string, any>;
        graphiti_data: Record<string, any>;
        detected_at_utc: string;
        resolved: boolean;
        metadata?: Record<string, any> | undefined;
        episode_id?: string | undefined;
        chunk_id?: string | undefined;
        entity_id?: string | undefined;
        suggested_resolution?: string | undefined;
        resolution_strategy?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
        resolution_notes?: string | undefined;
    }>, "many">;
    resolutions: z.ZodArray<z.ZodObject<{
        conflict_id: z.ZodString;
        strategy: z.ZodEnum<["source_wins", "target_wins", "newest_wins", "manual", "merge"]>;
        applied_at_utc: z.ZodString;
        notes: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        strategy: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge";
        conflict_id: string;
        applied_at_utc: string;
        notes?: string | undefined;
    }, {
        strategy: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge";
        conflict_id: string;
        applied_at_utc: string;
        notes?: string | undefined;
    }>, "many">;
    unresolved: z.ZodArray<z.ZodString, "many">;
    total_conflicts: z.ZodNumber;
    resolved_count: z.ZodNumber;
    unresolved_count: z.ZodNumber;
    timestamp_utc: z.ZodString;
    correlation_id: z.ZodString;
}, "strip", z.ZodTypeAny, {
    correlation_id: string;
    timestamp_utc: string;
    sync_operation_id: string;
    conflicts: {
        type: "entity_mismatch" | "temporal_inconsistency" | "semantic_conflict" | "data_collision" | "reference_missing" | "version_conflict";
        id: string;
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        ragbits_data: Record<string, any>;
        graphiti_data: Record<string, any>;
        detected_at_utc: string;
        resolved: boolean;
        metadata?: Record<string, any> | undefined;
        episode_id?: string | undefined;
        chunk_id?: string | undefined;
        entity_id?: string | undefined;
        suggested_resolution?: string | undefined;
        resolution_strategy?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
        resolution_notes?: string | undefined;
    }[];
    resolutions: {
        strategy: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge";
        conflict_id: string;
        applied_at_utc: string;
        notes?: string | undefined;
    }[];
    unresolved: string[];
    total_conflicts: number;
    resolved_count: number;
    unresolved_count: number;
}, {
    correlation_id: string;
    timestamp_utc: string;
    sync_operation_id: string;
    conflicts: {
        type: "entity_mismatch" | "temporal_inconsistency" | "semantic_conflict" | "data_collision" | "reference_missing" | "version_conflict";
        id: string;
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        ragbits_data: Record<string, any>;
        graphiti_data: Record<string, any>;
        detected_at_utc: string;
        resolved: boolean;
        metadata?: Record<string, any> | undefined;
        episode_id?: string | undefined;
        chunk_id?: string | undefined;
        entity_id?: string | undefined;
        suggested_resolution?: string | undefined;
        resolution_strategy?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
        resolution_notes?: string | undefined;
    }[];
    resolutions: {
        strategy: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge";
        conflict_id: string;
        applied_at_utc: string;
        notes?: string | undefined;
    }[];
    unresolved: string[];
    total_conflicts: number;
    resolved_count: number;
    unresolved_count: number;
}>;
export type ConflictReport = z.infer<typeof ConflictReportSchema>;
/**
 * Sync Specification
 *
 * Specifies what to sync and how
 */
export declare const SyncSpecSchema: z.ZodObject<{
    direction: z.ZodEnum<["ragbits_to_graphiti", "graphiti_to_ragbits", "bidirectional"]>;
    entity_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    episode_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    chunk_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    date_range: z.ZodOptional<z.ZodObject<{
        start: z.ZodString;
        end: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        start: string;
        end: string;
    }, {
        start: string;
        end: string;
    }>>;
    filters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    conflict_resolution: z.ZodOptional<z.ZodEnum<["source_wins", "target_wins", "newest_wins", "manual", "merge"]>>;
    timeout_ms: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    direction: "ragbits_to_graphiti" | "graphiti_to_ragbits" | "bidirectional";
    timeout_ms?: number | undefined;
    metadata?: Record<string, any> | undefined;
    filters?: Record<string, any> | undefined;
    conflict_resolution?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
    entity_ids?: string[] | undefined;
    episode_ids?: string[] | undefined;
    chunk_ids?: string[] | undefined;
    date_range?: {
        start: string;
        end: string;
    } | undefined;
}, {
    direction: "ragbits_to_graphiti" | "graphiti_to_ragbits" | "bidirectional";
    timeout_ms?: number | undefined;
    metadata?: Record<string, any> | undefined;
    filters?: Record<string, any> | undefined;
    conflict_resolution?: "manual" | "source_wins" | "target_wins" | "newest_wins" | "merge" | undefined;
    entity_ids?: string[] | undefined;
    episode_ids?: string[] | undefined;
    chunk_ids?: string[] | undefined;
    date_range?: {
        start: string;
        end: string;
    } | undefined;
}>;
export type SyncSpec = z.infer<typeof SyncSpecSchema>;
/**
 * Enhanced Query
 *
 * Represents a query enhanced with knowledge graph information
 */
export declare const EnhancedQuerySchema: z.ZodObject<{
    original_query: z.ZodString;
    enhanced_query: z.ZodString;
    entities: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        labels: z.ZodArray<z.ZodString, "many">;
        boost_factor: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        labels: string[];
        boost_factor: number;
    }, {
        name: string;
        id: string;
        labels: string[];
        boost_factor: number;
    }>, "many">;
    boost_factors: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    entities: {
        name: string;
        id: string;
        labels: string[];
        boost_factor: number;
    }[];
    original_query: string;
    enhanced_query: string;
    metadata?: Record<string, any> | undefined;
    boost_factors?: Record<string, number> | undefined;
}, {
    entities: {
        name: string;
        id: string;
        labels: string[];
        boost_factor: number;
    }[];
    original_query: string;
    enhanced_query: string;
    metadata?: Record<string, any> | undefined;
    boost_factors?: Record<string, number> | undefined;
}>;
export type EnhancedQuery = z.infer<typeof EnhancedQuerySchema>;
/**
 * Boost Factor
 *
 * Represents a boost factor for retrieval
 */
export declare const BoostFactorSchema: z.ZodObject<{
    entity_id: z.ZodString;
    boost_value: z.ZodNumber;
    reason: z.ZodString;
    confidence: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    confidence: number;
    reason: string;
    entity_id: string;
    boost_value: number;
}, {
    confidence: number;
    reason: string;
    entity_id: string;
    boost_value: number;
}>;
export type BoostFactor = z.infer<typeof BoostFactorSchema>;
/**
 * Validate sync operation
 */
export declare function validateSyncOperation(data: unknown): {
    success: boolean;
    data?: SyncOperation;
    errors?: string[];
};
/**
 * Validate sync config
 */
export declare function validateSyncConfig(data: unknown): {
    success: boolean;
    data?: SyncConfig;
    errors?: string[];
};
/**
 * Validate sync result
 */
export declare function validateSyncResult(data: unknown): {
    success: boolean;
    data?: SyncResult;
    errors?: string[];
};
/**
 * Validate conflict
 */
export declare function validateConflict(data: unknown): {
    success: boolean;
    data?: Conflict;
    errors?: string[];
};
/**
 * Validate sync spec
 */
export declare function validateSyncSpec(data: unknown): {
    success: boolean;
    data?: SyncSpec;
    errors?: string[];
};
/**
 * Check if data is a valid SyncOperation
 */
export declare function isSyncOperation(data: unknown): data is SyncOperation;
/**
 * Check if data is a valid SyncResult
 */
export declare function isSyncResult(data: unknown): data is SyncResult;
/**
 * Check if data is a valid Conflict
 */
export declare function isConflict(data: unknown): data is Conflict;
/**
 * Default sync configuration
 */
export declare const DEFAULT_SYNC_CONFIG: SyncConfig;
/**
 * Create a sync operation
 */
export declare function createSyncOperation(type: SyncOperation['type'], source: SyncOperation['source'], target: SyncOperation['target'], direction: SyncDirection, correlationId: string): SyncOperation;
/**
 * Create a sync result
 */
export declare function createSyncResult(operationId: string, status: SyncStatus, direction: SyncDirection, correlationId: string, durationMs?: number): SyncResult;
/**
 * Create a conflict
 */
export declare function createConflict(type: ConflictType, severity: ConflictSeverity, ragbitsData: Record<string, any>, graphitiData: Record<string, any>, description: string, correlationId: string): Conflict;
//# sourceMappingURL=canonical.d.ts.map