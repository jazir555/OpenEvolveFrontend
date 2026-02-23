/**
 * Sync Manager
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify operations execute
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Circuit breakers and retries
 * - Law of Configuration Explicitness: All config via env vars
 *
 * Main orchestration for bidirectional RAGBits-Graphiti synchronization
 */
import { RAGBitsToGraphitiConfig } from './ragbits-to-graphiti';
import { GraphitiToRAGBitsConfig } from './graphiti-to-ragbits';
import { ConflictDetectorConfig, ConflictReport } from './conflict-detector';
import { SyncResult, SyncConfig, SyncDirection, SyncSpec, Conflict, ConflictResolution } from './canonical';
/**
 * Document for ingestion sync
 */
export interface Document {
    id: string;
    content: string;
    source: string;
    metadata?: Record<string, any>;
}
/**
 * Sync strategy
 */
export interface SyncStrategy {
    direction: SyncDirection;
    conflict_resolution: ConflictResolution;
    batch_size: number;
    enable_enhancement: boolean;
}
/**
 * Sync operation result
 */
export interface SyncOperationResult {
    sync_result: SyncResult;
    conflict_report?: ConflictReport;
    errors: Error[];
}
/**
 * Sync Manager configuration
 */
export interface SyncManagerConfig {
    ragbits: RAGBitsToGraphitiConfig;
    graphiti: GraphitiToRAGBitsConfig;
    conflict_detector: ConflictDetectorConfig;
    sync: SyncConfig;
}
/**
 * Sync Manager
 *
 * Orchestrates bidirectional synchronization between RAGBits and Graphiti
 */
export declare class SyncManager {
    private readonly config;
    private readonly logger;
    private readonly ragbitsToGraphiti;
    private readonly graphitiToRAGBits;
    private readonly conflictDetector;
    private readonly circuitBreaker;
    private readonly serviceName;
    private activeOperations;
    private stats;
    constructor(config?: Partial<SyncManagerConfig>);
    /**
     * Sync on document ingestion
     *
     * Triggered when a new document is ingested into RAGBits
     *
     * @param document - Ingested document
     * @param correlationId - Correlation ID for tracing
     * @returns Sync operation result
     */
    syncOnIngest(document: Document, correlationId: string): Promise<SyncOperationResult>;
    /**
     * Scheduled sync
     *
     * Runs periodically to keep systems in sync
     *
     * @param correlationId - Correlation ID for tracing
     * @returns Sync operation result
     */
    syncOnSchedule(correlationId: string): Promise<SyncOperationResult>;
    /**
     * Manual sync with specification
     *
     * @param spec - Sync specification
     * @param correlationId - Correlation ID for tracing
     * @returns Sync operation result
     */
    syncManual(spec: SyncSpec, correlationId: string): Promise<SyncOperationResult>;
    /**
     * Resolve conflicts
     *
     * @param conflicts - Conflicts to resolve
     * @param resolutionStrategy - Resolution strategy
     * @param correlationId - Correlation ID for tracing
     * @returns Resolution result
     */
    resolveConflicts(conflicts: Conflict[], resolutionStrategy: ConflictResolution, correlationId: string): Promise<{
        resolved: string[];
        failed: string[];
        errors: Error[];
    }>;
    /**
     * Get statistics
     *
     * @returns Sync statistics
     */
    getStats(): {
        active_operations: number;
        success_rate: number;
        avg_duration_ms: number;
        conflict_rate: number;
        total_syncs: number;
        successful_syncs: number;
        failed_syncs: number;
        conflicts_detected: number;
        conflicts_resolved: number;
        total_duration_ms: number;
    };
    /**
     * Validate configuration from environment variables
     */
    private validateConfiguration;
    /**
     * Build configuration from environment and defaults
     */
    private buildConfig;
    /**
     * Perform bidirectional sync
     */
    private performBidirectionalSync;
    /**
     * Chunk document (placeholder)
     */
    private chunkDocument;
    /**
     * Apply conflict resolution strategy
     */
    private applyConflictResolution;
    /**
     * Fetch RAGBits data (placeholder)
     */
    private fetchRAGBitsData;
    /**
     * Fetch Graphiti data (placeholder)
     */
    private fetchGraphitiData;
    /**
     * Fetch chunks by IDs (placeholder)
     */
    private fetchChunks;
    /**
     * Fetch entities by IDs (placeholder)
     */
    private fetchEntities;
    /**
     * Update statistics
     */
    private updateStats;
}
export default SyncManager;
//# sourceMappingURL=sync-manager.d.ts.map