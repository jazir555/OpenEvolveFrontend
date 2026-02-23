/**
 * Conflict Detector
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Detect conflicts from actual data
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Graceful handling of detection errors
 *
 * Detects conflicts between RAGBits and Graphiti data during synchronization
 */
import { Conflict, ConflictReport, ConflictResolution, SyncOperation } from './canonical';
/**
 * RAGBits data for conflict detection
 */
export interface RAGBitsData {
    chunks?: DocumentChunkData[];
    entities?: Map<string, any>;
    metadata?: Record<string, any>;
}
/**
 * Document chunk data
 */
export interface DocumentChunkData {
    id: string;
    content: string;
    source: string;
    chunk_index: number;
    metadata?: Record<string, any>;
    timestamp: string;
}
/**
 * Graphiti data for conflict detection
 */
export interface GraphitiData {
    episodes?: GraphitiEpisodeData[];
    entities?: Map<string, GraphitiEntityData>;
    relationships?: Map<string, GraphitiRelationshipData>;
    metadata?: Record<string, any>;
}
/**
 * Graphiti episode data
 */
export interface GraphitiEpisodeData {
    id: string;
    name: string;
    content: string;
    valid_at: string;
    created_at: string;
    entity_edges: string[];
    metadata?: Record<string, any>;
}
/**
 * Graphiti entity data
 */
export interface GraphitiEntityData {
    id: string;
    name: string;
    labels: string[];
    summary?: string;
    attributes: Record<string, any>;
    created_at: string;
    updated_at?: string;
}
/**
 * Graphiti relationship data
 */
export interface GraphitiRelationshipData {
    id: string;
    source_entity_id: string;
    target_entity_id: string;
    relation_type: string;
    fact: string;
    valid_at: string;
    created_at: string;
}
/**
 * Conflict detection configuration
 */
export interface ConflictDetectorConfig {
    enable_entity_detection: boolean;
    enable_temporal_detection: boolean;
    enable_semantic_detection: boolean;
    semantic_similarity_threshold: number;
    temporal_drift_threshold_ms: number;
    auto_resolve_minor_conflicts: boolean;
}
/**
 * Conflict Detector
 *
 * Detects conflicts between RAGBits and Graphiti data
 */
export declare class ConflictDetector {
    private readonly config;
    private readonly logger;
    private readonly serviceName;
    constructor(config: ConflictDetectorConfig);
    /**
     * Detect conflicts between RAGBits and Graphiti data
     *
     * @param ragbitsData - Data from RAGBits system
     * @param graphitiData - Data from Graphiti system
     * @param syncOperation - Sync operation context
     * @returns Conflict report
     */
    detectConflicts(ragbitsData: RAGBitsData, graphitiData: GraphitiData, syncOperation: SyncOperation): Promise<ConflictReport>;
    /**
     * Detect entity conflicts between systems
     *
     * @param ragbitsData - RAGBits data
     * @param graphitiData - Graphiti data
     * @param correlationId - Correlation ID for tracing
     * @returns Array of entity conflicts
     */
    private detectEntityConflicts;
    /**
     * Detect temporal conflicts between systems
     *
     * @param ragbitsData - RAGBits data
     * @param graphitiData - Graphiti data
     * @param correlationId - Correlation ID for tracing
     * @returns Array of temporal conflicts
     */
    private detectTemporalConflicts;
    /**
     * Detect semantic conflicts between systems
     *
     * @param ragbitsData - RAGBits data
     * @param graphitiData - Graphiti data
     * @param correlationId - Correlation ID for tracing
     * @returns Array of semantic conflicts
     */
    private detectSemanticConflicts;
    /**
     * Detect attribute mismatches between entities
     *
     * @param ragbitsEntity - Entity from RAGBits
     * @param graphitiEntity - Entity from Graphiti
     * @param correlationId - Correlation ID for tracing
     * @returns Array of conflicts
     */
    private detectAttributeMismatches;
    /**
     * Extract entities from document chunks
     *
     * @param chunks - Document chunks
     * @returns Map of entity name to entity data
     */
    private extractEntitiesFromChunks;
    /**
     * Calculate semantic similarity between two texts
     *
     * @param text1 - First text
     * @param text2 - Second text
     * @returns Similarity score (0-1)
     */
    private calculateSemanticSimilarity;
    /**
     * Calculate severity of temporal conflict
     *
     * @param driftMs - Temporal drift in milliseconds
     * @returns Severity level
     */
    private calculateTemporalSeverity;
    /**
     * Auto-resolve conflicts if enabled
     *
     * @param conflicts - Array of conflicts
     * @param resolutionStrategy - Resolution strategy to apply
     * @returns Array of resolved conflict IDs
     */
    autoResolveConflicts(conflicts: Conflict[], resolutionStrategy: ConflictResolution): string[];
}
export default ConflictDetector;
//# sourceMappingURL=conflict-detector.d.ts.map