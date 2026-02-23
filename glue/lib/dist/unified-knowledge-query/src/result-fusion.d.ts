/**
 * Result Fusion
 *
 * Merges and normalizes results from multiple knowledge systems.
 *
 * Federation Constitution Compliance:
 * - Anti-Corruption Layer: All results normalized to canonical format
 * - Conflict Resolution: Detects and resolves data conflicts
 * - Idempotency: Merge operations are deterministic
 */
import { KnowledgeItem, SystemSource, UnifiedQueryResult } from './canonical';
/**
 * System Result with metadata
 */
interface SystemResult {
    system: SystemSource;
    items: KnowledgeItem[];
    queryTimeMs: number;
    success: boolean;
    error?: string;
}
/**
 * Result Fusion Class
 */
export declare class ResultFusion {
    private logger;
    constructor();
    /**
     * Merge results from multiple systems
     */
    merge(results: SystemResult[], query: string, correlationId: string): Promise<UnifiedQueryResult>;
    /**
     * Normalize scores across systems
     * Uses min-max normalization per system
     */
    private normalizeScores;
    /**
     * Detect conflicts between results
     * Currently checks for duplicate IDs with different content
     */
    private detectConflicts;
    /**
     * Resolve conflicts using highest confidence strategy
     */
    private resolveConflicts;
    /**
     * Rank results by combined score
     * Combines normalized score and confidence
     */
    private rankResults;
    /**
     * Build source metadata from results
     */
    private buildSourceMetadata;
    /**
     * Calculate overall confidence from all results
     */
    private calculateOverallConfidence;
    /**
     * Deduplicate results by ID
     * Keeps highest confidence version
     */
    deduplicateById(results: KnowledgeItem[]): KnowledgeItem[];
    /**
     * Filter results by minimum confidence
     */
    filterByConfidence(results: KnowledgeItem[], minConfidence: number): KnowledgeItem[];
    /**
     * Filter results by knowledge type
     */
    filterByType(results: KnowledgeItem[], types: string[]): KnowledgeItem[];
    /**
     * Limit results to top N
     */
    limitResults(results: KnowledgeItem[], limit: number): KnowledgeItem[];
}
/**
 * Default fusion instance
 */
export declare const resultFusion: ResultFusion;
export {};
//# sourceMappingURL=result-fusion.d.ts.map