/**
 * RESE Symbolic Constraint Engine (SCE) Adapter
 *
 * Adapter for RESE Phase I: Epistemic Audit functionality.
 * Implements the Anti-Corruption Layer (ACL) pattern.
 *
 * Architecture:
 * - Transforms canonical requests to SCE internal format
 * - Executes SCE operations with circuit breaker protection
 * - Transforms SCE results to canonical format
 * - Handles failures according to CLAUDE.md laws
 *
 * Failure Management:
 * - Transient failures → exponential backoff retry
 * - Logic failures → Dead Letter Queue
 * - System failures → Circuit breaker
 *
 * Follows CLAUDE.md Laws:
 * - Law of the "Air Gap": No direct imports from core-projects
 * - Law of Idempotency: All operations safe to run 100x
 * - Law of Configuration Explicitness: Config via env vars
 * - Law of UTC: All timestamps in UTC
 * - Timeout Enforcement: All operations have timeouts
 */
import { EpistemicAuditResult } from '../../schemas/rese-canonical';
import { Constraint } from '../../lib/rese-sce';
interface DLQEntry {
    id: string;
    timestamp: Date;
    operation: string;
    payload: any;
    error: string;
    correlation_id: string;
    retry_count: number;
}
/**
 * SCE Adapter Class
 *
 * Main adapter for RESE Symbolic Constraint Engine.
 * Implements the Anti-Corruption Layer pattern.
 */
export declare class SCEAdapter {
    private logger;
    private config;
    private sce;
    private circuitBreaker;
    private dlq;
    constructor();
    /**
     * Perform Epistemic Audit (Phase I)
     *
     * From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification
     *
     * Failure Management:
     * - Transient failures → exponential backoff retry
     * - Logic failures → Dead Letter Queue
     * - System failures → Circuit breaker
     *
     * @param request - Epistemic audit request
     * @returns Canonical EpistemicAuditResult
     */
    performEpistemicAudit(request: {
        problem_description: string;
        failure_patterns: Array<{
            pattern_description: string;
            failure_rate: number;
            data_points: number;
        }>;
        correlation_id?: string;
    }): Promise<EpistemicAuditResult>;
    /**
     * Add constraint to SCE
     *
     * Law of Idempotency: Safe to run multiple times
     *
     * @param constraint - Constraint to add
     * @param correlationId - Correlation ID for tracing
     * @returns Addition result
     */
    addConstraint(constraint: {
        constraint_id?: string;
        type: 'hard' | 'soft';
        category: ConstraintCategory;
        description: string;
        dependencies?: string[];
    }, correlationId?: string): Promise<{
        added: boolean;
        updated: boolean;
    }>;
    /**
     * Remove constraint from SCE
     *
     * Law of Idempotency: Safe to run multiple times
     *
     * @param constraintId - ID of constraint to remove
     * @param correlationId - Correlation ID for tracing
     * @returns Removal result
     */
    removeConstraint(constraintId: string, correlationId?: string): Promise<{
        removed: boolean;
    }>;
    /**
     * Get constraint by ID
     *
     * @param constraintId - ID of constraint to retrieve
     * @returns Constraint or null if not found
     */
    getConstraint(constraintId: string): Constraint | null;
    /**
     * Get all constraints
     *
     * @returns Array of all constraints
     */
    getAllConstraints(): Constraint[];
    /**
     * Detect contradictions in constraint set
     *
     * @param correlationId - Correlation ID for tracing
     * @returns Contradiction detection result
     */
    detectContradictions(correlationId?: string): Promise<{
        contradictions: Array<{
            constraint1_id: string;
            constraint2_id: string;
            type: string;
            contradiction_set_size: number;
            rollback_steps: number;
            affected_premises: string[];
            detected_at: Date;
        }>;
        total_checked: number;
        contradiction_found: boolean;
        largest_contradiction_set: number;
        detection_time_ms: number;
    }>;
    /**
     * Execute function with retry logic
     *
     * From CLAUDE.md: Transient failures → exponential backoff retry
     *
     * @private
     */
    private executeWithRetry;
    /**
     * Handle error according to failure type
     *
     * From CLAUDE.md:
     * - Transient failures → exponential backoff retry (already handled)
     * - Logic failures → Dead Letter Queue
     * - System failures → Circuit breaker (already handled)
     *
     * @private
     */
    private handleError;
    /**
     * Classify error type
     *
     * @private
     */
    private classifyError;
    /**
     * Get adapter statistics
     */
    getStats(): {
        circuit_breaker: any;
        dlq_size: number;
        sce_stats: any;
    };
    /**
     * Get Dead Letter Queue entries
     */
    getDLQEntries(): DLQEntry[];
    /**
     * Clear Dead Letter Queue
     */
    clearDLQ(): void;
    /**
     * Reset circuit breakers
     */
    resetCircuitBreakers(): void;
    /**
     * Health check
     */
    healthCheck(): Promise<{
        healthy: boolean;
        circuit_state: string;
        dlq_size: number;
        constraint_count: number;
    }>;
}
export default SCEAdapter;
//# sourceMappingURL=sce-adapter.d.ts.map