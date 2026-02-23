"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.SCEAdapter = void 0;
const uuid_1 = require("uuid");
const logger_1 = require("../../lib/logger");
const circuit_breaker_1 = require("../../lib/circuit-breaker");
const retry_1 = require("../../lib/retry");
const rese_canonical_1 = require("../../schemas/rese-canonical");
const rese_sce_1 = require("../../lib/rese-sce");
/**
 * Load configuration from environment variables
 *
 * Law of Configuration Explicitness: All config via env vars
 * Crashes immediately if required config is missing
 */
function loadAdapterConfig() {
    const config = {
        RESE_SCE_URL: process.env.RESE_SCE_URL,
        RESE_SCE_TIMEOUT_MS: parseInt(process.env.RESE_SCE_TIMEOUT_MS || '30000', 10),
        MAX_RETRIES: parseInt(process.env.SCE_ADAPTER_MAX_RETRIES || '3', 10),
        INITIAL_RETRY_DELAY_MS: parseInt(process.env.SCE_ADAPTER_INITIAL_DELAY_MS || '1000', 10),
        MAX_RETRY_DELAY_MS: parseInt(process.env.SCE_ADAPTER_MAX_DELAY_MS || '10000', 10),
        CIRCUIT_BREAKER_THRESHOLD: parseInt(process.env.SCE_ADAPTER_CB_THRESHOLD || '5', 10),
        CIRCUIT_BREAKER_TIMEOUT_MS: parseInt(process.env.SCE_ADAPTER_CB_TIMEOUT_MS || '60000', 10),
        DLQ_ENABLED: process.env.SCE_DLQ_ENABLED !== 'false', // default true
        DLQ_MAX_SIZE: parseInt(process.env.SCE_DLQ_MAX_SIZE || '1000', 10),
    };
    // Validate configuration
    if (config.RESE_SCE_TIMEOUT_MS <= 0) {
        throw new Error('RESE_SCE_TIMEOUT_MS must be positive');
    }
    if (config.MAX_RETRIES < 0) {
        throw new Error('SCE_ADAPTER_MAX_RETRIES must be non-negative');
    }
    return config;
}
/**
 * Dead Letter Queue for logic failures
 *
 * From CLAUDE.md: Logic failures → Dead Letter Queue
 * Do not block the pipeline for bad data
 */
class DeadLetterQueue {
    constructor(maxSize, logger) {
        this.queue = [];
        this.maxSize = maxSize;
        this.logger = logger.child({ component: 'DLQ' });
    }
    /**
     * Add entry to DLQ
     */
    add(operation, payload, error, correlationId, retryCount) {
        // Remove oldest entry if queue is full
        if (this.queue.length >= this.maxSize) {
            const removed = this.queue.shift();
            this.logger.warn('DLQ full, removing oldest entry', {
                removed_entry_id: removed?.id,
            });
        }
        const entry = {
            id: (0, uuid_1.v4)(),
            timestamp: new Date(),
            operation,
            payload,
            error,
            correlation_id: correlationId,
            retry_count: retryCount,
        };
        this.queue.push(entry);
        this.logger.error('Operation added to DLQ', {
            dlq_entry_id: entry.id,
            operation,
            error,
            correlation_id: correlationId,
            queue_size: this.queue.length,
        });
    }
    /**
     * Get all entries in DLQ
     */
    getAll() {
        return [...this.queue];
    }
    /**
     * Clear DLQ
     */
    clear() {
        this.queue = [];
        this.logger.info('DLQ cleared');
    }
    /**
     * Get DLQ size
     */
    size() {
        return this.queue.length;
    }
}
// ============================================================================
// SCE ADAPTER
// ============================================================================
/**
 * SCE Adapter Class
 *
 * Main adapter for RESE Symbolic Constraint Engine.
 * Implements the Anti-Corruption Layer pattern.
 */
class SCEAdapter {
    constructor() {
        this.logger = new logger_1.Logger('SCEAdapter');
        this.config = loadAdapterConfig();
        this.sce = new rese_sce_1.SymbolicConstraintEngine();
        // Initialize circuit breaker
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: this.config.CIRCUIT_BREAKER_THRESHOLD,
            timeout_ms: this.config.CIRCUIT_BREAKER_TIMEOUT_MS,
            onStateChange: (oldState, newState) => {
                this.logger.warn('SCEAdapter circuit breaker state changed', {
                    old_state: oldState,
                    new_state: newState,
                });
            },
        });
        // Initialize DLQ
        this.dlq = new DeadLetterQueue(this.config.DLQ_MAX_SIZE, this.logger);
        this.logger.info('SCEAdapter initialized', {
            max_retries: this.config.MAX_RETRIES,
            timeout_ms: this.config.RESE_SCE_TIMEOUT_MS,
            cb_threshold: this.config.CIRCUIT_BREAKER_THRESHOLD,
            dlq_enabled: this.config.DLQ_ENABLED,
        });
    }
    // ==========================================================================
    // EPISTEMIC AUDIT (PHASE I)
    // ==========================================================================
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
    async performEpistemicAudit(request) {
        const correlationId = request.correlation_id || (0, uuid_1.v4)();
        this.logger.info('Performing Epistemic Audit (Phase I)', {
            correlation_id: correlationId,
            problem_description: request.problem_description,
            pattern_count: request.failure_patterns.length,
        });
        try {
            // Execute through circuit breaker with retry logic
            const result = await this.executeWithRetry(() => this.circuitBreaker.execute(async () => {
                return await this.sce.performEpistemicAudit(request.problem_description, request.failure_patterns, correlationId);
            }), correlationId, 'performEpistemicAudit');
            // Validate result against canonical schema
            const validation = (0, rese_canonical_1.validateEpistemicAuditResult)(result);
            if (!validation.success) {
                throw new Error(`SCE result validation failed: ${validation.errors?.join(', ')}`);
            }
            this.logger.info('Epistemic Audit completed successfully', {
                correlation_id: correlationId,
                audit_id: result.audit_id,
                tacit_assumptions_found: result.tacit_assumptions?.length || 0,
                contradictions_found: result.contradictions?.length || 0,
            });
            return result;
        }
        catch (error) {
            return this.handleError('performEpistemicAudit', error, request, correlationId);
        }
    }
    // ==========================================================================
    // CONSTRAINT MANAGEMENT
    // ==========================================================================
    /**
     * Add constraint to SCE
     *
     * Law of Idempotency: Safe to run multiple times
     *
     * @param constraint - Constraint to add
     * @param correlationId - Correlation ID for tracing
     * @returns Addition result
     */
    async addConstraint(constraint, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info('Adding constraint', {
            correlation_id: cid,
            type: constraint.type,
            category: constraint.category,
            description: constraint.description,
        });
        try {
            const result = await this.executeWithRetry(() => this.circuitBreaker.execute(async () => {
                const internalConstraint = {
                    constraint_id: constraint.constraint_id || (0, uuid_1.v4)(),
                    type: constraint.type === 'hard' ? rese_sce_1.ConstraintType.HARD : rese_sce_1.ConstraintType.SOFT,
                    category: constraint.category,
                    description: constraint.description,
                    dependencies: constraint.dependencies || [],
                    created_at: new Date(),
                };
                return await this.sce.addConstraint(internalConstraint, cid);
            }), cid, 'addConstraint');
            this.logger.info('Constraint added successfully', {
                correlation_id: cid,
                added: result.added,
                updated: result.updated,
            });
            return result;
        }
        catch (error) {
            this.handleError('addConstraint', error, constraint, cid);
            throw error; // Re-throw for synchronous operations
        }
    }
    /**
     * Remove constraint from SCE
     *
     * Law of Idempotency: Safe to run multiple times
     *
     * @param constraintId - ID of constraint to remove
     * @param correlationId - Correlation ID for tracing
     * @returns Removal result
     */
    async removeConstraint(constraintId, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info('Removing constraint', {
            correlation_id: cid,
            constraint_id: constraintId,
        });
        try {
            const result = await this.executeWithRetry(() => this.circuitBreaker.execute(async () => {
                return await this.sce.removeConstraint(constraintId, cid);
            }), cid, 'removeConstraint');
            this.logger.info('Constraint removed successfully', {
                correlation_id: cid,
                removed: result.removed,
            });
            return result;
        }
        catch (error) {
            this.handleError('removeConstraint', error, { constraint_id: constraintId }, cid);
            throw error; // Re-throw for synchronous operations
        }
    }
    /**
     * Get constraint by ID
     *
     * @param constraintId - ID of constraint to retrieve
     * @returns Constraint or null if not found
     */
    getConstraint(constraintId) {
        return this.sce.getConstraint(constraintId);
    }
    /**
     * Get all constraints
     *
     * @returns Array of all constraints
     */
    getAllConstraints() {
        return this.sce.getAllConstraints();
    }
    // ==========================================================================
    // CONTRADICTION DETECTION
    // ==========================================================================
    /**
     * Detect contradictions in constraint set
     *
     * @param correlationId - Correlation ID for tracing
     * @returns Contradiction detection result
     */
    async detectContradictions(correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info('Detecting contradictions', {
            correlation_id: cid,
        });
        try {
            const result = await this.executeWithRetry(() => this.circuitBreaker.execute(async () => {
                return await this.sce.detectContradictions(cid);
            }), cid, 'detectContradictions');
            this.logger.info('Contradiction detection completed', {
                correlation_id: cid,
                contradictions_found: result.contradictions.length,
                detection_time_ms: result.detection_time_ms,
            });
            return result;
        }
        catch (error) {
            this.handleError('detectContradictions', error, {}, cid);
            throw error; // Re-throw for synchronous operations
        }
    }
    // ==========================================================================
    // UTILITY METHODS
    // ==========================================================================
    /**
     * Execute function with retry logic
     *
     * From CLAUDE.md: Transient failures → exponential backoff retry
     *
     * @private
     */
    async executeWithRetry(fn, correlationId, operation) {
        return (0, retry_1.retry)(fn, this.config.MAX_RETRIES, this.config.INITIAL_RETRY_DELAY_MS, this.config.MAX_RETRY_DELAY_MS, (error, attempt) => {
            this.logger.warn(`Retry attempt ${attempt}`, {
                correlation_id: correlationId,
                operation,
                error_name: error.name,
                error_message: error.message,
            });
        });
    }
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
    handleError(operation, error, payload, correlationId) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        const errorType = this.classifyError(error);
        if (errorType === 'logic') {
            // Logic failure → Dead Letter Queue
            if (this.config.DLQ_ENABLED) {
                this.dlq.add(operation, payload, errorMessage, correlationId, 0);
            }
            this.logger.error('Logic failure, operation sent to DLQ', error, {
                correlation_id: correlationId,
                operation,
                dlq_enabled: this.config.DLQ_ENABLED,
            });
        }
        else if (errorType === 'system') {
            // System failure → Circuit breaker (already triggered)
            this.logger.error('System failure, circuit breaker triggered', error, {
                correlation_id: correlationId,
                operation,
                circuit_state: this.circuitBreaker.getState(),
            });
        }
        else {
            // Transient failure (will be retried)
            this.logger.warn('Transient failure, will retry', {
                correlation_id: correlationId,
                operation,
                error_message: errorMessage,
            });
        }
        throw error;
    }
    /**
     * Classify error type
     *
     * @private
     */
    classifyError(error) {
        const errorMessage = error instanceof Error ? error.message.toLowerCase() : '';
        // System failures (circuit breaker)
        if (errorMessage.includes('circuit') ||
            errorMessage.includes('timeout') ||
            errorMessage.includes('econnrefused') ||
            errorMessage.includes('enotfound')) {
            return 'system';
        }
        // Logic failures (bad data, validation errors)
        if (errorMessage.includes('validation') ||
            errorMessage.includes('invalid') ||
            errorMessage.includes('not found') ||
            errorMessage.includes('duplicate')) {
            return 'logic';
        }
        // Default to transient
        return 'transient';
    }
    /**
     * Get adapter statistics
     */
    getStats() {
        return {
            circuit_breaker: this.circuitBreaker.getStats(),
            dlq_size: this.dlq.size(),
            sce_stats: this.sce.getStats(),
        };
    }
    /**
     * Get Dead Letter Queue entries
     */
    getDLQEntries() {
        return this.dlq.getAll();
    }
    /**
     * Clear Dead Letter Queue
     */
    clearDLQ() {
        this.dlq.clear();
    }
    /**
     * Reset circuit breakers
     */
    resetCircuitBreakers() {
        this.circuitBreaker.reset();
        this.sce.resetCircuitBreakers();
    }
    /**
     * Health check
     */
    async healthCheck() {
        const stats = this.getStats();
        const healthy = stats.circuit_breaker.state === circuit_breaker_1.CircuitState.CLOSED &&
            stats.dlq_size < this.config.DLQ_MAX_SIZE;
        return {
            healthy,
            circuit_state: stats.circuit_breaker.state,
            dlq_size: stats.dlq_size,
            constraint_count: stats.sce_stats.constraint_count,
        };
    }
}
exports.SCEAdapter = SCEAdapter;
// ============================================================================
// EXPORTS
// ============================================================================
exports.default = SCEAdapter;
//# sourceMappingURL=sce-adapter.js.map