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

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../../lib/logger';
import { CircuitBreaker, CircuitState } from '../../lib/circuit-breaker';
import { retry } from '../../lib/retry';
import {
  EpistemicAuditResult,
  validateEpistemicAuditResult,
} from '../../schemas/rese-canonical';
import {
  SymbolicConstraintEngine,
  Constraint,
  ConstraintType,
  ConstraintCategoryInternal,
} from '../../lib/rese-sce';

// ============================================================================
// CONFIGURATION (Law of Configuration Explicitness)
// ============================================================================

interface SCEAdapterConfig {
  // Service URLs
  RESE_SCE_URL?: string;
  RESE_SCE_TIMEOUT_MS: number;

  // Retry settings
  MAX_RETRIES: number;
  INITIAL_RETRY_DELAY_MS: number;
  MAX_RETRY_DELAY_MS: number;

  // Circuit breaker settings
  CIRCUIT_BREAKER_THRESHOLD: number;
  CIRCUIT_BREAKER_TIMEOUT_MS: number;

  // DLQ settings
  DLQ_ENABLED: boolean;
  DLQ_MAX_SIZE: number;
}

/**
 * Load configuration from environment variables
 *
 * Law of Configuration Explicitness: All config via env vars
 * Crashes immediately if required config is missing
 */
function loadAdapterConfig(): SCEAdapterConfig {
  const config: SCEAdapterConfig = {
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

// ============================================================================
// DEAD LETTER QUEUE
// ============================================================================

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
 * Dead Letter Queue for logic failures
 *
 * From CLAUDE.md: Logic failures → Dead Letter Queue
 * Do not block the pipeline for bad data
 */
class DeadLetterQueue {
  private queue: DLQEntry[] = [];
  private maxSize: number;
  private logger: Logger;

  constructor(maxSize: number, logger: Logger) {
    this.maxSize = maxSize;
    this.logger = logger.child({ component: 'DLQ' });
  }

  /**
   * Add entry to DLQ
   */
  add(
    operation: string,
    payload: any,
    error: string,
    correlationId: string,
    retryCount: number
  ): void {
    // Remove oldest entry if queue is full
    if (this.queue.length >= this.maxSize) {
      const removed = this.queue.shift();
      this.logger.warn('DLQ full, removing oldest entry', {
        removed_entry_id: removed?.id,
      });
    }

    const entry: DLQEntry = {
      id: uuidv4(),
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
  getAll(): DLQEntry[] {
    return [...this.queue];
  }

  /**
   * Clear DLQ
   */
  clear(): void {
    this.queue = [];
    this.logger.info('DLQ cleared');
  }

  /**
   * Get DLQ size
   */
  size(): number {
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
export class SCEAdapter {
  private logger: Logger;
  private config: SCEAdapterConfig;
  private sce: SymbolicConstraintEngine;
  private circuitBreaker: CircuitBreaker;
  private dlq: DeadLetterQueue;

  constructor() {
    this.logger = new Logger('SCEAdapter');
    this.config = loadAdapterConfig();
    this.sce = new SymbolicConstraintEngine();

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
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
  async performEpistemicAudit(request: {
    problem_description: string;
    failure_patterns: Array<{
      pattern_description: string;
      failure_rate: number;
      data_points: number;
    }>;
    correlation_id?: string;
  }): Promise<EpistemicAuditResult> {
    const correlationId = request.correlation_id || uuidv4();

    this.logger.info('Performing Epistemic Audit (Phase I)', {
      correlation_id: correlationId,
      problem_description: request.problem_description,
      pattern_count: request.failure_patterns.length,
    });

    try {
      // Execute through circuit breaker with retry logic
      const result = await this.executeWithRetry(
        () =>
          this.circuitBreaker.execute(async () => {
            return await this.sce.performEpistemicAudit(
              request.problem_description,
              request.failure_patterns,
              correlationId
            );
          }),
        correlationId,
        'performEpistemicAudit'
      );

      // Validate result against canonical schema
      const validation = validateEpistemicAuditResult(result);
      if (!validation.success) {
        throw new Error(
          `SCE result validation failed: ${validation.errors?.join(', ')}`
        );
      }

      this.logger.info('Epistemic Audit completed successfully', {
        correlation_id: correlationId,
        audit_id: result.audit_id,
        tacit_assumptions_found: result.tacit_assumptions?.length || 0,
        contradictions_found: result.contradictions?.length || 0,
      });

      return result;
    } catch (error) {
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
  async addConstraint(
    constraint: {
      constraint_id?: string;
      type: 'hard' | 'soft';
      category: ConstraintCategory;
      description: string;
      dependencies?: string[];
    },
    correlationId?: string
  ): Promise<{ added: boolean; updated: boolean }> {
    const cid = correlationId || uuidv4();

    this.logger.info('Adding constraint', {
      correlation_id: cid,
      type: constraint.type,
      category: constraint.category,
      description: constraint.description,
    });

    try {
      const result = await this.executeWithRetry(
        () =>
          this.circuitBreaker.execute(async () => {
            const internalConstraint: Constraint = {
              constraint_id: constraint.constraint_id || uuidv4(),
              type:
                constraint.type === 'hard' ? ConstraintType.HARD : ConstraintType.SOFT,
              category: constraint.category as any,
              description: constraint.description,
              dependencies: constraint.dependencies || [],
              created_at: new Date(),
            };

            return await this.sce.addConstraint(internalConstraint, cid);
          }),
        cid,
        'addConstraint'
      );

      this.logger.info('Constraint added successfully', {
        correlation_id: cid,
        added: result.added,
        updated: result.updated,
      });

      return result;
    } catch (error) {
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
  async removeConstraint(
    constraintId: string,
    correlationId?: string
  ): Promise<{ removed: boolean }> {
    const cid = correlationId || uuidv4();

    this.logger.info('Removing constraint', {
      correlation_id: cid,
      constraint_id: constraintId,
    });

    try {
      const result = await this.executeWithRetry(
        () => this.circuitBreaker.execute(async () => {
          return await this.sce.removeConstraint(constraintId, cid);
        }),
        cid,
        'removeConstraint'
      );

      this.logger.info('Constraint removed successfully', {
        correlation_id: cid,
        removed: result.removed,
      });

      return result;
    } catch (error) {
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
  getConstraint(constraintId: string): Constraint | null {
    return this.sce.getConstraint(constraintId);
  }

  /**
   * Get all constraints
   *
   * @returns Array of all constraints
   */
  getAllConstraints(): Constraint[] {
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
  async detectContradictions(correlationId?: string): Promise<{
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
  }> {
    const cid = correlationId || uuidv4();

    this.logger.info('Detecting contradictions', {
      correlation_id: cid,
    });

    try {
      const result = await this.executeWithRetry(
        () => this.circuitBreaker.execute(async () => {
          return await this.sce.detectContradictions(cid);
        }),
        cid,
        'detectContradictions'
      );

      this.logger.info('Contradiction detection completed', {
        correlation_id: cid,
        contradictions_found: result.contradictions.length,
        detection_time_ms: result.detection_time_ms,
      });

      return result;
    } catch (error) {
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
  private async executeWithRetry<T>(
    fn: () => Promise<T>,
    correlationId: string,
    operation: string
  ): Promise<T> {
    return retry(
      fn,
      this.config.MAX_RETRIES,
      this.config.INITIAL_RETRY_DELAY_MS,
      this.config.MAX_RETRY_DELAY_MS,
      (error, attempt) => {
        this.logger.warn(`Retry attempt ${attempt}`, {
          correlation_id: correlationId,
          operation,
          error_name: error.name,
          error_message: error.message,
        });
      }
    );
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
  private handleError(
    operation: string,
    error: any,
    payload: any,
    correlationId: string
  ): never {
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
    } else if (errorType === 'system') {
      // System failure → Circuit breaker (already triggered)
      this.logger.error('System failure, circuit breaker triggered', error, {
        correlation_id: correlationId,
        operation,
        circuit_state: this.circuitBreaker.getState(),
      });
    } else {
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
  private classifyError(error: any): 'transient' | 'logic' | 'system' {
    const errorMessage = error instanceof Error ? error.message.toLowerCase() : '';

    // System failures (circuit breaker)
    if (
      errorMessage.includes('circuit') ||
      errorMessage.includes('timeout') ||
      errorMessage.includes('econnrefused') ||
      errorMessage.includes('enotfound')
    ) {
      return 'system';
    }

    // Logic failures (bad data, validation errors)
    if (
      errorMessage.includes('validation') ||
      errorMessage.includes('invalid') ||
      errorMessage.includes('not found') ||
      errorMessage.includes('duplicate')
    ) {
      return 'logic';
    }

    // Default to transient
    return 'transient';
  }

  /**
   * Get adapter statistics
   */
  getStats(): {
    circuit_breaker: any;
    dlq_size: number;
    sce_stats: any;
  } {
    return {
      circuit_breaker: this.circuitBreaker.getStats(),
      dlq_size: this.dlq.size(),
      sce_stats: this.sce.getStats(),
    };
  }

  /**
   * Get Dead Letter Queue entries
   */
  getDLQEntries(): DLQEntry[] {
    return this.dlq.getAll();
  }

  /**
   * Clear Dead Letter Queue
   */
  clearDLQ(): void {
    this.dlq.clear();
  }

  /**
   * Reset circuit breakers
   */
  resetCircuitBreakers(): void {
    this.circuitBreaker.reset();
    this.sce.resetCircuitBreakers();
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    circuit_state: string;
    dlq_size: number;
    constraint_count: number;
  }> {
    const stats = this.getStats();
    const healthy =
      stats.circuit_breaker.state === CircuitState.CLOSED &&
      stats.dlq_size < this.config.DLQ_MAX_SIZE;

    return {
      healthy,
      circuit_state: stats.circuit_breaker.state,
      dlq_size: stats.dlq_size,
      constraint_count: stats.sce_stats.constraint_count,
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default SCEAdapter;
