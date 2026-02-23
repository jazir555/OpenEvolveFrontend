/**
 * BubbleLab Adapter
 *
 * Purpose: Main adapter integrating BubbleLab with OpenEvolve orchestration layer
 * Compliance: Federation Constitution - Full compliance with all laws
 *
 * Features:
 * - Circuit breaker for system failures
 * - Exponential backoff retry for transient failures
 * - JSON Lines structured logging
 * - UTC timestamps (Law of UTC)
 * - Idempotent operations
 * - Canonical schema mapping
 * - Environment variable validation
 */
import { CircuitState } from '../../../lib/circuit-breaker';
import { type EventBus } from '../../../orchestration/event-bus';
import { CanonicalBubbleFlow, CanonicalExecutionResult, CredentialType } from './bubblelab-canonical';
export interface BubbleLabAdapterConfig {
    api_url: string;
    timeout_ms: number;
    auth_token?: string;
    circuit_breaker_threshold?: number;
    circuit_breaker_timeout_ms?: number;
    retry_max_retries?: number;
    retry_base_delay_ms?: number;
    retry_max_delay_ms?: number;
    eventBus?: EventBus;
}
export interface AdapterMetrics {
    total_requests: number;
    successful_requests: number;
    failed_requests: number;
    circuit_breaker_state: CircuitState;
    circuit_breaker_failure_count: number;
    average_request_duration_ms: number;
}
export declare class BubbleLabAdapter {
    private readonly client;
    private readonly circuitBreaker;
    private readonly logger;
    private readonly metrics;
    private readonly eventBus;
    constructor(config: BubbleLabAdapterConfig);
    /**
     * Check adapter and BubbleLab API health
     * Idempotent: Safe to call multiple times
     */
    healthCheck(correlation_id?: string): Promise<boolean>;
    /**
     * List all BubbleFlows in canonical format
     * Idempotent: GET operation, safe to retry
     */
    listBubbleFlows(correlation_id?: string): Promise<CanonicalBubbleFlow[]>;
    /**
     * Get a specific BubbleFlow in canonical format
     * Idempotent: GET operation, safe to retry
     */
    getBubbleFlow(flowId: string, correlation_id?: string): Promise<CanonicalBubbleFlow>;
    /**
     * Create a new BubbleFlow from canonical format
     * NOT idempotent: Use upsert logic for idempotency
     */
    createBubbleFlow(canonicalFlow: CanonicalBubbleFlow, correlation_id?: string): Promise<CanonicalBubbleFlow>;
    /**
     * Upsert a BubbleFlow (create or update if exists)
     * Idempotent: Safe to run multiple times
     */
    upsertBubbleFlow(canonicalFlow: CanonicalBubbleFlow, correlation_id?: string): Promise<CanonicalBubbleFlow>;
    /**
     * Update a BubbleFlow
     * NOT idempotent: Multiple updates with same data OK
     */
    updateBubbleFlow(flowId: string, updates: Partial<CanonicalBubbleFlow>, correlation_id?: string): Promise<CanonicalBubbleFlow>;
    /**
     * Delete a BubbleFlow
     * Idempotent with check: Verifies deletion
     */
    deleteBubbleFlow(flowId: string, correlation_id?: string): Promise<void>;
    /**
     * Execute a BubbleFlow
     * NOT idempotent: Each execution creates a new run
     */
    executeBubbleFlow(flowId: string, payload?: any, credentials?: Record<CredentialType, number>, correlation_id?: string): Promise<CanonicalExecutionResult>;
    /**
     * Get execution history for a BubbleFlow
     * Idempotent: GET operation, safe to retry
     */
    getExecutionHistory(flowId: string, limit?: number, offset?: number, correlation_id?: string): Promise<CanonicalExecutionResult[]>;
    /**
     * Get current adapter metrics
     */
    getMetrics(): AdapterMetrics;
    /**
     * Reset circuit breaker (e.g., after manual health check)
     */
    resetCircuitBreaker(): void;
    /**
     * Emit canonical event to orchestration layer
     */
    private emitEvent;
    /**
     * Update metrics with latest request duration
     */
    private updateMetrics;
}
/**
 * Create a BubbleLab adapter from environment variables
 * Validates all required environment variables (Law of Configuration Explicitness)
 */
export declare function createBubbleLabAdapter(): BubbleLabAdapter;
//# sourceMappingURL=adapter.d.ts.map