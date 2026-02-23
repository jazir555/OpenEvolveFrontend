/**
 * Event-Enabled Adapter Base Class
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Event deduplication via IDs
 * - Failure Management: Transient → Retry, Logic → DLQ, System → Circuit Breaker
 * - Observability: JSON Lines logging with correlation IDs
 * - Law of UTC: All timestamps in UTC ISO-8601
 *
 * Provides base functionality for all adapters to integrate with the event bus
 */
import { EventBus } from '../orchestration/event-bus';
import { DeadLetterQueue } from '../orchestration/dead-letter-queue';
import { Event } from '../orchestration/event-types';
import { Logger } from './logger';
import { CircuitBreaker } from './circuit-breaker';
import { RetryConfig } from './retry';
export interface AdapterEventConfig {
    eventBus: EventBus;
    publishEvents?: boolean;
    subscribeToEvents?: boolean;
    dlqEnabled?: boolean;
    circuitBreakerEnabled?: boolean;
    retryConfig?: RetryConfig;
}
export interface AdapterOperationResult<T = any> {
    success: boolean;
    data?: T;
    error?: Error;
    event_published?: boolean;
    duration_ms: number;
    correlation_id: string;
}
/**
 * Base class for event-enabled adapters
 *
 * All adapters extend this class to get automatic:
 * - Event publishing
 * - Event subscription
 * - Circuit breaker protection
 * - Retry with exponential backoff
 * - DLQ integration
 * - Structured logging
 */
export declare class EventEnabledAdapter {
    protected logger: Logger;
    protected eventBus: EventBus;
    protected dlq?: DeadLetterQueue;
    protected circuitBreaker?: CircuitBreaker;
    protected retryConfig: RetryConfig;
    protected publishEvents: boolean;
    protected subscribeToEvents: boolean;
    protected adapterName: string;
    constructor(adapterName: string, config: AdapterEventConfig);
    /**
     * Execute an operation with automatic event publishing, retry, circuit breaker, and DLQ
     *
     * This is the main method that adapters should use to execute operations
     */
    protected executeOperation<T>(operationName: string, operation: () => Promise<T>, eventType?: string | null, eventData?: any): Promise<AdapterOperationResult<T>>;
    /**
     * Publish an event to the event bus
     */
    protected publishEvent(eventType: string, data: any, correlationId?: string): Promise<void>;
    /**
     * Subscribe to an event type
     */
    protected subscribeToEvent(eventType: string, handler: (event: Event) => Promise<void>): void;
    /**
     * Determine if an error is a logic failure (should go to DLQ) or transient (should retry)
     *
     * Transient errors: network timeouts, connection refused, rate limiting
     * Logic errors: validation errors, bad data, business logic violations
     */
    protected isLogicFailure(error: Error): boolean;
    /**
     * Get circuit breaker state
     */
    getCircuitBreakerState(): import("./circuit-breaker").CircuitBreakerStats | null;
    /**
     * Reset circuit breaker
     */
    resetCircuitBreaker(): void;
    /**
     * Get DLQ statistics
     */
    getDLQStats(): import("../orchestration/dead-letter-queue").DLQStats | null;
    /**
     * Cleanup resources
     */
    shutdown(): Promise<void>;
}
/**
 * Example usage:
 *
 * ```typescript
 * import { EventEnabledAdapter } from './event-enabled-adapter';
 * import { eventBus } from '../orchestration/event-bus';
 *
 * class MyAdapter extends EventEnabledAdapter {
 *   constructor() {
 *     super('my-adapter', {
 *       eventBus,
 *       publishEvents: true,
 *       subscribeToEvents: true,
 *       dlqEnabled: true,
 *       circuitBreakerEnabled: true,
 *       retryConfig: { max_retries: 3, base_delay_ms: 1000 }
 *     });
 *
 *     // Subscribe to events
 *     this.subscribeToEvent('KnowledgeExtracted', async (event) => {
 *       if (event.type === 'KnowledgeExtracted') {
 *         await this.processKnowledge(event.data);
 *       }
 *     });
 *   }
 *
 *   async processKnowledge(data: any) {
 *     return this.executeOperation(
 *       'process-knowledge',
 *       async () => {
 *         // Your operation here
 *         return { processed: true };
 *       },
 *       'KnowledgeProcessed',
 *       data
 *     );
 *   }
 * }
 * ```
 */
