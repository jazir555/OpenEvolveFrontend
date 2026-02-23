/**
 * Dead Letter Queue (DLQ)
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Support for replay and deduplication
 * - Failure Management: Logic failures go to DLQ, don't block pipeline
 * - Observability: JSON Lines logging with full context
 */
import { Event } from './event-types';
export interface DLQEntry {
    id: string;
    event: Event;
    error_message: string;
    error_stack?: string;
    retry_count: number;
    max_retries: number;
    next_retry_at?: string;
    created_at: string;
    last_attempt_at?: string;
    processed: boolean;
    metadata: Record<string, any>;
}
export interface RetryPolicy {
    max_retries: number;
    initial_delay_ms: number;
    max_delay_ms: number;
    backoff_multiplier: number;
    retry_on: Array<string>;
}
export interface DLQStats {
    total_entries: number;
    pending_entries: number;
    processed_entries: number;
    failed_permanently: number;
    by_event_type: Record<string, number>;
}
export declare const DEFAULT_RETRY_POLICY: RetryPolicy;
/**
 * Dead Letter Queue for handling failed events
 *
 * Logic failures (bad data, validation errors) go to DLQ
 * Transient failures (network blips) should use retry logic before DLQ
 */
export declare class DeadLetterQueue {
    private logger;
    private queue;
    private retryPolicy;
    constructor(retryPolicy?: Partial<RetryPolicy>);
    /**
     * Add an event to the DLQ
     */
    enqueue(event: Event, error: Error, metadata?: Record<string, any>): Promise<string>;
    /**
     * Process entries ready for retry
     */
    processRetry(handler: (event: Event) => Promise<void>): Promise<number>;
    /**
     * Get all DLQ entries
     */
    getEntries(filter?: {
        processed?: boolean;
        event_type?: string;
    }): DLQEntry[];
    /**
     * Get specific entry by ID
     */
    getEntry(id: string): DLQEntry | undefined;
    /**
     * Get DLQ statistics
     */
    getStats(): DLQStats;
    /**
     * Manually retry a specific entry
     */
    retryEntry(id: string, handler: (event: Event) => Promise<void>): Promise<boolean>;
    /**
     * Delete an entry from DLQ
     */
    deleteEntry(id: string): boolean;
    /**
     * Check if error should be retried
     */
    private shouldRetry;
    /**
     * Calculate next retry time with exponential backoff
     */
    private calculateNextRetry;
    /**
     * Cleanup processed entries older than specified time
     */
    private cleanup;
    /**
     * Clear all entries (use with caution)
     */
    clear(): void;
    /**
     * Export DLQ entries for analysis
     */
    export(): DLQEntry[];
    /**
     * Import DLQ entries (for recovery)
     */
    import(entries: DLQEntry[]): void;
}
/**
 * Singleton instance
 */
export declare const deadLetterQueue: DeadLetterQueue;
/**
 * Example usage:
 *
 * ```typescript
 * import { deadLetterQueue } from './dead-letter-queue';
 * import { createBaseEvent } from './event-types';
 *
 * // Enqueue failed event
 * try {
 *   await processEvent(event);
 * } catch (error) {
 *   await deadLetterQueue.enqueue(event, error, {
 *     handler: 'vector-db-adapter',
 *     operation: 'index-embeddings'
 *   });
 * }
 *
 * // Process retries (run periodically)
 * setInterval(async () => {
 *   const processed = await deadLetterQueue.processRetry(async (event) => {
 *     await processEvent(event);
 *   });
 *   console.log(`Retried ${processed} events`);
 * }, 30000); // Every 30 seconds
 *
 * // Get stats
 * const stats = deadLetterQueue.getStats();
 * console.log('DLQ Stats:', stats);
 *
 * // Manual retry
 * await deadLetterQueue.retryEntry('entry-id', async (event) => {
 *   await processEvent(event);
 * });
 * ```
 */
