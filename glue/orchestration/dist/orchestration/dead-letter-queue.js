"use strict";
/**
 * Dead Letter Queue (DLQ)
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Support for replay and deduplication
 * - Failure Management: Logic failures go to DLQ, don't block pipeline
 * - Observability: JSON Lines logging with full context
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.deadLetterQueue = exports.DeadLetterQueue = exports.DEFAULT_RETRY_POLICY = void 0;
const event_types_1 = require("./event-types");
const logger_1 = require("../lib/logger");
exports.DEFAULT_RETRY_POLICY = {
    max_retries: 3,
    initial_delay_ms: 1000,
    max_delay_ms: 60000,
    backoff_multiplier: 2,
    retry_on: ['NetworkError', 'TimeoutError', 'TransientError']
};
/**
 * Dead Letter Queue for handling failed events
 *
 * Logic failures (bad data, validation errors) go to DLQ
 * Transient failures (network blips) should use retry logic before DLQ
 */
class DeadLetterQueue {
    logger;
    queue = new Map();
    retryPolicy;
    constructor(retryPolicy) {
        this.logger = new logger_1.Logger('dead-letter-queue');
        this.retryPolicy = { ...exports.DEFAULT_RETRY_POLICY, ...retryPolicy };
    }
    /**
     * Add an event to the DLQ
     */
    async enqueue(event, error, metadata = {}) {
        const validation = (0, event_types_1.validateEvent)(event);
        if (!validation.valid) {
            this.logger.error('Invalid event enqueued to DLQ', error, {
                validation_errors: validation.errors
            });
            throw new Error(`Invalid event: ${validation.errors.join(', ')}`);
        }
        const entry = {
            id: `${event.id}-dlq`,
            event,
            error_message: error.message,
            error_stack: error.stack,
            retry_count: 0,
            max_retries: this.retryPolicy.max_retries,
            created_at: new Date().toISOString(),
            processed: false,
            metadata: {
                ...metadata,
                correlation_id: event.correlation_id,
                event_type: event.type
            }
        };
        // Calculate next retry time
        if (this.shouldRetry(error)) {
            entry.next_retry_at = this.calculateNextRetry(0);
        }
        this.queue.set(entry.id, entry);
        this.logger.warn('Event enqueued to DLQ', {
            dlq_entry_id: entry.id,
            event_id: event.id,
            event_type: event.type,
            error_message: error.message,
            retry_count: entry.retry_count,
            max_retries: entry.max_retries,
            will_retry: entry.retry_count < entry.max_retries
        });
        return entry.id;
    }
    /**
     * Process entries ready for retry
     */
    async processRetry(handler) {
        const now = new Date().toISOString();
        let processed = 0;
        for (const [id, entry] of this.queue.entries()) {
            if (entry.processed) {
                continue;
            }
            // Check if ready for retry
            if (entry.next_retry_at && entry.next_retry_at > now) {
                continue;
            }
            // Check if max retries exceeded
            if (entry.retry_count >= entry.max_retries) {
                this.logger.error('DLQ entry exceeded max retries', undefined, {
                    dlq_entry_id: id,
                    event_id: entry.event.id,
                    event_type: entry.event.type,
                    retry_count: entry.retry_count,
                    max_retries: entry.max_retries
                });
                continue;
            }
            // Attempt retry
            entry.last_attempt_at = now;
            entry.retry_count++;
            try {
                await handler(entry.event);
                // Success - mark as processed
                entry.processed = true;
                this.logger.info('DLQ entry processed successfully', {
                    dlq_entry_id: id,
                    event_id: entry.event.id,
                    event_type: entry.event.type,
                    retry_count: entry.retry_count
                });
                processed++;
            }
            catch (error) {
                const err = error;
                // Calculate next retry
                if (entry.retry_count < entry.max_retries && this.shouldRetry(err)) {
                    entry.next_retry_at = this.calculateNextRetry(entry.retry_count);
                }
                else {
                    // Mark for manual intervention
                    entry.next_retry_at = undefined;
                }
                this.logger.error('DLQ retry failed', err, {
                    dlq_entry_id: id,
                    event_id: entry.event.id,
                    event_type: entry.event.type,
                    retry_count: entry.retry_count,
                    max_retries: entry.max_retries,
                    will_retry_again: entry.retry_count < entry.max_retries
                });
            }
        }
        // Cleanup processed entries
        this.cleanup();
        return processed;
    }
    /**
     * Get all DLQ entries
     */
    getEntries(filter) {
        let entries = Array.from(this.queue.values());
        if (filter) {
            if (filter.processed !== undefined) {
                entries = entries.filter(e => e.processed === filter.processed);
            }
            if (filter.event_type) {
                entries = entries.filter(e => e.event.type === filter.event_type);
            }
        }
        // Sort by creation time (newest first)
        return entries.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    }
    /**
     * Get specific entry by ID
     */
    getEntry(id) {
        return this.queue.get(id);
    }
    /**
     * Get DLQ statistics
     */
    getStats() {
        const entries = Array.from(this.queue.values());
        const processed = entries.filter(e => e.processed);
        const pending = entries.filter(e => !e.processed && e.retry_count < e.max_retries);
        const failed = entries.filter(e => !e.processed && e.retry_count >= e.max_retries);
        const byEventType = {};
        for (const entry of entries) {
            const { type } = entry.event;
            byEventType[type] = (byEventType[type] || 0) + 1;
        }
        return {
            total_entries: entries.length,
            pending_entries: pending.length,
            processed_entries: processed.length,
            failed_permanently: failed.length,
            by_event_type: byEventType
        };
    }
    /**
     * Manually retry a specific entry
     */
    async retryEntry(id, handler) {
        const entry = this.queue.get(id);
        if (!entry) {
            throw new Error(`DLQ entry ${id} not found`);
        }
        entry.last_attempt_at = new Date().toISOString();
        entry.retry_count++;
        try {
            await handler(entry.event);
            entry.processed = true;
            this.logger.info('Manual DLQ retry successful', {
                dlq_entry_id: id,
                event_id: entry.event.id,
                retry_count: entry.retry_count
            });
            return true;
        }
        catch (error) {
            this.logger.error('Manual DLQ retry failed', error, {
                dlq_entry_id: id,
                event_id: entry.event.id,
                retry_count: entry.retry_count
            });
            return false;
        }
    }
    /**
     * Delete an entry from DLQ
     */
    deleteEntry(id) {
        return this.queue.delete(id);
    }
    /**
     * Check if error should be retried
     */
    shouldRetry(error) {
        return this.retryPolicy.retry_on.some(errorType => error.name.includes(errorType) || error.message.includes(errorType));
    }
    /**
     * Calculate next retry time with exponential backoff
     */
    calculateNextRetry(retryCount) {
        const delay = Math.min(this.retryPolicy.initial_delay_ms * Math.pow(this.retryPolicy.backoff_multiplier, retryCount), this.retryPolicy.max_delay_ms);
        const nextRetry = new Date(Date.now() + delay);
        return nextRetry.toISOString();
    }
    /**
     * Cleanup processed entries older than specified time
     */
    cleanup(maxAgeMs = 3600000) {
        const now = Date.now();
        const deleted = [];
        for (const [id, entry] of this.queue.entries()) {
            if (entry.processed) {
                const createdAt = new Date(entry.created_at).getTime();
                if (now - createdAt > maxAgeMs) {
                    this.queue.delete(id);
                    deleted.push(id);
                }
            }
        }
        if (deleted.length > 0) {
            this.logger.info('Cleaned up processed DLQ entries', {
                count: deleted.length
            });
        }
    }
    /**
     * Clear all entries (use with caution)
     */
    clear() {
        this.queue.clear();
        this.logger.warn('DLQ cleared', { count: this.queue.size });
    }
    /**
     * Export DLQ entries for analysis
     */
    export() {
        return Array.from(this.queue.values());
    }
    /**
     * Import DLQ entries (for recovery)
     */
    import(entries) {
        for (const entry of entries) {
            this.queue.set(entry.id, entry);
        }
        this.logger.info('DLQ entries imported', { count: entries.length });
    }
}
exports.DeadLetterQueue = DeadLetterQueue;
/**
 * Singleton instance
 */
exports.deadLetterQueue = new DeadLetterQueue();
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
//# sourceMappingURL=dead-letter-queue.js.map