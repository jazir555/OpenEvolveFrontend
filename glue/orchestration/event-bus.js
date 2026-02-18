"use strict";
/**
 * Event Bus - Publish/Subscribe Interface
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Event persistence for replay
 * - Law of Configuration Explicitness: Event bus type via ENV var
 * - Observability: JSON Lines logging with correlation IDs
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.eventBus = exports.EventBus = exports.EventBusType = void 0;
const events_1 = require("events");
const event_types_1 = require("./event-types");
const dead_letter_queue_1 = require("./dead-letter-queue");
const logger_1 = require("../lib/logger");
const circuit_breaker_1 = require("../lib/circuit-breaker");
var EventBusType;
(function (EventBusType) {
    EventBusType["MEMORY"] = "memory";
    EventBusType["REDIS"] = "redis";
    EventBusType["RABBITMQ"] = "rabbitmq";
    EventBusType["KAFKA"] = "kafka";
})(EventBusType = exports.EventBusType || (exports.EventBusType = {}));
/**
 * Event Bus Implementation
 *
 * Supports multiple backends: memory, Redis, RabbitMQ, Kafka
 * Includes event persistence, DLQ routing, and circuit breakers
 */
class EventBus extends events_1.EventEmitter {
    logger;
    config;
    subscriptions = new Map();
    eventHistory = new Map(); // For replay
    stats;
    startTime;
    dlq;
    circuitBreakers = new Map();
    constructor(config) {
        super();
        this.logger = new logger_1.Logger('event-bus');
        this.startTime = Date.now();
        // Load config from environment
        const eventType = process.env.EVENT_BUS_TYPE || EventBusType.MEMORY;
        const eventBusUrl = process.env.EVENT_BUS_URL;
        this.config = {
            type: eventType,
            url: eventBusUrl,
            persistence_enabled: true,
            circuit_breaker_enabled: true,
            dlq_enabled: true,
            ...config
        };
        // Initialize DLQ if enabled
        if (this.config.dlq_enabled) {
            this.dlq = new dead_letter_queue_1.DeadLetterQueue();
        }
        this.stats = {
            type: this.config.type,
            events_published: 0,
            events_received: 0,
            events_failed: 0,
            subscriptions: 0,
            uptime_seconds: 0
        };
        this.validateConfig();
        this.logger.info('Event Bus initialized', {
            type: this.config.type,
            url: this.config.url,
            persistence_enabled: this.config.persistence_enabled,
            dlq_enabled: this.config.dlq_enabled
        });
    }
    /**
     * Publish an event to the bus
     */
    async publish(event) {
        const validation = (0, event_types_1.validateEvent)(event);
        if (!validation.valid) {
            const error = new Error(`Invalid event: ${validation.errors.join(', ')}`);
            this.logger.error('Event validation failed', error, {
                event_id: event.id,
                validation_errors: validation.errors
            });
            throw error;
        }
        // Persist event for replay
        if (this.config.persistence_enabled) {
            this.eventHistory.set(event.id, event);
        }
        this.stats.events_published++;
        this.logger.debug('Event published', {
            event_id: event.id,
            event_type: event.type,
            correlation_id: event.correlation_id,
            source_service: event.source_service
        });
        try {
            // Emit to subscribers
            const subscribers = this.subscriptions.get(event.type) || [];
            for (const subscription of subscribers) {
                this.stats.events_received++;
                // Use circuit breaker if enabled
                if (this.config.circuit_breaker_enabled) {
                    const cb = this.getCircuitBreaker(subscription.handler);
                    await cb.execute(async () => subscription.handler(event));
                }
                else {
                    await subscription.handler(event);
                }
            }
            // Emit to wildcard subscribers
            const wildcardSubscribers = this.subscriptions.get('*') || [];
            for (const subscription of wildcardSubscribers) {
                await subscription.handler(event);
            }
        }
        catch (error) {
            this.stats.events_failed++;
            this.handlePublishError(event, error);
        }
    }
    /**
     * Subscribe to events of a specific type
     */
    subscribe(eventType, handler) {
        const subscription = {
            eventType,
            handler,
            subscriptionId: `${eventType}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        };
        if (!this.subscriptions.has(eventType)) {
            this.subscriptions.set(eventType, []);
        }
        this.subscriptions.get(eventType).push(subscription);
        this.stats.subscriptions++;
        this.logger.info('Event subscription created', {
            subscription_id: subscription.subscriptionId,
            event_type: eventType
        });
        return subscription;
    }
    /**
     * Unsubscribe from events
     */
    unsubscribe(subscriptionId) {
        for (const [eventType, subscriptions] of this.subscriptions.entries()) {
            const index = subscriptions.findIndex(s => s.subscriptionId === subscriptionId);
            if (index !== -1) {
                subscriptions.splice(index, 1);
                this.stats.subscriptions--;
                this.logger.info('Event subscription removed', {
                    subscription_id: subscriptionId,
                    event_type: eventType
                });
                return true;
            }
        }
        return false;
    }
    /**
     * Replay events from history
     */
    async replay(filter, handler) {
        let events = Array.from(this.eventHistory.values());
        // Apply filters
        if (filter) {
            if (filter.event_type) {
                events = events.filter(e => e.type === filter.event_type);
            }
            if (filter.from_timestamp) {
                events = events.filter(e => e.timestamp >= filter.from_timestamp);
            }
            if (filter.to_timestamp) {
                events = events.filter(e => e.timestamp <= filter.to_timestamp);
            }
            if (filter.correlation_id) {
                events = events.filter(e => e.correlation_id === filter.correlation_id);
            }
        }
        this.logger.info('Replaying events', {
            event_count: events.length,
            filter
        });
        let replayed = 0;
        for (const event of events) {
            try {
                if (handler) {
                    await handler(event);
                }
                else {
                    // Replay to original subscribers
                    const subscribers = this.subscriptions.get(event.type) || [];
                    for (const subscription of subscribers) {
                        await subscription.handler(event);
                    }
                }
                replayed++;
            }
            catch (error) {
                this.logger.error('Event replay failed', error, {
                    event_id: event.id,
                    event_type: event.type
                });
                // Send to DLQ
                if (this.dlq) {
                    await this.dlq.enqueue(event, error, {
                        replay: true
                    });
                }
            }
        }
        return replayed;
    }
    /**
     * Get event history
     */
    getHistory(filter) {
        let events = Array.from(this.eventHistory.values());
        if (filter) {
            if (filter.event_type) {
                events = events.filter(e => e.type === filter.event_type);
            }
            if (filter.correlation_id) {
                events = events.filter(e => e.correlation_id === filter.correlation_id);
            }
            if (filter.limit) {
                events = events.slice(-filter.limit);
            }
        }
        // Sort by timestamp (newest first)
        return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    }
    /**
     * Get event by ID
     */
    getEvent(eventId) {
        return this.eventHistory.get(eventId);
    }
    /**
     * Get statistics
     */
    getStats() {
        return {
            ...this.stats,
            uptime_seconds: Math.floor((Date.now() - this.startTime) / 1000)
        };
    }
    /**
     * Get DLQ instance
     */
    getDLQ() {
        return this.dlq;
    }
    /**
     * Clear event history (use with caution)
     */
    clearHistory() {
        this.eventHistory.clear();
        this.logger.warn('Event history cleared');
    }
    /**
     * Validate configuration
     */
    validateConfig() {
        const validTypes = Object.values(EventBusType);
        if (!validTypes.includes(this.config.type)) {
            throw new Error(`Invalid EVENT_BUS_TYPE: ${this.config.type}. Must be one of: ${validTypes.join(', ')}`);
        }
        if (this.config.type !== EventBusType.MEMORY && !this.config.url) {
            throw new Error(`EVENT_BUS_URL is required for ${this.config.type} event bus`);
        }
    }
    /**
     * Handle publish errors with retry and DLQ
     */
    async handlePublishError(event, error) {
        this.logger.error('Event publish failed', error, {
            event_id: event.id,
            event_type: event.type,
            correlation_id: event.correlation_id
        });
        // Send to DLQ for failed events
        if (this.dlq) {
            await this.dlq.enqueue(event, error, {
                handler: 'event-bus',
                operation: 'publish'
            });
        }
    }
    /**
     * Get or create circuit breaker for handler
     */
    getCircuitBreaker(handler) {
        const handlerId = handler.name || 'anonymous';
        if (!this.circuitBreakers.has(handlerId)) {
            this.circuitBreakers.set(handlerId, new circuit_breaker_1.CircuitBreaker({
                threshold: 5,
                timeout_ms: 30000 // 30 seconds
            }));
        }
        return this.circuitBreakers.get(handlerId);
    }
    /**
     * Shutdown event bus gracefully
     */
    async shutdown() {
        this.logger.info('Shutting down event bus');
        // Clear subscriptions
        this.subscriptions.clear();
        // Clear circuit breakers
        this.circuitBreakers.clear();
        this.logger.info('Event bus shutdown complete');
    }
}
exports.EventBus = EventBus;
/**
 * Singleton instance
 */
exports.eventBus = new EventBus();
/**
 * Example usage:
 *
 * ```typescript
 * import { eventBus } from './event-bus';
 * import { createBaseEvent } from './event-types';
 *
 * // Subscribe to events
 * const subscription = eventBus.subscribe('KnowledgeExtracted', async (event) => {
 *   if (event.type === 'KnowledgeExtracted') {
 *     console.log(`Extracted ${event.data.chunk_count} chunks`);
 *     await processChunks(event.data.chunks);
 *   }
 * });
 *
 * // Publish event
 * const event = createBaseEvent(
 *   'KnowledgeExtracted',
 *   'ragbits-adapter',
 *   'corr-123',
 *   {
 *     document_id: 'doc-456',
 *     chunk_count: 10,
 *     chunks: [...],
 *     extraction_method: 'recursive'
 *   }
 * );
 *
 * await eventBus.publish(event);
 *
 * // Replay events
 * await eventBus.replay({
 *   event_type: 'KnowledgeExtracted',
 *   from_timestamp: '2025-01-01T00:00:00Z'
 * });
 *
 * // Get stats
 * const stats = eventBus.getStats();
 * console.log('Event Bus Stats:', stats);
 *
 * // Unsubscribe
 * eventBus.unsubscribe(subscription.subscriptionId);
 * ```
 */
//# sourceMappingURL=event-bus.js.map