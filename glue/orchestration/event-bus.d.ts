/**
 * Event Bus - Publish/Subscribe Interface
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Event persistence for replay
 * - Law of Configuration Explicitness: Event bus type via ENV var
 * - Observability: JSON Lines logging with correlation IDs
 */
import { EventEmitter } from 'events';
import { Event, EventHandler, EventSubscription } from './event-types';
import { DeadLetterQueue } from './dead-letter-queue';
export declare enum EventBusType {
    MEMORY = "memory",
    REDIS = "redis",
    RABBITMQ = "rabbitmq",
    KAFKA = "kafka"
}
export interface EventBusConfig {
    type: EventBusType;
    url?: string;
    persistence_enabled?: boolean;
    circuit_breaker_enabled?: boolean;
    dlq_enabled?: boolean;
}
export interface EventBusStats {
    type: EventBusType;
    events_published: number;
    events_received: number;
    events_failed: number;
    subscriptions: number;
    uptime_seconds: number;
}
/**
 * Event Bus Implementation
 *
 * Supports multiple backends: memory, Redis, RabbitMQ, Kafka
 * Includes event persistence, DLQ routing, and circuit breakers
 */
export declare class EventBus extends EventEmitter {
    private logger;
    private config;
    private subscriptions;
    private eventHistory;
    private stats;
    private startTime;
    private dlq?;
    private circuitBreakers;
    constructor(config?: Partial<EventBusConfig>);
    /**
     * Publish an event to the bus
     */
    publish(event: Event): Promise<void>;
    /**
     * Subscribe to events of a specific type
     */
    subscribe(eventType: string, handler: EventHandler): EventSubscription;
    /**
     * Unsubscribe from events
     */
    unsubscribe(subscriptionId: string): boolean;
    /**
     * Replay events from history
     */
    replay(filter?: {
        event_type?: string;
        from_timestamp?: string;
        to_timestamp?: string;
        correlation_id?: string;
    }, handler?: EventHandler): Promise<number>;
    /**
     * Get event history
     */
    getHistory(filter?: {
        event_type?: string;
        correlation_id?: string;
        limit?: number;
    }): Event[];
    /**
     * Get event by ID
     */
    getEvent(eventId: string): Event | undefined;
    /**
     * Get statistics
     */
    getStats(): EventBusStats;
    /**
     * Get DLQ instance
     */
    getDLQ(): DeadLetterQueue | undefined;
    /**
     * Clear event history (use with caution)
     */
    clearHistory(): void;
    /**
     * Validate configuration
     */
    private validateConfig;
    /**
     * Handle publish errors with retry and DLQ
     */
    private handlePublishError;
    /**
     * Get or create circuit breaker for handler
     */
    private getCircuitBreaker;
    /**
     * Shutdown event bus gracefully
     */
    shutdown(): Promise<void>;
}
/**
 * In-memory implementation of EventBus
 * Suitable for testing and single-instance deployments
 */
export declare class InMemoryEventBus {
    private handlers;
    private eventHistory;
    private readonly maxHistorySize;
    private logger;
    constructor(maxHistorySize?: number);
    /**
     * Publish an event to the bus
     */
    publish(event: Event): Promise<void>;
    /**
     * Subscribe to events of a specific type
     */
    subscribe(eventType: string, handler: EventHandler): void;
    /**
     * Unsubscribe from events
     */
    unsubscribe(eventType: string, handler: EventHandler): void;
    /**
     * Replay events from history
     * Supports filtering to avoid loading everything into memory
     */
    replay(filter?: (event: Event) => boolean, limit?: number): Promise<Event[]>;
    /**
     * Get event history
     */
    getHistory(): Map<string, Event>;
    /**
     * Get events by type
     */
    getEventsByType(eventType: string): Event[];
    /**
     * Clear old events from history
     */
    clearHistory(olderThanMs: number): number;
    /**
     * Clear all history
     */
    clearAllHistory(): void;
    /**
     * Get statistics
     */
    getStats(): {
        total_events: number;
        subscriptions: {
            event_type: string;
            handler_count: number;
        }[];
        max_history_size: number;
    };
    /**
     * Execute handler with error handling
     */
    private executeHandler;
    /**
     * Shutdown event bus gracefully
     */
    shutdown(): Promise<void>;
}
/**
 * Singleton instance
 */
export declare const eventBus: EventBus;
/**
 * Singleton instance of InMemoryEventBus
 */
export declare const inMemoryEventBus: InMemoryEventBus;
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
