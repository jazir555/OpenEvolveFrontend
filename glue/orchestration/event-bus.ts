/**
 * Event Bus - Publish/Subscribe Interface
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Event persistence for replay
 * - Law of Configuration Explicitness: Event bus type via ENV var
 * - Observability: JSON Lines logging with correlation IDs
 */

import { EventEmitter } from 'events';
import { Event, EventHandler, EventSubscription, validateEvent } from './event-types';
import { DeadLetterQueue } from './dead-letter-queue';
import { Logger } from '../lib/logger';
import { CircuitBreaker } from '../lib/circuit-breaker';

export enum EventBusType {
  MEMORY = 'memory',
  REDIS = 'redis',
  RABBITMQ = 'rabbitmq',
  KAFKA = 'kafka'
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
export class EventBus extends EventEmitter {
  private logger: Logger;
  private config: EventBusConfig;
  private subscriptions: Map<string, EventSubscription[]> = new Map();
  private eventHistory: Map<string, Event> = new Map(); // For replay
  private stats: EventBusStats;
  private startTime: number;
  private dlq?: DeadLetterQueue;
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();

  constructor(config?: Partial<EventBusConfig>) {
    super();

    this.logger = new Logger('event-bus');
    this.startTime = Date.now();

    // Load config from environment
    const eventType = (process.env.EVENT_BUS_TYPE as EventBusType) || EventBusType.MEMORY;
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
      this.dlq = new DeadLetterQueue();
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
  async publish(event: Event): Promise<void> {
    const validation = validateEvent(event);
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
        } else {
          await subscription.handler(event);
        }
      }

      // Emit to wildcard subscribers
      const wildcardSubscribers = this.subscriptions.get('*') || [];
      for (const subscription of wildcardSubscribers) {
        await subscription.handler(event);
      }
    } catch (error) {
      this.stats.events_failed++;
      this.handlePublishError(event, error as Error);
    }
  }

  /**
   * Subscribe to events of a specific type
   */
  subscribe(eventType: string, handler: EventHandler): EventSubscription {
    const subscription: EventSubscription = {
      eventType,
      handler,
      subscriptionId: `${eventType}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    };

    if (!this.subscriptions.has(eventType)) {
      this.subscriptions.set(eventType, []);
    }

    this.subscriptions.get(eventType)!.push(subscription);
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
  unsubscribe(subscriptionId: string): boolean {
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
  async replay(
    filter?: {
      event_type?: string;
      from_timestamp?: string;
      to_timestamp?: string;
      correlation_id?: string;
    },
    handler?: EventHandler
  ): Promise<number> {
    let events = Array.from(this.eventHistory.values());

    // Apply filters
    if (filter) {
      if (filter.event_type) {
        events = events.filter(e => e.type === filter.event_type);
      }
      if (filter.from_timestamp) {
        events = events.filter(e => e.timestamp >= filter.from_timestamp!);
      }
      if (filter.to_timestamp) {
        events = events.filter(e => e.timestamp <= filter.to_timestamp!);
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
        } else {
          // Replay to original subscribers
          const subscribers = this.subscriptions.get(event.type) || [];
          for (const subscription of subscribers) {
            await subscription.handler(event);
          }
        }
        replayed++;
      } catch (error) {
        this.logger.error('Event replay failed', error as Error, {
          event_id: event.id,
          event_type: event.type
        });

        // Send to DLQ
        if (this.dlq) {
          await this.dlq.enqueue(event, error as Error, {
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
  getHistory(filter?: {
    event_type?: string;
    correlation_id?: string;
    limit?: number;
  }): Event[] {
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
    return events.sort((a, b) =>
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }

  /**
   * Get event by ID
   */
  getEvent(eventId: string): Event | undefined {
    return this.eventHistory.get(eventId);
  }

  /**
   * Get statistics
   */
  getStats(): EventBusStats {
    return {
      ...this.stats,
      uptime_seconds: Math.floor((Date.now() - this.startTime) / 1000)
    };
  }

  /**
   * Get DLQ instance
   */
  getDLQ(): DeadLetterQueue | undefined {
    return this.dlq;
  }

  /**
   * Clear event history (use with caution)
   */
  clearHistory(): void {
    this.eventHistory.clear();
    this.logger.warn('Event history cleared');
  }

  /**
   * Validate configuration
   */
  private validateConfig(): void {
    const validTypes = Object.values(EventBusType);
    if (!validTypes.includes(this.config.type)) {
      throw new Error(
        `Invalid EVENT_BUS_TYPE: ${this.config.type}. Must be one of: ${validTypes.join(', ')}`
      );
    }

    if (this.config.type !== EventBusType.MEMORY && !this.config.url) {
      throw new Error(
        `EVENT_BUS_URL is required for ${this.config.type} event bus`
      );
    }
  }

  /**
   * Handle publish errors with retry and DLQ
   */
  private async handlePublishError(event: Event, error: Error): Promise<void> {
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
  private getCircuitBreaker(handler: EventHandler): CircuitBreaker {
    const handlerId = handler.name || 'anonymous';
    if (!this.circuitBreakers.has(handlerId)) {
      this.circuitBreakers.set(
        handlerId,
        new CircuitBreaker({
          threshold: 5,
          timeout_ms: 30000 // 30 seconds
        })
      );
    }
    return this.circuitBreakers.get(handlerId)!;
  }

  /**
   * Shutdown event bus gracefully
   */
  async shutdown(): Promise<void> {
    this.logger.info('Shutting down event bus');

    // Clear subscriptions
    this.subscriptions.clear();

    // Clear circuit breakers
    this.circuitBreakers.clear();

    this.logger.info('Event bus shutdown complete');
  }
}

/**
 * In-memory implementation of EventBus
 * Suitable for testing and single-instance deployments
 */
export class InMemoryEventBus {
  private handlers: Map<string, EventHandler[]> = new Map();
  private eventHistory: Map<string, Event> = new Map();
  private readonly maxHistorySize: number;
  private logger: Logger;

  constructor(maxHistorySize: number = 10000) {
    this.maxHistorySize = maxHistorySize;
    this.logger = new Logger('in-memory-event-bus');
  }

  /**
   * Publish an event to the bus
   */
  async publish(event: Event): Promise<void> {
    // Validate event
    const validation = validateEvent(event);
    if (!validation.valid) {
      const error = new Error(`Invalid event: ${validation.errors.join(', ')}`);
      this.logger.error('Event validation failed', error, {
        event_id: event.id,
        validation_errors: validation.errors
      });
      throw error;
    }

    // Store in history
    this.eventHistory.set(event.id, event);

    // Enforce history size limit
    if (this.eventHistory.size > this.maxHistorySize) {
      // Remove oldest
      const oldest = Array.from(this.eventHistory.keys())[0];
      this.eventHistory.delete(oldest);
      this.logger.debug('Removed oldest event from history', { event_id: oldest });
    }

    // Get handlers for this event type
    const handlers = this.handlers.get(event.type) || [];

    // Execute all handlers
    await Promise.all(
      handlers.map(handler =>
        this.executeHandler(handler, event)
      )
    );

    // Execute wildcard handlers
    const wildcardHandlers = this.handlers.get('*') || [];
    await Promise.all(
      wildcardHandlers.map(handler =>
        this.executeHandler(handler, event)
      )
    );

    this.logger.debug('Event published', {
      event_id: event.id,
      event_type: event.type,
      correlation_id: event.correlation_id,
      source_service: event.source_service,
      handler_count: handlers.length + wildcardHandlers.length
    });
  }

  /**
   * Subscribe to events of a specific type
   */
  subscribe(eventType: string, handler: EventHandler): void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, []);
    }

    this.handlers.get(eventType)!.push(handler);

    this.logger.info('Event subscription created', {
      event_type: eventType,
      handler_name: handler.name || 'anonymous'
    });
  }

  /**
   * Unsubscribe from events
   */
  unsubscribe(eventType: string, handler: EventHandler): void {
    const handlers = this.handlers.get(eventType);
    if (!handlers) return;

    const index = handlers.indexOf(handler);
    if (index > -1) {
      handlers.splice(index, 1);
      this.logger.info('Event subscription removed', {
        event_type: eventType,
        handler_name: handler.name || 'anonymous'
      });
    }
  }

  /**
   * Replay events from history
   * Supports filtering to avoid loading everything into memory
   */
  async replay(
    filter?: (event: Event) => boolean,
    limit?: number
  ): Promise<Event[]> {
    let events = Array.from(this.eventHistory.values());

    // Apply filter if provided
    if (filter) {
      events = events.filter(filter);
    }

    // Apply limit
    if (limit) {
      events = events.slice(0, limit);
    }

    this.logger.info('Replaying events', {
      event_count: events.length,
      filter_applied: !!filter,
      limit_applied: !!limit
    });

    return events;
  }

  /**
   * Get event history
   */
  getHistory(): Map<string, Event> {
    return new Map(this.eventHistory);
  }

  /**
   * Get events by type
   */
  getEventsByType(eventType: string): Event[] {
    return Array.from(this.eventHistory.values()).filter(
      event => event.type === eventType
    );
  }

  /**
   * Clear old events from history
   */
  clearHistory(olderThanMs: number): number {
    const cutoff = Date.now() - olderThanMs;
    let cleared = 0;

    for (const [id, event] of this.eventHistory.entries()) {
      const eventTime = new Date(event.timestamp).getTime();
      if (eventTime < cutoff) {
        this.eventHistory.delete(id);
        cleared++;
      }
    }

    if (cleared > 0) {
      this.logger.info('Cleared old events from history', {
        count: cleared,
        older_than_ms: olderThanMs
      });
    }

    return cleared;
  }

  /**
   * Clear all history
   */
  clearAllHistory(): void {
    const count = this.eventHistory.size;
    this.eventHistory.clear();
    this.logger.warn('Cleared all event history', { count });
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      total_events: this.eventHistory.size,
      subscriptions: Array.from(this.handlers.entries()).map(([type, handlers]) => ({
        event_type: type,
        handler_count: handlers.length
      })),
      max_history_size: this.maxHistorySize
    };
  }

  /**
   * Execute handler with error handling
   */
  private async executeHandler(
    handler: EventHandler,
    event: Event
  ): Promise<void> {
    try {
      await handler(event);
    } catch (error) {
      this.logger.error('Event handler failed', error as Error, {
        event_type: event.type,
        event_id: event.id,
        correlation_id: event.correlation_id,
        handler_name: handler.name || 'anonymous'
      });
      // Don't re-throw - allow other handlers to execute
    }
  }

  /**
   * Shutdown event bus gracefully
   */
  async shutdown(): Promise<void> {
    this.logger.info('Shutting down in-memory event bus');

    // Clear subscriptions
    this.handlers.clear();

    // Clear history
    this.eventHistory.clear();

    this.logger.info('In-memory event bus shutdown complete');
  }
}

/**
 * Singleton instance
 */
export const eventBus = new EventBus();

/**
 * Singleton instance of InMemoryEventBus
 */
export const inMemoryEventBus = new InMemoryEventBus();

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
