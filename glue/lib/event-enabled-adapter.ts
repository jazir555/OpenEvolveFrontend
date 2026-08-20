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

import { randomUUID } from 'crypto';
import { EventBus } from '../orchestration/event-bus';
import { DeadLetterQueue } from '../orchestration/dead-letter-queue';
import { Event, createBaseEvent } from '../orchestration/event-types';
import { Logger } from './logger';
import { CircuitBreaker } from './circuit-breaker';
import { retryWithBackoff, RetryConfig } from './retry';

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
export class EventEnabledAdapter {
  protected logger: Logger;
  protected eventBus: EventBus;
  protected dlq?: DeadLetterQueue;
  protected circuitBreaker?: CircuitBreaker;
  protected retryConfig: RetryConfig;
  protected publishEvents: boolean;
  protected subscribeToEvents: boolean;
  protected adapterName: string;

  constructor(adapterName: string, config: AdapterEventConfig) {
    this.adapterName = adapterName;
    this.logger = new Logger(adapterName);
    this.eventBus = config.eventBus;
    this.publishEvents = config.publishEvents ?? true;
    this.subscribeToEvents = config.subscribeToEvents ?? true;
    this.retryConfig = config.retryConfig || { max_retries: 3, base_delay_ms: 1000 };

    // Initialize DLQ if enabled
    if (config.dlqEnabled && this.eventBus.getDLQ()) {
      this.dlq = this.eventBus.getDLQ();
    }

    // Initialize circuit breaker if enabled
    if (config.circuitBreakerEnabled) {
      this.circuitBreaker = new CircuitBreaker({
        threshold: 5,
        timeout_ms: 30000,
        onStateChange: (oldState, newState) => {
          this.logger.warn('Circuit breaker state changed', {
            old_state: oldState,
            new_state: newState,
            adapter: this.adapterName,
          });
        },
      });
    }

    this.logger.info('Event-enabled adapter initialized', {
      adapter: this.adapterName,
      publish_events: this.publishEvents,
      subscribe_events: this.subscribeToEvents,
      dlq_enabled: !!this.dlq,
      circuit_breaker_enabled: !!this.circuitBreaker,
    });
  }

  /**
   * Execute an operation with automatic event publishing, retry, circuit breaker, and DLQ
   *
   * This is the main method that adapters should use to execute operations
   */
  protected async executeOperation<T>(
    operationName: string,
    operation: () => Promise<T>,
    eventType: string | null = null,
    eventData?: any
  ): Promise<AdapterOperationResult<T>> {
    const correlationId = randomUUID();
    const startTime = Date.now();

    this.logger.info('Executing operation', {
      operation: operationName,
      correlation_id: correlationId,
      adapter: this.adapterName,
    });

    try {
      // Execute with circuit breaker if enabled
      let result: T;
      if (this.circuitBreaker) {
        result = await this.circuitBreaker.execute(async () => {
          // Execute with retry if configured
          if (this.retryConfig.max_retries > 0) {
            return await retryWithBackoff(operation, this.retryConfig);
          }
          return await operation();
        });
      } else {
        // Execute with retry if configured
        if (this.retryConfig.max_retries > 0) {
          result = await retryWithBackoff(operation, this.retryConfig);
        } else {
          result = await operation();
        }
      }

      const duration = Date.now() - startTime;

      // Publish success event if configured
      let eventPublished = false;
      if (this.publishEvents && eventType) {
        await this.publishEvent(eventType, {
          ...eventData,
          operation: operationName,
          success: true,
          duration_ms: duration,
        }, correlationId);
        eventPublished = true;
      }

      this.logger.info('Operation completed successfully', {
        operation: operationName,
        correlation_id: correlationId,
        duration_ms: duration,
        adapter: this.adapterName,
      });

      return {
        success: true,
        data: result,
        duration_ms: duration,
        correlation_id: correlationId,
        event_published: eventPublished,
      };
    } catch (error) {
      const err = error as Error;
      const duration = Date.now() - startTime;

      this.logger.error('Operation failed', err, {
        operation: operationName,
        correlation_id: correlationId,
        duration_ms: duration,
        adapter: this.adapterName,
      });

      // Publish failure event if configured
      if (this.publishEvents && eventType) {
        await this.publishEvent(`${eventType}.Failed`, {
          ...eventData,
          operation: operationName,
          error_message: err.message,
          error_name: err.name,
          duration_ms: duration,
        }, correlationId).catch((publishError) => {
          this.logger.error('Failed to publish error event', publishError as Error, {
            original_error: err.message,
            correlation_id: correlationId,
          });
        });
      }

      // Send to DLQ if this is a logic failure (not transient)
      if (this.dlq && this.isLogicFailure(err)) {
        await this.dlq.enqueue(
          createBaseEvent(
            (eventType || `${operationName}.Failed`) as Event['type'],
            this.adapterName,
            correlationId,
            {
              ...eventData,
              operation: operationName,
              error_message: err.message,
              error_stack: err.stack,
            }
          ),
          err,
          {
            handler: this.adapterName,
            operation: operationName,
          }
        );
      }

      return {
        success: false,
        error: err,
        duration_ms: duration,
        correlation_id: correlationId,
      };
    }
  }

  /**
   * Publish an event to the event bus
   */
  protected async publishEvent(
    eventType: string,
    data: any,
    correlationId?: string
  ): Promise<void> {
    if (!this.publishEvents) {
      return;
    }

    const event = createBaseEvent(
      eventType as Event['type'],
      this.adapterName,
      correlationId || randomUUID(),
      data
    );

    await this.eventBus.publish(event);
  }

  /**
   * Subscribe to an event type
   */
  protected subscribeToEvent(
    eventType: string,
    handler: (event: Event) => Promise<void>
  ): void {
    if (!this.subscribeToEvents) {
      this.logger.warn('Event subscription disabled', {
        event_type: eventType,
        adapter: this.adapterName,
      });
      return;
    }

    this.eventBus.subscribe(eventType, async (event) => {
      try {
        await handler(event);
      } catch (error) {
        this.logger.error('Event handler failed', error as Error, {
          event_type: eventType,
          event_id: event.id,
          correlation_id: event.correlation_id,
          adapter: this.adapterName,
        });

        // Send to DLQ
        if (this.dlq) {
          await this.dlq.enqueue(event, error as Error, {
            handler: this.adapterName,
            event_type: eventType,
          });
        }
      }
    });

    this.logger.info('Subscribed to event', {
      event_type: eventType,
      adapter: this.adapterName,
    });
  }

  /**
   * Determine if an error is a logic failure (should go to DLQ) or transient (should retry)
   *
   * Transient errors: network timeouts, connection refused, rate limiting
   * Logic errors: validation errors, bad data, business logic violations
   */
  protected isLogicFailure(error: Error): boolean {
    const transientPatterns = [
      'timeout',
      'ECONNREFUSED',
      'ECONNRESET',
      'ENOTFOUND',
      'ETIMEDOUT',
      'rate limit',
      'temporary',
      'transient',
    ];

    const errorMsg = error.message.toLowerCase();
    const isTransient = transientPatterns.some((pattern) =>
      errorMsg.includes(pattern.toLowerCase())
    );

    return !isTransient;
  }

  /**
   * Get circuit breaker state
   */
  getCircuitBreakerState() {
    if (!this.circuitBreaker) {
      return null;
    }
    return this.circuitBreaker.getStats();
  }

  /**
   * Reset circuit breaker
   */
  resetCircuitBreaker(): void {
    if (this.circuitBreaker) {
      this.circuitBreaker.reset();
      this.logger.info('Circuit breaker reset', {
        adapter: this.adapterName,
      });
    }
  }

  /**
   * Get DLQ statistics
   */
  getDLQStats() {
    if (!this.dlq) {
      return null;
    }
    return this.dlq.getStats();
  }

  /**
   * Cleanup resources
   */
  async shutdown(): Promise<void> {
    this.logger.info('Shutting down adapter', {
      adapter: this.adapterName,
    });

    // Clear circuit breakers
    if (this.circuitBreaker) {
      this.circuitBreaker.reset();
    }

    this.logger.info('Adapter shutdown complete', {
      adapter: this.adapterName,
    });
  }
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
