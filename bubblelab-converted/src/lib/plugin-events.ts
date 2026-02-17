/**
 * Plugin Event Bus Integration
 *
 * Integrates plugins with the central event bus for cross-plugin communication.
 * Enables plugins to publish events and subscribe to events from other plugins.
 *
 * Event Types:
 * - plugin.initialized: Plugin has been initialized
 * - plugin.error: Plugin encountered an error
 * - plugin.status.changed: Plugin status changed
 * - workflow.started: Workflow execution started
 * - workflow.completed: Workflow execution completed
 * - workflow.failed: Workflow execution failed
 * - data.processed: Data has been processed
 * - knowledge.indexed: Knowledge has been indexed
 * - search.executed: Search has been executed
 */

import { apiLogger, LogContext } from '../../../glue/lib/structuredLogger';
import { eventBus as defaultEventBus, type EventBus } from '../../../glue/orchestration/event-bus';
import type { PluginInterface } from './plugin-registry';
import type { WorkflowDefinition, WorkflowExecutionResult } from './workflow-orchestrator';

export interface PluginEvent {
  type: string;
  plugin: string;
  timestamp: string;
  data: Record<string, unknown>;
  correlationId?: string;
}

export interface PluginEventSubscriber {
  id: string;
  plugin: string;
  eventTypes: string[];
  handler: (event: PluginEvent) => void | Promise<void>;
  subscriptionIds: string[];
}

class PluginEventIntegration {
  private eventBus: EventBus;
  private subscribers = new Map<string, PluginEventSubscriber>();
  private internalSubscriptionIds: string[] = [];
  private correlationContext: LogContext;

  constructor(eventBus?: EventBus) {
    this.eventBus = eventBus || defaultEventBus;
    this.correlationContext = {
      correlation_id: `plugin-events-${Date.now()}`,
      source_service: 'plugin-event-integration',
      target_service: 'event-bus'
    };

    apiLogger.info('Plugin Event Integration initialized', this.correlationContext);
  }

  /**
   * Subscribe a plugin to events
   */
  subscribePlugin(
    plugin: PluginInterface,
    eventTypes: string[],
    handler: (event: PluginEvent) => void | Promise<void>
  ): void {
    const pluginName = plugin.metadata.name;

    apiLogger.info('Subscribing plugin to events', {
      ...this.correlationContext,
      plugin: pluginName,
      event_types: eventTypes
    });

    const subscriber: PluginEventSubscriber = {
      id: `${pluginName}-${Date.now()}`,
      plugin: pluginName,
      eventTypes,
      subscriptionIds: [],
      handler: async (event) => {
        try {
          const pluginEvent: PluginEvent = {
            type: event.type,
            plugin: (event as any).source_service || 'unknown',
            timestamp: event.timestamp,
            data: (event.data as Record<string, unknown>) || {},
            correlationId: (event as any).correlation_id
          };

          await handler(pluginEvent);
        } catch (error) {
          apiLogger.error('Plugin event handler failed', error as Error, {
            ...this.correlationContext,
            plugin: pluginName,
            event_type: event.type
          });
        }
      }
    };

    // Subscribe to each event type
    for (const eventType of eventTypes) {
      const subscription = this.eventBus.subscribe(eventType, subscriber.handler as any);
      subscriber.subscriptionIds.push(subscription.subscriptionId);
    }

    this.subscribers.set(`${pluginName}-sub`, subscriber);
  }

  /**
   * Unsubscribe a plugin from events
   */
  unsubscribePlugin(pluginName: string): void {
    const subscriberKey = `${pluginName}-sub`;
    const subscriber = this.subscribers.get(subscriberKey);

    if (subscriber) {
      for (const subscriptionId of subscriber.subscriptionIds) {
        this.eventBus.unsubscribe(subscriptionId);
      }

      this.subscribers.delete(subscriberKey);

      apiLogger.info('Unsubscribed plugin from events', {
        ...this.correlationContext,
        plugin: pluginName
      });
    }
  }

  /**
   * Publish an event from a plugin
   */
  async publishEvent(
    plugin: PluginInterface,
    eventType: string,
    data: Record<string, unknown>,
    correlationId?: string
  ): Promise<void> {
    const pluginName = plugin.metadata.name;

    apiLogger.debug('Publishing plugin event', {
      ...this.correlationContext,
      plugin: pluginName,
      event_type: eventType
    });

    await this.eventBus.publish({
      id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      type: eventType,
      source_service: pluginName,
      timestamp: new Date().toISOString(),
      data,
      correlation_id: correlationId || this.correlationContext.correlation_id,
      metadata: {},
    } as any);
  }

  /**
   * Emit plugin lifecycle events
   */
  async emitPluginInitialized(plugin: PluginInterface): Promise<void> {
    await this.publishEvent(plugin, 'plugin.initialized', {
      metadata: plugin.metadata,
      capabilities: plugin.capabilities,
      status: plugin.getStatus()
    });
  }

  async emitPluginError(
    plugin: PluginInterface,
    error: Error
  ): Promise<void> {
    await this.publishEvent(plugin, 'plugin.error', {
      error: error.message,
      stack: error.stack
    });
  }

  async emitPluginStatusChanged(
    plugin: PluginInterface,
    oldStatus: string,
    newStatus: string
  ): Promise<void> {
    await this.publishEvent(plugin, 'plugin.status.changed', {
      oldStatus,
      newStatus,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Emit workflow events
   */
  async emitWorkflowStarted(
    workflowId: string,
    executionId: string,
    input: Record<string, unknown>
  ): Promise<void> {
    await this.eventBus.publish({
      id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      type: 'workflow.started',
      source_service: 'workflow-orchestrator',
      timestamp: new Date().toISOString(),
      data: {
        workflowId,
        executionId,
        input
      },
      correlation_id: this.correlationContext.correlation_id,
      metadata: {},
    } as any);
  }

  async emitWorkflowCompleted(
    result: WorkflowExecutionResult
  ): Promise<void> {
    await this.eventBus.publish({
      id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      type: 'workflow.completed',
      source_service: 'workflow-orchestrator',
      timestamp: new Date().toISOString(),
      data: {
        result
      },
      correlation_id: this.correlationContext.correlation_id,
      metadata: {},
    } as any);
  }

  async emitWorkflowFailed(
    workflowId: string,
    executionId: string,
    error: string
  ): Promise<void> {
    await this.eventBus.publish({
      id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      type: 'workflow.failed',
      source_service: 'workflow-orchestrator',
      timestamp: new Date().toISOString(),
      data: {
        workflowId,
        executionId,
        error
      },
      correlation_id: this.correlationContext.correlation_id,
      metadata: {},
    } as any);
  }

  /**
   * Emit data processing events
   */
  async emitDataProcessed(
    plugin: PluginInterface,
    processingType: string,
    inputSize: number,
    outputSize: number,
    duration: number
  ): Promise<void> {
    this.publishEvent(plugin, 'data.processed', {
      processingType,
      inputSize,
      outputSize,
      duration,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Emit knowledge indexing events
   */
  async emitKnowledgeIndexed(
    plugin: PluginInterface,
    documentId: string,
    documentType: string,
    indexSize: number
  ): Promise<void> {
    this.publishEvent(plugin, 'knowledge.indexed', {
      documentId,
      documentType,
      indexSize,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Emit search events
   */
  async emitSearchExecuted(
    plugin: PluginInterface,
    query: string,
    resultsCount: number,
    duration: number
  ): Promise<void> {
    this.publishEvent(plugin, 'search.executed', {
      query,
      resultsCount,
      duration,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Setup cross-plugin event handlers
   */
  setupCrossPluginHandlers(): void {
    // When knowledge is indexed, notify search plugins to refresh
    const knowledgeIndexedSub = this.eventBus.subscribe('knowledge.indexed', async (event: any) => {
      apiLogger.info('Knowledge indexed, notifying search plugins', {
        ...this.correlationContext,
        document_id: event?.data?.documentId
      });

      // Publish refresh event for search plugins
      await this.eventBus.publish({
        id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        type: 'search.refresh',
        source_service: 'plugin-event-integration',
        timestamp: new Date().toISOString(),
        data: {
          documentId: event?.data?.documentId,
          documentType: event?.data?.documentType
        },
        correlation_id: this.correlationContext.correlation_id,
        metadata: {},
      } as any);
    });
    this.internalSubscriptionIds.push(knowledgeIndexedSub.subscriptionId);

    // When data is processed, trigger analytics
    const dataProcessedSub = this.eventBus.subscribe('data.processed', async (event: any) => {
      apiLogger.info('Data processed, triggering analytics', {
        ...this.correlationContext,
        processing_type: event?.data?.processingType
      });

      // Publish analytics event
      await this.eventBus.publish({
        id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        type: 'analytics.track',
        source_service: 'plugin-event-integration',
        timestamp: new Date().toISOString(),
        data: {
          processingType: event?.data?.processingType,
          inputSize: event?.data?.inputSize,
          outputSize: event?.data?.outputSize,
          duration: event?.data?.duration
        },
        correlation_id: this.correlationContext.correlation_id,
        metadata: {},
      } as any);
    });
    this.internalSubscriptionIds.push(dataProcessedSub.subscriptionId);

    // When workflow fails, trigger alert
    const workflowFailedSub = this.eventBus.subscribe('workflow.failed', async (event: any) => {
      apiLogger.warn('Workflow failed, triggering alert', {
        ...this.correlationContext,
        workflow_id: event?.data?.workflowId,
        execution_id: event?.data?.executionId
      });

      // Publish alert event
      await this.eventBus.publish({
        id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        type: 'alert.workflow',
        source_service: 'plugin-event-integration',
        timestamp: new Date().toISOString(),
        data: {
          workflowId: event?.data?.workflowId,
          executionId: event?.data?.executionId,
          error: event?.data?.error,
          severity: 'high'
        },
        correlation_id: this.correlationContext.correlation_id,
        metadata: {},
      } as any);
    });
    this.internalSubscriptionIds.push(workflowFailedSub.subscriptionId);

    apiLogger.info('Cross-plugin event handlers setup complete', this.correlationContext);
  }

  /**
   * Get event statistics
   */
  getEventStatistics(): {
    totalSubscribers: number;
    subscriberPlugins: string[];
    recentEvents: number;
  } {
    const stats = {
      totalSubscribers: this.subscribers.size,
      subscriberPlugins: Array.from(this.subscribers.values()).map(s => s.plugin),
      recentEvents: 0 // Would need to track this
    };

    return stats;
  }

  /**
   * Cleanup
   */
  destroy(): void {
    // Unsubscribe all plugin subscribers
    for (const subscriber of this.subscribers.values()) {
      for (const subscriptionId of subscriber.subscriptionIds) {
        this.eventBus.unsubscribe(subscriptionId);
      }
    }

    this.subscribers.clear();
    for (const subscriptionId of this.internalSubscriptionIds) {
      this.eventBus.unsubscribe(subscriptionId);
    }
    this.internalSubscriptionIds = [];

    apiLogger.info('Plugin Event Integration destroyed', this.correlationContext);
  }
}

// Global singleton instance
let globalPluginEvents: PluginEventIntegration | null = null;

/**
 * Get or create the global plugin event integration
 */
export function getPluginEventIntegration(eventBus?: EventBus): PluginEventIntegration {
  if (!globalPluginEvents) {
    globalPluginEvents = new PluginEventIntegration(eventBus);
    globalPluginEvents.setupCrossPluginHandlers();
  }
  return globalPluginEvents;
}

/**
 * Reset the global plugin event integration (for testing)
 */
export function resetPluginEventIntegration(): void {
  if (globalPluginEvents) {
    globalPluginEvents.destroy();
    globalPluginEvents = null;
  }
}

export { PluginEventIntegration };
