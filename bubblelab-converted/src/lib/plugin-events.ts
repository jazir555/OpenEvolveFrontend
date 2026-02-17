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
import { getEventBus, type EventBus, type EventSubscriber } from '../../../glue/orchestration/event-bus';
import type { PluginInterface } from './plugin-registry';
import type { WorkflowDefinition, WorkflowExecutionResult } from './workflow-orchestrator';

export interface PluginEvent {
  type: string;
  plugin: string;
  timestamp: string;
  data: Record<string, unknown>;
  correlationId?: string;
}

export interface PluginEventSubscriber extends EventSubscriber {
  plugin: string;
}

class PluginEventIntegration {
  private eventBus: EventBus;
  private subscribers = new Map<string, PluginEventSubscriber>();
  private correlationContext: LogContext;

  constructor(eventBus?: EventBus) {
    this.eventBus = eventBus || getEventBus();
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
      handler: async (event) => {
        try {
          const pluginEvent: PluginEvent = {
            type: event.type,
            plugin: event.source || 'unknown',
            timestamp: event.timestamp,
            data: event.data,
            correlationId: event.correlationId
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
      this.eventBus.subscribe(eventType, subscriber);
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
      for (const eventType of subscriber.eventTypes) {
        this.eventBus.unsubscribe(eventType, subscriber.id);
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
  publishEvent(
    plugin: PluginInterface,
    eventType: string,
    data: Record<string, unknown>,
    correlationId?: string
  ): void {
    const pluginName = plugin.metadata.name;

    apiLogger.debug('Publishing plugin event', {
      ...this.correlationContext,
      plugin: pluginName,
      event_type: eventType
    });

    this.eventBus.publish({
      type: eventType,
      source: pluginName,
      timestamp: new Date().toISOString(),
      data,
      correlationId: correlationId || this.correlationContext.correlation_id
    });
  }

  /**
   * Emit plugin lifecycle events
   */
  async emitPluginInitialized(plugin: PluginInterface): Promise<void> {
    this.publishEvent(plugin, 'plugin.initialized', {
      metadata: plugin.metadata,
      capabilities: plugin.capabilities,
      status: plugin.getStatus()
    });
  }

  async emitPluginError(
    plugin: PluginInterface,
    error: Error
  ): Promise<void> {
    this.publishEvent(plugin, 'plugin.error', {
      error: error.message,
      stack: error.stack
    });
  }

  async emitPluginStatusChanged(
    plugin: PluginInterface,
    oldStatus: string,
    newStatus: string
  ): Promise<void> {
    this.publishEvent(plugin, 'plugin.status.changed', {
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
    this.eventBus.publish({
      type: 'workflow.started',
      source: 'workflow-orchestrator',
      timestamp: new Date().toISOString(),
      data: {
        workflowId,
        executionId,
        input
      }
    });
  }

  async emitWorkflowCompleted(
    result: WorkflowExecutionResult
  ): Promise<void> {
    this.eventBus.publish({
      type: 'workflow.completed',
      source: 'workflow-orchestrator',
      timestamp: new Date().toISOString(),
      data: {
        result
      }
    });
  }

  async emitWorkflowFailed(
    workflowId: string,
    executionId: string,
    error: string
  ): Promise<void> {
    this.eventBus.publish({
      type: 'workflow.failed',
      source: 'workflow-orchestrator',
      timestamp: new Date().toISOString(),
      data: {
        workflowId,
        executionId,
        error
      }
    });
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
    this.eventBus.subscribe('knowledge.indexed', {
      id: 'cross-plugin-knowledge-indexed',
      eventTypes: ['knowledge.indexed'],
      handler: async (event) => {
        apiLogger.info('Knowledge indexed, notifying search plugins', {
          ...this.correlationContext,
          document_id: event.data.documentId
        });

        // Publish refresh event for search plugins
        this.eventBus.publish({
          type: 'search.refresh',
          source: 'plugin-event-integration',
          timestamp: new Date().toISOString(),
          data: {
            documentId: event.data.documentId,
            documentType: event.data.documentType
          }
        });
      }
    });

    // When data is processed, trigger analytics
    this.eventBus.subscribe('data.processed', {
      id: 'cross-plugin-data-processed',
      eventTypes: ['data.processed'],
      handler: async (event) => {
        apiLogger.info('Data processed, triggering analytics', {
          ...this.correlationContext,
          processing_type: event.data.processingType
        });

        // Publish analytics event
        this.eventBus.publish({
          type: 'analytics.track',
          source: 'plugin-event-integration',
          timestamp: new Date().toISOString(),
          data: {
            processingType: event.data.processingType,
            inputSize: event.data.inputSize,
            outputSize: event.data.outputSize,
            duration: event.data.duration
          }
        });
      }
    });

    // When workflow fails, trigger alert
    this.eventBus.subscribe('workflow.failed', {
      id: 'cross-plugin-workflow-failed',
      eventTypes: ['workflow.failed'],
      handler: async (event) => {
        apiLogger.warn('Workflow failed, triggering alert', {
          ...this.correlationContext,
          workflow_id: event.data.workflowId,
          execution_id: event.data.executionId
        });

        // Publish alert event
        this.eventBus.publish({
          type: 'alert.workflow',
          source: 'plugin-event-integration',
          timestamp: new Date().toISOString(),
          data: {
            workflowId: event.data.workflowId,
            executionId: event.data.executionId,
            error: event.data.error,
            severity: 'high'
          }
        });
      }
    });

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
      for (const eventType of subscriber.eventTypes) {
        this.eventBus.unsubscribe(eventType, subscriber.id);
      }
    }

    this.subscribers.clear();

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
