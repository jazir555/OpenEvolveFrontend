"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.PluginEventIntegration = void 0;
exports.getPluginEventIntegration = getPluginEventIntegration;
exports.resetPluginEventIntegration = resetPluginEventIntegration;
const structuredLogger_1 = require("../../../../lib/structuredLogger");
const event_bus_1 = require("../../../../orchestration/event-bus");
class PluginEventIntegration {
    constructor(eventBus) {
        this.subscribers = new Map();
        this.eventBus = eventBus || event_bus_1.eventBus;
        this.correlationContext = {
            correlation_id: `plugin-events-${Date.now()}`,
            source_service: 'plugin-event-integration',
            target_service: 'event-bus'
        };
        structuredLogger_1.apiLogger.info('Plugin Event Integration initialized', this.correlationContext);
    }
    /**
     * Subscribe a plugin to events
     */
    subscribePlugin(plugin, eventTypes, handler) {
        const pluginName = plugin.metadata.name;
        structuredLogger_1.apiLogger.info('Subscribing plugin to events', {
            ...this.correlationContext,
            plugin: pluginName,
            event_types: eventTypes
        });
        const subscriber = {
            id: `${pluginName}-${Date.now()}`,
            plugin: pluginName,
            eventTypes,
            subscriptionIds: [],
            handler: async (event) => {
                try {
                    const pluginEvent = {
                        type: event.type,
                        plugin: event.source_service || 'unknown',
                        timestamp: event.timestamp,
                        data: event.data,
                        correlationId: event.correlation_id
                    };
                    await handler(pluginEvent);
                }
                catch (error) {
                    structuredLogger_1.apiLogger.error('Plugin event handler failed', error, {
                        ...this.correlationContext,
                        plugin: pluginName,
                        event_type: event.type
                    });
                }
            }
        };
        // Subscribe to each event type
        for (const eventType of eventTypes) {
            const subscription = this.eventBus.subscribe(eventType, subscriber.handler);
            subscriber.subscriptionIds.push(subscription.subscriptionId);
        }
        this.subscribers.set(`${pluginName}-sub`, subscriber);
    }
    /**
     * Unsubscribe a plugin from events
     */
    unsubscribePlugin(pluginName) {
        const subscriberKey = `${pluginName}-sub`;
        const subscriber = this.subscribers.get(subscriberKey);
        if (subscriber) {
            for (const subscriptionId of subscriber.subscriptionIds) {
                this.eventBus.unsubscribe(subscriptionId);
            }
            this.subscribers.delete(subscriberKey);
            structuredLogger_1.apiLogger.info('Unsubscribed plugin from events', {
                ...this.correlationContext,
                plugin: pluginName
            });
        }
    }
    /**
     * Publish an event from a plugin
     */
    publishEvent(plugin, eventType, data, correlationId) {
        const pluginName = plugin.metadata.name;
        structuredLogger_1.apiLogger.debug('Publishing plugin event', {
            ...this.correlationContext,
            plugin: pluginName,
            event_type: eventType
        });
        void this.eventBus.publish(this.buildEvent(eventType, pluginName, data, correlationId || this.correlationContext.correlation_id));
    }
    /**
     * Emit plugin lifecycle events
     */
    async emitPluginInitialized(plugin) {
        this.publishEvent(plugin, 'plugin.initialized', {
            metadata: plugin.metadata,
            capabilities: plugin.capabilities,
            status: plugin.getStatus()
        });
    }
    async emitPluginError(plugin, error) {
        this.publishEvent(plugin, 'plugin.error', {
            error: error.message,
            stack: error.stack
        });
    }
    async emitPluginStatusChanged(plugin, oldStatus, newStatus) {
        this.publishEvent(plugin, 'plugin.status.changed', {
            oldStatus,
            newStatus,
            timestamp: new Date().toISOString()
        });
    }
    /**
     * Emit workflow events
     */
    async emitWorkflowStarted(workflowId, executionId, input) {
        await this.eventBus.publish(this.buildEvent('workflow.started', 'workflow-orchestrator', {
            workflowId,
            executionId,
            input,
        }));
    }
    async emitWorkflowCompleted(result) {
        await this.eventBus.publish(this.buildEvent('workflow.completed', 'workflow-orchestrator', {
            result,
        }));
    }
    async emitWorkflowFailed(workflowId, executionId, error) {
        await this.eventBus.publish(this.buildEvent('workflow.failed', 'workflow-orchestrator', {
            workflowId,
            executionId,
            error,
        }));
    }
    /**
     * Emit data processing events
     */
    async emitDataProcessed(plugin, processingType, inputSize, outputSize, duration) {
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
    async emitKnowledgeIndexed(plugin, documentId, documentType, indexSize) {
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
    async emitSearchExecuted(plugin, query, resultsCount, duration) {
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
    setupCrossPluginHandlers() {
        // When knowledge is indexed, notify search plugins to refresh
        this.eventBus.subscribe('knowledge.indexed', async (event) => {
            const eventData = this.asEventData(event);
            structuredLogger_1.apiLogger.info('Knowledge indexed, notifying search plugins', {
                ...this.correlationContext,
                document_id: eventData.documentId
            });
            // Publish refresh event for search plugins
            await this.eventBus.publish(this.buildEvent('search.refresh', 'plugin-event-integration', {
                documentId: eventData.documentId,
                documentType: eventData.documentType,
            }));
        });
        // When data is processed, trigger analytics
        this.eventBus.subscribe('data.processed', async (event) => {
            const eventData = this.asEventData(event);
            structuredLogger_1.apiLogger.info('Data processed, triggering analytics', {
                ...this.correlationContext,
                processing_type: eventData.processingType
            });
            // Publish analytics event
            await this.eventBus.publish(this.buildEvent('analytics.track', 'plugin-event-integration', {
                processingType: eventData.processingType,
                inputSize: eventData.inputSize,
                outputSize: eventData.outputSize,
                duration: eventData.duration,
            }));
        });
        // When workflow fails, trigger alert
        this.eventBus.subscribe('workflow.failed', async (event) => {
            const eventData = this.asEventData(event);
            const workflowId = typeof eventData.workflowId === 'string' ? eventData.workflowId : 'unknown';
            const executionId = typeof eventData.executionId === 'string' ? eventData.executionId : 'unknown';
            const errorMessage = typeof eventData.error === 'string' ? eventData.error : String(eventData.error ?? 'Unknown error');
            structuredLogger_1.apiLogger.warn('Workflow failed, triggering alert', {
                ...this.correlationContext,
                workflow_id: workflowId,
                execution_id: executionId
            });
            // Publish alert event
            await this.eventBus.publish(this.buildEvent('alert.workflow', 'plugin-event-integration', {
                workflowId,
                executionId,
                error: errorMessage,
                severity: 'high',
            }));
        });
        structuredLogger_1.apiLogger.info('Cross-plugin event handlers setup complete', this.correlationContext);
    }
    /**
     * Get event statistics
     */
    getEventStatistics() {
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
    destroy() {
        // Unsubscribe all plugin subscribers
        for (const subscriber of this.subscribers.values()) {
            for (const subscriptionId of subscriber.subscriptionIds) {
                this.eventBus.unsubscribe(subscriptionId);
            }
        }
        this.subscribers.clear();
        structuredLogger_1.apiLogger.info('Plugin Event Integration destroyed', this.correlationContext);
    }
    buildEvent(type, sourceService, data, correlationId) {
        const fallbackId = `${sourceService}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        const eventId = typeof globalThis.crypto?.randomUUID === 'function'
            ? globalThis.crypto.randomUUID()
            : fallbackId;
        return {
            id: eventId,
            type,
            timestamp: new Date().toISOString(),
            correlation_id: correlationId || this.correlationContext.correlation_id,
            source_service: sourceService,
            data,
            metadata: {},
        };
    }
    asEventData(event) {
        if (event.data && typeof event.data === 'object') {
            return event.data;
        }
        return {};
    }
}
exports.PluginEventIntegration = PluginEventIntegration;
// Global singleton instance
let globalPluginEvents = null;
/**
 * Get or create the global plugin event integration
 */
function getPluginEventIntegration(eventBus) {
    if (!globalPluginEvents) {
        globalPluginEvents = new PluginEventIntegration(eventBus);
        globalPluginEvents.setupCrossPluginHandlers();
    }
    return globalPluginEvents;
}
/**
 * Reset the global plugin event integration (for testing)
 */
function resetPluginEventIntegration() {
    if (globalPluginEvents) {
        globalPluginEvents.destroy();
        globalPluginEvents = null;
    }
}
//# sourceMappingURL=plugin-events.js.map