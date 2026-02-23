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
 * - gauntlet.execution.started: Gauntlet execution started
 * - gauntlet.round.completed: Gauntlet round completed
 * - gauntlet.execution.completed: Gauntlet execution completed
 * - gauntlet.execution.failed: Gauntlet execution failed
 * - decomposition.execution.started: Decomposition execution started
 * - decomposition.subproblem.solved: Sub-problem solved
 * - decomposition.execution.completed: Decomposition execution completed
 * - decomposition.execution.failed: Decomposition execution failed
 * - data.processed: Data has been processed
 * - knowledge.indexed: Knowledge has been indexed
 * - search.executed: Search has been executed
 */
import { type EventBus, type EventSubscriber } from '../../../glue/orchestration/event-bus';
import type { PluginInterface } from './plugin-registry';
import type { WorkflowExecutionResult } from './workflow-orchestrator';
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
declare class PluginEventIntegration {
    private eventBus;
    private subscribers;
    private correlationContext;
    constructor(eventBus?: EventBus);
    /**
     * Subscribe a plugin to events
     */
    subscribePlugin(plugin: PluginInterface, eventTypes: string[], handler: (event: PluginEvent) => void | Promise<void>): void;
    /**
     * Unsubscribe a plugin from events
     */
    unsubscribePlugin(pluginName: string): void;
    /**
     * Publish an event from a plugin
     */
    publishEvent(plugin: PluginInterface, eventType: string, data: Record<string, unknown>, correlationId?: string): void;
    /**
     * Emit plugin lifecycle events
     */
    emitPluginInitialized(plugin: PluginInterface): Promise<void>;
    emitPluginError(plugin: PluginInterface, error: Error): Promise<void>;
    emitPluginStatusChanged(plugin: PluginInterface, oldStatus: string, newStatus: string): Promise<void>;
    /**
     * Emit workflow events
     */
    emitWorkflowStarted(workflowId: string, executionId: string, input: Record<string, unknown>): Promise<void>;
    emitWorkflowCompleted(result: WorkflowExecutionResult): Promise<void>;
    emitWorkflowFailed(workflowId: string, executionId: string, error: string): Promise<void>;
    /**
     * Emit gauntlet execution events
     */
    emitGauntletExecutionStarted(gauntletName: string, executionId: string, content: string): Promise<void>;
    emitGauntletRoundCompleted(gauntletName: string, executionId: string, roundNumber: number, results: {
        team_scores: Record<string, number>;
        validation_results: Array<{
            team: string;
            approved: boolean;
            score: number;
            feedback: string;
        }>;
    }): Promise<void>;
    emitGauntletExecutionCompleted(gauntletName: string, executionId: string, finalResults: {
        approved: boolean;
        overall_score: number;
        rounds_completed: number;
        formal_verification?: boolean;
    }): Promise<void>;
    emitGauntletExecutionFailed(gauntletName: string, executionId: string, error: string, roundNumber?: number): Promise<void>;
    /**
     * Emit decomposition execution events
     */
    emitDecompositionStarted(workflowId: string, executionId: string, problemStatement: string): Promise<void>;
    emitDecompositionSubProblemSolved(workflowId: string, executionId: string, subProblemId: string, solution: string, dependenciesSolved: string[], executionTimeMs: number): Promise<void>;
    emitDecompositionCompleted(workflowId: string, executionId: string, finalSolution: string, subProblemsCount: number, executionTimeMs: number): Promise<void>;
    emitDecompositionFailed(workflowId: string, executionId: string, error: string, failedSubProblemId?: string): Promise<void>;
    /**
     * Emit data processing events
     */
    emitDataProcessed(plugin: PluginInterface, processingType: string, inputSize: number, outputSize: number, duration: number): Promise<void>;
    /**
     * Emit knowledge indexing events
     */
    emitKnowledgeIndexed(plugin: PluginInterface, documentId: string, documentType: string, indexSize: number): Promise<void>;
    /**
     * Emit search events
     */
    emitSearchExecuted(plugin: PluginInterface, query: string, resultsCount: number, duration: number): Promise<void>;
    /**
     * Setup cross-plugin event handlers
     */
    setupCrossPluginHandlers(): void;
    /**
     * Get event statistics
     */
    getEventStatistics(): {
        totalSubscribers: number;
        subscriberPlugins: string[];
        recentEvents: number;
    };
    /**
     * Cleanup
     */
    destroy(): void;
}
/**
 * Get or create the global plugin event integration
 */
export declare function getPluginEventIntegration(eventBus?: EventBus): PluginEventIntegration;
/**
 * Reset the global plugin event integration (for testing)
 */
export declare function resetPluginEventIntegration(): void;
export { PluginEventIntegration };
//# sourceMappingURL=plugin-events.d.ts.map