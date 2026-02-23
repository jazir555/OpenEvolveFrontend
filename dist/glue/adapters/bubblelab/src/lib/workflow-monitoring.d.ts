/**
 * Workflow Monitoring and Telemetry
 *
 * Collects and reports metrics for workflow executions.
 * Provides observability into workflow performance and failures.
 */
import type { WorkflowExecutionResult, WorkflowContext } from './workflow-orchestrator';
export interface WorkflowMetrics {
    workflowId: string;
    executionId: string;
    startTime: number;
    endTime: number;
    duration: number;
    status: string;
    stepCount: number;
    completedSteps: number;
    failedSteps: number;
    errorCount: number;
    pluginUsage: Record<string, number>;
}
export interface StepMetrics {
    workflowId: string;
    executionId: string;
    stepId: string;
    stepName: string;
    plugin: string;
    action: string;
    startTime: number;
    endTime: number;
    duration: number;
    status: 'success' | 'failure';
    errorMessage?: string;
    retryCount: number;
}
export interface TelemetryConfig {
    enabled?: boolean;
    sampleRate?: number;
    maxHistorySize?: number;
    persistToLocalStorage?: boolean;
    storageKey?: string;
}
declare class WorkflowMonitor {
    private config;
    private workflowMetrics;
    private stepMetrics;
    private correlationContext;
    constructor(config?: TelemetryConfig);
    /**
     * Record workflow start
     */
    recordWorkflowStart(context: WorkflowContext): void;
    /**
     * Record workflow completion
     */
    recordWorkflowCompletion(context: WorkflowContext, result: WorkflowExecutionResult): void;
    /**
     * Record step execution
     */
    recordStepExecution(context: WorkflowContext, stepId: string, stepName: string, plugin: string, action: string, startTime: number, endTime: number, success: boolean, errorMessage?: string, retryCount?: number): void;
    /**
     * Get metrics for a specific workflow execution
     */
    getWorkflowMetrics(executionId: string): WorkflowMetrics | undefined;
    /**
     * Get step metrics for a specific workflow execution
     */
    getStepMetrics(executionId: string): StepMetrics[];
    /**
     * Get all workflow metrics
     */
    getAllWorkflowMetrics(): WorkflowMetrics[];
    /**
     * Get metrics by workflow ID
     */
    getMetricsByWorkflow(workflowId: string): WorkflowMetrics[];
    /**
     * Get metrics by status
     */
    getMetricsByStatus(status: string): WorkflowMetrics[];
    /**
     * Get aggregate statistics
     */
    getAggregateStats(): {
        totalExecutions: number;
        successfulExecutions: number;
        failedExecutions: number;
        averageDuration: number;
        totalSteps: number;
        averageStepsPerWorkflow: number;
        mostUsedPlugins: Array<{
            plugin: string;
            count: number;
        }>;
        errorRate: number;
    };
    /**
     * Get performance percentiles
     */
    getPerformancePercentiles(): {
        p50: number;
        p75: number;
        p90: number;
        p95: number;
        p99: number;
    };
    /**
     * Get error summary
     */
    getErrorSummary(): Array<{
        workflowId: string;
        executionId: string;
        errorCount: number;
        lastError: string;
    }>;
    /**
     * Clear old metrics
     */
    clearMetrics(olderThan?: number): void;
    /**
     * Clear all metrics
     */
    clearAllMetrics(): void;
    /**
     * Enforce maximum history size
     */
    private enforceHistorySize;
    /**
     * Save metrics to localStorage
     */
    private saveToStorage;
    /**
     * Load metrics from localStorage
     */
    private loadFromStorage;
    /**
     * Export metrics as JSON
     */
    exportMetrics(): string;
    private hasLocalStorage;
}
/**
 * Get or create the global workflow monitor
 */
export declare function getWorkflowMonitor(config?: TelemetryConfig): WorkflowMonitor;
/**
 * Reset the global workflow monitor (for testing)
 */
export declare function resetWorkflowMonitor(): void;
export { WorkflowMonitor };
//# sourceMappingURL=workflow-monitoring.d.ts.map