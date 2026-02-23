"use strict";
/**
 * Workflow Monitoring and Telemetry
 *
 * Collects and reports metrics for workflow executions.
 * Provides observability into workflow performance and failures.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.WorkflowMonitor = void 0;
exports.getWorkflowMonitor = getWorkflowMonitor;
exports.resetWorkflowMonitor = resetWorkflowMonitor;
const structured_logger_1 = require("../../../glue/lib/structured-logger");
class WorkflowMonitor {
    constructor(config = {}) {
        this.workflowMetrics = new Map();
        this.stepMetrics = new Map();
        this.config = {
            enabled: true,
            sampleRate: 1.0, // Sample 100% by default
            maxHistorySize: 1000,
            persistToLocalStorage: true,
            storageKey: 'workflow-telemetry',
            ...config
        };
        this.correlationContext = {
            correlation_id: `workflow-monitor-${Date.now()}`,
            source_service: 'workflow-monitor',
            target_service: 'telemetry'
        };
        // Load existing metrics from localStorage
        if (this.config.persistToLocalStorage) {
            this.loadFromStorage();
        }
        structured_logger_1.apiLogger.info('Workflow Monitor initialized', {
            ...this.correlationContext,
            config: this.config
        });
    }
    /**
     * Record workflow start
     */
    recordWorkflowStart(context) {
        if (!this.config.enabled || Math.random() > this.config.sampleRate) {
            return;
        }
        const metrics = {
            workflowId: context.workflowId,
            executionId: context.executionId,
            startTime: context.startTime,
            endTime: 0,
            duration: 0,
            status: 'running',
            stepCount: 0,
            completedSteps: 0,
            failedSteps: 0,
            errorCount: 0,
            pluginUsage: {}
        };
        this.workflowMetrics.set(context.executionId, metrics);
        this.stepMetrics.set(context.executionId, []);
        structured_logger_1.apiLogger.debug('Workflow start recorded', {
            ...this.correlationContext,
            workflow_id: context.workflowId,
            execution_id: context.executionId
        });
    }
    /**
     * Record workflow completion
     */
    recordWorkflowCompletion(context, result) {
        if (!this.config.enabled) {
            return;
        }
        const metrics = this.workflowMetrics.get(context.executionId);
        if (!metrics) {
            return;
        }
        metrics.endTime = Date.now();
        metrics.duration = result.duration;
        metrics.status = result.status;
        metrics.stepCount = result.stepResults.size;
        metrics.completedSteps = Array.from(result.stepResults.values()).filter((r) => r.success !== false).length;
        metrics.failedSteps = result.errors.length;
        metrics.errorCount = result.errors.length;
        // Count plugin usage
        const stepMetrics = this.stepMetrics.get(context.executionId) || [];
        for (const step of stepMetrics) {
            metrics.pluginUsage[step.plugin] = (metrics.pluginUsage[step.plugin] || 0) + 1;
        }
        // Enforce max history size
        this.enforceHistorySize();
        // Persist to storage
        if (this.config.persistToLocalStorage) {
            this.saveToStorage();
        }
        structured_logger_1.apiLogger.info('Workflow completion recorded', {
            ...this.correlationContext,
            workflow_id: context.workflowId,
            execution_id: context.executionId,
            status: metrics.status,
            duration: metrics.duration
        });
    }
    /**
     * Record step execution
     */
    recordStepExecution(context, stepId, stepName, plugin, action, startTime, endTime, success, errorMessage, retryCount = 0) {
        if (!this.config.enabled) {
            return;
        }
        const metrics = {
            workflowId: context.workflowId,
            executionId: context.executionId,
            stepId,
            stepName,
            plugin,
            action,
            startTime,
            endTime,
            duration: endTime - startTime,
            status: success ? 'success' : 'failure',
            errorMessage,
            retryCount
        };
        const steps = this.stepMetrics.get(context.executionId) || [];
        steps.push(metrics);
        this.stepMetrics.set(context.executionId, steps);
        structured_logger_1.apiLogger.debug('Step execution recorded', {
            ...this.correlationContext,
            step_id: stepId,
            plugin,
            action,
            duration: metrics.duration,
            status: metrics.status
        });
    }
    /**
     * Get metrics for a specific workflow execution
     */
    getWorkflowMetrics(executionId) {
        return this.workflowMetrics.get(executionId);
    }
    /**
     * Get step metrics for a specific workflow execution
     */
    getStepMetrics(executionId) {
        return this.stepMetrics.get(executionId) || [];
    }
    /**
     * Get all workflow metrics
     */
    getAllWorkflowMetrics() {
        return Array.from(this.workflowMetrics.values());
    }
    /**
     * Get metrics by workflow ID
     */
    getMetricsByWorkflow(workflowId) {
        return Array.from(this.workflowMetrics.values()).filter(m => m.workflowId === workflowId);
    }
    /**
     * Get metrics by status
     */
    getMetricsByStatus(status) {
        return Array.from(this.workflowMetrics.values()).filter(m => m.status === status);
    }
    /**
     * Get aggregate statistics
     */
    getAggregateStats() {
        const allMetrics = Array.from(this.workflowMetrics.values());
        const totalExecutions = allMetrics.length;
        const successfulExecutions = allMetrics.filter(m => m.status === 'completed').length;
        const failedExecutions = allMetrics.filter(m => m.status === 'failed').length;
        const totalDuration = allMetrics.reduce((sum, m) => sum + m.duration, 0);
        const averageDuration = totalExecutions > 0 ? totalDuration / totalExecutions : 0;
        const totalSteps = allMetrics.reduce((sum, m) => sum + m.stepCount, 0);
        const averageStepsPerWorkflow = totalExecutions > 0 ? totalSteps / totalExecutions : 0;
        // Aggregate plugin usage
        const pluginUsage = new Map();
        for (const metrics of allMetrics) {
            for (const [plugin, count] of Object.entries(metrics.pluginUsage)) {
                pluginUsage.set(plugin, (pluginUsage.get(plugin) || 0) + count);
            }
        }
        const mostUsedPlugins = Array.from(pluginUsage.entries())
            .map(([plugin, count]) => ({ plugin, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);
        const totalErrors = allMetrics.reduce((sum, m) => sum + m.errorCount, 0);
        const errorRate = totalSteps > 0 ? totalErrors / totalSteps : 0;
        return {
            totalExecutions,
            successfulExecutions,
            failedExecutions,
            averageDuration,
            totalSteps,
            averageStepsPerWorkflow,
            mostUsedPlugins,
            errorRate
        };
    }
    /**
     * Get performance percentiles
     */
    getPerformancePercentiles() {
        const durations = Array.from(this.workflowMetrics.values())
            .map(m => m.duration)
            .sort((a, b) => a - b);
        if (durations.length === 0) {
            return { p50: 0, p75: 0, p90: 0, p95: 0, p99: 0 };
        }
        const percentile = (p) => {
            const index = Math.ceil((durations.length * p) / 100) - 1;
            return durations[Math.max(0, index)];
        };
        return {
            p50: percentile(50),
            p75: percentile(75),
            p90: percentile(90),
            p95: percentile(95),
            p99: percentile(99)
        };
    }
    /**
     * Get error summary
     */
    getErrorSummary() {
        return Array.from(this.workflowMetrics.values())
            .filter(m => m.errorCount > 0)
            .map(m => {
            const steps = this.stepMetrics.get(m.executionId) || [];
            const lastError = steps
                .filter(s => s.status === 'failure')
                .pop()?.errorMessage || 'Unknown error';
            return {
                workflowId: m.workflowId,
                executionId: m.executionId,
                errorCount: m.errorCount,
                lastError
            };
        })
            .sort((a, b) => b.errorCount - a.errorCount);
    }
    /**
     * Clear old metrics
     */
    clearMetrics(olderThan) {
        const cutoff = olderThan || Date.now() - 7 * 24 * 60 * 60 * 1000; // 7 days default
        for (const [executionId, metrics] of this.workflowMetrics.entries()) {
            if (metrics.startTime < cutoff) {
                this.workflowMetrics.delete(executionId);
                this.stepMetrics.delete(executionId);
            }
        }
        if (this.config.persistToLocalStorage) {
            this.saveToStorage();
        }
        structured_logger_1.apiLogger.info('Old metrics cleared', {
            ...this.correlationContext,
            cutoff: new Date(cutoff).toISOString()
        });
    }
    /**
     * Clear all metrics
     */
    clearAllMetrics() {
        this.workflowMetrics.clear();
        this.stepMetrics.clear();
        if (this.config.persistToLocalStorage) {
            this.saveToStorage();
        }
        structured_logger_1.apiLogger.info('All metrics cleared', this.correlationContext);
    }
    /**
     * Enforce maximum history size
     */
    enforceHistorySize() {
        if (this.workflowMetrics.size <= this.config.maxHistorySize) {
            return;
        }
        // Sort by start time and remove oldest
        const entries = Array.from(this.workflowMetrics.entries())
            .sort((a, b) => a[1].startTime - b[1].startTime);
        const toRemove = entries.slice(0, entries.length - this.config.maxHistorySize);
        for (const [executionId] of toRemove) {
            this.workflowMetrics.delete(executionId);
            this.stepMetrics.delete(executionId);
        }
    }
    /**
     * Save metrics to localStorage
     */
    saveToStorage() {
        try {
            const data = {
                workflowMetrics: Array.from(this.workflowMetrics.entries()),
                stepMetrics: Array.from(this.stepMetrics.entries())
            };
            localStorage.setItem(this.config.storageKey, JSON.stringify(data));
        }
        catch (error) {
            structured_logger_1.apiLogger.error('Failed to save metrics to localStorage', error, this.correlationContext);
        }
    }
    /**
     * Load metrics from localStorage
     */
    loadFromStorage() {
        try {
            const data = localStorage.getItem(this.config.storageKey);
            if (data) {
                const parsed = JSON.parse(data);
                this.workflowMetrics = new Map(parsed.workflowMetrics || []);
                this.stepMetrics = new Map(parsed.stepMetrics || []);
                structured_logger_1.apiLogger.info('Metrics loaded from localStorage', {
                    ...this.correlationContext,
                    count: this.workflowMetrics.size
                });
            }
        }
        catch (error) {
            structured_logger_1.apiLogger.error('Failed to load metrics from localStorage', error, this.correlationContext);
        }
    }
    /**
     * Export metrics as JSON
     */
    exportMetrics() {
        const data = {
            workflowMetrics: Array.from(this.workflowMetrics.entries()),
            stepMetrics: Array.from(this.stepMetrics.entries()),
            aggregateStats: this.getAggregateStats(),
            performancePercentiles: this.getPerformancePercentiles(),
            errorSummary: this.getErrorSummary(),
            exportedAt: new Date().toISOString()
        };
        return JSON.stringify(data, null, 2);
    }
}
exports.WorkflowMonitor = WorkflowMonitor;
// Global singleton instance
let globalMonitor = null;
/**
 * Get or create the global workflow monitor
 */
function getWorkflowMonitor(config) {
    if (!globalMonitor) {
        globalMonitor = new WorkflowMonitor(config);
    }
    return globalMonitor;
}
/**
 * Reset the global workflow monitor (for testing)
 */
function resetWorkflowMonitor() {
    if (globalMonitor) {
        globalMonitor.clearAllMetrics();
        globalMonitor = null;
    }
}
//# sourceMappingURL=workflow-monitoring.js.map