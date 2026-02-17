/**
 * Workflow Monitoring and Telemetry
 *
 * Collects and reports metrics for workflow executions.
 * Provides observability into workflow performance and failures.
 */

import { apiLogger, LogContext } from '../../../glue/lib/structuredLogger';
import type { WorkflowDefinition, WorkflowExecutionResult, WorkflowContext } from './workflow-orchestrator';

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
  sampleRate?: number; // 0.0 to 1.0
  maxHistorySize?: number;
  persistToLocalStorage?: boolean;
  storageKey?: string;
}

class WorkflowMonitor {
  private config: TelemetryConfig;
  private workflowMetrics = new Map<string, WorkflowMetrics>();
  private stepMetrics = new Map<string, StepMetrics[]>();
  private correlationContext: LogContext;

  constructor(config: TelemetryConfig = {}) {
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

    apiLogger.info('Workflow Monitor initialized', {
      ...this.correlationContext,
      config: this.config
    });
  }

  /**
   * Record workflow start
   */
  recordWorkflowStart(context: WorkflowContext): void {
    if (!this.config.enabled || Math.random() > this.config.sampleRate) {
      return;
    }

    const metrics: WorkflowMetrics = {
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

    apiLogger.debug('Workflow start recorded', {
      ...this.correlationContext,
      workflow_id: context.workflowId,
      execution_id: context.executionId
    });
  }

  /**
   * Record workflow completion
   */
  recordWorkflowCompletion(
    context: WorkflowContext,
    result: WorkflowExecutionResult
  ): void {
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
    metrics.completedSteps = Array.from(result.stepResults.values()).filter(
      (r: any) => r.success !== false
    ).length;
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

    apiLogger.info('Workflow completion recorded', {
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
  recordStepExecution(
    context: WorkflowContext,
    stepId: string,
    stepName: string,
    plugin: string,
    action: string,
    startTime: number,
    endTime: number,
    success: boolean,
    errorMessage?: string,
    retryCount: number = 0
  ): void {
    if (!this.config.enabled) {
      return;
    }

    const metrics: StepMetrics = {
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

    apiLogger.debug('Step execution recorded', {
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
  getWorkflowMetrics(executionId: string): WorkflowMetrics | undefined {
    return this.workflowMetrics.get(executionId);
  }

  /**
   * Get step metrics for a specific workflow execution
   */
  getStepMetrics(executionId: string): StepMetrics[] {
    return this.stepMetrics.get(executionId) || [];
  }

  /**
   * Get all workflow metrics
   */
  getAllWorkflowMetrics(): WorkflowMetrics[] {
    return Array.from(this.workflowMetrics.values());
  }

  /**
   * Get metrics by workflow ID
   */
  getMetricsByWorkflow(workflowId: string): WorkflowMetrics[] {
    return Array.from(this.workflowMetrics.values()).filter(
      m => m.workflowId === workflowId
    );
  }

  /**
   * Get metrics by status
   */
  getMetricsByStatus(status: string): WorkflowMetrics[] {
    return Array.from(this.workflowMetrics.values()).filter(
      m => m.status === status
    );
  }

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
    mostUsedPlugins: Array<{ plugin: string; count: number }>;
    errorRate: number;
  } {
    const allMetrics = Array.from(this.workflowMetrics.values());

    const totalExecutions = allMetrics.length;
    const successfulExecutions = allMetrics.filter(m => m.status === 'completed').length;
    const failedExecutions = allMetrics.filter(m => m.status === 'failed').length;

    const totalDuration = allMetrics.reduce((sum, m) => sum + m.duration, 0);
    const averageDuration = totalExecutions > 0 ? totalDuration / totalExecutions : 0;

    const totalSteps = allMetrics.reduce((sum, m) => sum + m.stepCount, 0);
    const averageStepsPerWorkflow = totalExecutions > 0 ? totalSteps / totalExecutions : 0;

    // Aggregate plugin usage
    const pluginUsage = new Map<string, number>();
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
  getPerformancePercentiles(): {
    p50: number;
    p75: number;
    p90: number;
    p95: number;
    p99: number;
  } {
    const durations = Array.from(this.workflowMetrics.values())
      .map(m => m.duration)
      .sort((a, b) => a - b);

    if (durations.length === 0) {
      return { p50: 0, p75: 0, p90: 0, p95: 0, p99: 0 };
    }

    const percentile = (p: number) => {
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
  getErrorSummary(): Array<{
    workflowId: string;
    executionId: string;
    errorCount: number;
    lastError: string;
  }> {
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
  clearMetrics(olderThan?: number): void {
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

    apiLogger.info('Old metrics cleared', {
      ...this.correlationContext,
      cutoff: new Date(cutoff).toISOString()
    });
  }

  /**
   * Clear all metrics
   */
  clearAllMetrics(): void {
    this.workflowMetrics.clear();
    this.stepMetrics.clear();

    if (this.config.persistToLocalStorage) {
      this.saveToStorage();
    }

    apiLogger.info('All metrics cleared', this.correlationContext);
  }

  /**
   * Enforce maximum history size
   */
  private enforceHistorySize(): void {
    if (this.workflowMetrics.size <= this.config.maxHistorySize!) {
      return;
    }

    // Sort by start time and remove oldest
    const entries = Array.from(this.workflowMetrics.entries())
      .sort((a, b) => a[1].startTime - b[1].startTime);

    const toRemove = entries.slice(0, entries.length - this.config.maxHistorySize!);
    for (const [executionId] of toRemove) {
      this.workflowMetrics.delete(executionId);
      this.stepMetrics.delete(executionId);
    }
  }

  /**
   * Save metrics to localStorage
   */
  private saveToStorage(): void {
    try {
      const data = {
        workflowMetrics: Array.from(this.workflowMetrics.entries()),
        stepMetrics: Array.from(this.stepMetrics.entries())
      };
      localStorage.setItem(this.config.storageKey!, JSON.stringify(data));
    } catch (error) {
      apiLogger.error('Failed to save metrics to localStorage', error as Error, this.correlationContext);
    }
  }

  /**
   * Load metrics from localStorage
   */
  private loadFromStorage(): void {
    try {
      const data = localStorage.getItem(this.config.storageKey!);
      if (data) {
        const parsed = JSON.parse(data);
        this.workflowMetrics = new Map(parsed.workflowMetrics || []);
        this.stepMetrics = new Map(parsed.stepMetrics || []);

        apiLogger.info('Metrics loaded from localStorage', {
          ...this.correlationContext,
          count: this.workflowMetrics.size
        });
      }
    } catch (error) {
      apiLogger.error('Failed to load metrics from localStorage', error as Error, this.correlationContext);
    }
  }

  /**
   * Export metrics as JSON
   */
  exportMetrics(): string {
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

// Global singleton instance
let globalMonitor: WorkflowMonitor | null = null;

/**
 * Get or create the global workflow monitor
 */
export function getWorkflowMonitor(config?: TelemetryConfig): WorkflowMonitor {
  if (!globalMonitor) {
    globalMonitor = new WorkflowMonitor(config);
  }
  return globalMonitor;
}

/**
 * Reset the global workflow monitor (for testing)
 */
export function resetWorkflowMonitor(): void {
  if (globalMonitor) {
    globalMonitor.clearAllMetrics();
    globalMonitor = null;
  }
}

export { WorkflowMonitor };
export type { WorkflowMetrics, StepMetrics, TelemetryConfig };
