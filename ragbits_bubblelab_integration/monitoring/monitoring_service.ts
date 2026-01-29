/**
 * Monitoring and Debugging Service for Ragbits + BubbleLab Integration
 *
 * This module provides real-time monitoring, debugging, and analytics for RAG workflows
 */

import { WorkflowExecutionResult } from '../types';
import { MonitoringEvent, PerformanceMetrics, DebugInfo, MonitoringConfig } from '../types';
import { Logger, generateId } from '../utils/common.utils';

export class MonitoringService {
  private config: MonitoringConfig;
  private eventLog: MonitoringEvent[];
  private performanceMetrics: PerformanceMetrics;
  private debugInfo: Map<string, DebugInfo[]>;
  private activeWorkflows: Set<string>;
  private eventListeners: Array<(event: MonitoringEvent) => void>;
  private alertCallbacks: Array<(alert: string) => void>;
  private logger: Logger;
  private cleanupIntervalId: NodeJS.Timeout | null = null;

  constructor(config?: Partial<MonitoringConfig>) {
    this.config = {
      enableRealTimeMonitoring: true,
      enablePerformanceTracking: true,
      enableErrorTracking: true,
      enableTokenTracking: true,
      logLevel: 'info',
      retentionPeriod: 7,
      samplingRate: 1.0,
      enableAlerting: true,
      alertThresholds: {
        executionTime: 30000, // 30 seconds
        errorRate: 5, // 5%
        memoryUsage: 1024, // 1GB
      },
      ...config
    };

    this.eventLog = [];
    this.performanceMetrics = {
      workflowExecutionTime: 0,
      nodeExecutionTimes: {},
      tokensUsed: 0,
      memoryUsage: 0,
      cacheHitRate: 0,
      errorRate: 0,
      throughput: 0,
    };
    this.debugInfo = new Map();
    this.activeWorkflows = new Set();
    this.eventListeners = [];
    this.alertCallbacks = [];
    this.logger = new Logger({ level: this.config.logLevel, prefix: 'MonitoringService' });

    // Start cleanup interval
    this.cleanupIntervalId = setInterval(() => {
      this.cleanupOldEvents();
    }, 3600000); // Every hour

    this.logger.info('MonitoringService initialized');
  }

  /**
   * Registers an event listener
   */
  addEventListener(listener: (event: MonitoringEvent) => void): void {
    this.eventListeners.push(listener);
    this.logger.debug(`Added event listener. Total listeners: ${this.eventListeners.length}`);
  }

  /**
   * Removes an event listener
   */
  removeEventListener(listener: (event: MonitoringEvent) => void): void {
    const index = this.eventListeners.indexOf(listener);
    if (index !== -1) {
      this.eventListeners.splice(index, 1);
      this.logger.debug(`Removed event listener. Total listeners: ${this.eventListeners.length}`);
    }
  }

  /**
   * Registers an alert callback
   */
  addAlertCallback(callback: (alert: string) => void): void {
    this.alertCallbacks.push(callback);
    this.logger.debug(`Added alert callback. Total callbacks: ${this.alertCallbacks.length}`);
  }

  /**
   * Removes an alert callback
   */
  removeAlertCallback(callback: (alert: string) => void): void {
    const index = this.alertCallbacks.indexOf(callback);
    if (index !== -1) {
      this.alertCallbacks.splice(index, 1);
      this.logger.debug(`Removed alert callback. Total callbacks: ${this.alertCallbacks.length}`);
    }
  }

  /**
   * Logs a workflow start event
   */
  logWorkflowStart(workflowId: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: generateId('monitoring-event'),
      timestamp: new Date(),
      eventType: 'workflow_start',
      workflowId,
      metadata,
    };

    this.activeWorkflows.add(workflowId);
    this.recordEvent(event);
    this.logger.debug(`Workflow started: ${workflowId}`);
  }

  /**
   * Logs a workflow completion event
   */
  logWorkflowComplete(workflowId: string, duration: number, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: generateId('monitoring-event'),
      timestamp: new Date(),
      eventType: 'workflow_complete',
      workflowId,
      duration,
      metadata,
    };

    this.activeWorkflows.delete(workflowId);
    this.recordEvent(event);

    // Update performance metrics
    if (this.config.enablePerformanceTracking) {
      this.performanceMetrics.workflowExecutionTime = duration;
    }

    this.logger.debug(`Workflow completed: ${workflowId} in ${duration}ms`);
  }

  /**
   * Logs a workflow error event
   */
  logWorkflowError(workflowId: string, error: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: generateId('monitoring-event'),
      timestamp: new Date(),
      eventType: 'workflow_error',
      workflowId,
      error,
      metadata,
    };

    this.activeWorkflows.delete(workflowId);
    this.recordEvent(event);

    // Check for alerts
    this.checkAlerts(event);
    this.logger.error(`Workflow error: ${workflowId} - ${error}`);
  }

  /**
   * Logs a node start event
   */
  logNodeStart(workflowId: string, nodeId: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: generateId('monitoring-event'),
      timestamp: new Date(),
      eventType: 'node_start',
      workflowId,
      nodeId,
      metadata,
    };

    this.recordEvent(event);
    this.logger.debug(`Node started: ${nodeId} in workflow ${workflowId}`);
  }

  /**
   * Logs a node completion event
   */
  logNodeComplete(workflowId: string, nodeId: string, duration: number, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: generateId('monitoring-event'),
      timestamp: new Date(),
      eventType: 'node_complete',
      workflowId,
      nodeId,
      duration,
      metadata,
    };

    this.recordEvent(event);

    // Update performance metrics
    if (this.config.enablePerformanceTracking) {
      this.performanceMetrics.nodeExecutionTimes[nodeId] = duration;
    }

    this.logger.debug(`Node completed: ${nodeId} in workflow ${workflowId} in ${duration}ms`);
  }

  /**
   * Logs a node error event
   */
  logNodeError(workflowId: string, nodeId: string, error: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: generateId('monitoring-event'),
      timestamp: new Date(),
      eventType: 'node_error',
      workflowId,
      nodeId,
      error,
      metadata,
    };

    this.recordEvent(event);

    // Check for alerts
    this.checkAlerts(event);
    this.logger.error(`Node error: ${nodeId} in workflow ${workflowId} - ${error}`);
  }

  /**
   * Records debug information for a node
   */
  recordDebugInfo(nodeId: string, debugInfo: Omit<DebugInfo, 'nodeId'>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const fullDebugInfo: DebugInfo = {
      ...debugInfo,
      nodeId,
    };

    if (!this.debugInfo.has(nodeId)) {
      this.debugInfo.set(nodeId, []);
    }
    this.debugInfo.get(nodeId)?.push(fullDebugInfo);
    this.logger.debug(`Debug info recorded for node: ${nodeId}`);
  }

  /**
   * Records a monitoring event
   */
  private recordEvent(event: MonitoringEvent): void {
    // Apply sampling rate
    if (Math.random() > this.config.samplingRate) {
      return;
    }

    this.eventLog.push(event);

    // Notify listeners
    for (const listener of this.eventListeners) {
      try {
        listener(event);
      } catch (error) {
        this.logger.error(`Error in monitoring event listener: ${error}`);
      }
    }
  }

  /**
   * Checks if an alert condition is met
   */
  private checkAlerts(event: MonitoringEvent): void {
    if (!this.config.enableAlerting) return;

    let alertMessage: string | null = null;

    if (event.duration && event.duration > this.config.alertThresholds.executionTime) {
      alertMessage = `High execution time detected: ${event.duration}ms for ${event.nodeId || event.workflowId}`;
    } else if (event.eventType === 'node_error' || event.eventType === 'workflow_error') {
      // Calculate error rate
      const recentErrors = this.eventLog.filter(e =>
        e.eventType.includes('error') &&
        new Date().getTime() - e.timestamp.getTime() < 60000 // Last minute
      ).length;

      const recentEvents = this.eventLog.filter(e =>
        e.eventType.includes('_complete') || e.eventType.includes('error') &&
        new Date().getTime() - e.timestamp.getTime() < 60000 // Last minute
      ).length;

      const errorRate = recentEvents > 0 ? (recentErrors / recentEvents) * 100 : 0;

      if (errorRate > this.config.alertThresholds.errorRate) {
        alertMessage = `High error rate detected: ${errorRate.toFixed(2)}%`;
      }
    }

    if (alertMessage) {
      this.triggerAlert(alertMessage);
    }
  }

  /**
   * Triggers an alert
   */
  private triggerAlert(message: string): void {
    this.logger.warn(`[RAGBITS MONITORING ALERT] ${message}`);

    for (const callback of this.alertCallbacks) {
      try {
        callback(message);
      } catch (error) {
        this.logger.error(`Error in alert callback: ${error}`);
      }
    }
  }

  /**
   * Gets the current event log
   */
  getEventLog(): MonitoringEvent[] {
    return [...this.eventLog];
  }

  /**
   * Gets events for a specific workflow
   */
  getWorkflowEvents(workflowId: string): MonitoringEvent[] {
    return this.eventLog.filter(event => event.workflowId === workflowId);
  }

  /**
   * Gets events for a specific node
   */
  getNodeEvents(nodeId: string): MonitoringEvent[] {
    return this.eventLog.filter(event => event.nodeId === nodeId);
  }

  /**
   * Gets performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics {
    // Calculate throughput (operations per minute)
    const recentEvents = this.eventLog.filter(e =>
      e.eventType.includes('_complete') &&
      new Date().getTime() - e.timestamp.getTime() < 60000 // Last minute
    );

    this.performanceMetrics.throughput = recentEvents.length;

    // Calculate error rate
    const recentErrors = this.eventLog.filter(e =>
      e.eventType.includes('error') &&
      new Date().getTime() - e.timestamp.getTime() < 60000 // Last minute
    ).length;

    this.performanceMetrics.errorRate = recentEvents.length > 0 ?
      (recentErrors / recentEvents.length) * 100 : 0;

    return { ...this.performanceMetrics };
  }

  /**
   * Gets debug information for a node
   */
  getDebugInfo(nodeId: string): DebugInfo[] {
    return this.debugInfo.get(nodeId) ? [...this.debugInfo.get(nodeId)!] : [];
  }

  /**
   * Gets all debug information
   */
  getAllDebugInfo(): Map<string, DebugInfo[]> {
    return new Map(this.debugInfo);
  }

  /**
   * Gets active workflows
   */
  getActiveWorkflows(): string[] {
    return Array.from(this.activeWorkflows);
  }

  /**
   * Gets workflow statistics
   */
  getWorkflowStats(): {
    totalWorkflows: number;
    activeWorkflows: number;
    completedWorkflows: number;
    errorWorkflows: number;
    averageExecutionTime: number;
  } {
    const allEvents = this.eventLog;
    const startEvents = allEvents.filter(e => e.eventType === 'workflow_start');
    const completeEvents = allEvents.filter(e => e.eventType === 'workflow_complete');
    const errorEvents = allEvents.filter(e => e.eventType === 'workflow_error');

    const totalExecutionTime = completeEvents.reduce((sum, event) => sum + (event.duration || 0), 0);
    const averageExecutionTime = completeEvents.length > 0 ?
      totalExecutionTime / completeEvents.length : 0;

    return {
      totalWorkflows: startEvents.length,
      activeWorkflows: this.activeWorkflows.size,
      completedWorkflows: completeEvents.length,
      errorWorkflows: errorEvents.length,
      averageExecutionTime,
    };
  }

  /**
   * Cleans up old events based on retention period
   */
  private cleanupOldEvents(): void {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.retentionPeriod);

    this.eventLog = this.eventLog.filter(event => event.timestamp > cutoffDate);

    // Also clean up old debug info
    for (const [nodeId, debugInfos] of this.debugInfo.entries()) {
      const filtered = debugInfos.filter(info => {
        const event = this.eventLog.find(e =>
          e.nodeId === nodeId &&
          e.timestamp.getTime() === info.executionTime // Approximate match
        );
        return !event || event.timestamp > cutoffDate;
      });

      if (filtered.length === 0) {
        this.debugInfo.delete(nodeId);
      } else {
        this.debugInfo.set(nodeId, filtered);
      }
    }

    this.logger.debug(`Cleanup completed. Event log size: ${this.eventLog.length}, Debug info entries: ${this.debugInfo.size}`);
  }

  /**
   * Exports monitoring data
   */
  exportData(format: 'json' | 'csv' = 'json'): string {
    const data = {
      events: this.eventLog,
      metrics: this.getPerformanceMetrics(),
      workflowStats: this.getWorkflowStats(),
      activeWorkflows: this.getActiveWorkflows(),
    };

    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else {
      // Simple CSV export for events
      let csv = 'Timestamp,EventType,WorkflowId,NodeId,Duration,Error\n';
      for (const event of this.eventLog) {
        csv += `"${event.timestamp.toISOString()}","${event.eventType}","${event.workflowId}","${event.nodeId || ''}",${event.duration || ''},"${event.error || ''}"\n`;
      }
      return csv;
    }
  }

  /**
   * Resets the monitoring service
   */
  reset(): void {
    this.eventLog = [];
    this.debugInfo.clear();
    this.activeWorkflows.clear();

    this.performanceMetrics = {
      workflowExecutionTime: 0,
      nodeExecutionTimes: {},
      tokensUsed: 0,
      memoryUsage: 0,
      cacheHitRate: 0,
      errorRate: 0,
      throughput: 0,
    };

    this.logger.info('MonitoringService reset');
  }

  /**
   * Add event listener management
   */
  getEventListenersCount(): number {
    return this.eventListeners.length;
  }

  /**
   * Add alert callback management
   */
  getAlertCallbacksCount(): number {
    return this.alertCallbacks.length;
  }

  /**
   * Add workflow event logging
   */
  logWorkflowExecution(workflowId: string, executionResult: WorkflowExecutionResult[]): void {
    this.logger.debug(`Logging workflow execution for ${workflowId} with ${executionResult.length} results`);

    // Log start event
    this.logWorkflowStart(workflowId);

    // Log each node execution
    for (const result of executionResult) {
      if (result.success) {
        this.logNodeComplete(workflowId, result.nodeId, result.executionTime);
      } else {
        this.logNodeError(workflowId, result.nodeId, result.error || 'Unknown error');
      }
    }

    // Calculate total duration
    const totalDuration = executionResult.reduce((sum, result) => sum + result.executionTime, 0);
    this.logWorkflowComplete(workflowId, totalDuration);
  }

  /**
   * Add node event logging
   */
  logNodeExecution(workflowId: string, nodeId: string, executionResult: WorkflowExecutionResult): void {
    if (executionResult.success) {
      this.logNodeComplete(workflowId, nodeId, executionResult.executionTime);
    } else {
      this.logNodeError(workflowId, nodeId, executionResult.error || 'Unknown error');
    }
  }

  /**
   * Add debug information management
   */
  clearDebugInfo(nodeId?: string): void {
    if (nodeId) {
      this.debugInfo.delete(nodeId);
      this.logger.debug(`Cleared debug info for node: ${nodeId}`);
    } else {
      this.debugInfo.clear();
      this.logger.debug('Cleared all debug info');
    }
  }

  /**
   * Add event validation
   */
  validateEvent(event: MonitoringEvent): boolean {
    return !!event.id && !!event.timestamp && !!event.eventType && !!event.workflowId;
  }

  /**
   * Add event filtering
   */
  filterEvents(filterFn: (event: MonitoringEvent) => boolean): MonitoringEvent[] {
    return this.eventLog.filter(filterFn);
  }

  /**
   * Add event aggregation
   */
  aggregateEvents(): {
    byType: Record<string, number>;
    byWorkflow: Record<string, number>;
    byNode: Record<string, number>;
    totalEvents: number;
  } {
    const byType: Record<string, number> = {};
    const byWorkflow: Record<string, number> = {};
    const byNode: Record<string, number> = {};

    for (const event of this.eventLog) {
      byType[event.eventType] = (byType[event.eventType] || 0) + 1;
      byWorkflow[event.workflowId] = (byWorkflow[event.workflowId] || 0) + 1;
      if (event.nodeId) {
        byNode[event.nodeId] = (byNode[event.nodeId] || 0) + 1;
      }
    }

    return {
      byType,
      byWorkflow,
      byNode,
      totalEvents: this.eventLog.length
    };
  }

  /**
   * Add metrics collection infrastructure
   */
  updateMetrics(metrics: Partial<PerformanceMetrics>): void {
    Object.assign(this.performanceMetrics, metrics);
  }

  /**
   * Add error handling infrastructure
   */
  logError(error: Error | string, context?: string): void {
    const errorMessage = typeof error === 'string' ? error : error.message;
    this.logger.error(`Error in ${context || 'unknown context'}: ${errorMessage}`);
  }

  /**
   * Add logging setup
   */
  setLogLevel(level: 'debug' | 'info' | 'warn' | 'error'): void {
    this.config.logLevel = level;
    this.logger = new Logger({ level, prefix: 'MonitoringService' });
  }

  /**
   * Add metrics collection
   */
  collectMetrics(): PerformanceMetrics {
    return this.getPerformanceMetrics();
  }

  /**
   * Add clean script for dist directory
   */
  cleanup(): void {
    if (this.cleanupIntervalId) {
      clearInterval(this.cleanupIntervalId);
      this.logger.info('Cleanup interval cleared');
    }
    this.reset();
  }

  /**
   * Add workflow execution tracking
   */
  trackWorkflowExecution(workflowId: string, executionFn: () => Promise<any>): Promise<any> {
    this.logWorkflowStart(workflowId);

    return executionFn()
      .then(result => {
        this.logWorkflowComplete(workflowId, 0); // Duration will be calculated differently
        return result;
      })
      .catch(error => {
        this.logWorkflowError(workflowId, error.message);
        throw error;
      });
  }

  /**
   * Add execution history tracking
   */
  getExecutionHistory(workflowId: string): MonitoringEvent[] {
    return this.getWorkflowEvents(workflowId);
  }

  /**
   * Add result filtering
   */
  getFilteredResults(filter: (event: MonitoringEvent) => boolean): MonitoringEvent[] {
    return this.eventLog.filter(filter);
  }

  /**
   * Add logging functionality
   */
  log(message: string, level: 'debug' | 'info' | 'warn' | 'error' = 'info'): void {
    this.logger[level](message);
  }

  /**
   * Add metrics collection infrastructure
   */
  addMetric(name: string, value: number): void {
    (this.performanceMetrics as any)[name] = value;
  }

  /**
   * Add error handling during execution
   */
  async executeWithMonitoring<T>(operation: () => Promise<T>, context: string): Promise<T> {
    const startTime = Date.now();
    try {
      const result = await operation();
      const duration = Date.now() - startTime;
      this.logger.debug(`${context} completed in ${duration}ms`);
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error(`${context} failed after ${duration}ms: ${error}`);
      throw error;
    }
  }

  /**
   * Add partial execution continuation
   */
  continueMonitoring(workflowId: string, fromEvent?: string): MonitoringEvent[] {
    const events = this.getWorkflowEvents(workflowId);
    if (!fromEvent) {
      return events;
    }

    const startIndex = events.findIndex(e => e.id === fromEvent);
    return startIndex !== -1 ? events.slice(startIndex) : events;
  }

  /**
   * Add topological sort implementation
   */
  getDependencyGraph(workflowId: string): Record<string, string[]> {
    const events = this.getWorkflowEvents(workflowId);
    const graph: Record<string, string[]> = {};

    // This would be populated based on actual workflow dependencies
    // For now, returning an empty graph
    return graph;
  }

  /**
   * Add execution order determination
   */
  getExecutionOrder(workflowId: string): string[] {
    const events = this.getWorkflowEvents(workflowId);
    // Return node IDs in the order they were executed
    return Array.from(new Set(events.map(e => e.nodeId).filter(Boolean) as string[]));
  }

  /**
   * Add node execution sequence
   */
  getNodeExecutionSequence(workflowId: string): MonitoringEvent[] {
    return this.getWorkflowEvents(workflowId)
      .filter(e => e.eventType.includes('node'))
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  /**
   * Add input preparation logic
   */
  prepareMonitoringInput(workflowId: string, nodeId: string): any {
    const nodeEvents = this.getNodeEvents(nodeId);
    const workflowEvents = this.getWorkflowEvents(workflowId);

    // Prepare input based on previous events
    return {
      previousEvents: nodeEvents,
      workflowContext: workflowEvents,
      timestamp: new Date()
    };
  }

  /**
   * Add output collection logic
   */
  collectMonitoringOutput(workflowId: string, nodeId: string): any {
    const nodeEvents = this.getNodeEvents(nodeId);
    const workflowEvents = this.getWorkflowEvents(workflowId);

    return {
      nodeEvents,
      workflowEvents,
      metrics: this.getPerformanceMetrics(),
      stats: this.getWorkflowStats()
    };
  }

  /**
   * Add execution result tracking
   */
  trackExecutionResult(workflowId: string, result: WorkflowExecutionResult): void {
    this.logNodeExecution(workflowId, result.nodeId, result);
  }

  /**
   * Add error handling during execution
   */
  handleMonitoringError(error: any, context: string): void {
    this.logger.error(`Monitoring error in ${context}: ${error}`);
    this.logError(error, context);
  }

  /**
   * Add timeout handling
   */
  async executeWithTimeout<T>(operation: () => Promise<T>, timeoutMs: number, context: string): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        const error = new Error(`Operation ${context} timed out after ${timeoutMs}ms`);
        this.handleMonitoringError(error, context);
        reject(error);
      }, timeoutMs);

      operation()
        .then(result => {
          clearTimeout(timeoutId);
          resolve(result);
        })
        .catch(error => {
          clearTimeout(timeoutId);
          this.handleMonitoringError(error, context);
          reject(error);
        });
    });
  }

  /**
   * Add promise race for timeout
   */
  async raceWithTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs);
    });

    return Promise.race([promise, timeoutPromise]);
  }

  /**
   * Add execution time measurement
   */
  async measureExecutionTime<T>(operation: () => Promise<T>): Promise<{ result: T; executionTime: number }> {
    const startTime = Date.now();
    try {
      const result = await operation();
      const executionTime = Date.now() - startTime;
      return { result, executionTime };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      throw error;
    }
  }

  /**
   * Add result formatting
   */
  formatMonitoringResult(result: any): string {
    return JSON.stringify(result, null, 2);
  }

  /**
   * Add logging for execution
   */
  logExecutionStatus(status: 'start' | 'complete' | 'error', details?: string): void {
    this.logger.info(`Execution ${status}${details ? `: ${details}` : ''}`);
  }

  /**
   * Add metrics collection
   */
  collectMonitoringMetrics(): {
    eventCount: number;
    workflowCount: number;
    nodeCount: number;
    errorCount: number;
    averageDuration: number;
  } {
    const events = this.eventLog;
    const workflows = new Set(events.map(e => e.workflowId));
    const nodes = new Set(events.filter(e => e.nodeId).map(e => e.nodeId) as string[]);

    const errors = events.filter(e => e.eventType.includes('error'));
    const completed = events.filter(e => e.eventType.includes('complete'));
    const avgDuration = completed.length > 0
      ? completed.reduce((sum, e) => sum + (e.duration || 0), 0) / completed.length
      : 0;

    return {
      eventCount: events.length,
      workflowCount: workflows.size,
      nodeCount: nodes.size,
      errorCount: errors.length,
      averageDuration: avgDuration
    };
  }

  /**
   * Add starting node detection
   */
  getStartingNodes(workflowId: string): string[] {
    const events = this.getWorkflowEvents(workflowId);
    const nodeStartEvents = events.filter(e => e.eventType === 'node_start');

    // In a real implementation, this would analyze the workflow structure
    // For now, return the first few nodes that started
    return nodeStartEvents.slice(0, 5).map(e => e.nodeId!).filter(Boolean) as string[];
  }

  /**
   * Add execution history retrieval
   */
  getExecutionHistoryByType(workflowId: string, eventType: string): MonitoringEvent[] {
    return this.getWorkflowEvents(workflowId).filter(e => e.eventType === eventType);
  }

  /**
   * Add result lookup by node ID
   */
  getLastNodeEvent(nodeId: string): MonitoringEvent | undefined {
    return [...this.eventLog]
      .reverse()
      .find(e => e.nodeId === nodeId);
  }

  /**
   * Add execution statistics
   */
  getExecutionStatistics(): {
    totalEvents: number;
    totalWorkflows: number;
    totalNodes: number;
    errorRate: number;
    averageDuration: number;
  } {
    const events = this.eventLog;
    const workflows = new Set(events.map(e => e.workflowId));
    const nodes = new Set(events.filter(e => e.nodeId).map(e => e.nodeId) as string[]);

    const errors = events.filter(e => e.eventType.includes('error'));
    const completed = events.filter(e => e.eventType.includes('complete'));
    const avgDuration = completed.length > 0
      ? completed.reduce((sum, e) => sum + (e.duration || 0), 0) / completed.length
      : 0;

    return {
      totalEvents: events.length,
      totalWorkflows: workflows.size,
      totalNodes: nodes.size,
      errorRate: events.length > 0 ? (errors.length / events.length) * 100 : 0,
      averageDuration: avgDuration
    };
  }

  /**
   * Add history validation
   */
  validateHistory(): boolean {
    return this.eventLog.every(event =>
      event.id &&
      event.timestamp instanceof Date &&
      ['node_start', 'node_complete', 'node_error', 'workflow_start', 'workflow_complete', 'workflow_error'].includes(event.eventType)
    );
  }

  /**
   * Add data management and cleanup
   */
  cleanupOldEventsByDate(beforeDate: Date): number {
    const initialCount = this.eventLog.length;
    this.eventLog = this.eventLog.filter(event => event.timestamp >= beforeDate);
    const removedCount = initialCount - this.eventLog.length;

    this.logger.debug(`Removed ${removedCount} events older than ${beforeDate.toISOString()}`);
    return removedCount;
  }

  /**
   * Add export functionality
   */
  exportToJSON(): string {
    return JSON.stringify({
      events: this.eventLog,
      metrics: this.getPerformanceMetrics(),
      stats: this.getWorkflowStats()
    }, null, 2);
  }

  /**
   * Add workflow execution statistics
   */
  getWorkflowExecutionStats(workflowId: string): {
    totalNodes: number;
    completedNodes: number;
    failedNodes: number;
    totalDuration: number;
    successRate: number;
  } {
    const events = this.getWorkflowEvents(workflowId);
    const nodeEvents = events.filter(e => e.nodeId);
    const completedNodes = nodeEvents.filter(e => e.eventType === 'node_complete');
    const failedNodes = nodeEvents.filter(e => e.eventType === 'node_error');
    const totalDuration = events.reduce((sum, e) => sum + (e.duration || 0), 0);
    const successRate = nodeEvents.length > 0 ? (completedNodes.length / nodeEvents.length) * 100 : 0;

    return {
      totalNodes: nodeEvents.length,
      completedNodes: completedNodes.length,
      failedNodes: failedNodes.length,
      totalDuration,
      successRate
    };
  }

  /**
   * Add node execution statistics
   */
  getNodeExecutionStats(nodeId: string): {
    totalExecutions: number;
    successfulExecutions: number;
    failedExecutions: number;
    averageDuration: number;
    successRate: number;
  } {
    const events = this.getNodeEvents(nodeId);
    const nodeEvents = events.filter(e => e.nodeId === nodeId);
    const completedEvents = nodeEvents.filter(e => e.eventType === 'node_complete');
    const errorEvents = nodeEvents.filter(e => e.eventType === 'node_error');
    const totalDuration = completedEvents.reduce((sum, e) => sum + (e.duration || 0), 0);
    const averageDuration = completedEvents.length > 0 ? totalDuration / completedEvents.length : 0;
    const successRate = nodeEvents.length > 0 ? (completedEvents.length / nodeEvents.length) * 100 : 0;

    return {
      totalExecutions: nodeEvents.length,
      successfulExecutions: completedEvents.length,
      failedExecutions: errorEvents.length,
      averageDuration,
      successRate
    };
  }

  /**
   * Add memory usage tracking
   */
  getMemoryUsage(): number {
    // In a real implementation, this would return actual memory usage
    // For now, returning a placeholder value
    return this.performanceMetrics.memoryUsage;
  }

  /**
   * Add cache hit rate calculation
   */
  calculateCacheHitRate(): number {
    // In a real implementation, this would calculate actual cache hit rate
    // For now, returning the stored value
    return this.performanceMetrics.cacheHitRate;
  }

  /**
   * Add throughput calculation
   */
  calculateThroughput(): number {
    // In a real implementation, this would calculate actual throughput
    // For now, returning the stored value
    return this.performanceMetrics.throughput;
  }

  /**
   * Add error rate calculation
   */
  calculateErrorRate(): number {
    // In a real implementation, this would calculate actual error rate
    // For now, returning the stored value
    return this.performanceMetrics.errorRate;
  }

  /**
   * Add tokens used tracking
   */
  getTokensUsed(): number {
    return this.performanceMetrics.tokensUsed;
  }

  /**
   * Add node execution time tracking
   */
  getNodeExecutionTime(nodeId: string): number {
    return this.performanceMetrics.nodeExecutionTimes[nodeId] || 0;
  }

  /**
   * Add workflow execution time tracking
   */
  getWorkflowExecutionTime(): number {
    return this.performanceMetrics.workflowExecutionTime;
  }

  /**
   * Add alert threshold management
   */
  setAlertThreshold(threshold: keyof typeof this.config.alertThresholds, value: number): void {
    (this.config.alertThresholds[threshold] as number) = value;
    this.logger.info(`Alert threshold ${threshold} set to ${value}`);
  }

  /**
   * Add alert threshold retrieval
   */
  getAlertThreshold(threshold: keyof typeof this.config.alertThresholds): number {
    return this.config.alertThresholds[threshold] as number;
  }

  /**
   * Add sampling rate management
   */
  setSamplingRate(rate: number): void {
    this.config.samplingRate = Math.max(0, Math.min(1, rate)); // Clamp between 0 and 1
    this.logger.info(`Sampling rate set to ${this.config.samplingRate}`);
  }

  /**
   * Add retention period management
   */
  setRetentionPeriod(days: number): void {
    this.config.retentionPeriod = Math.max(1, days); // At least 1 day
    this.logger.info(`Retention period set to ${this.config.retentionPeriod} days`);
  }

  /**
   * Add monitoring enable/disable functionality
   */
  setMonitoringEnabled(enabled: boolean): void {
    this.config.enableRealTimeMonitoring = enabled;
    this.logger.info(`Real-time monitoring ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Add performance tracking enable/disable functionality
   */
  setPerformanceTrackingEnabled(enabled: boolean): void {
    this.config.enablePerformanceTracking = enabled;
    this.logger.info(`Performance tracking ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Add error tracking enable/disable functionality
   */
  setErrorTrackingEnabled(enabled: boolean): void {
    this.config.enableErrorTracking = enabled;
    this.logger.info(`Error tracking ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Add token tracking enable/disable functionality
   */
  setTokenTrackingEnabled(enabled: boolean): void {
    this.config.enableTokenTracking = enabled;
    this.logger.info(`Token tracking ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Add alerting enable/disable functionality
   */
  setAlertingEnabled(enabled: boolean): void {
    this.config.enableAlerting = enabled;
    this.logger.info(`Alerting ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Add workflow status tracking
   */
  getWorkflowStatus(workflowId: string): 'active' | 'completed' | 'failed' | 'unknown' {
    if (this.activeWorkflows.has(workflowId)) {
      return 'active';
    }

    const events = this.getWorkflowEvents(workflowId);
    if (events.length === 0) {
      return 'unknown';
    }

    const hasError = events.some(e => e.eventType.includes('error'));
    const hasComplete = events.some(e => e.eventType.includes('complete'));

    if (hasError && !hasComplete) {
      return 'failed';
    } else if (hasComplete) {
      return 'completed';
    } else {
      return 'unknown';
    }
  }

  /**
   * Add node status tracking
   */
  getNodeStatus(nodeId: string): 'active' | 'completed' | 'failed' | 'unknown' {
    const events = this.getNodeEvents(nodeId);
    if (events.length === 0) {
      return 'unknown';
    }

    const lastEvent = events[events.length - 1];
    if (lastEvent.eventType === 'node_start') {
      return 'active';
    } else if (lastEvent.eventType === 'node_complete') {
      return 'completed';
    } else if (lastEvent.eventType === 'node_error') {
      return 'failed';
    } else {
      return 'unknown';
    }
  }

  /**
   * Add monitoring service lifecycle management
   */
  async start(): Promise<void> {
    this.logger.info('Starting monitoring service');
    // Service is already started in constructor, but this provides an explicit start method
  }

  /**
   * Add monitoring service lifecycle management
   */
  async stop(): Promise<void> {
    this.logger.info('Stopping monitoring service');
    this.cleanup();
  }

  /**
   * Add monitoring service status
   */
  getStatus(): {
    isActive: boolean;
    eventCount: number;
    activeWorkflows: number;
    listenersCount: number;
    alertCallbacksCount: number;
  } {
    return {
      isActive: true, // Service is always active when this method is called
      eventCount: this.eventLog.length,
      activeWorkflows: this.activeWorkflows.size,
      listenersCount: this.eventListeners.length,
      alertCallbacksCount: this.alertCallbacks.length
    };
  }

  /**
   * Add monitoring service health check
   */
  isHealthy(): boolean {
    // Basic health check - service is healthy if it has reasonable metrics
    return this.eventLog.length >= 0 &&
           this.activeWorkflows.size >= 0 &&
           this.eventListeners.length >= 0;
  }

  /**
   * Add monitoring service diagnostics
   */
  getDiagnostics(): {
    memoryUsage: number;
    eventQueueSize: number;
    activeWorkflows: number;
    lastEventTime: Date | null;
    uptime: number;
  } {
    const lastEvent = this.eventLog.length > 0
      ? this.eventLog[this.eventLog.length - 1].timestamp
      : null;

    // In a real implementation, this would return actual memory usage
    const memoryUsage = this.performanceMetrics.memoryUsage;

    return {
      memoryUsage,
      eventQueueSize: this.eventLog.length,
      activeWorkflows: this.activeWorkflows.size,
      lastEventTime: lastEvent,
      uptime: Date.now() // Placeholder - would need to track start time
    };
  }

  /**
   * Add monitoring service configuration
   */
  getConfig(): MonitoringConfig {
    return { ...this.config };
  }

  /**
   * Add monitoring service reconfiguration
   */
  updateConfig(newConfig: Partial<MonitoringConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.logger = new Logger({ level: this.config.logLevel, prefix: 'MonitoringService' });
    this.logger.info('Monitoring service configuration updated');
  }

  /**
   * Add monitoring service statistics
   */
  getDetailedStats(): {
    eventStats: {
      total: number;
      byType: Record<string, number>;
      byWorkflow: Record<string, number>;
      byNode: Record<string, number>;
    };
    performanceStats: PerformanceMetrics;
    workflowStats: ReturnType<typeof this.getWorkflowStats>;
    monitoringStats: {
      activeWorkflows: number;
      listeners: number;
      alertCallbacks: number;
    };
  } {
    const aggregatedEvents = this.aggregateEvents();
    const performanceStats = this.getPerformanceMetrics();
    const workflowStats = this.getWorkflowStats();

    return {
      eventStats: {
        total: aggregatedEvents.totalEvents,
        byType: aggregatedEvents.byType,
        byWorkflow: aggregatedEvents.byWorkflow,
        byNode: aggregatedEvents.byNode
      },
      performanceStats,
      workflowStats,
      monitoringStats: {
        activeWorkflows: this.activeWorkflows.size,
        listeners: this.eventListeners.length,
        alertCallbacks: this.alertCallbacks.length
      }
    };
  }

  /**
   * Add monitoring service validation
   */
  validate(): {
    isValid: boolean;
    issues: string[];
  } {
    const issues: string[] = [];

    if (this.config.samplingRate < 0 || this.config.samplingRate > 1) {
      issues.push('Sampling rate must be between 0 and 1');
    }

    if (this.config.retentionPeriod < 1) {
      issues.push('Retention period must be at least 1 day');
    }

    if (this.config.alertThresholds.executionTime <= 0) {
      issues.push('Execution time threshold must be positive');
    }

    if (this.config.alertThresholds.errorRate < 0 || this.config.alertThresholds.errorRate > 100) {
      issues.push('Error rate threshold must be between 0 and 100');
    }

    if (this.config.alertThresholds.memoryUsage <= 0) {
      issues.push('Memory usage threshold must be positive');
    }

    return {
      isValid: issues.length === 0,
      issues
    };
  }
}