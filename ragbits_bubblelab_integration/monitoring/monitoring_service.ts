/**
 * Monitoring and Debugging Service for Ragbits + BubbleLab Integration
 * 
 * This module provides real-time monitoring, debugging, and analytics for RAG workflows
 */

import { WorkflowExecutionResult } from '../types';
import { MonitoringEvent, PerformanceMetrics, DebugInfo, MonitoringConfig } from '../types';

export class MonitoringService {
  private config: MonitoringConfig;
  private eventLog: MonitoringEvent[];
  private performanceMetrics: PerformanceMetrics;
  private debugInfo: Map<string, DebugInfo[]>;
  private activeWorkflows: Set<string>;
  private eventListeners: Array<(event: MonitoringEvent) => void>;
  private alertCallbacks: Array<(alert: string) => void>;

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

    // Start cleanup interval
    setInterval(() => {
      this.cleanupOldEvents();
    }, 3600000); // Every hour
  }

  /**
   * Registers an event listener
   */
  addEventListener(listener: (event: MonitoringEvent) => void): void {
    this.eventListeners.push(listener);
  }

  /**
   * Removes an event listener
   */
  removeEventListener(listener: (event: MonitoringEvent) => void): void {
    const index = this.eventListeners.indexOf(listener);
    if (index !== -1) {
      this.eventListeners.splice(index, 1);
    }
  }

  /**
   * Registers an alert callback
   */
  addAlertCallback(callback: (alert: string) => void): void {
    this.alertCallbacks.push(callback);
  }

  /**
   * Removes an alert callback
   */
  removeAlertCallback(callback: (alert: string) => void): void {
    const index = this.alertCallbacks.indexOf(callback);
    if (index !== -1) {
      this.alertCallbacks.splice(index, 1);
    }
  }

  /**
   * Logs a workflow start event
   */
  logWorkflowStart(workflowId: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: this.generateId(),
      timestamp: new Date(),
      eventType: 'workflow_start',
      workflowId,
      metadata,
    };

    this.activeWorkflows.add(workflowId);
    this.recordEvent(event);
  }

  /**
   * Logs a workflow completion event
   */
  logWorkflowComplete(workflowId: string, duration: number, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: this.generateId(),
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
  }

  /**
   * Logs a workflow error event
   */
  logWorkflowError(workflowId: string, error: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: this.generateId(),
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
  }

  /**
   * Logs a node start event
   */
  logNodeStart(workflowId: string, nodeId: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: this.generateId(),
      timestamp: new Date(),
      eventType: 'node_start',
      workflowId,
      nodeId,
      metadata,
    };

    this.recordEvent(event);
  }

  /**
   * Logs a node completion event
   */
  logNodeComplete(workflowId: string, nodeId: string, duration: number, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: this.generateId(),
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
  }

  /**
   * Logs a node error event
   */
  logNodeError(workflowId: string, nodeId: string, error: string, metadata?: Record<string, any>): void {
    if (!this.config.enableRealTimeMonitoring) return;

    const event: MonitoringEvent = {
      id: this.generateId(),
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
        console.error('Error in monitoring event listener:', error);
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
    console.warn(`[RAGBITS MONITORING ALERT] ${message}`);
    
    for (const callback of this.alertCallbacks) {
      try {
        callback(message);
      } catch (error) {
        console.error('Error in alert callback:', error);
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
  }

  /**
   * Generates a unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
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
  }
}