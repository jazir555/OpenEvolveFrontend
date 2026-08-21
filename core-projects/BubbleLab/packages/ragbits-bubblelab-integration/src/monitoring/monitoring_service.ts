/**
 * Monitoring Service
 * Provides monitoring, debugging, and observability features for RAG workflows
 */

import {
  MonitoringEvent,
  PerformanceMetrics,
  DebugInfo,
  MonitoringConfig,
  WorkflowExecutionResult
} from '../types';

/**
 * MonitoringService - Comprehensive monitoring and debugging service
 * Tracks workflow execution, performance metrics, and provides debugging tools
 */
export class MonitoringService {
  private config: MonitoringConfig;
  private eventLog: MonitoringEvent[];
  private performanceMetrics: PerformanceMetrics;
  private debugInfo: Map<string, DebugInfo[]>;
  private activeWorkflows: Set<string>;
  private eventListeners: Array<(event: MonitoringEvent) => void>;
  private alertCallbacks: Array<(alert: { message: string; severity: string; data?: any }) => void>;
  private cleanupInterval: NodeJS.Timeout | null;
  private logger: Console;
  
  /**
   * Constructor
   * @param config - Monitoring configuration
   */
  constructor(config: Partial<MonitoringConfig> = {}) {
    this.config = this.validateConfig(config);
    this.eventLog = [];
    this.debugInfo = new Map();
    this.activeWorkflows = new Set();
    this.eventListeners = [];
    this.alertCallbacks = [];
    this.cleanupInterval = null;
    this.logger = console;
    
    // Initialize performance metrics
    this.performanceMetrics = {
      throughput: 0,
      avgExecutionTime: 0,
      errorRate: 0,
      memoryUsage: 0,
      cpuUsage: 0,
      activeWorkflows: 0,
      queueSize: 0
    };
    
    // Start cleanup interval
    this.startCleanupInterval();
  }
  
  /**
   * Validate and normalize configuration
   * @param config - Partial configuration
   * @returns Validated configuration
   */
  private validateConfig(config: Partial<MonitoringConfig>): MonitoringConfig {
    return {
      enabled: config.enabled !== undefined ? config.enabled : true,
      retentionPeriod: config.retentionPeriod || 86400000, // 24 hours
      maxEvents: config.maxEvents || 10000,
      samplingInterval: config.samplingInterval || 60000, // 1 minute
      alertThresholds: {
        executionTime: config.alertThresholds?.executionTime || 5000, // 5 seconds
        errorRate: config.alertThresholds?.errorRate || 0.1, // 10%
        memoryUsage: config.alertThresholds?.memoryUsage || 1024, // 1GB
        cpuUsage: config.alertThresholds?.cpuUsage || 80 // 80%
      }
    };
  }
  
  /**
   * Start cleanup interval for old events
   */
  private startCleanupInterval(): void {
    if (!this.config.enabled) {
      return;
    }
    
    this.cleanupInterval = setInterval(() => {
      try {
        this.cleanupOldEvents();
      } catch (error) {
        this.log('error', `Cleanup failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }, this.config.samplingInterval);
  }
  
  /**
   * Clean up old events based on retention period
   */
  private cleanupOldEvents(): void {
    if (!this.config.enabled || this.eventLog.length === 0) {
      return;
    }
    
    const now = Date.now();
    const retentionCutoff = now - this.config.retentionPeriod;
    
    // Filter out old events
    this.eventLog = this.eventLog.filter(event => {
      return event.timestamp.getTime() >= retentionCutoff;
    });
    
    // Also cleanup old debug info
    this.debugInfo.forEach((infos, nodeId) => {
      const filteredInfos = infos.filter(info => info.timestamp.getTime() >= retentionCutoff);
      if (filteredInfos.length === 0) {
        this.debugInfo.delete(nodeId);
      } else {
        this.debugInfo.set(nodeId, filteredInfos);
      }
    });
    
    this.log('debug', `Cleaned up events. Remaining: ${this.eventLog.length} events, ${this.debugInfo.size} nodes`);
  }
  
  /**
   * Log workflow start event
   * @param workflowId - Workflow ID
   * @param workflowName - Workflow name
   * @param timestamp - Event timestamp
   */
  public logWorkflowStart(workflowId: string, workflowName: string, timestamp: Date = new Date()): void {
    if (!this.config.enabled) {
      return;
    }
    
    const event: MonitoringEvent = {
      id: `workflow-start-${Date.now()}`,
      type: 'workflow_start',
      timestamp,
      data: {
        workflowId,
        workflowName
      },
      severity: 'info'
    };
    
    this.addEvent(event);
    this.activeWorkflows.add(workflowId);
    this.updatePerformanceMetrics();
  }
  
  /**
   * Log workflow completion event
   * @param workflowId - Workflow ID
   * @param workflowName - Workflow name
   * @param duration - Workflow duration in milliseconds
   * @param status - Completion status
   * @param timestamp - Event timestamp
   */
  public logWorkflowComplete(
    workflowId: string,
    workflowName: string,
    duration: number,
    status: 'success' | 'partial' | 'failed',
    timestamp: Date = new Date()
  ): void {
    if (!this.config.enabled) {
      return;
    }
    
    const event: MonitoringEvent = {
      id: `workflow-complete-${Date.now()}`,
      type: 'workflow_complete',
      timestamp,
      data: {
        workflowId,
        workflowName,
        duration,
        status
      },
      severity: status === 'failed' ? 'error' : 'info'
    };
    
    this.addEvent(event);
    this.activeWorkflows.delete(workflowId);
    this.updatePerformanceMetrics();
    
    // Check alerts
    this.checkAlerts();
  }
  
  /**
   * Log node start event
   * @param workflowId - Workflow ID
   * @param nodeId - Node ID
   * @param nodeType - Node type
   * @param timestamp - Event timestamp
   */
  public logNodeStart(
    workflowId: string,
    nodeId: string,
    nodeType: string,
    timestamp: Date = new Date()
  ): void {
    if (!this.config.enabled) {
      return;
    }
    
    const event: MonitoringEvent = {
      id: `node-start-${Date.now()}`,
      type: 'node_start',
      timestamp,
      data: {
        workflowId,
        nodeId,
        nodeType
      },
      severity: 'info',
      workflowId,
      nodeId
    };
    
    this.addEvent(event);
  }
  
  /**
   * Log node completion event
   * @param workflowId - Workflow ID
   * @param nodeId - Node ID
   * @param nodeType - Node type
   * @param duration - Node execution duration in milliseconds
   * @param timestamp - Event timestamp
   */
  public logNodeComplete(
    workflowId: string,
    nodeId: string,
    nodeType: string,
    duration: number,
    timestamp: Date = new Date()
  ): void {
    if (!this.config.enabled) {
      return;
    }
    
    const event: MonitoringEvent = {
      id: `node-complete-${Date.now()}`,
      type: 'node_complete',
      timestamp,
      data: {
        workflowId,
        nodeId,
        nodeType,
        duration
      },
      severity: 'info',
      workflowId,
      nodeId
    };
    
    this.addEvent(event);
    this.updatePerformanceMetrics();
  }
  
  /**
   * Log node error event
   * @param workflowId - Workflow ID
   * @param nodeId - Node ID
   * @param nodeType - Node type
   * @param error - Error information
   * @param timestamp - Event timestamp
   */
  public logNodeError(
    workflowId: string,
    nodeId: string,
    nodeType: string,
    error: Error | string,
    timestamp: Date = new Date()
  ): void {
    if (!this.config.enabled) {
      return;
    }
    
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : undefined;
    
    const event: MonitoringEvent = {
      id: `node-error-${Date.now()}`,
      type: 'node_error',
      timestamp,
      data: {
        workflowId,
        nodeId,
        nodeType,
        error: errorMessage,
        stack: errorStack
      },
      severity: 'error',
      workflowId,
      nodeId
    };
    
    this.addEvent(event);
    this.updatePerformanceMetrics();
    
    // Check alerts immediately for errors
    this.checkAlerts();
  }
  
  /**
   * Record debug information
   * @param nodeId - Node ID
   * @param debugInfo - Debug information
   */
  public recordDebugInfo(nodeId: string, debugInfo: DebugInfo): void {
    if (!this.config.enabled) {
      return;
    }
    
    if (!this.debugInfo.has(nodeId)) {
      this.debugInfo.set(nodeId, []);
    }
    
    const existingInfos = this.debugInfo.get(nodeId) || [];
    existingInfos.push(debugInfo);
    
    // Limit debug info per node
    if (existingInfos.length > 100) {
      existingInfos.shift(); // Remove oldest
    }
    
    this.debugInfo.set(nodeId, existingInfos);
  }
  
  /**
   * Get debug information for a specific node
   * @param nodeId - Node ID
   * @returns Array of debug information or empty array
   */
  public getDebugInfo(nodeId: string): DebugInfo[] {
    return this.debugInfo.get(nodeId) || [];
  }
  
  /**
   * Get all debug information
   * @returns Map of all debug information by node ID
   */
  public getAllDebugInfo(): Map<string, DebugInfo[]> {
    return new Map(this.debugInfo);
  }
  
  /**
   * Add event listener
   * @param listener - Event listener function
   */
  public addEventListener(listener: (event: MonitoringEvent) => void): void {
    if (this.eventListeners.includes(listener)) {
      return; // Prevent duplicate listeners
    }
    
    this.eventListeners.push(listener);
  }
  
  /**
   * Remove event listener
   * @param listener - Event listener function to remove
   */
  public removeEventListener(listener: (event: MonitoringEvent) => void): void {
    this.eventListeners = this.eventListeners.filter(l => l !== listener);
  }
  
  /**
   * Add alert callback
   * @param callback - Alert callback function
   */
  public addAlertCallback(callback: (alert: { message: string; severity: string; data?: any }) => void): void {
    if (this.alertCallbacks.includes(callback)) {
      return; // Prevent duplicate callbacks
    }
    
    this.alertCallbacks.push(callback);
  }
  
  /**
   * Remove alert callback
   * @param callback - Alert callback function to remove
   */
  public removeAlertCallback(callback: (alert: { message: string; severity: string; data?: any }) => void): void {
    this.alertCallbacks = this.alertCallbacks.filter(c => c !== callback);
  }
  
  /**
   * Add event to log and notify listeners
   * @param event - Monitoring event
   */
  private addEvent(event: MonitoringEvent): void {
    // Enforce max events limit
    if (this.config.maxEvents && this.eventLog.length >= this.config.maxEvents) {
      this.eventLog.shift(); // Remove oldest event
    }
    
    this.eventLog.push(event);
    
    // Notify listeners
    this.eventListeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        this.log('error', `Event listener failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    });
  }
  
  /**
   * Check for alert conditions
   */
  private checkAlerts(): void {
    if (!this.config.enabled || this.alertCallbacks.length === 0) {
      return;
    }
    
    const alerts: Array<{ message: string; severity: string; data?: any }> = [];
    
    // Check execution time alerts
    const slowNodes = this.getSlowNodes();
    if (slowNodes.length > 0) {
      alerts.push({
        message: `Slow node execution detected: ${slowNodes.length} nodes exceeded threshold`,
        severity: 'warning',
        data: { slowNodes }
      });
    }
    
    // Check error rate alerts
    const errorRate = this.getErrorRate();
    if (errorRate > this.config.alertThresholds!.errorRate!) {
      alerts.push({
        message: `High error rate detected: ${(errorRate * 100).toFixed(1)}%`,
        severity: 'error',
        data: { errorRate }
      });
    }
    
    // Check memory usage alerts (simulated)
    if (this.performanceMetrics.memoryUsage > this.config.alertThresholds!.memoryUsage!) {
      alerts.push({
        message: `High memory usage: ${this.performanceMetrics.memoryUsage}MB`,
        severity: 'warning',
        data: { memoryUsage: this.performanceMetrics.memoryUsage }
      });
    }
    
    // Check CPU usage alerts (simulated)
    if (this.performanceMetrics.cpuUsage > this.config.alertThresholds!.cpuUsage!) {
      alerts.push({
        message: `High CPU usage: ${this.performanceMetrics.cpuUsage}%`,
        severity: 'warning',
        data: { cpuUsage: this.performanceMetrics.cpuUsage }
      });
    }
    
    // Trigger alerts
    alerts.forEach(alert => {
      this.triggerAlert(alert);
    });
  }
  
  /**
   * Trigger alert to all callbacks
   * @param alert - Alert information
   */
  private triggerAlert(alert: { message: string; severity: string; data?: any }): void {
    this.alertCallbacks.forEach(callback => {
      try {
        callback(alert);
      } catch (error) {
        this.log('error', `Alert callback failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    });
    
    this.log('warn', `Alert triggered: ${alert.message}`);
  }
  
  /**
   * Get nodes with slow execution times
   * @returns Array of slow node information
   */
  private getSlowNodes(): Array<{ nodeId: string; avgDuration: number; count: number }> {
    const nodeDurations: Record<string, { total: number; count: number }> = {};
    
    // Collect duration data from node_complete events
    this.eventLog.forEach(event => {
      if (event.type === 'node_complete' && event.data?.duration && event.nodeId) {
        const nodeId = event.nodeId;
        if (!nodeDurations[nodeId]) {
          nodeDurations[nodeId] = { total: 0, count: 0 };
        }
        nodeDurations[nodeId].total += event.data.duration;
        nodeDurations[nodeId].count++;
      }
    });
    
    // Calculate averages and filter slow nodes
    const slowNodes: Array<{ nodeId: string; avgDuration: number; count: number }> = [];
    
    Object.entries(nodeDurations).forEach(([nodeId, data]) => {
      const avgDuration = data.total / data.count;
      if (avgDuration > this.config.alertThresholds!.executionTime!) {
        slowNodes.push({
          nodeId,
          avgDuration,
          count: data.count
        });
      }
    });
    
    return slowNodes.sort((a, b) => b.avgDuration - a.avgDuration);
  }
  
  /**
   * Get current error rate
   * @returns Error rate (0-1)
   */
  private getErrorRate(): number {
    const totalEvents = this.eventLog.length;
    if (totalEvents === 0) {
      return 0;
    }
    
    const errorEvents = this.eventLog.filter(event => event.severity === 'error').length;
    return errorEvents / totalEvents;
  }
  
  /**
   * Update performance metrics
   */
  private updatePerformanceMetrics(): void {
    const now = Date.now();
    const recentEvents = this.eventLog.filter(event => {
      return now - event.timestamp.getTime() < this.config.samplingInterval!;
    });
    
    // Calculate throughput (events per second)
    const samplingPeriodSeconds = this.config.samplingInterval! / 1000;
    this.performanceMetrics.throughput = recentEvents.length / samplingPeriodSeconds;
    
    // Calculate average execution time for nodes
    const nodeCompleteEvents = recentEvents.filter(event => event.type === 'node_complete');
    if (nodeCompleteEvents.length > 0) {
      const totalDuration = nodeCompleteEvents.reduce((sum, event) => {
        return sum + (event.data?.duration || 0);
      }, 0);
      this.performanceMetrics.avgExecutionTime = totalDuration / nodeCompleteEvents.length;
    }
    
    // Update error rate
    this.performanceMetrics.errorRate = this.getErrorRate();
    
    // Update active workflows count
    this.performanceMetrics.activeWorkflows = this.activeWorkflows.size;
    
    // Simulate memory and CPU usage
    this.performanceMetrics.memoryUsage = 256 + (Math.random() * 512); // 256-768 MB
    this.performanceMetrics.cpuUsage = 10 + (Math.random() * 60); // 10-70%
    
    // Queue size is simulated based on active workflows
    this.performanceMetrics.queueSize = this.performanceMetrics.activeWorkflows * 2;
  }
  
  /**
   * Get current performance metrics
   * @returns Performance metrics
   */
  public getPerformanceMetrics(): PerformanceMetrics {
    this.updatePerformanceMetrics();
    return { ...this.performanceMetrics };
  }
  
  /**
   * Get workflow statistics
   * @returns Workflow statistics
   */
  public getWorkflowStats(): {
    totalWorkflows: number;
    completedWorkflows: number;
    activeWorkflows: number;
    errorRate: number;
    avgExecutionTime: number;
  } {
    const totalWorkflows = this.eventLog.filter(event => event.type === 'workflow_start').length;
    const completedWorkflows = this.eventLog.filter(event => event.type === 'workflow_complete').length;
    
    return {
      totalWorkflows,
      completedWorkflows,
      activeWorkflows: this.activeWorkflows.size,
      errorRate: this.performanceMetrics.errorRate,
      avgExecutionTime: this.performanceMetrics.avgExecutionTime
    };
  }
  
  /**
   * Get recent events
   * @param limit - Maximum number of events to return
   * @returns Array of recent monitoring events
   */
  public getRecentEvents(limit: number = 100): MonitoringEvent[] {
    return [...this.eventLog].slice(-limit);
  }
  
  /**
   * Get events by type
   * @param type - Event type to filter
   * @param limit - Maximum number of events to return
   * @returns Array of filtered events
   */
  public getEventsByType(type: MonitoringEvent['type'], limit: number = 100): MonitoringEvent[] {
    return this.eventLog
      .filter(event => event.type === type)
      .slice(-limit);
  }
  
  /**
   * Get events by severity
   * @param severity - Severity level to filter
   * @param limit - Maximum number of events to return
   * @returns Array of filtered events
   */
  public getEventsBySeverity(severity: MonitoringEvent['severity'], limit: number = 100): MonitoringEvent[] {
    return this.eventLog
      .filter(event => event.severity === severity)
      .slice(-limit);
  }
  
  /**
   * Export monitoring data
   * @param format - Export format
   * @returns Exported data
   */
  public exportData(format: 'json' | 'csv' = 'json'): any {
    const data = {
      events: this.eventLog,
      debugInfo: Object.fromEntries(this.debugInfo),
      metrics: this.performanceMetrics,
      stats: this.getWorkflowStats(),
      exportedAt: new Date()
    };
    
    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else if (format === 'csv') {
      // Simple CSV export (would need more sophisticated implementation for full data)
      const csvHeader = 'timestamp,type,severity,workflowId,nodeId,message';
      const csvRows = this.eventLog.map(event => {
        const message = typeof event.data === 'object' ? JSON.stringify(event.data) : String(event.data);
        return `
${event.timestamp.toISOString()},${event.type},${event.severity || ''},${event.workflowId || ''},${event.nodeId || ''},"${message.replace(/\"/g, '""')}"`;
      });
      
      return csvHeader + csvRows.join('');
    }
    
    return data;
  }
  
  /**
   * Reset monitoring data
   */
  public reset(): void {
    this.eventLog = [];
    this.debugInfo.clear();
    this.activeWorkflows.clear();
    
    // Reset performance metrics
    this.performanceMetrics = {
      throughput: 0,
      avgExecutionTime: 0,
      errorRate: 0,
      memoryUsage: 0,
      cpuUsage: 0,
      activeWorkflows: 0,
      queueSize: 0
    };
    
    this.log('info', 'Monitoring data reset');
  }
  
  /**
   * Dispose the monitoring service
   */
  public dispose(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    
    this.eventListeners = [];
    this.alertCallbacks = [];
    
    this.log('info', 'Monitoring service disposed');
  }
  
  /**
   * Log a message
   * @param level - Log level
   * @param message - Message to log
   * @param data - Additional data
   */
  private log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: any): void {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [MonitoringService] [${level.toUpperCase()}] ${message}`;
    
    switch (level) {
      case 'debug':
        this.logger.debug(logMessage, data);
        break;
      case 'info':
        this.logger.info(logMessage, data);
        break;
      case 'warn':
        this.logger.warn(logMessage, data);
        break;
      case 'error':
        this.logger.error(logMessage, data);
        break;
    }
  }
  
  /**
   * Get monitoring configuration
   * @returns Monitoring configuration
   */
  public getConfig(): MonitoringConfig {
    return this.config;
  }
  
  /**
   * Set monitoring configuration
   * @param config - New monitoring configuration
   */
  public setConfig(config: Partial<MonitoringConfig>): void {
    this.config = this.validateConfig(config);
    
    // Restart cleanup interval with new settings
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.startCleanupInterval();
    
    this.log('info', 'Monitoring configuration updated');
  }
  
  /**
   * Check if monitoring is enabled
   * @returns True if enabled, false otherwise
   */
  public isEnabled(): boolean {
    return this.config.enabled;
  }
  
  /**
   * Enable monitoring
   */
  public enable(): void {
    this.config.enabled = true;
    this.startCleanupInterval();
    this.log('info', 'Monitoring enabled');
  }
  
  /**
   * Disable monitoring
   */
  public disable(): void {
    this.config.enabled = false;
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.log('info', 'Monitoring disabled');
  }
}