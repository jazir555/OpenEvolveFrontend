/**
 * Monitoring and Debug Types for Ragbits + BubbleLab Integration
 */

export interface MonitoringEvent {
  id: string;
  timestamp: Date;
  eventType: 'node_start' | 'node_complete' | 'node_error' | 'workflow_start' | 'workflow_complete' | 'workflow_error';
  workflowId: string;
  nodeId?: string;
  duration?: number;
  metadata?: Record<string, any>;
  error?: string;
}

export interface PerformanceMetrics {
  workflowExecutionTime: number;
  nodeExecutionTimes: Record<string, number>;
  tokensUsed: number;
  memoryUsage: number;
  cacheHitRate: number;
  errorRate: number;
  throughput: number; // operations per minute
}

export interface DebugInfo {
  nodeId: string;
  inputs: any;
  outputs: any;
  executionTime: number;
  logs: string[];
  error?: string;
}

export interface MonitoringConfig {
  enableRealTimeMonitoring: boolean;
  enablePerformanceTracking: boolean;
  enableErrorTracking: boolean;
  enableTokenTracking: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  retentionPeriod: number; // in days
  samplingRate: number; // 0.0 to 1.0
  enableAlerting: boolean;
  alertThresholds: {
    executionTime: number; // in ms
    errorRate: number; // percentage
    memoryUsage: number; // in MB
  };
}

export interface ProcessorIntegrationConfig {
  enableAutoIndexing?: boolean;
  autoIndexInterval?: number; // in seconds
  batchSize?: number;
  enableCaching?: boolean;
  cacheTTL?: number; // in seconds
  enableMonitoring?: boolean;
  maxConcurrentProcesses?: number;
}