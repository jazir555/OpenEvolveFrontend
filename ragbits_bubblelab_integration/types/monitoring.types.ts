/**
 * Monitoring and Debug Types for Ragbits + BubbleLab Integration
 */

/**
 * Event in the monitoring system
 */
export interface MonitoringEvent {
  /** Unique identifier for the event */
  id: string;
  /** Timestamp when the event occurred */
  timestamp: Date;
  /** Type of event that occurred */
  eventType: 'node_start' | 'node_complete' | 'node_error' | 'workflow_start' | 'workflow_complete' | 'workflow_error';
  /** ID of the workflow associated with the event */
  workflowId: string;
  /** ID of the node associated with the event (if applicable) */
  nodeId?: string;
  /** Duration of the operation in milliseconds (if applicable) */
  duration?: number;
  /** Additional metadata associated with the event */
  metadata?: Record<string, any>;
  /** Error message if the event represents an error */
  error?: string;
}

/**
 * Performance metrics collected by the monitoring system
 */
export interface PerformanceMetrics {
  /** Total time for the entire workflow execution in milliseconds */
  workflowExecutionTime: number;
  /** Execution times for individual nodes, keyed by node ID */
  nodeExecutionTimes: Record<string, number>;
  /** Number of tokens processed (for generation operations) */
  tokensUsed: number;
  /** Current memory usage in MB */
  memoryUsage: number;
  /** Percentage of cache hits vs misses */
  cacheHitRate: number;
  /** Error rate as a percentage */
  errorRate: number;
  /** Throughput measured as operations per minute */
  throughput: number;
}

/**
 * Debug information for a specific node
 */
export interface DebugInfo {
  /** ID of the node this debug info is for */
  nodeId: string;
  /** Inputs that were provided to the node */
  inputs: any;
  /** Outputs that were produced by the node */
  outputs: any;
  /** Execution time for the node in milliseconds */
  executionTime: number;
  /** Log messages generated during execution */
  logs: string[];
  /** Error message if the node execution failed */
  error?: string;
}

/**
 * Configuration for the monitoring service
 */
export interface MonitoringConfig {
  /** Whether to enable real-time monitoring of workflow events */
  enableRealTimeMonitoring: boolean;
  /** Whether to track performance metrics */
  enablePerformanceTracking: boolean;
  /** Whether to track error occurrences */
  enableErrorTracking: boolean;
  /** Whether to track token usage */
  enableTokenTracking: boolean;
  /** Minimum level of logs to capture */
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  /** Number of days to retain monitoring data */
  retentionPeriod: number;
  /** Sampling rate for monitoring events (0.0 to 1.0) */
  samplingRate: number;
  /** Whether to enable the alerting system */
  enableAlerting: boolean;
  /** Threshold values that trigger alerts */
  alertThresholds: {
    /** Maximum execution time in milliseconds before triggering an alert */
    executionTime: number;
    /** Maximum error rate percentage before triggering an alert */
    errorRate: number;
    /** Maximum memory usage in MB before triggering an alert */
    memoryUsage: number;
  };
}

/**
 * Configuration for the processor integration
 */
export interface ProcessorIntegrationConfig {
  /** Whether to enable automatic indexing of processed documents */
  enableAutoIndexing?: boolean;
  /** Interval in seconds between automatic indexing operations */
  autoIndexInterval?: number;
  /** Number of documents to process in each batch */
  batchSize?: number;
  /** Whether to enable caching of processor results */
  enableCaching?: boolean;
  /** Time-to-live for cached entries in seconds */
  cacheTTL?: number;
  /** Whether to enable monitoring for the processor */
  enableMonitoring?: boolean;
  /** Maximum number of concurrent processing operations */
  maxConcurrentProcesses?: number;
}