/**
 * Monitoring and Debugging Type Definitions
 */

/**
 * Monitoring Event Interface
 * Represents an event in the monitoring system
 */
export interface MonitoringEvent {
  /**
   * Event identifier
   */
  id: string;
  
  /**
   * Event type
   */
  type: 'workflow_start' | 'workflow_complete' | 'node_start' | 'node_complete' | 'node_error' | 'system';
  
  /**
   * Event timestamp
   */
  timestamp: Date;
  
  /**
   * Event data
   */
  data: any;
  
  /**
   * Event severity
   */
  severity?: 'info' | 'warning' | 'error' | 'critical';
  
  /**
   * Related workflow ID
   */
  workflowId?: string;
  
  /**
   * Related node ID
   */
  nodeId?: string;
}

/**
 * Performance Metrics Interface
 * Performance metrics for monitoring
 */
export interface PerformanceMetrics {
  /**
   * Throughput (operations per second)
   */
  throughput: number;
  
  /**
   * Average execution time (ms)
   */
  avgExecutionTime: number;
  
  /**
   * Error rate (errors per operation)
   */
  errorRate: number;
  
  /**
   * Memory usage (MB)
   */
  memoryUsage: number;
  
  /**
   * CPU usage (%)
   */
  cpuUsage: number;
  
  /**
   * Active workflows count
   */
  activeWorkflows: number;
  
  /**
   * Queue size
   */
  queueSize: number;
}

/**
 * Debug Information Interface
 * Debug information for troubleshooting
 */
export interface DebugInfo {
  /**
   * Debug session identifier
   */
  sessionId: string;
  
  /**
   * Node identifier
   */
  nodeId: string;
  
  /**
   * Debug data
   */
  data: {
    input?: any;
    output?: any;
    intermediateSteps?: any[];
    timing?: {
      startTime: Date;
      endTime: Date;
      duration: number;
    };
    errors?: Error[];
  };
  
  /**
   * Debug timestamp
   */
  timestamp: Date;
}

/**
 * Monitoring Configuration Interface
 * Configuration for monitoring system
 */
export interface MonitoringConfig {
  /**
   * Monitoring enabled
   */
  enabled: boolean;
  
  /**
   * Event retention period (ms)
   */
  retentionPeriod: number;
  
  /**
   * Maximum event count
   */
  maxEvents: number;
  
  /**
   * Performance sampling interval (ms)
   */
  samplingInterval: number;
  
  /**
   * Alert thresholds
   */
  alertThresholds?: {
    executionTime?: number;
    errorRate?: number;
    memoryUsage?: number;
    cpuUsage?: number;
  };
}

/**
 * Processor Integration Configuration Interface
 * Configuration for processor integration
 */
export interface ProcessorIntegrationConfig {
  /**
   * Processor type
   */
  processorType: 'ragbits' | 'custom';
  
  /**
   * Processor configuration
   */
  processorConfig?: {
    documentProcessor?: {
      chunkSize?: number;
      chunkOverlap?: number;
      embeddingModel?: string;
      vectorStoreType?: string;
    };
    searchConfig?: {
      topK?: number;
      similarityThreshold?: number;
      rerankModel?: string;
    };
    generationConfig?: {
      model?: string;
      temperature?: number;
      maxTokens?: number;
    };
  };
  
  /**
   * Auto-indexing configuration
   */
  autoIndexing?: {
    enabled: boolean;
    interval?: number;
    batchSize?: number;
  };
  
  /**
   * Caching configuration
   */
  caching?: {
    enabled: boolean;
    ttl?: number;
    maxSize?: number;
  };
}

/**
 * BubbleLab Workflow Configuration Interface
 * Configuration for BubbleLab workflow
 */
export interface BubbleLabWorkflowConfig {
  /**
   * Workflow identifier
   */
  id: string;
  
  /**
   * Workflow name
   */
  name: string;
  
  /**
   * Workflow description
   */
  description?: string;
  
  /**
   * Workflow nodes
   */
  nodes: any[];
  
  /**
   * Workflow edges
   */
  edges: any[];
  
  /**
   * Workflow metadata
   */
  metadata?: Record<string, any>;
}

/**
 * Ragbits Configuration Interface
 * Configuration for Ragbits
 */
export interface RagbitsConfig {
  /**
   * Configuration identifier
   */
  id: string;
  
  /**
   * Configuration name
   */
  name: string;
  
  /**
   * Document processor configuration
   */
  documentProcessor?: {
    chunkSize?: number;
    chunkOverlap?: number;
    embeddingModel?: string;
    vectorStoreType?: string;
  };
  
  /**
   * Search configuration
   */
  search?: {
    topK?: number;
    similarityThreshold?: number;
    rerankModel?: string;
  };
  
  /**
   * Generation configuration
   */
  generation?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
  };
}

/**
 * Ragbits Node Configuration Interface
 * Configuration for individual Ragbits nodes
 */
export interface RagbitsNodeConfig {
  /**
   * Node identifier
   */
  id: string;
  
  /**
   * Node type
   */
  type: 'ingest' | 'search' | 'generation' | 'index';
  
  /**
   * Node configuration
   */
  config: any;
  
  /**
   * Input connections
   */
  inputs?: string[];
  
  /**
   * Output connections
   */
  outputs?: string[];
}