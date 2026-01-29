/**
 * Type definitions for OpenTelemetry distributed tracing
 */

import { SpanStatusCode, SpanKind, Attributes } from '@opentelemetry/api';

/**
 * Main configuration for the tracing system
 */
export interface TracingConfig {
  /** Service name for this instance */
  serviceName: string;
  /** Enable/disable tracing globally */
  enabled: boolean;
  /** Sampling rate (0.0 to 1.0) */
  sampleRate: number;
  /** Exporter configuration */
  exporter: ExporterConfig;
  /** Batch export configuration */
  batchExport?: BatchExportConfig;
  /** Resource attributes */
  resourceAttributes?: Record<string, string>;
}

/**
 * Exporter configuration
 */
export interface ExporterConfig {
  /** Type of exporter to use */
  type: ExporterType;
  /** Exporter-specific options */
  options:
    | OtlpExporterOptions
    | ConsoleExporterOptions
    | CollectorExporterOptions;
}

/**
 * Exporter types
 */
export enum ExporterType {
  /** OpenTelemetry Collector (production) */
  COLLECTOR = 'collector',
  /** Console output (debugging) */
  CONSOLE = 'console',
  /** OTLP protocol */
  OTLP = 'otlp',
}

/**
 * OTLP exporter options
 */
export interface OtlpExporterOptions {
  /** OTLP endpoint URL */
  url: string;
  /** HTTP headers for authentication */
  headers?: Record<string, string>;
}

/**
 * Console exporter options
 */
export interface ConsoleExporterOptions {
  /** Enable colored output */
  colors?: boolean;
  /** Output format (json or pretty) */
  format?: 'json' | 'pretty';
}

/**
 * Collector exporter options
 */
export interface CollectorExporterOptions {
  /** Collector endpoint URL */
  endpoint: string;
  /** Authentication headers */
  headers?: Record<string, string>;
  /** Compression type */
  compression?: 'gzip' | 'none';
}

/**
 * Batch export configuration
 */
export interface BatchExportConfig {
  /** Export interval in milliseconds */
  exportIntervalMillis?: number;
  /** Maximum queue size */
  maxQueueSize?: number;
  /** Maximum batch size */
  maxExportBatchSize?: number;
  /** Export timeout in milliseconds */
  exportTimeoutMillis?: number;
}

/**
 * Configuration for individual trace operations
 */
export interface TraceConfig {
  /** Operation name */
  name: string;
  /** Span kind */
  kind?: SpanKind;
  /** Attributes to attach to the span */
  attributes?: SpanAttributes;
  /** Link to parent span */
  parentSpan?: string;
  /** Whether to record events */
  recordEvents?: boolean;
  /** Whether to trace this operation regardless of sampling */
  forceRecording?: boolean;
}

/**
 * Standard span attributes - extends OpenTelemetry Attributes
 */
export type SpanAttributes = Attributes & {
  // Standard OTEL attributes
  'http.method'?: string;
  'http.url'?: string;
  'http.status_code'?: number;
  'http.response_size'?: number;
  'http.request_size'?: number;
  'db.system'?: string;
  'db.name'?: string;
  'db.statement'?: string;
  'db.operation'?: string;
  'messaging.system'?: string;
  'messaging.destination'?: string;
  'messaging.message_id'?: string;

  // BubbleLab-specific attributes
  'bubble.name'?: string;
  'bubble.operation'?: string;
  'bubble.type'?: 'service' | 'tool' | 'workflow';
  'bubble.id'?: string;
  'bubble.variable_name'?: string;
  'bubbleflow.id'?: string;
  'bubbleflow.name'?: string;

  // Correlation and tracing
  'correlation.id'?: string;
  'execution.id'?: string;
  'trace.parent_id'?: string;

  // Error attributes
  'error.type'?: string;
  'error.message'?: string;
  'error.stack'?: string;

  // Performance attributes
  'duration.ms'?: number;
  'memory.used.mb'?: number;
  'cpu.used.percent'?: number;
};

/**
 * Span context information
 */
export interface SpanContext {
  /** Trace ID */
  traceId: string;
  /** Span ID */
  spanId: string;
  /** Parent span ID */
  parentSpanId?: string;
  /** Trace flags */
  traceFlags: number;
  /** Trace state */
  traceState?: string;
  /** Span attributes */
  attributes?: SpanAttributes;
}

/**
 * Trace metrics
 */
export interface TraceMetrics {
  /** Total number of traces */
  totalTraces: number;
  /** Total number of spans */
  totalSpans: number;
  /** Error rate (0-1) */
  errorRate: number;
  /** Average duration in milliseconds */
  avgDuration: number;
  /** P50 duration */
  p50Duration: number;
  /** P95 duration */
  p95Duration: number;
  /** P99 duration */
  p99Duration: number;
  /** Throughput (operations per second) */
  throughput: number;
  /** Slowest operations */
  slowestOperations: SlowOperation[];
}

/**
 * Slow operation information
 */
export interface SlowOperation {
  /** Operation name */
  name: string;
  /** Duration in milliseconds */
  duration: number;
  /** Timestamp */
  timestamp: Date;
  /** Error (if any) */
  error?: string;
}

/**
 * Performance analysis result
 */
export interface PerformanceAnalysis {
  /** Overall metrics */
  metrics: TraceMetrics;
  /** Bottlenecks identified */
  bottlenecks: Bottleneck[];
  /** Recommendations */
  recommendations: Recommendation[];
  /** Critical path analysis */
  criticalPath: CriticalPath[];
}

/**
 * Performance bottleneck
 */
export interface Bottleneck {
  /** Operation name */
  operation: string;
  /** Average duration */
  avgDuration: number;
  /** Impact level */
  impact: 'low' | 'medium' | 'high' | 'critical';
  /** Frequency */
  frequency: number;
  /** Suggested action */
  suggestedAction?: string;
}

/**
 * Performance recommendation
 */
export interface Recommendation {
  /** Type of recommendation */
  type: 'optimization' | 'caching' | 'parallelization' | 'error-handling' | 'scaling';
  /** Priority */
  priority: 'low' | 'medium' | 'high' | 'critical';
  /** Description */
  description: string;
  /** Expected impact */
  expectedImpact?: string;
}

/**
 * Critical path in workflow
 */
export interface CriticalPath {
  /** Chain of operations */
  operations: string[];
  /** Total duration */
  totalDuration: number;
  /** Percentage of total execution time */
  percentageOfTotal: number;
  /** Optimization potential */
  optimizationPotential: number;
}

/**
 * Alert rule configuration
 */
export interface AlertRule {
  /** Unique rule identifier */
  id: string;
  /** Rule name */
  name: string;
  /** Rule description */
  description: string;
  /** Condition expression */
  condition: AlertCondition;
  /** Threshold value */
  threshold: number;
  /** Time window in seconds */
  timeWindowSeconds: number;
  /** Severity level */
  severity: 'info' | 'warning' | 'error' | 'critical';
  /** Enable/disable rule */
  enabled: boolean;
  /** Notification channels */
  notificationChannels: string[];
}

/**
 * Alert condition
 */
export interface AlertCondition {
  /** Metric to monitor */
  metric: 'latency' | 'error_rate' | 'throughput' | 'missing_spans';
  /** Aggregation function */
  aggregation: 'avg' | 'p95' | 'p99' | 'max' | 'rate';
  /** Filter conditions */
  filters?: Record<string, string>;
}

/**
 * Alert trigger result
 */
export interface AlertTrigger {
  /** Rule that was triggered */
  rule: AlertRule;
  /** Actual value that triggered the alert */
  actualValue: number;
  /** Threshold value */
  threshold: number;
  /** Timestamp */
  timestamp: Date;
  /** Affected operations */
  affectedOperations: string[];
  /** Additional context */
  context: Record<string, unknown>;
}
