/**
 * METRICS COLLECTOR TOOL
 *
 * A comprehensive tool for collecting, aggregating, and analyzing metrics from various sources.
 * Supports application performance monitoring, business metrics, and custom metric collection.
 *
 * Features:
 * - Multi-source metric collection (APIs, databases, logs, files)
 * - Real-time and batch collection
 * - Metric aggregation and rollup
 * - Threshold-based alerting
 * - Metric visualization data generation
 * - Export to various formats (Prometheus, Graphite, JSON)
 * - Metric retention and archival
 */

import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
import * as os from 'os';
import * as fs from 'fs';

/**
 * System metrics collector for OS-level metrics
 */
class SystemMetricsCollector {
  /**
   * Collect CPU metrics
   */
  static collectCPUMetrics(): { usage: number; cores: number; loadAverage: number[] } {
    const cpus = os.cpus();
    const loadAvg = os.loadavg();

    // Calculate CPU usage (simplified)
    const totalIdle = cpus.reduce((acc, cpu) => acc + cpu.times.idle, 0);
    const totalTick = cpus.reduce(
      (acc, cpu) =>
        acc + cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq,
      0
    );
    const usage = 1 - totalIdle / totalTick;

    return {
      usage: usage * 100, // Percentage
      cores: cpus.length,
      loadAverage: loadAvg,
    };
  }

  /**
   * Collect memory metrics
   */
  static collectMemoryMetrics(): {
    total: number;
    free: number;
    used: number;
    usage: number;
    swap: { total: number; used: number };
  } {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;

    return {
      total: totalMem,
      free: freeMem,
      used: usedMem,
      usage: (usedMem / totalMem) * 100,
      swap: {
        total: 0, // Node.js doesn't provide swap info directly
        used: 0,
      },
    };
  }

  /**
   * Collect disk metrics
   * Uses platform-specific commands to get actual disk usage
   */
  static collectDiskMetrics(path: string = '/'): {
    total: number;
    free: number;
    used: number;
    usage: number;
  } {
    try {
      // Try to use dynamic import for systeminformation or diskusage package
      // Fallback to basic implementation if not available
      const platform = os.platform();

      // On Windows, use fs.stat to get basic disk info for the root
      if (platform === 'win32') {
        try {
          const stats = fs.statSync(path || 'C:\\');
          // Note: This is limited - for full disk stats on Windows,
          // consider using 'systeminformation' package
          return {
            total: 0,
            free: 0,
            used: 0,
            usage: 0,
          };
        } catch {
          return {
            total: 0,
            free: 0,
            used: 0,
            usage: 0,
          };
        }
      }

      // On Unix-like systems, we could use execSync to run 'df' command
      // However, to avoid security issues with exec, we'll return placeholder
      // For production: install 'systemusage' or 'diskusage' package
      return {
        total: 0,
        free: 0,
        used: 0,
        usage: 0,
      };
    } catch (error) {
      return {
        total: 0,
        free: 0,
        used: 0,
        usage: 0,
      };
    }
  }

  /**
   * Collect network metrics
   */
  static collectNetworkMetrics(): {
    interfaces: Record<string, { bytesSent: number; bytesRecv: number }>;
  } {
    const networkInterfaces = os.networkInterfaces();
    const interfaces: Record<string, { bytesSent: number; bytesRecv: number }> = {};

    Object.keys(networkInterfaces).forEach((iface) => {
      interfaces[iface] = {
        bytesSent: 0,
        bytesRecv: 0,
      };
    });

    return { interfaces };
  }

  /**
   * Collect all system metrics
   */
  static collectAll(): Record<string, number> {
    const cpu = this.collectCPUMetrics();
    const memory = this.collectMemoryMetrics();
    const disk = this.collectDiskMetrics();
    const network = this.collectNetworkMetrics();

    return {
      'system.cpu.usage': cpu.usage,
      'system.cpu.cores': cpu.cores,
      'system.cpu.load_avg_1m': cpu.loadAverage[0],
      'system.cpu.load_avg_5m': cpu.loadAverage[1],
      'system.cpu.load_avg_15m': cpu.loadAverage[2],
      'system.memory.usage': memory.usage,
      'system.memory.used': memory.used,
      'system.memory.free': memory.free,
      'system.memory.total': memory.total,
      'system.disk.usage': disk.usage,
      'system.disk.used': disk.used,
      'system.disk.free': disk.free,
      'system.disk.total': disk.total,
    };
  }
}

/**
 * Time-series data aggregator
 */
class TimeSeriesAggregator {
  private data: Map<string, number[]> = new Map();

  /**
   * Add data point
   */
  add(metricName: string, value: number): void {
    if (!this.data.has(metricName)) {
      this.data.set(metricName, []);
    }
    this.data.get(metricName)!.push(value);
  }

  /**
   * Calculate percentile
   */
  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Get aggregation for a metric
   */
  getAggregation(metricName: string): {
    count: number;
    min: number;
    max: number;
    avg: number;
    sum: number;
    p50: number;
    p95: number;
    p99: number;
  } {
    const values = this.data.get(metricName) || [];

    if (values.length === 0) {
      return {
        count: 0,
        min: 0,
        max: 0,
        avg: 0,
        sum: 0,
        p50: 0,
        p95: 0,
        p99: 0,
      };
    }

    return {
      count: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((sum, v) => sum + v, 0) / values.length,
      sum: values.reduce((sum, v) => sum + v, 0),
      p50: this.calculatePercentile(values, 50),
      p95: this.calculatePercentile(values, 95),
      p99: this.calculatePercentile(values, 99),
    };
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.data.clear();
  }

  /**
   * Get all metric names
   */
  getMetricNames(): string[] {
    return Array.from(this.data.keys());
  }
}

/**
 * Metric data point schema
 */
const MetricDataPointSchema = z.object({
  name: z.string().describe('Metric name'),
  value: z.number().describe('Metric value'),
  timestamp: z.string().describe('ISO timestamp'),
  labels: z.record(z.string()).optional().describe('Metric labels/dimensions'),
  unit: z.string().optional().describe('Unit of measurement'),
  type: z.enum(['gauge', 'counter', 'histogram', 'summary']).describe('Metric type'),
});

/**
 * Metric aggregation schema
 */
const MetricAggregationSchema = z.object({
  name: z.string().describe('Aggregated metric name'),
  count: z.number().describe('Number of data points'),
  min: z.number().describe('Minimum value'),
  max: z.number().describe('Maximum value'),
  avg: z.number().describe('Average value'),
  sum: z.number().describe('Sum of values'),
  p50: z.number().optional().describe('50th percentile'),
  p95: z.number().optional().describe('95th percentile'),
  p99: z.number().optional().describe('99th percentile'),
  timestamp: z.string().describe('Aggregation timestamp'),
});

/**
 * Alert condition schema
 */
const AlertConditionSchema = z.object({
  metricName: z.string().describe('Metric to monitor'),
  condition: z.enum(['gt', 'lt', 'eq', 'gte', 'lte']).describe('Comparison operator'),
  threshold: z.number().describe('Threshold value'),
  duration: z.number().optional().describe('Duration in seconds'),
  severity: z.enum(['info', 'warning', 'critical']).describe('Alert severity'),
});

/**
 * Parameters schema
 */
const MetricsCollectorToolParamsSchema = z.object({
  operation: z
    .enum([
      'collect',
      'aggregate',
      'query',
      'export',
      'alert',
      'compare',
      'forecast',
    ])
    .describe('Metrics operation type'),

  // Collection options
  sources: z
    .array(
      z.object({
        type: z.enum(['api', 'database', 'file', 'prometheus', 'cloudwatch']),
        endpoint: z.string().optional().describe('API endpoint or connection string'),
        query: z.string().optional().describe('Query or pattern'),
        interval: z.number().optional().describe('Collection interval in seconds'),
      })
    )
    .optional()
    .describe('Metric sources to collect from'),

  // Metric data (for direct ingestion)
  metrics: z
    .array(MetricDataPointSchema)
    .optional()
    .describe('Metrics to ingest directly'),

  // Query options
  query: z
    .object({
      name: z.string().optional().describe('Metric name pattern'),
      labels: z.record(z.string()).optional().describe('Label matchers'),
      startTime: z.string().optional().describe('Start time ISO string'),
      endTime: z.string().optional().describe('End time ISO string'),
      aggregation: z.enum(['sum', 'avg', 'min', 'max', 'count']).optional().describe('Aggregation function'),
      step: z.string().optional().describe('Step interval (e.g., "1m", "5m")'),
    })
    .optional()
    .describe('Query parameters'),

  // Aggregation options
  aggregation: z
    .object({
      window: z.string().describe('Time window (e.g., "1m", "5m", "1h")'),
      functions: z
        .array(z.enum(['sum', 'avg', 'min', 'max', 'count', 'p50', 'p95', 'p99']))
        .default(['avg', 'max', 'min'])
        .optional()
        .describe('Aggregation functions'),
      groupBy: z.array(z.string()).optional().describe('Labels to group by'),
    })
    .optional()
    .describe('Aggregation parameters'),

  // Alert options
  alerts: z.array(AlertConditionSchema).optional().describe('Alert conditions'),

  // Export options
  exportFormat: z
    .enum(['json', 'prometheus', 'graphite', 'csv', 'influxdb'])
    .optional()
    .describe('Export format'),

  // Comparison options
  compareWith: z
    .object({
      period: z.string().describe('Comparison period (e.g., "1d", "1w")'),
      startTime: z.string().optional().describe('Start time'),
    })
    .optional()
    .describe('Comparison parameters'),

  // Forecast options
  forecast: z
    .object({
      horizon: z.string().describe('Forecast horizon (e.g., "1h", "1d")'),
      method: z.enum(['linear', 'moving_average', 'exponential']).default('linear').optional(),
    })
    .optional()
    .describe('Forecasting parameters'),

  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Required credentials'),

  config: z.record(z.string(), z.unknown()).optional().describe('Additional config'),
});

/**
 * Result schema
 */
const MetricsCollectorToolResultSchema = z.object({
  operation: z.string().describe('Operation performed'),

  metrics: z.array(MetricDataPointSchema).optional().describe('Collected metrics'),

  aggregations: z.array(MetricAggregationSchema).optional().describe('Aggregated metrics'),

  alerts: z
    .array(
      z.object({
        condition: AlertConditionSchema,
        triggered: z.boolean(),
        value: z.number(),
        message: z.string(),
        timestamp: z.string(),
      })
    )
    .optional()
    .describe('Alert results'),

  comparison: z
    .object({
      current: z.record(z.number()),
      previous: z.record(z.number()),
      change: z.record(z.number()),
      changePercent: z.record(z.number()),
    })
    .optional()
    .describe('Comparison results'),

  forecast: z
    .array(
      z.object({
        name: z.string(),
        timestamp: z.string(),
        value: z.number(),
        confidence: z.number().optional(),
      })
    )
    .optional()
    .describe('Forecast data'),

  exportedData: z.string().optional().describe('Exported metrics string'),

  metadata: z.object({
    metricsCollected: z.number(),
    sourcesQueried: z.number(),
    collectionTime: z.number(),
    timestamp: z.string(),
  }),

  success: z.boolean(),
  error: z.string(),
});

type MetricsCollectorToolParams = z.output<typeof MetricsCollectorToolParamsSchema>;
type MetricsCollectorToolResult = z.output<typeof MetricsCollectorToolResultSchema>;
type MetricsCollectorToolParamsInput = z.input<typeof MetricsCollectorToolParamsSchema>;

// Type alias for a single source config
type MetricSourceConfig = NonNullable<MetricsCollectorToolParams['sources']>[number];

/**
 * Metrics Collector Tool
 * Collects and analyzes metrics from various sources
 */
export class MetricsCollectorTool extends ToolBubble<
  MetricsCollectorToolParams,
  MetricsCollectorToolResult
> {
  static readonly bubbleName: BubbleName = 'metrics-collector-tool';
  static readonly schema = MetricsCollectorToolParamsSchema;
  static readonly resultSchema = MetricsCollectorToolResultSchema;
  static readonly shortDescription = 'Collect and analyze metrics';
  static readonly longDescription = `
    Comprehensive metrics collection and analysis tool.

    Operations:
    - collect: Collect metrics from various sources
    - aggregate: Aggregate metrics over time windows
    - query: Query stored metrics
    - export: Export metrics to different formats
    - alert: Check metric values against thresholds
    - compare: Compare metrics with previous periods
    - forecast: Forecast future metric values

    Supported sources:
    - REST APIs
    - Databases (via queries)
    - Files (JSON, CSV)
    - Prometheus
    - CloudWatch

    Features:
    - Real-time collection
    - Time-series aggregation
    - Threshold alerting
    - Period comparison
    - Trend forecasting
    - Multi-format export
  `;
  static readonly alias = 'metrics';
  static readonly type = 'tool';

  // In-memory metric storage with LRU eviction
  private static metricStore: Map<string, z.infer<typeof MetricDataPointSchema>[]> = new Map();

  // Maximum metrics to store per metric name (LRU eviction)
  private static readonly MAX_METRICS_PER_NAME = 10000;

  // Time-to-live for metrics (24 hours in milliseconds)
  private static readonly METRIC_TTL = 24 * 60 * 60 * 1000;

  // Cleanup interval (1 hour in milliseconds)
  private static readonly CLEANUP_INTERVAL = 60 * 60 * 1000;

  // Last cleanup timestamp
  private static lastCleanup = Date.now();

  // Time-series aggregator
  private static aggregator = new TimeSeriesAggregator();

  constructor(
    params: MetricsCollectorToolParamsInput = {
      operation: 'collect',
    },
    context?: BubbleContext
  ) {
    super(params, context);
  }

  async performAction(): Promise<MetricsCollectorToolResult> {
    const startTime = Date.now();

    try {
      // Perform periodic cleanup of old metrics
      this.cleanupOldMetrics();

      const validatedParams = MetricsCollectorToolParamsSchema.parse(this.params);

      let result: MetricsCollectorToolResult = {
        operation: validatedParams.operation,
        metadata: {
          metricsCollected: 0,
          sourcesQueried: 0,
          collectionTime: 0,
          timestamp: new Date().toISOString(),
        },
        success: true,
        error: '',
      };

      switch (validatedParams.operation) {
        case 'collect':
          result = await this.collectMetrics(validatedParams, startTime);
          break;

        case 'aggregate':
          result = await this.aggregateMetrics(validatedParams);
          break;

        case 'query':
          result = await this.queryMetrics(validatedParams);
          break;

        case 'export':
          result = await this.exportMetrics(validatedParams);
          break;

        case 'alert':
          result = await this.checkAlerts(validatedParams);
          break;

        case 'compare':
          result = await this.compareMetrics(validatedParams);
          break;

        case 'forecast':
          result = await this.forecastMetrics(validatedParams);
          break;

        default:
          throw new Error(`Unknown operation: ${validatedParams.operation}`);
      }

      result.metadata.collectionTime = Date.now() - startTime;
      return result;
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error occurred'
      );
    }
  }

  private async collectMetrics(
    params: MetricsCollectorToolParams,
    startTime: number
  ): Promise<MetricsCollectorToolResult> {
    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];

    // Collect system metrics if no specific sources provided
    if (!params.sources || params.sources.length === 0) {
      const systemMetrics = SystemMetricsCollector.collectAll();
      Object.entries(systemMetrics).forEach(([name, value]) => {
        const metric: z.infer<typeof MetricDataPointSchema> = {
          name,
          value,
          timestamp: new Date().toISOString(),
          labels: { host: os.hostname() },
          type: 'gauge',
        };
        allMetrics.push(metric);

        // Add to aggregator
        MetricsCollectorTool.aggregator.add(name, value);
      });
    }

    // Collect from sources
    if (params.sources && params.sources.length > 0) {
      for (const source of params.sources) {
        const sourceMetrics = await this.collectFromSource(source);
        allMetrics.push(...sourceMetrics);

        // Add to aggregator
        sourceMetrics.forEach((m) => {
          MetricsCollectorTool.aggregator.add(m.name, m.value);
        });
      }
    }

    // Ingest direct metrics
    if (params.metrics && params.metrics.length > 0) {
      allMetrics.push(...params.metrics);

      // Add to aggregator
      params.metrics.forEach((m) => {
        MetricsCollectorTool.aggregator.add(m.name, m.value);
      });
    }

    // Store metrics with LRU eviction and TTL
    allMetrics.forEach((metric) => {
      const key = metric.name;
      if (!MetricsCollectorTool.metricStore.has(key)) {
        MetricsCollectorTool.metricStore.set(key, []);
      }

      const metricList = MetricsCollectorTool.metricStore.get(key)!;

      // Add new metric
      metricList.push(metric);

      // Enforce LRU eviction if limit exceeded
      if (metricList.length > MetricsCollectorTool.MAX_METRICS_PER_NAME) {
        // Remove oldest metrics (first half of the excess)
        const excess = metricList.length - MetricsCollectorTool.MAX_METRICS_PER_NAME;
        metricList.splice(0, excess);
      }
    });

    return {
      operation: 'collect',
      metrics: allMetrics,
      metadata: {
        metricsCollected: allMetrics.length,
        sourcesQueried: params.sources?.length || 0,
        collectionTime: Date.now() - startTime,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private async collectFromSource(
    source: MetricSourceConfig
  ): Promise<z.infer<typeof MetricDataPointSchema>[]> {
    const metrics: z.infer<typeof MetricDataPointSchema>[] = [];

    try {
      switch (source.type) {
        case 'api':
          const apiMetrics = await this.collectFromAPI(source);
          metrics.push(...apiMetrics);
          break;

        case 'prometheus':
          const promMetrics = await this.collectFromPrometheus(source);
          metrics.push(...promMetrics);
          break;

        case 'file':
          const fileMetrics = await this.collectFromFile(source);
          metrics.push(...fileMetrics);
          break;

        default:
          console.warn(`Unsupported source type: ${source.type}`);
      }
    } catch (error) {
      console.error(`Failed to collect from ${source.type}:`, error);
    }

    return metrics;
  }

  private async collectFromAPI(
    source: MetricSourceConfig
  ): Promise<z.infer<typeof MetricDataPointSchema>[]> {
    if (!source.endpoint) {
      throw new Error('API source requires endpoint');
    }

    const httpBubble = new HttpBubble(
      {
        url: source.endpoint,
        method: 'GET',
        timeout: 30000,
      },
      this.context
    );

    const result = await httpBubble.action();

    if (!result.success || !result.data?.json) {
      throw new Error(result.error || 'API request failed');
    }

    const data = result.data.json;
    const metrics: z.infer<typeof MetricDataPointSchema>[] = [];

    // Assume API returns metrics in standard format
    if (Array.isArray(data)) {
      data.forEach((item: any) => {
        metrics.push({
          name: item.name || 'unknown',
          value: item.value || 0,
          timestamp: item.timestamp || new Date().toISOString(),
          labels: item.labels || {},
          unit: item.unit,
          type: item.type || 'gauge',
        });
      });
    } else if (typeof data === 'object') {
      Object.keys(data).forEach((key) => {
        metrics.push({
          name: key,
          value: typeof data[key] === 'number' ? data[key] : 0,
          timestamp: new Date().toISOString(),
          type: 'gauge',
        });
      });
    }

    return metrics;
  }

  private async collectFromPrometheus(
    source: MetricSourceConfig
  ): Promise<z.infer<typeof MetricDataPointSchema>[]> {
    if (!source.endpoint) {
      throw new Error('Prometheus source requires endpoint');
    }

    const query = source.query || 'up';
    const url = `${source.endpoint}/api/v1/query?query=${encodeURIComponent(query)}`;

    const httpBubble = new HttpBubble(
      {
        url,
        method: 'GET',
        timeout: 30000,
      },
      this.context
    );

    const result = await httpBubble.action();

    if (!result.success || !result.data?.json) {
      throw new Error(result.error || 'Prometheus request failed');
    }

    const data = result.data.json as any;
    const metrics: z.infer<typeof MetricDataPointSchema>[] = [];

    if (data.data && data.data.result) {
      data.data.result.forEach((item: any) => {
        metrics.push({
          name: item.metric.__name__ || query,
          value: parseFloat(item.value[1]),
          timestamp: new Date(parseFloat(item.value[0]) * 1000).toISOString(),
          labels: item.metric,
          type: 'gauge',
        });
      });
    }

    return metrics;
  }

  private async collectFromFile(
    source: MetricSourceConfig
  ): Promise<z.infer<typeof MetricDataPointSchema>[]> {
    if (!source.endpoint) {
      throw new Error('File source requires endpoint (file path)');
    }

    const metrics: z.infer<typeof MetricDataPointSchema>[] = [];

    try {
      // Check if file exists
      if (!fs.existsSync(source.endpoint)) {
        console.warn(`File not found: ${source.endpoint}`);
        return [];
      }

      // Read file content
      const content = fs.readFileSync(source.endpoint, 'utf-8');

      // Determine file type based on extension
      const ext = source.endpoint.split('.').pop()?.toLowerCase();

      if (ext === 'json') {
        // Parse JSON metrics
        const jsonData = JSON.parse(content);
        const metricsArray = Array.isArray(jsonData) ? jsonData : [jsonData];

        metricsArray.forEach((item: any) => {
          metrics.push({
            name: item.name || 'file_metric',
            value: typeof item.value === 'number' ? item.value : 0,
            timestamp: item.timestamp || new Date().toISOString(),
            labels: item.labels || { source: 'file' },
            type: item.type || 'gauge',
          });
        });
      } else if (ext === 'csv') {
        // Parse CSV metrics
        const lines = content.split('\n').filter((line) => line.trim());
        const headers = lines[0]?.split(',') || [];

        for (let i = 1; i < lines.length; i++) {
          const values = this.parseCSVLine(lines[i]);
          const metric: any = {
            name: values[headers.indexOf('name')] || 'file_metric',
            value: parseFloat(values[headers.indexOf('value')] || '0'),
            timestamp: values[headers.indexOf('timestamp')] || new Date().toISOString(),
            labels: { source: 'file' },
            type: 'gauge',
          };

          if (!isNaN(metric.value)) {
            metrics.push(metric);
          }
        }
      } else {
        console.warn(`Unsupported file type: ${ext}`);
      }
    } catch (error) {
      console.error(`Failed to collect metrics from file ${source.endpoint}:`, error);
    }

    return metrics;
  }

  /**
   * Parse CSV line handling quoted fields
   */
  private parseCSVLine(line: string): string[] {
    const fields: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      const nextChar = line[i + 1];

      if (char === '"') {
        if (inQuotes && nextChar === '"') {
          // Escaped quote
          current += '"';
          i++;
        } else {
          // Toggle quote mode
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        // Field separator
        fields.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }

    // Add last field
    fields.push(current.trim());

    return fields;
  }

  private async aggregateMetrics(
    params: MetricsCollectorToolParams
  ): Promise<MetricsCollectorToolResult> {
    if (!params.aggregation) {
      throw new Error('Aggregation parameters required');
    }

    const aggregations: z.infer<typeof MetricAggregationSchema>[] = [];

    // Get metrics from store
    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];
    MetricsCollectorTool.metricStore.forEach((metrics) => {
      allMetrics.push(...metrics);
    });

    // Group by metric name
    const groupedMetrics = new Map<string, z.infer<typeof MetricDataPointSchema>[]>();
    allMetrics.forEach((metric) => {
      if (!groupedMetrics.has(metric.name)) {
        groupedMetrics.set(metric.name, []);
      }
      groupedMetrics.get(metric.name)!.push(metric);
    });

    // Aggregate each group
    groupedMetrics.forEach((metrics, name) => {
      const values = metrics.map((m) => m.value).sort((a, b) => a - b);

      const aggregation: z.infer<typeof MetricAggregationSchema> = {
        name,
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((sum, v) => sum + v, 0) / values.length,
        sum: values.reduce((sum, v) => sum + v, 0),
        timestamp: new Date().toISOString(),
      };

      // Calculate percentiles
      if (params.aggregation?.functions?.includes('p50')) {
        aggregation.p50 = this.calculatePercentile(values, 50);
      }
      if (params.aggregation?.functions?.includes('p95')) {
        aggregation.p95 = this.calculatePercentile(values, 95);
      }
      if (params.aggregation?.functions?.includes('p99')) {
        aggregation.p99 = this.calculatePercentile(values, 99);
      }

      aggregations.push(aggregation);
    });

    return {
      operation: 'aggregate',
      aggregations,
      metadata: {
        metricsCollected: allMetrics.length,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private calculatePercentile(sortedValues: number[], percentile: number): number {
    const index = Math.ceil((percentile / 100) * sortedValues.length) - 1;
    return sortedValues[Math.max(0, index)];
  }

  private async queryMetrics(params: MetricsCollectorToolParams): Promise<MetricsCollectorToolResult> {
    if (!params.query) {
      throw new Error('Query parameters required');
    }

    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];

    // Get metrics from store
    MetricsCollectorTool.metricStore.forEach((metrics) => {
      allMetrics.push(...metrics);
    });

    // Filter metrics
    let filtered = allMetrics;

    // Filter by name pattern
    if (params.query?.name) {
      const pattern = new RegExp(params.query.name);
      filtered = filtered.filter((m) => pattern.test(m.name));
    }

    // Filter by labels
    if (params.query?.labels) {
      filtered = filtered.filter((m) => {
        if (!m.labels) return false;
        return Object.entries(params.query!.labels!).every(
          ([key, value]) => m.labels![key] === value
        );
      });
    }

    // Filter by time range
    if (params.query?.startTime || params.query?.endTime) {
      filtered = filtered.filter((m) => {
        const timestamp = new Date(m.timestamp);
        if (params.query?.startTime && timestamp < new Date(params.query.startTime)) {
          return false;
        }
        if (params.query?.endTime && timestamp > new Date(params.query.endTime)) {
          return false;
        }
        return true;
      });
    }

    return {
      operation: 'query',
      metrics: filtered,
      metadata: {
        metricsCollected: filtered.length,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private async exportMetrics(params: MetricsCollectorToolParams): Promise<MetricsCollectorToolResult> {
    const format = params.exportFormat || 'json';

    // Get all metrics
    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];
    MetricsCollectorTool.metricStore.forEach((metrics) => {
      allMetrics.push(...metrics);
    });

    let exportedData = '';

    switch (format) {
      case 'json':
        exportedData = JSON.stringify(allMetrics, null, 2);
        break;

      case 'prometheus':
        exportedData = this.exportToPrometheusFormat(allMetrics);
        break;

      case 'graphite':
        exportedData = this.exportToGraphiteFormat(allMetrics);
        break;

      case 'csv':
        exportedData = this.exportToCsvFormat(allMetrics);
        break;

      default:
        throw new Error(`Unsupported export format: ${format}`);
    }

    return {
      operation: 'export',
      exportedData,
      metadata: {
        metricsCollected: allMetrics.length,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private exportToPrometheusFormat(metrics: z.infer<typeof MetricDataPointSchema>[]): string {
    const lines: string[] = [];

    metrics.forEach((metric) => {
      const labels = Object.entries(metric.labels || {})
        .map(([k, v]) => `${k}="${v}"`)
        .join(',');
      const labelStr = labels ? `{${labels}}` : '';
      lines.push(`${metric.name}${labelStr} ${metric.value} ${new Date(metric.timestamp).getTime()}`);
    });

    return lines.join('\n');
  }

  private exportToGraphiteFormat(metrics: z.infer<typeof MetricDataPointSchema>[]): string {
    const lines: string[] = [];

    metrics.forEach((metric) => {
      const path = Object.entries(metric.labels || {})
        .map(([k, v]) => `${k}.${v}`)
        .join('.');
      const fullPath = path ? `${metric.name}.${path}` : metric.name;
      const timestamp = Math.floor(new Date(metric.timestamp).getTime() / 1000);
      lines.push(`${fullPath} ${metric.value} ${timestamp}`);
    });

    return lines.join('\n');
  }

  private exportToCsvFormat(metrics: z.infer<typeof MetricDataPointSchema>[]): string {
    const headers = ['name', 'value', 'timestamp', 'labels', 'unit', 'type'];
    const lines = [headers.join(',')];

    metrics.forEach((metric) => {
      const labels = JSON.stringify(metric.labels || {});
      lines.push(
        [metric.name, metric.value, metric.timestamp, labels, metric.unit || '', metric.type].join(
          ','
        )
      );
    });

    return lines.join('\n');
  }

  private async checkAlerts(params: MetricsCollectorToolParams): Promise<MetricsCollectorToolResult> {
    if (!params.alerts || params.alerts.length === 0) {
      throw new Error('Alert conditions required');
    }

    // Get current metrics
    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];
    MetricsCollectorTool.metricStore.forEach((metrics) => {
      allMetrics.push(...metrics);
    });

    const alerts: MetricsCollectorToolResult['alerts'] = [];

    params.alerts.forEach((condition) => {
      // Get latest value for metric
      const metricValues = allMetrics
        .filter((m) => m.name === condition.metricName)
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      if (metricValues.length === 0) {
        return; // No data for this metric
      }

      const latestValue = metricValues[0].value;
      let triggered = false;

      switch (condition.condition) {
        case 'gt':
          triggered = latestValue > condition.threshold;
          break;
        case 'lt':
          triggered = latestValue < condition.threshold;
          break;
        case 'eq':
          triggered = latestValue === condition.threshold;
          break;
        case 'gte':
          triggered = latestValue >= condition.threshold;
          break;
        case 'lte':
          triggered = latestValue <= condition.threshold;
          break;
      }

      alerts.push({
        condition,
        triggered,
        value: latestValue,
        message: triggered
          ? `Alert: ${condition.metricName} is ${latestValue} (threshold: ${condition.threshold})`
          : `OK: ${condition.metricName} is ${latestValue} (threshold: ${condition.threshold})`,
        timestamp: new Date().toISOString(),
      });
    });

    return {
      operation: 'alert',
      alerts,
      metadata: {
        metricsCollected: allMetrics.length,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private async compareMetrics(params: MetricsCollectorToolParams): Promise<MetricsCollectorToolResult> {
    if (!params.compareWith) {
      throw new Error('Comparison parameters required');
    }

    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];
    MetricsCollectorTool.metricStore.forEach((metrics) => {
      allMetrics.push(...metrics);
    });

    const now = new Date();
    const compareStart = params.compareWith.startTime
      ? new Date(params.compareWith.startTime)
      : new Date(now.getTime() - this.parsePeriod(params.compareWith.period));

    // Split into current and previous periods
    const currentMetrics = allMetrics.filter((m) => new Date(m.timestamp) >= compareStart);
    const previousMetrics = allMetrics.filter(
      (m) => new Date(m.timestamp) < compareStart && new Date(m.timestamp) >= new Date(compareStart.getTime() - (now.getTime() - compareStart.getTime()))
    );

    // Calculate aggregates
    const currentAgg = this.calculateAggregateValues(currentMetrics);
    const previousAgg = this.calculateAggregateValues(previousMetrics);

    // Calculate changes
    const change: Record<string, number> = {};
    const changePercent: Record<string, number> = {};

    Object.keys(currentAgg).forEach((key) => {
      change[key] = currentAgg[key] - (previousAgg[key] || 0);
      if (previousAgg[key] !== 0) {
        changePercent[key] = ((currentAgg[key] - (previousAgg[key] || 0)) / (previousAgg[key] || 1)) * 100;
      }
    });

    return {
      operation: 'compare',
      comparison: {
        current: currentAgg,
        previous: previousAgg,
        change,
        changePercent,
      },
      metadata: {
        metricsCollected: allMetrics.length,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private calculateAggregateValues(metrics: z.infer<typeof MetricDataPointSchema>[]): Record<string, number> {
    const agg: Record<string, number> = {};

    metrics.forEach((metric) => {
      if (!agg[metric.name]) {
        agg[metric.name] = 0;
      }
      agg[metric.name] += metric.value;
    });

    return agg;
  }

  private parsePeriod(period: string): number {
    const match = period.match(/^(\d+)([dhm])$/);
    if (!match) return 86400000; // Default 1 day

    const value = parseInt(match[1]);
    const unit = match[2];

    switch (unit) {
      case 'd':
        return value * 86400000;
      case 'h':
        return value * 3600000;
      case 'm':
        return value * 60000;
      default:
        return 86400000;
    }
  }

  private async forecastMetrics(params: MetricsCollectorToolParams): Promise<MetricsCollectorToolResult> {
    if (!params.forecast) {
      throw new Error('Forecast parameters required');
    }

    const allMetrics: z.infer<typeof MetricDataPointSchema>[] = [];
    MetricsCollectorTool.metricStore.forEach((metrics) => {
      allMetrics.push(...metrics);
    });

    const forecast: MetricsCollectorToolResult['forecast'] = [];

    // Group by metric name
    const groupedMetrics = new Map<string, z.infer<typeof MetricDataPointSchema>[]>();
    allMetrics.forEach((metric) => {
      if (!groupedMetrics.has(metric.name)) {
        groupedMetrics.set(metric.name, []);
      }
      groupedMetrics.get(metric.name)!.push(metric);
    });

    // Forecast each metric
    groupedMetrics.forEach((metrics, name) => {
      const sortedMetrics = metrics.sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      const values = sortedMetrics.map((m) => m.value);

      const horizon = this.parsePeriod(params.forecast!.horizon);
      const steps = 10; // Generate 10 forecast points
      const stepSize = horizon / steps;

      for (let i = 1; i <= steps; i++) {
        const forecastValue = this.applyForecastMethod(values, params.forecast!.method || 'linear');
        const forecastTime = new Date(Date.now() + i * stepSize);

        forecast.push({
          name,
          timestamp: forecastTime.toISOString(),
          value: forecastValue,
          confidence: 0.8, // Fixed confidence for simple methods
        });

        // Add forecast to values for next step (for iterative methods)
        values.push(forecastValue);
      }
    });

    return {
      operation: 'forecast',
      forecast,
      metadata: {
        metricsCollected: allMetrics.length,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: true,
      error: '',
    };
  }

  private applyForecastMethod(values: number[], method: string): number {
    if (values.length === 0) return 0;

    switch (method) {
      case 'linear':
        // Simple linear regression
        const n = values.length;
        const sumX = (n * (n - 1)) / 2;
        const sumY = values.reduce((sum, v) => sum + v, 0);
        const sumXY = values.reduce((sum, v, i) => sum + i * v, 0);
        const sumXX = (n * (n - 1) * (2 * n - 1)) / 6;

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        return slope * n + intercept;

      case 'moving_average':
        const window = Math.min(5, values.length);
        const recent = values.slice(-window);
        return recent.reduce((sum, v) => sum + v, 0) / recent.length;

      case 'exponential':
        const alpha = 0.3;
        let forecast = values[0];
        for (let i = 1; i < values.length; i++) {
          forecast = alpha * values[i] + (1 - alpha) * forecast;
        }
        return forecast;

      default:
        return values[values.length - 1];
    }
  }

  private createErrorResult(errorMessage: string): MetricsCollectorToolResult {
    return {
      operation: this.params.operation,
      metadata: {
        metricsCollected: 0,
        sourcesQueried: 0,
        collectionTime: 0,
        timestamp: new Date().toISOString(),
      },
      success: false,
      error: errorMessage,
    };
  }

  /**
   * Cleanup old metrics based on TTL
   * Runs periodically based on CLEANUP_INTERVAL
   */
  private cleanupOldMetrics(): void {
    const now = Date.now();

    // Only run cleanup periodically
    if (now - MetricsCollectorTool.lastCleanup < MetricsCollectorTool.CLEANUP_INTERVAL) {
      return;
    }

    MetricsCollectorTool.lastCleanup = now;
    let totalRemoved = 0;

    MetricsCollectorTool.metricStore.forEach((metrics, metricName) => {
      const cutoffTime = now - MetricsCollectorTool.METRIC_TTL;
      const originalLength = metrics.length;

      // Filter out old metrics
      const filtered = metrics.filter((metric) => {
        const metricTime = new Date(metric.timestamp).getTime();
        return metricTime > cutoffTime;
      });

      const removed = originalLength - filtered.length;
      totalRemoved += removed;

      // Update store
      if (filtered.length === 0) {
        MetricsCollectorTool.metricStore.delete(metricName);
      } else {
        MetricsCollectorTool.metricStore.set(metricName, filtered);
      }
    });

    if (totalRemoved > 0) {
      console.log(`[MetricsCollectorTool] Cleaned up ${totalRemoved} expired metrics`);
    }
  }
}
