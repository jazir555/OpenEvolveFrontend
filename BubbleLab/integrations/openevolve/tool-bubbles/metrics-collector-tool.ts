/**
 * Metrics Collector Tool Bubble
 *
 * Collects, aggregates, and analyzes metrics from OpenEvolve services
 * including performance, resource usage, and operational metrics.
 */

import { z } from 'zod';
import { HttpBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';

const MetricTypeSchema = z.enum([
  'counter',
  'gauge',
  'histogram',
  'summary',
]);

const MetricsCollectorParamsSchema = z.object({
  operation: z.enum(['collect', 'aggregate', 'query', 'export', 'alert']).describe('Collector operation'),

  // Metric collection
  services: z.array(z.string()).optional().describe('Services to collect metrics from'),
  metricNames: z.array(z.string()).optional().describe('Specific metrics to collect'),
  timeRange: z.object({
    start: z.string().optional().describe('Start time (ISO 8601)'),
    end: z.string().optional().describe('End time (ISO 8601)'),
    duration: z.string().optional().describe('Duration (e.g., "1h", "24h", "7d")'),
  }).optional().describe('Time range for metrics'),

  // Aggregation
  aggregation: z.enum(['sum', 'avg', 'min', 'max', 'count', 'rate']).optional(),
  groupBy: z.array(z.string()).optional().describe('Fields to group by'),
  interval: z.string().default('1m').describe('Aggregation interval'),

  // Querying
  query: z.string().optional().describe('PromQL or query string'),
  labels: z.record(z.string()).optional().describe('Label selectors'),

  // Alerts
  alertRules: z.array(z.object({
    name: z.string(),
    condition: z.string(),
    threshold: z.number(),
    duration: z.string(),
  })).optional().describe('Alert rules to evaluate'),

  // General
  format: z.enum(['json', 'prometheus', 'csv', 'graphite']).default('json'),
  timeout: z.number().default(30000),
});

type MetricsCollectorParamsInput = z.input<typeof MetricsCollectorParamsSchema>;
type MetricsCollectorParams = z.output<typeof MetricsCollectorParamsSchema>;

const MetricSchema = z.object({
  name: z.string(),
  type: MetricTypeSchema,
  value: z.number(),
  timestamp: z.string(),
  labels: z.record(z.string()).optional(),
});

const MetricsCollectorResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  metrics: z.array(MetricSchema).optional(),
  series: z.array(z.object({
    name: z.string(),
    labels: z.record(z.string()),
    datapoints: z.array(z.tuple([z.number(), z.number()])),
  })).optional(),
  aggregations: z.record(z.number()).optional(),
  alerts: z.array(z.object({
    name: z.string(),
    state: z.enum(['firing', 'resolved', 'pending']),
    value: z.number(),
    condition: z.string(),
  })).optional(),
  summary: z.object({
    totalMetrics: z.number(),
    uniqueSeries: z.number(),
    datapoints: z.number(),
    timeRange: z.tuple([z.string(), z.string()]),
  }).optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type MetricsCollectorResult = z.output<typeof MetricsCollectorResultSchema>;

export class MetricsCollectorTool {
  private http: HttpBubble;
  private params: MetricsCollectorParams;
  private context?: BubbleContext;

  constructor(params: MetricsCollectorParamsInput, context?: BubbleContext) {
    this.params = MetricsCollectorParamsSchema.parse(params);
    this.context = context;

    this.http = new HttpBubble({
      url: 'http://localhost:9090', // Prometheus default
      method: 'GET',
      timeout: this.params.timeout,
    }, context);
  }

  private async prometheusQuery(endpoint: string, params: Record<string, string>): Promise<any> {
    const url = new URL(endpoint, 'http://localhost:9090');
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, value);
    });

    const response = await fetch(url.toString());
    return await response.json();
  }

  public async collect(): Promise<MetricsCollectorResult> {
    const startTime = Date.now();

    try {
      const metrics: any[] = [];

      if (this.params.services) {
        for (const service of this.params.services) {
          const query = `{service="${service}"}`;
          const result = await this.prometheusQuery('/api/v1/query', { query });

          if (result.data?.result) {
            for (const series of result.data.result) {
              metrics.push({
                name: series.metric.__name__,
                type: 'gauge',
                value: parseFloat(series.value[1]),
                timestamp: new Date(parseFloat(series.value[0]) * 1000).toISOString(),
                labels: series.metric,
              });
            }
          }
        }
      }

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'collect',
        metrics,
        summary: {
          totalMetrics: metrics.length,
          uniqueSeries: metrics.length,
          datapoints: metrics.length,
          timeRange: [
            new Date(Date.now() - 60000).toISOString(),
            new Date().toISOString(),
          ],
        },
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'collect',
        error: errorMessage,
        timing,
      };
    }
  }

  public async query(): Promise<MetricsCollectorResult> {
    if (!this.params.query) {
      throw new Error('query is required for query operation');
    }

    const startTime = Date.now();

    try {
      const timeRange = this.params.timeRange?.duration || '1h';
      const result = await this.prometheusQuery('/api/v1/query_range', {
        query: this.params.query,
        start: this.params.timeRange?.start || Math.floor((Date.now() - 3600000) / 1000).toString(),
        end: this.params.timeRange?.end || Math.floor(Date.now() / 1000).toString(),
        step: this.params.interval || '60',
      });

      const series: any[] = [];

      if (result.data?.result) {
        for (const s of result.data.result) {
          const datapoints = s.values.map((v: [string, string]) => [
            parseFloat(v[0]),
            parseFloat(v[1]),
          ]);

          series.push({
            name: s.metric.__name__,
            labels: s.metric,
            datapoints,
          });
        }
      }

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'query',
        series,
        summary: {
          totalMetrics: series.length,
          uniqueSeries: series.length,
          datapoints: series.reduce((sum, s) => sum + s.datapoints.length, 0),
          timeRange: [
            new Date(parseFloat(result.data?.result?.[0]?.values?.[0]?.[0] || '0') * 1000).toISOString(),
            new Date().toISOString(),
          ],
        },
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'query',
        error: errorMessage,
        timing,
      };
    }
  }

  public async aggregate(): Promise<MetricsCollectorResult> {
    const startTime = Date.now();

    try {
      let query = this.params.query || '';

      if (this.params.aggregation && query) {
        const aggregationFuncs: Record<string, string> = {
          sum: 'sum',
          avg: 'avg',
          min: 'min',
          max: 'max',
          count: 'count',
          rate: 'rate',
        };

        const func = aggregationFuncs[this.params.aggregation];
        if (func) {
          if (this.params.groupBy && this.params.groupBy.length > 0) {
            query = `${func} by (${this.params.groupBy.join(',')}) (${query})`;
          } else {
            query = `${func}(${query})`;
          }
        }
      }

      return this.query();
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'aggregate',
        error: errorMessage,
        timing,
      };
    }
  }

  public async alert(): Promise<MetricsCollectorResult> {
    if (!this.params.alertRules) {
      throw new Error('alertRules is required for alert operation');
    }

    const startTime = Date.now();

    try {
      const alerts: any[] = [];

      for (const rule of this.params.alertRules) {
        const result = await this.prometheusQuery('/api/v1/query', {
          query: rule.condition,
        });

        if (result.data?.result?.[0]?.value) {
          const value = parseFloat(result.data.result[0].value[1]);
          const firing = value > rule.threshold;

          alerts.push({
            name: rule.name,
            state: firing ? 'firing' : 'resolved',
            value,
            condition: rule.condition,
          });
        }
      }

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'alert',
        alerts,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'alert',
        error: errorMessage,
        timing,
      };
    }
  }

  public async action(): Promise<MetricsCollectorResult> {
    switch (this.params.operation) {
      case 'collect':
        return this.collect();
      case 'query':
        return this.query();
      case 'aggregate':
        return this.aggregate();
      case 'alert':
        return this.alert();
      default:
        return {
          success: false,
          operation: this.params.operation,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default MetricsCollectorTool;
