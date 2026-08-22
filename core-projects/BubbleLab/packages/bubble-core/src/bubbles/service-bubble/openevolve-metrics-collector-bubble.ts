import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const MetricsCollectorOperationSchema = z.enum(['collect', 'health_check']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const MetricsCollectorParamsSchema = z.object({
  operation: MetricsCollectorOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('X-API-Key'),

  metrics: z
    .array(
      z.object({
        name: z.string(),
        value: z.union([z.number(), z.string(), z.boolean()]),
      })
    )
    .optional(),
  workflow_id: z.string().optional(),
  fetch_analytics: z.boolean().default(true),
  aggregation: z.enum(['sum', 'avg', 'min', 'max', 'count']).default('avg'),
});

type MetricsCollectorParams = z.input<typeof MetricsCollectorParamsSchema> & ServiceBubbleParams;

const MetricsCollectorDataSchema = z.object({
  count: z.number(),
  numeric_count: z.number(),
  sum: z.number(),
  avg: z.number(),
  min: z.number(),
  max: z.number(),
  aggregation: z.string(),
  analytics: z.record(z.unknown()).optional(),
});

const MetricsCollectorResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: MetricsCollectorDataSchema.optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type MetricsCollectorResult = z.output<typeof MetricsCollectorResultSchema> & BubbleOperationResult;

export class OpenEvolveMetricsCollectorBubble extends ServiceBubble<
  MetricsCollectorParams,
  MetricsCollectorResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-metrics-collector' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = MetricsCollectorParamsSchema;
  static readonly resultSchema = MetricsCollectorResultSchema;
  static readonly shortDescription = 'OpenEvolve metrics collector & aggregator';
  static readonly longDescription = `
    Collects and aggregates metrics. Accepts inline metric samples and/or queries
    the OpenEvolve analytics endpoints (/api/analytics/*). Computes real
    aggregations (sum/avg/min/max/count) over numeric values.
  `;
  static readonly alias = 'openevolve-metrics-collector';

  constructor(params: MetricsCollectorParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<MetricsCollectorResult> {
    const startTime = Date.now();
    try {
      if (this.params.operation === 'health_check') {
        const r = await this.request('GET', '/health', undefined, startTime);
        return {
          success: r.success,
          operation: this.params.operation,
          data: undefined,
          error: r.error,
          timing: Date.now() - startTime,
        };
      }

      const analytics = this.params.fetch_analytics
        ? await this.fetchAnalytics()
        : undefined;

      const provided = (this.params.metrics || []).filter(
        (m) => typeof m.value === 'number'
      ) as Array<{ name: string; value: number }>;
      const values = provided.map((m) => m.value);

      const numericCount = values.length;
      const sum = values.reduce((a, b) => a + b, 0);
      const avg = numericCount > 0 ? sum / numericCount : 0;
      const min = numericCount > 0 ? Math.min(...values) : 0;
      const max = numericCount > 0 ? Math.max(...values) : 0;
      const count = (this.params.metrics || []).length;

      let aggregated = avg;
      if (this.params.aggregation === 'sum') aggregated = sum;
      else if (this.params.aggregation === 'min') aggregated = min;
      else if (this.params.aggregation === 'max') aggregated = max;
      else if (this.params.aggregation === 'count') aggregated = count;

      return {
        success: true,
        operation: this.params.operation,
        data: {
          count,
          numeric_count: numericCount,
          sum,
          avg,
          min,
          max,
          aggregation: this.params.aggregation ?? 'avg',
          analytics,
        },
        timing: Date.now() - startTime,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private async fetchAnalytics(): Promise<Record<string, unknown> | undefined> {
    const endpoints = [
      '/api/analytics/performance-metrics',
      '/api/analytics/workflow-metrics',
      '/api/analytics/knowledge-stats',
    ];
    const out: Record<string, unknown> = {};
    for (const ep of endpoints) {
      const res = await this.request('GET', ep, undefined, Date.now(), true);
      if (res.success) out[ep] = res.data;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }

  private buildHeaders(includeApiKey = false): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.params.headers) Object.assign(headers, this.params.headers);
    if (includeApiKey && this.params.auth_token) {
      const headerName = this.params.auth_header || 'X-API-Key';
      headers[headerName] = this.params.auth_token;
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number,
    includeApiKey = false
  ): Promise<{ success: boolean; data?: unknown; error?: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;
    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(includeApiKey),
        body: body && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);
      return {
        success: response.ok,
        data,
        error: response.ok ? undefined : ((data as any)?.detail as string) || response.statusText,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return { success: false, error: message };
    }
  }
}

export default OpenEvolveMetricsCollectorBubble;
