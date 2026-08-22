import { z } from 'zod';
import type { BubbleOperationResult, BubbleName } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const OpenEvolveMonitoringLogsParamsSchema = z.object({
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
  service: z.string().optional(),
  limit: z.number().int().min(1).max(10000).optional(),
});

type OpenEvolveMonitoringLogsParams =
  z.input<typeof OpenEvolveMonitoringLogsParamsSchema> & ServiceBubbleParams;

const OpenEvolveMonitoringLogsResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveMonitoringLogsResult =
  z.output<typeof OpenEvolveMonitoringLogsResultSchema> & BubbleOperationResult;

export class OpenEvolveMonitoringLogsBubble extends ServiceBubble<
  OpenEvolveMonitoringLogsParams,
  OpenEvolveMonitoringLogsResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-monitoring-logs' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveMonitoringLogsParamsSchema;
  static readonly resultSchema = OpenEvolveMonitoringLogsResultSchema;
  static readonly shortDescription = 'OpenEvolve monitoring logs bubble';
  static readonly longDescription = `
    Fetch the OpenEvolve monitoring logs, optionally filtered by service and limit.
  `;
  static readonly alias = 'openevolve-monitoring-logs';

  constructor(params: OpenEvolveMonitoringLogsParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveMonitoringLogsResult> {
    const startTime = Date.now();
    try {
      const query = new URLSearchParams();
      if (this.params.service) query.set('service', this.params.service);
      if (this.params.limit !== undefined) query.set('limit', String(this.params.limit));
      const qs = query.toString();
      const endpoint = `/api/monitoring/logs${qs ? `?${qs}` : ''}`;
      return await this.request('GET', endpoint, undefined, startTime);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'get_logs',
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.headers) {
      Object.assign(headers, this.params.headers);
    }
    if (this.params.auth_token) {
      const headerName = this.params.auth_header || 'Authorization';
      headers[headerName] =
        headerName.toLowerCase() === 'authorization' &&
        !this.params.auth_token.startsWith('Bearer ')
          ? `Bearer ${this.params.auth_token}`
          : this.params.auth_token;
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST' | 'PUT',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveMonitoringLogsResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: 'get_logs',
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'get_logs',
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveMonitoringLogsBubble;
