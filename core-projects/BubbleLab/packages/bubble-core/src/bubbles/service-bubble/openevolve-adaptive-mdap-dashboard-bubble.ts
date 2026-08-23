import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OperationSchema = z.enum(['get_dashboard']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const AdaptiveMdapDashboardParamsSchema = z.object({
  operation: OperationSchema.default('get_dashboard'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  include_history: z.boolean().optional(),
  window: z.string().optional(),
  options: z.record(z.unknown()).optional(),
});

type AdaptiveMdapDashboardParams = z.input<
  typeof AdaptiveMdapDashboardParamsSchema
> &
  ServiceBubbleParams;

const AdaptiveMdapDashboardResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type AdaptiveMdapDashboardResult = z.output<
  typeof AdaptiveMdapDashboardResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveAdaptiveMdapDashboardBubble extends ServiceBubble<
  AdaptiveMdapDashboardParams,
  AdaptiveMdapDashboardResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-adaptive-mdap-dashboard' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = AdaptiveMdapDashboardParamsSchema;
  static readonly resultSchema = AdaptiveMdapDashboardResultSchema;
  static readonly shortDescription =
    'OpenEvolve Adaptive MDAP dashboard overview bubble';
  static readonly longDescription = `
    Retrieves the Adaptive MDAP dashboard snapshot (GET /adaptive-mdap/dashboard),
    including allocation summaries, cost telemetry, and active profile state.
  `;
  static readonly alias = 'openevolve-adaptive-mdap-dashboard';

  constructor(
    params: AdaptiveMdapDashboardParams,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<AdaptiveMdapDashboardResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'get_dashboard': {
          const query = new URLSearchParams();
          if (this.params.include_history !== undefined) {
            query.set('include_history', String(this.params.include_history));
          }
          if (this.params.window) {
            query.set('window', this.params.window);
          }
          const suffix = query.toString() ? `?${query.toString()}` : '';
          return await this.request(
            'GET',
            `/adaptive-mdap/dashboard${suffix}`,
            undefined,
            startTime
          );
        }
        default:
          return {
            success: false,
            operation: this.params.operation as string,
            error: `Unsupported operation: ${this.params.operation as string}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation as string,
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
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<AdaptiveMdapDashboardResult> {
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
        operation: this.params.operation as string,
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation as string,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveAdaptiveMdapDashboardBubble;
