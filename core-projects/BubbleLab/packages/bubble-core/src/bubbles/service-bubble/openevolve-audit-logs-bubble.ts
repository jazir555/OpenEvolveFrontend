import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const AuditLogsOperationSchema = z.enum(['list']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const AuditLogsParamsSchema = z.object({
  operation: AuditLogsOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  limit: z.number().optional(),
  offset: z.number().optional(),
});

type AuditLogsParams = z.input<typeof AuditLogsParamsSchema> & ServiceBubbleParams;

const AuditLogsResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type AuditLogsResult = z.output<typeof AuditLogsResultSchema> & BubbleOperationResult;

export class OpenEvolveAuditLogsBubble extends ServiceBubble<
  AuditLogsParams,
  AuditLogsResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-audit-logs' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = AuditLogsParamsSchema;
  static readonly resultSchema = AuditLogsResultSchema;
  static readonly shortDescription = 'OpenEvolve audit logs bubble';
  static readonly longDescription = `
    Retrieve OpenEvolve audit logs with optional limit and offset.
  `;
  static readonly alias = 'openevolve-audit-logs';

  constructor(params: AuditLogsParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<AuditLogsResult> {
    const startTime = Date.now();
    const op = (this.params.operation as string) as string;
    try {
      switch (op) {
        case 'list': {
          const query = new URLSearchParams();
          if (this.params.limit !== undefined) {
            query.set('limit', String(this.params.limit));
          }
          if (this.params.offset !== undefined) {
            query.set('offset', String(this.params.offset));
          }
          const qs = query.toString();
          const endpoint = `/api/audit/logs${qs.length > 0 ? `?${qs}` : ''}`;
          return await this.request('GET', endpoint, undefined, startTime);
        }
        default:
          return {
            success: false,
            operation: op,
            error: `Unsupported operation: ${op}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: op,
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
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<AuditLogsResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body && method !== 'GET' && method !== 'DELETE' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok ? undefined : (data as any)?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveAuditLogsBubble;
