import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const SecurityOperationSchema = z.enum(['roles', 'api-keys', 'audit-logs']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const OpenEvolveSecurityParamsSchema = z.object({
  operation: SecurityOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('X-API-Key'),

  limit: z.number().int().min(1).max(1000).optional(),
});

type OpenEvolveSecurityParams = z.input<typeof OpenEvolveSecurityParamsSchema> & ServiceBubbleParams;

const OpenEvolveSecurityResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveSecurityResult = z.output<typeof OpenEvolveSecurityResultSchema> & BubbleOperationResult;

export class OpenEvolveSecurityBubble extends ServiceBubble<
  OpenEvolveSecurityParams,
  OpenEvolveSecurityResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-security' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveSecurityParamsSchema;
  static readonly resultSchema = OpenEvolveSecurityResultSchema;
  static readonly shortDescription = 'OpenEvolve security bubble (proxies to the :8001 engine)';
  static readonly longDescription = `
    OpenEvolve security operations: roles, api-keys, and audit-logs.
    Proxies to the :8001 security engine. The auth_token is sent as the
    X-API-Key header.
  `;
  static readonly alias = 'openevolve-security';

  constructor(params: OpenEvolveSecurityParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveSecurityResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'roles':
          return await this.request('GET', '/api/security/roles', undefined, startTime);
        case 'api-keys':
          return await this.request('GET', '/api/security/api-keys', undefined, startTime);
        case 'audit-logs':
          return await this.request('GET', '/api/security/audit-logs', undefined, startTime);
        default:
          return {
            success: false,
            operation: this.params.operation,
            error: `Unsupported operation: ${this.params.operation}`,
            timing: Date.now() - startTime,
          };
      }
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

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.headers) {
      Object.assign(headers, this.params.headers);
    }
    if (this.params.auth_token) {
      const headerName = this.params.auth_header || 'X-API-Key';
      headers[headerName] = this.params.auth_token;
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveSecurityResult> {
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
        operation: this.params.operation,
        data,
        error: response.ok ? undefined : (data as any)?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveSecurityBubble;
