import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OperationSchema = z.enum(['analyze_complexity']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const AdaptiveMdapComplexityParamsSchema = z.object({
  operation: OperationSchema.default('analyze_complexity'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  task: z.string().optional(),
  content: z.string().optional(),
  context: z.string().optional(),
  profile: z.string().optional(),
  options: z.record(z.unknown()).optional(),
});

type AdaptiveMdapComplexityParams = z.input<
  typeof AdaptiveMdapComplexityParamsSchema
> &
  ServiceBubbleParams;

const AdaptiveMdapComplexityResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type AdaptiveMdapComplexityResult = z.output<
  typeof AdaptiveMdapComplexityResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveAdaptiveMdapComplexityBubble extends ServiceBubble<
  AdaptiveMdapComplexityParams,
  AdaptiveMdapComplexityResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-adaptive-mdap-complexity' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = AdaptiveMdapComplexityParamsSchema;
  static readonly resultSchema = AdaptiveMdapComplexityResultSchema;
  static readonly shortDescription =
    'OpenEvolve Adaptive MDAP complexity analysis bubble';
  static readonly longDescription = `
    Scores task complexity for Adaptive MDAP planning (POST /adaptive-mdap/complexity)
    so downstream allocation can pick an appropriate profile.
  `;
  static readonly alias = 'openevolve-adaptive-mdap-complexity';

  constructor(params: AdaptiveMdapComplexityParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<AdaptiveMdapComplexityResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'analyze_complexity':
          return await this.request(
            'POST',
            '/adaptive-mdap/complexity',
            {
              task: this.params.task,
              content: this.params.content,
              context: this.params.context,
              profile: this.params.profile,
              options: this.params.options,
            },
            startTime
          );
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
  ): Promise<AdaptiveMdapComplexityResult> {
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

export default OpenEvolveAdaptiveMdapComplexityBubble;
