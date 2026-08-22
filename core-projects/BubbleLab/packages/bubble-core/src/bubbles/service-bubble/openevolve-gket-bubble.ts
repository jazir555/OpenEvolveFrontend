import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const GKETOperationSchema = z.enum(['health_check', 'extract']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.GKET_API_URL || process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8766';
  return base.replace(/\/$/, '');
};

const GKETParamsSchema = z.object({
  operation: GKETOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(180000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  // extract fields
  case: z.number().int().min(0).max(2).default(0),
  llm: z.string().default('openai'),
  text_or_file_ref: z.string().default(''),
  instruction: z.string().optional(),
  use_case: z.string().default('GketApiRun'),
  model_schema: z.record(z.unknown()).optional(),
});

type GKETParams = z.input<typeof GKETParamsSchema> & ServiceBubbleParams;

const GKETResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type GKETResult = z.output<typeof GKETResultSchema> & BubbleOperationResult;

export class OpenEvolveGKETBubble extends ServiceBubble<GKETParams, GKETResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'none' as const;
  static readonly bubbleName: BubbleName = 'openevolve-gket' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = GKETParamsSchema;
  static readonly resultSchema = GKETResultSchema;
  static readonly shortDescription = 'OpenEvolve Generic Knowledge Extraction Tool (GKET)';
  static readonly longDescription = `
    Generic Knowledge Extraction Tool. Wraps the GKET server (/healthz, /extract)
    for single-type (case 0), classification-routing (case 1) and hierarchical (case 2) extraction.
  `;
  static readonly alias = 'openevolve-gket';

  constructor(params: GKETParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<GKETResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'health_check':
          return await this.request('GET', '/healthz', undefined, startTime);
        case 'extract':
          return await this.request(
            'POST',
            '/extract',
            {
              case: this.params.case,
              llm: this.params.llm,
              text_or_file_ref: this.params.text_or_file_ref,
              instruction: this.params.instruction,
              use_case: this.params.use_case,
              model_schema: this.params.model_schema,
            },
            startTime
          );
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
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.params.headers) Object.assign(headers, this.params.headers);
    if (this.params.auth_token) {
      const headerName = this.params.auth_header || 'Authorization';
      const token = this.params.auth_token;
      headers[headerName] =
        headerName.toLowerCase() === 'authorization' && !token.startsWith('Bearer ')
          ? `Bearer ${token}`
          : token;
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<GKETResult> {
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
        error: response.ok ? undefined : ((data as any)?.error as string) || response.statusText,
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

export default OpenEvolveGKETBubble;
