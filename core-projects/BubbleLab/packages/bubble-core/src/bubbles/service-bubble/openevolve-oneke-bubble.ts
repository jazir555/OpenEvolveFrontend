import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OneKEOperationSchema = z.enum(['health_check', 'extract']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.ONEKE_API_URL || process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8765';
  return base.replace(/\/$/, '');
};

const OneKEParamsSchema = z.object({
  operation: OneKEOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(180000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  // extract fields
  task: z.enum(['NER', 'RE', 'EE', 'Triple', 'Base']).default('NER'),
  mode: z.enum(['quick', 'agent', 'customized']).default('quick'),
  text: z.string().optional(),
  file_ref: z.string().optional(),
  instruction: z.string().optional(),
  constraint: z.string().optional(),
  model_config: z.record(z.unknown()).optional(),
  api_key: z.string().optional(),
  model_base_url: z.string().optional(),
  model_name: z.string().optional(),
  construct: z.record(z.unknown()).optional(),
});

type OneKEParams = z.input<typeof OneKEParamsSchema> & ServiceBubbleParams;

const OneKEResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OneKEResult = z.output<typeof OneKEResultSchema> & BubbleOperationResult;

export class OpenEvolveOneKEBubble extends ServiceBubble<OneKEParams, OneKEResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'none' as const;
  static readonly bubbleName: BubbleName = 'openevolve-oneke' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OneKEParamsSchema;
  static readonly resultSchema = OneKEResultSchema;
  static readonly shortDescription = 'OpenEvolve OneKE knowledge extraction (schema-based IE)';
  static readonly longDescription = `
    OneKE information extraction. Wraps the OneKE server (/healthz, /extract).
    Supports NER / RE / EE / Triple / Base tasks over raw text or a file ref.
  `;
  static readonly alias = 'openevolve-oneke';

  constructor(params: OneKEParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OneKEResult> {
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
              task: this.params.task,
              mode: this.params.mode,
              text: this.params.text,
              file_ref: this.params.file_ref,
              instruction: this.params.instruction,
              constraint: this.params.constraint,
              model: this.params.model_config,
              api_key: this.params.api_key,
              base_url: this.params.model_base_url,
              model_name: this.params.model_name,
              construct: this.params.construct,
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
  ): Promise<OneKEResult> {
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

export default OpenEvolveOneKEBubble;
