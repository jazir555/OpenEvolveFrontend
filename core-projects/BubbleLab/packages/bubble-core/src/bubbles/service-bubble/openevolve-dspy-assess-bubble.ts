import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OperationSchema = z.enum(['assess']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const DspyAssessParamsSchema = z.object({
  operation: OperationSchema.default('assess'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  content: z.string().optional(),
  code: z.string().optional(),
  language: z.string().optional(),
  criteria: z.array(z.string()).optional(),
  model: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  options: z.record(z.unknown()).optional(),
});

type DspyAssessParams = z.input<typeof DspyAssessParamsSchema> &
  ServiceBubbleParams;

const DspyAssessResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type DspyAssessResult = z.output<typeof DspyAssessResultSchema> &
  BubbleOperationResult;

export class OpenEvolveDspyAssessBubble extends ServiceBubble<
  DspyAssessParams,
  DspyAssessResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'dspy-assess' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = DspyAssessParamsSchema;
  static readonly resultSchema = DspyAssessResultSchema;
  static readonly shortDescription =
    'OpenEvolve DSPy assessment bubble for content and code quality';
  static readonly longDescription = `
    Runs a DSPy-driven assessment of content or code
    (POST /api/openevolve/assess/dspy), returning scored findings per criterion.
  `;
  static readonly alias = 'openevolve-dspy-assess';

  constructor(params: DspyAssessParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<DspyAssessResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'assess': {
          if (!this.params.content && !this.params.code) {
            throw new Error('content or code is required for assess');
          }
          return await this.request(
            'POST',
            '/api/openevolve/assess/dspy',
            {
              content: this.params.content,
              code: this.params.code,
              language: this.params.language,
              criteria: this.params.criteria,
              model: this.params.model,
              temperature: this.params.temperature,
              options: this.params.options,
            },
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
  ): Promise<DspyAssessResult> {
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

export default OpenEvolveDspyAssessBubble;
