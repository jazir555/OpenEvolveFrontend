import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OperationSchema = z.enum(['fix']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const DspyFixParamsSchema = z.object({
  operation: OperationSchema.default('fix'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  content: z.string().optional(),
  code: z.string().optional(),
  language: z.string().optional(),
  issues: z.array(z.record(z.unknown())).optional(),
  assessment: z.record(z.unknown()).optional(),
  instructions: z.string().optional(),
  model: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  options: z.record(z.unknown()).optional(),
});

type DspyFixParams = z.input<typeof DspyFixParamsSchema> & ServiceBubbleParams;

const DspyFixResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type DspyFixResult = z.output<typeof DspyFixResultSchema> &
  BubbleOperationResult;

export class OpenEvolveDspyFixBubble extends ServiceBubble<
  DspyFixParams,
  DspyFixResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-dspy-fix' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = DspyFixParamsSchema;
  static readonly resultSchema = DspyFixResultSchema;
  static readonly shortDescription =
    'OpenEvolve DSPy remediation bubble that repairs assessed issues';
  static readonly longDescription = `
    Applies DSPy-generated fixes to content or code (POST /api/openevolve/fix/dspy),
    optionally driven by a prior assessment payload or explicit issue list.
  `;
  static readonly alias = 'openevolve-dspy-fix';

  constructor(params: DspyFixParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<DspyFixResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'fix': {
          if (!this.params.content && !this.params.code) {
            throw new Error('content or code is required for fix');
          }
          return await this.request(
            'POST',
            '/api/openevolve/fix/dspy',
            {
              content: this.params.content,
              code: this.params.code,
              language: this.params.language,
              issues: this.params.issues,
              assessment: this.params.assessment,
              instructions: this.params.instructions,
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
  ): Promise<DspyFixResult> {
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

export default OpenEvolveDspyFixBubble;
