import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const DecompositionExecutionStatusParamsSchema = z.object({
  operation: z.string().default('get_execution_status'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  execution_id: z.string().optional(),
});

type DecompositionExecutionStatusParams = z.input<typeof DecompositionExecutionStatusParamsSchema> & ServiceBubbleParams;

const DecompositionExecutionStatusResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type DecompositionExecutionStatusResult = z.output<typeof DecompositionExecutionStatusResultSchema> & BubbleOperationResult;

export class OpenEvolveDecompositionExecutionStatusBubble extends ServiceBubble<
  DecompositionExecutionStatusParams,
  DecompositionExecutionStatusResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-decomposition-execution-status' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = DecompositionExecutionStatusParamsSchema;
  static readonly resultSchema = DecompositionExecutionStatusResultSchema;
  static readonly shortDescription = 'OpenEvolve decomposition execution status bubble';
  static readonly longDescription = `
    Retrieve the status of an OpenEvolve decomposition execution.
  `;
  static readonly alias = 'openevolve-decomposition-execution-status';

  constructor(params: DecompositionExecutionStatusParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<DecompositionExecutionStatusResult> {
    const startTime = Date.now();
    const operation = ((this.params.operation as string) as string);
    try {
      switch (operation) {
        case 'get_execution_status':
          return await this.request(
            'GET',
            `/api/decomposition/executions/${this.requireParam('execution_id')}/status`,
            undefined,
            startTime
          );
        default:
          return {
            success: false,
            operation,
            error: `Unsupported operation: ${operation}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireParam(key: 'execution_id'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required for ${((this.params.operation as string) as string)}`);
    }
    return value;
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
  ): Promise<DecompositionExecutionStatusResult> {
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
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
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

export default OpenEvolveDecompositionExecutionStatusBubble;
