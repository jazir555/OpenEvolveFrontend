import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const WorkflowLifecycleOperationSchema = z.enum(['stop', 'pause', 'resume']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const OpenEvolveV1WorkflowLifecycleParamsSchema = z.object({
  operation: WorkflowLifecycleOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  workflow_id: z.string().min(1),
});

type OpenEvolveV1WorkflowLifecycleParams = z.input<typeof OpenEvolveV1WorkflowLifecycleParamsSchema> & ServiceBubbleParams;

const OpenEvolveV1WorkflowLifecycleResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveV1WorkflowLifecycleResult = z.output<typeof OpenEvolveV1WorkflowLifecycleResultSchema> & BubbleOperationResult;

export class OpenEvolveV1WorkflowLifecycleBubble extends ServiceBubble<
  OpenEvolveV1WorkflowLifecycleParams,
  OpenEvolveV1WorkflowLifecycleResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-v1-workflow-lifecycle' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveV1WorkflowLifecycleParamsSchema;
  static readonly resultSchema = OpenEvolveV1WorkflowLifecycleResultSchema;
  static readonly shortDescription = 'OpenEvolve v1 workflow lifecycle bubble';
  static readonly longDescription = `
    Control an OpenEvolve v1 workflow lifecycle: stop, pause, or resume.
  `;
  static readonly alias = 'openevolve-v1-workflow-lifecycle';

  constructor(params: OpenEvolveV1WorkflowLifecycleParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveV1WorkflowLifecycleResult> {
    const startTime = Date.now();
    try {
      const op = this.params.operation;
      return await this.request('POST', `/api/v1/workflows/${this.requireParam('workflow_id')}/${op}`, undefined, startTime);
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

  private requireParam(key: 'workflow_id'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required`);
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
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveV1WorkflowLifecycleResult> {
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

export default OpenEvolveV1WorkflowLifecycleBubble;
