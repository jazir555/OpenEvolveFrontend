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

const DecompositionTruthPackageParamsSchema = z.object({
  operation: z.string().default('create_truth_package'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  workflow_id: z.string().optional(),
  body: z.record(z.unknown()).optional(),
});

type DecompositionTruthPackageParams = z.input<typeof DecompositionTruthPackageParamsSchema> & ServiceBubbleParams;

const DecompositionTruthPackageResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type DecompositionTruthPackageResult = z.output<typeof DecompositionTruthPackageResultSchema> & BubbleOperationResult;

export class OpenEvolveDecompositionTruthPackageBubble extends ServiceBubble<
  DecompositionTruthPackageParams,
  DecompositionTruthPackageResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-decomposition-truth-package' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = DecompositionTruthPackageParamsSchema;
  static readonly resultSchema = DecompositionTruthPackageResultSchema;
  static readonly shortDescription = 'OpenEvolve decomposition truth package bubble';
  static readonly longDescription = `
    Create a truth package from the decomposition engine for an OpenEvolve workflow.
  `;
  static readonly alias = 'openevolve-decomposition-truth-package';

  constructor(params: DecompositionTruthPackageParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<DecompositionTruthPackageResult> {
    const startTime = Date.now();
    const operation = ((this.params.operation as string) as string);
    try {
      switch (operation) {
        case 'create_truth_package': {
          const body: Record<string, unknown> = {};
          if (this.params.body !== undefined) {
            Object.assign(body, this.params.body);
          }
          return await this.request(
            'POST',
            `/api/workflows/${this.requireParam('workflow_id')}/engine/truth-package`,
            body,
            startTime
          );
        }
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

  private requireParam(key: 'workflow_id'): string {
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
  ): Promise<DecompositionTruthPackageResult> {
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

export default OpenEvolveDecompositionTruthPackageBubble;
