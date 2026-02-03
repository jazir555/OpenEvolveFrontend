import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const DecompositionOperationSchema = z.enum(['plan', 'health']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_DECOMPOSITION_API_URL ||
        process.env.OPENEVOLVE_API_URL ||
        process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8001';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: DecompositionOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const DecompositionPlanSchema = BaseParamsSchema.extend({
  operation: z.literal('plan'),
  problem_statement: z.string().min(1),
  title: z.string().optional(),
  strategy: z.string().optional(),
  enable_adaptive_selection: z.boolean().optional(),
  maker_config: z.record(z.unknown()).optional(),
  openevolve_client_config: z.record(z.unknown()).optional(),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveDecompositionParamsSchema = z.discriminatedUnion('operation', [
  DecompositionPlanSchema,
  HealthSchema,
]);

type OpenEvolveDecompositionParams = z.input<typeof OpenEvolveDecompositionParamsSchema>;

const OpenEvolveDecompositionResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveDecompositionResult = z.output<typeof OpenEvolveDecompositionResultSchema>;

export class OpenEvolveDecompositionBubble extends ServiceBubble<
  OpenEvolveDecompositionParams,
  OpenEvolveDecompositionResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-decomposition' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveDecompositionParamsSchema;
  static readonly resultSchema = OpenEvolveDecompositionResultSchema;
  static readonly shortDescription = 'OpenEvolve decomposition planning (ProblemAnalyzer + DecompositionEngine)';
  static readonly longDescription = `
    OpenEvolve decomposition bubble for generating decomposition plans.

    Operations:
    - plan: analyze a problem and generate a decomposition plan
  `;
  static readonly alias = 'openevolve-decomposition';

  constructor(
    params: OpenEvolveDecompositionParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected chooseCredential(): string | undefined {
    return undefined;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveDecompositionResult> {
    const startTime = Date.now();
    const params = this.params;

    try {
      switch (params.operation) {
        case 'plan':
          return await this.request('POST', '/api/decomposition/plan', {
            problem_statement: params.problem_statement,
            title: params.title,
            strategy: params.strategy,
            enable_adaptive_selection: params.enable_adaptive_selection,
            maker_config: params.maker_config,
            openevolve_client_config: params.openevolve_client_config,
          }, startTime);
        case 'health':
          return await this.request('GET', '/health', undefined, startTime);
        default:
          return {
            success: false,
            status: 400,
            operation: params.operation,
            error: `Unsupported operation: ${params.operation}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 500,
        operation: params.operation,
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
      const token = this.params.auth_token;
      if (headerName.toLowerCase() === 'authorization' && !token.startsWith('Bearer ')) {
        headers[headerName] = `Bearer ${token}`;
      } else {
        headers[headerName] = token;
      }
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveDecompositionResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body !== undefined && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const text = await response.text();
      let data: unknown = undefined;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
      }

      return {
        success: response.ok,
        status: response.status,
        operation: this.params.operation,
        data,
        error: response.ok
          ? undefined
          : typeof data === 'object' && data && 'detail' in data
            ? String((data as any).detail)
            : response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 0,
        operation: this.params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveDecompositionBubble;
