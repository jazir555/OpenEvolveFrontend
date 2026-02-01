import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const DeterminismOperationSchema = z.enum(['generate', 'check', 'health']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_DETERMINISM_API_URL ||
        process.env.OPENEVOLVE_API_URL ||
        process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8001';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: DeterminismOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const DeterminismGenerateSchema = BaseParamsSchema.extend({
  operation: z.literal('generate'),
  prompt: z.string().min(1),
  schema: z.record(z.unknown()).optional(),
  constraints: z.string().optional(),
  context_document: z.string().optional(),
  mode: z.enum(['auto', 'cloud', 'local', 'hybrid', 'consensus']).default('auto'),
  cloud_provider: z.string().optional(),
  cloud_model: z.string().optional(),
  cloud_api_key: z.string().optional(),
  cloud_base_url: z.string().optional(),
  local_provider: z.string().optional(),
  local_model: z.string().optional(),
  local_device: z.string().optional(),
  local_dtype: z.string().optional(),
  config: z.record(z.unknown()).optional(),
  detllm_backend: z.string().optional(),
  detllm_model: z.string().optional(),
});

const DeterminismCheckSchema = BaseParamsSchema.extend({
  operation: z.literal('check'),
  prompt: z.string().min(1),
  tier: z.number().int().min(1).max(5).default(2),
  runs: z.number().int().min(1).max(20).default(3),
  provider: z.string().optional(),
  model: z.string().optional(),
  api_key: z.string().optional(),
  base_url: z.string().optional(),
  detllm_backend: z.string().optional(),
  detllm_model: z.string().optional(),
  device: z.string().optional(),
  dtype: z.string().optional(),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveDeterminismParamsSchema = z.discriminatedUnion('operation', [
  DeterminismGenerateSchema,
  DeterminismCheckSchema,
  HealthSchema,
]);

type OpenEvolveDeterminismParams = z.input<typeof OpenEvolveDeterminismParamsSchema>;

const OpenEvolveDeterminismResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveDeterminismResult = z.output<typeof OpenEvolveDeterminismResultSchema>;

export class OpenEvolveDeterminismBubble extends ServiceBubble<
  OpenEvolveDeterminismParams,
  OpenEvolveDeterminismResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-determinism' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveDeterminismParamsSchema;
  static readonly resultSchema = OpenEvolveDeterminismResultSchema;
  static readonly shortDescription = 'OpenEvolve determinism stack (generate + check)';
  static readonly longDescription = `
    OpenEvolve determinism bubble for deterministic generation and reproducibility checks.

    Operations:
    - generate: run deterministic generation pipeline
    - check: check reproducibility across tiers
  `;
  static readonly alias = 'openevolve-determinism';

  constructor(
    params: OpenEvolveDeterminismParams,
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

  protected async performAction(): Promise<OpenEvolveDeterminismResult> {
    const startTime = Date.now();
    const params = this.params;

    try {
      switch (params.operation) {
        case 'generate':
          return await this.request('POST', '/determinism/generate', {
            prompt: params.prompt,
            schema: params.schema,
            constraints: params.constraints,
            context_document: params.context_document,
            mode: params.mode,
            cloud_provider: params.cloud_provider,
            cloud_model: params.cloud_model,
            cloud_api_key: params.cloud_api_key,
            cloud_base_url: params.cloud_base_url,
            local_provider: params.local_provider,
            local_model: params.local_model,
            local_device: params.local_device,
            local_dtype: params.local_dtype,
            config: params.config,
            detllm_backend: params.detllm_backend,
            detllm_model: params.detllm_model,
          }, startTime);
        case 'check':
          return await this.request('POST', '/determinism/check', {
            prompt: params.prompt,
            tier: params.tier,
            runs: params.runs,
            provider: params.provider,
            model: params.model,
            api_key: params.api_key,
            base_url: params.base_url,
            detllm_backend: params.detllm_backend,
            detllm_model: params.detllm_model,
            device: params.device,
            dtype: params.dtype,
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
  ): Promise<OpenEvolveDeterminismResult> {
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

export default OpenEvolveDeterminismBubble;
