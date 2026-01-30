import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const SettingsOperationSchema = z.enum(['get_llm', 'update_llm', 'health']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8001';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: SettingsOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const LLMConfigSchema = z.object({
  provider: z.enum(['openai', 'anthropic', 'cohere', 'custom']).optional(),
  api_key: z.string().optional(),
  base_url: z.string().url().optional(),
  model_leanaide: z.string().optional(),
  model_text: z.string().optional(),
  model_img: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  top_p: z.number().min(0).max(1).optional(),
  max_tokens: z.number().int().min(1).optional(),
  frequency_penalty: z.number().min(-2).max(2).optional(),
  presence_penalty: z.number().min(-2).max(2).optional(),
});

const GetLLMSchema = BaseParamsSchema.extend({
  operation: z.literal('get_llm'),
});

const UpdateLLMSchema = BaseParamsSchema.extend({
  operation: z.literal('update_llm'),
  config: LLMConfigSchema,
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveSettingsParamsSchema = z.discriminatedUnion('operation', [
  GetLLMSchema,
  UpdateLLMSchema,
  HealthSchema,
]);

type OpenEvolveSettingsParams = z.input<typeof OpenEvolveSettingsParamsSchema>;

const OpenEvolveSettingsResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveSettingsResult = z.output<typeof OpenEvolveSettingsResultSchema>;

export class OpenEvolveSettingsBubble extends ServiceBubble<
  OpenEvolveSettingsParams,
  OpenEvolveSettingsResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-settings' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveSettingsParamsSchema;
  static readonly resultSchema = OpenEvolveSettingsResultSchema;
  static readonly shortDescription = 'OpenEvolve settings management (LLM config)';
  static readonly longDescription = `
    OpenEvolve settings bubble for managing LLM configuration.
  `;
  static readonly alias = 'openevolve-settings';

  constructor(
    params: OpenEvolveSettingsParams,
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

  protected async performAction(): Promise<OpenEvolveSettingsResult> {
    const startTime = Date.now();
    const params = this.params;

    try {
      switch (params.operation) {
        case 'get_llm':
          return await this.request('GET', '/api/settings/llm', undefined, startTime);
        case 'update_llm':
          return await this.request('PUT', '/api/settings/llm', params.config, startTime);
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
  ): Promise<OpenEvolveSettingsResult> {
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

export default OpenEvolveSettingsBubble;
