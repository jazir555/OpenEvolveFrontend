import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const LeanAideOperationSchema = z.enum([
  'health_check',
  'generate_proof',
  'verify_proof',
  'translate_theorem',
  'elaborate',
  'get_models',
  'math_query',
]);

const LeanAideModelSchema = z.enum([
  'gpt-4',
  'gpt-4-turbo',
  'gpt-3.5-turbo',
  'claude-3-opus',
  'claude-3-sonnet',
  'gemini-pro',
  'openrouter',
]);

const LeanAideParamsSchema = z.object({
  operation: LeanAideOperationSchema,
  base_url: z.string().url(),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  theorem: z.string().optional(),
  proof_attempt: z.string().optional(),
  model: LeanAideModelSchema.default('gpt-4'),
  temperature: z.number().min(0).max(2).default(0.7),

  code: z.string().optional(),

  statement: z.string().optional(),
  formalization_type: z.enum(['theorem', 'definition']).optional(),

  query: z.string().optional(),
  context: z.string().optional(),

  options: z.record(z.unknown()).optional(),
});

type LeanAideParams = z.input<typeof LeanAideParamsSchema>;

const LeanAideResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type LeanAideResult = z.output<typeof LeanAideResultSchema>;

export class OpenEvolveLeanAideBubble extends ServiceBubble<
  LeanAideParams,
  LeanAideResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-leanaide' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = LeanAideParamsSchema;
  static readonly resultSchema = LeanAideResultSchema;
  static readonly shortDescription = 'OpenEvolve LeanAide theorem prover bubble';
  static readonly longDescription = `
    LeanAide integration for proof generation, verification, and formalization.
  `;
  static readonly alias = 'openevolve-leanaide';

  constructor(params: LeanAideParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<LeanAideResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'health_check':
          return await this.request('GET', '/health', undefined, startTime);
        case 'generate_proof':
          return await this.request('POST', '/proof', {
            theorem: this.requireParam('theorem'),
            proof_attempt: this.params.proof_attempt,
            model: this.params.model,
            temperature: this.params.temperature,
            options: this.params.options,
          }, startTime);
        case 'verify_proof':
          return await this.request('POST', '/verify', {
            code: this.requireParam('code'),
            options: this.params.options,
          }, startTime);
        case 'translate_theorem':
          return await this.request('POST', '/translate', {
            statement: this.requireParam('statement'),
            formalization_type: this.params.formalization_type,
            options: this.params.options,
          }, startTime);
        case 'elaborate':
          return await this.request('POST', '/elaborate', {
            code: this.requireParam('code'),
            options: this.params.options,
          }, startTime);
        case 'math_query':
          return await this.request('POST', '/query', {
            query: this.requireParam('query'),
            context: this.params.context,
            options: this.params.options,
          }, startTime);
        case 'get_models':
          return {
            success: true,
            operation: this.params.operation,
            data: {
              models: [
                { provider: 'OpenAI', models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'] },
                { provider: 'Anthropic', models: ['claude-3-opus', 'claude-3-sonnet'] },
                { provider: 'Google', models: ['gemini-pro'] },
                { provider: 'OpenRouter', models: ['openai/gpt-4', 'anthropic/claude-3-opus'] },
              ],
            },
            timing: Date.now() - startTime,
          };
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

  private requireParam(key: 'theorem' | 'code' | 'statement' | 'query'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required for ${this.params.operation}`);
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
  ): Promise<LeanAideResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url.replace(/\/$/, '')}${endpoint}`;

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
        error: response.ok ? undefined : data?.error || response.statusText,
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

export default OpenEvolveLeanAideBubble;
