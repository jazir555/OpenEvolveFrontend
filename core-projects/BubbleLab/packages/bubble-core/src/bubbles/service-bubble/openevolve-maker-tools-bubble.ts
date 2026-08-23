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

const MakerToolsParamsSchema = z.object({
  operation: z.enum(['list', 'get', 'create']),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  tool_id: z.string().optional(),
  tool_spec: z.record(z.unknown()).optional(),

  options: z.record(z.unknown()).optional(),
});

type MakerToolsParams = z.input<typeof MakerToolsParamsSchema> & ServiceBubbleParams;

const MakerToolsResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type MakerToolsResult = z.output<typeof MakerToolsResultSchema> & BubbleOperationResult;

export class OpenEvolveMakerToolsBubble extends ServiceBubble<
  MakerToolsParams,
  MakerToolsResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'maker-tools' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = MakerToolsParamsSchema;
  static readonly resultSchema = MakerToolsResultSchema;
  static readonly shortDescription = 'OpenEvolve Maker tools bubble';
  static readonly longDescription = `
    List, get, and create tools exposed by the Maker service on the OpenEvolve backend.
  `;
  static readonly alias = 'openevolve-maker-tools';

  constructor(params: MakerToolsParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<MakerToolsResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'list':
          return await this.request('GET', '/maker/tools', undefined, startTime);
        case 'get':
          return await this.request(
            'GET',
            `/maker/tools/${this.requireParam('tool_id')}`,
            undefined,
            startTime
          );
        case 'create':
          return await this.request(
            'POST',
            '/maker/tools',
            { tool_spec: this.params.tool_spec },
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

  private requireParam(key: 'tool_id'): string {
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
  ): Promise<MakerToolsResult> {
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

export default OpenEvolveMakerToolsBubble;
