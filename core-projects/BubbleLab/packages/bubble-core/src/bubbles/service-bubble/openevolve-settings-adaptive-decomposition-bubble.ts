import { z } from 'zod';
import type { BubbleOperationResult, BubbleName } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const OpenEvolveSettingsAdaptiveDecompositionOperationSchema = z.enum(['get', 'update']);

const OpenEvolveSettingsAdaptiveDecompositionParamsSchema = z.object({
  operation: OpenEvolveSettingsAdaptiveDecompositionOperationSchema.default('get'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
  settings: z.record(z.unknown()).optional(),
});

type OpenEvolveSettingsAdaptiveDecompositionParams =
  z.input<typeof OpenEvolveSettingsAdaptiveDecompositionParamsSchema> & ServiceBubbleParams;

const OpenEvolveSettingsAdaptiveDecompositionResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveSettingsAdaptiveDecompositionResult =
  z.output<typeof OpenEvolveSettingsAdaptiveDecompositionResultSchema> & BubbleOperationResult;

export class OpenEvolveSettingsAdaptiveDecompositionBubble extends ServiceBubble<
  OpenEvolveSettingsAdaptiveDecompositionParams,
  OpenEvolveSettingsAdaptiveDecompositionResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-settings-adaptive-decomposition' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveSettingsAdaptiveDecompositionParamsSchema;
  static readonly resultSchema = OpenEvolveSettingsAdaptiveDecompositionResultSchema;
  static readonly shortDescription = 'OpenEvolve settings adaptive decomposition bubble';
  static readonly longDescription = `
    Get or update the OpenEvolve adaptive decomposition settings.
  `;
  static readonly alias = 'openevolve-settings-adaptive-decomposition';

  constructor(params: OpenEvolveSettingsAdaptiveDecompositionParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveSettingsAdaptiveDecompositionResult> {
    const startTime = Date.now();
    const operation = this.params.operation as string;
    try {
      switch (operation) {
        case 'get':
          return await this.request('GET', '/api/settings/adaptive-decomposition', undefined, startTime);
        case 'update':
          return await this.request('PUT', '/api/settings/adaptive-decomposition', this.params.settings, startTime);
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
  ): Promise<OpenEvolveSettingsAdaptiveDecompositionResult> {
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
        operation: (this.params.operation as string),
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: (this.params.operation as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveSettingsAdaptiveDecompositionBubble;
