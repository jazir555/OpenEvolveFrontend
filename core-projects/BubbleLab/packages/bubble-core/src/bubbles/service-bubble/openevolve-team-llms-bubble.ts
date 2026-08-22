import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const TeamLlmsOperationSchema = z.enum(['catalog', 'providers']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const TeamLlmsParamsSchema = z.object({
  operation: TeamLlmsOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

type TeamLlmsParams = z.input<typeof TeamLlmsParamsSchema> & ServiceBubbleParams;

const TeamLlmsResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type TeamLlmsResult = z.output<typeof TeamLlmsResultSchema> & BubbleOperationResult;

export class OpenEvolveTeamLlmsBubble extends ServiceBubble<
  TeamLlmsParams,
  TeamLlmsResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-team-llms' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = TeamLlmsParamsSchema;
  static readonly resultSchema = TeamLlmsResultSchema;
  static readonly shortDescription = 'OpenEvolve team LLM catalog bubble';
  static readonly longDescription = `
    Retrieve the OpenEvolve enhanced team LLM catalog and providers.
  `;
  static readonly alias = 'openevolve-team-llms';

  constructor(params: TeamLlmsParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<TeamLlmsResult> {
    const startTime = Date.now();
    const op = (this.params.operation as string) as string;
    try {
      switch (op) {
        case 'catalog':
          return await this.request(
            'GET',
            '/api/teams-enhanced/llms/catalog',
            undefined,
            startTime
          );
        case 'providers':
          return await this.request(
            'GET',
            '/api/teams-enhanced/llms/providers',
            undefined,
            startTime
          );
        default:
          return {
            success: false,
            operation: op,
            error: `Unsupported operation: ${op}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: op,
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
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<TeamLlmsResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body && method !== 'GET' && method !== 'DELETE' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok ? undefined : (data as any)?.error || response.statusText,
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

export default OpenEvolveTeamLlmsBubble;
