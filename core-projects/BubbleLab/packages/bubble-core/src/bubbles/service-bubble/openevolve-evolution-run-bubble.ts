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

const OpenEvolveEvolutionRunParamsSchema = z.object({
  operation: z.enum(['start', 'list', 'get', 'stop']).default('start'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  run_id: z.string().optional(),
  problem_statement: z.string().optional(),
  config: z.record(z.unknown()).optional(),
});

type OpenEvolveEvolutionRunParams = z.input<
  typeof OpenEvolveEvolutionRunParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveEvolutionRunResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveEvolutionRunResult = z.output<
  typeof OpenEvolveEvolutionRunResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveEvolutionRunBubble extends ServiceBubble<
  OpenEvolveEvolutionRunParams,
  OpenEvolveEvolutionRunResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-evolution-run' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveEvolutionRunParamsSchema;
  static readonly resultSchema = OpenEvolveEvolutionRunResultSchema;
  static readonly shortDescription = 'OpenEvolve evolution run service bubble';
  static readonly longDescription = `
    Manages OpenEvolve evolution runs: start a new run, list runs, get a run,
    or stop a running run.
  `;
  static readonly alias = 'openevolve-evolution-run';

  constructor(params: OpenEvolveEvolutionRunParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveEvolutionRunResult> {
    const startTime = Date.now();
    try {
      switch ((this.params.operation as string)) {
        case 'start':
          return await this.request(
            'POST',
            '/api/evolution/runs',
            {
              problem_statement: this.params.problem_statement,
              config: this.params.config,
            },
            startTime
          );
        case 'list':
          return await this.request('GET', '/api/evolution/runs', undefined, startTime);
        case 'get': {
          const id = this.requireRunId();
          return await this.request(
            'GET',
            `/api/evolution/runs/${encodeURIComponent(id)}`,
            undefined,
            startTime
          );
        }
        case 'stop': {
          const id = this.requireRunId();
          return await this.request(
            'POST',
            `/api/evolution/runs/${encodeURIComponent(id)}/stop`,
            undefined,
            startTime
          );
        }
        default:
          return {
            success: false,
            operation: (this.params.operation as string),
            error: `Unsupported operation: ${(this.params.operation as string)}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: (this.params.operation as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireRunId(): string {
    const id = this.params.run_id;
    if (!id) {
      throw new Error('run_id is required for this operation');
    }
    return id;
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
  ): Promise<OpenEvolveEvolutionRunResult> {
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

export default OpenEvolveEvolutionRunBubble;
