import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const GauntletOperationSchema = z.enum(['run_gauntlet', 'health_check', 'get_capabilities']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const GauntletParamsSchema = z.object({
  operation: GauntletOperationSchema,
  gauntlet_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  gauntlet_type: z.enum(['red', 'blue', 'gold', 'full']).default('full'),
  rounds: z.number().min(1).max(10).default(3),
  difficulty: z.enum(['easy', 'medium', 'hard', 'adaptive']).default('adaptive'),
  pass_threshold: z.number().min(0).max(100).default(70),

  solution: z.union([z.string(), z.record(z.unknown())]).optional(),
  solution_id: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),

  evaluation_criteria: z
    .array(
      z.enum([
        'correctness',
        'completeness',
        'efficiency',
        'clarity',
        'robustness',
        'security',
        'scalability',
        'maintainability',
      ])
    )
    .optional(),
});

type GauntletParams = z.input<typeof GauntletParamsSchema>;

const GauntletResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type GauntletResult = z.output<typeof GauntletResultSchema>;

export class OpenEvolveGauntletTestingBubble extends ServiceBubble<
  GauntletParams,
  GauntletResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-gauntlet-testing' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = GauntletParamsSchema;
  static readonly resultSchema = GauntletResultSchema;
  static readonly shortDescription =
    'OpenEvolve gauntlet testing (red/blue/gold team validation)';
  static readonly longDescription = `
    Runs OpenEvolve gauntlet tests with multi-round evaluation and scoring.
  `;
  static readonly alias = 'openevolve-gauntlet-testing';

  constructor(params: GauntletParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<GauntletResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'run_gauntlet':
          return await this.request('POST', '/api/gauntlets/run', {
            gauntlet_type: this.params.gauntlet_type,
            rounds: this.params.rounds,
            difficulty: this.params.difficulty,
            pass_threshold: this.params.pass_threshold,
            solution: this.params.solution,
            solution_id: this.params.solution_id,
            evaluation_criteria: this.params.evaluation_criteria,
            metadata: this.params.metadata,
          }, startTime);
        case 'health_check':
          return await this.request('GET', '/api/gauntlets/health', undefined, startTime);
        case 'get_capabilities':
          return await this.request('GET', '/api/gauntlets/capabilities', undefined, startTime);
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
  ): Promise<GauntletResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.gauntlet_url}${endpoint}`;

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

export default OpenEvolveGauntletTestingBubble;
