import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const Z3OperationSchema = z.enum([
  'health_check',
  'solve_smt',
  'optimize',
  'simplify',
  'apply_tactic',
  'fixedpoint_query',
  'get_tactics',
  'get_logics',
  'get_version',
]);

const Z3LogicSchema = z.enum([
  'AUFLIRA', 'AUFLIRF', 'AUFNIRA', 'BV', 'BVREF',
  'HORN', 'LIA', 'LRA', 'NIA', 'NRA', 'QF_ABV',
  'QF_AUFBV', 'QF_AUFLIA', 'QF_BV', 'QF_IDL',
  'QF_LIA', 'QF_LRA', 'QF_NIA', 'QF_NRA',
  'QF_UF', 'QF_UFBV', 'UFLRA', 'UF', 'UFBV',
  'QF_AX', 'QF_S', 'SMT', 'ALL',
]);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const Z3ParamsSchema = z.object({
  operation: Z3OperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(30000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
  smtlib2: z.string().optional(),
  logic: Z3LogicSchema.optional(),

  objectives: z
    .array(
      z.object({
        expression: z.string(),
        type: z.enum(['maximize', 'minimize']),
      })
    )
    .optional(),
  constraints: z.array(z.string()).optional(),

  expression: z.string().optional(),
  assumptions: z.array(z.string()).optional(),

  goal: z.string().optional(),
  tactic: z.string().optional(),
  tactic_params: z.record(z.unknown()).optional(),

  rules: z.array(z.string()).optional(),
  query: z.string().optional(),

  options: z.record(z.unknown()).optional(),
});

type Z3Params = z.input<typeof Z3ParamsSchema> & ServiceBubbleParams;

const Z3ResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type Z3Result = z.output<typeof Z3ResultSchema> & BubbleOperationResult;

export class OpenEvolveZ3ProverBubble extends ServiceBubble<Z3Params, Z3Result> {
  static readonly service = 'openevolve';
  static readonly authType = 'none' as const;
  static readonly bubbleName: BubbleName = 'openevolve-z3prover' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = Z3ParamsSchema;
  static readonly resultSchema = Z3ResultSchema;
  static readonly shortDescription = 'OpenEvolve Z3 SMT solver bubble';
  static readonly longDescription = `
    Z3 SMT solver integration for constraint solving, optimization, and tactics.
  `;
  static readonly alias = 'openevolve-z3prover';

  constructor(params: Z3Params, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return undefined;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      switch (((this.params.operation as string) as string)) {
        case 'health_check':
          return await this.request('GET', '/health', undefined, startTime);
        case 'solve_smt':
          return await this.request('POST', '/solve', {
            smtlib2: this.requireParam('smtlib2'),
            logic: this.params.logic,
            ...(this.params.options || {}),
          }, startTime);
        case 'optimize':
          return await this.request('POST', '/optimize', {
            objectives: this.params.objectives,
            constraints: this.params.constraints,
            ...(this.params.options || {}),
          }, startTime);
        case 'simplify':
          return await this.request('POST', '/simplify', {
            expression: this.requireParam('expression'),
            assumptions: this.params.assumptions,
            ...(this.params.options || {}),
          }, startTime);
        case 'apply_tactic':
          return await this.request('POST', '/tactic', {
            goal: this.requireParam('goal'),
            tactic: this.requireParam('tactic'),
            params: this.params.tactic_params,
          }, startTime);
        case 'fixedpoint_query':
          return await this.request('POST', '/fixedpoint', {
            rules: this.params.rules,
            query: this.requireParam('query'),
            ...(this.params.options || {}),
          }, startTime);
        case 'get_tactics':
          return await this.request('GET', '/tactics', undefined, startTime);
        case 'get_logics':
          return await this.request('GET', '/logics', undefined, startTime);
        case 'get_version':
          return await this.request('GET', '/version', undefined, startTime);
        default:
          return {
            success: false,
            operation: ((this.params.operation as string) as string),
            error: `Unsupported operation: ${((this.params.operation as string) as string)}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireParam(key: 'smtlib2' | 'expression' | 'goal' | 'tactic' | 'query'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required for ${((this.params.operation as string) as string)}`);
    }
    return value;
  }

  private buildHeaders(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      ...(this.params.headers || {}),
    };
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<Z3Result> {
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

export default OpenEvolveZ3ProverBubble;
