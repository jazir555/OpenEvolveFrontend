/**
 * Z3 Prover Service Bubble
 *
 * Provides SMT (Satisfiability Modulo Theories) solving capabilities through Z3.
 * Integrates with the Z3 server (port 7655) for constraint solving, optimization,
 * simplification, and tactic application.
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

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

const Z3SMTLogicSchema = z.enum([
  'AUFLIRA', 'AUFLIRF', 'AUFNIRA', 'BV', 'BVREF',
  'HORN', 'LIA', 'LRA', 'NIA', 'NRA', 'QF_ABV',
  'QF_AUFBV', 'QF_AUFLIA', 'QF_BV', 'QF_IDL',
  'QF_LIA', 'QF_LRA', 'QF_NIA', 'QF_NRA',
  'QF_UF', 'QF_UFBV', 'UFLRA', 'UF', 'UFBV',
  'QF_AX', 'QF_S', 'SMT', 'ALL',
]);

const Z3ParamsSchema = z.object({
  operation: Z3OperationSchema.describe('Z3 operation to execute'),

  // Server configuration
  baseUrl: z.string().url().default('http://localhost:7655')
    .describe('Z3 server URL'),
  timeout: z.number().min(1000).max(600000).default(30000)
    .describe('Request timeout in milliseconds'),

  // SMT solving
  smtlib2: z.string().optional().describe('SMTLIB2 commands to solve'),
  logic: Z3SMTLogicSchema.optional().describe('Optional logic specification'),

  // Optimization
  objectives: z.array(z.object({
    expression: z.string(),
    type: z.enum(['maximize', 'minimize']),
  })).optional().describe('Optimization objectives'),

  constraints: z.array(z.string()).optional().describe('Optional constraint strings'),

  // Simplification
  expression: z.string().optional().describe('Expression to simplify'),
  assumptions: z.array(z.string()).optional().describe('Optional assumptions'),

  // Tactics
  goal: z.string().optional().describe('Goal expression'),
  tactic: z.string().optional().describe('Tactic name'),
  tacticParams: z.record(z.unknown()).optional().describe('Tactic parameters'),

  // Fixedpoint
  rules: z.array(z.string()).optional().describe('Fixedpoint rules'),
  query: z.string().optional().describe('Fixedpoint query'),

  // Additional options
  options: z.record(z.unknown()).optional().describe('Additional operation-specific options'),
});

export type Z3ParamsInput = z.input<typeof Z3ParamsSchema>;
export type Z3Params = z.output<typeof Z3ParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const Z3ResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
  metadata: z.record(z.unknown()).optional(),
});

export type Z3Result = z.output<typeof Z3ResultSchema>;

// ============================================================================
// Z3PROVER BUBBLE
// ============================================================================

export class Z3ProverBubble extends ServiceBubble<Z3Params, Z3Result> {
  static readonly service = 'openevolve';
  static readonly authType = null as const; // No auth needed (local library)
  static readonly bubbleName = 'z3prover' as const;
  static readonly type = 'service' as const;
  static readonly schema = Z3ParamsSchema;
  static readonly resultSchema = Z3ResultSchema;
  static readonly credentialType = null as const;

  static readonly shortDescription = 'Z3 SMT solver integration';
  static readonly longDescription = `
    Z3 Prover service bubble for SMT solving.

    Features:
    - SMT solving (SAT/UNSAT/UNKNOWN)
    - Optimization (maximize/minimize objectives)
    - Expression simplification
    - Tactic application
    - Fixedpoint computation
    - Support for multiple theories (Booleans, Integers, Reals, Bit-vectors, Arrays)
  `;

  private resilience: ResilienceWrapper;

  constructor(params: Z3ParamsInput, context?: BubbleContext) {
    super(params, context);
    this.resilience = new ResilienceWrapper('z3prover', DEFAULT_RESILIENCE_CONFIG);
  }

  private buildHeaders(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
    };
  }

  private buildUrl(endpoint: string): string {
    const base = this.params.baseUrl.endsWith('/')
      ? this.params.baseUrl.slice(0, -1)
      : this.params.baseUrl;
    return `${base}${endpoint}`;
  }

  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown,
    timeout?: number
  ): Promise<Response> {
    const requestTimeout = timeout || this.params.timeout;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), requestTimeout);

    try {
      const response = await fetch(this.buildUrl(endpoint), {
        method,
        headers: this.buildHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error(`Z3 request timeout after ${requestTimeout}ms`);
      }
      throw error;
    }
  }

  private async healthCheck(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'z3-health',
        () => this.makeRequest('GET', '/health'),
        { operation: 'health_check' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.status === 'ok',
        operation: 'health_check',
        data: data.z3_available ? { available: true, version: data.version } : undefined,
        error: response.ok ? undefined : `Health check failed: ${response.status}`,
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'health_check',
        error: error.message || 'Health check failed',
        timing: Date.now() - startTime,
      };
    }
  }

  private async solveSMT(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const requestBody = {
        smtlib2: this.params.smtlib2,
        timeout: this.params.timeout,
        logic: this.params.logic,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'z3-solve',
        () => this.makeRequest('POST', '/solve', requestBody),
        { operation: 'solve_smt' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.result !== 'error',
        operation: 'solve_smt',
        data: {
          result: data.result,
          model: data.model,
          statistics: data.statistics,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'solve_smt',
        error: error.message || 'Failed to solve SMT problem',
        timing: Date.now() - startTime,
      };
    }
  }

  private async optimize(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const requestBody = {
        objectives: this.params.objectives,
        constraints: this.params.constraints,
        timeout: this.params.timeout,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'z3-optimize',
        () => this.makeRequest('POST', '/optimize', requestBody),
        { operation: 'optimize' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.status !== 'error',
        operation: 'optimize',
        data: {
          status: data.status,
          model: data.model,
          objectiveValues: data.objective_values,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'optimize',
        error: error.message || 'Failed to solve optimization problem',
        timing: Date.now() - startTime,
      };
    }
  }

  private async simplify(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const requestBody = {
        expression: this.params.expression,
        assumptions: this.params.assumptions,
        timeout: this.params.timeout,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'z3-simplify',
        () => this.makeRequest('POST', '/simplify', requestBody),
        { operation: 'simplify' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'simplify',
        data: {
          result: data.result,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'simplify',
        error: error.message || 'Failed to simplify expression',
        timing: Date.now() - startTime,
      };
    }
  }

  private async applyTactic(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const requestBody = {
        goal: this.params.goal,
        tactic: this.params.tactic,
        params: this.params.tacticParams,
        timeout: this.params.timeout,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'z3-tactic',
        () => this.makeRequest('POST', '/tactic', requestBody),
        { operation: 'apply_tactic' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.status !== 'error',
        operation: 'apply_tactic',
        data: {
          status: data.status,
          goals: data.goals,
          model: data.model,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'apply_tactic',
        error: error.message || 'Failed to apply tactic',
        timing: Date.now() - startTime,
      };
    }
  }

  private async fixedpointQuery(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const requestBody = {
        rules: this.params.rules,
        query: this.params.query,
        timeout: this.params.timeout,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'z3-fixedpoint',
        () => this.makeRequest('POST', '/fixedpoint', requestBody),
        { operation: 'fixedpoint_query' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.result !== 'error',
        operation: 'fixedpoint_query',
        data: {
          result: data.result,
          answer: data.answer,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'fixedpoint_query',
        error: error.message || 'Failed to execute fixedpoint query',
        timing: Date.now() - startTime,
      };
    }
  }

  private async getTactics(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'z3-tactics',
        () => this.makeRequest('GET', '/tactics', {}, 5000),
        { operation: 'get_tactics' }
      );

      const timing = Date.now() - startTime;
      const data: any = await response.json();

      return {
        success: response.ok,
        operation: 'get_tactics',
        data: { tactics: data.tactics },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'get_tactics',
        error: error.message || 'Failed to get tactics',
        timing: Date.now() - startTime,
      };
    }
  }

  private async getLogics(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'z3-logics',
        () => this.makeRequest('GET', '/logics', {}, 5000),
        { operation: 'get_logics' }
      );

      const timing = Date.now() - startTime;
      const data: any = await response.json();

      return {
        success: response.ok,
        operation: 'get_logics',
        data: { logics: data.logics },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'get_logics',
        error: error.message || 'Failed to get logics',
        timing: Date.now() - startTime,
      };
    }
  }

  private async getVersion(): Promise<Z3Result> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'z3-version',
        () => this.makeRequest('GET', '/version', {}, 5000),
        { operation: 'get_version' }
      );

      const timing = Date.now() - startTime;
      const data: any = await response.json();

      return {
        success: response.ok,
        operation: 'get_version',
        data: { version: data.version },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'get_version',
        error: error.message || 'Failed to get version',
        timing: Date.now() - startTime,
      };
    }
  }

  // ============================================================================
  // SERVICE BUBBLE EXECUTION
  // ============================================================================

  async execute(): Promise<Z3Result> {
    const operation = this.params.operation;

    switch (operation) {
      case 'health_check':
        return await this.healthCheck();

      case 'solve_smt':
        if (!this.params.smtlib2) {
          throw new Error('smtlib2 parameter is required for solve_smt operation');
        }
        return await this.solveSMT();

      case 'optimize':
        if (!this.params.objectives || this.params.objectives.length === 0) {
          throw new Error('objectives parameter is required for optimize operation');
        }
        return await this.optimize();

      case 'simplify':
        if (!this.params.expression) {
          throw new Error('expression parameter is required for simplify operation');
        }
        return await this.simplify();

      case 'apply_tactic':
        if (!this.params.goal) {
          throw new Error('goal parameter is required for apply_tactic operation');
        }
        if (!this.params.tactic) {
          throw new Error('tactic parameter is required for apply_tactic operation');
        }
        return await this.applyTactic();

      case 'fixedpoint_query':
        if (!this.params.query) {
          throw new Error('query parameter is required for fixedpoint_query operation');
        }
        return await this.fixedpointQuery();

      case 'get_tactics':
        return await this.getTactics();

      case 'get_logics':
        return await this.getLogics();

      case 'get_version':
        return await this.getVersion();

      default:
        return {
          success: false,
          operation,
          error: `Unknown operation: ${operation}`,
          timing: 0,
        };
    }
  }
}
