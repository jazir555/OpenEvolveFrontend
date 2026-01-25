/**
 * LeanAide Theorem Prover Service Bubble
 *
 * Provides Lean 4 theorem proving capabilities through the LeanAide server.
 * Integrates with the standalone LeanAide server (port 7654) for proof generation,
 * verification, and formalization tasks.
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

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
  operation: LeanAideOperationSchema.describe('LeanAide operation to execute'),

  // Server configuration
  baseUrl: z.string().url().default('http://localhost:7654')
    .describe('LeanAide server URL'),
  apiKey: z.string().optional().describe('Optional API key for authentication'),
  timeout: z.number().min(1000).max(600000).default(600000).describe('Request timeout in ms'),

  // Proof generation parameters
  theorem: z.string().optional().describe('Theorem statement to prove'),
  proofAttempt: z.string().optional().describe('Existing proof attempt to improve'),
  model: LeanAideModelSchema.default('gpt-4').describe('LLM model to use'),
  temperature: z.number().min(0).max(2).default(0.7).describe('Temperature for generation'),

  // Proof verification parameters
  code: z.string().optional().describe('Lean 4 code to verify'),

  // Translation parameters
  statement: z.string().optional().describe('Mathematical statement to translate'),
  formalizationType: z.enum(['theorem', 'definition']).optional().describe('Type of formalization'),

  // Math query parameters
  query: z.string().optional().describe('Mathematical query'),
  context: z.string().optional().describe('Additional context for query'),

  // Additional parameters
  options: z.record(z.unknown()).optional().describe('Additional operation-specific options'),
});

export type LeanAideParamsInput = z.input<typeof LeanAideParamsSchema>;
export type LeanAideParams = z.output<typeof LeanAideParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const LeanAideProofResultSchema = z.object({
  success: z.boolean(),
  leanCode: z.string().optional(),
  proof: z.string().optional(),
  error: z.string().optional(),
  logs: z.string().optional(),
  responseTime: z.number().optional(),
});

const LeanAideVerificationResultSchema = z.object({
  valid: z.boolean(),
  errors: z.array(z.string()).optional(),
  warnings: z.array(z.string()).optional(),
  tacticState: z.string().optional(),
  goals: z.array(z.string()).optional(),
});

const LeanAideTranslationResultSchema = z.object({
  success: z.boolean(),
  leanCode: z.string().optional(),
  originalStatement: z.string().optional(),
  error: z.string().optional(),
  explanation: z.string().optional(),
});

const LeanAideMathQueryResultSchema = z.object({
  success: z.boolean(),
  answer: z.string().optional(),
  reasoning: z.string().optional(),
  error: z.string().optional(),
  sources: z.array(z.string()).optional(),
});

const LeanAideResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
  metadata: z.record(z.unknown()).optional(),
});

export type LeanAideResult = z.output<typeof LeanAideResultSchema>;

// ============================================================================
// LEANAIDE BUBBLE
// ============================================================================

export class LeanAideBubble extends ServiceBubble<LeanAideParams, LeanAideResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'leanaide' as const;
  static readonly type = 'service' as const;
  static readonly schema = LeanAideParamsSchema;
  static readonly resultSchema = LeanAideResultSchema;
  static readonly credentialType = 'leanaide_api_key' as const;

  static readonly shortDescription = 'Lean 4 theorem prover integration';
  static readonly longDescription = `
    LeanAide service bubble for Lean 4 theorem proving.

    Features:
    - Generate Lean 4 proofs from theorem statements
    - Verify Lean 4 code against the Lean 4 kernel
    - Translate mathematical statements to Lean 4
    - Execute mathematical queries with formal verification
    - Support for multiple LLM backends (GPT-4, Claude, Gemini)
  `;

  private resilience: ResilienceWrapper;

  constructor(params: LeanAideParamsInput, context?: BubbleContext) {
    super(params, context);
    this.resilience = new ResilienceWrapper('leanaide', DEFAULT_RESILIENCE_CONFIG);
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.apiKey) {
      headers['Authorization'] = `Bearer ${this.params.apiKey}`;
    }
    return headers;
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
        throw new Error(`LeanAide request timeout after ${requestTimeout}ms`);
      }
      throw error;
    }
  }

  private async healthCheck(): Promise<LeanAideResult> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'leanaide-health',
        () => this.makeRequest('GET', '/'),
        { operation: 'health_check' }
      );
      const timing = Date.now() - startTime;

      return {
        success: response.ok,
        operation: 'health_check',
        data: response.ok ? { status: 'healthy' } : undefined,
        error: response.ok ? undefined : `Health check failed: ${response.status}`,
        timing,
        metadata: { statusCode: response.status },
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

  private async generateProof(): Promise<LeanAideResult> {
    const startTime = Date.now();
    try {
      const requestBody = {
        task: 'prove_for_formalization',
        theorem: this.params.theorem,
        proof_attempt: this.params.proofAttempt || '',
        model: this.params.model,
        temperature: this.params.temperature,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'leanaide-generate',
        () => this.makeRequest('POST', '/', requestBody),
        { operation: 'generate_proof' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.success !== false,
        operation: 'generate_proof',
        data: {
          leanCode: data.lean_code || data.code,
          proof: data.proof,
          logs: data.logs,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
        metadata: { responseTime: data.response_time },
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'generate_proof',
        error: error.message || 'Failed to generate proof',
        timing: Date.now() - startTime,
      };
    }
  }

  private async verifyProof(): Promise<LeanAideResult> {
    const startTime = Date.now();
    try {
      const requestBody = {
        task: 'elaborate',
        code: this.params.code,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'leanaide-verify',
        () => this.makeRequest('POST', '/', requestBody),
        { operation: 'verify_proof' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.success !== false && !data.error,
        operation: 'verify_proof',
        data: {
          valid: data.success !== false && !data.error,
          errors: data.error ? [data.error] : data.errors,
          warnings: data.warnings,
          tacticState: data.tactic_state,
          goals: data.goals,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'verify_proof',
        error: error.message || 'Failed to verify proof',
        timing: Date.now() - startTime,
      };
    }
  }

  private async translateTheorem(): Promise<LeanAideResult> {
    const startTime = Date.now();
    try {
      const taskType = this.params.formalizationType === 'definition'
        ? 'translate_def'
        : 'translate_thm';

      const requestBody = {
        task: taskType,
        statement: this.params.statement,
        model: this.params.model,
        temperature: this.params.temperature,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'leanaide-translate',
        () => this.makeRequest('POST', '/', requestBody),
        { operation: 'translate_theorem' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.success !== false,
        operation: 'translate_theorem',
        data: {
          leanCode: data.lean_code || data.code,
          originalStatement: this.params.statement,
          explanation: data.explanation,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'translate_theorem',
        error: error.message || 'Failed to translate theorem',
        timing: Date.now() - startTime,
      };
    }
  }

  private async mathQuery(): Promise<LeanAideResult> {
    const startTime = Date.now();
    try {
      const requestBody = {
        task: 'math_query',
        query: this.params.query,
        context: this.params.context,
        model: this.params.model,
        ...(this.params.options || {}),
      };

      const response = await this.resilience.execute(
        'leanaide-math-query',
        () => this.makeRequest('POST', '/', requestBody),
        { operation: 'math_query' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok && data.success !== false,
        operation: 'math_query',
        data: {
          answer: data.answer,
          reasoning: data.reasoning,
          sources: data.sources,
        },
        error: data.error || (response.ok ? undefined : `Server returned ${response.status}`),
        timing,
      };
    } catch (error: any) {
      return {
        success: false,
        operation: 'math_query',
        error: error.message || 'Failed to execute math query',
        timing: Date.now() - startTime,
      };
    }
  }

  private async getModels(): Promise<LeanAideResult> {
    // LeanAide doesn't provide a models endpoint, return static list
    const startTime = Date.now();
    const models = [
      { provider: 'OpenAI', models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'] },
      { provider: 'Anthropic', models: ['claude-3-opus', 'claude-3-sonnet'] },
      { provider: 'Google', models: ['gemini-pro'] },
      { provider: 'OpenRouter', models: ['openai/gpt-4', 'anthropic/claude-3-opus'] },
    ];

    return {
      success: true,
      operation: 'get_models',
      data: { models },
      timing: Date.now() - startTime,
    };
  }

  // ============================================================================
  // SERVICE BUBBLE EXECUTION
  // ============================================================================

  async execute(): Promise<LeanAideResult> {
    const operation = this.params.operation;

    switch (operation) {
      case 'health_check':
        return await this.healthCheck();

      case 'generate_proof':
        if (!this.params.theorem) {
          throw new Error('theorem parameter is required for generate_proof operation');
        }
        return await this.generateProof();

      case 'verify_proof':
        if (!this.params.code) {
          throw new Error('code parameter is required for verify_proof operation');
        }
        return await this.verifyProof();

      case 'translate_theorem':
        if (!this.params.statement) {
          throw new Error('statement parameter is required for translate_theorem operation');
        }
        return await this.translateTheorem();

      case 'elaborate':
        if (!this.params.code) {
          throw new Error('code parameter is required for elaborate operation');
        }
        return await this.verifyProof(); // Reuse verify logic

      case 'get_models':
        return await this.getModels();

      case 'math_query':
        if (!this.params.query) {
          throw new Error('query parameter is required for math_query operation');
        }
        return await this.mathQuery();

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
