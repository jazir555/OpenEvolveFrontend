/**
 * Gauntlet Testing Service Bubble
 *
 * Provides multi-stage quality control and adversarial testing for OpenEvolve.
 * Integrates with the Python GauntletManager for comprehensive validation.
 *
 * Features:
 * - Red/Blue/Gold team testing
 * - Multi-round evaluation with adaptive difficulty
 * - Comprehensive scoring and feedback
 * - Federation Constitution compliant
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// GAUNTLET-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const GauntletTypeSchema = z.enum([
  'red',
  'blue',
  'gold',
  'full'
]);

const DifficultySchema = z.enum([
  'easy',
  'medium',
  'hard',
  'adaptive'
]);

const EvaluationCriteriaSchema = z.enum([
  'correctness',
  'completeness',
  'efficiency',
  'clarity',
  'robustness',
  'security',
  'scalability',
  'maintainability'
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const GauntletParamsSchema = z.object({
  operation: z.enum([
    'run_gauntlet',
    'health_check',
    'get_capabilities'
  ]).describe('Operation to perform'),

  // REQUIRED: Gauntlet API URL (no magic defaults - Federation Constitution compliance)
  gauntletUrl: z.string().url().describe('Gauntlet API server URL (REQUIRED)'),
  apiKey: z.string().optional().describe('API key for authentication'),

  // Gauntlet configuration
  gauntletType: GauntletTypeSchema.default('full').describe('Type of gauntlet to run'),
  rounds: z.number().min(1).max(10).default(3).describe('Number of gauntlet rounds'),
  difficulty: DifficultySchema.default('adaptive').describe('Testing difficulty level'),
  passThreshold: z.number().min(0).max(100).default(70).describe('Pass threshold (0-100)'),

  // Solution to test
  solution: z.union([z.string(), z.object({})]).describe('Solution content to validate'),
  solutionId: z.string().optional().describe('Unique identifier for the solution'),

  // Evaluation criteria
  evaluationCriteria: z.array(EvaluationCriteriaSchema)
    .default(['correctness', 'completeness', 'efficiency', 'clarity', 'robustness'])
    .describe('Criteria to evaluate'),

  // Context and requirements
  requirements: z.array(z.string()).optional().describe('Requirements to validate against'),
  constraints: z.array(z.string()).optional().describe('Constraints to check'),
  context: z.record(z.unknown()).optional().describe('Additional context for evaluation'),

  // Advanced options
  enableLearning: z.boolean().default(true).describe('Adapt gauntlet based on previous runs'),
  timeout: z.number().min(5000).max(300000).default(120000).describe('Timeout in ms'),
});

type GauntletParamsInput = z.input<typeof GauntletParamsSchema>;
type GauntletParams = z.output<typeof GauntletParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const RoundResultSchema = z.object({
  round: z.number(),
  team: z.string(),
  score: z.number(),
  passed: z.boolean(),
  criteriaScores: z.record(z.number()).optional(),
  feedback: z.array(z.string()),
  timestamp: z.string(),
});

const TeamPerformanceSchema = z.object({
  team: z.string(),
  overallScore: z.number(),
  roundsParticipated: z.number(),
  strengths: z.array(z.string()),
  weaknesses: z.array(z.string()),
  recommendations: z.array(z.string()),
});

const GauntletResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),

  // Core results
  passed: z.boolean().describe('Whether solution passed the gauntlet'),
  score: z.number().describe('Overall score (0-100)'),

  // Detailed results
  roundResults: z.array(RoundResultSchema).describe('Results from each round'),
  feedback: z.array(z.string()).describe('Feedback items'),
  improvementsNeeded: z.array(z.string()).describe('Required improvements'),
  teamPerformances: z.array(TeamPerformanceSchema).describe('Team performance metrics'),

  // Summary metadata
  summary: z.object({
    gauntletType: z.string(),
    roundsCompleted: z.number(),
    totalRounds: z.number(),
    difficultyUsed: z.string(),
    criteriaEvaluated: z.array(z.string()),
    overallScore: z.number(),
    passThreshold: z.number(),
    passed: z.boolean(),
  }),

  // Execution metadata
  metadata: z.object({
    executionTime: z.number().describe('Execution time in ms'),
    gauntletVersion: z.string().optional(),
    teamConfigurations: z.unknown().optional(),
  }),

  // Error handling
  error: z.string().optional(),
  timing: z.number().describe('Response time in milliseconds'),
});

type GauntletResult = z.output<typeof GauntletResultSchema>;

// ============================================================================
// GAUNTLET BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class GauntletBubble extends ServiceBubble<GauntletParams, GauntletResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'gauntlet' as const;
  static readonly type = 'service' as const;
  static readonly schema = GauntletParamsSchema;
  static readonly resultSchema = GauntletResultSchema;
  static readonly credentialType = 'gauntlet_api_key' as const;

  static readonly shortDescription = 'Multi-stage quality control and adversarial testing';
  static readonly longDescription = `
    Gauntlet testing service bubble for OpenEvolve validation pipeline.

    Features:
    - Red Team: Adversarial testing and critique
    - Blue Team: Solution refinement and improvement
    - Gold Team: Final evaluation and certification
    - Full Gauntlet: All teams in sequence
    - Multi-round testing with adaptive difficulty
    - Comprehensive scoring and feedback
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - gauntletUrl: Gauntlet server API URL (no default - must be provided)
    - apiKey: Optional API key for authentication

    Federation Constitution Compliance:
    - No magic defaults (gauntletUrl is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: GauntletParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    GauntletBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('gauntlet', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - gauntletUrl is required by schema
    // Additional runtime validation can be added
  }

  /**
   * Build HTTP headers for Gauntlet API requests
   */
  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.params.apiKey) {
      headers['Authorization'] = `Bearer ${this.params.apiKey}`;
    }

    return headers;
  }

  /**
   * Build full URL for Gauntlet endpoint
   */
  private buildUrl(endpoint: string): string {
    const baseUrl = this.params.gauntletUrl.endsWith('/')
      ? this.params.gauntletUrl.slice(0, -1)
      : this.params.gauntletUrl;
    return `${baseUrl}${endpoint}`;
  }

  /**
   * Make HTTP request to Gauntlet API
   */
  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<Response> {
    const url = this.buildUrl(endpoint);

    return await fetch(url, {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  /**
   * Health check operation
   */
  private async healthCheck(): Promise<GauntletResult> {
    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        'gauntlet-healthcheck',
        () => this.makeRequest('GET', '/health'),
        { operation: 'health_check' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'health_check',
        data,
        passed: response.ok,
        score: response.ok ? 100 : 0,
        roundResults: [],
        feedback: response.ok ? ['Gauntlet service is healthy'] : ['Health check failed'],
        improvementsNeeded: [],
        teamPerformances: [],
        summary: {
          gauntletType: 'health_check',
          roundsCompleted: 0,
          totalRounds: 0,
          difficultyUsed: 'medium',
          criteriaEvaluated: [],
          overallScore: response.ok ? 100 : 0,
          passThreshold: 70,
          passed: response.ok,
        },
        metadata: {
          executionTime: timing,
        },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'health_check',
        passed: false,
        score: 0,
        roundResults: [],
        feedback: ['Health check failed'],
        improvementsNeeded: [],
        teamPerformances: [],
        summary: {
          gauntletType: 'health_check',
          roundsCompleted: 0,
          totalRounds: 0,
          difficultyUsed: 'medium',
          criteriaEvaluated: [],
          overallScore: 0,
          passThreshold: 70,
          passed: false,
        },
        metadata: {
          executionTime: timing,
        },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Run gauntlet operation
   */
  private async runGauntlet(): Promise<GauntletResult> {
    if (!this.params.solution) {
      throw new Error('solution is required for run_gauntlet operation');
    }

    const startTime = Date.now();

    try {
      const requestBody = {
        solution: this.params.solution,
        solution_id: this.params.solutionId,
        gauntlet_type: this.params.gauntletType,
        rounds: this.params.rounds,
        difficulty: this.params.difficulty,
        evaluation_criteria: this.params.evaluationCriteria,
        pass_threshold: this.params.passThreshold,
        requirements: this.params.requirements,
        constraints: this.params.constraints,
        context: this.params.context,
        enable_learning: this.params.enableLearning,
      };

      const cacheKey = `gauntlet-run-${this.params.solutionId || 'unknown'}-${this.params.gauntletType}`;

      const response = await this.resilience.execute(
        cacheKey,
        () => this.makeRequest('POST', '/gauntlet/run', requestBody),
        {
          operation: 'run_gauntlet',
          gauntletType: this.params.gauntletType,
          solutionId: this.params.solutionId,
        }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.message || 'Gauntlet execution failed');
      }

      // Transform response to match our schema
      return {
        success: true,
        operation: 'run_gauntlet',
        data,
        passed: data.passed ?? false,
        score: data.overall_score ?? data.score ?? 0,
        roundResults: this.formatRoundResults(data.round_results || data.rounds || []),
        feedback: this.formatFeedback(data.feedback || []),
        improvementsNeeded: data.improvements_needed || [],
        teamPerformances: this.formatTeamPerformances(data.team_performances || []),
        summary: {
          gauntletType: this.params.gauntletType,
          roundsCompleted: data.rounds_completed || data.round_results?.length || 0,
          totalRounds: this.params.rounds,
          difficultyUsed: data.difficulty_used || this.params.difficulty,
          criteriaEvaluated: this.params.evaluationCriteria,
          overallScore: data.overall_score ?? data.score ?? 0,
          passThreshold: this.params.passThreshold,
          passed: data.passed ?? false,
        },
        metadata: {
          executionTime: timing,
          gauntletVersion: data.version,
          teamConfigurations: data.team_configurations,
        },
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'run_gauntlet',
        passed: false,
        score: 0,
        roundResults: [],
        feedback: [`Gauntlet execution failed: ${errorMessage}`],
        improvementsNeeded: [],
        teamPerformances: [],
        summary: {
          gauntletType: this.params.gauntletType,
          roundsCompleted: 0,
          totalRounds: this.params.rounds,
          difficultyUsed: this.params.difficulty,
          criteriaEvaluated: this.params.evaluationCriteria,
          overallScore: 0,
          passThreshold: this.params.passThreshold,
          passed: false,
        },
        metadata: {
          executionTime: timing,
        },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Format round results from API response
   */
  private formatRoundResults(rounds: unknown[]): RoundResultSchema[] {
    return rounds.map((round: unknown) => ({
      round: (round as any).round_number || (round as any).round || 0,
      team: (round as any).team_type || (round as any).team || 'unknown',
      score: (round as any).score || 0,
      passed: (round as any).passed || false,
      criteriaScores: (round as any).criteria_scores || {},
      feedback: (round as any).feedback || [],
      timestamp: (round as any).timestamp || new Date().toISOString(),
    }));
  }

  /**
   * Format feedback from API response
   */
  private formatFeedback(feedback: unknown[]): string[] {
    return feedback.map((item: unknown) => {
      if (typeof item === 'string') return item;
      if (typeof item === 'object' && item !== null) {
        return (item as any).message || (item as any).suggestion || JSON.stringify(item);
      }
      return String(item);
    });
  }

  /**
   * Format team performances from API response
   */
  private formatTeamPerformances(performances: unknown[]): TeamPerformanceSchema[] {
    return performances.map((perf: unknown) => ({
      team: (perf as any).team_type || (perf as any).team || 'unknown',
      overallScore: (perf as any).score || (perf as any).overall_score || 0,
      roundsParticipated: (perf as any).rounds_count || (perf as any).rounds_participated || 0,
      strengths: (perf as any).strengths || [],
      weaknesses: (perf as any).weaknesses || [],
      recommendations: (perf as any).recommendations || [],
    }));
  }

  /**
   * Get capabilities operation
   */
  private async getCapabilities(): Promise<GauntletResult> {
    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        'gauntlet-capabilities',
        () => this.makeRequest('GET', '/capabilities'),
        { operation: 'get_capabilities' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'get_capabilities',
        data,
        passed: response.ok,
        score: response.ok ? 100 : 0,
        roundResults: [],
        feedback: response.ok ? ['Capabilities retrieved successfully'] : ['Failed to get capabilities'],
        improvementsNeeded: [],
        teamPerformances: [],
        summary: {
          gauntletType: 'capabilities',
          roundsCompleted: 0,
          totalRounds: 0,
          difficultyUsed: 'medium',
          criteriaEvaluated: [],
          overallScore: response.ok ? 100 : 0,
          passThreshold: 70,
          passed: response.ok,
        },
        metadata: {
          executionTime: timing,
        },
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_capabilities',
        passed: false,
        score: 0,
        roundResults: [],
        feedback: ['Failed to get capabilities'],
        improvementsNeeded: [],
        teamPerformances: [],
        summary: {
          gauntletType: 'capabilities',
          roundsCompleted: 0,
          totalRounds: 0,
          difficultyUsed: 'medium',
          criteriaEvaluated: [],
          overallScore: 0,
          passThreshold: 70,
          passed: false,
        },
        metadata: {
          executionTime: timing,
        },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<GauntletResult> {
    switch (this.params.operation) {
      case 'health_check':
        return this.healthCheck();
      case 'run_gauntlet':
        return this.runGauntlet();
      case 'get_capabilities':
        return this.getCapabilities();
      default:
        return {
          success: false,
          operation: this.params.operation,
          passed: false,
          score: 0,
          roundResults: [],
          feedback: [`Unknown operation: ${this.params.operation}`],
          improvementsNeeded: [],
          teamPerformances: [],
          summary: {
            gauntletType: 'unknown',
            roundsCompleted: 0,
            totalRounds: 0,
            difficultyUsed: 'medium',
            criteriaEvaluated: [],
            overallScore: 0,
            passThreshold: 70,
            passed: false,
          },
          metadata: {
            executionTime: 0,
          },
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default GauntletBubble;
