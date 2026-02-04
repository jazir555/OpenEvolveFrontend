/**
 * OpenEvolve Integration
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: NO IMPORTS from core-projects/openevolve
 * - We integrate via HTTP API only
 * - Law of Runtime Truth: Probe OpenEvolve API before use
 *
 * This module provides integration hooks between OpenEvolve and the
 * evolved code capture system.
 */

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../logger';
import { EvolvedCodeCapturer } from './capturer';
import {
  Problem,
  EvolutionMetrics,
  EvolvedCode,
  CaptureResult,
  LanguageEnum,
  validateProblem,
  validateEvolutionMetrics,
  validateEvolvedCode,
} from './canonical';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface OpenEvolveIntegrationConfig {
  // OpenEvolve API configuration
  openevolve_api_url: string;
  openevolve_api_key?: string;

  // Capturer configuration
  capturer: EvolvedCodeCapturer;

  // Webhook configuration
  webhook_enabled: boolean;
  webhook_path: string;

  // Capture trigger configuration
  auto_capture_on_completion: boolean;
  capture_threshold_fitness?: number;  // Only capture if fitness above threshold
  capture_top_n_solutions?: number;  // Capture top N solutions from final generation

  // Timeout and retry
  timeout_ms?: number;
  max_retries?: number;

  // Logging
  logger?: Logger;
}

const DEFAULT_CONFIG = {
  webhook_enabled: true,
  webhook_path: '/webhooks/openevolve/evolution-complete',
  auto_capture_on_completion: true,
  capture_top_n_solutions: 1,
  timeout_ms: 30000,  // 30 seconds
  max_retries: 3,
};

// ============================================================================
// OPENEVOLVE API CLIENT
// ============================================================================

/**
 * OpenEvolve API Client
 *
 * Lightweight client for interacting with OpenEvolve via HTTP
 * Following Law of the Air Gap: No direct imports, only HTTP
 */
export class OpenEvolveClient {
  private readonly apiUrl: string;
  private readonly apiKey: string | undefined;
  private readonly logger: Logger;
  private readonly timeout: number;

  constructor(apiUrl: string, apiKey?: string, logger?: Logger, timeout: number = 30000) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.logger = logger || new Logger('openevolve-client');
    this.timeout = timeout;
  }

  /**
   * Get evolution result by ID
   */
  async getEvolutionResult(evolutionId: string): Promise<any> {
    const correlationId = uuidv4();

    this.logger.info('Fetching evolution result from OpenEvolve', {
      correlation_id: correlationId,
      evolution_id: evolutionId,
      target_service: 'openevolve-core',
    });

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(
        `${this.apiUrl}/evolutions/${evolutionId}`,
        {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(this.timeout),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      this.logger.info('Evolution result fetched successfully', {
        correlation_id: correlationId,
        evolution_id: evolutionId,
      });

      return result;
    } catch (error) {
      this.logger.error('Failed to fetch evolution result', error as Error, {
        correlation_id: correlationId,
        evolution_id: evolutionId,
      });
      throw error;
    }
  }

  /**
   * List recent evolutions
   */
  async listEvolutions(limit: number = 10): Promise<any[]> {
    const correlationId = uuidv4();

    this.logger.info('Listing evolutions from OpenEvolve', {
      correlation_id: correlationId,
      limit,
    });

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(
        `${this.apiUrl}/evolutions?limit=${limit}`,
        {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(this.timeout),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      this.logger.info('Evolutions listed successfully', {
        correlation_id: correlationId,
        count: result.length || 0,
      });

      return result.evolutions || result;
    } catch (error) {
      this.logger.error('Failed to list evolutions', error as Error, {
        correlation_id: correlationId,
      });
      throw error;
    }
  }
}

// ============================================================================
// OPENEVOLVE INTEGRATION
// ============================================================================

/**
 * OpenEvolve Integration
 *
 * Orchestrates the capture of evolved code from OpenEvolve
 */
export class OpenEvolveIntegration {
  private readonly config: Required<Omit<OpenEvolveIntegrationConfig, 'logger'>> & {
    logger?: Logger;
  };

  private readonly logger: Logger;
  private readonly client: OpenEvolveClient;
  private readonly capturer: EvolvedCodeCapturer;
  private initialized: boolean = false;

  constructor(config: OpenEvolveIntegrationConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    this.logger = this.config.logger || new Logger('openevolve-integration');

    // Initialize OpenEvolve client
    this.client = new OpenEvolveClient(
      this.config.openevolve_api_url,
      this.config.openevolve_api_key,
      this.logger,
      this.config.timeout_ms
    );

    // Initialize capturer
    this.capturer = this.config.capturer;

    this.logger.info('OpenEvolveIntegration initialized', {
      correlation_id: 'integration-init',
      auto_capture: this.config.auto_capture_on_completion,
      webhook_enabled: this.config.webhook_enabled,
    });
  }

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  /**
   * Initialize integration
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn('OpenEvolveIntegration already initialized', {
        correlation_id: 'integration-init',
      });
      return;
    }

    const correlationId = uuidv4();

    this.logger.info('Initializing OpenEvolveIntegration', {
      correlation_id: correlationId,
    });

    try {
      // Initialize capturer
      await this.capturer.initialize();

      // Verify OpenEvolve connection
      await this.client.listEvolutions(1);

      this.initialized = true;
      this.logger.info('OpenEvolveIntegration initialized successfully', {
        correlation_id: correlationId,
      });
    } catch (error) {
      this.logger.error('Failed to initialize OpenEvolveIntegration', error as Error, {
        correlation_id: correlationId,
      });
      throw new Error(
        `OpenEvolveIntegration initialization failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // ========================================================================
  // CAPTURE OPERATIONS
  // ========================================================================

  /**
   * Capture evolution from OpenEvolve
   * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
   */
  async captureEvolution(evolutionId: string, correlationId?: string): Promise<CaptureResult[]> {
    const cid = correlationId || uuidv4();

    this.logger.info('Capturing evolution from OpenEvolve', {
      correlation_id: cid,
      evolution_id: evolutionId,
    });

    try {
      // Fetch evolution result
      const evolutionResult = await this.client.getEvolutionResult(evolutionId);

      // Convert to canonical format
      const problem = this.convertToProblem(evolutionResult);
      const solutions = this.convertToEvolvedCodes(evolutionResult);

      // Validate problem
      const problemValidation = validateProblem(problem);
      if (!problemValidation.success) {
        throw new Error(`Invalid problem: ${problemValidation.errors.join(', ')}`);
      }

      // Capture top N solutions
      const topNSolutions = solutions
        .filter(s => {
          if (this.config.capture_threshold_fitness) {
            return s.metrics.fitness_score >= this.config.capture_threshold_fitness;
          }
          return true;
        })
        .slice(0, this.config.capture_top_n_solutions);

      const captureResults: CaptureResult[] = [];

      for (const solution of topNSolutions) {
        const result = await this.capturer.captureEvolution(
          problem,
          solution,
          solution.metrics,
          cid
        );
        captureResults.push(result);
      }

      this.logger.info('Evolution captured successfully', {
        correlation_id: cid,
        evolution_id: evolutionId,
        solutions_captured: captureResults.length,
      });

      return captureResults;
    } catch (error) {
      this.logger.error('Failed to capture evolution', error as Error, {
        correlation_id: cid,
        evolution_id: evolutionId,
      });
      throw error;
    }
  }

  // ========================================================================
  // CONVERSION METHODS
  // ========================================================================

  /**
   * Convert OpenEvolve result to canonical Problem
   */
  private convertToProblem(evolutionResult: any): Problem {
    // Extract problem from OpenEvolve result
    const problemData = evolutionResult.problem || evolutionResult.config || {};

    return {
      description: problemData.description || problemData.name || 'Evolution problem',
      type: this.mapProblemType(problemData.type),
      constraints: problemData.constraints ? {
        max_memory_mb: problemData.constraints.max_memory_mb,
        max_runtime_ms: problemData.constraints.max_runtime_ms,
        required_libraries: problemData.constraints.required_libraries,
        language_version: problemData.constraints.language_version,
      } : undefined,
      input_spec: problemData.input_spec,
      output_spec: problemData.output_spec,
      test_cases: problemData.test_cases,
      difficulty: problemData.difficulty,
      tags: problemData.tags || [],
    };
  }

  /**
   * Convert OpenEvolve result to canonical EvolvedCode
   */
  private convertToEvolvedCodes(evolutionResult: any): EvolvedCode[] {
    const evolvedCodes: EvolvedCode[] = [];

    // Get best solution from final generation
    const bestSolution = evolutionResult.best_solution || evolutionResult.bestProgram;

    if (bestSolution) {
      evolvedCodes.push(this.convertSolutionToEvolvedCode(bestSolution, evolutionResult));
    }

    // Get other solutions from final generation if requested
    if (this.config.capture_top_n_solutions > 1) {
      const finalGeneration = evolutionResult.final_generation || evolutionResult.generation;
      if (finalGeneration && finalGeneration.solutions) {
        for (const solution of finalGeneration.solutions) {
          if (solution.id !== bestSolution?.id) {
            evolvedCodes.push(this.convertSolutionToEvolvedCode(solution, evolutionResult));
          }
        }
      }
    }

    return evolvedCodes;
  }

  /**
   * Convert single solution to canonical EvolvedCode
   */
  private convertSolutionToEvolvedCode(solution: any, evolutionResult: any): EvolvedCode {
    return {
      id: solution.id || uuidv4(),
      problem: this.convertToProblem(evolutionResult),
      language: this.mapLanguage(solution.language || evolutionResult.language),
      code: solution.code || solution.program || '',
      function_name: solution.function_name,
      class_name: solution.class_name,
      metrics: this.convertToMetrics(solution, evolutionResult),
      timestamp_utc: solution.timestamp || new Date().toISOString(),
      is_valid: solution.is_valid ?? true,
      validation_errors: solution.validation_errors,
      execution_time_ms: solution.execution_time_ms,
      memory_used_mb: solution.memory_used_mb,
      parent_code_id: solution.parent_id,
      generation_number: solution.generation || evolutionResult.generation,
      tags: solution.tags || [],
      metadata: solution.metadata || {},
    };
  }

  /**
   * Convert to canonical EvolutionMetrics
   */
  private convertToMetrics(solution: any, evolutionResult: any): EvolutionMetrics {
    return {
      iterations: evolutionResult.iterations || evolutionResult.generation || 0,
      fitness_score: solution.fitness || solution.score || 0,
      fitness_improvement: solution.fitness_improvement || 0,
      duration_ms: evolutionResult.duration_ms || 0,
      generations: evolutionResult.total_generations,
      population_size: evolutionResult.population_size,
      mutation_rate: evolutionResult.mutation_rate,
      crossover_rate: evolutionResult.crossover_rate,
      convergence_generation: evolutionResult.convergence_generation,
      total_evaluations: evolutionResult.total_evaluations,
      success_rate: solution.success_rate,
      benchmark_score: solution.benchmark_score,
    };
  }

  /**
   * Map OpenEvolve problem type to canonical type
   */
  private mapProblemType(type: string): Problem['type'] {
    const typeMap: Record<string, Problem['type']> = {
      'algotune': 'algorithm_optimization',
      'optimization': 'algorithm_optimization',
      'refactoring': 'code_refactoring',
      'performance': 'performance_tuning',
      'bugfix': 'bug_fix',
      'feature': 'feature_implementation',
      'migration': 'code_migration',
      'parallel': 'parallelization',
      'memory': 'memory_optimization',
      'numerical': 'numerical_computation',
      'datastructure': 'data_structure_design',
    };

    return typeMap[type.toLowerCase()] || 'other';
  }

  /**
   * Map language to canonical Language
   */
  private mapLanguage(language: string): Language {
    const langMap: Record<string, Language> = {
      'py': 'python',
      'python': 'python',
      'js': 'javascript',
      'javascript': 'javascript',
      'ts': 'typescript',
      'typescript': 'typescript',
      'java': 'java',
      'c': 'c',
      'cpp': 'cpp',
      'c++': 'cpp',
      'csharp': 'csharp',
      'c#': 'csharp',
      'go': 'go',
      'rs': 'rust',
      'rust': 'rust',
      'jl': 'julia',
      'julia': 'julia',
      'm': 'matlab',
      'matlab': 'matlab',
      'r': 'r',
    };

    return langMap[language.toLowerCase()] || 'other';
  }

  // ========================================================================
  // WEBHOOK HANDLER
  // ========================================================================

  /**
   * Handle webhook from OpenEvolve
   * Called when evolution completes
   */
  async handleWebhook(payload: any, correlationId?: string): Promise<CaptureResult[]> {
    const cid = correlationId || uuidv4();

    this.logger.info('Received webhook from OpenEvolve', {
      correlation_id: cid,
      event_type: payload.event_type,
      evolution_id: payload.evolution_id,
    });

    if (!this.config.auto_capture_on_completion) {
      this.logger.info('Auto-capture disabled, skipping webhook handling', {
        correlation_id: cid,
      });
      return [];
    }

    if (payload.event_type === 'evolution_complete' || payload.event_type === 'completion') {
      return await this.captureEvolution(payload.evolution_id, cid);
    } else {
      this.logger.warn('Unhandled webhook event type', {
        correlation_id: cid,
        event_type: payload.event_type,
      });
      return [];
    }
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  /**
   * Check integration health
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    initialized: boolean;
    openevolve_connected: boolean;
    capturer_healthy: boolean;
  }> {
    const capturerHealth = await this.capturer.healthCheck();

    try {
      if (!this.initialized) {
        return {
          healthy: false,
          initialized: false,
          openevolve_connected: false,
          capturer_healthy: capturerHealth.healthy,
        };
      }

      // Quick connectivity check to OpenEvolve
      await this.client.listEvolutions(1);

      return {
        healthy: capturerHealth.healthy,
        initialized: true,
        openevolve_connected: true,
        capturer_healthy: capturerHealth.healthy,
      };
    } catch (error) {
      return {
        healthy: false,
        initialized: true,
        openevolve_connected: false,
        capturer_healthy: capturerHealth.healthy,
      };
    }
  }

  // ========================================================================
  // CLEANUP
  // ========================================================================

  /**
   * Close integration and cleanup resources
   */
  async close(): Promise<void> {
    this.logger.info('Closing OpenEvolveIntegration', {
      correlation_id: 'integration-close',
    });

    try {
      await this.capturer.close();
      this.initialized = false;

      this.logger.info('OpenEvolveIntegration closed successfully', {
        correlation_id: 'integration-close',
      });
    } catch (error) {
      this.logger.error('Error closing OpenEvolveIntegration', error as Error, {
        correlation_id: 'integration-close',
      });
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export type { OpenEvolveIntegrationConfig };
