/**
 * Evolution Node
 *
 * Genetic algorithm evolution node for iterative content improvement.
 * Supports multiple evolution modes including standard, quality-diversity, and island model.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { evolutionApi } from '@/services/api';

/**
 * Evolution modes
 */
export type EvolutionMode = 'standard' | 'quality_diversity' | 'island_model';

/**
 * Evolution status types
 */
export type EvolutionStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed';

/**
 * Evolution configuration
 */
export interface EvolutionNodeConfig {
  mode?: EvolutionMode;
  maxIterations?: number;
  populationSize?: number;
  temperature?: number;
  topP?: number;
  timeoutMs?: number;
  enableWebSocket?: boolean;
}

/**
 * Individual in the population
 */
export interface EvolutionIndividual {
  id: string;
  content: string;
  fitness: number;
  generation: number;
  parentIds?: string[];
}

/**
 * Population metrics
 */
export interface PopulationMetrics {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  diversity: number;
  convergenceRate: number;
  populationSize: number;
}

/**
 * Evolution result
 */
export interface EvolutionResult {
  evolutionId: string;
  status: EvolutionStatus;
  mode: EvolutionMode;
  bestContent: string;
  bestFitness: number;
  generations: number;
  populationMetrics: PopulationMetrics[];
  finalPopulation: EvolutionIndividual[];
  metadata: {
    startedAt: Date;
    completedAt?: Date;
    executionTime: number;
    parameters: {
      maxIterations: number;
      populationSize: number;
      temperature: number;
      topP: number;
    };
  };
}

/**
 * Evolution Node
 *
 * Executes genetic algorithm evolution for content optimization.
 * Supports multiple evolution strategies and real-time progress tracking.
 */
export class EvolutionNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Evolution Engine';
  static readonly DESCRIPTION = 'Genetic algorithm evolution for iterative content improvement with multiple strategies';
  static readonly ICON = 'evolution';
  static readonly CATEGORY = 'optimization';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: EvolutionNodeConfig = {}) {
    super(id, {
      mode: 'standard',
      maxIterations: 10,
      populationSize: 5,
      temperature: 0.7,
      topP: 0.9,
      timeoutMs: 300000, // 5 minutes
      enableWebSocket: false,
      ...config
    });
  }

  /**
   * Execute evolution process
   *
   * @param inputs - Must contain 'content' and optionally 'models' configuration
   * @param context - Execution context
   * @returns Promise resolving to evolution result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const content = inputs.content as string;
      const models = inputs.models as Array<{ provider: string; model: string; api_key: string }> | undefined;
      const mode = (inputs.mode as EvolutionMode) || (this.config.mode as EvolutionMode);
      const parameters = inputs.parameters as Record<string, any> | undefined;

      // Validate required inputs
      if (!content || content.trim().length === 0) {
        return this.createErrorResult('Content is required and cannot be empty');
      }

      // Safely update progress if the method exists
      if (context && typeof context.updateProgress === 'function') {
        context.updateProgress(10, 'Preparing evolution parameters');
      }

      // Prepare evolution parameters
      const evolutionParams = {
        max_iterations: parameters?.maxIterations || this.config.maxIterations as number,
        population_size: parameters?.populationSize || this.config.populationSize as number,
        temperature: parameters?.temperature || this.config.temperature as number,
        top_p: parameters?.topP || this.config.topP as number
      };

      // Safely update progress if the method exists
      if (context && typeof context.updateProgress === 'function') {
        context.updateProgress(20, 'Starting evolution process');
      }

      // Start evolution via API with error handling
      let response;
      try {
        response = await evolutionApi.start({
          content,
          mode,
          parameters: evolutionParams,
          models: models || []
        });
      } catch (apiError) {
        return this.createErrorResult(
          `API error during evolution start: ${apiError instanceof Error ? apiError.message : String(apiError)}`
        );
      }

      const evolutionId = response.evolution_id;

      // Safely update progress if the method exists
      if (context && typeof context.updateProgress === 'function') {
        context.updateProgress(30, 'Evolution started, monitoring progress');
      }

      // Monitor evolution progress
      let result;
      try {
        result = await this.monitorEvolution(evolutionId, context);
      } catch (monitorError) {
        return this.createErrorResult(
          `Error monitoring evolution: ${monitorError instanceof Error ? monitorError.message : String(monitorError)}`
        );
      }

      const executionTime = Date.now() - startTime;

      const evolutionResult: EvolutionResult = {
        evolutionId,
        status: result.status as EvolutionStatus,
        mode,
        bestContent: result.best_content || content,
        bestFitness: result.best_fitness || 0,
        generations: result.generations || 0,
        populationMetrics: result.population_metrics || [],
        finalPopulation: result.final_population || [],
        metadata: {
          startedAt: new Date(result.started_at || startTime),
          completedAt: result.completed_at ? new Date(result.completed_at) : undefined,
          executionTime,
          parameters: evolutionParams
        }
      };

      // Safely update progress if the method exists
      if (context && typeof context.updateProgress === 'function') {
        context.updateProgress(100, `Evolution complete: ${evolutionResult.generations} generations`);
      }

      return this.createSuccessResult(evolutionResult);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during evolution'
      );
    }
  }

  /**
   * Monitor evolution progress until completion
   *
   * @param evolutionId - Evolution ID to monitor
   * @param context - Execution context
   * @returns Promise resolving to evolution status
   */
  private async monitorEvolution(
    evolutionId: string,
    context: ExecutionContext
  ): Promise<any> {
    const maxAttempts = 60; // 5 minutes with 5 second intervals
    let attempts = 0;
    const timeoutMs = this.config.timeoutMs as number;
    const startTime = Date.now();

    while (attempts < maxAttempts) {
      // Check timeout
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('Evolution monitoring timeout exceeded');
      }

      try {
        const status = await evolutionApi.getStatus(evolutionId);

        // Safely update progress if the method exists
        if (context && typeof context.updateProgress === 'function') {
          // Update progress based on generation
          const progress = status.current_iteration && status.total_iterations
            ? (status.current_iteration / status.total_iterations) * 80 + 20
            : 30 + (attempts / maxAttempts) * 70;

          context.updateProgress(
            Math.min(progress, 95),
            `Generation ${status.current_iteration || 0}/${status.total_iterations || '?'}`
          );
        }

        // Check if evolution is complete
        if (status.status === 'completed' || status.status === 'failed') {
          return status;
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;

      } catch (error) {
        // If polling fails, wait and retry
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;

        // Log the error but continue monitoring
        console.warn(`Evolution monitoring attempt ${attempts} failed:`, error);
      }
    }

    throw new Error('Evolution did not complete within the expected time');
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.content) {
      errors.push({
        field: 'content',
        message: 'Content is required',
        severity: 'error'
      });
    }

    if (inputs.content && typeof inputs.content !== 'string') {
      errors.push({
        field: 'content',
        message: 'Content must be a string',
        severity: 'error'
      });
    }

    if (inputs.content && inputs.content.length < 50) {
      errors.push({
        field: 'content',
        message: 'Content is too short for meaningful evolution (minimum 50 characters)',
        severity: 'warning'
      });
    }

    // Validate mode if provided
    if (inputs.mode && typeof inputs.mode === 'string') {
      const validModes = ['standard', 'quality_diversity', 'island_model'];
      if (!validModes.includes(inputs.mode)) {
        errors.push({
          field: 'mode',
          message: `Mode must be one of: ${validModes.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate parameters if provided
    if (inputs.parameters && typeof inputs.parameters !== 'object') {
      errors.push({
        field: 'parameters',
        message: 'Parameters must be an object',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          description: 'Evolution strategy mode',
          enum: ['standard', 'quality_diversity', 'island_model'],
          default: 'standard'
        },
        maxIterations: {
          type: 'number',
          description: 'Maximum number of evolution generations',
          minimum: 1,
          maximum: 100,
          default: 10
        },
        populationSize: {
          type: 'number',
          description: 'Size of the population in each generation',
          minimum: 2,
          maximum: 20,
          default: 5
        },
        temperature: {
          type: 'number',
          description: 'Temperature for content generation (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.7
        },
        topP: {
          type: 'number',
          description: 'Top-p sampling parameter (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.9
        },
        timeoutMs: {
          type: 'number',
          description: 'Timeout for evolution in milliseconds',
          minimum: 10000,
          maximum: 600000,
          default: 300000
        },
        enableWebSocket: {
          type: 'boolean',
          description: 'Enable WebSocket for real-time updates',
          default: false
        }
      },
      required: []
    };
  }

  /**
   * Pause running evolution
   *
   * @param evolutionId - Evolution ID to pause
   * @returns Promise resolving to pause result
   */
  async pauseEvolution(evolutionId: string): Promise<NodeResult> {
    try {
      const result = await evolutionApi.pause(evolutionId);
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to pause evolution'
      );
    }
  }

  /**
   * Resume paused evolution
   *
   * @param evolutionId - Evolution ID to resume
   * @returns Promise resolving to resume result
   */
  async resumeEvolution(evolutionId: string): Promise<NodeResult> {
    try {
      const result = await evolutionApi.resume(evolutionId);
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to resume evolution'
      );
    }
  }

  /**
   * Stop running evolution
   *
   * @param evolutionId - Evolution ID to stop
   * @returns Promise resolving to stop result with final results
   */
  async stopEvolution(evolutionId: string): Promise<NodeResult> {
    try {
      const result = await evolutionApi.stop(evolutionId);
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to stop evolution'
      );
    }
  }

  /**
   * Delete evolution
   *
   * @param evolutionId - Evolution ID to delete
   * @returns Promise resolving to deletion result
   */
  async deleteEvolution(evolutionId: string): Promise<NodeResult> {
    try {
      await evolutionApi.delete(evolutionId);
      return this.createSuccessResult({ evolutionId, deleted: true });
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to delete evolution'
      );
    }
  }
}

export default EvolutionNode;
