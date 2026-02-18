/**
 * EvolutionTriggerBubble
 *
 * Triggers OpenEvolve evolution workflows and monitors their progress.
 * This bubble serves as the entry point for evolutionary computation
 * within BubbleLab workflows.
 *
 * Architecture: Glue Layer Adapter
 * - Calls OpenEvolve API via openevolveApi
 * - Monitors execution progress
 * - Returns best evolved solutions
 * - UTC timestamp handling
 * - Circuit breaker pattern for resilience
 *
 * @see CLAUDE.md - Federation Constitution compliance
 */

import { z } from 'zod';
import { WorkflowBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { openevolveApi, type WorkflowResponse, type ExecutionResponse } from '@/services/openevolveApi';
import { logger } from '@/utils/logger';

// ==================== Canonical Schemas ====================

/**
 * Problem context extracted from workflow input
 */
const ProblemContextSchema = z.object({
  description: z.string().describe('Natural language description of the problem to solve'),
  constraints: z.array(z.string()).optional().describe('Hard constraints for the evolution'),
  objectives: z.array(z.string()).optional().describe('Optimization objectives'),
  context: z.string().optional().describe('Additional context for the evolution'),
});

/**
 * Evolution input parameters
 */
const EvolutionInputSchema = z.object({
  problemStatement: z.string().describe('Problem statement for evolution'),
  context: z.string().optional().describe('Additional context for evolution'),
  workflowType: z.enum(['evolution', 'adversarial', 'sovereign']).default('evolution'),
  iterations: z.number().int().min(1).max(10000).default(100).describe('Number of evolution iterations'),
  populationSize: z.number().int().min(1).max(1000).default(50).describe('Population size for genetic algorithm'),
  teams: z.array(z.string()).optional().describe('OpenEvolve team IDs to use'),
  gauntlets: z.array(z.string()).optional().describe('OpenEvolve gauntlet IDs to apply'),
});

/**
 * Evolution summary from OpenEvolve
 */
const EvolutionSummarySchema = z.object({
  evolutionId: z.string().describe('Unique evolution execution ID'),
  workflowId: z.string().describe('OpenEvolve workflow ID'),
  status: z.enum(['queued', 'running', 'paused', 'completed', 'failed', 'cancelled']),
  progress: z.number().min(0).max(100).describe('Progress percentage'),
  bestSolution: z.unknown().optional().describe('Best solution found'),
  fitness: z.number().optional().describe('Fitness score of best solution'),
  iterations: z.number().optional().describe('Number of iterations completed'),
  startedAt: z.string().datetime().describe('UTC ISO-8601 start timestamp'),
  completedAt: z.string().datetime().optional().describe('UTC ISO-8601 completion timestamp'),
});

/**
 * Evolution result returned by this bubble
 */
const EvolutionResultSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),

  evolutionId: z.string().optional().describe('Evolution execution ID'),
  workflowId: z.string().optional().describe('OpenEvolve workflow ID'),
  bestSolution: z.unknown().optional().describe('Best evolved solution'),
  fitness: z.number().optional().describe('Fitness score'),
  iterations: z.number().optional().describe('Iterations completed'),
  progress: z.number().optional().describe('Progress percentage'),

  timing: z.object({
    total: z.number().describe('Total execution time in ms'),
    evolution: z.number().optional().describe('Evolution time in ms'),
  }),
});

// ==================== Type Definitions ====================

type ProblemContext = z.output<typeof ProblemContextSchema>;
export type EvolutionInput = z.input<typeof EvolutionInputSchema>;
export type EvolutionRequest = z.output<typeof EvolutionInputSchema>;
type EvolutionSummary = z.output<typeof EvolutionSummarySchema>;
export type EvolutionResult = z.output<typeof EvolutionResultSchema>;

// ==================== Evolution Trigger Bubble ====================

/**
 * EvolutionTriggerBubble
 *
 * Triggers OpenEvolve evolution workflows and monitors their progress.
 *
 * Features:
 * - Creates and executes OpenEvolve workflows
 * - Monitors evolution progress with polling
 * - Returns best solution found
 * - Handles timeouts and failures gracefully
 * - UTC timestamp handling
 *
 * Usage:
 * ```typescript
 * const bubble = new EvolutionTriggerBubble({
 *   problemStatement: 'Optimize the sorting algorithm',
 *   iterations: 100,
 *   populationSize: 50,
 * });
 *
 * const result = await bubble.action();
 * if (result.success && result.bestSolution) {
 *   console.log('Best solution:', result.bestSolution);
 *   console.log('Fitness:', result.fitness);
 * }
 * ```
 */
export class EvolutionTriggerBubble extends WorkflowBubble<EvolutionInput, EvolutionResult> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName = 'evolution-trigger';
  static readonly schema = EvolutionInputSchema;
  static readonly resultSchema = EvolutionResultSchema;
  static readonly shortDescription = 'Triggers OpenEvolve evolution workflows';
  static readonly longDescription = `
    Initiates evolutionary computation via OpenEvolve and monitors progress.

    Features:
    - Creates OpenEvolve workflows with custom parameters
    - Monitors execution progress with adaptive polling
    - Returns best evolved solution and fitness metrics
    - Handles failures with circuit breaker pattern
    - UTC timestamp handling for all time values

    Supports evolution, adversarial, and sovereign workflow types.
  `;
  static readonly alias = 'evolve';

  // Circuit breaker state
  private static circuitBreakerState = {
    failures: 0,
    lastFailureTime: 0,
    isOpen: false,
    cooldownPeriod: 60000, // 1 minute cooldown
  };

  constructor(params: EvolutionInput, context?: BubbleContext) {
    super(params, context);
  }

  /**
   * Main action method that orchestrates the evolution
   */
  protected async performAction(_context?: BubbleContext): Promise<EvolutionResult> {
    const startTime = Date.now();

    try {
      // Check circuit breaker
      if (EvolutionTriggerBubble.isCircuitBreakerOpen()) {
        throw new Error('Circuit breaker is open due to repeated failures. Please try again later.');
      }

      logger.info({
        msg: 'Starting evolution workflow',
        component: 'EvolutionTriggerBubble',
        workflow_type: this.params.workflowType ?? 'evolution',
        iterations: this.params.iterations ?? 100,
        population_size: this.params.populationSize ?? 50,
      });

      // 1. Extract problem context
      const problemContext = this.analyzeProblem();

      // 2. Create evolution request
      const workflowType = this.params.workflowType ?? 'evolution';
      const iterations = this.params.iterations ?? 100;
      const populationSize = this.params.populationSize ?? 50;

      const evolutionRequest: EvolutionRequest = {
        problemStatement: problemContext.description,
        context: problemContext.context,
        workflowType,
        iterations,
        populationSize,
        teams: this.params.teams,
        gauntlets: this.params.gauntlets,
      };

      // 3. Trigger evolution via OpenEvolve API
      const openEvolveResult = await this.triggerEvolution(evolutionRequest);

      // 4. Monitor evolution progress
      const finalResult = await this.monitorProgress(openEvolveResult.execution_id);

      // Record success for circuit breaker
      EvolutionTriggerBubble.recordSuccess();

      const timing = {
        total: Date.now() - startTime,
        evolution: finalResult.completedAt && finalResult.startedAt
          ? new Date(finalResult.completedAt).getTime() - new Date(finalResult.startedAt).getTime()
          : undefined,
      };

      logger.info({
        msg: 'Evolution completed successfully',
        component: 'EvolutionTriggerBubble',
        evolution_id: finalResult.evolutionId,
        iterations: finalResult.iterations,
        fitness: finalResult.fitness,
        timing,
      });

      return {
        success: true,
        evolutionId: finalResult.evolutionId,
        workflowId: finalResult.workflowId,
        bestSolution: finalResult.bestSolution,
        fitness: finalResult.fitness,
        iterations: finalResult.iterations,
        progress: finalResult.progress,
        timing,
      };
    } catch (error) {
      // Record failure for circuit breaker
      EvolutionTriggerBubble.recordFailure();

      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      logger.error({
        msg: 'Evolution failed',
        component: 'EvolutionTriggerBubble',
        error: errorMessage,
        timing_total: Date.now() - startTime,
      });

      return {
        success: false,
        error: errorMessage,
        timing: {
          total: Date.now() - startTime,
        },
      };
    }
  }

  /**
   * Analyze problem from workflow input
   * Extracts and structures problem context for OpenEvolve
   */
  private analyzeProblem(): ProblemContext {
    const { problemStatement, context } = this.params;

    logger.debug({
      msg: 'Analyzing problem context',
      component: 'EvolutionTriggerBubble',
      problem_statement_length: problemStatement.length,
    });

    return {
      description: problemStatement,
      constraints: [],
      objectives: [],
      context: context || '',
    };
  }

  /**
   * Trigger evolution via OpenEvolve API
   * Creates workflow and starts execution
   */
  private async triggerEvolution(request: EvolutionRequest): Promise<ExecutionResponse & { workflowId: string }> {
    logger.debug({
      msg: 'Creating OpenEvolve workflow',
      component: 'EvolutionTriggerBubble',
      workflow_type: request.workflowType,
    });

    // Create workflow
    const workflow: WorkflowResponse = await openevolveApi.createWorkflow({
      name: `Evolution: ${request.problemStatement.slice(0, 50)}...`,
      description: request.context || 'Evolutionary computation workflow',
      workflow_type: request.workflowType,
      problem_statement: request.problemStatement,
      teams: request.teams,
      gauntlets: request.gauntlets,
      parameters: {
        iterations: request.iterations,
        populationSize: request.populationSize,
      },
    });

    logger.debug({
      msg: 'Workflow created, starting execution',
      component: 'EvolutionTriggerBubble',
      workflow_id: workflow.id,
    });

    // Execute workflow
    const execution: ExecutionResponse = await openevolveApi.executeWorkflow(workflow.id, {
      problem_statement: request.problemStatement,
      context: request.context,
    });

    return {
      ...execution,
      workflowId: workflow.id,
    };
  }

  /**
   * Monitor evolution progress with adaptive polling
   * Polls OpenEvolve API until completion or timeout
   */
  private async monitorProgress(executionId: string): Promise<EvolutionSummary> {
    const maxPollTime = 600000; // 10 minutes maximum
    const pollInterval = 2000; // Start with 2 second polling
    const startTime = Date.now();

    logger.info({
      msg: 'Monitoring evolution progress',
      component: 'EvolutionTriggerBubble',
      execution_id: executionId,
    });

    while (Date.now() - startTime < maxPollTime) {
      const status = await openevolveApi.getExecutionStatus(executionId);

      logger.debug({
        msg: 'Evolution status update',
        component: 'EvolutionTriggerBubble',
        execution_id: executionId,
        status: status.status,
        progress: status.progress,
      });

      // Check for completion
      if (status.status === 'completed') {
        const startedAt = status.started_at || new Date().toISOString();
        const completedAt = status.completed_at || new Date().toISOString();

        return {
          evolutionId: executionId,
          workflowId: status.workflow_id,
          status: status.status,
          progress: status.progress || 100,
          bestSolution: status.result,
          fitness: typeof status.result === 'object' && status.result && 'fitness' in status.result
            ? (status.result as any).fitness as number
            : undefined,
          iterations: typeof status.result === 'object' && status.result && 'iterations' in status.result
            ? (status.result as any).iterations as number
            : undefined,
          startedAt,
          completedAt,
        };
      }

      // Check for failure
      if (status.status === 'failed' || status.status === 'cancelled') {
        throw new Error(status.error || `Evolution ${status.status}`);
      }

      // Adaptive polling: slower as progress increases
      const adaptiveDelay = pollInterval + (status.progress || 0) * 10;
      await this.sleep(adaptiveDelay);
    }

    throw new Error('Evolution monitoring timeout');
  }

  /**
   * Sleep utility for polling delays
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Check if circuit breaker is open
   */
  private static isCircuitBreakerOpen(): boolean {
    const now = Date.now();
    const state = EvolutionTriggerBubble.circuitBreakerState;

    // Reset circuit breaker if cooldown period has passed
    if (state.isOpen && now - state.lastFailureTime > state.cooldownPeriod) {
      logger.info({
        msg: 'Circuit breaker cooldown period elapsed, resetting',
        component: 'EvolutionTriggerBubble',
      });
      state.isOpen = false;
      state.failures = 0;
    }

    return state.isOpen;
  }

  /**
   * Record successful operation
   */
  private static recordSuccess(): void {
    const state = EvolutionTriggerBubble.circuitBreakerState;
    state.failures = Math.max(0, state.failures - 1);

    logger.debug({
      msg: 'Recorded success, circuit breaker failures reduced',
      component: 'EvolutionTriggerBubble',
      remaining_failures: state.failures,
    });
  }

  /**
   * Record failed operation
   */
  private static recordFailure(): void {
    const state = EvolutionTriggerBubble.circuitBreakerState;
    state.failures += 1;
    state.lastFailureTime = Date.now();

    // Open circuit breaker after 3 consecutive failures
    if (state.failures >= 3) {
      state.isOpen = true;

      logger.error({
        msg: 'Circuit breaker opened due to repeated failures',
        component: 'EvolutionTriggerBubble',
        failures: state.failures,
      });
    }
  }
}

export default EvolutionTriggerBubble;
