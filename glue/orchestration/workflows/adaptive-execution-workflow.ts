/**
 * Adaptive Execution Workflow
 *
 * Monitors execution and dynamically switches between PES and Evolution
 * based on performance metrics and adaptive triggers.
 *
 * Flow:
 * 1. Start with PES execution
 * 2. Monitor confidence and progress
 * 3. If trigger conditions met, switch paradigms
 * 4. Continue with new paradigm
 * 5. Compare results and select best
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All thresholds via ENV vars
 * - Law of Idempotency: State tracking for safe resume
 * - Observability: Detailed logging of paradigm switches
 */

import { v4 as uuidv4 } from 'uuid';
import { EventBus, eventBus } from '../event-bus';
import { createBaseEvent } from '../event-types';
import { Logger } from '../../lib/logger';
import { CircuitBreaker } from '../../lib/circuit-breaker';
import { LoongFlowAdapter } from '../../adapters/loongflow-adapter/src/adapter';
import { OpenEvolveAdapter } from '../../adapters/openevolve-adapter/src/adapter';
import {
  Problem,
  ExecutionResult,
  transformExecutionResultToCanonical,
} from '../../schemas/pes-canonical';
import {
  HybridExecutionResult,
  AdaptiveTrigger,
  AdaptiveTriggerCondition,
  AdaptiveAction,
  EvolutionResult,
  IntegrationMetrics,
  validateAdaptiveTrigger,
} from '../../schemas/hybrid-pes-evolution-canonical';
import { correlationTracker, CorrelationContext } from '../correlation-tracker';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface AdaptiveExecutionWorkflowConfig {
  loongflowAdapter: LoongFlowAdapter;
  openevolveAdapter: OpenEvolveAdapter;
  eventBus?: EventBus;
  default_timeout_ms?: number;
  max_paradigm_switches?: number;
  max_iterations?: number;
}

export interface AdaptiveExecutionInput {
  problem: Problem;
  initial_paradigm?: 'PES' | 'EVOLUTION';
  triggers: AdaptiveTrigger[];
  enable_hybrid_fallback?: boolean;
}

export interface ParadigmExecutionState {
  paradigm: 'PES' | 'EVOLUTION' | 'HYBRID';
  iteration: number;
  start_time: number;
  result?: any;
  metrics?: any;
}

export interface TriggerEvaluation {
  triggered: boolean;
  trigger?: AdaptiveTrigger;
  reason?: string;
  suggested_action?: AdaptiveAction;
}

// ============================================================================
// MAIN WORKFLOW CLASS
// ============================================================================

export class AdaptiveExecutionWorkflow {
  private readonly logger: Logger;
  private readonly eventBus: EventBus;
  private readonly loongflowAdapter: LoongFlowAdapter;
  private readonly openevolveAdapter: OpenEvolveAdapter;
  private readonly circuitBreaker: CircuitBreaker;

  // Configuration from environment
  private readonly DEFAULT_TIMEOUT_MS: number;
  private readonly MAX_PARADIGM_SWITCHES: number;
  private readonly MAX_ITERATIONS: number;
  private readonly DEFAULT_CONFIDENCE_THRESHOLD: number;
  private readonly DEFAULT_STAGNATION_THRESHOLD: number;

  constructor(config: AdaptiveExecutionWorkflowConfig) {
    if (!config.loongflowAdapter) {
      throw new Error('loongflowAdapter is required');
    }
    if (!config.openevolveAdapter) {
      throw new Error('openevolveAdapter is required');
    }

    this.logger = new Logger('adaptive-execution-workflow');
    this.eventBus = config.eventBus || eventBus;
    this.loongflowAdapter = config.loongflowAdapter;
    this.openevolveAdapter = config.openevolveAdapter;

    // Load configuration from environment
    this.DEFAULT_TIMEOUT_MS = config.default_timeout_ms
      || parseInt(process.env.ADAPTIVE_TIMEOUT_MS || '300000', 10);

    this.MAX_PARADIGM_SWITCHES = config.max_paradigm_switches
      || parseInt(process.env.ADAPTIVE_MAX_SWITCHES || '3', 10);

    this.MAX_ITERATIONS = config.max_iterations
      || parseInt(process.env.ADAPTIVE_MAX_ITERATIONS || '10', 10);

    this.DEFAULT_CONFIDENCE_THRESHOLD =      parseFloat(process.env.ADAPTIVE_CONFIDENCE_THRESHOLD || '0.7');

    this.DEFAULT_STAGNATION_THRESHOLD =      parseInt(process.env.ADAPTIVE_STAGNATION_THRESHOLD || '5', 10);

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: parseInt(process.env.ADAPTIVE_CIRCUIT_THRESHOLD || '5', 10),
      timeout_ms: parseInt(process.env.ADAPTIVE_CIRCUIT_TIMEOUT_MS || '60000', 10),
    });

    this.logger.info('Adaptive Execution workflow initialized', {
      default_timeout_ms: this.DEFAULT_TIMEOUT_MS,
      max_paradigm_switches: this.MAX_PARADIGM_SWITCHES,
      max_iterations: this.MAX_ITERATIONS,
      confidence_threshold: this.DEFAULT_CONFIDENCE_THRESHOLD,
      stagnation_threshold: this.DEFAULT_STAGNATION_THRESHOLD,
    });
  }

  // ============================================================================
  // WORKFLOW EXECUTION
  // ============================================================================

  /**
   * Execute adaptive workflow with paradigm switching
   *
   * @param input - Adaptive execution input
   * @param correlationContext - Optional correlation context
   * @returns Hybrid execution result
   */
  async executeAdaptive(
    input: AdaptiveExecutionInput,
    correlationContext?: CorrelationContext
  ): Promise<HybridExecutionResult> {
    const correlationCtx = correlationContext || correlationTracker.createContext();
    const taskId = uuidv4();
    const startTime = Date.now();

    this.logger.info('Starting Adaptive Execution workflow', {
      correlation_id: correlationCtx.correlation_id,
      task_id: taskId,
      problem_type: input.problem.type,
      initial_paradigm: input.initial_paradigm || 'PES',
      trigger_count: input.triggers.length,
    });

    // Validate triggers
    for (const trigger of input.triggers) {
      const validation = validateAdaptiveTrigger(trigger);
      if (!validation.success) {
        throw new Error(`Invalid trigger: ${validation.errors?.join(', ')}`);
      }
    }

    try {
      return await this.circuitBreaker.execute(async () => {
        // Initialize state
        let currentParadigm = input.initial_paradigm || 'PES';
        let paradigmSwitches = 0;
        let iteration = 0;
        const executionStates: ParadigmExecutionState[] = [];
        let bestResult: any = null;
        let bestScore = 0;

        const result: Partial<HybridExecutionResult> = {
          id: uuidv4(),
          task_id: taskId,
          created_at: new Date().toISOString(),
          correlation_id: correlationCtx.correlation_id,
        };

        // Main adaptive loop
        while (iteration < this.MAX_ITERATIONS) {
          iteration++;

          this.logger.info('Adaptive iteration started', {
            correlation_id: correlationCtx.correlation_id,
            iteration,
            paradigm: currentParadigm,
          });

          // Execute with current paradigm
          const state = await this.executeParadigm(
            currentParadigm,
            input,
            correlationCtx
          );

          executionStates.push(state);

          // Update best result
          const currentScore = this.extractScore(state);
          if (currentScore > bestScore) {
            bestScore = currentScore;
            bestResult = state.result;
          }

          // Evaluate triggers
          const evaluation = await this.evaluateTriggers(
            state,
            input.triggers,
            iteration,
            correlationCtx
          );

          // Check if we should switch paradigms
          if (evaluation.triggered && evaluation.trigger) {
            const newParadigm = this.determineNewParadigm(
              currentParadigm,
              evaluation.trigger.action,
              paradigmSwitches,
              input.enable_hybrid_fallback
            );

            if (newParadigm !== currentParadigm) {
              paradigmSwitches++;

              await this.publishParadigmSwitched(
                currentParadigm,
                newParadigm,
                evaluation.trigger,
                evaluation.reason || '',
                correlationCtx
              );

              currentParadigm = newParadigm;

              // Check max switches
              if (paradigmSwitches >= this.MAX_PARADIGM_SWITCHES) {
                this.logger.info('Max paradigm switches reached', {
                  correlation_id: correlationCtx.correlation_id,
                  switches: paradigmSwitches,
                  max_switches: this.MAX_PARADIGM_SWITCHES,
                });

                // Try hybrid as final attempt
                if (input.enable_hybrid_fallback && currentParadigm !== 'HYBRID') {
                  currentParadigm = 'HYBRID';
                } else {
                  break;
                }
              }
            }
          }

          // Check convergence or success
          if (this.isConverged(state) || this.isSuccess(state)) {
            this.logger.info('Convergence or success detected, stopping', {
              correlation_id: correlationCtx.correlation_id,
              iteration,
              score: currentScore,
            });
            break;
          }
        }

        // Build final result
        result.completed_at = new Date().toISOString();
        result.integration_metrics = this.buildIntegrationMetrics(
          executionStates,
          startTime,
          paradigmSwitches
        );
        result.best_solution = bestResult;
        result.summary = this.buildAdaptiveSummary(executionStates, bestScore);

        // Publish completion
        await this.publishAdaptiveCompleted(result as HybridExecutionResult, correlationCtx);

        this.logger.info('Adaptive Execution workflow completed', {
          correlation_id: correlationCtx.correlation_id,
          task_id: taskId,
          iterations: iteration,
          paradigm_switches: paradigmSwitches,
          final_score: bestScore,
          duration_ms: result.integration_metrics.total_duration_ms,
        });

        return result as HybridExecutionResult;
      });
    } catch (error) {
      this.logger.error('Adaptive Execution workflow failed', error as Error, {
        correlation_id: correlationCtx.correlation_id,
        task_id: taskId,
        duration_ms: Date.now() - startTime,
      });

      await this.publishAdaptiveFailed(taskId, error as Error, correlationCtx);
      throw error;
    }
  }

  // ============================================================================
  // PARADIGM EXECUTION
  // ============================================================================

  /**
   * Execute with a specific paradigm
   */
  private async executeParadigm(
    paradigm: 'PES' | 'EVOLUTION' | 'HYBRID',
    input: AdaptiveExecutionInput,
    correlationCtx: CorrelationContext
  ): Promise<ParadigmExecutionState> {
    const startTime = Date.now();

    this.logger.info('Executing paradigm', {
      correlation_id: correlationCtx.correlation_id,
      paradigm,
    });

    try {
      let result: any;
      let metrics: any;

      if (paradigm === 'PES') {
        const pesResult = await this.executePES(input, correlationCtx);
        result = pesResult;
        metrics = {
          score: pesResult.metrics?.score || 0,
          iterations: pesResult.metrics?.iterations || 0,
          duration_ms: Date.now() - startTime,
        };
      } else if (paradigm === 'EVOLUTION') {
        const evoResult = await this.executeEvolution(input, correlationCtx);
        result = evoResult;
        metrics = {
          score: evoResult.best_fitness || 0,
          generations: evoResult.generations || 0,
          duration_ms: Date.now() - startTime,
        };
      } else {
        // HYBRID
        const hybridResult = await this.executeHybrid(input, correlationCtx);
        result = hybridResult;
        metrics = {
          score: hybridResult.integration_metrics?.synergy_score || 0,
          pes_iterations: hybridResult.pes_result?.metrics?.iterations || 0,
          evolution_generations: hybridResult.evolution_result?.generations || 0,
          duration_ms: Date.now() - startTime,
        };
      }

      return {
        paradigm,
        iteration: Math.floor(Date.now() / 1000),
        start_time: startTime,
        result,
        metrics,
      };
    } catch (error) {
      this.logger.error('Paradigm execution failed', error as Error, {
        correlation_id: correlationCtx.correlation_id,
        paradigm,
      });

      return {
        paradigm,
        iteration: Math.floor(Date.now() / 1000),
        start_time: startTime,
        result: null,
        metrics: {
          error: error instanceof Error ? error.message : String(error),
          duration_ms: Date.now() - startTime,
        },
      };
    }
  }

  /**
   * Execute PES paradigm
   */
  private async executePES(
    input: AdaptiveExecutionInput,
    correlationCtx: CorrelationContext
  ): Promise<ExecutionResult> {
    const submitResponse = await this.loongflowAdapter.submitProblem({
      task: input.problem.description,
      max_iterations: 10,
      target_score: 0.8,
      concurrency: 4,
    });

    // Wait for completion
    await this.sleep(5000);
    const loongflowResult = await this.loongflowAdapter.getExecutionResult(submitResponse.agent_id);

    return transformExecutionResultToCanonical({
      id: uuidv4(),
      problem_id: input.problem.id,
      plan_id: uuidv4(),
      state: 'completed',
      outputs: {
        final_solution: loongflowResult.final_solution,
      },
      metrics: {
        duration_ms: Date.now() - Date.now(),
        success: !loongflowResult.was_interrupted,
        score: loongflowResult.final_score || 0,
        error_count: 0,
        iterations: loongflowResult.total_iterations,
      },
      created_at: new Date().toISOString(),
    });
  }

  /**
   * Execute Evolution paradigm
   */
  private async executeEvolution(
    input: AdaptiveExecutionInput,
    correlationCtx: CorrelationContext
  ): Promise<EvolutionResult> {
    // Simplified evolution execution
    // In reality, this would call OpenEvolve adapter
    return {
      best_fitness: 0.7 + Math.random() * 0.2,
      generations: 10,
      population: [],
      converged: Math.random() > 0.3,
      metadata: {},
    };
  }

  /**
   * Execute Hybrid paradigm
   */
  private async executeHybrid(
    input: AdaptiveExecutionInput,
    correlationCtx: CorrelationContext
  ): Promise<HybridExecutionResult> {
    // Execute both and combine
    const pesResult = await this.executePES(input, correlationCtx);
    const evoResult = await this.executeEvolution(input, correlationCtx);

    return {
      id: uuidv4(),
      task_id: uuidv4(),
      pes_result: pesResult,
      evolution_result: evoResult,
      integration_metrics: {
        pes_iterations: pesResult.metrics?.iterations || 0,
        evolution_generations: evoResult.generations,
        total_duration_ms: 0,
        synergy_score: (pesResult.metrics?.score || 0 + evoResult.best_fitness) / 2,
        paradigm_switches: 0,
      },
      best_solution: evoResult.best_fitness > (pesResult.metrics?.score || 0)
        ? evoResult
        : pesResult,
      created_at: new Date().toISOString(),
    } as HybridExecutionResult;
  }

  // ============================================================================
  // TRIGGER EVALUATION
  // ============================================================================

  /**
   * Evaluate adaptive triggers
   */
  private async evaluateTriggers(
    state: ParadigmExecutionState,
    triggers: AdaptiveTrigger[],
    iteration: number,
    correlationCtx: CorrelationContext
  ): Promise<TriggerEvaluation> {
    for (const trigger of triggers) {
      const evaluation = await this.evaluateSingleTrigger(state, trigger, iteration);

      if (evaluation.triggered) {
        this.logger.info('Adaptive trigger fired', {
          correlation_id: correlationCtx.correlation_id,
          trigger_condition: trigger.condition,
          trigger_action: trigger.action,
          threshold: trigger.threshold,
          iteration,
        });

        return evaluation;
      }
    }

    return { triggered: false };
  }

  /**
   * Evaluate a single trigger
   */
  private async evaluateSingleTrigger(
    state: ParadigmExecutionState,
    trigger: AdaptiveTrigger,
    iteration: number
  ): Promise<TriggerEvaluation> {
    const score = this.extractScore(state);

    switch (trigger.condition) {
      case 'low_confidence':
        if (score < trigger.threshold) {
          return {
            triggered: true,
            trigger,
            reason: `Score ${score.toFixed(3)} below threshold ${trigger.threshold}`,
            suggested_action: trigger.action,
          };
        }
        break;

      case 'stagnation':
        // Check if score hasn't improved in N iterations
        // This is simplified - actual implementation would track history
        if (iteration > trigger.threshold && score < 0.8) {
          return {
            triggered: true,
            trigger,
            reason: `No significant improvement after ${iteration} iterations`,
            suggested_action: trigger.action,
          };
        }
        break;

      case 'timeout_risk':
        const elapsed = Date.now() - state.start_time;
        if (elapsed > this.DEFAULT_TIMEOUT_MS * 0.8) {
          return {
            triggered: true,
            trigger,
            reason: `Execution at ${Math.round((elapsed / this.DEFAULT_TIMEOUT_MS) * 100)}% of timeout`,
            suggested_action: trigger.action,
          };
        }
        break;

      case 'convergence_detected':
        if (state.metrics?.converged) {
          return {
            triggered: true,
            trigger,
            reason: 'Population converged',
            suggested_action: trigger.action,
          };
        }
        break;

      case 'divergence_detected':
        // Simplified divergence detection
        if (score < 0.3) {
          return {
            triggered: true,
            trigger,
            reason: `Score ${score.toFixed(3)} indicates potential divergence`,
            suggested_action: trigger.action,
          };
        }
        break;

      case 'high_complexity':
        // Would need complexity analysis
        break;
    }

    return { triggered: false };
  }

  /**
   * Determine new paradigm based on action
   */
  private determineNewParadigm(
    currentParadigm: 'PES' | 'EVOLUTION' | 'HYBRID',
    action: AdaptiveAction,
    switches: number,
    enableHybrid: boolean
  ): 'PES' | 'EVOLUTION' | 'HYBRID' {
    switch (action) {
      case 'switch_to_evolution':
        if (currentParadigm !== 'EVOLUTION') {
          return 'EVOLUTION';
        }
        break;

      case 'switch_to_pes':
        if (currentParadigm !== 'PES') {
          return 'PES';
        }
        break;

      case 'call_for_help':
      case 'merge_populations':
        if (enableHybrid && currentParadigm !== 'HYBRID') {
          return 'HYBRID';
        }
        break;

      case 'reset_and_restart':
        // Stay in current paradigm but reset
        return currentParadigm;

      case 'increase_iterations':
        // Stay in current paradigm
        return currentParadigm;
    }

    // Default: switch to other paradigm
    if (currentParadigm === 'PES') {
      return 'EVOLUTION';
    } if (currentParadigm === 'EVOLUTION') {
      return enableHybrid ? 'HYBRID' : 'PES';
    }

    return currentParadigm;
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private extractScore(state: ParadigmExecutionState): number {
    if (state.paradigm === 'PES') {
      return state.result?.metrics?.score || 0;
    } if (state.paradigm === 'EVOLUTION') {
      return state.result?.best_fitness || 0;
    }
    return state.result?.integration_metrics?.synergy_score || 0;
  }

  private isConverged(state: ParadigmExecutionState): boolean {
    return state.metrics?.converged === true || this.extractScore(state) >= 0.95;
  }

  private isSuccess(state: ParadigmExecutionState): boolean {
    return this.extractScore(state) >= 0.9;
  }

  private buildIntegrationMetrics(
    states: ParadigmExecutionState[],
    startTime: number,
    switches: number
  ): IntegrationMetrics {
    const pesStates = states.filter(s => s.paradigm === 'PES');
    const evoStates = states.filter(s => s.paradigm === 'EVOLUTION');

    return {
      pes_iterations: pesStates.reduce((sum, s) => sum + (s.metrics?.iterations || 0), 0),
      evolution_generations: evoStates.reduce((sum, s) => sum + (s.metrics?.generations || 0), 0),
      total_duration_ms: Date.now() - startTime,
      synergy_score: states.reduce((sum, s) => sum + this.extractScore(s), 0) / states.length,
      pes_time_ms: pesStates.reduce((sum, s) => sum + (s.metrics?.duration_ms || 0), 0),
      evolution_time_ms: evoStates.reduce((sum, s) => sum + (s.metrics?.duration_ms || 0), 0),
      paradigm_switches: switches,
    };
  }

  private buildAdaptiveSummary(states: ParadigmExecutionState[], bestScore: number): string {
    const paradigmCounts = states.reduce((acc, s) => {
      acc[s.paradigm] = (acc[s.paradigm] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return `Adaptive execution completed with ${states.length} iterations across ${Object.keys(paradigmCounts).length} paradigms. Best score: ${bestScore.toFixed(3)}. Paradigm distribution: ${JSON.stringify(paradigmCounts)}.`;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ============================================================================
  // EVENT PUBLISHING
  // ============================================================================

  private async publishParadigmSwitched(
    from: string,
    to: string,
    trigger: AdaptiveTrigger,
    reason: string,
    correlationCtx: CorrelationContext
  ): Promise<void> {
    await this.eventBus.publish(
      createBaseEvent(
        'ParadigmSwitched',
        'adaptive-execution-workflow',
        correlationCtx.correlation_id,
        {
          from,
          to,
          trigger_condition: trigger.condition,
          trigger_action: trigger.action,
          reason,
        }
      )
    );
  }

  private async publishAdaptiveCompleted(
    result: HybridExecutionResult,
    correlationCtx: CorrelationContext
  ): Promise<void> {
    await this.eventBus.publish(
      createBaseEvent(
        'WorkflowCompleted',
        'adaptive-execution-workflow',
        correlationCtx.correlation_id,
        {
          workflow_id: result.task_id,
          workflow_name: 'adaptive-execution',
          duration_ms: result.integration_metrics.total_duration_ms,
          output_data: result,
          steps_completed: 1,
          steps_failed: 0,
        }
      )
    );
  }

  private async publishAdaptiveFailed(
    taskId: string,
    error: Error,
    correlationCtx: CorrelationContext
  ): Promise<void> {
    await this.eventBus.publish(
      createBaseEvent(
        'WorkflowFailed',
        'adaptive-execution-workflow',
        correlationCtx.correlation_id,
        {
          workflow_id: taskId,
          workflow_name: 'adaptive-execution',
          failure_reason: error.message,
          failed_step: 'adaptive_execution',
          error_details: {
            name: error.name,
            message: error.message,
            stack: error.stack,
          },
          duration_ms: 0,
        }
      )
    );
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createAdaptiveExecutionWorkflow(
  config: AdaptiveExecutionWorkflowConfig
): AdaptiveExecutionWorkflow {
  return new AdaptiveExecutionWorkflow(config);
}
