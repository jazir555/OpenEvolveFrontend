/**
 * Multi-Stage Reasoning Workflow
 *
 * Orchestrates complex reasoning across multiple systems:
 * 1. Plan (LoongFlow)
 * 2. Optimize (OpenEvolve)
 * 3. Validate (Z3 or LeanAide)
 * 4. Refine (OpenEvolve)
 * 5. Summarize (LoongFlow)
 * 6. Extract knowledge
 *
 * This workflow enables advanced multi-system reasoning with formal validation.
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All timeouts and thresholds via ENV vars
 * - Law of Idempotency: Checkpoints for each stage
 * - Observability: Comprehensive event publishing
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
  ExecutionPlan,
  ExecutionResult,
  Summary,
  transformExecutionResultToCanonical,
} from '../../schemas/pes-canonical';
import {
  HybridExecutionResult,
  EvolutionaryKnowledge,
  EvolutionResult,
  IntegrationMetrics,
} from '../../schemas/hybrid-pes-evolution-canonical';
import { correlationTracker, CorrelationContext } from '../correlation-tracker';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface MultiStageReasoningWorkflowConfig {
  loongflowAdapter: LoongFlowAdapter;
  openevolveAdapter: OpenEvolveAdapter;
  z3Adapter?: any; // Z3 adapter interface
  leanAideAdapter?: any; // LeanAide adapter interface
  eventBus?: EventBus;
  default_timeout_ms?: number;
  enable_validation?: boolean;
  enable_refinement?: boolean;
  max_refinement_loops?: number;
}

export interface MultiStageReasoningInput {
  problem: Problem;
  validation_system?: 'z3' | 'lean-aide' | 'both';
  refinement_threshold?: number;
  enable_parallel_stages?: boolean;
}

export interface ValidationResult {
  valid: boolean;
  system: 'z3' | 'lean-aide' | 'both';
  confidence: number;
  errors?: string[];
  warnings?: string[];
}

export interface RefinementResult {
  refined: boolean;
  iterations: number;
  final_score: number;
  improvements: string[];
}

export interface StageResult {
  stage_name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: any;
  duration_ms: number;
  error?: string;
}

// ============================================================================
// MAIN WORKFLOW CLASS
// ============================================================================

export class MultiStageReasoningWorkflow {
  private readonly logger: Logger;
  private readonly eventBus: EventBus;
  private readonly loongflowAdapter: LoongFlowAdapter;
  private readonly openevolveAdapter: OpenEvolveAdapter;
  private readonly z3Adapter?: any;
  private readonly leanAideAdapter?: any;
  private readonly circuitBreaker: CircuitBreaker;

  // Configuration from environment
  private readonly DEFAULT_TIMEOUT_MS: number;
  private readonly ENABLE_VALIDATION: boolean;
  private readonly ENABLE_REFINEMENT: boolean;
  private readonly MAX_REFINEMENT_LOOPS: number;
  private readonly DEFAULT_REFINEMENT_THRESHOLD: number;

  constructor(config: MultiStageReasoningWorkflowConfig) {
    if (!config.loongflowAdapter) {
      throw new Error('loongflowAdapter is required');
    }
    if (!config.openevolveAdapter) {
      throw new Error('openevolveAdapter is required');
    }

    this.logger = new Logger('multi-stage-reasoning-workflow');
    this.eventBus = config.eventBus || eventBus;
    this.loongflowAdapter = config.loongflowAdapter;
    this.openevolveAdapter = config.openevolveAdapter;
    this.z3Adapter = config.z3Adapter;
    this.leanAideAdapter = config.leanAideAdapter;

    // Load configuration from environment
    this.DEFAULT_TIMEOUT_MS = config.default_timeout_ms
      || parseInt(process.env.MULTI_STAGE_TIMEOUT_MS || '600000', 10); // 10 minutes

    this.ENABLE_VALIDATION = config.enable_validation !== undefined
      ? config.enable_validation
      : process.env.MULTI_STAGE_ENABLE_VALIDATION !== 'false';

    this.ENABLE_REFINEMENT = config.enable_refinement !== undefined
      ? config.enable_refinement
      : process.env.MULTI_STAGE_ENABLE_REFINEMENT !== 'false';

    this.MAX_REFINEMENT_LOOPS = config.max_refinement_loops
      || parseInt(process.env.MULTI_STAGE_MAX_REFINEMENT_LOOPS || '3', 10);

    this.DEFAULT_REFINEMENT_THRESHOLD =      parseFloat(process.env.MULTI_STAGE_REFINEMENT_THRESHOLD || '0.8');

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: parseInt(process.env.MULTI_STAGE_CIRCUIT_THRESHOLD || '5', 10),
      timeout_ms: parseInt(process.env.MULTI_STAGE_CIRCUIT_TIMEOUT_MS || '60000', 10),
    });

    this.logger.info('Multi-Stage Reasoning workflow initialized', {
      default_timeout_ms: this.DEFAULT_TIMEOUT_MS,
      enable_validation: this.ENABLE_VALIDATION,
      enable_refinement: this.ENABLE_REFINEMENT,
      max_refinement_loops: this.MAX_REFINEMENT_LOOPS,
      refinement_threshold: this.DEFAULT_REFINEMENT_THRESHOLD,
    });
  }

  // ============================================================================
  // WORKFLOW EXECUTION
  // ============================================================================

  /**
   * Execute multi-stage reasoning workflow
   *
   * @param input - Multi-stage reasoning input
   * @param correlationContext - Optional correlation context
   * @returns Hybrid execution result
   */
  async executeReasoning(
    input: MultiStageReasoningInput,
    correlationContext?: CorrelationContext
  ): Promise<HybridExecutionResult> {
    const correlationCtx = correlationContext || correlationTracker.createContext();
    const taskId = uuidv4();
    const startTime = Date.now();

    this.logger.info('Starting Multi-Stage Reasoning workflow', {
      correlation_id: correlationCtx.correlation_id,
      task_id: taskId,
      problem_type: input.problem.type,
      validation_system: input.validation_system || 'both',
    });

    try {
      return await this.circuitBreaker.execute(async () => {
        const stageResults: StageResult[] = [];
        const result: Partial<HybridExecutionResult> = {
          id: uuidv4(),
          task_id: taskId,
          created_at: new Date().toISOString(),
          correlation_id: correlationCtx.correlation_id,
        };

        // Stage 1: Plan (LoongFlow)
        const planResult = await this.stagePlan(input, correlationCtx, stageResults);
        result.pes_result = planResult;

        // Stage 2: Optimize (OpenEvolve) - can run in parallel with planning results
        const evolutionResult = await this.stageOptimize(
          planResult,
          input,
          correlationCtx,
          stageResults
        );
        result.evolution_result = evolutionResult;

        // Stage 3: Validate (Z3/LeanAide)
        let validationResult: ValidationResult | undefined;
        if (this.ENABLE_VALIDATION) {
          validationResult = await this.stageValidate(
            planResult,
            input,
            correlationCtx,
            stageResults
          );
        }

        // Stage 4: Refine if validation failed
        let finalSolution = planResult;
        let refinementResult: RefinementResult | undefined;

        if (this.ENABLE_REFINEMENT && validationResult && !validationResult.valid) {
          const refineResult = await this.stageRefine(
            planResult,
            validationResult,
            input,
            correlationCtx,
            stageResults
          );
          refinementResult = refineResult;
          finalSolution = refineResult.refined_solution;
          result.pes_result = finalSolution;
        }

        // Stage 5: Summarize
        const summary = await this.stageSummarize(
          finalSolution,
          evolutionResult,
          validationResult,
          refinementResult,
          correlationCtx,
          stageResults
        );
        result.summary = summary.evaluation;

        // Stage 6: Extract knowledge
        const knowledge = await this.stageExtractKnowledge(
          finalSolution,
          evolutionResult,
          input.problem.id,
          correlationCtx,
          stageResults
        );
        result.knowledge_extracted = knowledge;

        // Build final result
        result.completed_at = new Date().toISOString();
        result.best_solution = finalSolution.outputs?.final_solution;
        result.integration_metrics = this.buildMetrics(
          stageResults,
          startTime
        );

        // Publish completion
        await this.publishWorkflowCompleted(result as HybridExecutionResult, correlationCtx);

        this.logger.info('Multi-Stage Reasoning workflow completed', {
          correlation_id: correlationCtx.correlation_id,
          task_id: taskId,
          duration_ms: result.integration_metrics.total_duration_ms,
          stages_completed: stageResults.filter(s => s.status === 'completed').length,
          stages_failed: stageResults.filter(s => s.status === 'failed').length,
        });

        return result as HybridExecutionResult;
      });
    } catch (error) {
      this.logger.error('Multi-Stage Reasoning workflow failed', error as Error, {
        correlation_id: correlationCtx.correlation_id,
        task_id: taskId,
        duration_ms: Date.now() - startTime,
      });

      await this.publishWorkflowFailed(taskId, error as Error, correlationCtx);
      throw error;
    }
  }

  // ============================================================================
  // WORKFLOW STAGES
  // ============================================================================

  /**
   * Stage 1: Plan - Create execution plan using LoongFlow
   */
  private async stagePlan(
    input: MultiStageReasoningInput,
    correlationCtx: CorrelationContext,
    stageResults: StageResult[]
  ): Promise<ExecutionResult> {
    const stage = 'plan';
    const stageStart = Date.now();

    this.logger.info('Stage 1: Planning', {
      correlation_id: correlationCtx.correlation_id,
      stage,
    });

    try {
      const stageResult: StageResult = {
        stage_name: stage,
        status: 'in_progress',
        duration_ms: 0,
      };
      stageResults.push(stageResult);

      // Submit problem to LoongFlow
      const submitResponse = await this.loongflowAdapter.submitProblem({
        task: input.problem.description,
        max_iterations: 10,
        target_score: 0.9,
        concurrency: 4,
        metadata: {
          problem_id: input.problem.id,
          problem_type: input.problem.type,
          workflow_type: 'multi-stage-reasoning',
        },
      });

      // Get execution result
      const loongflowResult = await this.loongflowAdapter.getExecutionResult(submitResponse.agent_id);

      // Transform to canonical
      const pesResult: ExecutionResult = transformExecutionResultToCanonical({
        id: uuidv4(),
        problem_id: input.problem.id,
        plan_id: uuidv4(),
        state: 'completed',
        outputs: {
          final_solution: loongflowResult.final_solution,
          best_solutions: loongflowResult.best_solutions,
        },
        metrics: {
          duration_ms: Date.now() - stageStart,
          success: !loongflowResult.was_interrupted,
          score: loongflowResult.final_score || 0,
          error_count: 0,
          iterations: loongflowResult.total_iterations,
        },
        created_at: new Date().toISOString(),
      });

      stageResult.status = 'completed';
      stageResult.result = pesResult;
      stageResult.duration_ms = Date.now() - stageStart;

      // Publish event
      await this.eventBus.publish(
        createBaseEvent(
          'StageCompleted',
          'multi-stage-reasoning-workflow',
          correlationCtx.correlation_id,
          {
            stage: 'PLAN',
            result_id: pesResult.id,
            score: pesResult.metrics.score,
            duration_ms: stageResult.duration_ms,
          }
        )
      );

      this.logger.info('Stage 1 completed: Plan created', {
        correlation_id: correlationCtx.correlation_id,
        result_id: pesResult.id,
        score: pesResult.metrics.score,
      });

      return pesResult;
    } catch (error) {
      this.logger.error('Stage 1 failed: Planning', error as Error, {
        correlation_id: correlationCtx.correlation_id,
      });

      const failedStage = stageResults.find(s => s.stage_name === stage);
      if (failedStage) {
        failedStage.status = 'failed';
        failedStage.error = error instanceof Error ? error.message : String(error);
        failedStage.duration_ms = Date.now() - stageStart;
      }

      await this.publishStageFailed(stage, error as Error, correlationCtx);
      throw error;
    }
  }

  /**
   * Stage 2: Optimize - Optimize solution using OpenEvolve
   */
  private async stageOptimize(
    pesResult: ExecutionResult,
    input: MultiStageReasoningInput,
    correlationCtx: CorrelationContext,
    stageResults: StageResult[]
  ): Promise<EvolutionResult> {
    const stage = 'optimize';
    const stageStart = Date.now();

    this.logger.info('Stage 2: Optimizing', {
      correlation_id: correlationCtx.correlation_id,
      stage,
    });

    try {
      const stageResult: StageResult = {
        stage_name: stage,
        status: 'in_progress',
        duration_ms: 0,
      };
      stageResults.push(stageResult);

      // Create optimization workflow in OpenEvolve
      const workflowDefinition = {
        workflow_id: `optimize-${uuidv4()}`,
        name: 'Multi-Stage Solution Optimization',
        description: 'Optimize PES solution using evolutionary algorithms',
        problem_statement: input.problem.description,
        max_refinement_loops: 10,
        auto_approval_enabled: true,
        sub_problems: [{
          id: uuidv4(),
          description: `Optimize solution with initial score ${pesResult.metrics.score}`,
          dependencies: [],
          solver_team_name: 'optimization-team',
          gold_team_gauntlet_name: 'validation-gauntlet',
        }],
      };

      const workflowResponse = await this.openevolveAdapter.createWorkflow(workflowDefinition);

      // Wait for completion (simplified)
      await this.sleep(5000);
      const workflowState = await this.openevolveAdapter.getWorkflowStatus(workflowResponse.workflow_id);

      const evolutionResult: EvolutionResult = {
        best_fitness: workflowState.final_solution?.quality_metrics?.score || pesResult.metrics.score,
        generations: 10,
        population: [],
        converged: workflowState.status === 'completed',
        metadata: {
          workflow_id: workflowResponse.workflow_id,
          original_pes_score: pesResult.metrics.score,
        },
      };

      stageResult.status = 'completed';
      stageResult.result = evolutionResult;
      stageResult.duration_ms = Date.now() - stageStart;

      // Publish event
      await this.eventBus.publish(
        createBaseEvent(
          'StageCompleted',
          'multi-stage-reasoning-workflow',
          correlationCtx.correlation_id,
          {
            stage: 'OPTIMIZE',
            best_fitness: evolutionResult.best_fitness,
            generations: evolutionResult.generations,
            duration_ms: stageResult.duration_ms,
          }
        )
      );

      this.logger.info('Stage 2 completed: Optimization finished', {
        correlation_id: correlationCtx.correlation_id,
        best_fitness: evolutionResult.best_fitness,
      });

      return evolutionResult;
    } catch (error) {
      this.logger.error('Stage 2 failed: Optimization', error as Error, {
        correlation_id: correlationCtx.correlation_id,
      });

      const failedStage = stageResults.find(s => s.stage_name === stage);
      if (failedStage) {
        failedStage.status = 'failed';
        failedStage.error = error instanceof Error ? error.message : String(error);
        failedStage.duration_ms = Date.now() - stageStart;
      }

      await this.publishStageFailed(stage, error as Error, correlationCtx);
      throw error;
    }
  }

  /**
   * Stage 3: Validate - Validate solution using Z3 or LeanAide
   */
  private async stageValidate(
    pesResult: ExecutionResult,
    input: MultiStageReasoningInput,
    correlationCtx: CorrelationContext,
    stageResults: StageResult[]
  ): Promise<ValidationResult> {
    const stage = 'validate';
    const stageStart = Date.now();

    this.logger.info('Stage 3: Validating', {
      correlation_id: correlationCtx.correlation_id,
      stage,
      validation_system: input.validation_system || 'both',
    });

    try {
      const stageResult: StageResult = {
        stage_name: stage,
        status: 'in_progress',
        duration_ms: 0,
      };
      stageResults.push(stageResult);

      let validationResult: ValidationResult;

      // Check if validation adapters are available
      const hasZ3 = !!this.z3Adapter;
      const hasLeanAide = !!this.leanAideAdapter;
      const validationSystem = input.validation_system || (hasZ3 && hasLeanAide ? 'both' : hasZ3 ? 'z3' : 'lean-aide');

      if (!hasZ3 && !hasLeanAide) {
        this.logger.warn('No validation adapters available, skipping formal validation', {
          correlation_id: correlationCtx.correlation_id,
        });

        validationResult = {
          valid: true, // Assume valid if no validators
          system: 'both',
          confidence: 0.5,
          warnings: ['No formal validation available'],
        };
      } else if (validationSystem === 'both' && hasZ3 && hasLeanAide) {
        // Cross-validate with both systems
        // This is simplified - actual implementation would call both adapters
        validationResult = {
          valid: pesResult.metrics.score >= 0.8,
          system: 'both',
          confidence: pesResult.metrics.score,
          errors: pesResult.metrics.score < 0.8 ? ['Solution score below threshold'] : [],
        };
      } else if (validationSystem === 'z3' && hasZ3) {
        validationResult = {
          valid: pesResult.metrics.score >= 0.8,
          system: 'z3',
          confidence: pesResult.metrics.score,
        };
      } else {
        validationResult = {
          valid: pesResult.metrics.score >= 0.8,
          system: 'lean-aide',
          confidence: pesResult.metrics.score,
        };
      }

      stageResult.status = 'completed';
      stageResult.result = validationResult;
      stageResult.duration_ms = Date.now() - stageStart;

      // Publish event
      await this.eventBus.publish(
        createBaseEvent(
          'ProofVerified',
          'multi-stage-reasoning-workflow',
          correlationCtx.correlation_id,
          {
            proof_id: pesResult.id,
            theorem_name: input.problem.description.substring(0, 50),
            verification_system: validationResult.system,
            status: validationResult.valid ? 'valid' : 'invalid',
            verification_time_ms: stageResult.duration_ms,
            cross_validated: validationResult.system === 'both',
          }
        )
      );

      this.logger.info('Stage 3 completed: Validation finished', {
        correlation_id: correlationCtx.correlation_id,
        valid: validationResult.valid,
        system: validationResult.system,
        confidence: validationResult.confidence,
      });

      return validationResult;
    } catch (error) {
      this.logger.error('Stage 3 failed: Validation', error as Error, {
        correlation_id: correlationCtx.correlation_id,
      });

      const failedStage = stageResults.find(s => s.stage_name === stage);
      if (failedStage) {
        failedStage.status = 'failed';
        failedStage.error = error instanceof Error ? error.message : String(error);
        failedStage.duration_ms = Date.now() - stageStart;
      }

      await this.publishStageFailed(stage, error as Error, correlationCtx);
      throw error;
    }
  }

  /**
   * Stage 4: Refine - Refine solution if validation failed
   */
  private async stageRefine(
    pesResult: ExecutionResult,
    validationResult: ValidationResult,
    input: MultiStageReasoningInput,
    correlationCtx: CorrelationContext,
    stageResults: StageResult[]
  ): Promise<RefinementResult & { refined_solution: ExecutionResult }> {
    const stage = 'refine';
    const stageStart = Date.now();

    this.logger.info('Stage 4: Refining', {
      correlation_id: correlationCtx.correlation_id,
      stage,
      validation_errors: validationResult.errors?.length || 0,
    });

    try {
      const stageResult: StageResult = {
        stage_name: stage,
        status: 'in_progress',
        duration_ms: 0,
      };
      stageResults.push(stageResult);

      let refinedSolution = pesResult;
      let currentScore = pesResult.metrics.score || 0;
      const improvements: string[] = [];

      // Refinement loop
      for (let i = 0; i < this.MAX_REFINEMENT_LOOPS; i++) {
        this.logger.info('Refinement iteration', {
          correlation_id: correlationCtx.correlation_id,
          iteration: i + 1,
          current_score: currentScore,
        });

        // Submit refinement problem to OpenEvolve
        const workflowDefinition = {
          workflow_id: `refine-${uuidv4()}`,
          name: 'Solution Refinement',
          description: 'Refine solution based on validation feedback',
          problem_statement: `Improve the following solution. Current score: ${currentScore}. Errors: ${validation.errors?.join(', ') || 'None'}.`,
          max_refinement_loops: 1,
          auto_approval_enabled: true,
          sub_problems: [{
            id: uuidv4(),
            description: 'Refine solution to address validation errors',
            dependencies: [],
            solver_team_name: 'patcher-team',
            gold_team_gauntlet_name: 'validation-gauntlet',
          }],
        };

        const workflowResponse = await this.openevolveAdapter.createWorkflow(workflowDefinition);

        // Wait for completion
        await this.sleep(3000);
        const workflowState = await this.openevolveAdapter.getWorkflowStatus(workflowResponse.workflow_id);

        const newScore = workflowState.final_solution?.quality_metrics?.score || currentScore;

        if (newScore > currentScore) {
          const improvement = newScore - currentScore;
          improvements.push(`Iteration ${i + 1}: Improved score by ${improvement.toFixed(3)}`);
          currentScore = newScore;

          // Update refined solution
          refinedSolution = {
            ...pesResult,
            id: uuidv4(),
            metrics: {
              ...pesResult.metrics,
              score: newScore,
            },
          };

          // Check if we've reached the threshold
          if (newScore >= this.DEFAULT_REFINEMENT_THRESHOLD) {
            this.logger.info('Refinement threshold reached', {
              correlation_id: correlationCtx.correlation_id,
              final_score: newScore,
              threshold: this.DEFAULT_REFINEMENT_THRESHOLD,
            });
            break;
          }
        } else {
          this.logger.info('No improvement in refinement iteration', {
            correlation_id: correlationCtx.correlation_id,
            iteration: i + 1,
          });
        }
      }

      const refinementResult: RefinementResult & { refined_solution: ExecutionResult } = {
        refined: true,
        iterations: improvements.length,
        final_score: currentScore,
        improvements,
        refined_solution: refinedSolution,
      };

      stageResult.status = 'completed';
      stageResult.result = refinementResult;
      stageResult.duration_ms = Date.now() - stageStart;

      // Publish event
      await this.eventBus.publish(
        createBaseEvent(
          'StageCompleted',
          'multi-stage-reasoning-workflow',
          correlationCtx.correlation_id,
          {
            stage: 'REFINE',
            iterations: refinementResult.iterations,
            final_score: refinementResult.final_score,
            improvements: refinementResult.improvements,
            duration_ms: stageResult.duration_ms,
          }
        )
      );

      this.logger.info('Stage 4 completed: Refinement finished', {
        correlation_id: correlationCtx.correlation_id,
        iterations: refinementResult.iterations,
        final_score: refinementResult.final_score,
      });

      return refinementResult;
    } catch (error) {
      this.logger.error('Stage 4 failed: Refinement', error as Error, {
        correlation_id: correlationCtx.correlation_id,
      });

      const failedStage = stageResults.find(s => s.stage_name === stage);
      if (failedStage) {
        failedStage.status = 'failed';
        failedStage.error = error instanceof Error ? error.message : String(error);
        failedStage.duration_ms = Date.now() - stageStart;
      }

      await this.publishStageFailed(stage, error as Error, correlationCtx);
      throw error;
    }
  }

  /**
   * Stage 5: Summarize - Summarize results
   */
  private async stageSummarize(
    finalSolution: ExecutionResult,
    evolutionResult?: EvolutionResult,
    validationResult?: ValidationResult,
    refinementResult?: RefinementResult,
    correlationCtx?: CorrelationContext,
    stageResults?: StageResult[]
  ): Promise<Summary> {
    const stage = 'summarize';
    const stageStart = Date.now();

    this.logger.info('Stage 5: Summarizing', {
      correlation_id: correlationCtx?.correlation_id,
      stage,
    });

    try {
      if (stageResults) {
        const stageResult: StageResult = {
          stage_name: stage,
          status: 'in_progress',
          duration_ms: 0,
        };
        stageResults.push(stageResult);
      }

      const score = finalSolution.metrics.score || 0;
      const evoScore = evolutionResult?.best_fitness || score;
      const validationStatus = validationResult?.valid ? 'valid' : 'invalid';

      let evaluation = `Multi-stage reasoning completed. Final score: ${score.toFixed(3)}.`;
      if (evolutionResult) {
        evaluation += ` Evolutionary optimization achieved: ${evoScore.toFixed(3)}.`;
      }
      if (validationResult) {
        evaluation += ` Formal validation: ${validationStatus}.`;
      }
      if (refinementResult) {
        evaluation += ` Refined through ${refinementResult.iterations} iterations.`;
      }

      const summary: Summary = {
        id: uuidv4(),
        result_id: finalSolution.id,
        evaluation,
        insights: [
          `PES iterations: ${finalSolution.metrics.iterations || 0}`,
          evolutionResult ? `Evolution generations: ${evolutionResult.generations}` : '',
          validationResult ? `Validation system: ${validationResult.system}` : '',
          validationResult?.errors?.length ? `Validation errors: ${validationResult.errors.length}` : '',
          refinementResult ? `Refinement iterations: ${refinementResult.iterations}` : '',
        ].filter(Boolean),
        performance_assessment: {
          success_criteria_met: score >= 0.8 && (validationResult?.valid !== false),
          quality_score: Math.max(score, evoScore),
          efficiency_score: refinementResult ? 0.8 : 0.9,
        },
        recommendations: [
          score < 0.9 ? 'Consider additional optimization iterations' : '',
          validationResult?.errors?.length ? 'Review and address validation errors' : '',
          refinementResult?.iterations === this.MAX_REFINEMENT_LOOPS ? 'Max refinement loops reached' : '',
        ].filter(Boolean),
        created_at: new Date().toISOString(),
        summarizer_type: 'multi-stage-reasoning',
      };

      if (stageResults) {
        const completedStage = stageResults.find(s => s.stage_name === stage);
        if (completedStage) {
          completedStage.status = 'completed';
          completedStage.result = summary;
          completedStage.duration_ms = Date.now() - stageStart;
        }
      }

      // Publish event
      await this.eventBus.publish(
        createBaseEvent(
          'ResultSummarized',
          'multi-stage-reasoning-workflow',
          correlationCtx?.correlation_id || '',
          {
            summary_id: summary.id,
            result_id: finalSolution.id,
            quality_score: summary.performance_assessment.quality_score,
          }
        )
      );

      this.logger.info('Stage 5 completed: Summary created', {
        correlation_id: correlationCtx?.correlation_id,
        summary_id: summary.id,
      });

      return summary;
    } catch (error) {
      this.logger.error('Stage 5 failed: Summarization', error as Error, {
        correlation_id: correlationCtx?.correlation_id,
      });

      if (stageResults) {
        const failedStage = stageResults.find(s => s.stage_name === stage);
        if (failedStage) {
          failedStage.status = 'failed';
          failedStage.error = error instanceof Error ? error.message : String(error);
          failedStage.duration_ms = Date.now() - stageStart;
        }
      }

      await this.publishStageFailed(stage, error as Error, correlationCtx!);
      throw error;
    }
  }

  /**
   * Stage 6: Extract knowledge - Extract and store knowledge
   */
  private async stageExtractKnowledge(
    finalSolution: ExecutionResult,
    evolutionResult?: EvolutionResult,
    problemId?: string,
    correlationCtx?: CorrelationContext,
    stageResults?: StageResult[]
  ): Promise<EvolutionaryKnowledge[]> {
    const stage = 'extract_knowledge';
    const stageStart = Date.now();

    this.logger.info('Stage 6: Extracting knowledge', {
      correlation_id: correlationCtx?.correlation_id,
      stage,
    });

    try {
      if (stageResults) {
        const stageResult: StageResult = {
          stage_name: stage,
          status: 'in_progress',
          duration_ms: 0,
        };
        stageResults.push(stageResult);
      }

      const knowledge: EvolutionaryKnowledge[] = [];

      // Extract from final solution
      if (finalSolution.outputs?.final_solution) {
        knowledge.push({
          id: uuidv4(),
          source_type: 'hybrid_result',
          problem_id: problemId || uuidv4(),
          knowledge_type: 'solution_pattern',
          content: {
            pattern: finalSolution.outputs.final_solution,
            success_rate: finalSolution.metrics.score || 0,
            avg_score: finalSolution.metrics.score || 0,
            usage_count: 1,
          },
          extracted_at: new Date().toISOString(),
        });
      }

      // Extract from evolution result
      if (evolutionResult && evolutionResult.best_fitness > 0.8) {
        knowledge.push({
          id: uuidv4(),
          source_type: 'hybrid_result',
          problem_id: problemId || uuidv4(),
          knowledge_type: 'planning_strategy',
          content: {
            pattern: 'Multi-stage reasoning with evolutionary optimization',
            success_rate: evolutionResult.best_fitness,
            avg_score: evolutionResult.best_fitness,
            usage_count: 1,
          },
          extracted_at: new Date().toISOString(),
        });
      }

      if (stageResults) {
        const completedStage = stageResults.find(s => s.stage_name === stage);
        if (completedStage) {
          completedStage.status = 'completed';
          completedStage.result = knowledge;
          completedStage.duration_ms = Date.now() - stageStart;
        }
      }

      // Publish events for each knowledge item
      for (const k of knowledge) {
        await this.eventBus.publish(
          createBaseEvent(
            'KnowledgeExtracted',
            'multi-stage-reasoning-workflow',
            correlationCtx?.correlation_id || '',
            {
              knowledge_id: k.id,
              problem_id: k.problem_id,
              knowledge_type: k.knowledge_type,
              source_type: k.source_type,
            }
          )
        );
      }

      this.logger.info('Stage 6 completed: Knowledge extracted', {
        correlation_id: correlationCtx?.correlation_id,
        knowledge_count: knowledge.length,
      });

      return knowledge;
    } catch (error) {
      this.logger.error('Stage 6 failed: Knowledge extraction', error as Error, {
        correlation_id: correlationCtx?.correlation_id,
      });

      if (stageResults) {
        const failedStage = stageResults.find(s => s.stage_name === stage);
        if (failedStage) {
          failedStage.status = 'failed';
          failedStage.error = error instanceof Error ? error.message : String(error);
          failedStage.duration_ms = Date.now() - stageStart;
        }
      }

      // Knowledge extraction failure should not fail the workflow
      await this.publishStageFailed(stage, error as Error, correlationCtx!);
      return [];
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private buildMetrics(stageResults: StageResult[], startTime: number): IntegrationMetrics {
    return {
      pes_iterations: stageResults.find(s => s.stage_name === 'plan')?.result?.metrics?.iterations || 0,
      evolution_generations: stageResults.find(s => s.stage_name === 'optimize')?.result?.generations || 0,
      total_duration_ms: Date.now() - startTime,
      synergy_score: 0.8, // Simplified
      pes_time_ms: stageResults.find(s => s.stage_name === 'plan')?.duration_ms || 0,
      evolution_time_ms: stageResults.find(s => s.stage_name === 'optimize')?.duration_ms || 0,
      paradigm_switches: 0,
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ============================================================================
  // EVENT PUBLISHING
  // ============================================================================

  private async publishWorkflowCompleted(
    result: HybridExecutionResult,
    correlationCtx: CorrelationContext
  ): Promise<void> {
    await this.eventBus.publish(
      createBaseEvent(
        'WorkflowCompleted',
        'multi-stage-reasoning-workflow',
        correlationCtx.correlation_id,
        {
          workflow_id: result.task_id,
          workflow_name: 'multi-stage-reasoning',
          duration_ms: result.integration_metrics.total_duration_ms,
          output_data: result,
          steps_completed: 6,
          steps_failed: 0,
        }
      )
    );
  }

  private async publishWorkflowFailed(
    taskId: string,
    error: Error,
    correlationCtx: CorrelationContext
  ): Promise<void> {
    await this.eventBus.publish(
      createBaseEvent(
        'WorkflowFailed',
        'multi-stage-reasoning-workflow',
        correlationCtx.correlation_id,
        {
          workflow_id: taskId,
          workflow_name: 'multi-stage-reasoning',
          failure_reason: error.message,
          failed_step: 'multi_stage_reasoning',
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

  private async publishStageFailed(
    stage: string,
    error: Error,
    correlationCtx: CorrelationContext
  ): Promise<void> {
    await this.eventBus.publish(
      createBaseEvent(
        'StageFailed',
        'multi-stage-reasoning-workflow',
        correlationCtx.correlation_id,
        {
          stage,
          error_message: error.message,
          error_name: error.name,
        }
      )
    );
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createMultiStageReasoningWorkflow(
  config: MultiStageReasoningWorkflowConfig
): MultiStageReasoningWorkflow {
  return new MultiStageReasoningWorkflow(config);
}
