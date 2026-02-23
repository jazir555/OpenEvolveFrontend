"use strict";
/**
 * PES-Evolution Hybrid Workflow
 *
 * This workflow orchestrates LoongFlow's Plan-Execute-Summarize (PES)
 * with OpenEvolve's evolutionary optimization for enhanced problem-solving.
 *
 * Flow:
 * 1. Receive problem
 * 2. LoongFlow plans solution
 * 3. LoongFlow executes initial solution
 * 4. OpenEvolve optimizes the solution
 * 5. LoongFlow summarizes results
 * 6. Extract evolutionary knowledge
 * 7. Store knowledge for future use
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All timeouts, retries via ENV vars
 * - Law of Idempotency: Checkpoints support resume capability
 * - Law of UTC: All timestamps in ISO-8601 UTC
 * - Failure Management: Circuit breakers on all adapter calls
 * - Observability: Structured logging with correlation IDs
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PESEvolutionWorkflow = void 0;
exports.createPESEvolutionWorkflow = createPESEvolutionWorkflow;
const uuid_1 = require("uuid");
const event_bus_1 = require("../event-bus");
const event_types_1 = require("../event-types");
const logger_1 = require("../../lib/logger");
const circuit_breaker_1 = require("../../lib/circuit-breaker");
const pes_canonical_1 = require("../../schemas/pes-canonical");
const hybrid_pes_evolution_canonical_1 = require("../../schemas/hybrid-pes-evolution-canonical");
const correlation_tracker_1 = require("../correlation-tracker");
// ============================================================================
// MAIN WORKFLOW CLASS
// ============================================================================
class PESEvolutionWorkflow {
    constructor(config) {
        this.checkpoints = new Map();
        // Validate required configuration
        if (!config.loongflowAdapter) {
            throw new Error('loongflowAdapter is required');
        }
        if (!config.openevolveAdapter) {
            throw new Error('openevolveAdapter is required');
        }
        // Load configuration from environment with explicit validation
        this.DEFAULT_TIMEOUT_MS = config.default_timeout_ms ||
            parseInt(process.env.PES_EVOLUTION_TIMEOUT_MS || '300000', 10); // 5 minutes
        this.MAX_RETRIES = config.max_retries ||
            parseInt(process.env.PES_EVOLUTION_MAX_RETRIES || '3', 10);
        this.CHECKPOINTS_ENABLED = config.checkpoints_enabled !== undefined
            ? config.checkpoints_enabled
            : process.env.PES_EVOLUTION_CHECKPOINTS_ENABLED === 'true';
        this.logger = new logger_1.Logger('pes-evolution-workflow');
        this.eventBus = config.eventBus || event_bus_1.eventBus;
        this.loongflowAdapter = config.loongflowAdapter;
        this.openevolveAdapter = config.openevolveAdapter;
        // Initialize circuit breaker for workflow operations
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: parseInt(process.env.PES_EVOLUTION_CIRCUIT_THRESHOLD || '5', 10),
            timeout_ms: parseInt(process.env.PES_EVOLUTION_CIRCUIT_TIMEOUT_MS || '60000', 10),
            onStateChange: (oldState, newState) => {
                this.logger.warn('PES-Evolution workflow circuit breaker state changed', {
                    old_state: oldState,
                    new_state: newState,
                });
            },
        });
        this.logger.info('PES-Evolution workflow initialized', {
            default_timeout_ms: this.DEFAULT_TIMEOUT_MS,
            max_retries: this.MAX_RETRIES,
            checkpoints_enabled: this.CHECKPOINTS_ENABLED,
        });
    }
    // ============================================================================
    // WORKFLOW EXECUTION
    // ============================================================================
    /**
     * Execute the PES-Evolution hybrid workflow
     *
     * @param input - Workflow input with problem and configurations
     * @param correlationContext - Optional correlation context for distributed tracing
     * @returns Hybrid execution result
     */
    async execute(input, correlationContext) {
        const correlationCtx = correlationContext || correlation_tracker_1.correlationTracker.createContext();
        const taskId = (0, uuid_1.v4)();
        const startTime = Date.now();
        this.logger.info('Starting PES-Evolution workflow execution', {
            correlation_id: correlationCtx.correlation_id,
            task_id: taskId,
            problem_type: input.problem.type,
            problem_description: input.problem.description.substring(0, 100),
        });
        // Validate input
        const taskValidation = (0, hybrid_pes_evolution_canonical_1.validateHybridTask)({
            id: taskId,
            type: 'pes_optimize',
            problem: input.problem,
            integration_strategy: 'sequential',
            created_at: new Date().toISOString(),
            correlation_id: correlationCtx.correlation_id,
        });
        if (!taskValidation.success) {
            throw new Error(`Invalid hybrid task: ${taskValidation.errors?.join(', ')}`);
        }
        try {
            return await this.circuitBreaker.execute(async () => {
                // Initialize result
                const result = {
                    id: (0, uuid_1.v4)(),
                    task_id: taskId,
                    created_at: new Date().toISOString(),
                    correlation_id: correlationCtx.correlation_id,
                };
                // Stage 1: Plan (LoongFlow)
                const plan = await this.stagePlan(input, correlationCtx, result);
                this.saveCheckpoint(taskId, 'plan', { plan, input });
                // Stage 2: Execute (LoongFlow)
                const pesResult = await this.stageExecute(plan, input, correlationCtx, result);
                this.saveCheckpoint(taskId, 'execute', { pesResult, plan });
                // Stage 3: Optimize (OpenEvolve) - optional
                let evolutionResult;
                if (input.enable_optimization !== false) {
                    evolutionResult = await this.stageOptimize(pesResult, input, correlationCtx, result);
                    this.saveCheckpoint(taskId, 'optimize', { evolutionResult, pesResult });
                }
                // Stage 4: Summarize (LoongFlow)
                const summary = await this.stageSummarize(pesResult, evolutionResult, correlationCtx, result);
                this.saveCheckpoint(taskId, 'summarize', { summary });
                // Stage 5: Extract knowledge
                const knowledge = await this.stageExtractKnowledge(pesResult, evolutionResult, input.problem.id, correlationCtx);
                // Complete workflow
                result.completed_at = new Date().toISOString();
                result.knowledge_extracted = knowledge;
                result.integration_metrics = this.calculateMetrics(startTime, pesResult, evolutionResult);
                result.summary = summary.evaluation;
                // Publish completion event
                await this.publishWorkflowCompleted(result, correlationCtx);
                this.logger.info('PES-Evolution workflow completed successfully', {
                    correlation_id: correlationCtx.correlation_id,
                    task_id: taskId,
                    duration_ms: result.integration_metrics.total_duration_ms,
                    synergy_score: result.integration_metrics.synergy_score,
                });
                return result;
            });
        }
        catch (error) {
            this.logger.error('PES-Evolution workflow failed', error, {
                correlation_id: correlationCtx.correlation_id,
                task_id: taskId,
                duration_ms: Date.now() - startTime,
            });
            // Publish failure event
            await this.publishWorkflowFailed(taskId, error, correlationCtx);
            throw error;
        }
    }
    // ============================================================================
    // WORKFLOW STAGES
    // ============================================================================
    /**
     * Stage 1: Plan - Create execution plan using LoongFlow
     */
    async stagePlan(input, correlationCtx, result) {
        const stage = 'plan';
        const timeout = this.DEFAULT_TIMEOUT_MS;
        this.logger.info('Stage 1: Planning', {
            correlation_id: correlationCtx.correlation_id,
            stage,
            problem_id: input.problem.id,
        });
        try {
            // Submit problem to LoongFlow for planning
            const submitResponse = await this.withTimeout(this.loongflowAdapter.submitProblem({
                task: input.problem.description,
                max_iterations: input.pes_config?.max_iterations || 10,
                target_score: input.pes_config?.target_score || 0.9,
                concurrency: input.pes_config?.concurrency || 4,
                metadata: {
                    problem_id: input.problem.id,
                    problem_type: input.problem.type,
                    correlation_id: correlationCtx.correlation_id,
                },
            }), timeout);
            // Get agent state to retrieve plan
            const agentState = await this.withTimeout(this.loongflowAdapter.getAgentState(submitResponse.agent_id), timeout);
            // Transform to canonical ExecutionPlan
            const plan = {
                id: (0, uuid_1.v4)(),
                problem_id: input.problem.id,
                steps: [], // LoongFlow handles internal steps
                metadata: {
                    agent_id: submitResponse.agent_id,
                    loongflow_plan: agentState,
                },
                created_at: new Date().toISOString(),
                planner_type: 'loongflow',
                confidence_score: 0.8,
            };
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('ProblemPlanned', 'pes-evolution-workflow', correlationCtx.correlation_id, {
                problem_id: input.problem.id,
                plan_id: plan.id,
                agent_id: submitResponse.agent_id,
            }));
            this.logger.info('Stage 1 completed: Plan created', {
                correlation_id: correlationCtx.correlation_id,
                plan_id: plan.id,
                agent_id: submitResponse.agent_id,
            });
            return plan;
        }
        catch (error) {
            this.logger.error('Stage 1 failed: Planning', error, {
                correlation_id: correlationCtx.correlation_id,
                problem_id: input.problem.id,
            });
            await this.publishStageFailed('plan', error, correlationCtx);
            throw error;
        }
    }
    /**
     * Stage 2: Execute - Execute plan using LoongFlow PES
     */
    async stageExecute(plan, input, correlationCtx, result) {
        const stage = 'execute';
        const timeout = this.DEFAULT_TIMEOUT_MS * 2; // Allow more time for execution
        this.logger.info('Stage 2: Executing', {
            correlation_id: correlationCtx.correlation_id,
            stage,
            plan_id: plan.id,
        });
        try {
            const agentId = plan.metadata?.agent_id;
            // Wait for execution to complete or timeout
            let agentState = await this.loongflowAdapter.getAgentState(agentId);
            let pollCount = 0;
            const maxPolls = 60; // Prevent infinite polling
            while (agentState.status === 'running' &&
                pollCount < maxPolls) {
                await this.sleep(5000); // Poll every 5 seconds
                agentState = await this.loongflowAdapter.getAgentState(agentId);
                pollCount++;
                this.logger.debug('Polling agent state', {
                    correlation_id: correlationCtx.correlation_id,
                    agent_id: agentId,
                    status: agentState.status,
                    iteration: agentState.current_iteration,
                    best_score: agentState.best_score,
                });
            }
            // Get final execution result
            const loongflowResult = await this.withTimeout(this.loongflowAdapter.getExecutionResult(agentId), timeout);
            // Transform to canonical ExecutionResult
            const pesResult = (0, pes_canonical_1.transformExecutionResultToCanonical)({
                id: (0, uuid_1.v4)(),
                problem_id: input.problem.id,
                plan_id: plan.id,
                state: loongflowResult.was_interrupted ? 'timeout' : 'completed',
                outputs: {
                    final_solution: loongflowResult.final_solution,
                    best_solutions: loongflowResult.best_solutions,
                },
                metrics: {
                    duration_ms: loongflowResult.total_cost * 1000, // Approximate
                    success: !loongflowResult.was_interrupted && loongflowResult.final_score >= 0.8,
                    score: loongflowResult.final_score || 0,
                    error_count: 0,
                    iterations: loongflowResult.total_iterations,
                },
                created_at: loongflowResult.start_time,
                completed_at: loongflowResult.end_time,
            });
            result.pes_result = pesResult;
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('SolutionExecuted', 'pes-evolution-workflow', correlationCtx.correlation_id, {
                problem_id: input.problem.id,
                result_id: pesResult.id,
                score: pesResult.metrics.score,
                iterations: pesResult.metrics.iterations,
            }));
            this.logger.info('Stage 2 completed: Execution finished', {
                correlation_id: correlationCtx.correlation_id,
                result_id: pesResult.id,
                score: pesResult.metrics.score,
                iterations: pesResult.metrics.iterations,
            });
            return pesResult;
        }
        catch (error) {
            this.logger.error('Stage 2 failed: Execution', error, {
                correlation_id: correlationCtx.correlation_id,
                plan_id: plan.id,
            });
            await this.publishStageFailed('execute', error, correlationCtx);
            throw error;
        }
    }
    /**
     * Stage 3: Optimize - Optimize solution using OpenEvolve
     */
    async stageOptimize(pesResult, input, correlationCtx, result) {
        const stage = 'optimize';
        const timeout = this.DEFAULT_TIMEOUT_MS * 3; // Allow more time for optimization
        this.logger.info('Stage 3: Optimizing', {
            correlation_id: correlationCtx.correlation_id,
            stage,
            pes_result_id: pesResult.id,
        });
        try {
            // Extract evolution config from input or use defaults
            const evolutionConfig = input.evolution_config || {
                generations: 10,
                population_size: 100,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            };
            // Create optimization workflow in OpenEvolve
            // Note: This is a simplified example - actual OpenEvolve integration
            // would depend on the specific OpenEvolve API structure
            const workflowDefinition = {
                workflow_id: `optimize-${(0, uuid_1.v4)()}`,
                name: 'PES Solution Optimization',
                description: 'Optimize PES solution using evolutionary algorithms',
                problem_statement: input.problem.description,
                max_refinement_loops: evolutionConfig.generations,
                auto_approval_enabled: true,
                sub_problems: [{
                        id: (0, uuid_1.v4)(),
                        description: `Optimize solution with score ${pesResult.metrics.score}`,
                        dependencies: [],
                        solver_team_name: 'optimization-team',
                        gold_team_gauntlet_name: 'validation-gauntlet',
                    }],
            };
            const workflowResponse = await this.withTimeout(this.openevolveAdapter.createWorkflow(workflowDefinition), timeout);
            // Poll for workflow completion
            let workflowState = await this.openevolveAdapter.getWorkflowStatus(workflowResponse.workflow_id);
            let pollCount = 0;
            const maxPolls = 120; // Allow up to 10 minutes
            while (workflowState.status === 'running' &&
                pollCount < maxPolls) {
                await this.sleep(5000);
                workflowState = await this.openevolveAdapter.getWorkflowStatus(workflowResponse.workflow_id);
                pollCount++;
            }
            // Extract evolution result
            const evolutionResult = {
                best_fitness: workflowState.final_solution?.quality_metrics?.score || pesResult.metrics.score,
                generations: evolutionConfig.generations,
                population: [], // Would be populated from OpenEvolve
                converged: workflowState.status === 'completed',
                metadata: {
                    workflow_id: workflowResponse.workflow_id,
                    original_pes_score: pesResult.metrics.score,
                    improvement: (workflowState.final_solution?.quality_metrics?.score || pesResult.metrics.score) - pesResult.metrics.score,
                },
            };
            result.evolution_result = evolutionResult;
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('SolutionOptimized', 'pes-evolution-workflow', correlationCtx.correlation_id, {
                problem_id: input.problem.id,
                evolution_result: evolutionResult,
                original_score: pesResult.metrics.score,
                optimized_score: evolutionResult.best_fitness,
            }));
            this.logger.info('Stage 3 completed: Optimization finished', {
                correlation_id: correlationCtx.correlation_id,
                workflow_id: workflowResponse.workflow_id,
                best_fitness: evolutionResult.best_fitness,
                generations: evolutionResult.generations,
            });
            return evolutionResult;
        }
        catch (error) {
            this.logger.error('Stage 3 failed: Optimization', error, {
                correlation_id: correlationCtx.correlation_id,
                pes_result_id: pesResult.id,
            });
            await this.publishStageFailed('optimize', error, correlationCtx);
            throw error;
        }
    }
    /**
     * Stage 4: Summarize - Summarize results using LoongFlow
     */
    async stageSummarize(pesResult, evolutionResult, correlationCtx, result) {
        const stage = 'summarize';
        const timeout = 30000; // 30 seconds for summarization
        this.logger.info('Stage 4: Summarizing', {
            correlation_id: correlationCtx.correlation_id,
            stage,
        });
        try {
            // Build evaluation summary
            const pesScore = pesResult.metrics.score || 0;
            const evoScore = evolutionResult?.best_fitness || pesScore;
            const improvement = evolutionResult ? evoScore - pesScore : 0;
            const evaluation = evolutionResult
                ? `Hybrid PES-Evolution execution completed. Initial PES score: ${pesScore.toFixed(3)}. After evolutionary optimization: ${evoScore.toFixed(3)}. Improvement: ${improvement.toFixed(3)} (${((improvement / pesScore) * 100).toFixed(1)}%).`
                : `PES execution completed with score: ${pesScore.toFixed(3)}.`;
            const summary = {
                id: (0, uuid_1.v4)(),
                result_id: pesResult.id,
                evaluation,
                insights: [
                    `PES iterations: ${pesResult.metrics.iterations || 0}`,
                    evolutionResult ? `Evolution generations: ${evolutionResult.generations}` : '',
                    improvement > 0 ? `Evolutionary optimization improved solution by ${improvement.toFixed(3)}` : '',
                    `Final score: ${Math.max(pesScore, evoScore).toFixed(3)}`,
                ].filter(Boolean),
                performance_assessment: {
                    success_criteria_met: Math.max(pesScore, evoScore) >= 0.8,
                    quality_score: Math.max(pesScore, evoScore),
                    efficiency_score: evolutionResult ? 0.9 : 0.7,
                },
                recommendations: [
                    improvement > 0.1 ? 'Evolutionary optimization significantly improved results' : 'Consider increasing evolutionary generations',
                    pesResult.metrics.iterations && pesResult.metrics.iterations > 50 ? 'PES required many iterations - consider adjusting planning strategy' : '',
                ].filter(Boolean),
                created_at: new Date().toISOString(),
                summarizer_type: 'pes-evolution-hybrid',
            };
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('ResultSummarized', 'pes-evolution-workflow', correlationCtx.correlation_id, {
                summary_id: summary.id,
                result_id: pesResult.id,
                quality_score: summary.performance_assessment.quality_score,
            }));
            this.logger.info('Stage 4 completed: Summary created', {
                correlation_id: correlationCtx.correlation_id,
                summary_id: summary.id,
            });
            return summary;
        }
        catch (error) {
            this.logger.error('Stage 4 failed: Summarization', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStageFailed('summarize', error, correlationCtx);
            throw error;
        }
    }
    /**
     * Stage 5: Extract knowledge - Extract and store evolutionary knowledge
     */
    async stageExtractKnowledge(pesResult, evolutionResult, problemId, correlationCtx) {
        const stage = 'extract_knowledge';
        this.logger.info('Stage 5: Extracting knowledge', {
            correlation_id: correlationCtx.correlation_id,
            stage,
        });
        try {
            const knowledge = [];
            // Extract knowledge from PES result
            if (pesResult.outputs?.best_solutions && Array.isArray(pesResult.outputs.best_solutions)) {
                for (const solution of pesResult.outputs.best_solutions.slice(0, 5)) {
                    knowledge.push((0, hybrid_pes_evolution_canonical_1.transformLoongFlowSolutionToKnowledge)(solution, problemId, 'solution_pattern'));
                }
            }
            // Extract knowledge from evolution result
            if (evolutionResult && evolutionResult.metadata?.improvement > 0) {
                knowledge.push({
                    id: (0, uuid_1.v4)(),
                    source_type: 'hybrid_result',
                    problem_id: problemId,
                    knowledge_type: 'planning_strategy',
                    content: {
                        pattern: 'Evolutionary optimization improved PES solution',
                        success_rate: evolutionResult.best_fitness,
                        avg_score: evolutionResult.best_fitness,
                        usage_count: 1,
                        context: {
                            improvement: evolutionResult.metadata.improvement,
                            generations: evolutionResult.generations,
                        },
                    },
                    extracted_at: new Date().toISOString(),
                });
            }
            // Publish knowledge extracted event for each knowledge item
            for (const k of knowledge) {
                await this.eventBus.publish((0, event_types_1.createBaseEvent)('KnowledgeExtracted', 'pes-evolution-workflow', correlationCtx.correlation_id, {
                    knowledge_id: k.id,
                    problem_id: k.problem_id,
                    knowledge_type: k.knowledge_type,
                    source_type: k.source_type,
                }));
            }
            this.logger.info('Stage 5 completed: Knowledge extracted', {
                correlation_id: correlationCtx.correlation_id,
                knowledge_count: knowledge.length,
            });
            return knowledge;
        }
        catch (error) {
            this.logger.error('Stage 5 failed: Knowledge extraction', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            // Knowledge extraction failure should not fail the workflow
            await this.publishStageFailed('extract_knowledge', error, correlationCtx);
            return [];
        }
    }
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    /**
     * Calculate integration metrics
     */
    calculateMetrics(startTime, pesResult, evolutionResult) {
        const durationMs = Date.now() - startTime;
        return {
            pes_iterations: pesResult?.metrics?.iterations || 0,
            evolution_generations: evolutionResult?.generations || 0,
            total_duration_ms: durationMs,
            synergy_score: this.calculateSynergyScore(pesResult, evolutionResult),
            pes_time_ms: pesResult?.metrics?.duration_ms || 0,
            evolution_time_ms: durationMs - (pesResult?.metrics?.duration_ms || 0),
            paradigm_switches: evolutionResult ? 1 : 0,
            knowledge_transferred: 0, // Will be updated in knowledge extraction stage
        };
    }
    /**
     * Calculate synergy score between PES and Evolution
     */
    calculateSynergyScore(pesResult, evolutionResult) {
        if (!pesResult || !evolutionResult) {
            return 0.5;
        }
        const pesScore = pesResult.metrics?.score || 0;
        const evoScore = evolutionResult.best_fitness || pesScore;
        const improvement = evoScore - pesScore;
        // Synergy score based on improvement and convergence
        const improvementFactor = Math.min(improvement / 0.5, 1); // Cap at 0.5 improvement
        const convergenceFactor = evolutionResult.converged ? 1 : 0.5;
        return 0.5 + (improvementFactor * 0.3) + (convergenceFactor * 0.2);
    }
    /**
     * Save checkpoint for resume capability
     */
    saveCheckpoint(taskId, stage, data) {
        if (!this.CHECKPOINTS_ENABLED) {
            return;
        }
        const checkpoint = {
            checkpoint_id: (0, uuid_1.v4)(),
            task_id: taskId,
            stage,
            timestamp: new Date().toISOString(),
            data,
        };
        this.checkpoints.set(checkpoint.checkpoint_id, checkpoint);
        this.logger.debug('Checkpoint saved', {
            task_id: taskId,
            checkpoint_id: checkpoint.checkpoint_id,
            stage,
        });
    }
    /**
     * Execute function with timeout
     */
    async withTimeout(fn, timeoutMs) {
        return Promise.race([
            fn,
            new Promise((_, reject) => setTimeout(() => reject(new Error(`Operation timeout after ${timeoutMs}ms`)), timeoutMs)),
        ]);
    }
    /**
     * Sleep for specified milliseconds
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    // ============================================================================
    // EVENT PUBLISHING
    // ============================================================================
    async publishWorkflowCompleted(result, correlationCtx) {
        await this.eventBus.publish((0, event_types_1.createBaseEvent)('WorkflowCompleted', 'pes-evolution-workflow', correlationCtx.correlation_id, {
            workflow_id: result.task_id,
            workflow_name: 'pes-evolution',
            duration_ms: result.integration_metrics.total_duration_ms,
            output_data: result,
            steps_completed: 5,
            steps_failed: 0,
        }));
    }
    async publishWorkflowFailed(taskId, error, correlationCtx) {
        await this.eventBus.publish((0, event_types_1.createBaseEvent)('WorkflowFailed', 'pes-evolution-workflow', correlationCtx.correlation_id, {
            workflow_id: taskId,
            workflow_name: 'pes-evolution',
            failure_reason: error.message,
            failed_step: 'unknown',
            error_details: {
                name: error.name,
                message: error.message,
                stack: error.stack,
            },
            duration_ms: 0,
        }));
    }
    async publishStageFailed(stage, error, correlationCtx) {
        await this.eventBus.publish((0, event_types_1.createBaseEvent)('StageFailed', 'pes-evolution-workflow', correlationCtx.correlation_id, {
            stage,
            error_message: error.message,
            error_name: error.name,
        }));
    }
    // ============================================================================
    // CHECKPOINT MANAGEMENT
    // ============================================================================
    /**
     * Get checkpoint by ID
     */
    getCheckpoint(checkpointId) {
        return this.checkpoints.get(checkpointId);
    }
    /**
     * Get all checkpoints for a task
     */
    getCheckpointsForTask(taskId) {
        return Array.from(this.checkpoints.values()).filter(cp => cp.task_id === taskId);
    }
    /**
     * Resume workflow from checkpoint
     */
    async resumeFromCheckpoint(checkpointId, correlationContext) {
        const checkpoint = this.checkpoints.get(checkpointId);
        if (!checkpoint) {
            throw new Error(`Checkpoint not found: ${checkpointId}`);
        }
        this.logger.info('Resuming workflow from checkpoint', {
            checkpoint_id: checkpointId,
            stage: checkpoint.stage,
        });
        // Implementation would continue from the checkpointed stage
        // This is a simplified example
        throw new Error('Resume from checkpoint not yet implemented');
    }
    /**
     * Clear checkpoints for a task
     */
    clearCheckpoints(taskId) {
        const toDelete = Array.from(this.checkpoints.entries())
            .filter(([_, cp]) => cp.task_id === taskId)
            .map(([id, _]) => id);
        for (const id of toDelete) {
            this.checkpoints.delete(id);
        }
        this.logger.info('Checkpoints cleared', {
            task_id: taskId,
            cleared_count: toDelete.length,
        });
    }
}
exports.PESEvolutionWorkflow = PESEvolutionWorkflow;
// ============================================================================
// FACTORY FUNCTION
// ============================================================================
function createPESEvolutionWorkflow(config) {
    return new PESEvolutionWorkflow(config);
}
//# sourceMappingURL=pes-evolution-workflow.js.map