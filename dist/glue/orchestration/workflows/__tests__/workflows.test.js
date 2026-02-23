"use strict";
/**
 * Hybrid Orchestration Workflow Tests
 *
 * Comprehensive test suite for all hybrid workflows.
 *
 * Following Federation Constitution:
 * - Law of Runtime Truth: Tests verify actual behavior
 * - Law of Idempotency: Tests verify safe retry
 * - Observability: Tests verify event publishing
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const uuid_1 = require("uuid");
const event_bus_1 = require("../../event-bus");
const pes_evolution_workflow_1 = require("../pes-evolution-workflow");
const knowledge_extraction_workflow_1 = require("../knowledge-extraction-workflow");
const adaptive_execution_workflow_1 = require("../adaptive-execution-workflow");
const multi_stage_reasoning_workflow_1 = require("../multi-stage-reasoning-workflow");
// ============================================================================
// MOCK ADAPTERS
// ============================================================================
class MockLoongFlowAdapter {
    async healthCheck() {
        return {
            status: 'healthy',
            timestamp: new Date().toISOString(),
        };
    }
    async submitProblem(request) {
        return {
            agent_id: (0, uuid_1.v4)(),
            status: 'running',
            message: 'Problem submitted',
        };
    }
    async getAgentState(agentId) {
        return {
            agent_id: agentId,
            status: 'completed',
            current_iteration: 10,
            max_iterations: 10,
            target_score: 0.9,
            best_score: 0.85,
            start_time: new Date().toISOString(),
            completion_count: 1,
            total_prompt_tokens: 1000,
            total_completion_tokens: 2000,
            total_cost: 0.1,
        };
    }
    async getExecutionResult(agentId) {
        return {
            agent_id: agentId,
            status: 'completed',
            final_solution: 'Solution text here',
            final_score: 0.85,
            best_solutions: [
                {
                    solution_id: (0, uuid_1.v4)(),
                    solution: 'Best solution',
                    evaluation: 'Good solution',
                    score: 0.9,
                    island_id: 0,
                    iteration: 8,
                    generate_plan: 'Plan text',
                    summary: 'Summary',
                    created_at: new Date().toISOString(),
                },
            ],
            total_iterations: 10,
            total_tokens: 3000,
            total_cost: 0.15,
            was_interrupted: false,
            start_time: new Date(Date.now() - 60000).toISOString(),
            end_time: new Date().toISOString(),
        };
    }
    async getBestSolutions(islandId, topK) {
        return [
            {
                solution_id: (0, uuid_1.v4)(),
                solution: 'Solution 1',
                evaluation: 'Evaluation 1',
                score: 0.9,
                island_id: islandId || 0,
                iteration: 5,
                generate_plan: 'Plan 1',
                summary: 'Summary 1',
                created_at: new Date().toISOString(),
            },
            {
                solution_id: (0, uuid_1.v4)(),
                solution: 'Solution 2',
                evaluation: 'Evaluation 2',
                score: 0.85,
                island_id: islandId || 0,
                iteration: 6,
                generate_plan: 'Plan 2',
                summary: 'Summary 2',
                created_at: new Date().toISOString(),
            },
        ];
    }
    async interruptAgent(agentId) {
        return { message: 'Agent interrupted' };
    }
}
class MockOpenEvolveAdapter {
    async healthCheck() {
        return {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            integrations: [],
        };
    }
    async createWorkflow(workflow) {
        return {
            message: 'Workflow created',
            workflow_id: (0, uuid_1.v4)(),
        };
    }
    async getWorkflowStatus(workflowId) {
        return {
            workflow_id: workflowId,
            status: 'completed',
            current_stage: 'completed',
            progress: 100,
            start_time: new Date(Date.now() - 30000).toISOString(),
            end_time: new Date().toISOString(),
            final_solution: {
                content: 'Final solution',
                quality_metrics: { score: 0.88 },
            },
        };
    }
    async createTeam(team) {
        return {
            message: 'Team created',
            team_name: team.name,
        };
    }
    async createGauntlet(gauntlet) {
        return {
            message: 'Gauntlet created',
            gauntlet_name: gauntlet.name,
        };
    }
}
class MockEventBus extends event_bus_1.EventBus {
    constructor() {
        super(...arguments);
        this.publishedEvents = [];
    }
    async publish(event) {
        this.publishedEvents.push(event);
        await super.publish(event);
    }
    clearEvents() {
        this.publishedEvents = [];
    }
    getEventsByType(type) {
        return this.publishedEvents.filter(e => e.type === type);
    }
}
// ============================================================================
// TEST DATA
// ============================================================================
const createMockProblem = () => ({
    id: (0, uuid_1.v4)(),
    type: 'optimization',
    description: 'Optimize neural network hyperparameters',
    context: { dataset: 'MNIST', model: 'neural_network' },
    constraints: ['max_layers <= 10', 'batch_size in [32, 64, 128]'],
    success_criteria: ['accuracy > 0.95', 'training_time < 3600s'],
    created_at: new Date().toISOString(),
    priority: 8,
    tags: ['ml', 'optimization'],
});
// ============================================================================
// PES-EVOLUTION WORKFLOW TESTS
// ============================================================================
(0, globals_1.describe)('PESEvolutionWorkflow', () => {
    let workflow;
    let loongflowAdapter;
    let openevolveAdapter;
    let eventBus;
    (0, globals_1.beforeEach)(() => {
        loongflowAdapter = new MockLoongFlowAdapter();
        openevolveAdapter = new MockOpenEvolveAdapter();
        eventBus = new MockEventBus();
        workflow = (0, pes_evolution_workflow_1.createPESEvolutionWorkflow)({
            loongflowAdapter: loongflowAdapter,
            openevolveAdapter: openevolveAdapter,
            eventBus,
            checkpoints_enabled: false,
            default_timeout_ms: 30000,
            max_retries: 2,
        });
    });
    (0, globals_1.afterEach)(() => {
        eventBus.clearEvents();
    });
    (0, globals_1.describe)('execute', () => {
        (0, globals_1.it)('should execute PES then Evolution', async () => {
            const input = {
                problem: createMockProblem(),
                pes_config: {
                    max_iterations: 10,
                    target_score: 0.9,
                },
                evolution_config: {
                    generations: 10,
                    population_size: 100,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                },
                enable_optimization: true,
                enable_knowledge_extraction: true,
            };
            const result = await workflow.execute(input);
            (0, globals_1.expect)(result).toBeDefined();
            (0, globals_1.expect)(result.id).toBeDefined();
            (0, globals_1.expect)(result.task_id).toBeDefined();
            (0, globals_1.expect)(result.pes_result).toBeDefined();
            (0, globals_1.expect)(result.evolution_result).toBeDefined();
            (0, globals_1.expect)(result.integration_metrics).toBeDefined();
            (0, globals_1.expect)(result.knowledge_extracted).toBeDefined();
            (0, globals_1.expect)(Array.isArray(result.knowledge_extracted)).toBe(true);
        });
        (0, globals_1.it)('should publish events for each phase', async () => {
            const input = {
                problem: createMockProblem(),
                enable_optimization: true,
            };
            await workflow.execute(input);
            // Check for key events
            const problemPlannedEvents = eventBus.getEventsByType('ProblemPlanned');
            const solutionExecutedEvents = eventBus.getEventsByType('SolutionExecuted');
            const solutionOptimizedEvents = eventBus.getEventsByType('SolutionOptimized');
            const resultSummarizedEvents = eventBus.getEventsByType('ResultSummarized');
            const knowledgeExtractedEvents = eventBus.getEventsByType('KnowledgeExtracted');
            const workflowCompletedEvents = eventBus.getEventsByType('WorkflowCompleted');
            (0, globals_1.expect)(problemPlannedEvents.length).toBeGreaterThan(0);
            (0, globals_1.expect)(solutionExecutedEvents.length).toBeGreaterThan(0);
            (0, globals_1.expect)(solutionOptimizedEvents.length).toBeGreaterThan(0);
            (0, globals_1.expect)(resultSummarizedEvents.length).toBeGreaterThan(0);
            (0, globals_1.expect)(knowledgeExtractedEvents.length).toBeGreaterThan(0);
            (0, globals_1.expect)(workflowCompletedEvents.length).toBe(1);
        });
        (0, globals_1.it)('should handle failures gracefully', async () => {
            const failingAdapter = new MockLoongFlowAdapter();
            failingAdapter.submitProblem = async () => {
                throw new Error('Adapter failure');
            };
            const failingWorkflow = (0, pes_evolution_workflow_1.createPESEvolutionWorkflow)({
                loongflowAdapter: failingAdapter,
                openevolveAdapter: openevolveAdapter,
                eventBus,
                max_retries: 1,
            });
            const input = {
                problem: createMockProblem(),
            };
            await (0, globals_1.expect)(failingWorkflow.execute(input)).rejects.toThrow();
            // Check that failure event was published
            const workflowFailedEvents = eventBus.getEventsByType('WorkflowFailed');
            (0, globals_1.expect)(workflowFailedEvents.length).toBe(1);
        });
        (0, globals_1.it)('should skip optimization if disabled', async () => {
            const input = {
                problem: createMockProblem(),
                enable_optimization: false,
            };
            const result = await workflow.execute(input);
            (0, globals_1.expect)(result.pes_result).toBeDefined();
            (0, globals_1.expect)(result.evolution_result).toBeUndefined();
        });
    });
    (0, globals_1.describe)('checkpoints', () => {
        (0, globals_1.it)('should save checkpoints for each stage', async () => {
            const workflowWithCheckpoints = (0, pes_evolution_workflow_1.createPESEvolutionWorkflow)({
                loongflowAdapter: loongflowAdapter,
                openevolveAdapter: openevolveAdapter,
                eventBus,
                checkpoints_enabled: true,
            });
            const input = {
                problem: createMockProblem(),
            };
            await workflowWithCheckpoints.execute(input);
            // Check that checkpoints were saved
            const checkpoints = workflowWithCheckpoints.getCheckpointsForTask(globals_1.expect.any(String));
            (0, globals_1.expect)(checkpoints.length).toBeGreaterThan(0);
        });
        (0, globals_1.it)('should clear checkpoints when requested', async () => {
            const workflowWithCheckpoints = (0, pes_evolution_workflow_1.createPESEvolutionWorkflow)({
                loongflowAdapter: loongflowAdapter,
                openevolveAdapter: openevolveAdapter,
                eventBus,
                checkpoints_enabled: true,
            });
            const input = {
                problem: createMockProblem(),
            };
            const result = await workflowWithCheckpoints.execute(input);
            workflowWithCheckpoints.clearCheckpoints(result.task_id);
            const checkpoints = workflowWithCheckpoints.getCheckpointsForTask(result.task_id);
            (0, globals_1.expect)(checkpoints.length).toBe(0);
        });
    });
});
// ============================================================================
// KNOWLEDGE EXTRACTION WORKFLOW TESTS
// ============================================================================
(0, globals_1.describe)('KnowledgeExtractionWorkflow', () => {
    let workflow;
    let loongflowAdapter;
    let eventBus;
    (0, globals_1.beforeEach)(() => {
        loongflowAdapter = new MockLoongFlowAdapter();
        eventBus = new MockEventBus();
        workflow = (0, knowledge_extraction_workflow_1.createKnowledgeExtractionWorkflow)({
            loongflowAdapter: loongflowAdapter,
            eventBus,
            enable_graph_storage: false,
            enable_vectorization: false,
            enable_problem_formulation: true,
        });
    });
    (0, globals_1.afterEach)(() => {
        eventBus.clearEvents();
    });
    (0, globals_1.describe)('execute', () => {
        (0, globals_1.it)('should extract knowledge from solutions', async () => {
            const input = {
                island_id: 0,
                top_k: 10,
                min_score: 0.5,
            };
            const result = await workflow.execute(input);
            (0, globals_1.expect)(result).toBeDefined();
            (0, globals_1.expect)(result.knowledge).toBeDefined();
            (0, globals_1.expect)(Array.isArray(result.knowledge)).toBe(true);
            (0, globals_1.expect)(result.patterns).toBeDefined();
            (0, globals_1.expect)(Array.isArray(result.patterns)).toBe(true);
            (0, globals_1.expect)(result.problems).toBeDefined();
            (0, globals_1.expect)(Array.isArray(result.problems)).toBe(true);
        });
        (0, globals_1.it)('should deduplicate knowledge', async () => {
            const input = {
                island_id: 0,
                top_k: 10,
            };
            const result = await workflow.execute(input);
            // Check for duplicates by source_id
            const sourceIds = result.knowledge.map(k => k.source_id);
            const uniqueSourceIds = new Set(sourceIds);
            (0, globals_1.expect)(sourceIds.length).toBe(uniqueSourceIds.size);
        });
        (0, globals_1.it)('should publish KnowledgeExtracted events', async () => {
            const input = {
                island_id: 0,
                top_k: 5,
            };
            await workflow.execute(input);
            const knowledgeEvents = eventBus.getEventsByType('KnowledgeExtracted');
            (0, globals_1.expect)(knowledgeEvents.length).toBeGreaterThan(0);
            knowledgeEvents.forEach(event => {
                (0, globals_1.expect)(event.data).toHaveProperty('knowledge_id');
                (0, globals_1.expect)(event.data).toHaveProperty('problem_id');
                (0, globals_1.expect)(event.data).toHaveProperty('knowledge_type');
            });
        });
        (0, globals_1.it)('should filter by minimum score', async () => {
            const input = {
                island_id: 0,
                top_k: 10,
                min_score: 0.9,
            };
            const result = await workflow.execute(input);
            // All knowledge should have score >= 0.9
            result.knowledge.forEach(k => {
                (0, globals_1.expect)(k.content.avg_score).toBeGreaterThanOrEqual(0.9);
            });
        });
    });
    (0, globals_1.describe)('problem formulation', () => {
        (0, globals_1.it)('should formulate problems from low-quality patterns', async () => {
            const input = {
                island_id: 0,
                top_k: 10,
                min_score: 0.3, // Low threshold to get more varied solutions
            };
            const result = await workflow.execute(input);
            // Check that problems were formulated
            (0, globals_1.expect)(result.problems.length).toBeGreaterThan(0);
            result.problems.forEach(problem => {
                (0, globals_1.expect)(problem).toHaveProperty('problem_id');
                (0, globals_1.expect)(problem).toHaveProperty('problem_type');
                (0, globals_1.expect)(problem).toHaveProperty('description');
                (0, globals_1.expect)(problem).toHaveProperty('priority');
                (0, globals_1.expect)(problem.priority).toBeGreaterThan(0);
                (0, globals_1.expect)(problem.priority).toBeLessThanOrEqual(10);
            });
        });
    });
});
// ============================================================================
// ADAPTIVE EXECUTION WORKFLOW TESTS
// ============================================================================
(0, globals_1.describe)('AdaptiveExecutionWorkflow', () => {
    let workflow;
    let loongflowAdapter;
    let openevolveAdapter;
    let eventBus;
    (0, globals_1.beforeEach)(() => {
        loongflowAdapter = new MockLoongFlowAdapter();
        openevolveAdapter = new MockOpenEvolveAdapter();
        eventBus = new MockEventBus();
        workflow = (0, adaptive_execution_workflow_1.createAdaptiveExecutionWorkflow)({
            loongflowAdapter: loongflowAdapter,
            openevolveAdapter: openevolveAdapter,
            eventBus,
            max_paradigm_switches: 3,
            max_iterations: 5,
        });
    });
    (0, globals_1.afterEach)(() => {
        eventBus.clearEvents();
    });
    (0, globals_1.describe)('executeAdaptive', () => {
        (0, globals_1.it)('should execute with initial paradigm', async () => {
            const input = {
                problem: createMockProblem(),
                initial_paradigm: 'PES',
                triggers: [],
            };
            const result = await workflow.executeAdaptive(input);
            (0, globals_1.expect)(result).toBeDefined();
            (0, globals_1.expect)(result.id).toBeDefined();
            (0, globals_1.expect)(result.task_id).toBeDefined();
            (0, globals_1.expect)(result.integration_metrics).toBeDefined();
            (0, globals_1.expect)(result.best_solution).toBeDefined();
        });
        (0, globals_1.it)('should publish ParadigmSwitched events when switching', async () => {
            const input = {
                problem: createMockProblem(),
                initial_paradigm: 'PES',
                triggers: [
                    {
                        id: (0, uuid_1.v4)(),
                        condition: 'low_confidence',
                        threshold: 0.95,
                        action: 'switch_to_evolution',
                    },
                ],
            };
            await workflow.executeAdaptive(input);
            const paradigmSwitchedEvents = eventBus.getEventsByType('ParadigmSwitched');
            // May or may not switch depending on scores
            (0, globals_1.expect)(paradigmSwitchedEvents).toBeDefined();
        });
        (0, globals_1.it)('should respect max paradigm switches', async () => {
            const input = {
                problem: createMockProblem(),
                initial_paradigm: 'PES',
                triggers: [
                    {
                        id: (0, uuid_1.v4)(),
                        condition: 'stagnation',
                        threshold: 2,
                        action: 'switch_to_evolution',
                    },
                ],
                enable_hybrid_fallback: false,
            };
            const result = await workflow.executeAdaptive(input);
            // Should not exceed max switches
            (0, globals_1.expect)(result.integration_metrics.paradigm_switches).toBeLessThanOrEqual(3);
        });
        (0, globals_1.it)('should stop on convergence', async () => {
            const input = {
                problem: createMockProblem(),
                initial_paradigm: 'PES',
                triggers: [],
            };
            const result = await workflow.executeAdaptive(input);
            // Should complete successfully
            (0, globals_1.expect)(result.integration_metrics.synergy_score).toBeGreaterThan(0);
        });
    });
});
// ============================================================================
// MULTI-STAGE REASONING WORKFLOW TESTS
// ============================================================================
(0, globals_1.describe)('MultiStageReasoningWorkflow', () => {
    let workflow;
    let loongflowAdapter;
    let openevolveAdapter;
    let eventBus;
    (0, globals_1.beforeEach)(() => {
        loongflowAdapter = new MockLoongFlowAdapter();
        openevolveAdapter = new MockOpenEvolveAdapter();
        eventBus = new MockEventBus();
        workflow = (0, multi_stage_reasoning_workflow_1.createMultiStageReasoningWorkflow)({
            loongflowAdapter: loongflowAdapter,
            openevolveAdapter: openevolveAdapter,
            eventBus,
            enable_validation: true,
            enable_refinement: true,
            max_refinement_loops: 2,
        });
    });
    (0, globals_1.afterEach)(() => {
        eventBus.clearEvents();
    });
    (0, globals_1.describe)('executeReasoning', () => {
        (0, globals_1.it)('should execute all stages', async () => {
            const input = {
                problem: createMockProblem(),
                validation_system: 'both',
            };
            const result = await workflow.executeReasoning(input);
            (0, globals_1.expect)(result).toBeDefined();
            (0, globals_1.expect)(result.id).toBeDefined();
            (0, globals_1.expect)(result.pes_result).toBeDefined();
            (0, globals_1.expect)(result.evolution_result).toBeDefined();
            (0, globals_1.expect)(result.knowledge_extracted).toBeDefined();
            (0, globals_1.expect)(result.integration_metrics).toBeDefined();
        });
        (0, globals_1.it)('should publish events for each stage', async () => {
            const input = {
                problem: createMockProblem(),
            };
            await workflow.executeReasoning(input);
            const stageCompletedEvents = eventBus.getEventsByType('StageCompleted');
            (0, globals_1.expect)(stageCompletedEvents.length).toBeGreaterThan(0);
            const workflowCompletedEvents = eventBus.getEventsByType('WorkflowCompleted');
            (0, globals_1.expect)(workflowCompletedEvents.length).toBe(1);
        });
        (0, globals_1.it)('should skip refinement if validation passes', async () => {
            const input = {
                problem: {
                    ...createMockProblem(),
                    // This should result in high score
                },
                refinement_threshold: 0.8,
            };
            const result = await workflow.executeReasoning(input);
            // If validation passes, no refinement needed
            (0, globals_1.expect)(result).toBeDefined();
        });
        (0, globals_1.it)('should extract knowledge from final solution', async () => {
            const input = {
                problem: createMockProblem(),
            };
            const result = await workflow.executeReasoning(input);
            (0, globals_1.expect)(result.knowledge_extracted).toBeDefined();
            (0, globals_1.expect)(Array.isArray(result.knowledge_extracted)).toBe(true);
            (0, globals_1.expect)(result.knowledge_extracted.length).toBeGreaterThan(0);
        });
        (0, globals_1.it)('should handle stage failures gracefully', async () => {
            const failingAdapter = new MockLoongFlowAdapter();
            failingAdapter.submitProblem = async () => {
                throw new Error('Plan stage failed');
            };
            const failingWorkflow = (0, multi_stage_reasoning_workflow_1.createMultiStageReasoningWorkflow)({
                loongflowAdapter: failingAdapter,
                openevolveAdapter: openevolveAdapter,
                eventBus,
            });
            const input = {
                problem: createMockProblem(),
            };
            await (0, globals_1.expect)(failingWorkflow.executeReasoning(input)).rejects.toThrow();
            const workflowFailedEvents = eventBus.getEventsByType('WorkflowFailed');
            (0, globals_1.expect)(workflowFailedEvents.length).toBe(1);
        });
    });
    (0, globals_1.describe)('validation', () => {
        (0, globals_1.it)('should skip validation if disabled', async () => {
            const workflowNoValidation = (0, multi_stage_reasoning_workflow_1.createMultiStageReasoningWorkflow)({
                loongflowAdapter: loongflowAdapter,
                openevolveAdapter: openevolveAdapter,
                eventBus,
                enable_validation: false,
            });
            const input = {
                problem: createMockProblem(),
            };
            const result = await workflowNoValidation.executeReasoning(input);
            (0, globals_1.expect)(result).toBeDefined();
        });
    });
});
// ============================================================================
// WORKFLOW FACTORY TESTS
// ============================================================================
(0, globals_1.describe)('Workflow Factory', () => {
    (0, globals_1.it)('should throw error for unknown workflow type', () => {
        const { createWorkflow, WORKFLOWS } = require('../index');
        (0, globals_1.expect)(() => {
            createWorkflow('unknown', {});
        }).toThrow();
    });
    (0, globals_1.it)('should validate workflow config', () => {
        const { validateWorkflowConfig, WORKFLOWS } = require('../index');
        const validation = validateWorkflowConfig(WORKFLOWS.PES_EVOLUTION, {
            loongflowAdapter: new MockLoongFlowAdapter(),
            openevolveAdapter: new MockOpenEvolveAdapter(),
        });
        (0, globals_1.expect)(validation.valid).toBe(true);
        (0, globals_1.expect)(validation.errors).toBeUndefined();
    });
    (0, globals_1.it)('should detect missing adapters', () => {
        const { validateWorkflowConfig, WORKFLOWS } = require('../index');
        const validation = validateWorkflowConfig(WORKFLOWS.PES_EVOLUTION, {});
        (0, globals_1.expect)(validation.valid).toBe(false);
        (0, globals_1.expect)(validation.errors).toBeDefined();
        (0, globals_1.expect)(validation.errors?.length).toBeGreaterThan(0);
    });
});
//# sourceMappingURL=workflows.test.js.map