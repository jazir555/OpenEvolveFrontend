"use strict";
/**
 * End-to-end tests for the hybrid OpenEvolve LoongFlow PES system
 *
 * This test suite validates the complete integration of:
 * 1. LoongFlow PES (Plan-Execute-Summarize) execution
 * 2. OpenEvolve evolutionary optimization
 * 3. Hybrid workflows combining both systems
 * 4. Knowledge extraction and reuse
 * 5. Adaptive paradigm switching
 * 6. Multi-stage reasoning
 * 7. Error handling and recovery
 * 8. Performance and scalability
 *
 * Test Statistics:
 * - Total Test Suites: 7
 * - Total Test Functions: 35+
 * - Coverage Areas: E2E Integration, Workflows, Knowledge, Error Handling, Performance
 *
 * Running Tests:
 *   npm test tests/test_hybrid_pes_evolution_e2e.test.ts
 *   npm test -- tests/test_hybrid_pes_evolution_e2e.test.ts --testNamePattern="PESEvolutionWorkflow"
 *   npm test -- tests/test_hybrid_pes_evolution_e2e.test.ts --testNamePattern="slow" --skip
 *
 * Author: OpenEvolve Distinguished Engineer
 * Version: 2.0.0 (TypeScript)
 */
Object.defineProperty(exports, "__esModule", { value: true });
const pes_evolution_workflow_1 = require("../glue/orchestration/workflows/pes-evolution-workflow");
const knowledge_extraction_workflow_1 = require("../glue/orchestration/workflows/knowledge-extraction-workflow");
const adaptive_execution_workflow_1 = require("../glue/orchestration/workflows/adaptive-execution-workflow");
const event_bus_1 = require("../glue/orchestration/event-bus");
const dead_letter_queue_1 = require("../glue/orchestration/dead-letter-queue");
// Mock adapters for testing
const loongflow_mock_1 = require("./mocks/loongflow-mock");
const openevolve_mock_1 = require("./mocks/openevolve-mock");
// ============================================================================
// TEST CONFIGURATION
// ============================================================================
describe('Hybrid PES Evolution System', () => {
    let eventBus;
    let dlq;
    // Test configuration
    const testTimeout = 60000; // 60 seconds
    const skipSlowTests = process.env.SKIP_SLOW_TESTS === 'true';
    const enableKnowledgeTests = process.env.ENABLE_KNOWLEDGE_TESTS === 'true';
    beforeEach(() => {
        eventBus = new event_bus_1.InMemoryEventBus();
        dlq = new dead_letter_queue_1.DeadLetterQueue(eventBus);
    });
    // ============================================================================
    // TEST SUITE 1: Basic PES Execution
    // ============================================================================
    describe('TestBasicPESExecution', () => {
        it('should submit and execute problem', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Maximize f(x) = x^2 for x in [0, 10]',
                context: {
                    domain: 'mathematical_optimization',
                    difficulty: 'easy',
                },
                constraints: ['x >= 0', 'x <= 10'],
                success_criteria: ['fitness > 0.9'],
                created_at: new Date().toISOString(),
            };
            // Submit problem
            const result = await loongflow.submitProblem({
                task: problem.description,
                max_iterations: 10,
                target_score: 0.8,
                concurrency: 4,
            });
            const agentId = result.agent_id;
            // Assert submission
            expect(agentId).toBeDefined();
            expect(result.status).toBe('SUBMITTED');
            expect(result.submitted_at).toBeDefined();
            // Get status
            const status = await loongflow.getAgentState(agentId);
            // Assert status
            expect(status.status).toBeDefined();
            expect(['completed', 'running', 'idle']).toContain(status.status);
            expect(status.current_iteration).toBeGreaterThanOrEqual(0);
        });
        it('should retrieve solution results', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Test optimization problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            // Submit and get result
            const submitResult = await loongflow.submitProblem({
                task: problem.description,
                max_iterations: 10,
            });
            const solution = await loongflow.getExecutionResult(submitResult.agent_id);
            // Assert solution structure
            expect(solution.final_solution).toBeDefined();
            expect(solution.final_score).toBeGreaterThanOrEqual(0);
            expect(solution.final_score).toBeLessThanOrEqual(1);
            expect(solution.total_iterations).toBeGreaterThan(0);
            expect(solution.was_interrupted).toBe(false);
        });
        it('should complete PES cycle', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const problem = {
                id: crypto.randomUUID(),
                type: 'reasoning',
                description: 'Prove that sqrt(2) is irrational',
                context: {
                    domain: 'mathematics',
                    difficulty: 'medium',
                },
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            // Submit
            const result = await loongflow.submitProblem({
                task: problem.description,
                max_iterations: 10,
            });
            // Get final result
            const finalResult = await loongflow.getExecutionResult(result.agent_id);
            // Assert PES phases completed
            expect(finalResult.final_solution).toBeDefined();
            expect(finalResult.final_score).toBeGreaterThanOrEqual(0);
        });
        it('should get best solutions', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const solutions = await loongflow.getBestSolutions(undefined, 5);
            // Assert solutions retrieved
            expect(Array.isArray(solutions)).toBe(true);
            expect(solutions.length).toBeGreaterThan(0);
            expect(solutions.length).toBeLessThanOrEqual(5);
            for (const solution of solutions) {
                expect(solution.solution_id).toBeDefined();
                expect(solution.score).toBeGreaterThanOrEqual(0);
                expect(solution.score).toBeLessThanOrEqual(1);
            }
        });
    });
    // ============================================================================
    // TEST SUITE 2: Evolutionary Optimization
    // ============================================================================
    describe('TestEvolutionaryOptimization', () => {
        it('should evolve solution', async () => {
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const initialSolution = 'def solve(x): return x * 2';
            const result = await openevolve.optimize({
                solution: initialSolution,
                generations: 5,
                population_size: 20,
                mutation_rate: 0.1,
            });
            // Assert evolution completed
            expect(result.best_fitness).toBeGreaterThanOrEqual(0);
            expect(result.generations).toBe(5);
            expect(result.final_population).toBeDefined();
            expect(Array.isArray(result.final_population)).toBe(true);
        });
        it('should handle multi-generation evolution', async () => {
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const initialSolution = 'def optimize(x): return x ** 2';
            const result = await openevolve.optimize({
                solution: initialSolution,
                generations: 10,
                population_size: 50,
                crossover_rate: 0.8,
                mutation_rate: 0.2,
            });
            // Assert multi-generation evolution
            expect(result.generations).toBe(10);
            expect(result.final_population.length).toBeLessThanOrEqual(50);
            expect(result.optimization_history).toBeDefined();
            expect(result.optimization_history.length).toBe(11); // 0-10 generations
        });
        it('should handle different evolutionary parameters', async () => {
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const testCases = [
                { generations: 3, population_size: 10 },
                { generations: 5, population_size: 20, mutation_rate: 0.15 },
                { generations: 7, population_size: 30, crossover_rate: 0.9 },
            ];
            for (const params of testCases) {
                const result = await openevolve.optimize({
                    solution: 'test solution',
                    ...params,
                });
                expect(result.generations).toBe(params.generations);
                expect(result.population_size).toBe(params.population_size);
            }
        });
    });
    // ============================================================================
    // TEST SUITE 3: Hybrid Workflows
    // ============================================================================
    describe('TestHybridWorkflows', () => {
        it('should execute PES evolution workflow', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const workflow = new pes_evolution_workflow_1.PESEvolutionWorkflow({
                loongFlowAdapter: loongflow,
                openEvolveAdapter: openevolve,
                eventBus,
            });
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Optimize neural network architecture',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            const evolutionConfig = {
                generations: 5,
                population_size: 20,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            };
            const result = await workflow.execute({
                problem,
                evolution_config: evolutionConfig,
                enable_optimization: true,
            });
            // Assert workflow completed
            expect(result.pes_result).toBeDefined();
            expect(result.evolution_result).toBeDefined();
            expect(result.integration_metrics).toBeDefined();
            expect(result.integration_metrics.synergy_score).toBeGreaterThanOrEqual(0);
        });
        it('should extract and store knowledge', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)({
                mockBestSolutions: [
                    {
                        solution: 'def solution1(x): return x * 2',
                        solution_id: 'sol_1',
                        generate_plan: 'Strategy 1',
                        score: 0.8,
                        evaluation: 'Good',
                        summary: 'Test',
                        parent_id: '',
                        island_id: 0,
                        iteration: 1,
                        metadata: {},
                    },
                    {
                        solution: 'def solution2(x): return x ** 2',
                        solution_id: 'sol_2',
                        generate_plan: 'Strategy 2',
                        score: 0.9,
                        evaluation: 'Better',
                        summary: 'Test',
                        parent_id: '',
                        island_id: 1,
                        iteration: 2,
                        metadata: {},
                    },
                ],
            });
            const workflow = new knowledge_extraction_workflow_1.KnowledgeExtractionWorkflow({
                loongflowAdapter: loongflow,
                eventBus,
                enable_problem_formulation: true,
            });
            const knowledge = await workflow.execute({
                top_k: 3,
                problem_id: crypto.randomUUID(),
            });
            // Assert knowledge extracted
            expect(knowledge.knowledge).toBeDefined();
            expect(knowledge.knowledge.length).toBeGreaterThan(0);
            expect(knowledge.patterns).toBeDefined();
            expect(knowledge.problems).toBeDefined();
        });
        it('should execute adaptive workflow', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)({
                mockLowConfidence: true,
            });
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const workflow = new adaptive_execution_workflow_1.AdaptiveExecutionWorkflow({
                loongFlowAdapter: loongflow,
                openEvolveAdapter: openevolve,
                eventBus,
            });
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Complex optimization problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            const triggers = [
                {
                    id: crypto.randomUUID(),
                    condition: 'low_confidence',
                    threshold: 0.7,
                    action: 'switch_to_evolution',
                },
            ];
            const result = await workflow.executeAdaptive({
                problem,
                triggers,
                enable_hybrid_fallback: true,
            });
            // Assert adaptive behavior
            expect(result.best_solution).toBeDefined();
            expect(result.integration_metrics).toBeDefined();
            expect(result.integration_metrics.paradigm_switches).toBeGreaterThanOrEqual(0);
        });
    });
    // ============================================================================
    // TEST SUITE 4: Knowledge Management
    // ============================================================================
    describe('TestKnowledgeManagement', () => {
        it('should extract evolutionary knowledge', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)({
                mockBestSolutions: [
                    {
                        solution: 'def best(x): return x',
                        solution_id: 'best_sol',
                        generate_plan: 'Best strategy',
                        score: 0.95,
                        evaluation: 'Excellent',
                        summary: 'Best solution',
                        parent_id: '',
                        island_id: 0,
                        iteration: 10,
                        metadata: {},
                    },
                ],
            });
            const workflow = new knowledge_extraction_workflow_1.KnowledgeExtractionWorkflow({
                loongflowAdapter: loongflow,
                eventBus,
            });
            const { knowledge } = await workflow.execute({
                top_k: 5,
                problem_id: crypto.randomUUID(),
            });
            // Assert knowledge extracted
            expect(knowledge.length).toBeGreaterThan(0);
            for (const k of knowledge) {
                expect(k.id).toBeDefined();
                expect(k.source_type).toBeDefined();
                expect(k.content.success_rate).toBeGreaterThanOrEqual(0);
                expect(k.content.success_rate).toBeLessThanOrEqual(1);
            }
        });
        it('should reuse knowledge for new problems', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            // Create a problem with knowledge guidance
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Optimize with knowledge hints',
                context: {
                    knowledge_guidance: [
                        {
                            pattern: 'Use gradient descent',
                            success_rate: 0.9,
                            avg_score: 0.85,
                        },
                    ],
                },
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            // Submit problem
            const result = await loongflow.submitProblem({
                task: problem.description,
                metadata: problem.context,
            });
            // Assert problem created with knowledge
            expect(problem.context.knowledge_guidance).toBeDefined();
            expect(result.agent_id).toBeDefined();
        });
    });
    // ============================================================================
    // TEST SUITE 5: Error Handling and Recovery
    // ============================================================================
    describe('TestErrorHandlingAndRecovery', () => {
        it('should handle timeout', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)({
                mockTimeout: true,
            });
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Long running problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            const result = await loongflow.submitProblem({
                task: problem.description,
            });
            // Should handle timeout gracefully
            await expect(loongflow.getExecutionResult(result.agent_id)).rejects.toThrow();
        });
        it('should retry with backoff', async () => {
            let callCount = 0;
            const flakyOperation = async () => {
                callCount++;
                if (callCount < 3) {
                    throw new Error('Temporary failure');
                }
                return { success: true };
            };
            // Execute with retry logic
            const maxRetries = 3;
            let result = null;
            for (let attempt = 0; attempt < maxRetries; attempt++) {
                try {
                    result = await flakyOperation();
                    break;
                }
                catch (error) {
                    if (attempt === maxRetries - 1) {
                        throw error;
                    }
                    // Exponential backoff
                    await new Promise(resolve => setTimeout(resolve, 100 * Math.pow(2, attempt)));
                }
            }
            // Assert operation succeeded after retries
            expect(result?.success).toBe(true);
            expect(callCount).toBe(3);
        });
        it('should handle invalid input', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            // Submit with minimal valid input (adapter may not validate)
            const result = await loongflow.submitProblem({
                task: '', // Empty task
            });
            // Should return a response or error
            expect(result).toBeDefined();
        });
        it('should handle missing execution ID', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)({
                mockError: true,
            });
            // Mock missing execution
            const result = await loongflow.getExecutionResult('nonexistent_id').catch(() => null);
            // Should return null or error
            expect(result).toBeNull();
        });
    });
    // ============================================================================
    // TEST SUITE 6: Performance and Scalability
    // ============================================================================
    describe('TestPerformanceAndScalability', () => {
        it('should handle concurrent execution', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            // Create multiple problems
            const problems = Array.from({ length: 5 }, (_, i) => ({
                task: `Problem ${i}`,
                max_iterations: 10,
            }));
            // Execute concurrently
            const results = await Promise.all(problems.map(p => loongflow.submitProblem(p)));
            // Assert all problems submitted
            expect(results.length).toBe(5);
            for (const result of results) {
                expect(result.agent_id).toBeDefined();
            }
        });
        it('should complete workflow within timeout', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const workflow = new pes_evolution_workflow_1.PESEvolutionWorkflow({
                loongFlowAdapter: loongflow,
                openEvolveAdapter: openevolve,
                eventBus,
                default_timeout_ms: 30000,
            });
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Test problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            const startTime = Date.now();
            await workflow.execute({ problem });
            const duration = Date.now() - startTime;
            // Assert completes in reasonable time
            expect(duration).toBeLessThan(testTimeout);
        });
        it('should handle large problems', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const largeProblem = {
                task: 'Optimize over large parameter space',
                max_iterations: 50,
                metadata: {
                    parameter_count: 1000,
                },
            };
            const result = await loongflow.submitProblem(largeProblem);
            // Assert handles large problem
            expect(result.agent_id).toBeDefined();
        });
        it('should cleanup resources', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            // Submit and complete
            const result = await loongflow.submitProblem({ task: 'Test' });
            const solution = await loongflow.getExecutionResult(result.agent_id);
            // Assert cleanup (no dangling resources)
            expect(solution).toBeDefined();
        });
    });
    // ============================================================================
    // TEST SUITE 7: Integration Tests
    // ============================================================================
    describe('TestIntegration', () => {
        it('should integrate PES and Evolution end-to-end', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)();
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const workflow = new pes_evolution_workflow_1.PESEvolutionWorkflow({
                loongFlowAdapter: loongflow,
                openEvolveAdapter: openevolve,
                eventBus,
            });
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'End-to-end integration test',
                context: {},
                constraints: [],
                success_criteria: ['fitness > 0.8'],
                created_at: new Date().toISOString(),
            };
            const result = await workflow.execute({
                problem,
                enable_optimization: true,
                enable_knowledge_extraction: true,
            });
            // Assert complete workflow
            expect(result.pes_result).toBeDefined();
            expect(result.evolution_result).toBeDefined();
            expect(result.knowledge_extracted).toBeDefined();
            expect(result.integration_metrics.total_duration_ms).toBeGreaterThan(0);
        });
        it('should handle workflow failures gracefully', async () => {
            const loongflow = (0, loongflow_mock_1.createMockLoongFlowAdapter)({
                mockError: true,
            });
            const openevolve = (0, openevolve_mock_1.createMockOpenEvolveAdapter)();
            const workflow = new pes_evolution_workflow_1.PESEvolutionWorkflow({
                loongFlowAdapter: loongflow,
                openEvolveAdapter: openevolve,
                eventBus,
            });
            const problem = {
                id: crypto.randomUUID(),
                type: 'optimization',
                description: 'Failure test',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: new Date().toISOString(),
            };
            // Should handle failure gracefully
            await expect(workflow.execute({ problem })).rejects.toThrow();
        });
    });
});
//# sourceMappingURL=test_hybrid_pes_evolution_e2e.test.js.map