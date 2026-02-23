"use strict";
/**
 * Hybrid PES-Evolution Schemas Validation Tests
 *
 * Comprehensive test suite for Hybrid PES-Evolution canonical schemas.
 * Tests all schemas for proper validation, transformation, and type safety.
 *
 * Law of Runtime Truth: These tests verify that schemas actually work as expected.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const hybrid_pes_evolution_canonical_1 = require("../hybrid-pes-evolution-canonical");
const pes_canonical_1 = require("../pes-canonical");
(0, globals_1.describe)('Hybrid PES-Evolution Canonical Schemas', () => {
    (0, globals_1.describe)('EvolutionConfig Schema', () => {
        (0, globals_1.it)('should validate a valid evolution config', () => {
            const validConfig = {
                generations: 100,
                population_size: 50,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
                selection_method: 'tournament',
                elitism_count: 2,
                mutation_strength: 0.5,
                representation_type: 'binary',
                metadata: { experiment: 'exp_001' },
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionConfig.safeParse(validConfig);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.generations).toBe(100);
                (0, globals_1.expect)(result.data.mutation_rate).toBe(0.1);
                (0, globals_1.expect)(result.data.crossover_rate).toBe(0.8);
            }
        });
        (0, globals_1.it)('should reject mutation_rate out of range', () => {
            const invalidConfig = {
                generations: 100,
                population_size: 50,
                mutation_rate: 1.5, // Invalid: must be 0-1
                crossover_rate: 0.8,
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative generations', () => {
            const invalidConfig = {
                generations: -10, // Invalid: must be positive
                population_size: 50,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject crossover_rate out of range', () => {
            const invalidConfig = {
                generations: 100,
                population_size: 50,
                mutation_rate: 0.1,
                crossover_rate: -0.1, // Invalid: must be 0-1
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('HybridTask Schema', () => {
        const createValidProblem = () => ({
            id: (0, pes_canonical_1.createPESCorrelationId)(),
            type: 'optimization',
            description: 'Test optimization problem',
            context: {},
            constraints: [],
            success_criteria: [],
            created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
        });
        const createValidLoongFlowConfig = () => ({
            task_name: 'Test task',
            llm_config: {
                provider: 'openai',
                model: 'gpt-4',
                temperature: 0.7,
                max_tokens: 1000,
            },
            workers: {
                planner: 'Planner',
                executor: 'Executor',
                summarizer: 'Summarizer',
            },
            evolution: {
                iterations: 10,
                islands: 2,
                sample_size: 20,
            },
        });
        (0, globals_1.it)('should validate a valid hybrid task', () => {
            const validTask = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'adaptive_execute',
                problem: createValidProblem(),
                pes_config: createValidLoongFlowConfig(),
                evolution_config: {
                    generations: 50,
                    population_size: 100,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                },
                integration_strategy: 'adaptive',
                adaptive_triggers: [
                    {
                        condition: 'stagnation',
                        threshold: 5,
                        action: 'switch_to_evolution',
                    },
                    {
                        condition: 'low_confidence',
                        threshold: 0.6,
                        action: 'increase_iterations',
                    },
                ],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                correlation_id: (0, pes_canonical_1.createPESCorrelationId)(),
                priority: 7,
                timeout_ms: 300000,
            };
            const result = hybrid_pes_evolution_canonical_1.HybridTask.safeParse(validTask);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.type).toBe('adaptive_execute');
                (0, globals_1.expect)(result.data.integration_strategy).toBe('adaptive');
                (0, globals_1.expect)(result.data.adaptive_triggers).toHaveLength(2);
            }
        });
        (0, globals_1.it)('should validate task without optional pes_config', () => {
            const validTask = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'evolve_solve',
                problem: createValidProblem(),
                evolution_config: {
                    generations: 100,
                    population_size: 50,
                    mutation_rate: 0.2,
                    crossover_rate: 0.7,
                },
                integration_strategy: 'sequential',
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.HybridTask.safeParse(validTask);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.it)('should reject invalid priority range', () => {
            const invalidTask = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'pes_optimize',
                problem: createValidProblem(),
                pes_config: createValidLoongFlowConfig(),
                evolution_config: {
                    generations: 50,
                    population_size: 100,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                },
                integration_strategy: 'parallel',
                priority: 15, // Invalid: must be 1-10
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.HybridTask.safeParse(invalidTask);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('EvolutionaryKnowledge Schema', () => {
        (0, globals_1.it)('should validate a valid knowledge', () => {
            const validKnowledge = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_type: 'loongflow_solution',
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                knowledge_type: 'solution_pattern',
                content: {
                    pattern: 'Use gradient descent for convex optimization',
                    success_rate: 0.9,
                    avg_score: 0.85,
                    usage_count: 10,
                    context: { domain: 'optimization' },
                },
                source_id: 'solution_123',
                valid_until: (0, pes_canonical_1.createPESUTCTimestamp)(),
                extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                metadata: { verified: true },
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionaryKnowledge.safeParse(validKnowledge);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.source_type).toBe('loongflow_solution');
                (0, globals_1.expect)(result.data.knowledge_type).toBe('solution_pattern');
                (0, globals_1.expect)(result.data.content.success_rate).toBe(0.9);
            }
        });
        (0, globals_1.it)('should reject success_rate out of range', () => {
            const invalidKnowledge = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_type: 'openevolve_optimization',
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                knowledge_type: 'planning_strategy',
                content: {
                    pattern: 'Test pattern',
                    success_rate: 1.5, // Invalid: must be 0-1
                    avg_score: 0.8,
                    usage_count: 5,
                },
                extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionaryKnowledge.safeParse(invalidKnowledge);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative usage_count', () => {
            const invalidKnowledge = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_type: 'hybrid_result',
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                knowledge_type: 'execution_approach',
                content: {
                    pattern: 'Test pattern',
                    success_rate: 0.8,
                    avg_score: 0.75,
                    usage_count: -1, // Invalid: must be non-negative
                },
                extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionaryKnowledge.safeParse(invalidKnowledge);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('PopulationIndividual Schema', () => {
        (0, globals_1.it)('should validate a valid individual', () => {
            const validIndividual = {
                id: 'individual_123',
                genes: [1, 0, 1, 1, 0],
                fitness: 0.95,
                metadata: { generation: 5 },
                parent_ids: ['parent_1', 'parent_2'],
                generation: 10,
            };
            const result = hybrid_pes_evolution_canonical_1.PopulationIndividual.safeParse(validIndividual);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.fitness).toBe(0.95);
                (0, globals_1.expect)(result.data.genes).toEqual([1, 0, 1, 1, 0]);
                (0, globals_1.expect)(result.data.parent_ids).toHaveLength(2);
            }
        });
        (0, globals_1.it)('should reject negative fitness', () => {
            const invalidIndividual = {
                id: 'ind_1',
                genes: [1, 0],
                fitness: -0.5, // Invalid: must be >= 0
            };
            const result = hybrid_pes_evolution_canonical_1.PopulationIndividual.safeParse(invalidIndividual);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative generation', () => {
            const invalidIndividual = {
                id: 'ind_1',
                genes: [1],
                fitness: 0.8,
                generation: -5, // Invalid: must be non-negative
            };
            const result = hybrid_pes_evolution_canonical_1.PopulationIndividual.safeParse(invalidIndividual);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('EvolutionResult Schema', () => {
        (0, globals_1.it)('should validate a valid evolution result', () => {
            const validResult = {
                best_fitness: 0.98,
                generations: 50,
                population: [
                    {
                        id: 'ind_1',
                        genes: [1, 0, 1],
                        fitness: 0.98,
                    },
                    {
                        id: 'ind_2',
                        genes: [0, 1, 0],
                        fitness: 0.95,
                    },
                ],
                converged: true,
                convergence_generation: 45,
                diversity_score: 0.15,
                metadata: { algorithm: 'genetic_algorithm' },
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionResult.safeParse(validResult);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.best_fitness).toBe(0.98);
                (0, globals_1.expect)(result.data.population).toHaveLength(2);
                (0, globals_1.expect)(result.data.converged).toBe(true);
            }
        });
        (0, globals_1.it)('should reject negative generations', () => {
            const invalidResult = {
                best_fitness: 0.9,
                generations: -10, // Invalid: must be non-negative
                population: [],
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionResult.safeParse(invalidResult);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject diversity_score out of range', () => {
            const invalidResult = {
                best_fitness: 0.9,
                generations: 50,
                population: [],
                diversity_score: 1.5, // Invalid: must be 0-1
            };
            const result = hybrid_pes_evolution_canonical_1.EvolutionResult.safeParse(invalidResult);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('IntegrationMetrics Schema', () => {
        (0, globals_1.it)('should validate valid integration metrics', () => {
            const validMetrics = {
                pes_iterations: 5,
                evolution_generations: 50,
                total_duration_ms: 120000,
                synergy_score: 0.85,
                pes_time_ms: 30000,
                evolution_time_ms: 90000,
                paradigm_switches: 3,
                knowledge_transferred: 7,
            };
            const result = hybrid_pes_evolution_canonical_1.IntegrationMetrics.safeParse(validMetrics);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.synergy_score).toBe(0.85);
                (0, globals_1.expect)(result.data.paradigm_switches).toBe(3);
            }
        });
        (0, globals_1.it)('should reject negative values', () => {
            const invalidMetrics = {
                pes_iterations: -5, // Invalid: must be non-negative
                evolution_generations: 50,
                total_duration_ms: 100000,
                synergy_score: 0.8,
            };
            const result = hybrid_pes_evolution_canonical_1.IntegrationMetrics.safeParse(invalidMetrics);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject synergy_score out of range', () => {
            const invalidMetrics = {
                pes_iterations: 5,
                evolution_generations: 50,
                total_duration_ms: 100000,
                synergy_score: 1.2, // Invalid: must be 0-1
            };
            const result = hybrid_pes_evolution_canonical_1.IntegrationMetrics.safeParse(invalidMetrics);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('HybridExecutionResult Schema', () => {
        (0, globals_1.it)('should validate a valid hybrid execution result', () => {
            const validResult = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                task_id: (0, pes_canonical_1.createPESCorrelationId)(),
                pes_result: {
                    id: (0, pes_canonical_1.createPESCorrelationId)(),
                    problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                    plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                    state: 'completed',
                    outputs: {},
                    metrics: {
                        duration_ms: 5000,
                        success: true,
                        error_count: 0,
                    },
                    created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                },
                evolution_result: {
                    best_fitness: 0.95,
                    generations: 100,
                    population: [],
                },
                integration_metrics: {
                    pes_iterations: 3,
                    evolution_generations: 100,
                    total_duration_ms: 60000,
                    synergy_score: 0.92,
                },
                best_solution: {
                    solution: 'best solution',
                    solution_id: 'best_1',
                    generate_plan: 'plan',
                    score: 0.95,
                    evaluation: 'excellent',
                    summary: 'summary',
                    parent_id: '',
                    island_id: 0,
                    iteration: 10,
                },
                summary: 'Hybrid execution completed successfully',
                knowledge_extracted: [
                    {
                        id: (0, pes_canonical_1.createPESCorrelationId)(),
                        source_type: 'loongflow_solution',
                        problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                        knowledge_type: 'solution_pattern',
                        content: {
                            pattern: 'Test pattern',
                            success_rate: 0.9,
                            avg_score: 0.85,
                            usage_count: 5,
                        },
                        extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                    },
                ],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                completed_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                correlation_id: (0, pes_canonical_1.createPESCorrelationId)(),
            };
            const result = hybrid_pes_evolution_canonical_1.HybridExecutionResult.safeParse(validResult);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.pes_result).toBeDefined();
                (0, globals_1.expect)(result.data.evolution_result).toBeDefined();
                (0, globals_1.expect)(result.data.knowledge_extracted).toHaveLength(1);
            }
        });
        (0, globals_1.it)('should validate failed result with error', () => {
            const failedResult = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                task_id: (0, pes_canonical_1.createPESCorrelationId)(),
                integration_metrics: {
                    pes_iterations: 0,
                    evolution_generations: 0,
                    total_duration_ms: 1000,
                    synergy_score: 0.0,
                },
                error: {
                    code: 'TIMEOUT',
                    message: 'Execution exceeded timeout',
                    details: { timeout_ms: 300000 },
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                completed_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.HybridExecutionResult.safeParse(failedResult);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.error).toBeDefined();
                (0, globals_1.expect)(result.data.error?.code).toBe('TIMEOUT');
            }
        });
    });
    (0, globals_1.describe)('AdaptiveTrigger Schema', () => {
        (0, globals_1.it)('should validate a valid adaptive trigger', () => {
            const validTrigger = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                condition: 'stagnation',
                threshold: 5,
                action: 'switch_to_evolution',
                cooldown_ms: 10000,
                metadata: { description: 'Switch to evolution if no improvement for 5 iterations' },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                last_triggered_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.AdaptiveTrigger.safeParse(validTrigger);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.condition).toBe('stagnation');
                (0, globals_1.expect)(result.data.action).toBe('switch_to_evolution');
            }
        });
        (0, globals_1.it)('should reject negative cooldown', () => {
            const invalidTrigger = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                condition: 'low_confidence',
                threshold: 0.6,
                action: 'increase_iterations',
                cooldown_ms: -1000, // Invalid: must be positive
            };
            const result = hybrid_pes_evolution_canonical_1.AdaptiveTrigger.safeParse(invalidTrigger);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('KnowledgeTransfer Schema', () => {
        (0, globals_1.it)('should validate a valid knowledge transfer', () => {
            const validTransfer = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_paradigm: 'pes',
                destination_paradigm: 'evolution',
                knowledge: {
                    id: (0, pes_canonical_1.createPESCorrelationId)(),
                    source_type: 'loongflow_solution',
                    problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                    knowledge_type: 'solution_pattern',
                    content: {
                        pattern: 'Test pattern',
                        success_rate: 0.9,
                        avg_score: 0.85,
                        usage_count: 5,
                    },
                    extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                },
                status: 'completed',
                transfer_result: { success: true },
                initiated_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                completed_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                metadata: { transfer_method: 'direct' },
            };
            const result = hybrid_pes_evolution_canonical_1.KnowledgeTransfer.safeParse(validTransfer);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.source_paradigm).toBe('pes');
                (0, globals_1.expect)(result.data.destination_paradigm).toBe('evolution');
                (0, globals_1.expect)(result.data.status).toBe('completed');
            }
        });
        (0, globals_1.it)('should reject invalid status', () => {
            const invalidTransfer = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_paradigm: 'evolution',
                destination_paradigm: 'pes',
                knowledge: {
                    id: (0, pes_canonical_1.createPESCorrelationId)(),
                    source_type: 'hybrid_result',
                    problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                    knowledge_type: 'planning_strategy',
                    content: {
                        pattern: 'Test',
                        success_rate: 0.8,
                        avg_score: 0.75,
                        usage_count: 3,
                    },
                    extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                },
                status: 'invalid_status', // Invalid enum value
                initiated_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = hybrid_pes_evolution_canonical_1.KnowledgeTransfer.safeParse(invalidTransfer);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Validation Functions', () => {
        const createValidProblem = () => ({
            id: (0, pes_canonical_1.createPESCorrelationId)(),
            type: 'optimization',
            description: 'Test',
            context: {},
            constraints: [],
            success_criteria: [],
            created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
        });
        (0, globals_1.it)('should validate hybrid task using validateHybridTask', () => {
            const task = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'adaptive_execute',
                problem: createValidProblem(),
                evolution_config: {
                    generations: 50,
                    population_size: 100,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                },
                integration_strategy: 'adaptive',
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, hybrid_pes_evolution_canonical_1.validateHybridTask)(task);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate evolutionary knowledge using validateEvolutionaryKnowledge', () => {
            const knowledge = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_type: 'loongflow_solution',
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                knowledge_type: 'solution_pattern',
                content: {
                    pattern: 'Test pattern',
                    success_rate: 0.9,
                    avg_score: 0.85,
                    usage_count: 5,
                },
                extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, hybrid_pes_evolution_canonical_1.validateEvolutionaryKnowledge)(knowledge);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate hybrid execution result using validateHybridExecutionResult', () => {
            const result = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                task_id: (0, pes_canonical_1.createPESCorrelationId)(),
                integration_metrics: {
                    pes_iterations: 5,
                    evolution_generations: 50,
                    total_duration_ms: 60000,
                    synergy_score: 0.85,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, hybrid_pes_evolution_canonical_1.validateHybridExecutionResult)(result);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate adaptive trigger using validateAdaptiveTrigger', () => {
            const trigger = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                condition: 'stagnation',
                threshold: 5,
                action: 'switch_to_evolution',
            };
            const validation = (0, hybrid_pes_evolution_canonical_1.validateAdaptiveTrigger)(trigger);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
    });
    (0, globals_1.describe)('Type Guards', () => {
        const createValidProblem = () => ({
            id: (0, pes_canonical_1.createPESCorrelationId)(),
            type: 'optimization',
            description: 'Test',
            context: {},
            constraints: [],
            success_criteria: [],
            created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
        });
        (0, globals_1.it)('should check if data is a valid HybridTask', () => {
            const task = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'pes_optimize',
                problem: createValidProblem(),
                evolution_config: {
                    generations: 50,
                    population_size: 100,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                },
                integration_strategy: 'sequential',
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isHybridTask)(task)).toBe(true);
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isHybridTask)({ not: 'a task' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid HybridExecutionResult', () => {
            const result = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                task_id: (0, pes_canonical_1.createPESCorrelationId)(),
                integration_metrics: {
                    pes_iterations: 5,
                    evolution_generations: 50,
                    total_duration_ms: 60000,
                    synergy_score: 0.85,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isHybridExecutionResult)(result)).toBe(true);
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isHybridExecutionResult)({ not: 'a result' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid EvolutionaryKnowledge', () => {
            const knowledge = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                source_type: 'loongflow_solution',
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                knowledge_type: 'solution_pattern',
                content: {
                    pattern: 'Test',
                    success_rate: 0.9,
                    avg_score: 0.85,
                    usage_count: 5,
                },
                extracted_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isEvolutionaryKnowledge)(knowledge)).toBe(true);
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isEvolutionaryKnowledge)({ not: 'knowledge' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid AdaptiveTrigger', () => {
            const trigger = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                condition: 'low_confidence',
                threshold: 0.6,
                action: 'increase_iterations',
            };
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isAdaptiveTrigger)(trigger)).toBe(true);
            (0, globals_1.expect)((0, hybrid_pes_evolution_canonical_1.isAdaptiveTrigger)({ not: 'a trigger' })).toBe(false);
        });
    });
    (0, globals_1.describe)('Transformation Functions', () => {
        (0, globals_1.it)('should transform LoongFlow solution to evolutionary knowledge', () => {
            const solution = {
                solution: 'test solution',
                solution_id: 'sol_123',
                generate_plan: 'test plan',
                score: 0.9,
                evaluation: 'good',
                summary: 'test summary',
                parent_id: '',
                island_id: 2,
                iteration: 5,
            };
            const problemId = (0, pes_canonical_1.createPESCorrelationId)();
            const knowledge = (0, hybrid_pes_evolution_canonical_1.transformLoongFlowSolutionToKnowledge)(solution, problemId, 'solution_pattern');
            (0, globals_1.expect)(knowledge.source_type).toBe('loongflow_solution');
            (0, globals_1.expect)(knowledge.problem_id).toBe(problemId);
            (0, globals_1.expect)(knowledge.knowledge_type).toBe('solution_pattern');
            (0, globals_1.expect)(knowledge.content.success_rate).toBe(0.9);
        });
        (0, globals_1.it)('should transform hybrid result to summary', () => {
            const result = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                task_id: (0, pes_canonical_1.createPESCorrelationId)(),
                integration_metrics: {
                    pes_iterations: 5,
                    evolution_generations: 50,
                    total_duration_ms: 60000,
                    synergy_score: 0.85,
                    pes_time_ms: 10000,
                    evolution_time_ms: 50000,
                    paradigm_switches: 3,
                },
                best_solution: {
                    solution: 'best',
                    solution_id: 'best_1',
                    generate_plan: 'plan',
                    score: 0.95,
                    evaluation: 'excellent',
                    summary: 'summary',
                    parent_id: '',
                    island_id: 0,
                    iteration: 10,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                completed_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const summary = (0, hybrid_pes_evolution_canonical_1.transformHybridResultToSummary)(result);
            (0, globals_1.expect)(summary.result_id).toBe(result.id);
            (0, globals_1.expect)(summary.performance_assessment.success_criteria_met).toBe(true);
            (0, globals_1.expect)(summary.performance_assessment.quality_score).toBe(0.95);
            (0, globals_1.expect)(summary.insights).toContain('PES iterations: 5');
            (0, globals_1.expect)(summary.insights).toContain('Evolution generations: 50');
        });
    });
    (0, globals_1.describe)('Enum Validation', () => {
        (0, globals_1.it)('should accept all valid hybrid task types', () => {
            const validTypes = [
                'pes_optimize',
                'evolve_solve',
                'adaptive_execute',
                'parallel_hybrid',
                'interleaved',
            ];
            validTypes.forEach((type) => {
                const result = hybrid_pes_evolution_canonical_1.HybridTaskType.safeParse(type);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.it)('should accept all valid integration strategies', () => {
            const validStrategies = ['sequential', 'parallel', 'interleaved', 'adaptive'];
            validStrategies.forEach((strategy) => {
                const result = hybrid_pes_evolution_canonical_1.IntegrationStrategy.safeParse(strategy);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.it)('should accept all valid adaptive trigger conditions', () => {
            const validConditions = [
                'low_confidence',
                'high_complexity',
                'stagnation',
                'timeout_risk',
                'convergence_detected',
                'divergence_detected',
            ];
            validConditions.forEach((condition) => {
                const result = hybrid_pes_evolution_canonical_1.AdaptiveTriggerCondition.safeParse(condition);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.it)('should accept all valid adaptive actions', () => {
            const validActions = [
                'switch_to_evolution',
                'switch_to_pes',
                'increase_iterations',
                'call_for_help',
                'merge_populations',
                'reset_and_restart',
            ];
            validActions.forEach((action) => {
                const result = hybrid_pes_evolution_canonical_1.AdaptiveAction.safeParse(action);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
    });
});
//# sourceMappingURL=hybrid-schemas.test.js.map