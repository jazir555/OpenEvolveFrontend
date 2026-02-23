"use strict";
/**
 * LoongFlow Schemas Validation Tests
 *
 * Comprehensive test suite for LoongFlow canonical schemas.
 * Tests all schemas for proper validation, transformation, and type safety.
 *
 * Law of Runtime Truth: These tests verify that schemas actually work as expected.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const loongflow_canonical_1 = require("../loongflow-canonical");
(0, globals_1.describe)('LoongFlow Canonical Schemas', () => {
    (0, globals_1.describe)('LoongFlowSolution Schema', () => {
        (0, globals_1.it)('should validate a valid solution', () => {
            const validSolution = {
                solution: 'def optimize(): return 42',
                solution_id: 'solution_123',
                generate_plan: 'Use gradient descent',
                score: 0.95,
                evaluation: 'Optimal solution found',
                summary: 'Successfully converged to global optimum',
                parent_id: 'solution_122',
                island_id: 2,
                iteration: 5,
                metadata: { algorithm: 'gradient_descent' },
                created_at: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
                fitness_map_key: { performance: 'high', complexity: 'low' },
            };
            const result = loongflow_canonical_1.LoongFlowSolution.safeParse(validSolution);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.score).toBe(0.95);
                (0, globals_1.expect)(result.data.island_id).toBe(2);
                (0, globals_1.expect)(result.data.iteration).toBe(5);
            }
        });
        (0, globals_1.it)('should reject score out of range', () => {
            const invalidSolution = {
                solution: 'test',
                solution_id: 'sol_1',
                generate_plan: 'plan',
                score: 1.5, // Invalid: must be 0-1
                evaluation: 'eval',
                summary: 'summary',
                parent_id: '',
                island_id: 0,
                iteration: 0,
            };
            const result = loongflow_canonical_1.LoongFlowSolution.safeParse(invalidSolution);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative island_id', () => {
            const invalidSolution = {
                solution: 'test',
                solution_id: 'sol_1',
                generate_plan: 'plan',
                score: 0.8,
                evaluation: 'eval',
                summary: 'summary',
                parent_id: '',
                island_id: -1, // Invalid: must be non-negative
                iteration: 0,
            };
            const result = loongflow_canonical_1.LoongFlowSolution.safeParse(invalidSolution);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative iteration', () => {
            const invalidSolution = {
                solution: 'test',
                solution_id: 'sol_1',
                generate_plan: 'plan',
                score: 0.8,
                evaluation: 'eval',
                summary: 'summary',
                parent_id: '',
                island_id: 0,
                iteration: -5, // Invalid: must be non-negative
            };
            const result = loongflow_canonical_1.LoongFlowSolution.safeParse(invalidSolution);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('LLMConfig Schema', () => {
        (0, globals_1.it)('should validate a valid LLM config', () => {
            const validConfig = {
                provider: 'openai',
                model: 'gpt-4',
                temperature: 0.7,
                max_tokens: 2000,
                top_p: 0.9,
                top_k: 50,
                api_key: 'sk-...',
                base_url: 'https://api.openai.com/v1',
                timeout_ms: 30000,
            };
            const result = loongflow_canonical_1.LLMConfig.safeParse(validConfig);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.provider).toBe('openai');
                (0, globals_1.expect)(result.data.temperature).toBe(0.7);
            }
        });
        (0, globals_1.it)('should reject temperature out of range', () => {
            const invalidConfig = {
                provider: 'anthropic',
                model: 'claude-3',
                temperature: 3.0, // Invalid: must be 0-2
                max_tokens: 1000,
            };
            const result = loongflow_canonical_1.LLMConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative temperature', () => {
            const invalidConfig = {
                provider: 'local',
                model: 'llama-2',
                temperature: -0.5, // Invalid: must be >= 0
                max_tokens: 1000,
            };
            const result = loongflow_canonical_1.LLMConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject invalid base_url', () => {
            const invalidConfig = {
                provider: 'openai',
                model: 'gpt-4',
                temperature: 0.5,
                max_tokens: 1000,
                base_url: 'not-a-valid-url', // Invalid: must be URL
            };
            const result = loongflow_canonical_1.LLMConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('WorkerConfig Schema', () => {
        (0, globals_1.it)('should validate a valid worker config', () => {
            const validConfig = {
                planner: 'LLMPlanner',
                executor: 'CodeExecutor',
                summarizer: 'QualitySummarizer',
                planner_config: { temperature: 0.8 },
                executor_config: { timeout: 30 },
                summarizer_config: { verbose: true },
            };
            const result = loongflow_canonical_1.WorkerConfig.safeParse(validConfig);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.planner).toBe('LLMPlanner');
                (0, globals_1.expect)(result.data.executor).toBe('CodeExecutor');
                (0, globals_1.expect)(result.data.summarizer).toBe('QualitySummarizer');
            }
        });
        (0, globals_1.it)('should reject empty planner', () => {
            const invalidConfig = {
                planner: '', // Invalid: cannot be empty
                executor: 'Executor',
                summarizer: 'Summarizer',
            };
            const result = loongflow_canonical_1.WorkerConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('EvolutionConfig Schema', () => {
        (0, globals_1.it)('should validate a valid evolution config', () => {
            const validConfig = {
                iterations: 100,
                islands: 4,
                sample_size: 50,
                map_elites_enabled: true,
                map_dimensions: [
                    { name: 'complexity', bins: 10 },
                    { name: 'performance', bins: 10 },
                ],
                temperature_initial: 1.0,
                temperature_decay: 0.95,
                migration_interval: 10,
                migration_size: 5,
                tournament_size: 3,
                elite_size: 2,
            };
            const result = loongflow_canonical_1.EvolutionConfig.safeParse(validConfig);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.iterations).toBe(100);
                (0, globals_1.expect)(result.data.islands).toBe(4);
                (0, globals_1.expect)(result.data.map_elites_enabled).toBe(true);
            }
        });
        (0, globals_1.it)('should reject negative iterations', () => {
            const invalidConfig = {
                iterations: -10, // Invalid: must be positive
                islands: 4,
                sample_size: 50,
            };
            const result = loongflow_canonical_1.EvolutionConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject mutation_rate out of range', () => {
            const invalidConfig = {
                iterations: 100,
                islands: 4,
                sample_size: 50,
                mutation_rate: 1.5, // Invalid: must be 0-1
            };
            const result = loongflow_canonical_1.EvolutionConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('LoongFlowConfig Schema', () => {
        (0, globals_1.it)('should validate a valid LoongFlow config', () => {
            const validConfig = {
                task_name: 'Optimize neural network',
                llm_config: {
                    provider: 'anthropic',
                    model: 'claude-3-opus',
                    temperature: 0.7,
                    max_tokens: 2000,
                },
                workers: {
                    planner: 'LLMPlanner',
                    executor: 'CodeExecutor',
                    summarizer: 'QualitySummarizer',
                },
                evolution: {
                    iterations: 50,
                    islands: 4,
                    sample_size: 100,
                },
                timeout_ms: 300000,
                checkpoint_interval: 10,
                checkpoint_dir: '/checkpoints',
                metadata: { experiment: 'exp_001' },
            };
            const result = loongflow_canonical_1.LoongFlowConfig.safeParse(validConfig);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.task_name).toBe('Optimize neural network');
                (0, globals_1.expect)(result.data.llm_config.provider).toBe('anthropic');
                (0, globals_1.expect)(result.data.evolution.iterations).toBe(50);
            }
        });
        (0, globals_1.it)('should reject empty task_name', () => {
            const invalidConfig = {
                task_name: '', // Invalid: cannot be empty
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
            };
            const result = loongflow_canonical_1.LoongFlowConfig.safeParse(invalidConfig);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('LoongFlowRequest Schema', () => {
        (0, globals_1.it)('should validate a valid LoongFlow request', () => {
            const validRequest = {
                problem_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                task_config: {
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
                },
                initial_context: { dataset: 'MNIST' },
                seed_solutions: [
                    {
                        solution: 'seed solution',
                        solution_id: 'seed_1',
                        score: 0.8,
                        evaluation: 'good',
                        summary: 'summary',
                        parent_id: '',
                        island_id: 0,
                        iteration: 0,
                    },
                ],
                correlation_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                created_at: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
            };
            const result = loongflow_canonical_1.LoongFlowRequest.safeParse(validRequest);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.seed_solutions).toBeDefined();
                (0, globals_1.expect)(result.data.seed_solutions).toHaveLength(1);
            }
        });
        (0, globals_1.it)('should reject invalid UUID for problem_id', () => {
            const invalidRequest = {
                problem_id: 'not-a-uuid', // Invalid: must be UUID
                task_config: {
                    task_name: 'Test',
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
                },
                initial_context: {},
            };
            const result = loongflow_canonical_1.LoongFlowRequest.safeParse(invalidRequest);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('LoongFlowResponse Schema', () => {
        (0, globals_1.it)('should validate a valid LoongFlow response', () => {
            const validResponse = {
                request_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                current_solution: {
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
                best_solutions: [
                    {
                        solution: 'solution 1',
                        solution_id: 'sol_1',
                        generate_plan: 'plan 1',
                        score: 0.9,
                        evaluation: 'good',
                        summary: 'summary 1',
                        parent_id: '',
                        island_id: 0,
                        iteration: 5,
                    },
                    {
                        solution: 'solution 2',
                        solution_id: 'sol_2',
                        generate_plan: 'plan 2',
                        score: 0.95,
                        evaluation: 'excellent',
                        summary: 'summary 2',
                        parent_id: '',
                        island_id: 1,
                        iteration: 8,
                    },
                ],
                iteration: 10,
                checkpoints: ['checkpoint_1', 'checkpoint_2'],
                metrics: {
                    total_duration_ms: 60000,
                    planning_time_ms: 10000,
                    execution_time_ms: 40000,
                    summarization_time_ms: 10000,
                    solutions_generated: 100,
                    islands_completed: 2,
                },
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
                correlation_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                metadata: { experiment_id: 'exp_001' },
            };
            const result = loongflow_canonical_1.LoongFlowResponse.safeParse(validResponse);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.state).toBe('completed');
                (0, globals_1.expect)(result.data.best_solutions).toHaveLength(2);
                (0, globals_1.expect)(result.data.checkpoints).toHaveLength(2);
            }
        });
        (0, globals_1.it)('should validate failed response with error', () => {
            const failedResponse = {
                request_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                state: 'failed',
                best_solutions: [],
                iteration: 3,
                checkpoints: [],
                error: {
                    code: 'TIMEOUT',
                    message: 'Execution exceeded timeout',
                    details: { timeout_ms: 300000 },
                },
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
            };
            const result = loongflow_canonical_1.LoongFlowResponse.safeParse(failedResponse);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.state).toBe('failed');
                (0, globals_1.expect)(result.data.error).toBeDefined();
                (0, globals_1.expect)(result.data.error?.code).toBe('TIMEOUT');
            }
        });
        (0, globals_1.it)('should reject negative iteration', () => {
            const invalidResponse = {
                request_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                state: 'running',
                best_solutions: [],
                iteration: -5, // Invalid: must be non-negative
                checkpoints: [],
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
            };
            const result = loongflow_canonical_1.LoongFlowResponse.safeParse(invalidResponse);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('LoongFlowCheckpoint Schema', () => {
        (0, globals_1.it)('should validate a valid checkpoint', () => {
            const validCheckpoint = {
                checkpoint_id: 'checkpoint_5',
                iteration: 5,
                state: 'evolving',
                populations: [
                    {
                        island_id: 0,
                        solutions: [
                            {
                                solution: 'solution',
                                solution_id: 'sol_1',
                                generate_plan: 'plan',
                                score: 0.8,
                                evaluation: 'good',
                                summary: 'summary',
                                parent_id: '',
                                island_id: 0,
                                iteration: 5,
                            },
                        ],
                    },
                    {
                        island_id: 1,
                        solutions: [
                            {
                                solution: 'solution 2',
                                solution_id: 'sol_2',
                                generate_plan: 'plan 2',
                                score: 0.85,
                                evaluation: 'better',
                                summary: 'summary 2',
                                parent_id: '',
                                island_id: 1,
                                iteration: 5,
                            },
                        ],
                    },
                ],
                fitness_maps: [
                    {
                        island_id: 0,
                        map: {
                            'low_complexity': [
                                {
                                    solution: 'solution',
                                    solution_id: 'sol_1',
                                    generate_plan: 'plan',
                                    score: 0.8,
                                    evaluation: 'good',
                                    summary: 'summary',
                                    parent_id: '',
                                    island_id: 0,
                                    iteration: 5,
                                },
                            ],
                        },
                    },
                ],
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
                metadata: { convergence_score: 0.9 },
            };
            const result = loongflow_canonical_1.LoongFlowCheckpoint.safeParse(validCheckpoint);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.iteration).toBe(5);
                (0, globals_1.expect)(result.data.populations).toHaveLength(2);
                (0, globals_1.expect)(result.data.fitness_maps).toBeDefined();
            }
        });
        (0, globals_1.it)('should reject negative iteration', () => {
            const invalidCheckpoint = {
                checkpoint_id: 'checkpoint_1',
                iteration: -1, // Invalid: must be non-negative
                state: 'running',
                populations: [],
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
            };
            const result = loongflow_canonical_1.LoongFlowCheckpoint.safeParse(invalidCheckpoint);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Validation Functions', () => {
        (0, globals_1.it)('should validate solution using validateLoongFlowSolution', () => {
            const solution = {
                solution: 'test solution',
                solution_id: 'sol_1',
                generate_plan: 'test plan',
                score: 0.9,
                evaluation: 'good',
                summary: 'test summary',
                parent_id: '',
                island_id: 0,
                iteration: 1,
            };
            const validation = (0, loongflow_canonical_1.validateLoongFlowSolution)(solution);
            (0, globals_1.expect)(validation.success).toBe(true);
            (0, globals_1.expect)(validation.data).toBeDefined();
        });
        (0, globals_1.it)('should validate config using validateLoongFlowConfig', () => {
            const config = {
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
            };
            const validation = (0, loongflow_canonical_1.validateLoongFlowConfig)(config);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate request using validateLoongFlowRequest', () => {
            const request = {
                problem_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                task_config: {
                    task_name: 'Test',
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
                },
                initial_context: {},
            };
            const validation = (0, loongflow_canonical_1.validateLoongFlowRequest)(request);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate response using validateLoongFlowResponse', () => {
            const response = {
                request_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                best_solutions: [],
                iteration: 10,
                checkpoints: [],
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, loongflow_canonical_1.validateLoongFlowResponse)(response);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
    });
    (0, globals_1.describe)('Type Guards', () => {
        (0, globals_1.it)('should check if data is a valid LoongFlowSolution', () => {
            const solution = {
                solution: 'test',
                solution_id: 'sol_1',
                generate_plan: 'plan',
                score: 0.8,
                evaluation: 'eval',
                summary: 'summary',
                parent_id: '',
                island_id: 0,
                iteration: 0,
            };
            (0, globals_1.expect)((0, loongflow_canonical_1.isLoongFlowSolution)(solution)).toBe(true);
            (0, globals_1.expect)((0, loongflow_canonical_1.isLoongFlowSolution)({ not: 'a solution' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid LoongFlowRequest', () => {
            const request = {
                problem_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                task_config: {
                    task_name: 'Test',
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
                },
                initial_context: {},
            };
            (0, globals_1.expect)((0, loongflow_canonical_1.isLoongFlowRequest)(request)).toBe(true);
            (0, globals_1.expect)((0, loongflow_canonical_1.isLoongFlowRequest)({ not: 'a request' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid LoongFlowResponse', () => {
            const response = {
                request_id: (0, loongflow_canonical_1.createPESCorrelationId)(),
                state: 'running',
                best_solutions: [],
                iteration: 5,
                checkpoints: [],
                timestamp: (0, loongflow_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, loongflow_canonical_1.isLoongFlowResponse)(response)).toBe(true);
            (0, globals_1.expect)((0, loongflow_canonical_1.isLoongFlowResponse)({ not: 'a response' })).toBe(false);
        });
    });
    (0, globals_1.describe)('Transformation Functions', () => {
        (0, globals_1.it)('should transform raw solution to canonical format', () => {
            const rawSolution = {
                code: 'def foo(): return 42',
                id: 'sol_123',
                plan: 'Use direct return',
                score: 0.95,
                eval: 'Optimal',
                reflection: 'Excellent',
                parent: 'sol_122',
                island: 2,
                iter: 5,
            };
            const canonical = (0, loongflow_canonical_1.transformLoongFlowSolutionToCanonical)(rawSolution);
            (0, globals_1.expect)(canonical.solution).toBe('def foo(): return 42');
            (0, globals_1.expect)(canonical.solution_id).toBe('sol_123');
            (0, globals_1.expect)(canonical.generate_plan).toBe('Use direct return');
            (0, globals_1.expect)(canonical.island_id).toBe(2);
            (0, globals_1.expect)(canonical.iteration).toBe(5);
        });
        (0, globals_1.it)('should transform canonical solution to raw format', () => {
            const canonical = {
                solution: 'test',
                solution_id: 'sol_1',
                generate_plan: 'plan',
                score: 0.9,
                evaluation: 'eval',
                summary: 'summary',
                parent_id: 'parent',
                island_id: 1,
                iteration: 2,
            };
            const raw = (0, loongflow_canonical_1.transformCanonicalToLoongFlowSolution)(canonical);
            (0, globals_1.expect)(raw.solution).toBe(canonical.solution);
            (0, globals_1.expect)(raw.solution_id).toBe(canonical.solution_id);
            (0, globals_1.expect)(raw.island_id).toBe(canonical.island_id);
        });
    });
    (0, globals_1.describe)('Enum Validation', () => {
        (0, globals_1.it)('should accept all valid LoongFlow states', () => {
            const validStates = [
                'idle',
                'planning',
                'executing',
                'summarizing',
                'evolving',
                'completed',
                'failed',
                'cancelled',
            ];
            validStates.forEach((state) => {
                const result = loongflow_canonical_1.LoongFlowState.safeParse(state);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.it)('should accept all valid worker types', () => {
            const validTypes = ['planner', 'executor', 'summarizer', 'evolver'];
            validTypes.forEach((type) => {
                const result = loongflow_canonical_1.LoongFlowWorkerType.safeParse(type);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
    });
});
//# sourceMappingURL=loongflow-schemas.test.js.map