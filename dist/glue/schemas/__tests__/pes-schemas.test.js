"use strict";
/**
 * PES Schemas Validation Tests
 *
 * Comprehensive test suite for PES (Plan-Execute-Summarize) canonical schemas.
 * Tests all schemas for proper validation, transformation, and type safety.
 *
 * Law of Runtime Truth: These tests verify that schemas actually work as expected.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const pes_canonical_1 = require("../pes-canonical");
(0, globals_1.describe)('PES Canonical Schemas', () => {
    (0, globals_1.describe)('Problem Schema', () => {
        (0, globals_1.it)('should validate a valid problem', () => {
            const validProblem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'optimization',
                description: 'Optimize neural network hyperparameters',
                context: { dataset: 'MNIST', max_layers: 10 },
                constraints: ['max_layers <= 10', 'learning_rate in [0.001, 0.1]'],
                success_criteria: ['accuracy > 0.95', 'training_time < 3600s'],
                metadata: { priority: 'high' },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                priority: 8,
                tags: ['ml', 'optimization', 'neural-network'],
            };
            const result = pes_canonical_1.Problem.safeParse(validProblem);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.type).toBe('optimization');
                (0, globals_1.expect)(result.data.priority).toBe(8);
                (0, globals_1.expect)(result.data.tags).toHaveLength(3);
            }
        });
        (0, globals_1.it)('should reject invalid problem type', () => {
            const invalidProblem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'invalid_type',
                description: 'Test problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Problem.safeParse(invalidProblem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject empty description', () => {
            const invalidProblem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'reasoning',
                description: '',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Problem.safeParse(invalidProblem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject invalid priority range', () => {
            const invalidProblem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'generation',
                description: 'Generate code',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                priority: 15, // Invalid: must be 1-10
            };
            const result = pes_canonical_1.Problem.safeParse(invalidProblem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should enforce UUID format for id', () => {
            const invalidProblem = {
                id: 'not-a-uuid',
                type: 'validation',
                description: 'Test validation',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Problem.safeParse(invalidProblem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should enforce UTC ISO-8601 timestamp format', () => {
            const invalidProblem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'classification',
                description: 'Classify data',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: '2024-01-01', // Missing time and timezone
            };
            const result = pes_canonical_1.Problem.safeParse(invalidProblem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('ExecutionStep Schema', () => {
        (0, globals_1.it)('should validate a valid execution step', () => {
            const validStep = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'execute',
                description: 'Execute the generated code',
                worker_type: 'CodeExecutor',
                parameters: { language: 'python', timeout: 30 },
                timeout_ms: 5000,
                retry_config: {
                    max_retries: 3,
                    backoff_ms: 1000,
                    max_backoff_ms: 10000,
                },
                depends_on: [(0, pes_canonical_1.createPESCorrelationId)()],
            };
            const result = pes_canonical_1.ExecutionStep.safeParse(validStep);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.it)('should reject negative timeout', () => {
            const invalidStep = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'plan',
                description: 'Plan execution',
                worker_type: 'Planner',
                parameters: {},
                timeout_ms: -100,
            };
            const result = pes_canonical_1.ExecutionStep.safeParse(invalidStep);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject timeout exceeding 1 hour', () => {
            const invalidStep = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'summarize',
                description: 'Summarize results',
                worker_type: 'Summarizer',
                parameters: {},
                timeout_ms: 3600001, // 1 hour + 1ms
            };
            const result = pes_canonical_1.ExecutionStep.safeParse(invalidStep);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('ExecutionPlan Schema', () => {
        (0, globals_1.it)('should validate a valid execution plan', () => {
            const planId = (0, pes_canonical_1.createPESCorrelationId)();
            const problemId = (0, pes_canonical_1.createPESCorrelationId)();
            const validPlan = {
                id: planId,
                problem_id: problemId,
                steps: [
                    {
                        id: (0, pes_canonical_1.createPESCorrelationId)(),
                        type: 'plan',
                        description: 'Plan approach',
                        worker_type: 'Planner',
                        parameters: {},
                        timeout_ms: 5000,
                    },
                    {
                        id: (0, pes_canonical_1.createPESCorrelationId)(),
                        type: 'execute',
                        description: 'Execute plan',
                        worker_type: 'Executor',
                        parameters: {},
                        timeout_ms: 10000,
                    },
                ],
                resource_requirements: {
                    estimated_duration_ms: 15000,
                    required_workers: ['Planner', 'Executor'],
                    memory_mb: 512,
                    cpu_cores: 2,
                },
                metadata: {},
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                planner_type: 'LLMPlanner',
                confidence_score: 0.85,
            };
            const result = pes_canonical_1.ExecutionPlan.safeParse(validPlan);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.steps).toHaveLength(2);
                (0, globals_1.expect)(result.data.confidence_score).toBe(0.85);
            }
        });
        (0, globals_1.it)('should reject empty steps array', () => {
            const invalidPlan = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                steps: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.ExecutionPlan.safeParse(invalidPlan);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject invalid confidence score', () => {
            const invalidPlan = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                steps: [
                    {
                        id: (0, pes_canonical_1.createPESCorrelationId)(),
                        type: 'plan',
                        description: 'Plan',
                        worker_type: 'Planner',
                        parameters: {},
                        timeout_ms: 5000,
                    },
                ],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                confidence_score: 1.5, // Invalid: must be 0-1
            };
            const result = pes_canonical_1.ExecutionPlan.safeParse(invalidPlan);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('ExecutionResult Schema', () => {
        (0, globals_1.it)('should validate a valid execution result', () => {
            const validResult = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                outputs: { answer: '42', confidence: 0.95 },
                metrics: {
                    duration_ms: 5000,
                    success: true,
                    score: 0.9,
                    error_count: 0,
                    steps_completed: 5,
                    steps_failed: 0,
                },
                artifacts: [
                    {
                        type: 'code',
                        uri: '/artifacts/solution.py',
                        size_bytes: 1024,
                        checksum: 'abc123',
                    },
                ],
                logs: [
                    {
                        timestamp: (0, pes_canonical_1.createPESUTCTimestamp)(),
                        level: 'info',
                        message: 'Execution started',
                    },
                ],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                completed_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                correlation_id: (0, pes_canonical_1.createPESCorrelationId)(),
            };
            const result = pes_canonical_1.ExecutionResult.safeParse(validResult);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.state).toBe('completed');
                (0, globals_1.expect)(result.data.artifacts).toHaveLength(1);
                (0, globals_1.expect)(result.data.logs).toHaveLength(1);
            }
        });
        (0, globals_1.it)('should validate failed execution result with error', () => {
            const failedResult = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                state: 'failed',
                outputs: {},
                metrics: {
                    duration_ms: 2000,
                    success: false,
                    error_count: 1,
                },
                error: {
                    code: 'TIMEOUT',
                    message: 'Execution exceeded timeout',
                    details: { timeout_ms: 5000 },
                    stack_trace: 'Error: Timeout\n    at execute...',
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                completed_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.ExecutionResult.safeParse(failedResult);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.state).toBe('failed');
                (0, globals_1.expect)(result.data.error).toBeDefined();
                (0, globals_1.expect)(result.data.error?.code).toBe('TIMEOUT');
            }
        });
        (0, globals_1.it)('should reject invalid score range', () => {
            const invalidResult = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                outputs: {},
                metrics: {
                    duration_ms: 1000,
                    success: true,
                    score: 1.5, // Invalid: must be 0-1
                    error_count: 0,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.ExecutionResult.safeParse(invalidResult);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject negative duration', () => {
            const invalidResult = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                outputs: {},
                metrics: {
                    duration_ms: -100, // Invalid: must be non-negative
                    success: true,
                    error_count: 0,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.ExecutionResult.safeParse(invalidResult);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Summary Schema', () => {
        (0, globals_1.it)('should validate a valid summary', () => {
            const validSummary = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                result_id: (0, pes_canonical_1.createPESCorrelationId)(),
                evaluation: 'The solution successfully optimized the hyperparameters',
                insights: [
                    'Learning rate of 0.01 performed best',
                    'Batch size had minimal impact',
                    'Deeper networks converged slower',
                ],
                performance_assessment: {
                    success_criteria_met: true,
                    quality_score: 0.92,
                    efficiency_score: 0.88,
                    criteria_scores: {
                        accuracy: 0.95,
                        speed: 0.85,
                    },
                    improvement_suggestions: [
                        'Try different activation functions',
                        'Consider early stopping',
                    ],
                },
                recommendations: [
                    'Use learning rate 0.01 for similar tasks',
                    'Start with smaller batch sizes',
                ],
                metadata: { reviewer: 'AI_System' },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                summarizer_type: 'QualitySummarizer',
                confidence_score: 0.9,
            };
            const result = pes_canonical_1.Summary.safeParse(validSummary);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.insights).toHaveLength(3);
                (0, globals_1.expect)(result.data.performance_assessment.quality_score).toBe(0.92);
                (0, globals_1.expect)(result.data.recommendations).toHaveLength(2);
            }
        });
        (0, globals_1.it)('should reject empty evaluation', () => {
            const invalidSummary = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                result_id: (0, pes_canonical_1.createPESCorrelationId)(),
                evaluation: '', // Invalid: cannot be empty
                insights: [],
                performance_assessment: {
                    success_criteria_met: true,
                    quality_score: 0.8,
                    efficiency_score: 0.7,
                },
                recommendations: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Summary.safeParse(invalidSummary);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should reject performance assessment scores out of range', () => {
            const invalidSummary = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                result_id: (0, pes_canonical_1.createPESCorrelationId)(),
                evaluation: 'Test evaluation',
                insights: [],
                performance_assessment: {
                    success_criteria_met: true,
                    quality_score: 1.2, // Invalid: must be 0-1
                    efficiency_score: 0.8,
                },
                recommendations: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Summary.safeParse(invalidSummary);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('Validation Functions', () => {
        (0, globals_1.it)('should validate problem using validateProblem', () => {
            const problem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'optimization',
                description: 'Test problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, pes_canonical_1.validateProblem)(problem);
            (0, globals_1.expect)(validation.success).toBe(true);
            (0, globals_1.expect)(validation.data).toBeDefined();
            (0, globals_1.expect)(validation.errors).toBeUndefined();
        });
        (0, globals_1.it)('should return errors for invalid problem', () => {
            const invalidProblem = {
                id: 'invalid-uuid',
                type: 'invalid',
                description: '',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: 'invalid-date',
            };
            const validation = (0, pes_canonical_1.validateProblem)(invalidProblem);
            (0, globals_1.expect)(validation.success).toBe(false);
            (0, globals_1.expect)(validation.errors).toBeDefined();
            (0, globals_1.expect)(validation.errors.length).toBeGreaterThan(0);
        });
        (0, globals_1.it)('should validate execution plan using validateExecutionPlan', () => {
            const plan = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                steps: [
                    {
                        id: (0, pes_canonical_1.createPESCorrelationId)(),
                        type: 'plan',
                        description: 'Plan',
                        worker_type: 'Planner',
                        parameters: {},
                        timeout_ms: 5000,
                    },
                ],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, pes_canonical_1.validateExecutionPlan)(plan);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate execution result using validateExecutionResult', () => {
            const result = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                outputs: {},
                metrics: {
                    duration_ms: 1000,
                    success: true,
                    error_count: 0,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, pes_canonical_1.validateExecutionResult)(result);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
        (0, globals_1.it)('should validate summary using validateSummary', () => {
            const summary = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                result_id: (0, pes_canonical_1.createPESCorrelationId)(),
                evaluation: 'Good solution',
                insights: ['Insight 1'],
                performance_assessment: {
                    success_criteria_met: true,
                    quality_score: 0.9,
                    efficiency_score: 0.8,
                },
                recommendations: ['Recommendation 1'],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const validation = (0, pes_canonical_1.validateSummary)(summary);
            (0, globals_1.expect)(validation.success).toBe(true);
        });
    });
    (0, globals_1.describe)('Utility Functions', () => {
        (0, globals_1.it)('should create valid UTC timestamp', () => {
            const timestamp = (0, pes_canonical_1.createPESUTCTimestamp)();
            (0, globals_1.expect)(timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
        });
        (0, globals_1.it)('should create valid correlation ID (UUID)', () => {
            const correlationId = (0, pes_canonical_1.createPESCorrelationId)();
            (0, globals_1.expect)(correlationId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
        });
        (0, globals_1.it)('should check if data is a valid Problem', () => {
            const problem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'reasoning',
                description: 'Test',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, pes_canonical_1.isProblem)(problem)).toBe(true);
            (0, globals_1.expect)((0, pes_canonical_1.isProblem)({ not: 'a problem' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid ExecutionResult', () => {
            const result = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                problem_id: (0, pes_canonical_1.createPESCorrelationId)(),
                plan_id: (0, pes_canonical_1.createPESCorrelationId)(),
                state: 'completed',
                outputs: {},
                metrics: {
                    duration_ms: 100,
                    success: true,
                    error_count: 0,
                },
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, pes_canonical_1.isExecutionResult)(result)).toBe(true);
            (0, globals_1.expect)((0, pes_canonical_1.isExecutionResult)({ not: 'a result' })).toBe(false);
        });
        (0, globals_1.it)('should check if data is a valid Summary', () => {
            const summary = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                result_id: (0, pes_canonical_1.createPESCorrelationId)(),
                evaluation: 'Test',
                insights: [],
                performance_assessment: {
                    success_criteria_met: true,
                    quality_score: 0.8,
                    efficiency_score: 0.7,
                },
                recommendations: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            (0, globals_1.expect)((0, pes_canonical_1.isSummary)(summary)).toBe(true);
            (0, globals_1.expect)((0, pes_canonical_1.isSummary)({ not: 'a summary' })).toBe(false);
        });
    });
    (0, globals_1.describe)('Transformation Functions', () => {
        (0, globals_1.it)('should transform raw problem to canonical format', () => {
            const rawProblem = {
                type: 'generation',
                problem: 'Generate Python code for sorting',
                objectives: ['Correctness', 'Efficiency'],
            };
            const canonical = (0, pes_canonical_1.transformProblemToCanonical)(rawProblem);
            (0, globals_1.expect)(canonical.type).toBe('generation');
            (0, globals_1.expect)(canonical.description).toBe('Generate Python code for sorting');
            (0, globals_1.expect)(canonical.success_criteria).toEqual(['Correctness', 'Efficiency']);
            (0, globals_1.expect)(canonical.id).toBeDefined();
            (0, globals_1.expect)(canonical.created_at).toBeDefined();
        });
        (0, globals_1.it)('should transform canonical problem to external format', () => {
            const canonical = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'optimization',
                description: 'Optimize function',
                context: { param: 'value' },
                constraints: ['constraint1'],
                success_criteria: ['criteria1'],
                metadata: {},
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const external = (0, pes_canonical_1.transformCanonicalToProblem)(canonical);
            (0, globals_1.expect)(external.id).toBe(canonical.id);
            (0, globals_1.expect)(external.type).toBe(canonical.type);
            (0, globals_1.expect)(external.description).toBe(canonical.description);
        });
        (0, globals_1.it)('should transform raw execution result to canonical format', () => {
            const rawResult = {
                task_id: (0, pes_canonical_1.createPESCorrelationId)(),
                workflow_id: (0, pes_canonical_1.createPESCorrelationId)(),
                status: 'completed',
                result: { answer: 42 },
                duration_ms: 5000,
                success: true,
            };
            const canonical = (0, pes_canonical_1.transformExecutionResultToCanonical)(rawResult);
            (0, globals_1.expect)(canonical.state).toBe('completed');
            (0, globals_1.expect)(canonical.outputs).toEqual({ answer: 42 });
            (0, globals_1.expect)(canonical.metrics.duration_ms).toBe(5000);
            (0, globals_1.expect)(canonical.metrics.success).toBe(true);
        });
    });
    (0, globals_1.describe)('Type Guards and Enums', () => {
        (0, globals_1.it)('should accept all valid problem types', () => {
            const validTypes = [
                'optimization',
                'reasoning',
                'generation',
                'validation',
                'classification',
                'prediction',
                'synthesis',
            ];
            validTypes.forEach((type) => {
                const result = pes_canonical_1.ProblemType.safeParse(type);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.it)('should accept all valid execution step types', () => {
            const validTypes = [
                'plan',
                'execute',
                'summarize',
                'validate',
                'transform',
                'aggregate',
            ];
            validTypes.forEach((type) => {
                const result = pes_canonical_1.ExecutionStepType.safeParse(type);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.it)('should accept all valid execution states', () => {
            const validStates = [
                'pending',
                'running',
                'completed',
                'failed',
                'cancelled',
                'timeout',
                'paused',
            ];
            validStates.forEach((state) => {
                const result = pes_canonical_1.ExecutionState.safeParse(state);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
    });
    (0, globals_1.describe)('Edge Cases and Error Handling', () => {
        (0, globals_1.it)('should handle missing optional fields gracefully', () => {
            const minimalProblem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'reasoning',
                description: 'Minimal problem',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Problem.safeParse(minimalProblem);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.it)('should handle empty arrays for optional array fields', () => {
            const problem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'generation',
                description: 'Test',
                context: {},
                constraints: [],
                success_criteria: [],
                tags: [], // Empty array for optional field
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Problem.safeParse(problem);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.it)('should reject null for required fields', () => {
            const invalidProblem = {
                id: null, // Invalid: must be string
                type: 'reasoning',
                description: 'Test',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
            };
            const result = pes_canonical_1.Problem.safeParse(invalidProblem);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should handle large metadata objects', () => {
            const problem = {
                id: (0, pes_canonical_1.createPESCorrelationId)(),
                type: 'validation',
                description: 'Test',
                context: {},
                constraints: [],
                success_criteria: [],
                created_at: (0, pes_canonical_1.createPESUTCTimestamp)(),
                metadata: {
                    key1: 'value1',
                    key2: 123,
                    key3: { nested: 'object' },
                    key4: ['array', 'of', 'values'],
                },
            };
            const result = pes_canonical_1.Problem.safeParse(problem);
            (0, globals_1.expect)(result.success).toBe(true);
        });
    });
});
//# sourceMappingURL=pes-schemas.test.js.map