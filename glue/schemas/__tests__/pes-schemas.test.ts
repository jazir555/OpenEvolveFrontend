/**
 * PES Schemas Validation Tests
 *
 * Comprehensive test suite for PES (Plan-Execute-Summarize) canonical schemas.
 * Tests all schemas for proper validation, transformation, and type safety.
 *
 * Law of Runtime Truth: These tests verify that schemas actually work as expected.
 */

import { describe, it, expect } from '@jest/globals';
import {
  // PES Schemas
  Problem,
  ProblemType,
  ExecutionStep,
  ExecutionStepType,
  ExecutionPlan,
  ExecutionState,
  ExecutionResult,
  ExecutionMetrics,
  LogEntry,
  Artifact,
  Summary,
  PerformanceAssessment,
  // Validation functions
  validateProblem,
  validateExecutionPlan,
  validateExecutionResult,
  validateSummary,
  // Utility functions
  createPESUTCTimestamp,
  createPESCorrelationId,
  isProblem,
  isExecutionResult,
  isSummary,
  // Transformation functions
  transformProblemToCanonical,
  transformCanonicalToProblem,
  transformExecutionResultToCanonical,
  transformCanonicalToSummary,
} from '../pes-canonical';

describe('PES Canonical Schemas', () => {
  describe('Problem Schema', () => {
    it('should validate a valid problem', () => {
      const validProblem = {
        id: createPESCorrelationId(),
        type: 'optimization' as const,
        description: 'Optimize neural network hyperparameters',
        context: { dataset: 'MNIST', max_layers: 10 },
        constraints: ['max_layers <= 10', 'learning_rate in [0.001, 0.1]'],
        success_criteria: ['accuracy > 0.95', 'training_time < 3600s'],
        metadata: { priority: 'high' },
        created_at: createPESUTCTimestamp(),
        priority: 8,
        tags: ['ml', 'optimization', 'neural-network'],
      };

      const result = Problem.safeParse(validProblem);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('optimization');
        expect(result.data.priority).toBe(8);
        expect(result.data.tags).toHaveLength(3);
      }
    });

    it('should reject invalid problem type', () => {
      const invalidProblem = {
        id: createPESCorrelationId(),
        type: 'invalid_type',
        description: 'Test problem',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Problem.safeParse(invalidProblem);
      expect(result.success).toBe(false);
    });

    it('should reject empty description', () => {
      const invalidProblem = {
        id: createPESCorrelationId(),
        type: 'reasoning' as const,
        description: '',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Problem.safeParse(invalidProblem);
      expect(result.success).toBe(false);
    });

    it('should reject invalid priority range', () => {
      const invalidProblem = {
        id: createPESCorrelationId(),
        type: 'generation' as const,
        description: 'Generate code',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
        priority: 15, // Invalid: must be 1-10
      };

      const result = Problem.safeParse(invalidProblem);
      expect(result.success).toBe(false);
    });

    it('should enforce UUID format for id', () => {
      const invalidProblem = {
        id: 'not-a-uuid',
        type: 'validation' as const,
        description: 'Test validation',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Problem.safeParse(invalidProblem);
      expect(result.success).toBe(false);
    });

    it('should enforce UTC ISO-8601 timestamp format', () => {
      const invalidProblem = {
        id: createPESCorrelationId(),
        type: 'classification' as const,
        description: 'Classify data',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: '2024-01-01', // Missing time and timezone
      };

      const result = Problem.safeParse(invalidProblem);
      expect(result.success).toBe(false);
    });
  });

  describe('ExecutionStep Schema', () => {
    it('should validate a valid execution step', () => {
      const validStep = {
        id: createPESCorrelationId(),
        type: 'execute' as const,
        description: 'Execute the generated code',
        worker_type: 'CodeExecutor',
        parameters: { language: 'python', timeout: 30 },
        timeout_ms: 5000,
        retry_config: {
          max_retries: 3,
          backoff_ms: 1000,
          max_backoff_ms: 10000,
        },
        depends_on: [createPESCorrelationId()],
      };

      const result = ExecutionStep.safeParse(validStep);
      expect(result.success).toBe(true);
    });

    it('should reject negative timeout', () => {
      const invalidStep = {
        id: createPESCorrelationId(),
        type: 'plan' as const,
        description: 'Plan execution',
        worker_type: 'Planner',
        parameters: {},
        timeout_ms: -100,
      };

      const result = ExecutionStep.safeParse(invalidStep);
      expect(result.success).toBe(false);
    });

    it('should reject timeout exceeding 1 hour', () => {
      const invalidStep = {
        id: createPESCorrelationId(),
        type: 'summarize' as const,
        description: 'Summarize results',
        worker_type: 'Summarizer',
        parameters: {},
        timeout_ms: 3600001, // 1 hour + 1ms
      };

      const result = ExecutionStep.safeParse(invalidStep);
      expect(result.success).toBe(false);
    });
  });

  describe('ExecutionPlan Schema', () => {
    it('should validate a valid execution plan', () => {
      const planId = createPESCorrelationId();
      const problemId = createPESCorrelationId();

      const validPlan = {
        id: planId,
        problem_id: problemId,
        steps: [
          {
            id: createPESCorrelationId(),
            type: 'plan' as const,
            description: 'Plan approach',
            worker_type: 'Planner',
            parameters: {},
            timeout_ms: 5000,
          },
          {
            id: createPESCorrelationId(),
            type: 'execute' as const,
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
        created_at: createPESUTCTimestamp(),
        planner_type: 'LLMPlanner',
        confidence_score: 0.85,
      };

      const result = ExecutionPlan.safeParse(validPlan);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.steps).toHaveLength(2);
        expect(result.data.confidence_score).toBe(0.85);
      }
    });

    it('should reject empty steps array', () => {
      const invalidPlan = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        steps: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = ExecutionPlan.safeParse(invalidPlan);
      expect(result.success).toBe(false);
    });

    it('should reject invalid confidence score', () => {
      const invalidPlan = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        steps: [
          {
            id: createPESCorrelationId(),
            type: 'plan' as const,
            description: 'Plan',
            worker_type: 'Planner',
            parameters: {},
            timeout_ms: 5000,
          },
        ],
        created_at: createPESUTCTimestamp(),
        confidence_score: 1.5, // Invalid: must be 0-1
      };

      const result = ExecutionPlan.safeParse(invalidPlan);
      expect(result.success).toBe(false);
    });
  });

  describe('ExecutionResult Schema', () => {
    it('should validate a valid execution result', () => {
      const validResult = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        plan_id: createPESCorrelationId(),
        state: 'completed' as const,
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
            timestamp: createPESUTCTimestamp(),
            level: 'info' as const,
            message: 'Execution started',
          },
        ],
        created_at: createPESUTCTimestamp(),
        completed_at: createPESUTCTimestamp(),
        correlation_id: createPESCorrelationId(),
      };

      const result = ExecutionResult.safeParse(validResult);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.state).toBe('completed');
        expect(result.data.artifacts).toHaveLength(1);
        expect(result.data.logs).toHaveLength(1);
      }
    });

    it('should validate failed execution result with error', () => {
      const failedResult = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        plan_id: createPESCorrelationId(),
        state: 'failed' as const,
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
        created_at: createPESUTCTimestamp(),
        completed_at: createPESUTCTimestamp(),
      };

      const result = ExecutionResult.safeParse(failedResult);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.state).toBe('failed');
        expect(result.data.error).toBeDefined();
        expect(result.data.error?.code).toBe('TIMEOUT');
      }
    });

    it('should reject invalid score range', () => {
      const invalidResult = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        plan_id: createPESCorrelationId(),
        state: 'completed' as const,
        outputs: {},
        metrics: {
          duration_ms: 1000,
          success: true,
          score: 1.5, // Invalid: must be 0-1
          error_count: 0,
        },
        created_at: createPESUTCTimestamp(),
      };

      const result = ExecutionResult.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });

    it('should reject negative duration', () => {
      const invalidResult = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        plan_id: createPESCorrelationId(),
        state: 'completed' as const,
        outputs: {},
        metrics: {
          duration_ms: -100, // Invalid: must be non-negative
          success: true,
          error_count: 0,
        },
        created_at: createPESUTCTimestamp(),
      };

      const result = ExecutionResult.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });
  });

  describe('Summary Schema', () => {
    it('should validate a valid summary', () => {
      const validSummary = {
        id: createPESCorrelationId(),
        result_id: createPESCorrelationId(),
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
        created_at: createPESUTCTimestamp(),
        summarizer_type: 'QualitySummarizer',
        confidence_score: 0.9,
      };

      const result = Summary.safeParse(validSummary);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.insights).toHaveLength(3);
        expect(result.data.performance_assessment.quality_score).toBe(0.92);
        expect(result.data.recommendations).toHaveLength(2);
      }
    });

    it('should reject empty evaluation', () => {
      const invalidSummary = {
        id: createPESCorrelationId(),
        result_id: createPESCorrelationId(),
        evaluation: '', // Invalid: cannot be empty
        insights: [],
        performance_assessment: {
          success_criteria_met: true,
          quality_score: 0.8,
          efficiency_score: 0.7,
        },
        recommendations: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Summary.safeParse(invalidSummary);
      expect(result.success).toBe(false);
    });

    it('should reject performance assessment scores out of range', () => {
      const invalidSummary = {
        id: createPESCorrelationId(),
        result_id: createPESCorrelationId(),
        evaluation: 'Test evaluation',
        insights: [],
        performance_assessment: {
          success_criteria_met: true,
          quality_score: 1.2, // Invalid: must be 0-1
          efficiency_score: 0.8,
        },
        recommendations: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Summary.safeParse(invalidSummary);
      expect(result.success).toBe(false);
    });
  });

  describe('Validation Functions', () => {
    it('should validate problem using validateProblem', () => {
      const problem = {
        id: createPESCorrelationId(),
        type: 'optimization' as const,
        description: 'Test problem',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      const validation = validateProblem(problem);
      expect(validation.success).toBe(true);
      expect(validation.data).toBeDefined();
      expect(validation.errors).toBeUndefined();
    });

    it('should return errors for invalid problem', () => {
      const invalidProblem = {
        id: 'invalid-uuid',
        type: 'invalid',
        description: '',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: 'invalid-date',
      };

      const validation = validateProblem(invalidProblem);
      expect(validation.success).toBe(false);
      expect(validation.errors).toBeDefined();
      expect(validation.errors!.length).toBeGreaterThan(0);
    });

    it('should validate execution plan using validateExecutionPlan', () => {
      const plan = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        steps: [
          {
            id: createPESCorrelationId(),
            type: 'plan' as const,
            description: 'Plan',
            worker_type: 'Planner',
            parameters: {},
            timeout_ms: 5000,
          },
        ],
        created_at: createPESUTCTimestamp(),
      };

      const validation = validateExecutionPlan(plan);
      expect(validation.success).toBe(true);
    });

    it('should validate execution result using validateExecutionResult', () => {
      const result = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        plan_id: createPESCorrelationId(),
        state: 'completed' as const,
        outputs: {},
        metrics: {
          duration_ms: 1000,
          success: true,
          error_count: 0,
        },
        created_at: createPESUTCTimestamp(),
      };

      const validation = validateExecutionResult(result);
      expect(validation.success).toBe(true);
    });

    it('should validate summary using validateSummary', () => {
      const summary = {
        id: createPESCorrelationId(),
        result_id: createPESCorrelationId(),
        evaluation: 'Good solution',
        insights: ['Insight 1'],
        performance_assessment: {
          success_criteria_met: true,
          quality_score: 0.9,
          efficiency_score: 0.8,
        },
        recommendations: ['Recommendation 1'],
        created_at: createPESUTCTimestamp(),
      };

      const validation = validateSummary(summary);
      expect(validation.success).toBe(true);
    });
  });

  describe('Utility Functions', () => {
    it('should create valid UTC timestamp', () => {
      const timestamp = createPESUTCTimestamp();
      expect(timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
    });

    it('should create valid correlation ID (UUID)', () => {
      const correlationId = createPESCorrelationId();
      expect(correlationId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    });

    it('should check if data is a valid Problem', () => {
      const problem = {
        id: createPESCorrelationId(),
        type: 'reasoning' as const,
        description: 'Test',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      expect(isProblem(problem)).toBe(true);
      expect(isProblem({ not: 'a problem' })).toBe(false);
    });

    it('should check if data is a valid ExecutionResult', () => {
      const result = {
        id: createPESCorrelationId(),
        problem_id: createPESCorrelationId(),
        plan_id: createPESCorrelationId(),
        state: 'completed' as const,
        outputs: {},
        metrics: {
          duration_ms: 100,
          success: true,
          error_count: 0,
        },
        created_at: createPESUTCTimestamp(),
      };

      expect(isExecutionResult(result)).toBe(true);
      expect(isExecutionResult({ not: 'a result' })).toBe(false);
    });

    it('should check if data is a valid Summary', () => {
      const summary = {
        id: createPESCorrelationId(),
        result_id: createPESCorrelationId(),
        evaluation: 'Test',
        insights: [],
        performance_assessment: {
          success_criteria_met: true,
          quality_score: 0.8,
          efficiency_score: 0.7,
        },
        recommendations: [],
        created_at: createPESUTCTimestamp(),
      };

      expect(isSummary(summary)).toBe(true);
      expect(isSummary({ not: 'a summary' })).toBe(false);
    });
  });

  describe('Transformation Functions', () => {
    it('should transform raw problem to canonical format', () => {
      const rawProblem = {
        type: 'generation',
        problem: 'Generate Python code for sorting',
        objectives: ['Correctness', 'Efficiency'],
      };

      const canonical = transformProblemToCanonical(rawProblem);

      expect(canonical.type).toBe('generation');
      expect(canonical.description).toBe('Generate Python code for sorting');
      expect(canonical.success_criteria).toEqual(['Correctness', 'Efficiency']);
      expect(canonical.id).toBeDefined();
      expect(canonical.created_at).toBeDefined();
    });

    it('should transform canonical problem to external format', () => {
      const canonical: ProblemType = {
        id: createPESCorrelationId(),
        type: 'optimization',
        description: 'Optimize function',
        context: { param: 'value' },
        constraints: ['constraint1'],
        success_criteria: ['criteria1'],
        metadata: {},
        created_at: createPESUTCTimestamp(),
      };

      const external = transformCanonicalToProblem(canonical);

      expect(external.id).toBe(canonical.id);
      expect(external.type).toBe(canonical.type);
      expect(external.description).toBe(canonical.description);
    });

    it('should transform raw execution result to canonical format', () => {
      const rawResult = {
        task_id: createPESCorrelationId(),
        workflow_id: createPESCorrelationId(),
        status: 'completed',
        result: { answer: 42 },
        duration_ms: 5000,
        success: true,
      };

      const canonical = transformExecutionResultToCanonical(rawResult);

      expect(canonical.state).toBe('completed');
      expect(canonical.outputs).toEqual({ answer: 42 });
      expect(canonical.metrics.duration_ms).toBe(5000);
      expect(canonical.metrics.success).toBe(true);
    });
  });

  describe('Type Guards and Enums', () => {
    it('should accept all valid problem types', () => {
      const validTypes = [
        'optimization',
        'reasoning',
        'generation',
        'validation',
        'classification',
        'prediction',
        'synthesis',
      ] as const;

      validTypes.forEach((type) => {
        const result = ProblemType.safeParse(type);
        expect(result.success).toBe(true);
      });
    });

    it('should accept all valid execution step types', () => {
      const validTypes = [
        'plan',
        'execute',
        'summarize',
        'validate',
        'transform',
        'aggregate',
      ] as const;

      validTypes.forEach((type) => {
        const result = ExecutionStepType.safeParse(type);
        expect(result.success).toBe(true);
      });
    });

    it('should accept all valid execution states', () => {
      const validStates = [
        'pending',
        'running',
        'completed',
        'failed',
        'cancelled',
        'timeout',
        'paused',
      ] as const;

      validStates.forEach((state) => {
        const result = ExecutionState.safeParse(state);
        expect(result.success).toBe(true);
      });
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle missing optional fields gracefully', () => {
      const minimalProblem = {
        id: createPESCorrelationId(),
        type: 'reasoning' as const,
        description: 'Minimal problem',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Problem.safeParse(minimalProblem);
      expect(result.success).toBe(true);
    });

    it('should handle empty arrays for optional array fields', () => {
      const problem = {
        id: createPESCorrelationId(),
        type: 'generation' as const,
        description: 'Test',
        context: {},
        constraints: [],
        success_criteria: [],
        tags: [], // Empty array for optional field
        created_at: createPESUTCTimestamp(),
      };

      const result = Problem.safeParse(problem);
      expect(result.success).toBe(true);
    });

    it('should reject null for required fields', () => {
      const invalidProblem = {
        id: null, // Invalid: must be string
        type: 'reasoning' as const,
        description: 'Test',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
      };

      const result = Problem.safeParse(invalidProblem);
      expect(result.success).toBe(false);
    });

    it('should handle large metadata objects', () => {
      const problem = {
        id: createPESCorrelationId(),
        type: 'validation' as const,
        description: 'Test',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: createPESUTCTimestamp(),
        metadata: {
          key1: 'value1',
          key2: 123,
          key3: { nested: 'object' },
          key4: ['array', 'of', 'values'],
        },
      };

      const result = Problem.safeParse(problem);
      expect(result.success).toBe(true);
    });
  });
});
