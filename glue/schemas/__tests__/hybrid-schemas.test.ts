/**
 * Hybrid PES-Evolution Schemas Validation Tests
 *
 * Comprehensive test suite for Hybrid PES-Evolution canonical schemas.
 * Tests all schemas for proper validation, transformation, and type safety.
 *
 * Law of Runtime Truth: These tests verify that schemas actually work as expected.
 */

import { describe, it, expect } from '@jest/globals';
import {
  // Hybrid Schemas
  HybridTask,
  HybridTaskType,
  IntegrationStrategy,
  AdaptiveTriggerCondition,
  AdaptiveAction,
  KnowledgeSourceType,
  KnowledgeType,
  EvolutionConfig,
  EvolutionaryKnowledge,
  PopulationIndividual,
  EvolutionResult,
  IntegrationMetrics,
  HybridExecutionResult,
  AdaptiveTrigger,
  KnowledgeTransfer,
  // Validation functions
  validateHybridTask,
  validateEvolutionaryKnowledge,
  validateHybridExecutionResult,
  validateAdaptiveTrigger,
  // Utility functions
  isHybridTask,
  isHybridExecutionResult,
  isEvolutionaryKnowledge,
  isAdaptiveTrigger,
  // Transformation functions
  transformLoongFlowSolutionToKnowledge,
  transformHybridResultToSummary,
} from '../hybrid-pes-evolution-canonical';
import {
  Problem,
  ExecutionResult,
  Summary,
  createPESUTCTimestamp,
  createPESCorrelationId,
} from '../pes-canonical';
import {
  LoongFlowConfig,
  LoongFlowSolution,
  LoongFlowState,
} from '../loongflow-canonical';

describe('Hybrid PES-Evolution Canonical Schemas', () => {
  describe('EvolutionConfig Schema', () => {
    it('should validate a valid evolution config', () => {
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

      const result = EvolutionConfig.safeParse(validConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.generations).toBe(100);
        expect(result.data.mutation_rate).toBe(0.1);
        expect(result.data.crossover_rate).toBe(0.8);
      }
    });

    it('should reject mutation_rate out of range', () => {
      const invalidConfig = {
        generations: 100,
        population_size: 50,
        mutation_rate: 1.5, // Invalid: must be 0-1
        crossover_rate: 0.8,
      };

      const result = EvolutionConfig.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should reject negative generations', () => {
      const invalidConfig = {
        generations: -10, // Invalid: must be positive
        population_size: 50,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
      };

      const result = EvolutionConfig.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should reject crossover_rate out of range', () => {
      const invalidConfig = {
        generations: 100,
        population_size: 50,
        mutation_rate: 0.1,
        crossover_rate: -0.1, // Invalid: must be 0-1
      };

      const result = EvolutionConfig.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('HybridTask Schema', () => {
    const createValidProblem = (): Problem => ({
      id: createPESCorrelationId(),
      type: 'optimization',
      description: 'Test optimization problem',
      context: {},
      constraints: [],
      success_criteria: [],
      created_at: createPESUTCTimestamp(),
    });

    const createValidLoongFlowConfig = (): LoongFlowConfig => ({
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

    it('should validate a valid hybrid task', () => {
      const validTask = {
        id: createPESCorrelationId(),
        type: 'adaptive_execute' as const,
        problem: createValidProblem(),
        pes_config: createValidLoongFlowConfig(),
        evolution_config: {
          generations: 50,
          population_size: 100,
          mutation_rate: 0.1,
          crossover_rate: 0.8,
        },
        integration_strategy: 'adaptive' as const,
        adaptive_triggers: [
          {
            condition: 'stagnation' as const,
            threshold: 5,
            action: 'switch_to_evolution' as const,
          },
          {
            condition: 'low_confidence' as const,
            threshold: 0.6,
            action: 'increase_iterations' as const,
          },
        ],
        created_at: createPESUTCTimestamp(),
        correlation_id: createPESCorrelationId(),
        priority: 7,
        timeout_ms: 300000,
      };

      const result = HybridTask.safeParse(validTask);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('adaptive_execute');
        expect(result.data.integration_strategy).toBe('adaptive');
        expect(result.data.adaptive_triggers).toHaveLength(2);
      }
    });

    it('should validate task without optional pes_config', () => {
      const validTask = {
        id: createPESCorrelationId(),
        type: 'evolve_solve' as const,
        problem: createValidProblem(),
        evolution_config: {
          generations: 100,
          population_size: 50,
          mutation_rate: 0.2,
          crossover_rate: 0.7,
        },
        integration_strategy: 'sequential' as const,
        created_at: createPESUTCTimestamp(),
      };

      const result = HybridTask.safeParse(validTask);
      expect(result.success).toBe(true);
    });

    it('should reject invalid priority range', () => {
      const invalidTask = {
        id: createPESCorrelationId(),
        type: 'pes_optimize' as const,
        problem: createValidProblem(),
        pes_config: createValidLoongFlowConfig(),
        evolution_config: {
          generations: 50,
          population_size: 100,
          mutation_rate: 0.1,
          crossover_rate: 0.8,
        },
        integration_strategy: 'parallel' as const,
        priority: 15, // Invalid: must be 1-10
        created_at: createPESUTCTimestamp(),
      };

      const result = HybridTask.safeParse(invalidTask);
      expect(result.success).toBe(false);
    });
  });

  describe('EvolutionaryKnowledge Schema', () => {
    it('should validate a valid knowledge', () => {
      const validKnowledge = {
        id: createPESCorrelationId(),
        source_type: 'loongflow_solution' as const,
        problem_id: createPESCorrelationId(),
        knowledge_type: 'solution_pattern' as const,
        content: {
          pattern: 'Use gradient descent for convex optimization',
          success_rate: 0.9,
          avg_score: 0.85,
          usage_count: 10,
          context: { domain: 'optimization' },
        },
        source_id: 'solution_123',
        valid_until: createPESUTCTimestamp(),
        extracted_at: createPESUTCTimestamp(),
        metadata: { verified: true },
      };

      const result = EvolutionaryKnowledge.safeParse(validKnowledge);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.source_type).toBe('loongflow_solution');
        expect(result.data.knowledge_type).toBe('solution_pattern');
        expect(result.data.content.success_rate).toBe(0.9);
      }
    });

    it('should reject success_rate out of range', () => {
      const invalidKnowledge = {
        id: createPESCorrelationId(),
        source_type: 'openevolve_optimization' as const,
        problem_id: createPESCorrelationId(),
        knowledge_type: 'planning_strategy' as const,
        content: {
          pattern: 'Test pattern',
          success_rate: 1.5, // Invalid: must be 0-1
          avg_score: 0.8,
          usage_count: 5,
        },
        extracted_at: createPESUTCTimestamp(),
      };

      const result = EvolutionaryKnowledge.safeParse(invalidKnowledge);
      expect(result.success).toBe(false);
    });

    it('should reject negative usage_count', () => {
      const invalidKnowledge = {
        id: createPESCorrelationId(),
        source_type: 'hybrid_result' as const,
        problem_id: createPESCorrelationId(),
        knowledge_type: 'execution_approach' as const,
        content: {
          pattern: 'Test pattern',
          success_rate: 0.8,
          avg_score: 0.75,
          usage_count: -1, // Invalid: must be non-negative
        },
        extracted_at: createPESUTCTimestamp(),
      };

      const result = EvolutionaryKnowledge.safeParse(invalidKnowledge);
      expect(result.success).toBe(false);
    });
  });

  describe('PopulationIndividual Schema', () => {
    it('should validate a valid individual', () => {
      const validIndividual = {
        id: 'individual_123',
        genes: [1, 0, 1, 1, 0],
        fitness: 0.95,
        metadata: { generation: 5 },
        parent_ids: ['parent_1', 'parent_2'],
        generation: 10,
      };

      const result = PopulationIndividual.safeParse(validIndividual);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.fitness).toBe(0.95);
        expect(result.data.genes).toEqual([1, 0, 1, 1, 0]);
        expect(result.data.parent_ids).toHaveLength(2);
      }
    });

    it('should reject negative fitness', () => {
      const invalidIndividual = {
        id: 'ind_1',
        genes: [1, 0],
        fitness: -0.5, // Invalid: must be >= 0
      };

      const result = PopulationIndividual.safeParse(invalidIndividual);
      expect(result.success).toBe(false);
    });

    it('should reject negative generation', () => {
      const invalidIndividual = {
        id: 'ind_1',
        genes: [1],
        fitness: 0.8,
        generation: -5, // Invalid: must be non-negative
      };

      const result = PopulationIndividual.safeParse(invalidIndividual);
      expect(result.success).toBe(false);
    });
  });

  describe('EvolutionResult Schema', () => {
    it('should validate a valid evolution result', () => {
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

      const result = EvolutionResult.safeParse(validResult);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.best_fitness).toBe(0.98);
        expect(result.data.population).toHaveLength(2);
        expect(result.data.converged).toBe(true);
      }
    });

    it('should reject negative generations', () => {
      const invalidResult = {
        best_fitness: 0.9,
        generations: -10, // Invalid: must be non-negative
        population: [],
      };

      const result = EvolutionResult.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });

    it('should reject diversity_score out of range', () => {
      const invalidResult = {
        best_fitness: 0.9,
        generations: 50,
        population: [],
        diversity_score: 1.5, // Invalid: must be 0-1
      };

      const result = EvolutionResult.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });
  });

  describe('IntegrationMetrics Schema', () => {
    it('should validate valid integration metrics', () => {
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

      const result = IntegrationMetrics.safeParse(validMetrics);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.synergy_score).toBe(0.85);
        expect(result.data.paradigm_switches).toBe(3);
      }
    });

    it('should reject negative values', () => {
      const invalidMetrics = {
        pes_iterations: -5, // Invalid: must be non-negative
        evolution_generations: 50,
        total_duration_ms: 100000,
        synergy_score: 0.8,
      };

      const result = IntegrationMetrics.safeParse(invalidMetrics);
      expect(result.success).toBe(false);
    });

    it('should reject synergy_score out of range', () => {
      const invalidMetrics = {
        pes_iterations: 5,
        evolution_generations: 50,
        total_duration_ms: 100000,
        synergy_score: 1.2, // Invalid: must be 0-1
      };

      const result = IntegrationMetrics.safeParse(invalidMetrics);
      expect(result.success).toBe(false);
    });
  });

  describe('HybridExecutionResult Schema', () => {
    it('should validate a valid hybrid execution result', () => {
      const validResult = {
        id: createPESCorrelationId(),
        task_id: createPESCorrelationId(),
        pes_result: {
          id: createPESCorrelationId(),
          problem_id: createPESCorrelationId(),
          plan_id: createPESCorrelationId(),
          state: 'completed' as const,
          outputs: {},
          metrics: {
            duration_ms: 5000,
            success: true,
            error_count: 0,
          },
          created_at: createPESUTCTimestamp(),
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
            id: createPESCorrelationId(),
            source_type: 'loongflow_solution' as const,
            problem_id: createPESCorrelationId(),
            knowledge_type: 'solution_pattern' as const,
            content: {
              pattern: 'Test pattern',
              success_rate: 0.9,
              avg_score: 0.85,
              usage_count: 5,
            },
            extracted_at: createPESUTCTimestamp(),
          },
        ],
        created_at: createPESUTCTimestamp(),
        completed_at: createPESUTCTimestamp(),
        correlation_id: createPESCorrelationId(),
      };

      const result = HybridExecutionResult.safeParse(validResult);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.pes_result).toBeDefined();
        expect(result.data.evolution_result).toBeDefined();
        expect(result.data.knowledge_extracted).toHaveLength(1);
      }
    });

    it('should validate failed result with error', () => {
      const failedResult = {
        id: createPESCorrelationId(),
        task_id: createPESCorrelationId(),
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
        created_at: createPESUTCTimestamp(),
        completed_at: createPESUTCTimestamp(),
      };

      const result = HybridExecutionResult.safeParse(failedResult);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.error).toBeDefined();
        expect(result.data.error?.code).toBe('TIMEOUT');
      }
    });
  });

  describe('AdaptiveTrigger Schema', () => {
    it('should validate a valid adaptive trigger', () => {
      const validTrigger = {
        id: createPESCorrelationId(),
        condition: 'stagnation' as const,
        threshold: 5,
        action: 'switch_to_evolution' as const,
        cooldown_ms: 10000,
        metadata: { description: 'Switch to evolution if no improvement for 5 iterations' },
        created_at: createPESUTCTimestamp(),
        last_triggered_at: createPESUTCTimestamp(),
      };

      const result = AdaptiveTrigger.safeParse(validTrigger);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.condition).toBe('stagnation');
        expect(result.data.action).toBe('switch_to_evolution');
      }
    });

    it('should reject negative cooldown', () => {
      const invalidTrigger = {
        id: createPESCorrelationId(),
        condition: 'low_confidence' as const,
        threshold: 0.6,
        action: 'increase_iterations' as const,
        cooldown_ms: -1000, // Invalid: must be positive
      };

      const result = AdaptiveTrigger.safeParse(invalidTrigger);
      expect(result.success).toBe(false);
    });
  });

  describe('KnowledgeTransfer Schema', () => {
    it('should validate a valid knowledge transfer', () => {
      const validTransfer = {
        id: createPESCorrelationId(),
        source_paradigm: 'pes' as const,
        destination_paradigm: 'evolution' as const,
        knowledge: {
          id: createPESCorrelationId(),
          source_type: 'loongflow_solution' as const,
          problem_id: createPESCorrelationId(),
          knowledge_type: 'solution_pattern' as const,
          content: {
            pattern: 'Test pattern',
            success_rate: 0.9,
            avg_score: 0.85,
            usage_count: 5,
          },
          extracted_at: createPESUTCTimestamp(),
        },
        status: 'completed' as const,
        transfer_result: { success: true },
        initiated_at: createPESUTCTimestamp(),
        completed_at: createPESUTCTimestamp(),
        metadata: { transfer_method: 'direct' },
      };

      const result = KnowledgeTransfer.safeParse(validTransfer);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.source_paradigm).toBe('pes');
        expect(result.data.destination_paradigm).toBe('evolution');
        expect(result.data.status).toBe('completed');
      }
    });

    it('should reject invalid status', () => {
      const invalidTransfer = {
        id: createPESCorrelationId(),
        source_paradigm: 'evolution' as const,
        destination_paradigm: 'pes' as const,
        knowledge: {
          id: createPESCorrelationId(),
          source_type: 'hybrid_result' as const,
          problem_id: createPESCorrelationId(),
          knowledge_type: 'planning_strategy' as const,
          content: {
            pattern: 'Test',
            success_rate: 0.8,
            avg_score: 0.75,
            usage_count: 3,
          },
          extracted_at: createPESUTCTimestamp(),
        },
        status: 'invalid_status' as const, // Invalid enum value
        initiated_at: createPESUTCTimestamp(),
      };

      const result = KnowledgeTransfer.safeParse(invalidTransfer);
      expect(result.success).toBe(false);
    });
  });

  describe('Validation Functions', () => {
    const createValidProblem = (): Problem => ({
      id: createPESCorrelationId(),
      type: 'optimization',
      description: 'Test',
      context: {},
      constraints: [],
      success_criteria: [],
      created_at: createPESUTCTimestamp(),
    });

    it('should validate hybrid task using validateHybridTask', () => {
      const task = {
        id: createPESCorrelationId(),
        type: 'adaptive_execute' as const,
        problem: createValidProblem(),
        evolution_config: {
          generations: 50,
          population_size: 100,
          mutation_rate: 0.1,
          crossover_rate: 0.8,
        },
        integration_strategy: 'adaptive' as const,
        created_at: createPESUTCTimestamp(),
      };

      const validation = validateHybridTask(task);
      expect(validation.success).toBe(true);
    });

    it('should validate evolutionary knowledge using validateEvolutionaryKnowledge', () => {
      const knowledge = {
        id: createPESCorrelationId(),
        source_type: 'loongflow_solution' as const,
        problem_id: createPESCorrelationId(),
        knowledge_type: 'solution_pattern' as const,
        content: {
          pattern: 'Test pattern',
          success_rate: 0.9,
          avg_score: 0.85,
          usage_count: 5,
        },
        extracted_at: createPESUTCTimestamp(),
      };

      const validation = validateEvolutionaryKnowledge(knowledge);
      expect(validation.success).toBe(true);
    });

    it('should validate hybrid execution result using validateHybridExecutionResult', () => {
      const result = {
        id: createPESCorrelationId(),
        task_id: createPESCorrelationId(),
        integration_metrics: {
          pes_iterations: 5,
          evolution_generations: 50,
          total_duration_ms: 60000,
          synergy_score: 0.85,
        },
        created_at: createPESUTCTimestamp(),
      };

      const validation = validateHybridExecutionResult(result);
      expect(validation.success).toBe(true);
    });

    it('should validate adaptive trigger using validateAdaptiveTrigger', () => {
      const trigger = {
        id: createPESCorrelationId(),
        condition: 'stagnation' as const,
        threshold: 5,
        action: 'switch_to_evolution' as const,
      };

      const validation = validateAdaptiveTrigger(trigger);
      expect(validation.success).toBe(true);
    });
  });

  describe('Type Guards', () => {
    const createValidProblem = (): Problem => ({
      id: createPESCorrelationId(),
      type: 'optimization',
      description: 'Test',
      context: {},
      constraints: [],
      success_criteria: [],
      created_at: createPESUTCTimestamp(),
    });

    it('should check if data is a valid HybridTask', () => {
      const task = {
        id: createPESCorrelationId(),
        type: 'pes_optimize' as const,
        problem: createValidProblem(),
        evolution_config: {
          generations: 50,
          population_size: 100,
          mutation_rate: 0.1,
          crossover_rate: 0.8,
        },
        integration_strategy: 'sequential' as const,
        created_at: createPESUTCTimestamp(),
      };

      expect(isHybridTask(task)).toBe(true);
      expect(isHybridTask({ not: 'a task' })).toBe(false);
    });

    it('should check if data is a valid HybridExecutionResult', () => {
      const result = {
        id: createPESCorrelationId(),
        task_id: createPESCorrelationId(),
        integration_metrics: {
          pes_iterations: 5,
          evolution_generations: 50,
          total_duration_ms: 60000,
          synergy_score: 0.85,
        },
        created_at: createPESUTCTimestamp(),
      };

      expect(isHybridExecutionResult(result)).toBe(true);
      expect(isHybridExecutionResult({ not: 'a result' })).toBe(false);
    });

    it('should check if data is a valid EvolutionaryKnowledge', () => {
      const knowledge = {
        id: createPESCorrelationId(),
        source_type: 'loongflow_solution' as const,
        problem_id: createPESCorrelationId(),
        knowledge_type: 'solution_pattern' as const,
        content: {
          pattern: 'Test',
          success_rate: 0.9,
          avg_score: 0.85,
          usage_count: 5,
        },
        extracted_at: createPESUTCTimestamp(),
      };

      expect(isEvolutionaryKnowledge(knowledge)).toBe(true);
      expect(isEvolutionaryKnowledge({ not: 'knowledge' })).toBe(false);
    });

    it('should check if data is a valid AdaptiveTrigger', () => {
      const trigger = {
        id: createPESCorrelationId(),
        condition: 'low_confidence' as const,
        threshold: 0.6,
        action: 'increase_iterations' as const,
      };

      expect(isAdaptiveTrigger(trigger)).toBe(true);
      expect(isAdaptiveTrigger({ not: 'a trigger' })).toBe(false);
    });
  });

  describe('Transformation Functions', () => {
    it('should transform LoongFlow solution to evolutionary knowledge', () => {
      const solution: LoongFlowSolution = {
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

      const problemId = createPESCorrelationId();
      const knowledge = transformLoongFlowSolutionToKnowledge(
        solution,
        problemId,
        'solution_pattern'
      );

      expect(knowledge.source_type).toBe('loongflow_solution');
      expect(knowledge.problem_id).toBe(problemId);
      expect(knowledge.knowledge_type).toBe('solution_pattern');
      expect(knowledge.content.success_rate).toBe(0.9);
    });

    it('should transform hybrid result to summary', () => {
      const result: HybridExecutionResult = {
        id: createPESCorrelationId(),
        task_id: createPESCorrelationId(),
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
        created_at: createPESUTCTimestamp(),
        completed_at: createPESUTCTimestamp(),
      };

      const summary = transformHybridResultToSummary(result);

      expect(summary.result_id).toBe(result.id);
      expect(summary.performance_assessment.success_criteria_met).toBe(true);
      expect(summary.performance_assessment.quality_score).toBe(0.95);
      expect(summary.insights).toContain('PES iterations: 5');
      expect(summary.insights).toContain('Evolution generations: 50');
    });
  });

  describe('Enum Validation', () => {
    it('should accept all valid hybrid task types', () => {
      const validTypes = [
        'pes_optimize',
        'evolve_solve',
        'adaptive_execute',
        'parallel_hybrid',
        'interleaved',
      ] as const;

      validTypes.forEach((type) => {
        const result = HybridTaskType.safeParse(type);
        expect(result.success).toBe(true);
      });
    });

    it('should accept all valid integration strategies', () => {
      const validStrategies = ['sequential', 'parallel', 'interleaved', 'adaptive'] as const;

      validStrategies.forEach((strategy) => {
        const result = IntegrationStrategy.safeParse(strategy);
        expect(result.success).toBe(true);
      });
    });

    it('should accept all valid adaptive trigger conditions', () => {
      const validConditions = [
        'low_confidence',
        'high_complexity',
        'stagnation',
        'timeout_risk',
        'convergence_detected',
        'divergence_detected',
      ] as const;

      validConditions.forEach((condition) => {
        const result = AdaptiveTriggerCondition.safeParse(condition);
        expect(result.success).toBe(true);
      });
    });

    it('should accept all valid adaptive actions', () => {
      const validActions = [
        'switch_to_evolution',
        'switch_to_pes',
        'increase_iterations',
        'call_for_help',
        'merge_populations',
        'reset_and_restart',
      ] as const;

      validActions.forEach((action) => {
        const result = AdaptiveAction.safeParse(action);
        expect(result.success).toBe(true);
      });
    });
  });
});
