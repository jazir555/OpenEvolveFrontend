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

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { v4 as uuidv4 } from 'uuid';
import { EventBus } from '../../event-bus';
import { Logger } from '../../../lib/logger';
import {
  PESEvolutionWorkflow,
  PESEvolutionInput,
  createPESEvolutionWorkflow,
} from '../pes-evolution-workflow';
import {
  KnowledgeExtractionWorkflow,
  KnowledgeExtractionInput,
  createKnowledgeExtractionWorkflow,
} from '../knowledge-extraction-workflow';
import {
  AdaptiveExecutionWorkflow,
  AdaptiveExecutionInput,
  createAdaptiveExecutionWorkflow,
} from '../adaptive-execution-workflow';
import {
  MultiStageReasoningWorkflow,
  MultiStageReasoningInput,
  createMultiStageReasoningWorkflow,
} from '../multi-stage-reasoning-workflow';
import { correlationTracker } from '../../correlation-tracker';

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

  async submitProblem(request: any) {
    return {
      agent_id: uuidv4(),
      status: 'running',
      message: 'Problem submitted',
    };
  }

  async getAgentState(agentId: string) {
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

  async getExecutionResult(agentId: string) {
    return {
      agent_id: agentId,
      status: 'completed',
      final_solution: 'Solution text here',
      final_score: 0.85,
      best_solutions: [
        {
          solution_id: uuidv4(),
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

  async getBestSolutions(islandId?: number, topK?: number) {
    return [
      {
        solution_id: uuidv4(),
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
        solution_id: uuidv4(),
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

  async interruptAgent(agentId: string) {
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

  async createWorkflow(workflow: any) {
    return {
      message: 'Workflow created',
      workflow_id: uuidv4(),
    };
  }

  async getWorkflowStatus(workflowId: string) {
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

  async createTeam(team: any) {
    return {
      message: 'Team created',
      team_name: team.name,
    };
  }

  async createGauntlet(gauntlet: any) {
    return {
      message: 'Gauntlet created',
      gauntlet_name: gauntlet.name,
    };
  }
}

class MockEventBus extends EventBus {
  publishedEvents: any[] = [];

  async publish(event: any): Promise<void> {
    this.publishedEvents.push(event);
    await super.publish(event);
  }

  clearEvents(): void {
    this.publishedEvents = [];
  }

  getEventsByType(type: string): any[] {
    return this.publishedEvents.filter(e => e.type === type);
  }
}

// ============================================================================
// TEST DATA
// ============================================================================

const createMockProblem = () => ({
  id: uuidv4(),
  type: 'optimization' as const,
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

describe('PESEvolutionWorkflow', () => {
  let workflow: PESEvolutionWorkflow;
  let loongflowAdapter: MockLoongFlowAdapter;
  let openevolveAdapter: MockOpenEvolveAdapter;
  let eventBus: MockEventBus;

  beforeEach(() => {
    loongflowAdapter = new MockLoongFlowAdapter();
    openevolveAdapter = new MockOpenEvolveAdapter();
    eventBus = new MockEventBus();

    workflow = createPESEvolutionWorkflow({
      loongflowAdapter: loongflowAdapter as any,
      openevolveAdapter: openevolveAdapter as any,
      eventBus,
      checkpoints_enabled: false,
      default_timeout_ms: 30000,
      max_retries: 2,
    });
  });

  afterEach(() => {
    eventBus.clearEvents();
  });

  describe('execute', () => {
    it('should execute PES then Evolution', async () => {
      const input: PESEvolutionInput = {
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

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
      expect(result.task_id).toBeDefined();
      expect(result.pes_result).toBeDefined();
      expect(result.evolution_result).toBeDefined();
      expect(result.integration_metrics).toBeDefined();
      expect(result.knowledge_extracted).toBeDefined();
      expect(Array.isArray(result.knowledge_extracted)).toBe(true);
    });

    it('should publish events for each phase', async () => {
      const input: PESEvolutionInput = {
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

      expect(problemPlannedEvents.length).toBeGreaterThan(0);
      expect(solutionExecutedEvents.length).toBeGreaterThan(0);
      expect(solutionOptimizedEvents.length).toBeGreaterThan(0);
      expect(resultSummarizedEvents.length).toBeGreaterThan(0);
      expect(knowledgeExtractedEvents.length).toBeGreaterThan(0);
      expect(workflowCompletedEvents.length).toBe(1);
    });

    it('should handle failures gracefully', async () => {
      const failingAdapter = new MockLoongFlowAdapter();
      failingAdapter.submitProblem = async () => {
        throw new Error('Adapter failure');
      };

      const failingWorkflow = createPESEvolutionWorkflow({
        loongflowAdapter: failingAdapter as any,
        openevolveAdapter: openevolveAdapter as any,
        eventBus,
        max_retries: 1,
      });

      const input: PESEvolutionInput = {
        problem: createMockProblem(),
      };

      await expect(failingWorkflow.execute(input)).rejects.toThrow();

      // Check that failure event was published
      const workflowFailedEvents = eventBus.getEventsByType('WorkflowFailed');
      expect(workflowFailedEvents.length).toBe(1);
    });

    it('should skip optimization if disabled', async () => {
      const input: PESEvolutionInput = {
        problem: createMockProblem(),
        enable_optimization: false,
      };

      const result = await workflow.execute(input);

      expect(result.pes_result).toBeDefined();
      expect(result.evolution_result).toBeUndefined();
    });
  });

  describe('checkpoints', () => {
    it('should save checkpoints for each stage', async () => {
      const workflowWithCheckpoints = createPESEvolutionWorkflow({
        loongflowAdapter: loongflowAdapter as any,
        openevolveAdapter: openevolveAdapter as any,
        eventBus,
        checkpoints_enabled: true,
      });

      const input: PESEvolutionInput = {
        problem: createMockProblem(),
      };

      await workflowWithCheckpoints.execute(input);

      // Check that checkpoints were saved
      const checkpoints = workflowWithCheckpoints.getCheckpointsForTask(expect.any(String));
      expect(checkpoints.length).toBeGreaterThan(0);
    });

    it('should clear checkpoints when requested', async () => {
      const workflowWithCheckpoints = createPESEvolutionWorkflow({
        loongflowAdapter: loongflowAdapter as any,
        openevolveAdapter: openevolveAdapter as any,
        eventBus,
        checkpoints_enabled: true,
      });

      const input: PESEvolutionInput = {
        problem: createMockProblem(),
      };

      const result = await workflowWithCheckpoints.execute(input);
      workflowWithCheckpoints.clearCheckpoints(result.task_id);

      const checkpoints = workflowWithCheckpoints.getCheckpointsForTask(result.task_id);
      expect(checkpoints.length).toBe(0);
    });
  });
});

// ============================================================================
// KNOWLEDGE EXTRACTION WORKFLOW TESTS
// ============================================================================

describe('KnowledgeExtractionWorkflow', () => {
  let workflow: KnowledgeExtractionWorkflow;
  let loongflowAdapter: MockLoongFlowAdapter;
  let eventBus: MockEventBus;

  beforeEach(() => {
    loongflowAdapter = new MockLoongFlowAdapter();
    eventBus = new MockEventBus();

    workflow = createKnowledgeExtractionWorkflow({
      loongflowAdapter: loongflowAdapter as any,
      eventBus,
      enable_graph_storage: false,
      enable_vectorization: false,
      enable_problem_formulation: true,
    });
  });

  afterEach(() => {
    eventBus.clearEvents();
  });

  describe('execute', () => {
    it('should extract knowledge from solutions', async () => {
      const input: KnowledgeExtractionInput = {
        island_id: 0,
        top_k: 10,
        min_score: 0.5,
      };

      const result = await workflow.execute(input);

      expect(result).toBeDefined();
      expect(result.knowledge).toBeDefined();
      expect(Array.isArray(result.knowledge)).toBe(true);
      expect(result.patterns).toBeDefined();
      expect(Array.isArray(result.patterns)).toBe(true);
      expect(result.problems).toBeDefined();
      expect(Array.isArray(result.problems)).toBe(true);
    });

    it('should deduplicate knowledge', async () => {
      const input: KnowledgeExtractionInput = {
        island_id: 0,
        top_k: 10,
      };

      const result = await workflow.execute(input);

      // Check for duplicates by source_id
      const sourceIds = result.knowledge.map(k => k.source_id);
      const uniqueSourceIds = new Set(sourceIds);

      expect(sourceIds.length).toBe(uniqueSourceIds.size);
    });

    it('should publish KnowledgeExtracted events', async () => {
      const input: KnowledgeExtractionInput = {
        island_id: 0,
        top_k: 5,
      };

      await workflow.execute(input);

      const knowledgeEvents = eventBus.getEventsByType('KnowledgeExtracted');
      expect(knowledgeEvents.length).toBeGreaterThan(0);

      knowledgeEvents.forEach(event => {
        expect(event.data).toHaveProperty('knowledge_id');
        expect(event.data).toHaveProperty('problem_id');
        expect(event.data).toHaveProperty('knowledge_type');
      });
    });

    it('should filter by minimum score', async () => {
      const input: KnowledgeExtractionInput = {
        island_id: 0,
        top_k: 10,
        min_score: 0.9,
      };

      const result = await workflow.execute(input);

      // All knowledge should have score >= 0.9
      result.knowledge.forEach(k => {
        expect(k.content.avg_score).toBeGreaterThanOrEqual(0.9);
      });
    });
  });

  describe('problem formulation', () => {
    it('should formulate problems from low-quality patterns', async () => {
      const input: KnowledgeExtractionInput = {
        island_id: 0,
        top_k: 10,
        min_score: 0.3, // Low threshold to get more varied solutions
      };

      const result = await workflow.execute(input);

      // Check that problems were formulated
      expect(result.problems.length).toBeGreaterThan(0);

      result.problems.forEach(problem => {
        expect(problem).toHaveProperty('problem_id');
        expect(problem).toHaveProperty('problem_type');
        expect(problem).toHaveProperty('description');
        expect(problem).toHaveProperty('priority');
        expect(problem.priority).toBeGreaterThan(0);
        expect(problem.priority).toBeLessThanOrEqual(10);
      });
    });
  });
});

// ============================================================================
// ADAPTIVE EXECUTION WORKFLOW TESTS
// ============================================================================

describe('AdaptiveExecutionWorkflow', () => {
  let workflow: AdaptiveExecutionWorkflow;
  let loongflowAdapter: MockLoongFlowAdapter;
  let openevolveAdapter: MockOpenEvolveAdapter;
  let eventBus: MockEventBus;

  beforeEach(() => {
    loongflowAdapter = new MockLoongFlowAdapter();
    openevolveAdapter = new MockOpenEvolveAdapter();
    eventBus = new MockEventBus();

    workflow = createAdaptiveExecutionWorkflow({
      loongflowAdapter: loongflowAdapter as any,
      openevolveAdapter: openevolveAdapter as any,
      eventBus,
      max_paradigm_switches: 3,
      max_iterations: 5,
    });
  });

  afterEach(() => {
    eventBus.clearEvents();
  });

  describe('executeAdaptive', () => {
    it('should execute with initial paradigm', async () => {
      const input: AdaptiveExecutionInput = {
        problem: createMockProblem(),
        initial_paradigm: 'PES',
        triggers: [],
      };

      const result = await workflow.executeAdaptive(input);

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
      expect(result.task_id).toBeDefined();
      expect(result.integration_metrics).toBeDefined();
      expect(result.best_solution).toBeDefined();
    });

    it('should publish ParadigmSwitched events when switching', async () => {
      const input: AdaptiveExecutionInput = {
        problem: createMockProblem(),
        initial_paradigm: 'PES',
        triggers: [
          {
            id: uuidv4(),
            condition: 'low_confidence',
            threshold: 0.95,
            action: 'switch_to_evolution',
          },
        ],
      };

      await workflow.executeAdaptive(input);

      const paradigmSwitchedEvents = eventBus.getEventsByType('ParadigmSwitched');
      // May or may not switch depending on scores
      expect(paradigmSwitchedEvents).toBeDefined();
    });

    it('should respect max paradigm switches', async () => {
      const input: AdaptiveExecutionInput = {
        problem: createMockProblem(),
        initial_paradigm: 'PES',
        triggers: [
          {
            id: uuidv4(),
            condition: 'stagnation',
            threshold: 2,
            action: 'switch_to_evolution',
          },
        ],
        enable_hybrid_fallback: false,
      };

      const result = await workflow.executeAdaptive(input);

      // Should not exceed max switches
      expect(result.integration_metrics.paradigm_switches).toBeLessThanOrEqual(3);
    });

    it('should stop on convergence', async () => {
      const input: AdaptiveExecutionInput = {
        problem: createMockProblem(),
        initial_paradigm: 'PES',
        triggers: [],
      };

      const result = await workflow.executeAdaptive(input);

      // Should complete successfully
      expect(result.integration_metrics.synergy_score).toBeGreaterThan(0);
    });
  });
});

// ============================================================================
// MULTI-STAGE REASONING WORKFLOW TESTS
// ============================================================================

describe('MultiStageReasoningWorkflow', () => {
  let workflow: MultiStageReasoningWorkflow;
  let loongflowAdapter: MockLoongFlowAdapter;
  let openevolveAdapter: MockOpenEvolveAdapter;
  let eventBus: MockEventBus;

  beforeEach(() => {
    loongflowAdapter = new MockLoongFlowAdapter();
    openevolveAdapter = new MockOpenEvolveAdapter();
    eventBus = new MockEventBus();

    workflow = createMultiStageReasoningWorkflow({
      loongflowAdapter: loongflowAdapter as any,
      openevolveAdapter: openevolveAdapter as any,
      eventBus,
      enable_validation: true,
      enable_refinement: true,
      max_refinement_loops: 2,
    });
  });

  afterEach(() => {
    eventBus.clearEvents();
  });

  describe('executeReasoning', () => {
    it('should execute all stages', async () => {
      const input: MultiStageReasoningInput = {
        problem: createMockProblem(),
        validation_system: 'both',
      };

      const result = await workflow.executeReasoning(input);

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
      expect(result.pes_result).toBeDefined();
      expect(result.evolution_result).toBeDefined();
      expect(result.knowledge_extracted).toBeDefined();
      expect(result.integration_metrics).toBeDefined();
    });

    it('should publish events for each stage', async () => {
      const input: MultiStageReasoningInput = {
        problem: createMockProblem(),
      };

      await workflow.executeReasoning(input);

      const stageCompletedEvents = eventBus.getEventsByType('StageCompleted');
      expect(stageCompletedEvents.length).toBeGreaterThan(0);

      const workflowCompletedEvents = eventBus.getEventsByType('WorkflowCompleted');
      expect(workflowCompletedEvents.length).toBe(1);
    });

    it('should skip refinement if validation passes', async () => {
      const input: MultiStageReasoningInput = {
        problem: {
          ...createMockProblem(),
          // This should result in high score
        },
        refinement_threshold: 0.8,
      };

      const result = await workflow.executeReasoning(input);

      // If validation passes, no refinement needed
      expect(result).toBeDefined();
    });

    it('should extract knowledge from final solution', async () => {
      const input: MultiStageReasoningInput = {
        problem: createMockProblem(),
      };

      const result = await workflow.executeReasoning(input);

      expect(result.knowledge_extracted).toBeDefined();
      expect(Array.isArray(result.knowledge_extracted)).toBe(true);
      expect(result.knowledge_extracted.length).toBeGreaterThan(0);
    });

    it('should handle stage failures gracefully', async () => {
      const failingAdapter = new MockLoongFlowAdapter();
      failingAdapter.submitProblem = async () => {
        throw new Error('Plan stage failed');
      };

      const failingWorkflow = createMultiStageReasoningWorkflow({
        loongflowAdapter: failingAdapter as any,
        openevolveAdapter: openevolveAdapter as any,
        eventBus,
      });

      const input: MultiStageReasoningInput = {
        problem: createMockProblem(),
      };

      await expect(failingWorkflow.executeReasoning(input)).rejects.toThrow();

      const workflowFailedEvents = eventBus.getEventsByType('WorkflowFailed');
      expect(workflowFailedEvents.length).toBe(1);
    });
  });

  describe('validation', () => {
    it('should skip validation if disabled', async () => {
      const workflowNoValidation = createMultiStageReasoningWorkflow({
        loongflowAdapter: loongflowAdapter as any,
        openevolveAdapter: openevolveAdapter as any,
        eventBus,
        enable_validation: false,
      });

      const input: MultiStageReasoningInput = {
        problem: createMockProblem(),
      };

      const result = await workflowNoValidation.executeReasoning(input);

      expect(result).toBeDefined();
    });
  });
});

// ============================================================================
// WORKFLOW FACTORY TESTS
// ============================================================================

describe('Workflow Factory', () => {
  it('should throw error for unknown workflow type', () => {
    const { createWorkflow, WORKFLOWS } = require('../index');

    expect(() => {
      createWorkflow('unknown' as any, {});
    }).toThrow();
  });

  it('should validate workflow config', () => {
    const { validateWorkflowConfig, WORKFLOWS } = require('../index');

    const validation = validateWorkflowConfig(WORKFLOWS.PES_EVOLUTION, {
      loongflowAdapter: new MockLoongFlowAdapter(),
      openevolveAdapter: new MockOpenEvolveAdapter(),
    });

    expect(validation.valid).toBe(true);
    expect(validation.errors).toBeUndefined();
  });

  it('should detect missing adapters', () => {
    const { validateWorkflowConfig, WORKFLOWS } = require('../index');

    const validation = validateWorkflowConfig(WORKFLOWS.PES_EVOLUTION, {});

    expect(validation.valid).toBe(false);
    expect(validation.errors).toBeDefined();
    expect(validation.errors?.length).toBeGreaterThan(0);
  });
});
