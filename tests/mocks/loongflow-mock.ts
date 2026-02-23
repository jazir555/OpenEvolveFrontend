/**
 * Mock LoongFlow Adapter for Testing
 *
 * Provides a mock implementation of the LoongFlow adapter interface
 * for testing purposes without requiring actual LoongFlow service.
 */

import { LoongFlowSolution } from '../../glue/schemas/loongflow-canonical';

export interface MockLoongFlowConfig {
  mockSolution?: Partial<LoongFlowSolution>;
  mockBestSolutions?: LoongFlowSolution[];
  mockLowConfidence?: boolean;
  mockTimeout?: boolean;
  mockError?: boolean;
}

export interface AgentSubmitResponse {
  agent_id: string;
  status: string;
  submitted_at: string;
}

export interface AgentState {
  status: 'idle' | 'planning' | 'executing' | 'summarizing' | 'evolving' | 'completed' | 'failed';
  current_iteration: number;
  best_score: number;
  total_cost: number;
}

export interface ExecutionResult {
  final_solution: string;
  final_score: number;
  was_interrupted: boolean;
  total_iterations: number;
  best_solutions?: LoongFlowSolution[];
  start_time: string;
  end_time: string;
}

/**
 * Create a mock LoongFlow adapter
 */
export function createMockLoongFlowAdapter(config: MockLoongFlowConfig = {}) {
  const {
    mockSolution,
    mockBestSolutions,
    mockLowConfidence = false,
    mockTimeout = false,
    mockError = false,
  } = config;

  const defaultSolution: LoongFlowSolution = {
    solution: 'def solve(x): return x * 2',
    solution_id: 'sol_mock_123',
    generate_plan: 'Use mathematical optimization',
    score: mockLowConfidence ? 0.6 : 0.92,
    evaluation: 'Solution is correct',
    summary: 'Successfully solved the problem',
    parent_id: '',
    island_id: 0,
    iteration: 1,
    metadata: {},
    created_at: new Date().toISOString(),
  };

  const bestSolutions: LoongFlowSolution[] = mockBestSolutions || [
    {
      solution: 'def solve(x): return x * 2',
      solution_id: 'sol_1',
      generate_plan: 'Evolutionary strategy',
      score: 0.95,
      evaluation: 'Excellent solution',
      summary: 'Best solution found',
      parent_id: '',
      island_id: 0,
      iteration: 5,
      metadata: {},
      created_at: new Date().toISOString(),
    },
    {
      solution: 'def solve(x): return x ** 2',
      solution_id: 'sol_2',
      generate_plan: 'Gradient descent',
      score: 0.90,
      evaluation: 'Good solution',
      summary: 'Second best solution',
      parent_id: '',
      island_id: 1,
      iteration: 3,
      metadata: {},
      created_at: new Date().toISOString(),
    },
  ];

  let executionId = 0;

  return {
    /**
     * Submit a problem for execution
     */
    async submitProblem(params: {
      task: string;
      max_iterations?: number;
      target_score?: number;
      concurrency?: number;
      metadata?: Record<string, any>;
    }): Promise<AgentSubmitResponse> {
      if (mockError) {
        throw new Error('Mock error: Failed to submit problem');
      }

      executionId++;
      return {
        agent_id: `agent_${executionId}`,
        status: 'SUBMITTED',
        submitted_at: new Date().toISOString(),
      };
    },

    /**
     * Get current agent state
     */
    async getAgentState(agentId: string): Promise<AgentState> {
      if (mockError) {
        throw new Error('Mock error: Failed to get agent state');
      }

      if (mockTimeout) {
        throw new Error('Timeout: Operation took too long');
      }

      return {
        status: 'completed',
        current_iteration: 10,
        best_score: mockLowConfidence ? 0.6 : 0.92,
        total_cost: 5.0,
      };
    },

    /**
     * Get execution result
     */
    async getExecutionResult(agentId: string): Promise<ExecutionResult> {
      if (mockError) {
        throw new Error('Mock error: Failed to get execution result');
      }

      if (mockTimeout) {
        throw new Error('Timeout: Operation took too long');
      }

      const solution = mockSolution || defaultSolution;

      return {
        final_solution: solution.solution,
        final_score: solution.score,
        was_interrupted: false,
        total_iterations: solution.iteration || 10,
        best_solutions: bestSolutions,
        start_time: new Date(Date.now() - 5000).toISOString(),
        end_time: new Date().toISOString(),
      };
    },

    /**
     * Get best solutions from evolutionary database
     */
    async getBestSolutions(
      islandId?: number,
      topK: number = 10
    ): Promise<LoongFlowSolution[]> {
      if (mockError) {
        throw new Error('Mock error: Failed to get best solutions');
      }

      let solutions = bestSolutions;

      // Filter by island if specified
      if (islandId !== undefined) {
        solutions = solutions.filter(s => s.island_id === islandId);
      }

      // Return top K
      return solutions.slice(0, topK);
    },

    /**
     * Get specific solution by ID
     */
    async getSolution(solutionId: string): Promise<LoongFlowSolution | null> {
      if (mockError) {
        throw new Error('Mock error: Failed to get solution');
      }

      const solution = bestSolutions.find(s => s.solution_id === solutionId);
      return solution || null;
    },
  };
}

/**
 * Create a mock LoongFlow adapter with realistic delays
 */
export function createRealisticMockLoongFlowAdapter(config: MockLoongFlowConfig = {}) {
  const adapter = createMockLoongFlowAdapter(config);

  // Add delays to simulate real execution time
  const originalSubmitProblem = adapter.submitProblem.bind(adapter);
  adapter.submitProblem = async (params) => {
    await new Promise(resolve => setTimeout(resolve, 100));
    return originalSubmitProblem(params);
  };

  const originalGetAgentState = adapter.getAgentState.bind(adapter);
  adapter.getAgentState = async (agentId) => {
    await new Promise(resolve => setTimeout(resolve, 50));
    return originalGetAgentState(agentId);
  };

  const originalGetExecutionResult = adapter.getExecutionResult.bind(adapter);
  adapter.getExecutionResult = async (agentId) => {
    await new Promise(resolve => setTimeout(resolve, 100));
    return originalGetExecutionResult(agentId);
  };

  return adapter;
}
