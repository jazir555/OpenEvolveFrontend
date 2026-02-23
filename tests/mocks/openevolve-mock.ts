/**
 * Mock OpenEvolve Adapter for Testing
 *
 * Provides a mock implementation of the OpenEvolve adapter interface
 * for testing purposes without requiring actual OpenEvolve service.
 */

export interface MockOpenEvolveConfig {
  mockOptimized?: {
    solution: any;
    fitness: number;
  };
  mockGenerations?: number;
  mockPopulationSize?: number;
  mockConverged?: boolean;
  mockTimeout?: boolean;
  mockError?: boolean;
}

export interface WorkflowDefinition {
  workflow_id: string;
  name: string;
  description: string;
  problem_statement: string;
  max_refinement_loops: number;
  auto_approval_enabled: boolean;
  sub_problems: Array<{
    id: string;
    description: string;
    dependencies: string[];
    solver_team_name: string;
    gold_team_gauntlet_name: string;
  }>;
}

export interface WorkflowResponse {
  workflow_id: string;
  status: string;
  created_at: string;
}

export interface WorkflowState {
  status: 'running' | 'completed' | 'failed';
  final_solution?: {
    quality_metrics?: {
      score: number;
    };
  };
}

export interface OptimizationResult {
  best_fitness: number;
  generations: number;
  population_size: number;
  final_population: Array<{
    solution: any;
    fitness: number;
  }>;
  optimization_history: Array<{
    generation: number;
    best_fitness: number;
  }>;
}

/**
 * Create a mock OpenEvolve adapter
 */
export function createMockOpenEvolveAdapter(config: MockOpenEvolveConfig = {}) {
  const {
    mockOptimized,
    mockGenerations = 5,
    mockPopulationSize = 20,
    mockConverged = true,
    mockTimeout = false,
    mockError = false,
  } = config;

  let workflowId = 0;

  return {
    /**
     * Create an optimization workflow
     */
    async createWorkflow(workflow: WorkflowDefinition): Promise<WorkflowResponse> {
      if (mockError) {
        throw new Error('Mock error: Failed to create workflow');
      }

      workflowId++;
      return {
        workflow_id: workflow.workflow_id || `workflow_${workflowId}`,
        status: 'created',
        created_at: new Date().toISOString(),
      };
    },

    /**
     * Get workflow status
     */
    async getWorkflowStatus(workflowId: string): Promise<WorkflowState> {
      if (mockError) {
        throw new Error('Mock error: Failed to get workflow status');
      }

      if (mockTimeout) {
        return {
          status: 'running',
        };
      }

      const fitness = mockOptimized?.fitness || 0.95;

      return {
        status: 'completed',
        final_solution: {
          quality_metrics: {
            score: fitness,
          },
        },
      };
    },

    /**
     * Optimize a solution using evolutionary algorithms
     */
    async optimize(params: {
      solution: any;
      generations?: number;
      population_size?: number;
      mutation_rate?: number;
      crossover_rate?: number;
    }): Promise<OptimizationResult> {
      if (mockError) {
        throw new Error('Mock error: Failed to optimize');
      }

      if (mockTimeout) {
        throw new Error('Timeout: Optimization took too long');
      }

      const generations = params.generations || mockGenerations;
      const populationSize = params.population_size || mockPopulationSize;
      const initialFitness = 0.7;
      const finalFitness = mockOptimized?.fitness || 0.95;

      // Generate optimization history
      const optimizationHistory: Array<{ generation: number; best_fitness: number }> = [];
      for (let i = 0; i <= generations; i++) {
        const progress = i / generations;
        const fitness = initialFitness + (finalFitness - initialFitness) * progress;
        optimizationHistory.push({
          generation: i,
          best_fitness: fitness,
        });
      }

      // Generate final population
      const finalPopulation: Array<{ solution: any; fitness: number }> = [];
      for (let i = 0; i < Math.min(populationSize, 5); i++) {
        finalPopulation.push({
          solution: params.solution,
          fitness: finalFitness - (i * 0.05),
        });
      }

      return {
        best_fitness: finalFitness,
        generations,
        population_size: populationSize,
        final_population: finalPopulation,
        optimization_history: optimizationHistory,
      };
    },

    /**
     * Validate a formula against constraints
     */
    async validate(formula: any, constraints: any): Promise<{
      valid: boolean;
      satisfied: boolean;
      errors?: string[];
    }> {
      if (mockError) {
        throw new Error('Mock error: Failed to validate');
      }

      return {
        valid: true,
        satisfied: true,
      };
    },
  };
}

/**
 * Create a mock OpenEvolve adapter with realistic delays
 */
export function createRealisticMockOpenEvolveAdapter(config: MockOpenEvolveConfig = {}) {
  const adapter = createMockOpenEvolveAdapter(config);

  // Add delays to simulate real execution time
  const originalCreateWorkflow = adapter.createWorkflow.bind(adapter);
  adapter.createWorkflow = async (workflow) => {
    await new Promise(resolve => setTimeout(resolve, 100));
    return originalCreateWorkflow(workflow);
  };

  const originalGetWorkflowStatus = adapter.getWorkflowStatus.bind(adapter);
  adapter.getWorkflowStatus = async (workflowId) => {
    await new Promise(resolve => setTimeout(resolve, 50));
    return originalGetWorkflowStatus(workflowId);
  };

  const originalOptimize = adapter.optimize.bind(adapter);
  adapter.optimize = async (params) => {
    await new Promise(resolve => setTimeout(resolve, 200));
    return originalOptimize(params);
  };

  return adapter;
}
