"use strict";
/**
 * Mock LoongFlow Adapter for Testing
 *
 * Provides a mock implementation of the LoongFlow adapter interface
 * for testing purposes without requiring actual LoongFlow service.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.createMockLoongFlowAdapter = createMockLoongFlowAdapter;
exports.createRealisticMockLoongFlowAdapter = createRealisticMockLoongFlowAdapter;
/**
 * Create a mock LoongFlow adapter
 */
function createMockLoongFlowAdapter(config = {}) {
    const { mockSolution, mockBestSolutions, mockLowConfidence = false, mockTimeout = false, mockError = false, } = config;
    const defaultSolution = {
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
    const bestSolutions = mockBestSolutions || [
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
        async submitProblem(params) {
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
        async getAgentState(agentId) {
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
        async getExecutionResult(agentId) {
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
        async getBestSolutions(islandId, topK = 10) {
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
        async getSolution(solutionId) {
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
function createRealisticMockLoongFlowAdapter(config = {}) {
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
//# sourceMappingURL=loongflow-mock.js.map