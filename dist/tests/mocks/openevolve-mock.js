"use strict";
/**
 * Mock OpenEvolve Adapter for Testing
 *
 * Provides a mock implementation of the OpenEvolve adapter interface
 * for testing purposes without requiring actual OpenEvolve service.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.createMockOpenEvolveAdapter = createMockOpenEvolveAdapter;
exports.createRealisticMockOpenEvolveAdapter = createRealisticMockOpenEvolveAdapter;
/**
 * Create a mock OpenEvolve adapter
 */
function createMockOpenEvolveAdapter(config = {}) {
    const { mockOptimized, mockGenerations = 5, mockPopulationSize = 20, mockConverged = true, mockTimeout = false, mockError = false, } = config;
    let workflowId = 0;
    return {
        /**
         * Create an optimization workflow
         */
        async createWorkflow(workflow) {
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
        async getWorkflowStatus(workflowId) {
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
        async optimize(params) {
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
            const optimizationHistory = [];
            for (let i = 0; i <= generations; i++) {
                const progress = i / generations;
                const fitness = initialFitness + (finalFitness - initialFitness) * progress;
                optimizationHistory.push({
                    generation: i,
                    best_fitness: fitness,
                });
            }
            // Generate final population
            const finalPopulation = [];
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
        async validate(formula, constraints) {
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
function createRealisticMockOpenEvolveAdapter(config = {}) {
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
//# sourceMappingURL=openevolve-mock.js.map