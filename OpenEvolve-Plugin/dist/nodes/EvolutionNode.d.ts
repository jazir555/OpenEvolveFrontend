import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Evolution modes
 */
export type EvolutionMode = 'standard' | 'quality_diversity' | 'island_model';
/**
 * Evolution status types
 */
export type EvolutionStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed';
/**
 * Evolution configuration
 */
export interface EvolutionNodeConfig {
    mode?: EvolutionMode;
    maxIterations?: number;
    populationSize?: number;
    temperature?: number;
    topP?: number;
    timeoutMs?: number;
    enableWebSocket?: boolean;
}
/**
 * Individual in the population
 */
export interface EvolutionIndividual {
    id: string;
    content: string;
    fitness: number;
    generation: number;
    parentIds?: string[];
}
/**
 * Population metrics
 */
export interface PopulationMetrics {
    generation: number;
    bestFitness: number;
    averageFitness: number;
    diversity: number;
    convergenceRate: number;
    populationSize: number;
}
/**
 * Evolution result
 */
export interface EvolutionResult {
    evolutionId: string;
    status: EvolutionStatus;
    mode: EvolutionMode;
    bestContent: string;
    bestFitness: number;
    generations: number;
    populationMetrics: PopulationMetrics[];
    finalPopulation: EvolutionIndividual[];
    metadata: {
        startedAt: Date;
        completedAt?: Date;
        executionTime: number;
        parameters: {
            maxIterations: number;
            populationSize: number;
            temperature: number;
            topP: number;
        };
    };
}
/**
 * Evolution Node
 *
 * Executes genetic algorithm evolution for content optimization.
 * Supports multiple evolution strategies and real-time progress tracking.
 */
export declare class EvolutionNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Evolution Engine";
    static readonly DESCRIPTION = "Genetic algorithm evolution for iterative content improvement with multiple strategies";
    static readonly ICON = "evolution";
    static readonly CATEGORY = "optimization";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: EvolutionNodeConfig);
    /**
     * Execute evolution process
     *
     * @param inputs - Must contain 'content' and optionally 'models' configuration
     * @param context - Execution context
     * @returns Promise resolving to evolution result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Monitor evolution progress until completion
     *
     * @param evolutionId - Evolution ID to monitor
     * @param context - Execution context
     * @returns Promise resolving to evolution status
     */
    private monitorEvolution;
    /**
     * Validate input data
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get JSON Schema for configuration parameters
     *
     * @returns Parameter schema
     */
    getParameterSchema(): ParameterSchema;
    /**
     * Pause running evolution
     *
     * @param evolutionId - Evolution ID to pause
     * @returns Promise resolving to pause result
     */
    pauseEvolution(evolutionId: string): Promise<NodeResult>;
    /**
     * Resume paused evolution
     *
     * @param evolutionId - Evolution ID to resume
     * @returns Promise resolving to resume result
     */
    resumeEvolution(evolutionId: string): Promise<NodeResult>;
    /**
     * Stop running evolution
     *
     * @param evolutionId - Evolution ID to stop
     * @returns Promise resolving to stop result with final results
     */
    stopEvolution(evolutionId: string): Promise<NodeResult>;
    /**
     * Delete evolution
     *
     * @param evolutionId - Evolution ID to delete
     * @returns Promise resolving to deletion result
     */
    deleteEvolution(evolutionId: string): Promise<NodeResult>;
}
export default EvolutionNode;
