import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Solution generation strategies
 */
export type SolutionStrategy = 'MAKER' | 'MCTS' | 'Evolutionary' | 'Hybrid';
/**
 * Solution interface
 */
export interface Solution {
    id: string;
    content: string;
    strategy: SolutionStrategy;
    qualityScore: number;
    confidence: number;
    iteration: number;
    metadata: {
        generatedAt: Date;
        executionTime: number;
        problemHash: string;
        [key: string]: any;
    };
    qualityMetrics: {
        completeness: number;
        correctness: number;
        clarity: number;
        efficiency: number;
        innovation: number;
    };
    alternatives?: Solution[];
}
/**
 * Convergence metrics
 */
export interface ConvergenceMetrics {
    iterations: number;
    qualityHistory: number[];
    convergenceRate: number;
    converged: boolean;
    finalQuality: number;
    bestIteration: number;
}
/**
 * Solution node configuration
 */
export interface SolutionNodeConfig {
    strategy?: SolutionStrategy;
    maxIterations?: number;
    qualityThreshold?: number;
    temperature?: number;
    generateAlternatives?: boolean;
    numAlternatives?: number;
    enableCaching?: boolean;
    timeoutMs?: number;
}
/**
 * Solution generation result
 */
export interface SolutionResult {
    bestSolution: Solution;
    allSolutions: Solution[];
    convergenceMetrics: ConvergenceMetrics;
    metadata: {
        problem: string;
        strategyUsed: SolutionStrategy;
        totalExecutionTime: number;
        iterationsCompleted: number;
        cacheHits: number;
        [key: string]: any;
    };
}
/**
 * Solution Generation Node
 *
 * Generates high-quality solutions using various strategies.
 * Iterates until quality threshold is met or max iterations reached.
 */
export declare class SolutionNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Solution Generation";
    static readonly DESCRIPTION = "Generate solutions for problems using MAKER, MCTS, Evolutionary, or Hybrid strategies with quality tracking";
    static readonly ICON = "solution";
    static readonly CATEGORY = "generation";
    static readonly VERSION = "1.0.0";
    private static solutionCache;
    private readonly maxCacheSize;
    constructor(id: string, config?: SolutionNodeConfig);
    /**
     * Execute solution generation
     *
     * @param inputs - Must contain 'problem' string
     * @param context - Execution context
     * @returns Promise resolving to solution result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
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
     * Generate solutions iteratively until convergence or max iterations
     *
     * @param problem - Problem statement
     * @param analysis - Problem analysis
     * @param requirements - Optional requirements
     * @param constraints - Optional constraints
     * @param startTime - Start time for timeout
     * @returns Solutions and convergence metrics
     */
    private generateSolutionsIteratively;
    /**
     * Generate a single solution
     *
     * @param problem - Problem statement
     * @param analysis - Problem analysis
     * @param iteration - Current iteration
     * @param strategy - Generation strategy
     * @param requirements - Optional requirements
     * @param constraints - Optional constraints
     * @returns Generated solution
     */
    private generateSingleSolution;
    /**
     * Score a solution on multiple quality metrics
     *
     * @param solution - Solution to score
     * @param problem - Original problem
     * @param requirements - Optional requirements
     * @returns Scored solution
     */
    private scoreSolution;
    /**
     * Select best solution from array
     *
     * @param solutions - Array of solutions
     * @returns Best solution
     */
    private selectBestSolution;
    /**
     * Select alternative solutions
     *
     * @param solutions - All solutions
     * @param bestSolution - Best solution to exclude
     * @returns Alternative solutions
     */
    private selectAlternatives;
    private generateMakerSolution;
    private generateMCTSSolution;
    private generateEvolutionarySolution;
    private generateHybridSolution;
    private analyzeProblem;
    private extractKeyPoints;
    private generateDetailedContent;
    private extractKeywords;
    private analyzeProblemSpace;
    private generateExplorationPaths;
    private calculateConfidence;
    private hashProblem;
    private cacheSolution;
    private checkConvergence;
    private calculateConvergenceRate;
    private createMockConvergenceMetrics;
    private calculateCompleteness;
    private calculateCorrectness;
    private calculateClarity;
    private calculateEfficiency;
    private calculateInnovation;
}
export default SolutionNode;
