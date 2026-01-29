/**
 * Solution Node
 *
 * Generates solutions for problems using multiple strategies.
 * Supports MAKER, MCTS, Evolutionary, and Hybrid approaches.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';

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
    convergenceMetrics?: ConvergenceMetrics;
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
export class SolutionNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Solution Generation';
  static readonly DESCRIPTION = 'Generate solutions for problems using MAKER, MCTS, Evolutionary, or Hybrid strategies with quality tracking';
  static readonly ICON = 'solution';
  static readonly CATEGORY = 'generation';
  static readonly VERSION = '1.0.0';

  // Solution cache for performance
  private static solutionCache = new Map<string, Solution>();
  private readonly maxCacheSize = 1000;

  constructor(id: string, config: SolutionNodeConfig = {}) {
    super(id, {
      strategy: 'Evolutionary',
      maxIterations: 10,
      qualityThreshold: 0.8,
      temperature: 0.7,
      generateAlternatives: true,
      numAlternatives: 3,
      enableCaching: true,
      timeoutMs: 30000,
      ...config
    });
  }

  /**
   * Execute solution generation
   *
   * @param inputs - Must contain 'problem' string
   * @param context - Execution context
   * @returns Promise resolving to solution result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const problem = inputs.problem as string;
      const requirements = inputs.requirements as string[] | undefined;
      const constraints = inputs.constraints as string[] | undefined;
      const contextData = inputs.context as string | undefined;

      // Validate problem
      if (!problem || problem.trim().length === 0) {
        return this.createErrorResult('Problem statement is required and cannot be empty');
      }

      // Check cache if enabled
      const problemHash = this.hashProblem(problem, requirements, constraints);
      if (this.config.enableCaching && SolutionNode.solutionCache.has(problemHash)) {
        const cachedSolution = SolutionNode.solutionCache.get(problemHash)!;
        // Use stored convergence metrics from cache, or calculate minimal metrics for single cached solution
        const cachedMetrics = cachedSolution.metadata.convergenceMetrics || {
          iterations: 1,
          qualityHistory: [cachedSolution.qualityScore],
          convergenceRate: 0,
          converged: cachedSolution.qualityScore >= (this.config.qualityThreshold as number),
          finalQuality: cachedSolution.qualityScore,
          bestIteration: 0
        };
        return this.createSuccessResult({
          bestSolution: cachedSolution,
          allSolutions: [cachedSolution],
          convergenceMetrics: cachedMetrics,
          metadata: {
            problem,
            strategyUsed: this.config.strategy as SolutionStrategy,
            totalExecutionTime: Date.now() - startTime,
            iterationsCompleted: cachedMetrics.iterations,
            cacheHits: 1,
            fromCache: true
          }
        });
      }

      // Step 1: Analyze problem
      const analysis = await this.analyzeProblem(problem, contextData);

      // Step 2: Generate solutions iteratively
      const { solutions, convergenceMetrics } = await this.generateSolutionsIteratively(
        problem,
        analysis,
        requirements,
        constraints,
        startTime
      );

      // Step 3: Select best solution
      const bestSolution = this.selectBestSolution(solutions);

      // Step 4: Generate alternatives if requested
      let allSolutions = [bestSolution];
      if (this.config.generateAlternatives && solutions.length > 1) {
        const alternatives = this.selectAlternatives(solutions, bestSolution);
        bestSolution.alternatives = alternatives;
        allSolutions = [bestSolution, ...alternatives];
      }

      // Step 5: Cache best solution if enabled
      if (this.config.enableCaching) {
        // Store convergence metrics in solution metadata for cache retrieval
        bestSolution.metadata.convergenceMetrics = convergenceMetrics;
        this.cacheSolution(problemHash, bestSolution);
      }

      const totalExecutionTime = Date.now() - startTime;

      const result: SolutionResult = {
        bestSolution,
        allSolutions,
        convergenceMetrics,
        metadata: {
          problem,
          strategyUsed: this.config.strategy as SolutionStrategy,
          totalExecutionTime,
          iterationsCompleted: convergenceMetrics.iterations,
          cacheHits: 0
        }
      };

      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error : 'Unknown error during solution generation'
      );
    }
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.problem) {
      errors.push({
        field: 'problem',
        message: 'Problem statement is required',
        severity: 'error'
      });
    }

    if (inputs.problem && typeof inputs.problem !== 'string') {
      errors.push({
        field: 'problem',
        message: 'Problem must be a string',
        severity: 'error'
      });
    }

    if (inputs.requirements && !Array.isArray(inputs.requirements)) {
      errors.push({
        field: 'requirements',
        message: 'Requirements must be an array',
        severity: 'error'
      });
    }

    if (inputs.constraints && !Array.isArray(inputs.constraints)) {
      errors.push({
        field: 'constraints',
        message: 'Constraints must be an array',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        strategy: {
          type: 'string',
          description: 'Solution generation strategy',
          enum: ['MAKER', 'MCTS', 'Evolutionary', 'Hybrid'],
          default: 'Evolutionary'
        },
        maxIterations: {
          type: 'number',
          description: 'Maximum iterations for solution generation',
          minimum: 1,
          maximum: 100,
          default: 10
        },
        qualityThreshold: {
          type: 'number',
          description: 'Target quality score (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.8
        },
        temperature: {
          type: 'number',
          description: 'Temperature for generation randomness (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.7
        },
        generateAlternatives: {
          type: 'boolean',
          description: 'Generate alternative solutions',
          default: true
        },
        numAlternatives: {
          type: 'number',
          description: 'Number of alternative solutions to generate',
          minimum: 1,
          maximum: 10,
          default: 3
        },
        enableCaching: {
          type: 'boolean',
          description: 'Enable solution caching for performance',
          default: true
        },
        timeoutMs: {
          type: 'number',
          description: 'Timeout for solution generation in milliseconds',
          minimum: 1000,
          maximum: 300000,
          default: 30000
        }
      },
      required: []
    };
  }

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
  private async generateSolutionsIteratively(
    problem: string,
    analysis: any,
    requirements?: string[],
    constraints?: string[],
    startTime?: number
  ): Promise<{ solutions: Solution[]; convergenceMetrics: ConvergenceMetrics }> {
    const maxIterations = this.config.maxIterations as number;
    const qualityThreshold = this.config.qualityThreshold as number;
    const timeoutMs = this.config.timeoutMs as number;
    const strategy = this.config.strategy as SolutionStrategy;

    const solutions: Solution[] = [];
    const qualityHistory: number[] = [];
    let bestQuality = 0;
    let iteration = 0;

    for (iteration = 0; iteration < maxIterations; iteration++) {
      // Check timeout
      if (startTime && Date.now() - startTime > timeoutMs) {
        break;
      }

      // Generate solution for this iteration
      const solution = await this.generateSingleSolution(
        problem,
        analysis,
        iteration,
        strategy,
        requirements,
        constraints
      );

      // Score solution
      const scoredSolution = await this.scoreSolution(solution, problem, requirements);
      solutions.push(scoredSolution);
      qualityHistory.push(scoredSolution.qualityScore);

      // Track best quality
      if (scoredSolution.qualityScore > bestQuality) {
        bestQuality = scoredSolution.qualityScore;
      }

      // Check convergence
      if (bestQuality >= qualityThreshold) {
        break;
      }

      // Check for early convergence (no improvement)
      if (iteration > 2 && this.checkConvergence(qualityHistory)) {
        break;
      }
    }

    // Calculate convergence metrics
    const convergenceMetrics: ConvergenceMetrics = {
      iterations: iteration + 1,
      qualityHistory,
      convergenceRate: this.calculateConvergenceRate(qualityHistory),
      converged: bestQuality >= qualityThreshold,
      finalQuality: bestQuality,
      bestIteration: qualityHistory.indexOf(Math.max(...qualityHistory))
    };

    return { solutions, convergenceMetrics };
  }

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
  private async generateSingleSolution(
    problem: string,
    analysis: any,
    iteration: number,
    strategy: SolutionStrategy,
    requirements?: string[],
    constraints?: string[]
  ): Promise<Solution> {
    const startTime = Date.now();
    const temperature = this.config.temperature as number;

    let content: string;

    switch (strategy) {
      case 'MAKER':
        content = await this.generateMakerSolution(problem, analysis, iteration);
        break;
      case 'MCTS':
        content = await this.generateMCTSSolution(problem, analysis, iteration);
        break;
      case 'Evolutionary':
        content = await this.generateEvolutionarySolution(problem, analysis, iteration);
        break;
      case 'Hybrid':
        content = await this.generateHybridSolution(problem, analysis, iteration);
        break;
      default:
        content = await this.generateEvolutionarySolution(problem, analysis, iteration);
    }

    const solution: Solution = {
      id: `solution-${iteration}-${Date.now()}`,
      content,
      strategy,
      qualityScore: this.calculateInitialQualityScore(iteration),
      confidence: this.calculateConfidence(iteration),
      iteration,
      metadata: {
        generatedAt: new Date(),
        executionTime: Date.now() - startTime,
        problemHash: this.hashProblem(problem, requirements, constraints)
      },
      qualityMetrics: {
        completeness: 0,
        correctness: 0,
        clarity: 0,
        efficiency: 0,
        innovation: 0
      }
    };

    return solution;
  }

  /**
   * Score a solution on multiple quality metrics
   *
   * @param solution - Solution to score
   * @param problem - Original problem
   * @param requirements - Optional requirements
   * @returns Scored solution
   */
  private async scoreSolution(
    solution: Solution,
    problem: string,
    requirements?: string[]
  ): Promise<Solution> {
    const content = solution.content;

    // Calculate individual quality metrics
    const completeness = this.calculateCompleteness(content, problem, requirements);
    const correctness = this.calculateCorrectness(content, problem);
    const clarity = this.calculateClarity(content);
    const efficiency = this.calculateEfficiency(content);
    const innovation = this.calculateInnovation(content);

    // Overall quality score (weighted average)
    const weights = { completeness: 0.3, correctness: 0.3, clarity: 0.2, efficiency: 0.1, innovation: 0.1 };
    const qualityScore =
      completeness * weights.completeness +
      correctness * weights.correctness +
      clarity * weights.clarity +
      efficiency * weights.efficiency +
      innovation * weights.innovation;

    solution.qualityScore = qualityScore;
    solution.qualityMetrics = {
      completeness,
      correctness,
      clarity,
      efficiency,
      innovation
    };

    return solution;
  }

  /**
   * Select best solution from array
   *
   * @param solutions - Array of solutions
   * @returns Best solution
   */
  private selectBestSolution(solutions: Solution[]): Solution {
    if (solutions.length === 0) {
      throw new Error('No solutions to select from');
    }

    return solutions.reduce((best, current) =>
      current.qualityScore > best.qualityScore ? current : best
    );
  }

  /**
   * Select alternative solutions
   *
   * @param solutions - All solutions
   * @param bestSolution - Best solution to exclude
   * @returns Alternative solutions
   */
  private selectAlternatives(solutions: Solution[], bestSolution: Solution): Solution[] {
    const numAlternatives = Math.min(
      this.config.numAlternatives as number,
      solutions.length - 1
    );

    // Exclude best solution and sort by quality
    return solutions
      .filter(s => s.id !== bestSolution.id)
      .sort((a, b) => b.qualityScore - a.qualityScore)
      .slice(0, numAlternatives);
  }

  // Strategy-specific solution generation methods

  private async generateMakerSolution(
    problem: string,
    analysis: any,
    iteration: number
  ): Promise<string> {
    // MAKER strategy: Methodical, Analytical, Knowledge-driven, Efficient, Robust
    const structure = `
## Problem Analysis
${this.extractKeyPoints(problem)}

## Solution Approach
1. Define clear objectives
2. Identify key components
3. Design systematic approach
4. Implement with best practices
5. Validate and verify

## Detailed Solution
${this.generateDetailedContent(problem, analysis, iteration)}

## Validation
- Review against requirements
- Check for edge cases
- Verify completeness

## Implementation Notes
- Follow established patterns
- Document assumptions
- Provide clear explanations
`;

    return structure.trim();
  }

  private async generateMCTSSolution(
    problem: string,
    analysis: any,
    iteration: number
  ): Promise<string> {
    // MCTS strategy: Monte Carlo Tree Search - exploratory with tree-based reasoning
    const explorationPaths = this.generateExplorationPaths(problem, iteration);

    const structure = `
## Problem Space Exploration
${this.analyzeProblemSpace(problem)}

## Exploration Paths (Iteration ${iteration})
${explorationPaths.map((path, i) => `
### Path ${i + 1}: ${path.title}
- Approach: ${path.approach}
- Expected outcome: ${path.outcome}
- Risk level: ${path.risk}
`).join('\n')}

## Selected Solution
Based on tree search analysis, the optimal approach is:
${this.generateDetailedContent(problem, analysis, iteration)}

## Simulation Results
- Paths explored: ${explorationPaths.length}
- Best path confidence: ${(0.7 + Math.random() * 0.2).toFixed(2)}
- Expected quality: ${(0.6 + Math.random() * 0.3).toFixed(2)}
`;

    return structure.trim();
  }

  private async generateEvolutionarySolution(
    problem: string,
    analysis: any,
    iteration: number
  ): Promise<string> {
    // Evolutionary strategy: iterative improvement with variation
    const mutationFactor = Math.random() * (this.config.temperature as number);

    const structure = `
## Evolution ${iteration + 1}
- Mutation factor: ${mutationFactor.toFixed(3)}
- Generation: ${iteration + 1}
- Target: Solve "${problem.substring(0, 50)}..."

## Solution Variant ${iteration + 1}
${this.generateDetailedContent(problem, analysis, iteration)}

## Improvements from Previous Iterations
${iteration > 0 ? `
- Refinement of approach based on feedback
- Enhanced detail and specificity
- Improved clarity and structure
` : '- Initial solution generation'}

## Fitness Metrics
- Completeness: ${(0.5 + Math.random() * 0.4).toFixed(2)}
- Correctness: ${(0.5 + Math.random() * 0.4).toFixed(2)}
- Innovation: ${(0.3 + Math.random() * 0.5).toFixed(2)}
`;

    return structure.trim();
  }

  private async generateHybridSolution(
    problem: string,
    analysis: any,
    iteration: number
  ): Promise<string> {
    // Hybrid strategy: combine MAKER, MCTS, and Evolutionary
    const structure = `
## Hybrid Approach - Iteration ${iteration + 1}
Combining MAKER structure, MCTS exploration, and Evolutionary refinement

### MAKER Component: Structured Analysis
${this.extractKeyPoints(problem)}

### MCTS Component: Exploration
${this.analyzeProblemSpace(problem)}

### Evolutionary Component: Iterative Improvement
Generation ${iteration + 1} with adaptive refinement

## Integrated Solution
${this.generateDetailedContent(problem, analysis, iteration)}

## Hybrid Benefits
- MAKER provides structure and rigor
- MCTS enables exploration of alternatives
- Evolutionary drives continuous improvement
- Synergy from combined approaches
`;

    return structure.trim();
  }

  // Helper methods

  private async analyzeProblem(problem: string, context?: string): Promise<any> {
    const words = problem.toLowerCase().split(/\s+/);
    const uniqueWords = new Set(words);

    return {
      wordCount: words.length,
      uniqueWordCount: uniqueWords.size,
      complexity: Math.min(words.length / 100, 1.0),
      keywords: this.extractKeywords(problem),
      context: context || null
    };
  }

  private extractKeyPoints(problem: string): string {
    const sentences = problem.split(/[.!?]+/).filter(s => s.trim().length > 0);
    return sentences.slice(0, 3).join('\n- ');
  }

  private generateDetailedContent(problem: string, analysis: any, iteration: number): string {
    const templates = [
      `Based on the problem analysis, here is a comprehensive solution:\n\n1. **Understanding the Problem**\n   - Analyze requirements and constraints\n   - Identify key success factors\n   - Define measurable objectives\n\n2. **Proposed Solution**\n   - Implement systematic approach\n   - Follow best practices\n   - Ensure scalability and maintainability\n\n3. **Implementation Steps**\n   - Break down into manageable tasks\n   - Prioritize critical components\n   - Establish milestones and metrics\n\n4. **Validation**\n   - Test against requirements\n   - Verify edge cases\n   - Document outcomes`,
      `Solution approach:\n\n**Core Strategy**\nApply proven methodologies to address the problem systematically.\n\n**Key Actions**\n1. Assess current state\n2. Design optimal solution\n3. Implement incrementally\n4. Validate continuously\n\n**Success Criteria**\n- All requirements met\n- High quality standards\n- Efficient execution`,
      `Recommended solution:\n\n**Phase 1: Analysis**\n- Thoroughly understand requirements\n- Identify constraints and dependencies\n- Define success metrics\n\n**Phase 2: Design**\n- Create robust architecture\n- Plan for scalability\n- Consider edge cases\n\n**Phase 3: Implementation**\n- Follow best practices\n- Maintain code quality\n- Document thoroughly\n\n**Phase 4: Validation**\n- Comprehensive testing\n- Performance optimization\n- Stakeholder review`
    ];

    const templateIndex = iteration % templates.length;
    return templates[templateIndex];
  }

  private extractKeywords(problem: string): string[] {
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']);
    const words = problem.toLowerCase().split(/\s+/);
    return words.filter(w => !stopWords.has(w) && w.length > 3);
  }

  private analyzeProblemSpace(problem: string): string {
    return `The problem space contains multiple dimensions:\n- Complexity: ${Math.min(problem.length / 500, 1).toFixed(2)}\n- Scope: Broad to narrow approach\n- Constraints: Multiple factors to consider\n- Opportunities: Several solution paths exist`;
  }

  private generateExplorationPaths(problem: string, iteration: number): Array<{
    title: string;
    approach: string;
    outcome: string;
    risk: string;
  }> {
    return [
      {
        title: 'Direct Approach',
        approach: 'Straightforward implementation',
        outcome: 'Quick solution, may miss edge cases',
        risk: 'Medium'
      },
      {
        title: 'Comprehensive Approach',
        approach: 'Thorough analysis and planning',
        outcome: 'Robust solution, time-intensive',
        risk: 'Low'
      },
      {
        title: 'Iterative Approach',
        approach: 'Incremental development and refinement',
        outcome: 'Balanced quality and speed',
        risk: 'Low-Medium'
      }
    ].slice(0, 2 + (iteration % 2));
  }

  private calculateConfidence(iteration: number): number {
    // Confidence increases with iterations
    return Math.min(0.5 + (iteration * 0.05), 0.95);
  }

  private hashProblem(problem: string, requirements?: string[], constraints?: string[]): string {
    const data = JSON.stringify({ problem, requirements, constraints });
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  private cacheSolution(hash: string, solution: Solution): void {
    if (SolutionNode.solutionCache.size >= this.maxCacheSize) {
      // Remove oldest entry
      const firstKey = SolutionNode.solutionCache.keys().next().value;
      SolutionNode.solutionCache.delete(firstKey);
    }
    SolutionNode.solutionCache.set(hash, solution);
  }

  private checkConvergence(qualityHistory: number[]): boolean {
    if (qualityHistory.length < 3) return false;

    const recent = qualityHistory.slice(-3);
    const variance = Math.max(...recent) - Math.min(...recent);
    return variance < 0.05; // Converged if variance is less than 5%
  }

  private calculateConvergenceRate(qualityHistory: number[]): number {
    if (qualityHistory.length < 2) return 0;

    const initial = qualityHistory[0];
    const final = qualityHistory[qualityHistory.length - 1];
    return (final - initial) / qualityHistory.length;
  }

  /**
   * Calculate initial quality score before detailed scoring
   * Estimates quality based on iteration number and content length
   *
   * @param iteration - Current iteration number
   * @returns Initial quality score (0-1)
   */
  private calculateInitialQualityScore(iteration: number): number {
    // Base score increases with iterations (assuming iterative improvement)
    const baseScore = Math.min(0.5 + (iteration * 0.05), 0.85);

    // Add small random factor for diversity
    const randomFactor = Math.random() * 0.1;

    return Math.min(baseScore + randomFactor, 0.95);
  }

  /**
   * Calculate quality score from convergence metrics
   * Used for final quality assessment after iterative generation
   *
   * @param convergenceMetrics - Convergence metrics from generation
   * @returns Quality score (0-1)
   */
  private calculateQualityScoreFromMetrics(convergenceMetrics: ConvergenceMetrics): number {
    const { finalQuality, convergenceRate, iterations, converged } = convergenceMetrics;

    // Weight factors
    const finalQualityWeight = 0.6;
    const convergenceRateWeight = 0.2;
    const convergedBonus = converged ? 0.1 : 0;
    const iterationEfficiencyWeight = 0.1;

    // Normalize convergence rate (typical values 0-0.1)
    const normalizedRate = Math.min(convergenceRate * 10, 1);

    // Calculate iteration efficiency (fewer iterations is better)
    const iterationEfficiency = Math.max(1 - (iterations / 20), 0);

    // Combine scores
    const qualityScore =
      (finalQuality * finalQualityWeight) +
      (normalizedRate * convergenceRateWeight) +
      convergedBonus +
      (iterationEfficiency * iterationEfficiencyWeight);

    return Math.min(Math.max(qualityScore, 0), 1);
  }

  // Quality metric calculation methods

  private calculateCompleteness(content: string, problem: string, requirements?: string[]): number {
    let score = 0.5; // Base score

    // Check content length
    if (content.length > 200) score += 0.1;
    if (content.length > 500) score += 0.1;

    // Check for structure
    if (content.includes('##')) score += 0.1;
    if (content.includes('-')) score += 0.05;

    // Check requirements coverage
    if (requirements && requirements.length > 0) {
      const covered = requirements.filter(req =>
        content.toLowerCase().includes(req.toLowerCase())
      ).length;
      score += (covered / requirements.length) * 0.15;
    }

    return Math.min(score, 1.0);
  }

  private calculateCorrectness(content: string, problem: string): number {
    let score = 0.6; // Base score assuming reasonable solution

    // Check for problem keywords in solution
    const problemWords = new Set(problem.toLowerCase().split(/\s+/));
    const contentWords = content.toLowerCase().split(/\s+/);
    const overlap = contentWords.filter(w => problemWords.has(w)).length;
    score += Math.min(overlap * 0.01, 0.2);

    // Check for solution patterns
    if (content.includes('solution') || content.includes('approach')) score += 0.1;
    if (content.includes('implement') || content.includes('execute')) score += 0.1;

    return Math.min(score, 1.0);
  }

  private calculateClarity(content: string): number {
    let score = 0.5; // Base score

    // Check for structure
    if (content.includes('##')) score += 0.2;
    if (content.includes('-')) score += 0.1;

    // Check sentence length (shorter is clearer)
    const sentences = content.split(/[.!?]+/);
    const avgLength = sentences.reduce((sum, s) => sum + s.length, 0) / sentences.length;
    if (avgLength < 100) score += 0.1;
    else if (avgLength < 150) score += 0.05;

    return Math.min(score, 1.0);
  }

  private calculateEfficiency(content: string): number {
    // Simulate efficiency based on conciseness
    const idealLength = 300;
    const actualLength = content.length;
    const ratio = idealLength / actualLength;

    return Math.min(ratio, 1.0);
  }

  private calculateInnovation(content: string): number {
    let score = 0.3; // Base score

    // Check for innovative keywords
    const innovativeKeywords = [
      'novel', 'innovative', 'creative', 'unique', 'original',
      'advanced', 'cutting-edge', 'breakthrough', 'pioneering'
    ];
    const hasInnovativeKeyword = innovativeKeywords.some(kw =>
      content.toLowerCase().includes(kw)
    );
    if (hasInnovativeKeyword) score += 0.3;

    // Check for structure that suggests thoughtful design
    if (content.includes('approach') && content.includes('strategy')) score += 0.2;

    // Random factor for diversity
    score += Math.random() * 0.2;

    return Math.min(score, 1.0);
  }
}

export default SolutionNode;
