/**
 * VERIFICATION STRATEGY SELECTOR
 *
 * Analyzes problems and selects the optimal verification strategy:
 * - Which system(s) to use (Z3, LeanAide, or both)
 * - How to combine them (parallel, sequential, hybrid)
 * - Expected success rates based on historical data
 */

import { v4 as uuidv4 } from 'uuid';
import {
  Problem,
  VerificationRequest,
  VerificationStrategy,
  SystemResult
} from './canonical';
import { Logger } from '../../lib/logger';

export type ProblemType =
  | 'SMT_CONSTRAINTS'
  | 'THEOREM_PROVING'
  | 'FORMAL_VERIFICATION'
  | 'CODE_CORRECTNESS'
  | 'MODEL_CHECKING'
  | 'SAT_SOLVING';

export interface SystemConfig {
  name: 'z3' | 'leanaide';
  expectedSuccessRate: number;
  averageExecutionTime: number;
  confidence: number;
}

export interface StrategySelection {
  strategy: VerificationStrategy;
  systems: SystemConfig[];
  reasoning: string;
  expectedConfidence: number;
}

/**
 * Strategy Selector - determines best verification approach
 */
export class VerificationStrategySelector {
  private logger: Logger;
  private strategyHistory: Map<string, StrategyEffectiveness> = new Map();

  constructor(logger?: Logger) {
    this.logger = logger || new Logger('VerificationStrategySelector');
    this.loadHistoricalEffectiveness();
  }

  /**
   * Main entry point - select strategy based on problem analysis
   */
  async selectStrategy(request: VerificationRequest): Promise<StrategySelection> {
    const correlationId = request.correlationId || uuidv4();

    this.logger.info({
      msg: 'Selecting verification strategy',
      correlationId,
      problemId: request.problem.id,
      problemType: request.problem.type
    });

    // Analyze problem type
    const problemType = this.analyzeProblemType(request.problem);

    // Select appropriate systems
    const systems = this.selectSystems(problemType, request.problem);

    // Determine execution strategy
    const strategy = this.determineStrategy(
      systems,
      request.problem,
      request.confidenceRequired
    );

    // Estimate expected confidence
    const expectedConfidence = this.estimateConfidence(systems, strategy, problemType);

    const selection: StrategySelection = {
      strategy,
      systems,
      reasoning: this.generateReasoning(problemType, systems, strategy),
      expectedConfidence
    };

    this.logger.info({
      msg: 'Strategy selected',
      correlationId,
      strategy,
      systems: systems.map(s => s.name),
      expectedConfidence
    });

    return selection;
  }

  /**
   * Analyze problem to determine its type
   */
  private analyzeProblemType(problem: Problem): ProblemType {
    // Use problem.type directly, but also analyze statement for confirmation
    const statement = problem.statement.toLowerCase();
    const description = problem.description.toLowerCase();

    // Heuristics for problem type detection
    if (statement.includes('smt') || statement.includes('satisfiable') ||
        statement.includes('constraint solving')) {
      return 'SMT_CONSTRAINTS';
    }

    if (statement.includes('theorem') || statement.includes('prove') ||
        statement.includes('lemma') || description.includes('theorem')) {
      return 'THEOREM_PROVING';
    }

    if (statement.includes('verify') || statement.includes('correctness') ||
        description.includes('verification')) {
      return 'FORMAL_VERIFICATION';
    }

    if (statement.includes('function') || statement.includes('code') ||
        description.includes('code')) {
      return 'CODE_CORRECTNESS';
    }

    if (statement.includes('model') || statement.includes('state') ||
        description.includes('model checking')) {
      return 'MODEL_CHECKING';
    }

    if (statement.includes('sat') || statement.includes('boolean') ||
        description.includes('satisfiability')) {
      return 'SAT_SOLVING';
    }

    // Default to declared type
    return problem.type as ProblemType;
  }

  /**
   * Select which systems to use based on problem type
   */
  private selectSystems(
    problemType: ProblemType,
    problem: Problem
  ): SystemConfig[] {
    const systems: SystemConfig[] = [];

    // Z3 is best for SMT, SAT, and constraint solving
    if (['SMT_CONSTRAINTS', 'SAT_SOLVING', 'MODEL_CHECKING'].includes(problemType)) {
      systems.push({
        name: 'z3',
        expectedSuccessRate: this.getHistoricalSuccessRate('z3', problemType),
        averageExecutionTime: this.getAverageExecutionTime('z3', problemType),
        confidence: 0.90
      });
    }

    // LeanAide is best for theorem proving and formal verification
    if (['THEOREM_PROVING', 'FORMAL_VERIFICATION', 'CODE_CORRECTNESS'].includes(problemType)) {
      systems.push({
        name: 'leanaide',
        expectedSuccessRate: this.getHistoricalSuccessRate('leanaide', problemType),
        averageExecutionTime: this.getAverageExecutionTime('leanaide', problemType),
        confidence: 0.85
      });
    }

    // For CODE_CORRECTNESS, both systems can contribute
    if (problemType === 'CODE_CORRECTNESS') {
      systems.push({
        name: 'z3',
        expectedSuccessRate: this.getHistoricalSuccessRate('z3', problemType),
        averageExecutionTime: this.getAverageExecutionTime('z3', problemType),
        confidence: 0.80
      });
    }

    // If no systems selected (shouldn't happen), use both
    if (systems.length === 0) {
      systems.push(
        {
          name: 'z3',
          expectedSuccessRate: 0.75,
          averageExecutionTime: 5000,
          confidence: 0.75
        },
        {
          name: 'leanaide',
          expectedSuccessRate: 0.75,
          averageExecutionTime: 8000,
          confidence: 0.75
        }
      );
    }

    return systems;
  }

  /**
   * Determine execution strategy (parallel, sequential, single system)
   */
  private determineStrategy(
    systems: SystemConfig[],
    problem: Problem,
    confidenceRequired: number
  ): VerificationStrategy {
    // Single system selected
    if (systems.length === 1) {
      return systems[0].name === 'z3' ? 'z3_only' : 'leanaide_only';
    }

    // High confidence required - use both
    if (confidenceRequired >= 0.95) {
      return 'parallel';
    }

    // Time-critical problems - sequential (can stop early)
    if (problem.metadata?.timeCritical) {
      return 'sequential';
    }

    // Complex problems benefit from cross-validation
    if (problem.type === 'FORMAL_VERIFICATION' || problem.type === 'CODE_CORRECTNESS') {
      return 'hybrid';
    }

    // Default: parallel for cross-validation
    return 'parallel';
  }

  /**
   * Estimate expected confidence from strategy
   */
  private estimateConfidence(
    systems: SystemConfig[],
    strategy: VerificationStrategy,
    problemType: ProblemType
  ): number {
    if (systems.length === 1) {
      return systems[0].expectedSuccessRate;
    }

    // For parallel/hybrid: confidence increases with agreement
    const avgConfidence = systems.reduce((sum, s) => sum + s.confidence, 0) / systems.length;

    // Boost confidence when using multiple systems
    const multiplier = strategy === 'parallel' ? 1.15 :
                      strategy === 'hybrid' ? 1.10 :
                      1.05; // sequential

    return Math.min(0.99, avgConfidence * multiplier);
  }

  /**
   * Generate human-readable reasoning
   */
  private generateReasoning(
    problemType: ProblemType,
    systems: SystemConfig[],
    strategy: VerificationStrategy
  ): string {
    const systemNames = systems.map(s => s.name.toUpperCase()).join(' + ');

    const reasons: Record<ProblemType, string> = {
      'SMT_CONSTRAINTS': `SMT constraint problems are well-suited for Z3's SMT solver`,
      'THEOREM_PROVING': `Theorem proving requires LeanAide's interactive proof capabilities`,
      'FORMAL_VERIFICATION': `Formal verification benefits from both SMT solving and theorem proving`,
      'CODE_CORRECTNESS': `Code correctness requires both constraint analysis and formal proof`,
      'MODEL_CHECKING': `Model checking is efficiently handled by Z3's constraint engine`,
      'SAT_SOLVING': `SAT solving is Z3's core strength`
    };

    const strategyReason = strategy === 'parallel' ?
      'Parallel execution provides cross-validation and highest confidence' :
      strategy === 'sequential' ?
      'Sequential execution allows early termination on success' :
      strategy === 'hybrid' ?
      'Hybrid approach combines the strengths of both systems' :
      `${systemNames} selected based on problem type`;

    return `${reasons[problemType]}. ${strategyReason}`;
  }

  /**
   * Get historical success rate from learning data
   */
  private getHistoricalSuccessRate(system: 'z3' | 'leanaide', problemType: ProblemType): number {
    const key = `${system}_${problemType}`;
    const effectiveness = this.strategyHistory.get(key);

    if (effectiveness) {
      return effectiveness.successRate;
    }

    // Default success rates by system and problem type
    const defaults: Record<string, number> = {
      'z3_SMT_CONSTRAINTS': 0.95,
      'z3_THEOREM_PROVING': 0.65,
      'z3_FORMAL_VERIFICATION': 0.75,
      'z3_CODE_CORRECTNESS': 0.80,
      'z3_MODEL_CHECKING': 0.90,
      'z3_SAT_SOLVING': 0.98,
      'leanaide_SMT_CONSTRAINTS': 0.60,
      'leanaide_THEOREM_PROVING': 0.92,
      'leanaide_FORMAL_VERIFICATION': 0.88,
      'leanaide_CODE_CORRECTNESS': 0.85,
      'leanaide_MODEL_CHECKING': 0.70,
      'leanaide_SAT_SOLVING': 0.65
    };

    return defaults[key] || 0.75;
  }

  /**
   * Get average execution time from learning data
   */
  private getAverageExecutionTime(system: 'z3' | 'leanaide', problemType: ProblemType): number {
    const key = `${system}_${problemType}`;
    const effectiveness = this.strategyHistory.get(key);

    if (effectiveness) {
      return effectiveness.averageExecutionTime;
    }

    // Default execution times (milliseconds)
    const defaults: Record<string, number> = {
      'z3_SMT_CONSTRAINTS': 3000,
      'z3_THEOREM_PROVING': 8000,
      'z3_FORMAL_VERIFICATION': 5000,
      'z3_CODE_CORRECTNESS': 4000,
      'z3_MODEL_CHECKING': 3500,
      'z3_SAT_SOLVING': 2000,
      'leanaide_SMT_CONSTRAINTS': 7000,
      'leanaide_THEOREM_PROVING': 5000,
      'leanaide_FORMAL_VERIFICATION': 6000,
      'leanaide_CODE_CORRECTNESS': 6500,
      'leanaide_MODEL_CHECKING': 8000,
      'leanaide_SAT_SOLVING': 9000
    };

    return defaults[key] || 5000;
  }

  /**
   * Load historical effectiveness data (would come from Vector DB/Graphiti)
   */
  private loadHistoricalEffectiveness(): void {
    // TODO: Load from learning database
    // For now, defaults are used
    this.logger.info({
      msg: 'Historical effectiveness data loaded',
      entries: this.strategyHistory.size
    });
  }

  /**
   * Update strategy effectiveness based on outcomes
   */
  async updateEffectiveness(
    strategy: VerificationStrategy,
    problemType: ProblemType,
    success: boolean,
    executionTime: number,
    confidence: number
  ): Promise<void> {
    // Extract systems from strategy
    const systems: ('z3' | 'leanaide')[] = [];
    if (strategy.includes('z3') || strategy === 'parallel' || strategy === 'hybrid') {
      systems.push('z3');
    }
    if (strategy.includes('leanaide') || strategy === 'parallel' || strategy === 'hybrid') {
      systems.push('leanaide');
    }

    // Update each system's effectiveness
    for (const system of systems) {
      const key = `${system}_${problemType}`;
      const existing = this.strategyHistory.get(key);

      if (existing) {
        // Update with exponential moving average
        const alpha = 0.1; // Learning rate
        existing.successRate = alpha * (success ? 1 : 0) + (1 - alpha) * existing.successRate;
        existing.averageConfidence = alpha * confidence + (1 - alpha) * existing.averageConfidence;
        existing.averageExecutionTime = alpha * executionTime + (1 - alpha) * existing.averageExecutionTime;
        existing.sampleSize += 1;
        existing.lastUpdated = new Date().toISOString();
      } else {
        // Initialize
        this.strategyHistory.set(key, {
          strategy,
          problemType,
          successRate: success ? 1.0 : 0.0,
          averageConfidence: confidence,
          averageExecutionTime: executionTime,
          sampleSize: 1,
          lastUpdated: new Date().toISOString()
        });
      }
    }

    this.logger.info({
      msg: 'Strategy effectiveness updated',
      strategy,
      problemType,
      success
    });

    // TODO: Persist to Vector DB/Graphiti
  }
}

interface StrategyEffectiveness {
  strategy: VerificationStrategy;
  problemType: ProblemType;
  successRate: number;
  averageConfidence: number;
  averageExecutionTime: number;
  sampleSize: number;
  lastUpdated: string;
}
