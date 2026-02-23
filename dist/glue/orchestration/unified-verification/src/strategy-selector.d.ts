/**
 * VERIFICATION STRATEGY SELECTOR
 *
 * Analyzes problems and selects the optimal verification strategy:
 * - Which system(s) to use (Z3, LeanAide, or both)
 * - How to combine them (parallel, sequential, hybrid)
 * - Expected success rates based on historical data
 */
import { VerificationRequest, VerificationStrategy } from './canonical';
import { Logger } from '../../lib/logger';
export type ProblemType = 'SMT_CONSTRAINTS' | 'THEOREM_PROVING' | 'FORMAL_VERIFICATION' | 'CODE_CORRECTNESS' | 'MODEL_CHECKING' | 'SAT_SOLVING';
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
export declare class VerificationStrategySelector {
    private logger;
    private strategyHistory;
    constructor(logger?: Logger);
    /**
     * Main entry point - select strategy based on problem analysis
     */
    selectStrategy(request: VerificationRequest): Promise<StrategySelection>;
    /**
     * Analyze problem to determine its type
     */
    private analyzeProblemType;
    /**
     * Select which systems to use based on problem type
     */
    private selectSystems;
    /**
     * Determine execution strategy (parallel, sequential, single system)
     */
    private determineStrategy;
    /**
     * Estimate expected confidence from strategy
     */
    private estimateConfidence;
    /**
     * Generate human-readable reasoning
     */
    private generateReasoning;
    /**
     * Get historical success rate from learning data
     */
    private getHistoricalSuccessRate;
    /**
     * Get average execution time from learning data
     */
    private getAverageExecutionTime;
    /**
     * Load historical effectiveness data (would come from Vector DB/Graphiti)
     */
    private loadHistoricalEffectiveness;
    /**
     * Update strategy effectiveness based on outcomes
     */
    updateEffectiveness(strategy: VerificationStrategy, problemType: ProblemType, success: boolean, executionTime: number, confidence: number): Promise<void>;
}
//# sourceMappingURL=strategy-selector.d.ts.map