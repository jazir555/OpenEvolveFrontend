/**
 * UNIFIED VERIFICATION ORCHESTRATOR
 *
 * Main entry point for unified formal verification:
 * - Coordinates strategy selection, execution, and cross-validation
 * - Provides simple API for verification requests
 * - Handles storage and learning feedback loops
 * - Follows idempotency and timeout laws
 */
import { Problem, Constraints, VerificationOptions, VerificationResult, CrossValidationResult } from './canonical';
import { Logger } from '../../lib/logger';
/**
 * Main Orchestrator - coordinates all verification components
 */
export declare class UnifiedVerificationOrchestrator {
    private logger;
    private strategySelector;
    private crossValidator;
    private confidenceAggregator;
    private resultHistory;
    constructor(z3Url: string, leanaideUrl: string, logger?: Logger);
    /**
     * Simple verification - single system or basic approach
     */
    verify(problem: Problem, constraints: Constraints, options?: VerificationOptions): Promise<VerificationResult>;
    /**
     * Verification with cross-validation - multiple systems
     */
    verifyWithCrossValidation(problem: Problem, options?: VerificationOptions): Promise<CrossValidationResult>;
    /**
     * Batch verification - multiple problems
     */
    verifyBatch(problems: Problem[], constraints: Constraints, options?: VerificationOptions): Promise<Map<string, VerificationResult>>;
    /**
     * Store verification result for learning
     */
    private storeResults;
    /**
     * Store cross-validation result for learning
     */
    private storeCrossValidationResults;
    /**
     * Learn from verification outcomes to improve strategy selection
     */
    private learnFromOutcome;
    /**
     * Get statistics about verification performance
     */
    getStatistics(): Promise<{
        totalVerifications: number;
        successRate: number;
        averageConfidence: number;
        averageExecutionTime: number;
        systemBreakdown: {
            z3: {
                count: number;
                successRate: number;
            };
            leanaide: {
                count: number;
                successRate: number;
            };
        };
    }>;
    /**
     * Health check for the orchestrator
     */
    healthCheck(): Promise<{
        healthy: boolean;
        components: {
            orchestrator: boolean;
            strategySelector: boolean;
            crossValidator: boolean;
            confidenceAggregator: boolean;
        };
    }>;
    /**
     * Helper: Convert CrossValidationResult to SystemResult[]
     */
    private crossValidatorToSystemResults;
}
//# sourceMappingURL=orchestrator.d.ts.map