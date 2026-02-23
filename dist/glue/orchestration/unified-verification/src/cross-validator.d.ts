/**
 * CROSS VALIDATOR
 *
 * Executes verification across multiple systems and validates results:
 * - Parallel execution: Run both systems simultaneously
 * - Sequential execution: Run one, then the other if needed
 * - Compare results and detect disagreements
 * - Resolve conflicts based on confidence and evidence
 */
import { VerificationRequest, CrossValidationResult } from './canonical';
import { Logger } from '../../lib/logger';
/**
 * Cross Validator - orchestrates multi-system verification
 */
export declare class CrossValidator {
    private logger;
    private systems;
    constructor(z3Url: string, leanaideUrl: string, logger?: Logger);
    /**
     * Main entry point - validate with cross-checking
     */
    validate(request: VerificationRequest): Promise<CrossValidationResult>;
    /**
     * Execute verification based on strategy
     */
    private executeVerification;
    /**
     * Execute Z3 verification
     */
    private executeZ3;
    /**
     * Execute LeanAide verification
     */
    private executeLeanAide;
    /**
     * Execute both systems in parallel
     */
    private executeParallel;
    /**
     * Execute systems sequentially (stop if first succeeds)
     */
    private executeSequential;
    /**
     * Execute hybrid approach (Z3 first, then LeanAide for proof refinement)
     */
    private executeHybrid;
    /**
     * Compare results from multiple systems
     */
    private compareResults;
    /**
     * Detect disagreements between results
     */
    private detectDisagreements;
    /**
     * Determine final resolution based on results and conflicts
     */
    private determineResolution;
    /**
     * Calculate aggregate confidence from multiple results
     */
    private calculateAggregateConfidence;
    /**
     * Infer strategy from request if not specified
     */
    private inferStrategy;
    /**
     * Convert SystemResult[] to VerificationResult[]
     */
    private toVerificationResults;
    /**
     * Generate human-readable comparison details
     */
    private generateComparisonDetails;
    /**
     * Check system health (circuit breaker prerequisite)
     */
    private checkSystemHealth;
}
//# sourceMappingURL=cross-validator.d.ts.map