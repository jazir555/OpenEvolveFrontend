/**
 * Unified Verification Service
 *
 * Federation Constitution - Unified Verification System (ADR-007)
 *
 * Orchestrates proof verification across multiple systems (Z3, LeanAide, etc.)
 * using canonical proof format.
 *
 * Features:
 * - Single API for all proof systems
 * - Cross-validation between systems
 * - Dependency tracking and revalidation
 * - Circuit breaker protection per system
 * - Retry with exponential backoff
 */
import { CircuitBreaker } from '../../lib/circuit-breaker';
import type { VerificationRequest, VerificationOptions, CrossValidationResult, SystemResult } from './canonical';
/**
 * Proof verifier interface
 * All system verifiers must implement this
 */
export interface ProofVerifier {
    systemName: 'z3' | 'leanaide' | 'lean' | 'coq';
    circuitBreaker?: CircuitBreaker;
    verify(request: VerificationRequest): Promise<SystemResult>;
    healthCheck(): Promise<boolean>;
}
/**
 * Unified Verification Service
 */
export declare class UnifiedVerificationService {
    private verifiers;
    private loggerContext;
    constructor();
    /**
     * Register a proof verifier
     */
    registerVerifier(verifier: ProofVerifier): void;
    /**
     * Verify a proof using specified strategy
     */
    verifyProof(request: VerificationRequest, options?: VerificationOptions): Promise<CrossValidationResult>;
    /**
     * Batch verify multiple proofs
     */
    batchVerify(requests: VerificationRequest[], options?: VerificationOptions): Promise<CrossValidationResult[]>;
    /**
     * Revalidate proofs when a dependency changes
     */
    revalidateOnDependencyChange(changedProofId: string, dependentProofs: VerificationRequest[], options?: VerificationOptions): Promise<CrossValidationResult[]>;
    /**
     * Execute verification strategy
     */
    private executeStrategy;
    /**
     * Verify with a single system
     */
    private verifyWithSystem;
    /**
     * Verify in parallel (all systems simultaneously)
     */
    private verifyInParallel;
    /**
     * Verify sequentially (systems one after another)
     */
    private verifySequentially;
    /**
     * Hybrid verification: Z3 first, then LeanAide if needed
     */
    private verifyHybrid;
    /**
     * Cross-validate results from multiple systems
     */
    private crossValidate;
    /**
     * Calculate combined confidence score
     */
    private calculateConfidence;
    /**
     * Get system weight for confidence calculation
     */
    private getSystemWeight;
    /**
     * Check if confidence scores align across systems
     */
    private confidencesAlign;
    /**
     * Detect conflicts between system results
     */
    private detectConflicts;
    /**
     * Determine resolution when systems disagree
     */
    private determineResolution;
    /**
     * Determine best verification strategy for a given problem
     */
    private determineStrategy;
    /**
     * Format system result for API response
     */
    private formatResult;
}
//# sourceMappingURL=verification-service.d.ts.map