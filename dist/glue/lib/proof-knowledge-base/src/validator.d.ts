/**
 * Proof Validator
 *
 * Validates formal proofs and checks dependencies.
 * Integrates with Z3 and LeanAide for proof verification.
 *
 * Federation Constitution Compliance:
 * - Law of Idempotency: Safe to re-validate
 * - Law of UTC: All timestamps in UTC
 * - Circuit Breaker: Prevents cascading failures
 * - Retry Logic: Handles transient failures
 */
import { FormalProof, ProofValidation, RevalidationResult } from './canonical';
/**
 * Proof Validator Configuration
 */
interface ProofValidatorConfig {
    z3ApiUrl?: string;
    leanaideApiUrl?: string;
    timeoutMs: number;
    maxRetries: number;
}
/**
 * Proof Validator
 *
 * Validates formal proofs and manages dependency checking
 */
export declare class ProofValidator {
    private config;
    private z3CircuitBreaker;
    private leanaideCircuitBreaker;
    constructor(config?: Partial<ProofValidatorConfig>);
    /**
     * Validate a proof
     *
     * Executes the proof in its native system and returns validation result
     *
     * @param proofId - ID of the proof to validate
     * @param revalidateDependencies - Whether to revalidate dependencies
     * @param correlationId - Optional correlation ID for tracing
     * @returns Validation result
     */
    validateProof(proofId: string, proof: FormalProof, revalidateDependencies?: boolean, correlationId?: string): Promise<ProofValidation>;
    /**
     * Batch validate multiple proofs
     *
     * @param proofs - Array of proofs with their IDs
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of validation results
     */
    batchValidate(proofs: Array<{
        proofId: string;
        proof: FormalProof;
    }>, correlationId?: string): Promise<ProofValidation[]>;
    /**
     * Check validation status of a single dependency
     *
     * @param dependencyId - ID of dependency to check
     * @param correlationId - Optional correlation ID
     * @returns Whether dependency is currently valid
     */
    private checkDependencyValidationStatus;
    /**
     * Check if dependencies are valid
     *
     * @param proofId - ID of the proof
     * @param proof - The proof with dependencies
     * @param correlationId - Optional correlation ID for tracing
     * @returns Whether all dependencies are valid
     */
    checkDependenciesValid(proofId: string, proof: FormalProof, correlationId?: string): Promise<boolean>;
    /**
     * Revalidate proofs when a dependency changes
     *
     * When a proof is updated, all proofs that depend on it must be revalidated
     *
     * @param changedProofId - ID of the proof that changed
     * @param dependentProofIds - IDs of proofs that depend on the changed proof
     * @param proofs - Map of proof IDs to proofs
     * @param correlationId - Optional correlation ID for tracing
     * @returns Revalidation result
     */
    revalidateOnDependencyChange(changedProofId: string, dependentProofIds: string[], proofs: Map<string, FormalProof>, correlationId?: string): Promise<RevalidationResult>;
    /**
     * Execute a proof in its native system
     *
     * @param proof - The proof to execute
     * @param correlationId - Optional correlation ID for tracing
     * @returns Execution result
     */
    private executeProof;
    /**
     * Execute a Z3 proof
     *
     * @param proof - The proof to execute
     * @param correlationId - Optional correlation ID for tracing
     * @returns Execution result
     */
    private executeZ3Proof;
    /**
     * Execute a LeanAide proof
     *
     * @param proof - The proof to execute
     * @param correlationId - Optional correlation ID for tracing
     * @returns Execution result
     */
    private executeLeanAideProof;
    /**
     * Generate a UUID v4
     */
    private generateId;
    /**
     * Update proof validation in storage
     *
     * @param proofId - ID of the proof
     * @param validation - Validation result
     */
    private updateProofValidation;
}
export {};
/**
 * Example usage:
 *
 * ```typescript
 * import { ProofValidator } from './validator';
 * import { FormalProof } from './canonical';
 *
 * // Create validator
 * const validator = new ProofValidator({
 *   z3ApiUrl: 'http://z3-core:8000',
 *   leanaideApiUrl: 'http://leanaide-core:8000',
 *   timeoutMs: 30000,
 *   maxRetries: 3,
 * });
 *
 * // Validate a single proof
 * const proof: FormalProof = { ... };
 * const validation = await validator.validateProof(
 *   proof.id,
 *   proof,
 *   true, // revalidate dependencies
 *   'correlation-123'
 * );
 *
 * // Batch validate
 * const validations = await validator.batchValidate([
 *   { proofId: proof1.id, proof: proof1 },
 *   { proofId: proof2.id, proof: proof2 },
 * ], 'correlation-123');
 *
 * // Revalidate on dependency change
 * const result = await validator.revalidateOnDependencyChange(
 *   changedProofId,
 *   dependentProofIds,
 *   proofsMap,
 *   'correlation-123'
 * );
 * ```
 */
//# sourceMappingURL=validator.d.ts.map