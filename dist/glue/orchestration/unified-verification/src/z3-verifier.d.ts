/**
 * Z3 Proof Verifier
 *
 * Verifies SMT-LIB formatted proofs using Z3 SMT solver.
 */
import { CircuitBreaker } from '../../../lib/circuit-breaker';
import type { ProofVerifier, SystemResult } from './verification-service';
import type { VerificationRequest } from './canonical';
/**
 * Z3 Verifier
 */
export declare class Z3Verifier implements ProofVerifier {
    systemName: 'z3';
    circuitBreaker: CircuitBreaker;
    private apiUrl;
    private timeout;
    private loggerContext;
    constructor(apiUrl: string, timeout?: number);
    /**
     * Verify a proof with Z3
     */
    verify(request: VerificationRequest): Promise<SystemResult>;
    /**
     * Execute Z3 proof
     */
    private executeZ3;
    /**
     * Check if Z3 service is healthy
     */
    healthCheck(): Promise<boolean>;
}
//# sourceMappingURL=z3-verifier.d.ts.map