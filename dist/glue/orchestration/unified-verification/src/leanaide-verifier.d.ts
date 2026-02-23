/**
 * LeanAide Proof Verifier
 *
 * Verifies Lean proofs using LeanAide (AI-assisted theorem proving).
 */
import { CircuitBreaker } from '../../../lib/circuit-breaker';
import type { ProofVerifier, SystemResult } from './verification-service';
import type { VerificationRequest } from './canonical';
/**
 * LeanAide Verifier
 */
export declare class LeanAideVerifier implements ProofVerifier {
    systemName: 'leanaide';
    circuitBreaker: CircuitBreaker;
    private apiUrl;
    private timeout;
    private loggerContext;
    constructor(apiUrl: string, timeout?: number);
    /**
     * Verify a proof with LeanAide
     */
    verify(request: VerificationRequest): Promise<SystemResult>;
    /**
     * Execute LeanAide proof verification
     */
    private executeLeanAide;
    /**
     * Check if LeanAide service is healthy
     */
    healthCheck(): Promise<boolean>;
}
//# sourceMappingURL=leanaide-verifier.d.ts.map