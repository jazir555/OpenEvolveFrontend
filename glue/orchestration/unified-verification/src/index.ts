/**
 * Unified Verification Module
 *
 * Main entry point for unified proof verification functionality.
 * Exports verification service, verifiers, and canonical schemas.
 */

export { UnifiedVerificationService } from './verification-service';
export { Z3Verifier } from './z3-verifier';
export { LeanAideVerifier } from './leanaide-verifier';
export type { ProofVerifier } from './verification-service';
export * from './canonical';
