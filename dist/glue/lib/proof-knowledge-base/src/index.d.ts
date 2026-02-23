/**
 * Proof Knowledge Base - Main Export
 *
 * Centralized system for storing, searching, and reusing formal proofs
 * from Z3, LeanAide, and other formal verification systems.
 *
 * Federation Constitution Compliance:
 * - Law of the "Air Gap": No imports from core-projects
 * - Law of Configuration Explicitness: All env vars validated
 * - Law of Idempotency: All operations safe to retry
 * - Law of UTC: All timestamps in UTC
 */
export { Theorem, FormalProof, ProofDependency, ProofValidation, SimilarProof, ProofLineage, ProofHistory, StorageResult, UpdateResult, IndexResult, ProofMetrics, RevalidationResult, ProofSystem, ProofStatus, TheoremType, DependencyType, validateTheorem, validateFormalProof, validateProofValidation, transformZ3ResponseToFormalProof, transformLeanAideResponseToFormalProof, ProofKnowledgeBaseExamples, } from './canonical';
export { ProofVectorIndex } from './vector-index';
export { ProofGraphIndex } from './graph-index';
export { ProofValidator } from './validator';
export { ProofKnowledgeBase } from './repository';
export declare const VERSION = "1.0.0";
/**
 * Quick start example:
 *
 * ```typescript
 * import { ProofKnowledgeBase, FormalProof, Theorem } from '@openevolve/proof-knowledge-base';
 *
 * // Create knowledge base
 * const kb = new ProofKnowledgeBase({
 *   vectorIndexEnabled: true,
 *   graphIndexEnabled: true,
 *   validationEnabled: true,
 * });
 *
 * // Store a theorem
 * const theorem: Theorem = {
 *   id: 'theorem-1',
 *   statement: 'For all natural numbers n, n + 0 = n',
 *   type: 'theorem',
 *   constraints: ['n ∈ Nat'],
 *   dependencies: [],
 *   created_at: new Date().toISOString(),
 * };
 * await kb.storeTheorem(theorem);
 *
 * // Store a proof
 * const proof: FormalProof = {
 *   id: 'proof-1',
 *   theorem_id: 'theorem-1',
 *   theorem: 'For all natural numbers n, n + 0 = n',
 *   proof: 'theorem add_zero (n : Nat) : n + 0 = n := by ...',
 *   system: 'leanaide',
 *   status: 'valid',
 *   confidence: 1.0,
 *   tactics: ['induction', 'rfl'],
 *   dependencies: [],
 *   timestamp_utc: new Date().toISOString(),
 * };
 * await kb.storeProof(proof);
 *
 * // Search for similar proofs
 * const similar = await kb.searchSimilar(theorem, 10);
 *
 * // Get proof lineage
 * const lineage = await kb.getProofLineage('proof-1', 3);
 *
 * // Get metrics
 * const metrics = await kb.getMetrics();
 * ```
 */
//# sourceMappingURL=index.d.ts.map