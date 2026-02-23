"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.VERSION = exports.ProofKnowledgeBase = exports.ProofValidator = exports.ProofGraphIndex = exports.ProofVectorIndex = exports.ProofKnowledgeBaseExamples = exports.transformLeanAideResponseToFormalProof = exports.transformZ3ResponseToFormalProof = exports.validateProofValidation = exports.validateFormalProof = exports.validateTheorem = exports.DependencyType = exports.TheoremType = exports.ProofStatus = exports.ProofSystem = exports.RevalidationResult = exports.ProofMetrics = exports.IndexResult = exports.UpdateResult = exports.StorageResult = exports.ProofHistory = exports.ProofLineage = exports.SimilarProof = exports.ProofValidation = exports.ProofDependency = exports.FormalProof = exports.Theorem = void 0;
// Export canonical schemas
var canonical_1 = require("./canonical");
// Schemas
Object.defineProperty(exports, "Theorem", { enumerable: true, get: function () { return canonical_1.Theorem; } });
Object.defineProperty(exports, "FormalProof", { enumerable: true, get: function () { return canonical_1.FormalProof; } });
Object.defineProperty(exports, "ProofDependency", { enumerable: true, get: function () { return canonical_1.ProofDependency; } });
Object.defineProperty(exports, "ProofValidation", { enumerable: true, get: function () { return canonical_1.ProofValidation; } });
Object.defineProperty(exports, "SimilarProof", { enumerable: true, get: function () { return canonical_1.SimilarProof; } });
Object.defineProperty(exports, "ProofLineage", { enumerable: true, get: function () { return canonical_1.ProofLineage; } });
Object.defineProperty(exports, "ProofHistory", { enumerable: true, get: function () { return canonical_1.ProofHistory; } });
Object.defineProperty(exports, "StorageResult", { enumerable: true, get: function () { return canonical_1.StorageResult; } });
Object.defineProperty(exports, "UpdateResult", { enumerable: true, get: function () { return canonical_1.UpdateResult; } });
Object.defineProperty(exports, "IndexResult", { enumerable: true, get: function () { return canonical_1.IndexResult; } });
Object.defineProperty(exports, "ProofMetrics", { enumerable: true, get: function () { return canonical_1.ProofMetrics; } });
Object.defineProperty(exports, "RevalidationResult", { enumerable: true, get: function () { return canonical_1.RevalidationResult; } });
// Types
Object.defineProperty(exports, "ProofSystem", { enumerable: true, get: function () { return canonical_1.ProofSystem; } });
Object.defineProperty(exports, "ProofStatus", { enumerable: true, get: function () { return canonical_1.ProofStatus; } });
Object.defineProperty(exports, "TheoremType", { enumerable: true, get: function () { return canonical_1.TheoremType; } });
Object.defineProperty(exports, "DependencyType", { enumerable: true, get: function () { return canonical_1.DependencyType; } });
// Validation functions
Object.defineProperty(exports, "validateTheorem", { enumerable: true, get: function () { return canonical_1.validateTheorem; } });
Object.defineProperty(exports, "validateFormalProof", { enumerable: true, get: function () { return canonical_1.validateFormalProof; } });
Object.defineProperty(exports, "validateProofValidation", { enumerable: true, get: function () { return canonical_1.validateProofValidation; } });
// Transformation functions
Object.defineProperty(exports, "transformZ3ResponseToFormalProof", { enumerable: true, get: function () { return canonical_1.transformZ3ResponseToFormalProof; } });
Object.defineProperty(exports, "transformLeanAideResponseToFormalProof", { enumerable: true, get: function () { return canonical_1.transformLeanAideResponseToFormalProof; } });
// Examples
Object.defineProperty(exports, "ProofKnowledgeBaseExamples", { enumerable: true, get: function () { return canonical_1.ProofKnowledgeBaseExamples; } });
// Export vector index
var vector_index_1 = require("./vector-index");
Object.defineProperty(exports, "ProofVectorIndex", { enumerable: true, get: function () { return vector_index_1.ProofVectorIndex; } });
// Export graph index
var graph_index_1 = require("./graph-index");
Object.defineProperty(exports, "ProofGraphIndex", { enumerable: true, get: function () { return graph_index_1.ProofGraphIndex; } });
// Export validator
var validator_1 = require("./validator");
Object.defineProperty(exports, "ProofValidator", { enumerable: true, get: function () { return validator_1.ProofValidator; } });
// Export main repository
var repository_1 = require("./repository");
Object.defineProperty(exports, "ProofKnowledgeBase", { enumerable: true, get: function () { return repository_1.ProofKnowledgeBase; } });
// Version
exports.VERSION = '1.0.0';
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
//# sourceMappingURL=index.js.map