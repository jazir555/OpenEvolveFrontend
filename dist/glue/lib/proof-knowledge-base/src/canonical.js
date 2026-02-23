"use strict";
/**
 * Proof Knowledge Base Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for formal proofs from Z3 and LeanAide.
 * All proof storage systems must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for proof data in the glue layer.
 * Do not pass raw proof API responses between services.
 *
 * Federation Constitution Compliance:
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Idempotency: All operations are safe to retry
 * - Law of Configuration Explicitness: No magic defaults
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProofKnowledgeBaseExamples = exports.RevalidationResult = exports.ProofMetrics = exports.IndexResult = exports.UpdateResult = exports.StorageResult = exports.ProofHistory = exports.ProofLineage = exports.SimilarProof = exports.ProofValidation = exports.ProofDependency = exports.FormalProof = exports.Theorem = exports.DependencyType = exports.TheoremType = exports.ProofStatus = exports.ProofSystem = void 0;
exports.validateTheorem = validateTheorem;
exports.validateFormalProof = validateFormalProof;
exports.validateProofValidation = validateProofValidation;
exports.transformZ3ResponseToFormalProof = transformZ3ResponseToFormalProof;
exports.transformLeanAideResponseToFormalProof = transformLeanAideResponseToFormalProof;
const zod_1 = require("zod");
/**
 * Proof System Types
 *
 * Supported formal proof systems
 */
exports.ProofSystem = zod_1.z.enum([
    'z3',
    'leanaide',
    'coq',
    'isabelle',
    'hol',
    'other',
]);
/**
 * Proof Status Types
 */
exports.ProofStatus = zod_1.z.enum([
    'valid',
    'invalid',
    'partial',
    'unknown',
    'pending_validation',
]);
/**
 * Theorem Types
 */
exports.TheoremType = zod_1.z.enum([
    'lemma',
    'theorem',
    'corollary',
    'proposition',
    'definition',
    'axiom',
    'conjecture',
    'other',
]);
/**
 * Dependency Types
 */
exports.DependencyType = zod_1.z.enum([
    'direct_dependency',
    'indirect_dependency',
    'import',
    'definition_used',
    'theorem_applied',
    'axiom_referenced',
    'other',
]);
/**
 * Theorem Schema
 *
 * Represents a mathematical theorem or statement
 */
exports.Theorem = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique identifier for the theorem"),
    // The theorem statement
    statement: zod_1.z.string()
        .min(1, "Theorem statement cannot be empty")
        .describe("The formal statement of the theorem"),
    // Type of theorem
    type: exports.TheoremType.describe("Type of theorem (lemma, theorem, etc.)"),
    // Optional constraints or preconditions
    constraints: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Constraints or preconditions for the theorem"),
    // Dependencies on other theorems/axioms
    dependencies: zod_1.z.array(zod_1.z.string().uuid()).optional()
        .describe("IDs of theorems or axioms this theorem depends on"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional metadata about the theorem"),
    // Timestamp in UTC (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when theorem was created (ISO-8601)"),
    // Optional last updated timestamp
    updated_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when theorem was last updated (ISO-8601)"),
});
/**
 * Formal Proof Schema
 *
 * Represents a complete formal proof
 */
exports.FormalProof = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique identifier for the proof"),
    // Reference to the theorem being proved
    theorem_id: zod_1.z.string().uuid().describe("ID of the theorem being proved"),
    // The theorem statement (denormalized for search)
    theorem: zod_1.z.string()
        .min(1, "Theorem statement cannot be empty")
        .describe("The theorem statement being proved"),
    // The proof content (system-specific format)
    proof: zod_1.z.string()
        .min(1, "Proof content cannot be empty")
        .describe("The formal proof content in system-specific format"),
    // Proof system that generated/verified this proof
    system: exports.ProofSystem.describe("The proof system (z3, leanaide, etc.)"),
    // Proof status
    status: exports.ProofStatus.describe("Validation status of the proof"),
    // Confidence score (0-1) for AI-generated proofs
    confidence: zod_1.z.number()
        .min(0, "Confidence must be at least 0")
        .max(1, "Confidence must be at most 1")
        .optional()
        .describe("Confidence score for AI-generated proofs (0-1)"),
    // List of tactics/techniques used
    tactics: zod_1.z.array(zod_1.z.string()).optional()
        .describe("List of tactics or techniques used in the proof"),
    // Dependencies on other proofs
    dependencies: zod_1.z.array(zod_1.z.string().uuid()).optional()
        .describe("IDs of proofs this proof depends on"),
    // Execution metadata
    metadata: zod_1.z.object({
        proof_length: zod_1.z.number().int().positive().optional()
            .describe("Length of the proof in lines/steps"),
        verification_time_ms: zod_1.z.number().optional()
            .describe("Time taken to verify the proof"),
        memory_used_mb: zod_1.z.number().optional()
            .describe("Memory used during verification"),
        solver_version: zod_1.z.string().optional()
            .describe("Version of the proof system used"),
        tactics_count: zod_1.z.number().int().nonnegative().optional()
            .describe("Number of tactics used"),
    }).optional().describe("Execution metadata"),
    // Additional metadata
    custom_metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional custom metadata"),
    // Timestamp in UTC (Law of UTC)
    timestamp_utc: zod_1.z.string().datetime()
        .describe("UTC timestamp when proof was created (ISO-8601)"),
    // Optional correlation ID for distributed tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Proof Dependency Schema
 *
 * Represents a dependency relationship between proofs
 */
exports.ProofDependency = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique identifier for the dependency"),
    // The proof that has the dependency
    proof_id: zod_1.z.string().uuid().describe("ID of the proof that has the dependency"),
    // The proof being depended on
    depends_on_proof_id: zod_1.z.string().uuid()
        .describe("ID of the proof that is being depended on"),
    // Type of dependency
    type: exports.DependencyType.describe("Type of dependency relationship"),
    // Whether the dependency is still valid
    validity: zod_1.z.boolean().describe("Whether the dependency is valid"),
    // Optional description of the dependency
    description: zod_1.z.string().optional()
        .describe("Description of why this dependency exists"),
    // Timestamp in UTC
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when dependency was created (ISO-8601)"),
});
/**
 * Proof Validation Schema
 *
 * Represents the validation status of a proof
 */
exports.ProofValidation = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique identifier for the validation"),
    // Reference to the proof
    proof_id: zod_1.z.string().uuid().describe("ID of the proof being validated"),
    // Whether the proof is valid
    is_valid: zod_1.z.boolean().describe("Whether the proof is valid"),
    // Validation errors (if any)
    errors: zod_1.z.array(zod_1.z.object({
        line: zod_1.z.number().int().positive().optional(),
        column: zod_1.z.number().int().nonnegative().optional(),
        message: zod_1.z.string().describe("Error message"),
        code: zod_1.z.string().optional(),
    })).optional().describe("Validation errors (if any)"),
    // Validation timestamp
    validated_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when validation was performed (ISO-8601)"),
    // Who/what performed the validation
    validated_by: zod_1.z.enum(['system', 'user', 'automated_check']).optional()
        .describe("Who or what performed the validation"),
    // Proof system version used for validation
    validator_version: zod_1.z.string().optional()
        .describe("Version of the proof system used for validation"),
    // Optional correlation ID
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Similar Proof Schema
 *
 * Represents a proof similar to a query
 */
exports.SimilarProof = zod_1.z.object({
    // The similar proof
    proof: exports.FormalProof.describe("The similar proof"),
    // Similarity score (0-1)
    similarity_score: zod_1.z.number()
        .min(0, "Similarity score must be at least 0")
        .max(1, "Similarity score must be at most 1")
        .describe("Similarity score (0-1, higher is more similar)"),
    // Matching theorems or concepts
    matching_theorems: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Theorems or concepts that match"),
    // Explanation of similarity
    explanation: zod_1.z.string().optional()
        .describe("Explanation of why these proofs are similar"),
});
/**
 * Proof Lineage Schema
 *
 * Represents the ancestry/descendency of a proof
 */
exports.ProofLineage = zod_1.z.object({
    // The proof ID
    proof_id: zod_1.z.string().uuid().describe("ID of the proof"),
    // Ancestor proofs (proofs this depends on)
    ancestors: zod_1.z.array(zod_1.z.object({
        proof_id: zod_1.z.string().uuid(),
        depth: zod_1.z.number().int().nonnegative(),
        relationship: zod_1.z.string(),
    })).describe("Ancestor proofs in the dependency graph"),
    // Descendant proofs (proofs that depend on this)
    descendants: zod_1.z.array(zod_1.z.object({
        proof_id: zod_1.z.string().uuid(),
        depth: zod_1.z.number().int().nonnegative(),
        relationship: zod_1.z.string(),
    })).describe("Descendant proofs in the dependency graph"),
    // Full dependency tree
    full_tree: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Full dependency tree representation"),
    // Timestamp when lineage was computed
    computed_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when lineage was computed (ISO-8601)"),
});
/**
 * Proof History Schema
 *
 * Represents the history of changes to a proof
 */
exports.ProofHistory = zod_1.z.object({
    // The proof ID
    proof_id: zod_1.z.string().uuid().describe("ID of the proof"),
    // List of historical versions
    versions: zod_1.z.array(zod_1.z.object({
        version_id: zod_1.z.string().uuid(),
        timestamp: zod_1.z.string().datetime(),
        changes: zod_1.z.array(zod_1.z.string()),
        proof: exports.FormalProof.optional(),
    })).describe("Historical versions of the proof"),
    // Total number of versions
    version_count: zod_1.z.number().int().nonnegative()
        .describe("Total number of versions"),
    // Current version ID
    current_version_id: zod_1.z.string().uuid().optional()
        .describe("ID of the current version"),
});
/**
 * Storage Result Schema
 */
exports.StorageResult = zod_1.z.object({
    success: zod_1.z.boolean(),
    proof_id: zod_1.z.string().uuid().optional(),
    error: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Update Result Schema
 */
exports.UpdateResult = zod_1.z.object({
    success: zod_1.z.boolean(),
    proof_id: zod_1.z.string().uuid().optional(),
    previous_version_id: zod_1.z.string().uuid().optional(),
    new_version_id: zod_1.z.string().uuid().optional(),
    error: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Index Result Schema
 */
exports.IndexResult = zod_1.z.object({
    success: zod_1.z.boolean(),
    embedding_id: zod_1.z.string().optional(),
    vector_indexed: zod_1.z.boolean(),
    graph_indexed: zod_1.z.boolean(),
    error: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Proof Metrics Schema
 */
exports.ProofMetrics = zod_1.z.object({
    total_proofs: zod_1.z.number().int().nonnegative(),
    proofs_by_system: zod_1.z.record(zod_1.z.number().int().nonnegative()),
    proofs_by_status: zod_1.z.record(zod_1.z.number().int().nonnegative()),
    average_confidence: zod_1.z.number().optional(),
    total_dependencies: zod_1.z.number().int().nonnegative(),
    indexed_proofs: zod_1.z.number().int().nonnegative(),
    last_updated: zod_1.z.string().datetime(),
});
/**
 * Revalidation Result Schema
 */
exports.RevalidationResult = zod_1.z.object({
    success: zod_1.z.boolean(),
    proofs_revalidated: zod_1.z.number().int().nonnegative(),
    proofs_invalidated: zod_1.z.number().int().nonnegative(),
    errors: zod_1.z.array(zod_1.z.string()).optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Validation Functions
 */
/**
 * Validate a Theorem against the schema
 */
function validateTheorem(data) {
    const result = exports.Theorem.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate a FormalProof against the schema
 */
function validateFormalProof(data) {
    const result = exports.FormalProof.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate a ProofValidation against the schema
 */
function validateProofValidation(data) {
    const result = exports.ProofValidation.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Transformation Functions
 */
/**
 * Transform Z3 SolverResponse to FormalProof
 */
function transformZ3ResponseToFormalProof(z3Response, theoremId, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        id: theoremId,
        theorem_id: theoremId,
        theorem: z3Response.problem || 'Unknown theorem',
        proof: z3Response.proof || '',
        system: 'z3',
        status: z3Response.result === 'unsat' ? 'valid' : 'unknown',
        confidence: z3Response.result === 'unsat' ? 1.0 : 0.5,
        tactics: z3Response.metadata?.tactics_applied || [],
        dependencies: [],
        metadata: {
            verification_time_ms: z3Response.metadata?.solve_time_ms,
            memory_used_mb: z3Response.metadata?.memory_used_mb,
            solver_version: z3Response.metadata?.solver_version,
        },
        timestamp_utc: timestamp,
        correlation_id: correlationId,
    };
}
/**
 * Transform LeanAide ProofVerificationResponse to FormalProof
 */
function transformLeanAideResponseToFormalProof(leanResponse, theoremId, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        id: theoremId,
        theorem_id: theoremId,
        theorem: leanResponse.theorem || 'Unknown theorem',
        proof: leanResponse.proof_code || '',
        system: 'leanaide',
        status: leanResponse.verified ? 'valid' : 'invalid',
        confidence: leanResponse.verified ? 1.0 : 0.0,
        tactics: leanResponse.tactics_used || [],
        dependencies: [],
        metadata: {
            proof_length: leanResponse.proof_code?.split('\n').length,
            verification_time_ms: leanResponse.metadata?.verification_time_ms,
            memory_used_mb: leanResponse.metadata?.memory_used_mb,
            solver_version: leanResponse.metadata?.lean_version,
            tactics_count: leanResponse.metadata?.tactics_count,
        },
        timestamp_utc: timestamp,
        correlation_id: correlationId,
    };
}
/**
 * Example usage
 */
exports.ProofKnowledgeBaseExamples = {
    validTheorem: {
        id: "550e8400-e29b-41d4-a716-446655440000",
        statement: "For all natural numbers n, n + 0 = n",
        type: "theorem",
        constraints: ["n ∈ Nat"],
        dependencies: [],
        metadata: {
            difficulty: "beginner",
            domain: "arithmetic",
        },
        created_at: "2025-02-03T12:34:56.789Z",
    },
    validFormalProof: {
        id: "550e8400-e29b-41d4-a716-446655440001",
        theorem_id: "550e8400-e29b-41d4-a716-446655440000",
        theorem: "For all natural numbers n, n + 0 = n",
        proof: `
theorem add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [add_succ, ih]
    `,
        system: "leanaide",
        status: "valid",
        confidence: 1.0,
        tactics: ["induction", "rfl", "rw"],
        dependencies: [],
        metadata: {
            proof_length: 5,
            verification_time_ms: 125,
            solver_version: "4.7.0",
            tactics_count: 3,
        },
        timestamp_utc: "2025-02-03T12:34:56.789Z",
        correlation_id: "550e8400-e29b-41d4-a716-446655440002",
    },
    validProofValidation: {
        id: "550e8400-e29b-41d4-a716-446655440003",
        proof_id: "550e8400-e29b-41d4-a716-446655440001",
        is_valid: true,
        errors: [],
        validated_at: "2025-02-03T12:34:56.789Z",
        validated_by: "system",
        validator_version: "4.7.0",
    },
};
//# sourceMappingURL=canonical.js.map