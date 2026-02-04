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

import { z } from 'zod';

/**
 * Proof System Types
 *
 * Supported formal proof systems
 */
export const ProofSystem = z.enum([
  'z3',
  'leanaide',
  'coq',
  'isabelle',
  'hol',
  'other',
]);

export type ProofSystem = z.infer<typeof ProofSystem>;

/**
 * Proof Status Types
 */
export const ProofStatus = z.enum([
  'valid',
  'invalid',
  'partial',
  'unknown',
  'pending_validation',
]);

export type ProofStatus = z.infer<typeof ProofStatus>;

/**
 * Theorem Types
 */
export const TheoremType = z.enum([
  'lemma',
  'theorem',
  'corollary',
  'proposition',
  'definition',
  'axiom',
  'conjecture',
  'other',
]);

export type TheoremType = z.infer<typeof TheoremType>;

/**
 * Dependency Types
 */
export const DependencyType = z.enum([
  'direct_dependency',
  'indirect_dependency',
  'import',
  'definition_used',
  'theorem_applied',
  'axiom_referenced',
  'other',
]);

export type DependencyType = z.infer<typeof DependencyType>;

/**
 * Theorem Schema
 *
 * Represents a mathematical theorem or statement
 */
export const Theorem = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique identifier for the theorem"),

  // The theorem statement
  statement: z.string()
    .min(1, "Theorem statement cannot be empty")
    .describe("The formal statement of the theorem"),

  // Type of theorem
  type: TheoremType.describe("Type of theorem (lemma, theorem, etc.)"),

  // Optional constraints or preconditions
  constraints: z.array(z.string()).optional()
    .describe("Constraints or preconditions for the theorem"),

  // Dependencies on other theorems/axioms
  dependencies: z.array(z.string().uuid()).optional()
    .describe("IDs of theorems or axioms this theorem depends on"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Additional metadata about the theorem"),

  // Timestamp in UTC (Law of UTC)
  created_at: z.string().datetime()
    .describe("UTC timestamp when theorem was created (ISO-8601)"),

  // Optional last updated timestamp
  updated_at: z.string().datetime().optional()
    .describe("UTC timestamp when theorem was last updated (ISO-8601)"),
});

export type Theorem = z.infer<typeof Theorem>;

/**
 * Formal Proof Schema
 *
 * Represents a complete formal proof
 */
export const FormalProof = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique identifier for the proof"),

  // Reference to the theorem being proved
  theorem_id: z.string().uuid().describe("ID of the theorem being proved"),

  // The theorem statement (denormalized for search)
  theorem: z.string()
    .min(1, "Theorem statement cannot be empty")
    .describe("The theorem statement being proved"),

  // The proof content (system-specific format)
  proof: z.string()
    .min(1, "Proof content cannot be empty")
    .describe("The formal proof content in system-specific format"),

  // Proof system that generated/verified this proof
  system: ProofSystem.describe("The proof system (z3, leanaide, etc.)"),

  // Proof status
  status: ProofStatus.describe("Validation status of the proof"),

  // Confidence score (0-1) for AI-generated proofs
  confidence: z.number()
    .min(0, "Confidence must be at least 0")
    .max(1, "Confidence must be at most 1")
    .optional()
    .describe("Confidence score for AI-generated proofs (0-1)"),

  // List of tactics/techniques used
  tactics: z.array(z.string()).optional()
    .describe("List of tactics or techniques used in the proof"),

  // Dependencies on other proofs
  dependencies: z.array(z.string().uuid()).optional()
    .describe("IDs of proofs this proof depends on"),

  // Execution metadata
  metadata: z.object({
    proof_length: z.number().int().positive().optional()
      .describe("Length of the proof in lines/steps"),
    verification_time_ms: z.number().optional()
      .describe("Time taken to verify the proof"),
    memory_used_mb: z.number().optional()
      .describe("Memory used during verification"),
    solver_version: z.string().optional()
      .describe("Version of the proof system used"),
    tactics_count: z.number().int().nonnegative().optional()
      .describe("Number of tactics used"),
  }).optional().describe("Execution metadata"),

  // Additional metadata
  custom_metadata: z.record(z.any()).optional()
    .describe("Additional custom metadata"),

  // Timestamp in UTC (Law of UTC)
  timestamp_utc: z.string().datetime()
    .describe("UTC timestamp when proof was created (ISO-8601)"),

  // Optional correlation ID for distributed tracing
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type FormalProof = z.infer<typeof FormalProof>;

/**
 * Proof Dependency Schema
 *
 * Represents a dependency relationship between proofs
 */
export const ProofDependency = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique identifier for the dependency"),

  // The proof that has the dependency
  proof_id: z.string().uuid().describe("ID of the proof that has the dependency"),

  // The proof being depended on
  depends_on_proof_id: z.string().uuid()
    .describe("ID of the proof that is being depended on"),

  // Type of dependency
  type: DependencyType.describe("Type of dependency relationship"),

  // Whether the dependency is still valid
  validity: z.boolean().describe("Whether the dependency is valid"),

  // Optional description of the dependency
  description: z.string().optional()
    .describe("Description of why this dependency exists"),

  // Timestamp in UTC
  created_at: z.string().datetime()
    .describe("UTC timestamp when dependency was created (ISO-8601)"),
});

export type ProofDependency = z.infer<typeof ProofDependency>;

/**
 * Proof Validation Schema
 *
 * Represents the validation status of a proof
 */
export const ProofValidation = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique identifier for the validation"),

  // Reference to the proof
  proof_id: z.string().uuid().describe("ID of the proof being validated"),

  // Whether the proof is valid
  is_valid: z.boolean().describe("Whether the proof is valid"),

  // Validation errors (if any)
  errors: z.array(z.object({
    line: z.number().int().positive().optional(),
    column: z.number().int().nonnegative().optional(),
    message: z.string().describe("Error message"),
    code: z.string().optional(),
  })).optional().describe("Validation errors (if any)"),

  // Validation timestamp
  validated_at: z.string().datetime()
    .describe("UTC timestamp when validation was performed (ISO-8601)"),

  // Who/what performed the validation
  validated_by: z.enum(['system', 'user', 'automated_check']).optional()
    .describe("Who or what performed the validation"),

  // Proof system version used for validation
  validator_version: z.string().optional()
    .describe("Version of the proof system used for validation"),

  // Optional correlation ID
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type ProofValidation = z.infer<typeof ProofValidation>;

/**
 * Similar Proof Schema
 *
 * Represents a proof similar to a query
 */
export const SimilarProof = z.object({
  // The similar proof
  proof: FormalProof.describe("The similar proof"),

  // Similarity score (0-1)
  similarity_score: z.number()
    .min(0, "Similarity score must be at least 0")
    .max(1, "Similarity score must be at most 1")
    .describe("Similarity score (0-1, higher is more similar)"),

  // Matching theorems or concepts
  matching_theorems: z.array(z.string()).optional()
    .describe("Theorems or concepts that match"),

  // Explanation of similarity
  explanation: z.string().optional()
    .describe("Explanation of why these proofs are similar"),
});

export type SimilarProof = z.infer<typeof SimilarProof>;

/**
 * Proof Lineage Schema
 *
 * Represents the ancestry/descendency of a proof
 */
export const ProofLineage = z.object({
  // The proof ID
  proof_id: z.string().uuid().describe("ID of the proof"),

  // Ancestor proofs (proofs this depends on)
  ancestors: z.array(z.object({
    proof_id: z.string().uuid(),
    depth: z.number().int().nonnegative(),
    relationship: z.string(),
  })).describe("Ancestor proofs in the dependency graph"),

  // Descendant proofs (proofs that depend on this)
  descendants: z.array(z.object({
    proof_id: z.string().uuid(),
    depth: z.number().int().nonnegative(),
    relationship: z.string(),
  })).describe("Descendant proofs in the dependency graph"),

  // Full dependency tree
  full_tree: z.record(z.any()).optional()
    .describe("Full dependency tree representation"),

  // Timestamp when lineage was computed
  computed_at: z.string().datetime()
    .describe("UTC timestamp when lineage was computed (ISO-8601)"),
});

export type ProofLineage = z.infer<typeof ProofLineage>;

/**
 * Proof History Schema
 *
 * Represents the history of changes to a proof
 */
export const ProofHistory = z.object({
  // The proof ID
  proof_id: z.string().uuid().describe("ID of the proof"),

  // List of historical versions
  versions: z.array(z.object({
    version_id: z.string().uuid(),
    timestamp: z.string().datetime(),
    changes: z.array(z.string()),
    proof: FormalProof.optional(),
  })).describe("Historical versions of the proof"),

  // Total number of versions
  version_count: z.number().int().nonnegative()
    .describe("Total number of versions"),

  // Current version ID
  current_version_id: z.string().uuid().optional()
    .describe("ID of the current version"),
});

export type ProofHistory = z.infer<typeof ProofHistory>;

/**
 * Storage Result Schema
 */
export const StorageResult = z.object({
  success: z.boolean(),
  proof_id: z.string().uuid().optional(),
  error: z.string().optional(),
  timestamp: z.string().datetime(),
});

export type StorageResult = z.infer<typeof StorageResult>;

/**
 * Update Result Schema
 */
export const UpdateResult = z.object({
  success: z.boolean(),
  proof_id: z.string().uuid().optional(),
  previous_version_id: z.string().uuid().optional(),
  new_version_id: z.string().uuid().optional(),
  error: z.string().optional(),
  timestamp: z.string().datetime(),
});

export type UpdateResult = z.infer<typeof UpdateResult>;

/**
 * Index Result Schema
 */
export const IndexResult = z.object({
  success: z.boolean(),
  embedding_id: z.string().optional(),
  vector_indexed: z.boolean(),
  graph_indexed: z.boolean(),
  error: z.string().optional(),
  timestamp: z.string().datetime(),
});

export type IndexResult = z.infer<typeof IndexResult>;

/**
 * Proof Metrics Schema
 */
export const ProofMetrics = z.object({
  total_proofs: z.number().int().nonnegative(),
  proofs_by_system: z.record(z.number().int().nonnegative()),
  proofs_by_status: z.record(z.number().int().nonnegative()),
  average_confidence: z.number().optional(),
  total_dependencies: z.number().int().nonnegative(),
  indexed_proofs: z.number().int().nonnegative(),
  last_updated: z.string().datetime(),
});

export type ProofMetrics = z.infer<typeof ProofMetrics>;

/**
 * Revalidation Result Schema
 */
export const RevalidationResult = z.object({
  success: z.boolean(),
  proofs_revalidated: z.number().int().nonnegative(),
  proofs_invalidated: z.number().int().nonnegative(),
  errors: z.array(z.string()).optional(),
  timestamp: z.string().datetime(),
});

export type RevalidationResult = z.infer<typeof RevalidationResult>;

/**
 * Validation Functions
 */

/**
 * Validate a Theorem against the schema
 */
export function validateTheorem(data: unknown): {
  success: boolean;
  data?: Theorem;
  errors?: string[];
} {
  const result = Theorem.safeParse(data);

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
export function validateFormalProof(data: unknown): {
  success: boolean;
  data?: FormalProof;
  errors?: string[];
} {
  const result = FormalProof.safeParse(data);

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
export function validateProofValidation(data: unknown): {
  success: boolean;
  data?: ProofValidation;
  errors?: string[];
} {
  const result = ProofValidation.safeParse(data);

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
export function transformZ3ResponseToFormalProof(
  z3Response: any,
  theoremId: string,
  correlationId?: string
): FormalProof {
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
export function transformLeanAideResponseToFormalProof(
  leanResponse: any,
  theoremId: string,
  correlationId?: string
): FormalProof {
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
export const ProofKnowledgeBaseExamples = {
  validTheorem: {
    id: "550e8400-e29b-41d4-a716-446655440000",
    statement: "For all natural numbers n, n + 0 = n",
    type: "theorem" as const,
    constraints: ["n ∈ Nat"],
    dependencies: [],
    metadata: {
      difficulty: "beginner",
      domain: "arithmetic",
    },
    created_at: "2025-02-03T12:34:56.789Z",
  } as Theorem,

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
    system: "leanaide" as const,
    status: "valid" as const,
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
  } as FormalProof,

  validProofValidation: {
    id: "550e8400-e29b-41d4-a716-446655440003",
    proof_id: "550e8400-e29b-41d4-a716-446655440001",
    is_valid: true,
    errors: [],
    validated_at: "2025-02-03T12:34:56.789Z",
    validated_by: "system" as const,
    validator_version: "4.7.0",
  } as ProofValidation,
};
