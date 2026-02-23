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
export declare const ProofSystem: z.ZodEnum<["z3", "leanaide", "coq", "isabelle", "hol", "other"]>;
export type ProofSystem = z.infer<typeof ProofSystem>;
/**
 * Proof Status Types
 */
export declare const ProofStatus: z.ZodEnum<["valid", "invalid", "partial", "unknown", "pending_validation"]>;
export type ProofStatus = z.infer<typeof ProofStatus>;
/**
 * Theorem Types
 */
export declare const TheoremType: z.ZodEnum<["lemma", "theorem", "corollary", "proposition", "definition", "axiom", "conjecture", "other"]>;
export type TheoremType = z.infer<typeof TheoremType>;
/**
 * Dependency Types
 */
export declare const DependencyType: z.ZodEnum<["direct_dependency", "indirect_dependency", "import", "definition_used", "theorem_applied", "axiom_referenced", "other"]>;
export type DependencyType = z.infer<typeof DependencyType>;
/**
 * Theorem Schema
 *
 * Represents a mathematical theorem or statement
 */
export declare const Theorem: z.ZodObject<{
    id: z.ZodString;
    statement: z.ZodString;
    type: z.ZodEnum<["lemma", "theorem", "corollary", "proposition", "definition", "axiom", "conjecture", "other"]>;
    constraints: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
    updated_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: "theorem" | "definition" | "other" | "lemma" | "corollary" | "proposition" | "axiom" | "conjecture";
    id: string;
    created_at: string;
    statement: string;
    metadata?: Record<string, any> | undefined;
    dependencies?: string[] | undefined;
    constraints?: string[] | undefined;
    updated_at?: string | undefined;
}, {
    type: "theorem" | "definition" | "other" | "lemma" | "corollary" | "proposition" | "axiom" | "conjecture";
    id: string;
    created_at: string;
    statement: string;
    metadata?: Record<string, any> | undefined;
    dependencies?: string[] | undefined;
    constraints?: string[] | undefined;
    updated_at?: string | undefined;
}>;
export type Theorem = z.infer<typeof Theorem>;
/**
 * Formal Proof Schema
 *
 * Represents a complete formal proof
 */
export declare const FormalProof: z.ZodObject<{
    id: z.ZodString;
    theorem_id: z.ZodString;
    theorem: z.ZodString;
    proof: z.ZodString;
    system: z.ZodEnum<["z3", "leanaide", "coq", "isabelle", "hol", "other"]>;
    status: z.ZodEnum<["valid", "invalid", "partial", "unknown", "pending_validation"]>;
    confidence: z.ZodOptional<z.ZodNumber>;
    tactics: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodObject<{
        proof_length: z.ZodOptional<z.ZodNumber>;
        verification_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        solver_version: z.ZodOptional<z.ZodString>;
        tactics_count: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        verification_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
        proof_length?: number | undefined;
        solver_version?: string | undefined;
    }, {
        verification_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
        proof_length?: number | undefined;
        solver_version?: string | undefined;
    }>>;
    custom_metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp_utc: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
    id: string;
    theorem: string;
    proof: string;
    system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
    timestamp_utc: string;
    theorem_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        verification_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
        proof_length?: number | undefined;
        solver_version?: string | undefined;
    } | undefined;
    dependencies?: string[] | undefined;
    confidence?: number | undefined;
    tactics?: string[] | undefined;
    custom_metadata?: Record<string, any> | undefined;
}, {
    status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
    id: string;
    theorem: string;
    proof: string;
    system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
    timestamp_utc: string;
    theorem_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        verification_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
        proof_length?: number | undefined;
        solver_version?: string | undefined;
    } | undefined;
    dependencies?: string[] | undefined;
    confidence?: number | undefined;
    tactics?: string[] | undefined;
    custom_metadata?: Record<string, any> | undefined;
}>;
export type FormalProof = z.infer<typeof FormalProof>;
/**
 * Proof Dependency Schema
 *
 * Represents a dependency relationship between proofs
 */
export declare const ProofDependency: z.ZodObject<{
    id: z.ZodString;
    proof_id: z.ZodString;
    depends_on_proof_id: z.ZodString;
    type: z.ZodEnum<["direct_dependency", "indirect_dependency", "import", "definition_used", "theorem_applied", "axiom_referenced", "other"]>;
    validity: z.ZodBoolean;
    description: z.ZodOptional<z.ZodString>;
    created_at: z.ZodString;
}, "strip", z.ZodTypeAny, {
    type: "import" | "other" | "direct_dependency" | "indirect_dependency" | "definition_used" | "theorem_applied" | "axiom_referenced";
    id: string;
    proof_id: string;
    created_at: string;
    depends_on_proof_id: string;
    validity: boolean;
    description?: string | undefined;
}, {
    type: "import" | "other" | "direct_dependency" | "indirect_dependency" | "definition_used" | "theorem_applied" | "axiom_referenced";
    id: string;
    proof_id: string;
    created_at: string;
    depends_on_proof_id: string;
    validity: boolean;
    description?: string | undefined;
}>;
export type ProofDependency = z.infer<typeof ProofDependency>;
/**
 * Proof Validation Schema
 *
 * Represents the validation status of a proof
 */
export declare const ProofValidation: z.ZodObject<{
    id: z.ZodString;
    proof_id: z.ZodString;
    is_valid: z.ZodBoolean;
    errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
    }, {
        message: string;
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
    }>, "many">>;
    validated_at: z.ZodString;
    validated_by: z.ZodOptional<z.ZodEnum<["system", "user", "automated_check"]>>;
    validator_version: z.ZodOptional<z.ZodString>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    proof_id: string;
    is_valid: boolean;
    validated_at: string;
    correlation_id?: string | undefined;
    errors?: {
        message: string;
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
    }[] | undefined;
    validated_by?: "system" | "user" | "automated_check" | undefined;
    validator_version?: string | undefined;
}, {
    id: string;
    proof_id: string;
    is_valid: boolean;
    validated_at: string;
    correlation_id?: string | undefined;
    errors?: {
        message: string;
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
    }[] | undefined;
    validated_by?: "system" | "user" | "automated_check" | undefined;
    validator_version?: string | undefined;
}>;
export type ProofValidation = z.infer<typeof ProofValidation>;
/**
 * Similar Proof Schema
 *
 * Represents a proof similar to a query
 */
export declare const SimilarProof: z.ZodObject<{
    proof: z.ZodObject<{
        id: z.ZodString;
        theorem_id: z.ZodString;
        theorem: z.ZodString;
        proof: z.ZodString;
        system: z.ZodEnum<["z3", "leanaide", "coq", "isabelle", "hol", "other"]>;
        status: z.ZodEnum<["valid", "invalid", "partial", "unknown", "pending_validation"]>;
        confidence: z.ZodOptional<z.ZodNumber>;
        tactics: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodObject<{
            proof_length: z.ZodOptional<z.ZodNumber>;
            verification_time_ms: z.ZodOptional<z.ZodNumber>;
            memory_used_mb: z.ZodOptional<z.ZodNumber>;
            solver_version: z.ZodOptional<z.ZodString>;
            tactics_count: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            verification_time_ms?: number | undefined;
            memory_used_mb?: number | undefined;
            tactics_count?: number | undefined;
            proof_length?: number | undefined;
            solver_version?: string | undefined;
        }, {
            verification_time_ms?: number | undefined;
            memory_used_mb?: number | undefined;
            tactics_count?: number | undefined;
            proof_length?: number | undefined;
            solver_version?: string | undefined;
        }>>;
        custom_metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        timestamp_utc: z.ZodString;
        correlation_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
        id: string;
        theorem: string;
        proof: string;
        system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
        timestamp_utc: string;
        theorem_id: string;
        correlation_id?: string | undefined;
        metadata?: {
            verification_time_ms?: number | undefined;
            memory_used_mb?: number | undefined;
            tactics_count?: number | undefined;
            proof_length?: number | undefined;
            solver_version?: string | undefined;
        } | undefined;
        dependencies?: string[] | undefined;
        confidence?: number | undefined;
        tactics?: string[] | undefined;
        custom_metadata?: Record<string, any> | undefined;
    }, {
        status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
        id: string;
        theorem: string;
        proof: string;
        system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
        timestamp_utc: string;
        theorem_id: string;
        correlation_id?: string | undefined;
        metadata?: {
            verification_time_ms?: number | undefined;
            memory_used_mb?: number | undefined;
            tactics_count?: number | undefined;
            proof_length?: number | undefined;
            solver_version?: string | undefined;
        } | undefined;
        dependencies?: string[] | undefined;
        confidence?: number | undefined;
        tactics?: string[] | undefined;
        custom_metadata?: Record<string, any> | undefined;
    }>;
    similarity_score: z.ZodNumber;
    matching_theorems: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    explanation: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    proof: {
        status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
        id: string;
        theorem: string;
        proof: string;
        system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
        timestamp_utc: string;
        theorem_id: string;
        correlation_id?: string | undefined;
        metadata?: {
            verification_time_ms?: number | undefined;
            memory_used_mb?: number | undefined;
            tactics_count?: number | undefined;
            proof_length?: number | undefined;
            solver_version?: string | undefined;
        } | undefined;
        dependencies?: string[] | undefined;
        confidence?: number | undefined;
        tactics?: string[] | undefined;
        custom_metadata?: Record<string, any> | undefined;
    };
    similarity_score: number;
    matching_theorems?: string[] | undefined;
    explanation?: string | undefined;
}, {
    proof: {
        status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
        id: string;
        theorem: string;
        proof: string;
        system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
        timestamp_utc: string;
        theorem_id: string;
        correlation_id?: string | undefined;
        metadata?: {
            verification_time_ms?: number | undefined;
            memory_used_mb?: number | undefined;
            tactics_count?: number | undefined;
            proof_length?: number | undefined;
            solver_version?: string | undefined;
        } | undefined;
        dependencies?: string[] | undefined;
        confidence?: number | undefined;
        tactics?: string[] | undefined;
        custom_metadata?: Record<string, any> | undefined;
    };
    similarity_score: number;
    matching_theorems?: string[] | undefined;
    explanation?: string | undefined;
}>;
export type SimilarProof = z.infer<typeof SimilarProof>;
/**
 * Proof Lineage Schema
 *
 * Represents the ancestry/descendency of a proof
 */
export declare const ProofLineage: z.ZodObject<{
    proof_id: z.ZodString;
    ancestors: z.ZodArray<z.ZodObject<{
        proof_id: z.ZodString;
        depth: z.ZodNumber;
        relationship: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        proof_id: string;
        depth: number;
        relationship: string;
    }, {
        proof_id: string;
        depth: number;
        relationship: string;
    }>, "many">;
    descendants: z.ZodArray<z.ZodObject<{
        proof_id: z.ZodString;
        depth: z.ZodNumber;
        relationship: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        proof_id: string;
        depth: number;
        relationship: string;
    }, {
        proof_id: string;
        depth: number;
        relationship: string;
    }>, "many">;
    full_tree: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    computed_at: z.ZodString;
}, "strip", z.ZodTypeAny, {
    proof_id: string;
    ancestors: {
        proof_id: string;
        depth: number;
        relationship: string;
    }[];
    descendants: {
        proof_id: string;
        depth: number;
        relationship: string;
    }[];
    computed_at: string;
    full_tree?: Record<string, any> | undefined;
}, {
    proof_id: string;
    ancestors: {
        proof_id: string;
        depth: number;
        relationship: string;
    }[];
    descendants: {
        proof_id: string;
        depth: number;
        relationship: string;
    }[];
    computed_at: string;
    full_tree?: Record<string, any> | undefined;
}>;
export type ProofLineage = z.infer<typeof ProofLineage>;
/**
 * Proof History Schema
 *
 * Represents the history of changes to a proof
 */
export declare const ProofHistory: z.ZodObject<{
    proof_id: z.ZodString;
    versions: z.ZodArray<z.ZodObject<{
        version_id: z.ZodString;
        timestamp: z.ZodString;
        changes: z.ZodArray<z.ZodString, "many">;
        proof: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            theorem_id: z.ZodString;
            theorem: z.ZodString;
            proof: z.ZodString;
            system: z.ZodEnum<["z3", "leanaide", "coq", "isabelle", "hol", "other"]>;
            status: z.ZodEnum<["valid", "invalid", "partial", "unknown", "pending_validation"]>;
            confidence: z.ZodOptional<z.ZodNumber>;
            tactics: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            metadata: z.ZodOptional<z.ZodObject<{
                proof_length: z.ZodOptional<z.ZodNumber>;
                verification_time_ms: z.ZodOptional<z.ZodNumber>;
                memory_used_mb: z.ZodOptional<z.ZodNumber>;
                solver_version: z.ZodOptional<z.ZodString>;
                tactics_count: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            }, {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            }>>;
            custom_metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            timestamp_utc: z.ZodString;
            correlation_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
            id: string;
            theorem: string;
            proof: string;
            system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
            timestamp_utc: string;
            theorem_id: string;
            correlation_id?: string | undefined;
            metadata?: {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            } | undefined;
            dependencies?: string[] | undefined;
            confidence?: number | undefined;
            tactics?: string[] | undefined;
            custom_metadata?: Record<string, any> | undefined;
        }, {
            status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
            id: string;
            theorem: string;
            proof: string;
            system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
            timestamp_utc: string;
            theorem_id: string;
            correlation_id?: string | undefined;
            metadata?: {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            } | undefined;
            dependencies?: string[] | undefined;
            confidence?: number | undefined;
            tactics?: string[] | undefined;
            custom_metadata?: Record<string, any> | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        changes: string[];
        version_id: string;
        proof?: {
            status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
            id: string;
            theorem: string;
            proof: string;
            system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
            timestamp_utc: string;
            theorem_id: string;
            correlation_id?: string | undefined;
            metadata?: {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            } | undefined;
            dependencies?: string[] | undefined;
            confidence?: number | undefined;
            tactics?: string[] | undefined;
            custom_metadata?: Record<string, any> | undefined;
        } | undefined;
    }, {
        timestamp: string;
        changes: string[];
        version_id: string;
        proof?: {
            status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
            id: string;
            theorem: string;
            proof: string;
            system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
            timestamp_utc: string;
            theorem_id: string;
            correlation_id?: string | undefined;
            metadata?: {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            } | undefined;
            dependencies?: string[] | undefined;
            confidence?: number | undefined;
            tactics?: string[] | undefined;
            custom_metadata?: Record<string, any> | undefined;
        } | undefined;
    }>, "many">;
    version_count: z.ZodNumber;
    current_version_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    proof_id: string;
    versions: {
        timestamp: string;
        changes: string[];
        version_id: string;
        proof?: {
            status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
            id: string;
            theorem: string;
            proof: string;
            system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
            timestamp_utc: string;
            theorem_id: string;
            correlation_id?: string | undefined;
            metadata?: {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            } | undefined;
            dependencies?: string[] | undefined;
            confidence?: number | undefined;
            tactics?: string[] | undefined;
            custom_metadata?: Record<string, any> | undefined;
        } | undefined;
    }[];
    version_count: number;
    current_version_id?: string | undefined;
}, {
    proof_id: string;
    versions: {
        timestamp: string;
        changes: string[];
        version_id: string;
        proof?: {
            status: "unknown" | "partial" | "valid" | "invalid" | "pending_validation";
            id: string;
            theorem: string;
            proof: string;
            system: "z3" | "leanaide" | "other" | "coq" | "isabelle" | "hol";
            timestamp_utc: string;
            theorem_id: string;
            correlation_id?: string | undefined;
            metadata?: {
                verification_time_ms?: number | undefined;
                memory_used_mb?: number | undefined;
                tactics_count?: number | undefined;
                proof_length?: number | undefined;
                solver_version?: string | undefined;
            } | undefined;
            dependencies?: string[] | undefined;
            confidence?: number | undefined;
            tactics?: string[] | undefined;
            custom_metadata?: Record<string, any> | undefined;
        } | undefined;
    }[];
    version_count: number;
    current_version_id?: string | undefined;
}>;
export type ProofHistory = z.infer<typeof ProofHistory>;
/**
 * Storage Result Schema
 */
export declare const StorageResult: z.ZodObject<{
    success: z.ZodBoolean;
    proof_id: z.ZodOptional<z.ZodString>;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    error?: string | undefined;
    proof_id?: string | undefined;
}, {
    timestamp: string;
    success: boolean;
    error?: string | undefined;
    proof_id?: string | undefined;
}>;
export type StorageResult = z.infer<typeof StorageResult>;
/**
 * Update Result Schema
 */
export declare const UpdateResult: z.ZodObject<{
    success: z.ZodBoolean;
    proof_id: z.ZodOptional<z.ZodString>;
    previous_version_id: z.ZodOptional<z.ZodString>;
    new_version_id: z.ZodOptional<z.ZodString>;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    error?: string | undefined;
    proof_id?: string | undefined;
    previous_version_id?: string | undefined;
    new_version_id?: string | undefined;
}, {
    timestamp: string;
    success: boolean;
    error?: string | undefined;
    proof_id?: string | undefined;
    previous_version_id?: string | undefined;
    new_version_id?: string | undefined;
}>;
export type UpdateResult = z.infer<typeof UpdateResult>;
/**
 * Index Result Schema
 */
export declare const IndexResult: z.ZodObject<{
    success: z.ZodBoolean;
    embedding_id: z.ZodOptional<z.ZodString>;
    vector_indexed: z.ZodBoolean;
    graph_indexed: z.ZodBoolean;
    error: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    vector_indexed: boolean;
    graph_indexed: boolean;
    error?: string | undefined;
    embedding_id?: string | undefined;
}, {
    timestamp: string;
    success: boolean;
    vector_indexed: boolean;
    graph_indexed: boolean;
    error?: string | undefined;
    embedding_id?: string | undefined;
}>;
export type IndexResult = z.infer<typeof IndexResult>;
/**
 * Proof Metrics Schema
 */
export declare const ProofMetrics: z.ZodObject<{
    total_proofs: z.ZodNumber;
    proofs_by_system: z.ZodRecord<z.ZodString, z.ZodNumber>;
    proofs_by_status: z.ZodRecord<z.ZodString, z.ZodNumber>;
    average_confidence: z.ZodOptional<z.ZodNumber>;
    total_dependencies: z.ZodNumber;
    indexed_proofs: z.ZodNumber;
    last_updated: z.ZodString;
}, "strip", z.ZodTypeAny, {
    last_updated: string;
    total_proofs: number;
    proofs_by_system: Record<string, number>;
    proofs_by_status: Record<string, number>;
    total_dependencies: number;
    indexed_proofs: number;
    average_confidence?: number | undefined;
}, {
    last_updated: string;
    total_proofs: number;
    proofs_by_system: Record<string, number>;
    proofs_by_status: Record<string, number>;
    total_dependencies: number;
    indexed_proofs: number;
    average_confidence?: number | undefined;
}>;
export type ProofMetrics = z.infer<typeof ProofMetrics>;
/**
 * Revalidation Result Schema
 */
export declare const RevalidationResult: z.ZodObject<{
    success: z.ZodBoolean;
    proofs_revalidated: z.ZodNumber;
    proofs_invalidated: z.ZodNumber;
    errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    proofs_revalidated: number;
    proofs_invalidated: number;
    errors?: string[] | undefined;
}, {
    timestamp: string;
    success: boolean;
    proofs_revalidated: number;
    proofs_invalidated: number;
    errors?: string[] | undefined;
}>;
export type RevalidationResult = z.infer<typeof RevalidationResult>;
/**
 * Validation Functions
 */
/**
 * Validate a Theorem against the schema
 */
export declare function validateTheorem(data: unknown): {
    success: boolean;
    data?: Theorem;
    errors?: string[];
};
/**
 * Validate a FormalProof against the schema
 */
export declare function validateFormalProof(data: unknown): {
    success: boolean;
    data?: FormalProof;
    errors?: string[];
};
/**
 * Validate a ProofValidation against the schema
 */
export declare function validateProofValidation(data: unknown): {
    success: boolean;
    data?: ProofValidation;
    errors?: string[];
};
/**
 * Transformation Functions
 */
/**
 * Transform Z3 SolverResponse to FormalProof
 */
export declare function transformZ3ResponseToFormalProof(z3Response: any, theoremId: string, correlationId?: string): FormalProof;
/**
 * Transform LeanAide ProofVerificationResponse to FormalProof
 */
export declare function transformLeanAideResponseToFormalProof(leanResponse: any, theoremId: string, correlationId?: string): FormalProof;
/**
 * Example usage
 */
export declare const ProofKnowledgeBaseExamples: {
    validTheorem: Theorem;
    validFormalProof: FormalProof;
    validProofValidation: ProofValidation;
};
//# sourceMappingURL=canonical.d.ts.map