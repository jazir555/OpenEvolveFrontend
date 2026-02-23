/**
 * RESE Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for RESE (Recursive Epistemic Solvability Engine) interactions.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for RESE data in the glue layer.
 * Do not pass raw RESE API responses between services.
 *
 * RESE Technical Manual Reference:
 * - Phase I: Epistemic Audit and Falsification (Red Team Protocol)
 * - Phase II: Isomorphic Resonance and Constraint Inversion
 * - Phase III: Monte Carlo Metacognitive Refinement (MCTS)
 * - Phase IV: Architectural Synthesis and Validation
 */
import { z } from 'zod';
/**
 * RESE Phase Types
 */
export declare const RESEPhase: z.ZodEnum<["phase1_epistemic_audit", "phase2_isomorphic_mapping", "phase3_mcts_refinement", "phase4_architecture_assembly"]>;
export type RESEPhase = z.infer<typeof RESEPhase>;
/**
 * Constraint Categories
 */
export declare const ConstraintCategory: z.ZodEnum<["hard_parameter_inequality", "soft_statistical", "tacit_assumption", "inverted_constraint"]>;
export type ConstraintCategory = z.infer<typeof ConstraintCategory>;
/**
 * Logical Fallacy Types
 */
export declare const LogicalFallacy: z.ZodEnum<["circulus_in_probando", "confirmation_bias", "hasty_generalization", "false_cause", "ad_hominem", "straw_man", "contradiction", "inconsistency", "other"]>;
export type LogicalFallacy = z.infer<typeof LogicalFallacy>;
/**
 * ============================================================================
 * PHASE I: EPISTEMIC AUDIT RESULTS
 * ============================================================================
 */
/**
 * Tacit Assumption Schema
 *
 * Represents an unstated, heuristic constraint inferred from failure patterns.
 * From RESE Manual §3.1.5: Tacit Assumption Mining (Φ₁.₅)
 */
export declare const TacitAssumption: z.ZodObject<{
    id: z.ZodString;
    description: z.ZodString;
    source_pattern: z.ZodOptional<z.ZodString>;
    confidence_score: z.ZodNumber;
    supporting_evidence_count: z.ZodOptional<z.ZodNumber>;
    formalized_in_lean4: z.ZodOptional<z.ZodBoolean>;
    lean4_proposition: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    description: string;
    confidence_score: number;
    source_pattern?: string | undefined;
    supporting_evidence_count?: number | undefined;
    formalized_in_lean4?: boolean | undefined;
    lean4_proposition?: string | undefined;
}, {
    id: string;
    description: string;
    confidence_score: number;
    source_pattern?: string | undefined;
    supporting_evidence_count?: number | undefined;
    formalized_in_lean4?: boolean | undefined;
    lean4_proposition?: string | undefined;
}>;
export type TacitAssumption = z.infer<typeof TacitAssumption>;
/**
 * Contradiction Detection Schema
 *
 * Represents a logical contradiction detected by the Symbolic Constraint Engine.
 * From RESE Manual §3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
 */
export declare const ContradictionDetection: z.ZodObject<{
    id: z.ZodString;
    fallacy_type: z.ZodEnum<["circulus_in_probando", "confirmation_bias", "hasty_generalization", "false_cause", "ad_hominem", "straw_man", "contradiction", "inconsistency", "other"]>;
    contradiction_set_size: z.ZodNumber;
    rollback_steps: z.ZodOptional<z.ZodNumber>;
    affected_premises: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    resolved: z.ZodOptional<z.ZodBoolean>;
    resolution_strategy: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    fallacy_type: "contradiction" | "other" | "circulus_in_probando" | "confirmation_bias" | "hasty_generalization" | "false_cause" | "ad_hominem" | "straw_man" | "inconsistency";
    contradiction_set_size: number;
    resolved?: boolean | undefined;
    resolution_strategy?: string | undefined;
    rollback_steps?: number | undefined;
    affected_premises?: string[] | undefined;
}, {
    id: string;
    fallacy_type: "contradiction" | "other" | "circulus_in_probando" | "confirmation_bias" | "hasty_generalization" | "false_cause" | "ad_hominem" | "straw_man" | "inconsistency";
    contradiction_set_size: number;
    resolved?: boolean | undefined;
    resolution_strategy?: string | undefined;
    rollback_steps?: number | undefined;
    affected_premises?: string[] | undefined;
}>;
export type ContradictionDetection = z.infer<typeof ContradictionDetection>;
/**
 * Falsification Result Schema
 *
 * Represents the result of attempting to falsify a hypothesis.
 * From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification
 */
export declare const FalsificationResult: z.ZodObject<{
    hypothesis_id: z.ZodString;
    falsified: z.ZodBoolean;
    degree_of_violation: z.ZodOptional<z.ZodNumber>;
    hypothesis_robustness_score: z.ZodNumber;
    falsifying_evidence: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    counter_examples: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    hypothesis_id: string;
    falsified: boolean;
    hypothesis_robustness_score: number;
    degree_of_violation?: number | undefined;
    falsifying_evidence?: string[] | undefined;
    counter_examples?: string[] | undefined;
}, {
    hypothesis_id: string;
    falsified: boolean;
    hypothesis_robustness_score: number;
    degree_of_violation?: number | undefined;
    falsifying_evidence?: string[] | undefined;
    counter_examples?: string[] | undefined;
}>;
export type FalsificationResult = z.infer<typeof FalsificationResult>;
/**
 * Epistemic Audit Result Schema
 *
 * Complete result from Phase I: Epistemic Audit and Falsification.
 * From RESE Manual §3.0
 */
export declare const EpistemicAuditResult: z.ZodObject<{
    phase: z.ZodLiteral<"phase1_epistemic_audit">;
    audit_id: z.ZodString;
    problem_description: z.ZodString;
    hardened_constraints: z.ZodOptional<z.ZodArray<z.ZodObject<{
        category: z.ZodEnum<["hard_parameter_inequality", "soft_statistical", "tacit_assumption", "inverted_constraint"]>;
        description: z.ZodString;
        formalized: z.ZodBoolean;
        lean4_theorem: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        description: string;
        category: "hard_parameter_inequality" | "soft_statistical" | "tacit_assumption" | "inverted_constraint";
        formalized: boolean;
        lean4_theorem?: string | undefined;
    }, {
        description: string;
        category: "hard_parameter_inequality" | "soft_statistical" | "tacit_assumption" | "inverted_constraint";
        formalized: boolean;
        lean4_theorem?: string | undefined;
    }>, "many">>;
    tacit_assumptions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        description: z.ZodString;
        source_pattern: z.ZodOptional<z.ZodString>;
        confidence_score: z.ZodNumber;
        supporting_evidence_count: z.ZodOptional<z.ZodNumber>;
        formalized_in_lean4: z.ZodOptional<z.ZodBoolean>;
        lean4_proposition: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        description: string;
        confidence_score: number;
        source_pattern?: string | undefined;
        supporting_evidence_count?: number | undefined;
        formalized_in_lean4?: boolean | undefined;
        lean4_proposition?: string | undefined;
    }, {
        id: string;
        description: string;
        confidence_score: number;
        source_pattern?: string | undefined;
        supporting_evidence_count?: number | undefined;
        formalized_in_lean4?: boolean | undefined;
        lean4_proposition?: string | undefined;
    }>, "many">>;
    contradictions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        fallacy_type: z.ZodEnum<["circulus_in_probando", "confirmation_bias", "hasty_generalization", "false_cause", "ad_hominem", "straw_man", "contradiction", "inconsistency", "other"]>;
        contradiction_set_size: z.ZodNumber;
        rollback_steps: z.ZodOptional<z.ZodNumber>;
        affected_premises: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        resolved: z.ZodOptional<z.ZodBoolean>;
        resolution_strategy: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        fallacy_type: "contradiction" | "other" | "circulus_in_probando" | "confirmation_bias" | "hasty_generalization" | "false_cause" | "ad_hominem" | "straw_man" | "inconsistency";
        contradiction_set_size: number;
        resolved?: boolean | undefined;
        resolution_strategy?: string | undefined;
        rollback_steps?: number | undefined;
        affected_premises?: string[] | undefined;
    }, {
        id: string;
        fallacy_type: "contradiction" | "other" | "circulus_in_probando" | "confirmation_bias" | "hasty_generalization" | "false_cause" | "ad_hominem" | "straw_man" | "inconsistency";
        contradiction_set_size: number;
        resolved?: boolean | undefined;
        resolution_strategy?: string | undefined;
        rollback_steps?: number | undefined;
        affected_premises?: string[] | undefined;
    }>, "many">>;
    falsification_results: z.ZodOptional<z.ZodArray<z.ZodObject<{
        hypothesis_id: z.ZodString;
        falsified: z.ZodBoolean;
        degree_of_violation: z.ZodOptional<z.ZodNumber>;
        hypothesis_robustness_score: z.ZodNumber;
        falsifying_evidence: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        counter_examples: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        hypothesis_id: string;
        falsified: boolean;
        hypothesis_robustness_score: number;
        degree_of_violation?: number | undefined;
        falsifying_evidence?: string[] | undefined;
        counter_examples?: string[] | undefined;
    }, {
        hypothesis_id: string;
        falsified: boolean;
        hypothesis_robustness_score: number;
        degree_of_violation?: number | undefined;
        falsifying_evidence?: string[] | undefined;
        counter_examples?: string[] | undefined;
    }>, "many">>;
    metrics: z.ZodOptional<z.ZodObject<{
        total_assumptions_analyzed: z.ZodOptional<z.ZodNumber>;
        confirmed_contradictions: z.ZodOptional<z.ZodNumber>;
        hypotheses_falsified: z.ZodOptional<z.ZodNumber>;
        reduction_in_failure_rate: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_assumptions_analyzed?: number | undefined;
        confirmed_contradictions?: number | undefined;
        hypotheses_falsified?: number | undefined;
        reduction_in_failure_rate?: number | undefined;
    }, {
        total_assumptions_analyzed?: number | undefined;
        confirmed_contradictions?: number | undefined;
        hypotheses_falsified?: number | undefined;
        reduction_in_failure_rate?: number | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        lean4_version: z.ZodOptional<z.ZodString>;
        epoch_number: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    }, {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    problem_description: string;
    phase: "phase1_epistemic_audit";
    audit_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    } | undefined;
    metrics?: {
        total_assumptions_analyzed?: number | undefined;
        confirmed_contradictions?: number | undefined;
        hypotheses_falsified?: number | undefined;
        reduction_in_failure_rate?: number | undefined;
    } | undefined;
    hardened_constraints?: {
        description: string;
        category: "hard_parameter_inequality" | "soft_statistical" | "tacit_assumption" | "inverted_constraint";
        formalized: boolean;
        lean4_theorem?: string | undefined;
    }[] | undefined;
    tacit_assumptions?: {
        id: string;
        description: string;
        confidence_score: number;
        source_pattern?: string | undefined;
        supporting_evidence_count?: number | undefined;
        formalized_in_lean4?: boolean | undefined;
        lean4_proposition?: string | undefined;
    }[] | undefined;
    contradictions?: {
        id: string;
        fallacy_type: "contradiction" | "other" | "circulus_in_probando" | "confirmation_bias" | "hasty_generalization" | "false_cause" | "ad_hominem" | "straw_man" | "inconsistency";
        contradiction_set_size: number;
        resolved?: boolean | undefined;
        resolution_strategy?: string | undefined;
        rollback_steps?: number | undefined;
        affected_premises?: string[] | undefined;
    }[] | undefined;
    falsification_results?: {
        hypothesis_id: string;
        falsified: boolean;
        hypothesis_robustness_score: number;
        degree_of_violation?: number | undefined;
        falsifying_evidence?: string[] | undefined;
        counter_examples?: string[] | undefined;
    }[] | undefined;
}, {
    timestamp: string;
    problem_description: string;
    phase: "phase1_epistemic_audit";
    audit_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    } | undefined;
    metrics?: {
        total_assumptions_analyzed?: number | undefined;
        confirmed_contradictions?: number | undefined;
        hypotheses_falsified?: number | undefined;
        reduction_in_failure_rate?: number | undefined;
    } | undefined;
    hardened_constraints?: {
        description: string;
        category: "hard_parameter_inequality" | "soft_statistical" | "tacit_assumption" | "inverted_constraint";
        formalized: boolean;
        lean4_theorem?: string | undefined;
    }[] | undefined;
    tacit_assumptions?: {
        id: string;
        description: string;
        confidence_score: number;
        source_pattern?: string | undefined;
        supporting_evidence_count?: number | undefined;
        formalized_in_lean4?: boolean | undefined;
        lean4_proposition?: string | undefined;
    }[] | undefined;
    contradictions?: {
        id: string;
        fallacy_type: "contradiction" | "other" | "circulus_in_probando" | "confirmation_bias" | "hasty_generalization" | "false_cause" | "ad_hominem" | "straw_man" | "inconsistency";
        contradiction_set_size: number;
        resolved?: boolean | undefined;
        resolution_strategy?: string | undefined;
        rollback_steps?: number | undefined;
        affected_premises?: string[] | undefined;
    }[] | undefined;
    falsification_results?: {
        hypothesis_id: string;
        falsified: boolean;
        hypothesis_robustness_score: number;
        degree_of_violation?: number | undefined;
        falsifying_evidence?: string[] | undefined;
        counter_examples?: string[] | undefined;
    }[] | undefined;
}>;
export type EpistemicAuditResult = z.infer<typeof EpistemicAuditResult>;
/**
 * ============================================================================
 * PHASE II: ISOMORPHIC MAPPING
 * ============================================================================
 */
/**
 * Functional Dependency Graph Schema
 *
 * Represents causal connections and dependencies in a system.
 * From RESE Manual §4.2: Mechanistic Isomorphism Validation (ℑ_mech)
 */
export declare const FunctionalDependencyGraph: z.ZodObject<{
    nodes: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        type: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        id: string;
        description?: string | undefined;
    }, {
        type: string;
        id: string;
        description?: string | undefined;
    }>, "many">;
    edges: z.ZodArray<z.ZodObject<{
        source: z.ZodString;
        target: z.ZodString;
        relation_type: z.ZodString;
        strength: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        source: string;
        target: string;
        relation_type: string;
        strength?: number | undefined;
    }, {
        source: string;
        target: string;
        relation_type: string;
        strength?: number | undefined;
    }>, "many">;
    verified_in_lean4: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    edges: {
        source: string;
        target: string;
        relation_type: string;
        strength?: number | undefined;
    }[];
    nodes: {
        type: string;
        id: string;
        description?: string | undefined;
    }[];
    verified_in_lean4?: boolean | undefined;
}, {
    edges: {
        source: string;
        target: string;
        relation_type: string;
        strength?: number | undefined;
    }[];
    nodes: {
        type: string;
        id: string;
        description?: string | undefined;
    }[];
    verified_in_lean4?: boolean | undefined;
}>;
export type FunctionalDependencyGraph = z.infer<typeof FunctionalDependencyGraph>;
/**
 * Cross-Domain Pattern Schema
 *
 * Represents a pattern identified from a different domain.
 * From RESE Manual §4.2: Cross-Domain Ontology Mapping
 */
export declare const CrossDomainPattern: z.ZodObject<{
    id: z.ZodString;
    source_domain: z.ZodString;
    source_description: z.ZodString;
    target_domain: z.ZodString;
    target_description: z.ZodString;
    mechanistic_isomorphism_score: z.ZodNumber;
    source_fdg: z.ZodOptional<z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            type: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            id: string;
            description?: string | undefined;
        }, {
            type: string;
            id: string;
            description?: string | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            target: z.ZodString;
            relation_type: z.ZodString;
            strength: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }, {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }>, "many">;
        verified_in_lean4: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    }, {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    }>>;
    target_fdg: z.ZodOptional<z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            type: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            id: string;
            description?: string | undefined;
        }, {
            type: string;
            id: string;
            description?: string | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            target: z.ZodString;
            relation_type: z.ZodString;
            strength: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }, {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }>, "many">;
        verified_in_lean4: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    }, {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    }>>;
    fdg_overlap: z.ZodOptional<z.ZodNumber>;
    verified_predictive: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    id: string;
    source_domain: string;
    source_description: string;
    target_domain: string;
    target_description: string;
    mechanistic_isomorphism_score: number;
    source_fdg?: {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    } | undefined;
    target_fdg?: {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    } | undefined;
    fdg_overlap?: number | undefined;
    verified_predictive?: boolean | undefined;
}, {
    id: string;
    source_domain: string;
    source_description: string;
    target_domain: string;
    target_description: string;
    mechanistic_isomorphism_score: number;
    source_fdg?: {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    } | undefined;
    target_fdg?: {
        edges: {
            source: string;
            target: string;
            relation_type: string;
            strength?: number | undefined;
        }[];
        nodes: {
            type: string;
            id: string;
            description?: string | undefined;
        }[];
        verified_in_lean4?: boolean | undefined;
    } | undefined;
    fdg_overlap?: number | undefined;
    verified_predictive?: boolean | undefined;
}>;
export type CrossDomainPattern = z.infer<typeof CrossDomainPattern>;
/**
 * Inverted Constraint Schema
 *
 * Represents a constraint that has been inverted to define solution requirements.
 * From RESE Manual §4.3: Constraint Inversion and Parameter Space Rotation
 */
export declare const InvertedConstraint: z.ZodObject<{
    id: z.ZodString;
    original_constraint: z.ZodString;
    inverted_constraint: z.ZodString;
    parameter_space_rotation: z.ZodOptional<z.ZodString>;
    formalized_in_lean4: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    id: string;
    inverted_constraint: string;
    original_constraint: string;
    formalized_in_lean4?: boolean | undefined;
    parameter_space_rotation?: string | undefined;
}, {
    id: string;
    inverted_constraint: string;
    original_constraint: string;
    formalized_in_lean4?: boolean | undefined;
    parameter_space_rotation?: string | undefined;
}>;
export type InvertedConstraint = z.infer<typeof InvertedConstraint>;
/**
 * Isomorphic Mapping Result Schema
 *
 * Complete result from Phase II: Isomorphic Resonance and Constraint Inversion.
 * From RESE Manual §4.0
 */
export declare const IsomorphicMapping: z.ZodObject<{
    phase: z.ZodLiteral<"phase2_isomorphic_mapping">;
    mapping_id: z.ZodString;
    problem_description: z.ZodString;
    cross_domain_patterns: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        source_domain: z.ZodString;
        source_description: z.ZodString;
        target_domain: z.ZodString;
        target_description: z.ZodString;
        mechanistic_isomorphism_score: z.ZodNumber;
        source_fdg: z.ZodOptional<z.ZodObject<{
            nodes: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                type: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: string;
                id: string;
                description?: string | undefined;
            }, {
                type: string;
                id: string;
                description?: string | undefined;
            }>, "many">;
            edges: z.ZodArray<z.ZodObject<{
                source: z.ZodString;
                target: z.ZodString;
                relation_type: z.ZodString;
                strength: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }, {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }>, "many">;
            verified_in_lean4: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        }, {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        }>>;
        target_fdg: z.ZodOptional<z.ZodObject<{
            nodes: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                type: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: string;
                id: string;
                description?: string | undefined;
            }, {
                type: string;
                id: string;
                description?: string | undefined;
            }>, "many">;
            edges: z.ZodArray<z.ZodObject<{
                source: z.ZodString;
                target: z.ZodString;
                relation_type: z.ZodString;
                strength: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }, {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }>, "many">;
            verified_in_lean4: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        }, {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        }>>;
        fdg_overlap: z.ZodOptional<z.ZodNumber>;
        verified_predictive: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        source_domain: string;
        source_description: string;
        target_domain: string;
        target_description: string;
        mechanistic_isomorphism_score: number;
        source_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        target_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        fdg_overlap?: number | undefined;
        verified_predictive?: boolean | undefined;
    }, {
        id: string;
        source_domain: string;
        source_description: string;
        target_domain: string;
        target_description: string;
        mechanistic_isomorphism_score: number;
        source_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        target_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        fdg_overlap?: number | undefined;
        verified_predictive?: boolean | undefined;
    }>, "many">>;
    inverted_constraints: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        original_constraint: z.ZodString;
        inverted_constraint: z.ZodString;
        parameter_space_rotation: z.ZodOptional<z.ZodString>;
        formalized_in_lean4: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        inverted_constraint: string;
        original_constraint: string;
        formalized_in_lean4?: boolean | undefined;
        parameter_space_rotation?: string | undefined;
    }, {
        id: string;
        inverted_constraint: string;
        original_constraint: string;
        formalized_in_lean4?: boolean | undefined;
        parameter_space_rotation?: string | undefined;
    }>, "many">>;
    metrics: z.ZodOptional<z.ZodObject<{
        total_domains_searched: z.ZodOptional<z.ZodNumber>;
        patterns_identified: z.ZodOptional<z.ZodNumber>;
        high_isomorphism_count: z.ZodOptional<z.ZodNumber>;
        average_isomorphism_score: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_domains_searched?: number | undefined;
        patterns_identified?: number | undefined;
        high_isomorphism_count?: number | undefined;
        average_isomorphism_score?: number | undefined;
    }, {
        total_domains_searched?: number | undefined;
        patterns_identified?: number | undefined;
        high_isomorphism_count?: number | undefined;
        average_isomorphism_score?: number | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        lean4_version: z.ZodOptional<z.ZodString>;
        epoch_number: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    }, {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    problem_description: string;
    phase: "phase2_isomorphic_mapping";
    mapping_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    } | undefined;
    metrics?: {
        total_domains_searched?: number | undefined;
        patterns_identified?: number | undefined;
        high_isomorphism_count?: number | undefined;
        average_isomorphism_score?: number | undefined;
    } | undefined;
    cross_domain_patterns?: {
        id: string;
        source_domain: string;
        source_description: string;
        target_domain: string;
        target_description: string;
        mechanistic_isomorphism_score: number;
        source_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        target_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        fdg_overlap?: number | undefined;
        verified_predictive?: boolean | undefined;
    }[] | undefined;
    inverted_constraints?: {
        id: string;
        inverted_constraint: string;
        original_constraint: string;
        formalized_in_lean4?: boolean | undefined;
        parameter_space_rotation?: string | undefined;
    }[] | undefined;
}, {
    timestamp: string;
    problem_description: string;
    phase: "phase2_isomorphic_mapping";
    mapping_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        lean4_version?: string | undefined;
        epoch_number?: number | undefined;
    } | undefined;
    metrics?: {
        total_domains_searched?: number | undefined;
        patterns_identified?: number | undefined;
        high_isomorphism_count?: number | undefined;
        average_isomorphism_score?: number | undefined;
    } | undefined;
    cross_domain_patterns?: {
        id: string;
        source_domain: string;
        source_description: string;
        target_domain: string;
        target_description: string;
        mechanistic_isomorphism_score: number;
        source_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        target_fdg?: {
            edges: {
                source: string;
                target: string;
                relation_type: string;
                strength?: number | undefined;
            }[];
            nodes: {
                type: string;
                id: string;
                description?: string | undefined;
            }[];
            verified_in_lean4?: boolean | undefined;
        } | undefined;
        fdg_overlap?: number | undefined;
        verified_predictive?: boolean | undefined;
    }[] | undefined;
    inverted_constraints?: {
        id: string;
        inverted_constraint: string;
        original_constraint: string;
        formalized_in_lean4?: boolean | undefined;
        parameter_space_rotation?: string | undefined;
    }[] | undefined;
}>;
export type IsomorphicMapping = z.infer<typeof IsomorphicMapping>;
/**
 * ============================================================================
 * PHASE III: MCTS SEARCH RESULT
 * ============================================================================
 */
/**
 * Search Tree Node Schema
 *
 * Represents a node in the MCTS search tree.
 * From RESE Manual §5.0: Monte Carlo Metacognitive Refinement
 */
export declare const SearchTreeNode: z.ZodObject<{
    id: z.ZodString;
    visit_count: z.ZodNumber;
    value: z.ZodNumber;
    ucb1_score: z.ZodOptional<z.ZodNumber>;
    hypothesis: z.ZodOptional<z.ZodString>;
    parent_id: z.ZodOptional<z.ZodString>;
    child_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    id: string;
    value: number;
    visit_count: number;
    parent_id?: string | undefined;
    ucb1_score?: number | undefined;
    hypothesis?: string | undefined;
    child_ids?: string[] | undefined;
}, {
    id: string;
    value: number;
    visit_count: number;
    parent_id?: string | undefined;
    ucb1_score?: number | undefined;
    hypothesis?: string | undefined;
    child_ids?: string[] | undefined;
}>;
export type SearchTreeNode = z.infer<typeof SearchTreeNode>;
/**
 * Hypothesis Schema
 *
 * Represents a hypothesis generated during MCTS search.
 */
export declare const Hypothesis: z.ZodObject<{
    id: z.ZodString;
    statement: z.ZodString;
    confidence: z.ZodNumber;
    expected_value: z.ZodOptional<z.ZodNumber>;
    visit_count: z.ZodOptional<z.ZodNumber>;
    validated: z.ZodOptional<z.ZodBoolean>;
    falsified: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    id: string;
    confidence: number;
    statement: string;
    validated?: boolean | undefined;
    falsified?: boolean | undefined;
    visit_count?: number | undefined;
    expected_value?: number | undefined;
}, {
    id: string;
    confidence: number;
    statement: string;
    validated?: boolean | undefined;
    falsified?: boolean | undefined;
    visit_count?: number | undefined;
    expected_value?: number | undefined;
}>;
export type Hypothesis = z.infer<typeof Hypothesis>;
/**
 * Validation Metrics Schema
 *
 * Represents metrics for hypothesis validation.
 * From RESE Manual §5.2: High-Entropy Data Analysis
 */
export declare const ValidationMetrics: z.ZodObject<{
    disorder_entropy: z.ZodOptional<z.ZodNumber>;
    causal_coherence: z.ZodOptional<z.ZodNumber>;
    aci_score: z.ZodOptional<z.ZodNumber>;
    convergence_rate: z.ZodOptional<z.ZodNumber>;
    epochs_to_convergence: z.ZodOptional<z.ZodNumber>;
    predictive_accuracy: z.ZodOptional<z.ZodNumber>;
    reduction_in_aci: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    disorder_entropy?: number | undefined;
    causal_coherence?: number | undefined;
    aci_score?: number | undefined;
    convergence_rate?: number | undefined;
    epochs_to_convergence?: number | undefined;
    predictive_accuracy?: number | undefined;
    reduction_in_aci?: number | undefined;
}, {
    disorder_entropy?: number | undefined;
    causal_coherence?: number | undefined;
    aci_score?: number | undefined;
    convergence_rate?: number | undefined;
    epochs_to_convergence?: number | undefined;
    predictive_accuracy?: number | undefined;
    reduction_in_aci?: number | undefined;
}>;
export type ValidationMetrics = z.infer<typeof ValidationMetrics>;
/**
 * MCTS Search Result Schema
 *
 * Complete result from Phase III: Monte Carlo Metacognitive Refinement.
 * From RESE Manual §5.0
 */
export declare const MCTSSearchResult: z.ZodObject<{
    phase: z.ZodLiteral<"phase3_mcts_refinement">;
    search_id: z.ZodString;
    problem_description: z.ZodString;
    search_tree: z.ZodOptional<z.ZodObject<{
        root_node: z.ZodObject<{
            id: z.ZodString;
            visit_count: z.ZodNumber;
            value: z.ZodNumber;
            ucb1_score: z.ZodOptional<z.ZodNumber>;
            hypothesis: z.ZodOptional<z.ZodString>;
            parent_id: z.ZodOptional<z.ZodString>;
            child_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            value: number;
            visit_count: number;
            parent_id?: string | undefined;
            ucb1_score?: number | undefined;
            hypothesis?: string | undefined;
            child_ids?: string[] | undefined;
        }, {
            id: string;
            value: number;
            visit_count: number;
            parent_id?: string | undefined;
            ucb1_score?: number | undefined;
            hypothesis?: string | undefined;
            child_ids?: string[] | undefined;
        }>;
        total_nodes: z.ZodNumber;
        max_depth: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        max_depth: number;
        root_node: {
            id: string;
            value: number;
            visit_count: number;
            parent_id?: string | undefined;
            ucb1_score?: number | undefined;
            hypothesis?: string | undefined;
            child_ids?: string[] | undefined;
        };
        total_nodes: number;
    }, {
        max_depth: number;
        root_node: {
            id: string;
            value: number;
            visit_count: number;
            parent_id?: string | undefined;
            ucb1_score?: number | undefined;
            hypothesis?: string | undefined;
            child_ids?: string[] | undefined;
        };
        total_nodes: number;
    }>>;
    hypotheses: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        statement: z.ZodString;
        confidence: z.ZodNumber;
        expected_value: z.ZodOptional<z.ZodNumber>;
        visit_count: z.ZodOptional<z.ZodNumber>;
        validated: z.ZodOptional<z.ZodBoolean>;
        falsified: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }, {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }>, "many">>;
    top_hypotheses: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        statement: z.ZodString;
        confidence: z.ZodNumber;
        expected_value: z.ZodOptional<z.ZodNumber>;
        visit_count: z.ZodOptional<z.ZodNumber>;
        validated: z.ZodOptional<z.ZodBoolean>;
        falsified: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }, {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }>, "many">>;
    validation_metrics: z.ZodOptional<z.ZodObject<{
        disorder_entropy: z.ZodOptional<z.ZodNumber>;
        causal_coherence: z.ZodOptional<z.ZodNumber>;
        aci_score: z.ZodOptional<z.ZodNumber>;
        convergence_rate: z.ZodOptional<z.ZodNumber>;
        epochs_to_convergence: z.ZodOptional<z.ZodNumber>;
        predictive_accuracy: z.ZodOptional<z.ZodNumber>;
        reduction_in_aci: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        disorder_entropy?: number | undefined;
        causal_coherence?: number | undefined;
        aci_score?: number | undefined;
        convergence_rate?: number | undefined;
        epochs_to_convergence?: number | undefined;
        predictive_accuracy?: number | undefined;
        reduction_in_aci?: number | undefined;
    }, {
        disorder_entropy?: number | undefined;
        causal_coherence?: number | undefined;
        aci_score?: number | undefined;
        convergence_rate?: number | undefined;
        epochs_to_convergence?: number | undefined;
        predictive_accuracy?: number | undefined;
        reduction_in_aci?: number | undefined;
    }>>;
    converged: z.ZodOptional<z.ZodBoolean>;
    convergence_criteria_met: z.ZodOptional<z.ZodString>;
    metrics: z.ZodOptional<z.ZodObject<{
        total_simulations: z.ZodOptional<z.ZodNumber>;
        hypotheses_tested: z.ZodOptional<z.ZodNumber>;
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        epoch_number: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        epoch_number?: number | undefined;
        total_simulations?: number | undefined;
        hypotheses_tested?: number | undefined;
    }, {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        epoch_number?: number | undefined;
        total_simulations?: number | undefined;
        hypotheses_tested?: number | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        lean4_version: z.ZodOptional<z.ZodString>;
        mcts_strategy: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        lean4_version?: string | undefined;
        mcts_strategy?: string | undefined;
    }, {
        lean4_version?: string | undefined;
        mcts_strategy?: string | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    problem_description: string;
    phase: "phase3_mcts_refinement";
    search_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        lean4_version?: string | undefined;
        mcts_strategy?: string | undefined;
    } | undefined;
    metrics?: {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        epoch_number?: number | undefined;
        total_simulations?: number | undefined;
        hypotheses_tested?: number | undefined;
    } | undefined;
    hypotheses?: {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }[] | undefined;
    search_tree?: {
        max_depth: number;
        root_node: {
            id: string;
            value: number;
            visit_count: number;
            parent_id?: string | undefined;
            ucb1_score?: number | undefined;
            hypothesis?: string | undefined;
            child_ids?: string[] | undefined;
        };
        total_nodes: number;
    } | undefined;
    top_hypotheses?: {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }[] | undefined;
    validation_metrics?: {
        disorder_entropy?: number | undefined;
        causal_coherence?: number | undefined;
        aci_score?: number | undefined;
        convergence_rate?: number | undefined;
        epochs_to_convergence?: number | undefined;
        predictive_accuracy?: number | undefined;
        reduction_in_aci?: number | undefined;
    } | undefined;
    converged?: boolean | undefined;
    convergence_criteria_met?: string | undefined;
}, {
    timestamp: string;
    problem_description: string;
    phase: "phase3_mcts_refinement";
    search_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        lean4_version?: string | undefined;
        mcts_strategy?: string | undefined;
    } | undefined;
    metrics?: {
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        epoch_number?: number | undefined;
        total_simulations?: number | undefined;
        hypotheses_tested?: number | undefined;
    } | undefined;
    hypotheses?: {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }[] | undefined;
    search_tree?: {
        max_depth: number;
        root_node: {
            id: string;
            value: number;
            visit_count: number;
            parent_id?: string | undefined;
            ucb1_score?: number | undefined;
            hypothesis?: string | undefined;
            child_ids?: string[] | undefined;
        };
        total_nodes: number;
    } | undefined;
    top_hypotheses?: {
        id: string;
        confidence: number;
        statement: string;
        validated?: boolean | undefined;
        falsified?: boolean | undefined;
        visit_count?: number | undefined;
        expected_value?: number | undefined;
    }[] | undefined;
    validation_metrics?: {
        disorder_entropy?: number | undefined;
        causal_coherence?: number | undefined;
        aci_score?: number | undefined;
        convergence_rate?: number | undefined;
        epochs_to_convergence?: number | undefined;
        predictive_accuracy?: number | undefined;
        reduction_in_aci?: number | undefined;
    } | undefined;
    converged?: boolean | undefined;
    convergence_criteria_met?: string | undefined;
}>;
export type MCTSSearchResult = z.infer<typeof MCTSSearchResult>;
/**
 * ============================================================================
 * PHASE IV: ARCHITECTURE ASSEMBLY
 * ============================================================================
 */
/**
 * Paradigm Shift Schema
 *
 * Represents a paradigm shift in the theoretical framework.
 * From RESE Manual §6.0: Architectural Synthesis and Validation
 */
export declare const ParadigmShift: z.ZodObject<{
    id: z.ZodString;
    incumbent_paradigm: z.ZodString;
    new_paradigm: z.ZodString;
    anomalies_explained: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    predictive_improvements: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    validated: z.ZodOptional<z.ZodBoolean>;
    validation_criteria: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    incumbent_paradigm: string;
    new_paradigm: string;
    validated?: boolean | undefined;
    anomalies_explained?: string[] | undefined;
    predictive_improvements?: string[] | undefined;
    validation_criteria?: string | undefined;
}, {
    id: string;
    incumbent_paradigm: string;
    new_paradigm: string;
    validated?: boolean | undefined;
    anomalies_explained?: string[] | undefined;
    predictive_improvements?: string[] | undefined;
    validation_criteria?: string | undefined;
}>;
export type ParadigmShift = z.infer<typeof ParadigmShift>;
/**
 * Synthesized Knowledge Schema
 *
 * Represents knowledge synthesized from the RESE process.
 */
export declare const SynthesizedKnowledge: z.ZodObject<{
    id: z.ZodString;
    knowledge_type: z.ZodEnum<["theoretical_model", "algorithm", "experimental_protocol", "verification_proof", "other"]>;
    title: z.ZodString;
    description: z.ZodString;
    lean4_proof: z.ZodOptional<z.ZodString>;
    verified: z.ZodOptional<z.ZodBoolean>;
    applicable_domains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    depends_on: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    id: string;
    title: string;
    description: string;
    knowledge_type: "algorithm" | "other" | "theoretical_model" | "experimental_protocol" | "verification_proof";
    verified?: boolean | undefined;
    depends_on?: string[] | undefined;
    lean4_proof?: string | undefined;
    applicable_domains?: string[] | undefined;
}, {
    id: string;
    title: string;
    description: string;
    knowledge_type: "algorithm" | "other" | "theoretical_model" | "experimental_protocol" | "verification_proof";
    verified?: boolean | undefined;
    depends_on?: string[] | undefined;
    lean4_proof?: string | undefined;
    applicable_domains?: string[] | undefined;
}>;
export type SynthesizedKnowledge = z.infer<typeof SynthesizedKnowledge>;
/**
 * Architecture Assembly Result Schema
 *
 * Complete result from Phase IV: Architectural Synthesis and Validation.
 * From RESE Manual §6.0
 */
export declare const ArchitectureAssembly: z.ZodObject<{
    phase: z.ZodLiteral<"phase4_architecture_assembly">;
    assembly_id: z.ZodString;
    problem_description: z.ZodString;
    paradigm_shifts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        incumbent_paradigm: z.ZodString;
        new_paradigm: z.ZodString;
        anomalies_explained: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        predictive_improvements: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        validated: z.ZodOptional<z.ZodBoolean>;
        validation_criteria: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        incumbent_paradigm: string;
        new_paradigm: string;
        validated?: boolean | undefined;
        anomalies_explained?: string[] | undefined;
        predictive_improvements?: string[] | undefined;
        validation_criteria?: string | undefined;
    }, {
        id: string;
        incumbent_paradigm: string;
        new_paradigm: string;
        validated?: boolean | undefined;
        anomalies_explained?: string[] | undefined;
        predictive_improvements?: string[] | undefined;
        validation_criteria?: string | undefined;
    }>, "many">>;
    synthesized_knowledge: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        knowledge_type: z.ZodEnum<["theoretical_model", "algorithm", "experimental_protocol", "verification_proof", "other"]>;
        title: z.ZodString;
        description: z.ZodString;
        lean4_proof: z.ZodOptional<z.ZodString>;
        verified: z.ZodOptional<z.ZodBoolean>;
        applicable_domains: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        depends_on: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        title: string;
        description: string;
        knowledge_type: "algorithm" | "other" | "theoretical_model" | "experimental_protocol" | "verification_proof";
        verified?: boolean | undefined;
        depends_on?: string[] | undefined;
        lean4_proof?: string | undefined;
        applicable_domains?: string[] | undefined;
    }, {
        id: string;
        title: string;
        description: string;
        knowledge_type: "algorithm" | "other" | "theoretical_model" | "experimental_protocol" | "verification_proof";
        verified?: boolean | undefined;
        depends_on?: string[] | undefined;
        lean4_proof?: string | undefined;
        applicable_domains?: string[] | undefined;
    }>, "many">>;
    validation_results: z.ZodOptional<z.ZodObject<{
        predictive_model_efficacy: z.ZodOptional<z.ZodBoolean>;
        aci_reduction_achieved: z.ZodOptional<z.ZodNumber>;
        testable_predictions_count: z.ZodOptional<z.ZodNumber>;
        predictions_verified: z.ZodOptional<z.ZodNumber>;
        verification_success_rate: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        predictive_model_efficacy?: boolean | undefined;
        aci_reduction_achieved?: number | undefined;
        testable_predictions_count?: number | undefined;
        predictions_verified?: number | undefined;
        verification_success_rate?: number | undefined;
    }, {
        predictive_model_efficacy?: boolean | undefined;
        aci_reduction_achieved?: number | undefined;
        testable_predictions_count?: number | undefined;
        predictions_verified?: number | undefined;
        verification_success_rate?: number | undefined;
    }>>;
    metrics: z.ZodOptional<z.ZodObject<{
        total_epochs_completed: z.ZodOptional<z.ZodNumber>;
        total_execution_time_ms: z.ZodOptional<z.ZodNumber>;
        peak_memory_mb: z.ZodOptional<z.ZodNumber>;
        lean4_theorems_proved: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_execution_time_ms?: number | undefined;
        total_epochs_completed?: number | undefined;
        peak_memory_mb?: number | undefined;
        lean4_theorems_proved?: number | undefined;
    }, {
        total_execution_time_ms?: number | undefined;
        total_epochs_completed?: number | undefined;
        peak_memory_mb?: number | undefined;
        lean4_theorems_proved?: number | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        lean4_version: z.ZodOptional<z.ZodString>;
        rese_version: z.ZodOptional<z.ZodString>;
        converged: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        lean4_version?: string | undefined;
        converged?: boolean | undefined;
        rese_version?: string | undefined;
    }, {
        lean4_version?: string | undefined;
        converged?: boolean | undefined;
        rese_version?: string | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    problem_description: string;
    phase: "phase4_architecture_assembly";
    assembly_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        lean4_version?: string | undefined;
        converged?: boolean | undefined;
        rese_version?: string | undefined;
    } | undefined;
    metrics?: {
        total_execution_time_ms?: number | undefined;
        total_epochs_completed?: number | undefined;
        peak_memory_mb?: number | undefined;
        lean4_theorems_proved?: number | undefined;
    } | undefined;
    paradigm_shifts?: {
        id: string;
        incumbent_paradigm: string;
        new_paradigm: string;
        validated?: boolean | undefined;
        anomalies_explained?: string[] | undefined;
        predictive_improvements?: string[] | undefined;
        validation_criteria?: string | undefined;
    }[] | undefined;
    synthesized_knowledge?: {
        id: string;
        title: string;
        description: string;
        knowledge_type: "algorithm" | "other" | "theoretical_model" | "experimental_protocol" | "verification_proof";
        verified?: boolean | undefined;
        depends_on?: string[] | undefined;
        lean4_proof?: string | undefined;
        applicable_domains?: string[] | undefined;
    }[] | undefined;
    validation_results?: {
        predictive_model_efficacy?: boolean | undefined;
        aci_reduction_achieved?: number | undefined;
        testable_predictions_count?: number | undefined;
        predictions_verified?: number | undefined;
        verification_success_rate?: number | undefined;
    } | undefined;
}, {
    timestamp: string;
    problem_description: string;
    phase: "phase4_architecture_assembly";
    assembly_id: string;
    correlation_id?: string | undefined;
    metadata?: {
        lean4_version?: string | undefined;
        converged?: boolean | undefined;
        rese_version?: string | undefined;
    } | undefined;
    metrics?: {
        total_execution_time_ms?: number | undefined;
        total_epochs_completed?: number | undefined;
        peak_memory_mb?: number | undefined;
        lean4_theorems_proved?: number | undefined;
    } | undefined;
    paradigm_shifts?: {
        id: string;
        incumbent_paradigm: string;
        new_paradigm: string;
        validated?: boolean | undefined;
        anomalies_explained?: string[] | undefined;
        predictive_improvements?: string[] | undefined;
        validation_criteria?: string | undefined;
    }[] | undefined;
    synthesized_knowledge?: {
        id: string;
        title: string;
        description: string;
        knowledge_type: "algorithm" | "other" | "theoretical_model" | "experimental_protocol" | "verification_proof";
        verified?: boolean | undefined;
        depends_on?: string[] | undefined;
        lean4_proof?: string | undefined;
        applicable_domains?: string[] | undefined;
    }[] | undefined;
    validation_results?: {
        predictive_model_efficacy?: boolean | undefined;
        aci_reduction_achieved?: number | undefined;
        testable_predictions_count?: number | undefined;
        predictions_verified?: number | undefined;
        verification_success_rate?: number | undefined;
    } | undefined;
}>;
export type ArchitectureAssembly = z.infer<typeof ArchitectureAssembly>;
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform raw RESE Phase I response to canonical EpistemicAuditResult
 */
export declare function transformEpistemicAuditToCanonical(rawResponse: any, correlationId?: string): EpistemicAuditResult;
/**
 * Transform raw RESE Phase II response to canonical IsomorphicMapping
 */
export declare function transformIsomorphicMappingToCanonical(rawResponse: any, correlationId?: string): IsomorphicMapping;
/**
 * Transform raw RESE Phase III response to canonical MCTSSearchResult
 */
export declare function transformMCTSSearchToCanonical(rawResponse: any, correlationId?: string): MCTSSearchResult;
/**
 * Transform raw RESE Phase IV response to canonical ArchitectureAssembly
 */
export declare function transformArchitectureAssemblyToCanonical(rawResponse: any, correlationId?: string): ArchitectureAssembly;
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate an EpistemicAuditResult against the schema
 */
export declare function validateEpistemicAuditResult(data: unknown): {
    success: boolean;
    data?: EpistemicAuditResult;
    errors?: string[];
};
/**
 * Validate an IsomorphicMapping against the schema
 */
export declare function validateIsomorphicMapping(data: unknown): {
    success: boolean;
    data?: IsomorphicMapping;
    errors?: string[];
};
/**
 * Validate an MCTSSearchResult against the schema
 */
export declare function validateMCTSSearchResult(data: unknown): {
    success: boolean;
    data?: MCTSSearchResult;
    errors?: string[];
};
/**
 * Validate an ArchitectureAssembly against the schema
 */
export declare function validateArchitectureAssembly(data: unknown): {
    success: boolean;
    data?: ArchitectureAssembly;
    errors?: string[];
};
/**
 * Create a UTC timestamp in ISO-8601 format
 */
export declare function createUTCTimestamp(): string;
/**
 * Create a correlation ID (UUID v4)
 */
export declare function createCorrelationId(): string;
/**
 * ============================================================================
 * EXAMPLE DATA
 * ============================================================================
 */
/**
 * Example RESE data for testing and documentation
 */
export declare const RESEExamples: {
    validEpistemicAuditResult: EpistemicAuditResult;
    validIsomorphicMapping: IsomorphicMapping;
    validMCTSSearchResult: MCTSSearchResult;
    validArchitectureAssembly: ArchitectureAssembly;
};
//# sourceMappingURL=rese-canonical.d.ts.map