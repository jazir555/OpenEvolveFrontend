"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.RESEExamples = exports.ArchitectureAssembly = exports.SynthesizedKnowledge = exports.ParadigmShift = exports.MCTSSearchResult = exports.ValidationMetrics = exports.Hypothesis = exports.SearchTreeNode = exports.IsomorphicMapping = exports.InvertedConstraint = exports.CrossDomainPattern = exports.FunctionalDependencyGraph = exports.EpistemicAuditResult = exports.FalsificationResult = exports.ContradictionDetection = exports.TacitAssumption = exports.LogicalFallacy = exports.ConstraintCategory = exports.RESEPhase = void 0;
exports.transformEpistemicAuditToCanonical = transformEpistemicAuditToCanonical;
exports.transformIsomorphicMappingToCanonical = transformIsomorphicMappingToCanonical;
exports.transformMCTSSearchToCanonical = transformMCTSSearchToCanonical;
exports.transformArchitectureAssemblyToCanonical = transformArchitectureAssemblyToCanonical;
exports.validateEpistemicAuditResult = validateEpistemicAuditResult;
exports.validateIsomorphicMapping = validateIsomorphicMapping;
exports.validateMCTSSearchResult = validateMCTSSearchResult;
exports.validateArchitectureAssembly = validateArchitectureAssembly;
exports.createUTCTimestamp = createUTCTimestamp;
exports.createCorrelationId = createCorrelationId;
const zod_1 = require("zod");
/**
 * RESE Phase Types
 */
exports.RESEPhase = zod_1.z.enum([
    'phase1_epistemic_audit',
    'phase2_isomorphic_mapping',
    'phase3_mcts_refinement',
    'phase4_architecture_assembly',
]);
/**
 * Constraint Categories
 */
exports.ConstraintCategory = zod_1.z.enum([
    'hard_parameter_inequality', // Category A: Physical laws
    'soft_statistical', // Category B: Heuristics
    'tacit_assumption', // Category C: Unstated beliefs
    'inverted_constraint', // Category D: Solution requirements
]);
/**
 * Logical Fallacy Types
 */
exports.LogicalFallacy = zod_1.z.enum([
    'circulus_in_probando', // Circular reasoning
    'confirmation_bias', // Bias towards confirmation
    'hasty_generalization', // Insufficient evidence
    'false_cause', // Causation error
    'ad_hominem', // Personal attack
    'straw_man', // Misrepresentation
    'contradiction', // Direct contradiction
    'inconsistency', // Inconsistent premises
    'other',
]);
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
exports.TacitAssumption = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the assumption"),
    description: zod_1.z.string()
        .min(1, "Assumption description cannot be empty")
        .describe("The tacit assumption statement"),
    // Source of the assumption
    source_pattern: zod_1.z.string().optional()
        .describe("Pattern in null data that revealed this assumption"),
    // Confidence metrics
    confidence_score: zod_1.z.number()
        .min(0, "Confidence must be between 0 and 1")
        .max(1, "Confidence must be between 0 and 1")
        .describe("Statistical confidence in this assumption (0-1)"),
    supporting_evidence_count: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of data points supporting this assumption"),
    // Formalization status
    formalized_in_lean4: zod_1.z.boolean().optional()
        .describe("Whether this assumption has been formalized in Lean 4"),
    lean4_proposition: zod_1.z.string().optional()
        .describe("The formal proposition in Lean 4 syntax"),
});
/**
 * Contradiction Detection Schema
 *
 * Represents a logical contradiction detected by the Symbolic Constraint Engine.
 * From RESE Manual §3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
 */
exports.ContradictionDetection = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the contradiction"),
    // Type of fallacy
    fallacy_type: exports.LogicalFallacy.describe("Type of logical fallacy detected"),
    // Contradiction metrics
    contradiction_set_size: zod_1.z.number().int().nonnegative()
        .describe("Size of the contradiction set (CSS) - number of propositions involved"),
    rollback_steps: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of steps required to rollback to root premise violation"),
    // Location in proof tree
    affected_premises: zod_1.z.array(zod_1.z.string()).optional()
        .describe("IDs of premises involved in the contradiction"),
    // Resolution status
    resolved: zod_1.z.boolean().optional()
        .describe("Whether the contradiction has been resolved"),
    resolution_strategy: zod_1.z.string().optional()
        .describe("Strategy used to resolve the contradiction"),
});
/**
 * Falsification Result Schema
 *
 * Represents the result of attempting to falsify a hypothesis.
 * From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification
 */
exports.FalsificationResult = zod_1.z.object({
    hypothesis_id: zod_1.z.string().uuid().describe("ID of the hypothesis being tested"),
    // Falsification status
    falsified: zod_1.z.boolean().describe("Whether the hypothesis was falsified"),
    // Metrics
    degree_of_violation: zod_1.z.number().optional()
        .describe("Degree of logical constraint violation against known physics"),
    hypothesis_robustness_score: zod_1.z.number()
        .min(0, "Robustness score must be between 0 and 1")
        .max(1, "Robustness score must be between 0 and 1")
        .describe("HRS - Hypothesis Robustness Score (0-1)"),
    // Evidence
    falsifying_evidence: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Evidence that falsifies the hypothesis"),
    counter_examples: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Specific counter-examples found"),
});
/**
 * Epistemic Audit Result Schema
 *
 * Complete result from Phase I: Epistemic Audit and Falsification.
 * From RESE Manual §3.0
 */
exports.EpistemicAuditResult = zod_1.z.object({
    // Phase identification
    phase: zod_1.z.literal('phase1_epistemic_audit')
        .describe("RESE phase identifier"),
    // Audit metadata
    audit_id: zod_1.z.string().uuid().describe("Unique identifier for this audit"),
    problem_description: zod_1.z.string()
        .min(1, "Problem description cannot be empty")
        .describe("Description of the problem being audited"),
    // Constraint hardening results
    hardened_constraints: zod_1.z.array(zod_1.z.object({
        category: exports.ConstraintCategory,
        description: zod_1.z.string(),
        formalized: zod_1.z.boolean(),
        lean4_theorem: zod_1.z.string().optional(),
    })).optional().describe("Hardened constraints from Φ₁"),
    // Tacit assumptions mined
    tacit_assumptions: zod_1.z.array(exports.TacitAssumption).optional()
        .describe("Tacit assumptions inferred from Φ₁.₅"),
    // Contradictions detected
    contradictions: zod_1.z.array(exports.ContradictionDetection).optional()
        .describe("Contradictions detected by Φ₃"),
    // Falsification results
    falsification_results: zod_1.z.array(exports.FalsificationResult).optional()
        .describe("Results from red team protocol Φ₄"),
    // Overall audit metrics
    metrics: zod_1.z.object({
        total_assumptions_analyzed: zod_1.z.number().int().nonnegative().optional(),
        confirmed_contradictions: zod_1.z.number().int().nonnegative().optional(),
        hypotheses_falsified: zod_1.z.number().int().nonnegative().optional(),
        reduction_in_failure_rate: zod_1.z.number().optional() // Expected reduction in null results
    }).optional().describe("Audit performance metrics"),
    // Execution metadata
    metadata: zod_1.z.object({
        execution_time_ms: zod_1.z.number().optional()
            .describe("Actual audit execution time in milliseconds"),
        memory_used_mb: zod_1.z.number().optional()
            .describe("Memory usage in MB"),
        lean4_version: zod_1.z.string().optional()
            .describe("Lean 4 version used for formal verification"),
        epoch_number: zod_1.z.number().int().positive().optional()
            .describe("Current RESE epoch number"),
    }).optional().describe("Execution metadata"),
    // Correlation ID for distributed tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // UTC timestamp (Law of UTC)
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the audit result (ISO-8601)"),
});
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
exports.FunctionalDependencyGraph = zod_1.z.object({
    nodes: zod_1.z.array(zod_1.z.object({
        id: zod_1.z.string(),
        type: zod_1.z.string(),
        description: zod_1.z.string().optional(),
    })).describe("Nodes in the dependency graph"),
    edges: zod_1.z.array(zod_1.z.object({
        source: zod_1.z.string(),
        target: zod_1.z.string(),
        relation_type: zod_1.z.string(),
        strength: zod_1.z.number().min(0).max(1).optional(),
    })).describe("Causal edges between nodes"),
    // Verification status
    verified_in_lean4: zod_1.z.boolean().optional()
        .describe("Whether FDG has been verified in Lean 4"),
});
/**
 * Cross-Domain Pattern Schema
 *
 * Represents a pattern identified from a different domain.
 * From RESE Manual §4.2: Cross-Domain Ontology Mapping
 */
exports.CrossDomainPattern = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the pattern"),
    // Source domain
    source_domain: zod_1.z.string()
        .min(1, "Source domain cannot be empty")
        .describe("Name of the domain the pattern comes from"),
    source_description: zod_1.z.string()
        .min(1, "Source description cannot be empty")
        .describe("Description of the pattern in its source domain"),
    // Target domain
    target_domain: zod_1.z.string()
        .min(1, "Target domain cannot be empty")
        .describe("Name of the domain where pattern is being applied"),
    target_description: zod_1.z.string()
        .min(1, "Target description cannot be empty")
        .describe("Description of how the pattern maps to target domain"),
    // Isomorphism validation
    mechanistic_isomorphism_score: zod_1.z.number()
        .min(0, "Isomorphism score must be between 0 and 1")
        .max(1, "Isomorphism score must be between 0 and 1")
        .describe("ℑ_mech - Mechanistic Isomorphism Validation Score (0-1)"),
    // Functional dependency graphs
    source_fdg: exports.FunctionalDependencyGraph.optional()
        .describe("Functional dependency graph of source solution"),
    target_fdg: exports.FunctionalDependencyGraph.optional()
        .describe("Functional dependency graph required by target problem"),
    fdg_overlap: zod_1.z.number().min(0).max(1).optional()
        .describe("Quantified overlap between source and target FDGs"),
    // Verification
    verified_predictive: zod_1.z.boolean().optional()
        .describe("Whether isomorphism has been verified to be predictive"),
});
/**
 * Inverted Constraint Schema
 *
 * Represents a constraint that has been inverted to define solution requirements.
 * From RESE Manual §4.3: Constraint Inversion and Parameter Space Rotation
 */
exports.InvertedConstraint = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the constraint"),
    // Original constraint
    original_constraint: zod_1.z.string()
        .min(1, "Original constraint cannot be empty")
        .describe("The original constraint being inverted"),
    // Inverted version
    inverted_constraint: zod_1.z.string()
        .min(1, "Inverted constraint cannot be empty")
        .describe("The inverted constraint defining solution requirements"),
    // Parameter space changes
    parameter_space_rotation: zod_1.z.string().optional()
        .describe("Description of how parameter space is rotated"),
    // Formalization
    formalized_in_lean4: zod_1.z.boolean().optional()
        .describe("Whether inverted constraint is formalized in Lean 4"),
});
/**
 * Isomorphic Mapping Result Schema
 *
 * Complete result from Phase II: Isomorphic Resonance and Constraint Inversion.
 * From RESE Manual §4.0
 */
exports.IsomorphicMapping = zod_1.z.object({
    // Phase identification
    phase: zod_1.z.literal('phase2_isomorphic_mapping')
        .describe("RESE phase identifier"),
    // Mapping metadata
    mapping_id: zod_1.z.string().uuid().describe("Unique identifier for this mapping"),
    problem_description: zod_1.z.string()
        .min(1, "Problem description cannot be empty")
        .describe("Description of the problem requiring isomorphic solution"),
    // Cross-domain patterns identified
    cross_domain_patterns: zod_1.z.array(exports.CrossDomainPattern).optional()
        .describe("Patterns identified from other domains (Ψ₂)"),
    // Inverted constraints
    inverted_constraints: zod_1.z.array(exports.InvertedConstraint).optional()
        .describe("Constraints inverted to define solution architecture (Ψ₃)"),
    // Overall mapping metrics
    metrics: zod_1.z.object({
        total_domains_searched: zod_1.z.number().int().nonnegative().optional(),
        patterns_identified: zod_1.z.number().int().nonnegative().optional(),
        high_isomorphism_count: zod_1.z.number().int().nonnegative().optional(),
        average_isomorphism_score: zod_1.z.number().min(0).max(1).optional(),
    }).optional().describe("Mapping performance metrics"),
    // Execution metadata
    metadata: zod_1.z.object({
        execution_time_ms: zod_1.z.number().optional()
            .describe("Actual mapping execution time in milliseconds"),
        memory_used_mb: zod_1.z.number().optional()
            .describe("Memory usage in MB"),
        lean4_version: zod_1.z.string().optional()
            .describe("Lean 4 version used for verification"),
        epoch_number: zod_1.z.number().int().positive().optional()
            .describe("Current RESE epoch number"),
    }).optional().describe("Execution metadata"),
    // Correlation ID for distributed tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // UTC timestamp (Law of UTC)
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the mapping result (ISO-8601)"),
});
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
exports.SearchTreeNode = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the node"),
    // Node state
    visit_count: zod_1.z.number().int().nonnegative()
        .describe("Number of times this node has been visited"),
    value: zod_1.z.number()
        .describe("Expected value of this node"),
    // MCTS-specific metrics
    ucb1_score: zod_1.z.number().optional()
        .describe("UCB1 score for node selection"),
    // Node content
    hypothesis: zod_1.z.string().optional()
        .describe("Hypothesis represented by this node"),
    // Tree structure
    parent_id: zod_1.z.string().uuid().optional()
        .describe("ID of parent node"),
    child_ids: zod_1.z.array(zod_1.z.string().uuid()).optional()
        .describe("IDs of child nodes"),
});
/**
 * Hypothesis Schema
 *
 * Represents a hypothesis generated during MCTS search.
 */
exports.Hypothesis = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the hypothesis"),
    statement: zod_1.z.string()
        .min(1, "Hypothesis statement cannot be empty")
        .describe("The hypothesis statement"),
    // Validation metrics
    confidence: zod_1.z.number()
        .min(0, "Confidence must be between 0 and 1")
        .max(1, "Confidence must be between 0 and 1")
        .describe("Confidence in this hypothesis (0-1)"),
    expected_value: zod_1.z.number().optional()
        .describe("Expected value from MCTS simulation"),
    visit_count: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of times this hypothesis was explored"),
    // Status
    validated: zod_1.z.boolean().optional()
        .describe("Whether hypothesis has been validated"),
    falsified: zod_1.z.boolean().optional()
        .describe("Whether hypothesis has been falsified"),
});
/**
 * Validation Metrics Schema
 *
 * Represents metrics for hypothesis validation.
 * From RESE Manual §5.2: High-Entropy Data Analysis
 */
exports.ValidationMetrics = zod_1.z.object({
    // Anomaly Characterization Index (ACI) components
    disorder_entropy: zod_1.z.number().nonnegative().optional()
        .describe("𝔔_D - Disorder Entropy measuring randomness in data"),
    causal_coherence: zod_1.z.number().min(0).max(1).optional()
        .describe("𝔙_C - Causal Coherence measuring correlation strength"),
    aci_score: zod_1.z.number().nonnegative().optional()
        .describe("Composite ACI score (higher = more anomalous)"),
    // Convergence metrics
    convergence_rate: zod_1.z.number().min(0).max(1).optional()
        .describe("Rate of convergence to solution"),
    epochs_to_convergence: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of epochs until convergence"),
    // Predictive efficacy
    predictive_accuracy: zod_1.z.number().min(0).max(1).optional()
        .describe("Accuracy of predictions made by this hypothesis"),
    reduction_in_aci: zod_1.z.number().optional()
        .describe("Reduction in ACI relative to incumbent paradigm"),
});
/**
 * MCTS Search Result Schema
 *
 * Complete result from Phase III: Monte Carlo Metacognitive Refinement.
 * From RESE Manual §5.0
 */
exports.MCTSSearchResult = zod_1.z.object({
    // Phase identification
    phase: zod_1.z.literal('phase3_mcts_refinement')
        .describe("RESE phase identifier"),
    // Search metadata
    search_id: zod_1.z.string().uuid().describe("Unique identifier for this search"),
    problem_description: zod_1.z.string()
        .min(1, "Problem description cannot be empty")
        .describe("Description of the problem being solved"),
    // Search tree
    search_tree: zod_1.z.object({
        root_node: exports.SearchTreeNode.describe("Root node of the search tree"),
        total_nodes: zod_1.z.number().int().positive().describe("Total nodes in tree"),
        max_depth: zod_1.z.number().int().positive().describe("Maximum depth of tree"),
    }).optional().describe("The MC-NEST search tree"),
    // Hypotheses generated
    hypotheses: zod_1.z.array(exports.Hypothesis).optional()
        .describe("Hypotheses generated during search"),
    // Top hypotheses
    top_hypotheses: zod_1.z.array(exports.Hypothesis).optional()
        .describe("Top-ranked hypotheses by expected value"),
    // Validation metrics
    validation_metrics: exports.ValidationMetrics.optional()
        .describe("Validation metrics for top hypotheses"),
    // Convergence status
    converged: zod_1.z.boolean().optional()
        .describe("Whether the search converged to a solution"),
    convergence_criteria_met: zod_1.z.string().optional()
        .describe("Description of convergence criteria if met"),
    // Overall search metrics
    metrics: zod_1.z.object({
        total_simulations: zod_1.z.number().int().nonnegative().optional(),
        hypotheses_tested: zod_1.z.number().int().nonnegative().optional(),
        execution_time_ms: zod_1.z.number().optional(),
        memory_used_mb: zod_1.z.number().optional(),
        epoch_number: zod_1.z.number().int().positive().optional(),
    }).optional().describe("Search performance metrics"),
    // Execution metadata
    metadata: zod_1.z.object({
        lean4_version: zod_1.z.string().optional()
            .describe("Lean 4 version used for formal verification"),
        mcts_strategy: zod_1.z.string().optional()
            .describe("MCTS exploration strategy used"),
    }).optional().describe("Execution metadata"),
    // Correlation ID for distributed tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // UTC timestamp (Law of UTC)
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the search result (ISO-8601)"),
});
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
exports.ParadigmShift = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the paradigm shift"),
    // Old paradigm
    incumbent_paradigm: zod_1.z.string()
        .min(1, "Incumbent paradigm description cannot be empty")
        .describe("Description of the incumbent paradigm"),
    // New paradigm
    new_paradigm: zod_1.z.string()
        .min(1, "New paradigm description cannot be empty")
        .describe("Description of the new paradigm"),
    // Justification
    anomalies_explained: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Anomalies that the new paradigm explains"),
    predictive_improvements: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Predictive improvements over incumbent paradigm"),
    // Validation
    validated: zod_1.z.boolean().optional()
        .describe("Whether paradigm shift has been validated"),
    validation_criteria: zod_1.z.string().optional()
        .describe("Criteria used to validate the paradigm shift"),
});
/**
 * Synthesized Knowledge Schema
 *
 * Represents knowledge synthesized from the RESE process.
 */
exports.SynthesizedKnowledge = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the knowledge"),
    knowledge_type: zod_1.z.enum([
        'theoretical_model',
        'algorithm',
        'experimental_protocol',
        'verification_proof',
        'other',
    ]).describe("Type of synthesized knowledge"),
    title: zod_1.z.string()
        .min(1, "Title cannot be empty")
        .describe("Title of the knowledge"),
    description: zod_1.z.string()
        .min(1, "Description cannot be empty")
        .describe("Detailed description of the knowledge"),
    // Formal verification
    lean4_proof: zod_1.z.string().optional()
        .describe("Lean 4 proof if applicable"),
    verified: zod_1.z.boolean().optional()
        .describe("Whether knowledge has been formally verified"),
    // Applicability
    applicable_domains: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Domains where this knowledge is applicable"),
    // Dependencies
    depends_on: zod_1.z.array(zod_1.z.string().uuid()).optional()
        .describe("IDs of knowledge this depends on"),
});
/**
 * Architecture Assembly Result Schema
 *
 * Complete result from Phase IV: Architectural Synthesis and Validation.
 * From RESE Manual §6.0
 */
exports.ArchitectureAssembly = zod_1.z.object({
    // Phase identification
    phase: zod_1.z.literal('phase4_architecture_assembly')
        .describe("RESE phase identifier"),
    // Assembly metadata
    assembly_id: zod_1.z.string().uuid().describe("Unique identifier for this assembly"),
    problem_description: zod_1.z.string()
        .min(1, "Problem description cannot be empty")
        .describe("Description of the problem being solved"),
    // Paradigm shifts
    paradigm_shifts: zod_1.z.array(exports.ParadigmShift).optional()
        .describe("Paradigm shifts identified during synthesis"),
    // Synthesized knowledge
    synthesized_knowledge: zod_1.z.array(exports.SynthesizedKnowledge).optional()
        .describe("Knowledge synthesized from all RESE phases"),
    // Validation results
    validation_results: zod_1.z.object({
        predictive_model_efficacy: zod_1.z.boolean().optional()
            .describe("Whether predictive model is efficacious (see §6.3)"),
        aci_reduction_achieved: zod_1.z.number().optional()
            .describe("Reduction in Anomaly Characterization Index achieved"),
        testable_predictions_count: zod_1.z.number().int().nonnegative().optional()
            .describe("Number of testable predictions generated"),
        predictions_verified: zod_1.z.number().int().nonnegative().optional()
            .describe("Number of predictions that have been verified"),
        verification_success_rate: zod_1.z.number().min(0).max(1).optional()
            .describe("Success rate of verification (0-1)"),
    }).optional().describe("Validation results for the assembled architecture"),
    // Overall assembly metrics
    metrics: zod_1.z.object({
        total_epochs_completed: zod_1.z.number().int().positive().optional(),
        total_execution_time_ms: zod_1.z.number().optional(),
        peak_memory_mb: zod_1.z.number().optional(),
        lean4_theorems_proved: zod_1.z.number().int().nonnegative().optional(),
    }).optional().describe("Assembly performance metrics"),
    // Execution metadata
    metadata: zod_1.z.object({
        lean4_version: zod_1.z.string().optional()
            .describe("Lean 4 version used for formal verification"),
        rese_version: zod_1.z.string().optional()
            .describe("RESE implementation version"),
        converged: zod_1.z.boolean().optional()
            .describe("Whether RESE converged to a solution"),
    }).optional().describe("Execution metadata"),
    // Correlation ID for distributed tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // UTC timestamp (Law of UTC)
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the assembly result (ISO-8601)"),
});
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform raw RESE Phase I response to canonical EpistemicAuditResult
 */
function transformEpistemicAuditToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        phase: 'phase1_epistemic_audit',
        audit_id: rawResponse.audit_id || rawResponse.id,
        problem_description: rawResponse.problem_description || rawResponse.problem || '',
        tacit_assumptions: (rawResponse.tacit_assumptions || []).map((assumption) => ({
            id: assumption.id || generateUUID(),
            description: assumption.description || assumption.assumption || '',
            source_pattern: assumption.source_pattern,
            confidence_score: assumption.confidence_score || assumption.confidence || 0.5,
            supporting_evidence_count: assumption.supporting_evidence_count,
            formalized_in_lean4: assumption.formalized_in_lean4 || false,
            lean4_proposition: assumption.lean4_proposition,
        })),
        contradictions: (rawResponse.contradictions || []).map((contradiction) => ({
            id: contradiction.id || generateUUID(),
            fallacy_type: exports.LogicalFallacy.parse(contradiction.fallacy_type || contradiction.type || 'other'),
            contradiction_set_size: contradiction.contradiction_set_size || contradiction.set_size || 0,
            rollback_steps: contradiction.rollback_steps,
            affected_premises: contradiction.affected_premises,
            resolved: contradiction.resolved || false,
            resolution_strategy: contradiction.resolution_strategy,
        })),
        falsification_results: (rawResponse.falsification_results || []).map((result) => ({
            hypothesis_id: result.hypothesis_id,
            falsified: result.falsified || false,
            degree_of_violation: result.degree_of_violation,
            hypothesis_robustness_score: result.hypothesis_robustness_score || result.robustness || 0.5,
            falsifying_evidence: result.falsifying_evidence,
            counter_examples: result.counter_examples,
        })),
        metrics: rawResponse.metrics ? {
            total_assumptions_analyzed: rawResponse.metrics.total_assumptions,
            confirmed_contradictions: rawResponse.metrics.contradictions,
            hypotheses_falsified: rawResponse.metrics.falsified,
            reduction_in_failure_rate: rawResponse.metrics.failure_reduction,
        } : undefined,
        metadata: rawResponse.metadata ? {
            execution_time_ms: rawResponse.metadata.time || rawResponse.metadata.execution_time,
            memory_used_mb: rawResponse.metadata.memory,
            lean4_version: rawResponse.metadata.lean4_version,
            epoch_number: rawResponse.metadata.epoch,
        } : undefined,
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * Transform raw RESE Phase II response to canonical IsomorphicMapping
 */
function transformIsomorphicMappingToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        phase: 'phase2_isomorphic_mapping',
        mapping_id: rawResponse.mapping_id || rawResponse.id,
        problem_description: rawResponse.problem_description || rawResponse.problem || '',
        cross_domain_patterns: (rawResponse.cross_domain_patterns || rawResponse.patterns || []).map((pattern) => ({
            id: pattern.id || generateUUID(),
            source_domain: pattern.source_domain || pattern.source || '',
            source_description: pattern.source_description || pattern.source_desc || '',
            target_domain: pattern.target_domain || pattern.target || '',
            target_description: pattern.target_description || pattern.target_desc || '',
            mechanistic_isomorphism_score: pattern.mechanistic_isomorphism_score || pattern.isomorphism_score || 0.5,
            source_fdg: pattern.source_fdg,
            target_fdg: pattern.target_fdg,
            fdg_overlap: pattern.fdg_overlap,
            verified_predictive: pattern.verified_predictive || false,
        })),
        inverted_constraints: (rawResponse.inverted_constraints || rawResponse.constraints || []).map((constraint) => ({
            id: constraint.id || generateUUID(),
            original_constraint: constraint.original_constraint || constraint.original || '',
            inverted_constraint: constraint.inverted_constraint || constraint.inverted || '',
            parameter_space_rotation: constraint.parameter_space_rotation,
            formalized_in_lean4: constraint.formalized_in_lean4 || false,
        })),
        metrics: rawResponse.metrics ? {
            total_domains_searched: rawResponse.metrics.domains_searched,
            patterns_identified: rawResponse.metrics.patterns,
            high_isomorphism_count: rawResponse.metrics.high_isomorphisms,
            average_isomorphism_score: rawResponse.metrics.avg_score,
        } : undefined,
        metadata: rawResponse.metadata ? {
            execution_time_ms: rawResponse.metadata.time || rawResponse.metadata.execution_time,
            memory_used_mb: rawResponse.metadata.memory,
            lean4_version: rawResponse.metadata.lean4_version,
            epoch_number: rawResponse.metadata.epoch,
        } : undefined,
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * Transform raw RESE Phase III response to canonical MCTSSearchResult
 */
function transformMCTSSearchToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        phase: 'phase3_mcts_refinement',
        search_id: rawResponse.search_id || rawResponse.id,
        problem_description: rawResponse.problem_description || rawResponse.problem || '',
        search_tree: rawResponse.search_tree || rawResponse.tree ? {
            root_node: {
                id: rawResponse.search_tree.root_node?.id || rawResponse.tree.root_id || generateUUID(),
                visit_count: rawResponse.search_tree.root_node?.visit_count || rawResponse.tree.visits || 0,
                value: rawResponse.search_tree.root_node?.value || rawResponse.tree.value || 0,
                ucb1_score: rawResponse.search_tree.root_node?.ucb1,
                hypothesis: rawResponse.search_tree.root_node?.hypothesis,
                parent_id: rawResponse.search_tree.root_node?.parent_id,
                child_ids: rawResponse.search_tree.root_node?.child_ids,
            },
            total_nodes: rawResponse.search_tree.total_nodes || rawResponse.tree.nodes || 0,
            max_depth: rawResponse.search_tree.max_depth || rawResponse.tree.depth || 0,
        } : undefined,
        hypotheses: (rawResponse.hypotheses || []).map((hyp) => ({
            id: hyp.id || generateUUID(),
            statement: hyp.statement || hyp.hypothesis || '',
            confidence: hyp.confidence || 0.5,
            expected_value: hyp.expected_value || hyp.value,
            visit_count: hyp.visit_count || hyp.visits,
            validated: hyp.validated || false,
            falsified: hyp.falsified || false,
        })),
        top_hypotheses: (rawResponse.top_hypotheses || rawResponse.top || []).map((hyp) => ({
            id: hyp.id || generateUUID(),
            statement: hyp.statement || hyp.hypothesis || '',
            confidence: hyp.confidence || 0.5,
            expected_value: hyp.expected_value || hyp.value,
            visit_count: hyp.visit_count || hyp.visits,
            validated: hyp.validated || false,
            falsified: hyp.falsified || false,
        })),
        validation_metrics: rawResponse.validation_metrics ? {
            disorder_entropy: rawResponse.validation_metrics.disorder_entropy,
            causal_coherence: rawResponse.validation_metrics.causal_coherence,
            aci_score: rawResponse.validation_metrics.aci,
            convergence_rate: rawResponse.validation_metrics.convergence_rate,
            epochs_to_convergence: rawResponse.validation_metrics.epochs,
            predictive_accuracy: rawResponse.validation_metrics.accuracy,
            reduction_in_aci: rawResponse.validation_metrics.aci_reduction,
        } : undefined,
        converged: rawResponse.converged || false,
        convergence_criteria_met: rawResponse.convergence_criteria,
        metrics: rawResponse.metrics ? {
            total_simulations: rawResponse.metrics.simulations,
            hypotheses_tested: rawResponse.metrics.hypotheses_tested,
            execution_time_ms: rawResponse.metrics.time || rawResponse.metrics.execution_time,
            memory_used_mb: rawResponse.metrics.memory,
            epoch_number: rawResponse.metrics.epoch,
        } : undefined,
        metadata: rawResponse.metadata ? {
            lean4_version: rawResponse.metadata.lean4_version,
            mcts_strategy: rawResponse.metadata.strategy || rawResponse.metadata.mcts_strategy,
        } : undefined,
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * Transform raw RESE Phase IV response to canonical ArchitectureAssembly
 */
function transformArchitectureAssemblyToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        phase: 'phase4_architecture_assembly',
        assembly_id: rawResponse.assembly_id || rawResponse.id,
        problem_description: rawResponse.problem_description || rawResponse.problem || '',
        paradigm_shifts: (rawResponse.paradigm_shifts || rawResponse.shifts || []).map((shift) => ({
            id: shift.id || generateUUID(),
            incumbent_paradigm: shift.incumbent_paradigm || shift.incumbent || '',
            new_paradigm: shift.new_paradigm || shift.new || '',
            anomalies_explained: shift.anomalies_explained,
            predictive_improvements: shift.predictive_improvements,
            validated: shift.validated || false,
            validation_criteria: shift.validation_criteria,
        })),
        synthesized_knowledge: (rawResponse.synthesized_knowledge || rawResponse.knowledge || []).map((knowledge) => ({
            id: knowledge.id || generateUUID(),
            knowledge_type: knowledge.knowledge_type || knowledge.type || 'other',
            title: knowledge.title || '',
            description: knowledge.description || '',
            lean4_proof: knowledge.lean4_proof || knowledge.proof,
            verified: knowledge.verified || false,
            applicable_domains: knowledge.applicable_domains,
            depends_on: knowledge.depends_on,
        })),
        validation_results: rawResponse.validation_results ? {
            predictive_model_efficacy: rawResponse.validation_results.predictive_efficacy,
            aci_reduction_achieved: rawResponse.validation_results.aci_reduction,
            testable_predictions_count: rawResponse.validation_results.predictions_count,
            predictions_verified: rawResponse.validation_results.predictions_verified,
            verification_success_rate: rawResponse.validation_results.success_rate,
        } : undefined,
        metrics: rawResponse.metrics ? {
            total_epochs_completed: rawResponse.metrics.epochs,
            total_execution_time_ms: rawResponse.metrics.time || rawResponse.metrics.execution_time,
            peak_memory_mb: rawResponse.metrics.peak_memory,
            lean4_theorems_proved: rawResponse.metrics.theorems,
        } : undefined,
        metadata: rawResponse.metadata ? {
            lean4_version: rawResponse.metadata.lean4_version,
            rese_version: rawResponse.metadata.rese_version,
            converged: rawResponse.metadata.converged || rawResponse.converged,
        } : undefined,
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate an EpistemicAuditResult against the schema
 */
function validateEpistemicAuditResult(data) {
    const result = exports.EpistemicAuditResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an IsomorphicMapping against the schema
 */
function validateIsomorphicMapping(data) {
    const result = exports.IsomorphicMapping.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an MCTSSearchResult against the schema
 */
function validateMCTSSearchResult(data) {
    const result = exports.MCTSSearchResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an ArchitectureAssembly against the schema
 */
function validateArchitectureAssembly(data) {
    const result = exports.ArchitectureAssembly.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================
 */
/**
 * Generate a UUID v4
 */
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
/**
 * Create a UTC timestamp in ISO-8601 format
 */
function createUTCTimestamp() {
    return new Date().toISOString();
}
/**
 * Create a correlation ID (UUID v4)
 */
function createCorrelationId() {
    return generateUUID();
}
/**
 * ============================================================================
 * EXAMPLE DATA
 * ============================================================================
 */
/**
 * Example RESE data for testing and documentation
 */
exports.RESEExamples = {
    validEpistemicAuditResult: {
        phase: 'phase1_epistemic_audit',
        audit_id: '550e8400-e29b-41d4-a716-446655440000',
        problem_description: 'LENR thermal coefficient inconsistency',
        tacit_assumptions: [
            {
                id: '650e8400-e29b-41d4-a716-446655440001',
                description: 'Lattice defects are uniformly distributed',
                source_pattern: '50% null results in palladium-deuterium systems',
                confidence_score: 0.85,
                supporting_evidence_count: 23,
                formalized_in_lean4: true,
                lean4_proposition: 'theorem lattice_defects_uniform : ∀ (x : Lattice), ...',
            },
        ],
        contradictions: [
            {
                id: '750e8400-e29b-41d4-a716-446655440001',
                fallacy_type: 'circulus_in_probando',
                contradiction_set_size: 3,
                rollback_steps: 2,
                affected_premises: ['premise-1', 'premise-3', 'premise-7'],
                resolved: false,
            },
        ],
        falsification_results: [
            {
                hypothesis_id: '850e8400-e29b-41d4-a716-446655440001',
                falsified: true,
                degree_of_violation: 0.73,
                hypothesis_robustness_score: 0.27,
                falsifying_evidence: ['Experimental data from NASA LCF experiments'],
                counter_examples: ['Case where lattice confinement produced no excess heat'],
            },
        ],
        metrics: {
            total_assumptions_analyzed: 15,
            confirmed_contradictions: 3,
            hypotheses_falsified: 7,
            reduction_in_failure_rate: 0.35,
        },
        metadata: {
            execution_time_ms: 15420,
            memory_used_mb: 512,
            lean4_version: '4.7.0',
            epoch_number: 1,
        },
        correlation_id: '550e8400-e29b-41d4-a716-446655440000',
        timestamp: '2025-02-04T12:34:56.789Z',
    },
    validIsomorphicMapping: {
        phase: 'phase2_isomorphic_mapping',
        mapping_id: '550e8400-e29b-41d4-a716-446655440002',
        problem_description: 'Need isolated state with local computation and controlled release',
        cross_domain_patterns: [
            {
                id: '650e8400-e29b-41d4-a716-446655440002',
                source_domain: 'Homomorphic Encryption',
                source_description: 'Encrypted data remains isolated during computation',
                target_domain: 'Lattice Confinement Fusion',
                target_description: 'Nuclear reactions occur in isolated lattice sites',
                mechanistic_isomorphism_score: 0.89,
                source_fdg: {
                    nodes: [
                        { id: 'encrypted_state', type: 'state', description: 'Encrypted data' },
                        { id: 'computation', type: 'process', description: 'Operations on encrypted data' },
                    ],
                    edges: [
                        { source: 'encrypted_state', target: 'computation', relation_type: 'enables', strength: 1.0 },
                    ],
                    verified_in_lean4: true,
                },
                verified_predictive: true,
            },
        ],
        inverted_constraints: [
            {
                id: '750e8400-e29b-41d4-a716-446655440002',
                original_constraint: 'Thermal energy dissipates immediately in the lattice',
                inverted_constraint: 'Thermal energy must be contained within lattice sites for controlled release',
                formalized_in_lean4: true,
            },
        ],
        metrics: {
            total_domains_searched: 12,
            patterns_identified: 5,
            high_isomorphism_count: 2,
            average_isomorphism_score: 0.76,
        },
        metadata: {
            execution_time_ms: 8730,
            memory_used_mb: 384,
            lean4_version: '4.7.0',
            epoch_number: 2,
        },
        correlation_id: '550e8400-e29b-41d4-a716-446655440002',
        timestamp: '2025-02-04T12:35:12.456Z',
    },
    validMCTSSearchResult: {
        phase: 'phase3_mcts_refinement',
        search_id: '550e8400-e29b-41d4-a716-446655440003',
        problem_description: 'Find optimal lattice configuration for LENR',
        search_tree: {
            root_node: {
                id: '650e8400-e29b-41d4-a716-446655440003',
                visit_count: 150,
                value: 0.73,
                ucb1_score: 1.42,
                hypothesis: 'Palladium-deuterium lattice with specific defect patterns',
            },
            total_nodes: 1247,
            max_depth: 8,
        },
        hypotheses: [
            {
                id: '750e8400-e29b-41d4-a716-446655440003',
                statement: 'Hexagonal close-packed lattice yields optimal confinement',
                confidence: 0.82,
                expected_value: 0.78,
                visit_count: 45,
                validated: true,
                falsified: false,
            },
        ],
        top_hypotheses: [
            {
                id: '850e8400-e29b-41d4-a716-446655440003',
                statement: 'Specific loading ratio maximizes excess heat',
                confidence: 0.91,
                expected_value: 0.87,
                visit_count: 67,
                validated: false,
                falsified: false,
            },
        ],
        validation_metrics: {
            disorder_entropy: 0.34,
            causal_coherence: 0.78,
            aci_score: 0.26,
            convergence_rate: 0.85,
            epochs_to_convergence: 4,
            predictive_accuracy: 0.82,
            reduction_in_aci: 0.54,
        },
        converged: true,
        convergence_criteria_met: 'ACI reduction > 50% sustained over 3 epochs',
        metrics: {
            total_simulations: 5432,
            hypotheses_tested: 127,
            execution_time_ms: 45230,
            memory_used_mb: 1024,
            epoch_number: 4,
        },
        metadata: {
            lean4_version: '4.7.0',
            mcts_strategy: 'UCB1 with adaptive exploration',
        },
        correlation_id: '550e8400-e29b-41d4-a716-446655440003',
        timestamp: '2025-02-04T12:41:23.789Z',
    },
    validArchitectureAssembly: {
        phase: 'phase4_architecture_assembly',
        assembly_id: '550e8400-e29b-41d4-a716-446655440004',
        problem_description: 'Complete LENR theoretical framework',
        paradigm_shifts: [
            {
                id: '650e8400-e29b-41d4-a716-446655440004',
                incumbent_paradigm: 'LENR is theoretically impossible due to Coulomb barrier',
                new_paradigm: 'Lattice confinement enables quantum tunneling at room temperature',
                anomalies_explained: [
                    'Excess heat in palladium-deuterium systems',
                    'Transmutation products without high energy radiation',
                ],
                predictive_improvements: [
                    'Predicts optimal loading ratios',
                    'Explains reproducibility issues',
                ],
                validated: true,
                validation_criteria: 'Predictive model efficacy with 50%+ ACI reduction',
            },
        ],
        synthesized_knowledge: [
            {
                id: '750e8400-e29b-41d4-a716-446655440004',
                knowledge_type: 'theoretical_model',
                title: 'Lattice Confinement Fusion Theory',
                description: 'Complete theoretical framework for LENR based on lattice confinement',
                lean4_proof: 'theorem lattice_confinement_fusion : ...',
                verified: true,
                applicable_domains: ['Low-Energy Nuclear Reactions', 'Condensed Matter Nuclear Science'],
            },
        ],
        validation_results: {
            predictive_model_efficacy: true,
            aci_reduction_achieved: 0.62,
            testable_predictions_count: 15,
            predictions_verified: 12,
            verification_success_rate: 0.80,
        },
        metrics: {
            total_epochs_completed: 5,
            total_execution_time_ms: 124500,
            peak_memory_mb: 2048,
            lean4_theorems_proved: 47,
        },
        metadata: {
            lean4_version: '4.7.0',
            rese_version: '1.0.0',
            converged: true,
        },
        correlation_id: '550e8400-e29b-41d4-a716-446655440004',
        timestamp: '2025-02-04T12:55:34.123Z',
    },
};
//# sourceMappingURL=rese-canonical.js.map