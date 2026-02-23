/**
 * RESE Symbolic Constraint Engine (SCE) - Glue Layer Implementation
 *
 * Foundation for all RESE phases - enforces logical consistency using
 * formal logic and contradiction detection.
 *
 * Follows CLAUDE.md Laws:
 * - Law of Idempotency: All operations safe to run 100x
 * - Law of Configuration Explicitness: All config via env vars
 * - Circuit Breaker Pattern: Detect system failures
 * - Structured Logging: JSON with correlation_id
 * - Timeout Enforcement: Every operation has timeout
 *
 * Technical Manual Reference:
 * - Section 2.1: Symbolic Constraint Engine (SCE)
 * - Section 3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
 * - Section 3.1.5: Tacit Assumption Mining (Φ₁.₅)
 *
 * Source: bytecode analysis of rese/core/symbolic_constraint_engine.py
 * - 3 classes extracted: ConstraintType, Constraint, SymbolicConstraintEngine
 * - 2 functions extracted
 */
import { Logger } from './logger';
import { EpistemicAuditResult, TacitAssumption } from '../schemas/rese-canonical';
interface SCEConfig {
    TIMEOUT_MS: number;
    CONSTRAINT_TIMEOUT_MS: number;
    CONTRADICTION_DETECTION_TIMEOUT_MS: number;
    MAX_ITERATIONS: number;
    MAX_CONSTRAINTS: number;
    MAX_CONTRADICTION_SET_SIZE: number;
    CIRCUIT_BREAKER_THRESHOLD: number;
    CIRCUIT_BREAKER_TIMEOUT_MS: number;
    ENABLE_LEAN4_INTEGRATION: boolean;
    ENABLE_TACIT_ASSUMPTION_MINING: boolean;
}
/**
 * Constraint Type Enum
 *
 * From bytecode analysis: ConstraintType enum with HARD, SOFT values
 */
export declare enum ConstraintType {
    HARD = "hard",
    SOFT = "soft"
}
/**
 * Constraint Category
 *
 * Maps to canonical schema categories:
 * - Category A: Hard Parameter Inequality (Physical laws)
 * - Category B: Soft Statistical (Heuristics)
 * - Category C: Tacit Assumptions (Unstated beliefs)
 * - Category D: Inverted Constraints (Solution requirements)
 */
export declare enum ConstraintCategoryInternal {
    HARD_PARAMETER_INEQUALITY = "hard_parameter_inequality",
    SOFT_STATISTICAL = "soft_statistical",
    TACIT_ASSUMPTION = "tacit_assumption",
    INVERTED_CONSTRAINT = "inverted_constraint"
}
/**
 * Logical Fallacy Types
 *
 * From RESE Technical Manual §3.3: Formal Logic Audit
 */
export declare enum LogicalFallacyInternal {
    CIRCULUS_IN_PROBANDO = "circulus_in_probando",// Circular reasoning
    CONFIRMATION_BIAS = "confirmation_bias",
    HASTY_GENERALIZATION = "hasty_generalization",
    FALSE_CAUSE = "false_cause",
    AD_HOMINEM = "ad_hominem",
    STRAW_MAN = "straw_man",
    CONTRADICTION = "contradiction",
    INCONSISTENCY = "inconsistency",
    OTHER = "other"
}
/**
 * Constraint Data Structure
 *
 * From bytecode analysis: Constraint dataclass with fields:
 * - constraint_id: str
 * - type: ConstraintType
 * - description: str
 * - expression: Any
 * - dependencies: List[str]
 */
export interface Constraint {
    constraint_id: string;
    type: ConstraintType;
    category: ConstraintCategoryInternal;
    description: string;
    expression?: any;
    dependencies: string[];
    formalized_in_lean4?: boolean;
    lean4_theorem?: string;
    created_at: Date;
}
/**
 * Contradiction Pair
 *
 * Represents a pair of constraints that contradict each other
 */
export interface ContradictionPair {
    constraint1_id: string;
    constraint2_id: string;
    type: LogicalFallacyInternal;
    contradiction_set_size: number;
    rollback_steps: number;
    affected_premises: string[];
    detected_at: Date;
}
/**
 * Contradiction Detection Result
 *
 * Result from contradiction detection analysis
 */
export interface ContradictionDetectionResult {
    contradictions: ContradictionPair[];
    total_checked: number;
    contradiction_found: boolean;
    largest_contradiction_set: number;
    detection_time_ms: number;
}
/**
 * ContradictionDetector Class
 *
 * Identifies logical contradictions in constraint sets.
 * From bytecode analysis: One of 3 classes in symbolic_constraint_engine.py
 *
 * Algorithms:
 * - Naive O(n²) pairwise comparison (baseline)
 * - DITO optimization (O(n log n)) - to be implemented separately
 *
 * Follows Law of Idempotency: Safe to run multiple times
 */
export declare class ContradictionDetector {
    private logger;
    private config;
    private circuitBreaker;
    constructor(logger: Logger, config: SCEConfig);
    /**
     * Detect contradictions in constraint set
     *
     * Law of Timeout Enforcement: Enforced timeout via circuit breaker
     * Law of Idempotency: Pure function, safe to retry
     *
     * @param constraints - Set of constraints to check
     * @param correlationId - Distributed tracing correlation ID
     * @returns Contradiction detection result
     */
    detectContradictions(constraints: Constraint[], correlationId: string): Promise<ContradictionDetectionResult>;
    /**
     * Internal contradiction detection implementation
     *
     * Uses naive O(n²) pairwise comparison.
     * DITO optimization (O(n log n)) will be implemented separately.
     *
     * @private
     */
    private detectContradictionsInternal;
    /**
     * Check if two constraints contradict each other
     *
     * @private
     */
    private checkPairwiseContradiction;
    /**
     * Check if description2 is a direct negation of description1
     *
     * @private
     */
    private isDirectNegation;
    /**
     * Check for circular dependency between constraints
     *
     * @private
     */
    private hasCircularDependency;
    /**
     * Check if constraints have mismatched types on same premise
     *
     * @private
     */
    private isHardSoftMismatch;
    /**
     * Calculate contradiction set size (CSS)
     *
     * From RESE Manual §3.3: Contradiction Set Size (CSS)
     *
     * @private
     */
    private calculateContradictionSetSize;
    /**
     * Calculate rollback steps to root premise
     *
     * From RESE Manual §3.3: Rollback steps for root premise isolation
     *
     * @private
     */
    private calculateRollbackSteps;
    /**
     * Get circuit breaker stats
     */
    getStats(): import("./circuit-breaker").CircuitBreakerStats;
    /**
     * Reset circuit breaker
     */
    reset(): void;
}
/**
 * ConsistencyChecker Class
 *
 * Verifies internal consistency of constraint sets.
 * From bytecode analysis: One of 3 classes in symbolic_constraint_engine.py
 *
 * Follows Law of Idempotency: Pure function, safe to retry
 */
export declare class ConsistencyChecker {
    private logger;
    private config;
    constructor(logger: Logger, config: SCEConfig);
    /**
     * Check consistency of constraint set
     *
     * @param constraints - Set of constraints to check
     * @param correlationId - Distributed tracing correlation ID
     * @returns Consistency check result
     */
    checkConsistency(constraints: Constraint[], correlationId: string): Promise<{
        consistent: boolean;
        issues: string[];
        checked_at: Date;
    }>;
    /**
     * Detect dependency cycles using DFS
     *
     * @private
     */
    private detectDependencyCycles;
}
/**
 * SymbolicConstraintEngine Class
 *
 * Main engine class for RESE Phase I: Epistemic Audit.
 * From bytecode analysis: Main class in symbolic_constraint_engine.py
 *
 * Responsibilities:
 * - Constraint management (add, remove, query)
 * - Contradiction detection
 * - Consistency checking
 * - Tacit assumption mining (Φ₁.₅)
 * - Lean 4 integration (formal verification)
 *
 * Follows CLAUDE.md Laws:
 * - Law of Idempotency: All operations safe to run 100x
 * - Law of Configuration Explicitness: Config from env vars
 * - Circuit Breaker Pattern: Failure detection
 * - Structured Logging: JSON with correlation_id
 * - Timeout Enforcement: All operations have timeouts
 */
export declare class SymbolicConstraintEngine {
    private logger;
    private config;
    private constraints;
    private contradictionDetector;
    private consistencyChecker;
    constructor();
    /**
     * Add a constraint to the engine
     *
     * Law of Idempotency: Check before create (UPSERT logic)
     *
     * @param constraint - Constraint to add
     * @param correlationId - Distributed tracing correlation ID
     * @returns True if added, false if updated
     */
    addConstraint(constraint: Constraint, correlationId: string): Promise<{
        added: boolean;
        updated: boolean;
    }>;
    /**
     * Remove a constraint from the engine
     *
     * Law of Idempotency: Safe to run multiple times
     *
     * @param constraintId - ID of constraint to remove
     * @param correlationId - Distributed tracing correlation ID
     * @returns True if removed, false if not found
     */
    removeConstraint(constraintId: string, correlationId: string): Promise<{
        removed: boolean;
    }>;
    /**
     * Get a constraint by ID
     *
     * @param constraintId - ID of constraint to retrieve
     * @returns Constraint or null if not found
     */
    getConstraint(constraintId: string): Constraint | null;
    /**
     * Get all constraints
     *
     * @returns Array of all constraints
     */
    getAllConstraints(): Constraint[];
    /**
     * Get constraints by type
     *
     * @param type - Constraint type filter
     * @returns Array of matching constraints
     */
    getConstraintsByType(type: ConstraintType): Constraint[];
    /**
     * Get constraints by category
     *
     * @param category - Constraint category filter
     * @returns Array of matching constraints
     */
    getConstraintsByCategory(category: ConstraintCategoryInternal): Constraint[];
    /**
     * Detect contradictions in the current constraint set
     *
     * From RESE Manual §3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
     *
     * @param correlationId - Distributed tracing correlation ID
     * @returns Contradiction detection result
     */
    detectContradictions(correlationId: string): Promise<ContradictionDetectionResult>;
    /**
     * Check consistency of the current constraint set
     *
     * @param correlationId - Distributed tracing correlation ID
     * @returns Consistency check result
     */
    checkConsistency(correlationId: string): Promise<{
        consistent: boolean;
        issues: string[];
        checked_at: Date;
    }>;
    /**
     * Mine tacit assumptions from failure patterns
     *
     * From RESE Manual §3.1.5: Tacit Assumption Mining (Φ₁.₅)
     *
     * "Inverse Inference Analysis: The DEE analyzes high-entropy data
     * (e.g., 50% null results) via statistical correlation. This performs
     * inverse inference, inferring the unstated rule set (C_tacit) by
     * correlating patterns of failure with known, unmeasured variables."
     *
     * @param failurePatterns - Patterns of failure to analyze
     * @param correlationId - Distributed tracing correlation ID
     * @returns Mined tacit assumptions
     */
    mineTacitAssumptions(failurePatterns: Array<{
        pattern_description: string;
        failure_rate: number;
        data_points: number;
    }>, correlationId: string): Promise<TacitAssumption[]>;
    /**
     * Infer assumption from failure pattern description
     *
     * @private
     */
    private inferAssumptionFromPattern;
    /**
     * Perform full Epistemic Audit (Phase I)
     *
     * From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification
     *
     * Combines:
     * - Φ₁: Initial Hypothesis Cluster Definition (Constraint Hardening)
     * - Φ₁.₅: Tacit Assumption Mining
     * - Φ₃: Formal Logic Audit and Contradiction Detection
     *
     * @param problemDescription - Description of the problem to audit
     * @param failurePatterns - Failure patterns for tacit assumption mining
     * @param correlationId - Distributed tracing correlation ID
     * @returns Canonical EpistemicAuditResult
     */
    performEpistemicAudit(problemDescription: string, failurePatterns: Array<{
        pattern_description: string;
        failure_rate: number;
        data_points: number;
    }>, correlationId: string): Promise<EpistemicAuditResult>;
    /**
     * Map internal logical fallacy to canonical schema
     *
     * @private
     */
    private mapLogicalFallacy;
    /**
     * Clear all constraints from the engine
     *
     * Useful for testing and isolation
     */
    clear(): void;
    /**
     * Get engine statistics
     */
    getStats(): {
        constraint_count: number;
        hard_constraints: number;
        soft_constraints: number;
        contradiction_detector_stats: any;
    };
    /**
     * Reset circuit breakers
     */
    resetCircuitBreakers(): void;
}
export default SymbolicConstraintEngine;
//# sourceMappingURL=rese-sce.d.ts.map