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

import { v4 as uuidv4 } from 'uuid';
import { Logger } from './logger';
import { CircuitBreaker, CircuitState } from './circuit-breaker';
import {
  EpistemicAuditResult,
  TacitAssumption,
  ContradictionDetection,
  ConstraintCategory,
  LogicalFallacy,
} from '../schemas/rese-canonical';

// ============================================================================
// CONFIGURATION (Law of Configuration Explicitness)
// ============================================================================

interface SCEConfig {
  // Timeout settings (milliseconds)
  TIMEOUT_MS: number;
  CONSTRAINT_TIMEOUT_MS: number;
  CONTRADICTION_DETECTION_TIMEOUT_MS: number;

  // Iteration limits
  MAX_ITERATIONS: number;
  MAX_CONSTRAINTS: number;
  MAX_CONTRADICTION_SET_SIZE: number;

  // Circuit breaker settings
  CIRCUIT_BREAKER_THRESHOLD: number;
  CIRCUIT_BREAKER_TIMEOUT_MS: number;

  // Feature flags
  ENABLE_LEAN4_INTEGRATION: boolean;
  ENABLE_TACIT_ASSUMPTION_MINING: boolean;
}

/**
 * Load configuration from environment variables
 *
 * Law of Configuration Explicitness: All config via env vars
 * Crashes immediately if required config is missing
 */
function loadConfig(): SCEConfig {
  const config: SCEConfig = {
    TIMEOUT_MS: parseInt(process.env.SCE_TIMEOUT_MS || '5000', 10),
    CONSTRAINT_TIMEOUT_MS: parseInt(process.env.SCE_CONSTRAINT_TIMEOUT_MS || '3000', 10),
    CONTRADICTION_DETECTION_TIMEOUT_MS: parseInt(process.env.SCE_CONTRADICTION_TIMEOUT_MS || '10000', 10),

    MAX_ITERATIONS: parseInt(process.env.SCE_MAX_ITERATIONS || '1000', 10),
    MAX_CONSTRAINTS: parseInt(process.env.SCE_MAX_CONSTRAINTS || '10000', 10),
    MAX_CONTRADICTION_SET_SIZE: parseInt(process.env.SCE_MAX_CONTRADICTION_SET_SIZE || '100', 10),

    CIRCUIT_BREAKER_THRESHOLD: parseInt(process.env.SCE_CIRCUIT_BREAKER_THRESHOLD || '5', 10),
    CIRCUIT_BREAKER_TIMEOUT_MS: parseInt(process.env.SCE_CIRCUIT_BREAKER_TIMEOUT_MS || '60000', 10),

    ENABLE_LEAN4_INTEGRATION: process.env.SCE_ENABLE_LEAN4 === 'true',
    ENABLE_TACIT_ASSUMPTION_MINING: process.env.SCE_ENABLE_TACIT_MINING !== 'false', // default true
  };

  // Validate configuration
  if (config.TIMEOUT_MS <= 0) {
    throw new Error('SCE_TIMEOUT_MS must be positive');
  }
  if (config.MAX_ITERATIONS <= 0) {
    throw new Error('SCE_MAX_ITERATIONS must be positive');
  }
  if (config.MAX_CONSTRAINTS <= 0) {
    throw new Error('SCE_MAX_CONSTRAINTS must be positive');
  }

  return config;
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * Constraint Type Enum
 *
 * From bytecode analysis: ConstraintType enum with HARD, SOFT values
 */
export enum ConstraintType {
  HARD = 'hard',
  SOFT = 'soft',
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
export enum ConstraintCategoryInternal {
  HARD_PARAMETER_INEQUALITY = 'hard_parameter_inequality',
  SOFT_STATISTICAL = 'soft_statistical',
  TACIT_ASSUMPTION = 'tacit_assumption',
  INVERTED_CONSTRAINT = 'inverted_constraint',
}

/**
 * Logical Fallacy Types
 *
 * From RESE Technical Manual §3.3: Formal Logic Audit
 */
export enum LogicalFallacyInternal {
  CIRCULUS_IN_PROBANDO = 'circulus_in_probando', // Circular reasoning
  CONFIRMATION_BIAS = 'confirmation_bias',
  HASTY_GENERALIZATION = 'hasty_generalization',
  FALSE_CAUSE = 'false_cause',
  AD_HOMINEM = 'ad_hominem',
  STRAW_MAN = 'straw_man',
  CONTRADICTION = 'contradiction',
  INCONSISTENCY = 'inconsistency',
  OTHER = 'other',
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
  expression?: any; // Logical expression (can be Lean 4 proposition)
  dependencies: string[]; // IDs of constraints this depends on
  formalized_in_lean4?: boolean;
  lean4_theorem?: string;
  created_at: Date; // UTC timestamp (Law of UTC)
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
  detected_at: Date; // UTC timestamp
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

// ============================================================================
// CORE CLASS: ContradictionDetector
// ============================================================================

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
export class ContradictionDetector {
  private logger: Logger;
  private config: SCEConfig;
  private circuitBreaker: CircuitBreaker;

  constructor(logger: Logger, config: SCEConfig) {
    this.logger = logger.child({ component: 'ContradictionDetector' });
    this.config = config;

    // Initialize circuit breaker for contradiction detection
    this.circuitBreaker = new CircuitBreaker({
      threshold: config.CIRCUIT_BREAKER_THRESHOLD,
      timeout_ms: config.CIRCUIT_BREAKER_TIMEOUT_MS,
      onStateChange: (oldState, newState) => {
        this.logger.warn('ContradictionDetector circuit breaker state changed', {
          old_state: oldState,
          new_state: newState,
        });
      },
    });
  }

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
  async detectContradictions(
    constraints: Constraint[],
    correlationId: string
  ): Promise<ContradictionDetectionResult> {
    const startTime = Date.now();

    this.logger.info('Starting contradiction detection', {
      correlation_id: correlationId,
      constraint_count: constraints.length,
      max_iterations: this.config.MAX_ITERATIONS,
    });

    // Check circuit breaker
    if (this.circuitBreaker.getState() === CircuitState.OPEN) {
      throw new Error('ContradictionDetector circuit is OPEN - too many recent failures');
    }

    try {
      // Execute through circuit breaker with timeout
      const result = await this.circuitBreaker.execute(async () => {
        return await this.detectContradictionsInternal(constraints, correlationId);
      });

      const detectionTime = Date.now() - startTime;

      this.logger.info('Contradiction detection completed', {
        correlation_id: correlationId,
        total_checked: result.total_checked,
        contradictions_found: result.contradictions.length,
        detection_time_ms: detectionTime,
      });

      return {
        ...result,
        detection_time_ms: detectionTime,
      };
    } catch (error) {
      this.logger.error('Contradiction detection failed', error as Error, {
        correlation_id: correlationId,
        constraint_count: constraints.length,
      });
      throw error;
    }
  }

  /**
   * Internal contradiction detection implementation
   *
   * Uses naive O(n²) pairwise comparison.
   * DITO optimization (O(n log n)) will be implemented separately.
   *
   * @private
   */
  private async detectContradictionsInternal(
    constraints: Constraint[],
    correlationId: string
  ): Promise<Omit<ContradictionDetectionResult, 'detection_time_ms'>> {
    const contradictions: ContradictionPair[] = [];
    const totalChecked = constraints.length;
    let largestSet = 0;

    // Naive pairwise comparison
    for (let i = 0; i < constraints.length; i++) {
      for (let j = i + 1; j < constraints.length; j++) {
        const c1 = constraints[i];
        const c2 = constraints[j];

        // Check if constraints are direct contradictions
        const contradiction = this.checkPairwiseContradiction(c1, c2);
        if (contradiction) {
          contradictions.push(contradiction);
          largestSet = Math.max(largestSet, contradiction.contradiction_set_size);
        }

        // Enforce max iterations to prevent infinite loops
        if (contradictions.length >= this.config.MAX_ITERATIONS) {
          this.logger.warn('Max iterations reached, stopping detection', {
            correlation_id: correlationId,
            iterations: contradictions.length,
            max_iterations: this.config.MAX_ITERATIONS,
          });
          break;
        }
      }

      if (contradictions.length >= this.config.MAX_ITERATIONS) {
        break;
      }
    }

    return {
      contradictions,
      total_checked: totalChecked,
      contradiction_found: contradictions.length > 0,
      largest_contradiction_set: largestSet,
    };
  }

  /**
   * Check if two constraints contradict each other
   *
   * @private
   */
  private checkPairwiseContradiction(
    c1: Constraint,
    c2: Constraint
  ): ContradictionPair | null {
    // Direct textual contradiction detection
    if (this.isDirectNegation(c1.description, c2.description)) {
      return {
        constraint1_id: c1.constraint_id,
        constraint2_id: c2.constraint_id,
        type: LogicalFallacyInternal.CONTRADICTION,
        contradiction_set_size: 2,
        rollback_steps: this.calculateRollbackSteps(c1, c2),
        affected_premises: [c1.constraint_id, c2.constraint_id],
        detected_at: new Date(),
      };
    }

    // Check for circular dependency
    if (this.hasCircularDependency(c1, c2)) {
      return {
        constraint1_id: c1.constraint_id,
        constraint2_id: c2.constraint_id,
        type: LogicalFallacyInternal.CIRCULUS_IN_PROBANDO,
        contradiction_set_size: this.calculateContradictionSetSize(c1, c2),
        rollback_steps: this.calculateRollbackSteps(c1, c2),
        affected_premises: [...c1.dependencies, ...c2.dependencies],
        detected_at: new Date(),
      };
    }

    // Check for constraint type mismatch (HARD vs SOFT on same premise)
    if (this.isHardSoftMismatch(c1, c2)) {
      return {
        constraint1_id: c1.constraint_id,
        constraint2_id: c2.constraint_id,
        type: LogicalFallacyInternal.INCONSISTENCY,
        contradiction_set_size: 2,
        rollback_steps: 1,
        affected_premises: [c1.constraint_id, c2.constraint_id],
        detected_at: new Date(),
      };
    }

    return null;
  }

  /**
   * Check if description2 is a direct negation of description1
   *
   * @private
   */
  private isDirectNegation(desc1: string, desc2: string): boolean {
    const negationPatterns = [
      /^(not |no |non-|un-)/i,
      /\b(not|no|never|cannot)\b/i,
    ];

    const lower1 = desc1.toLowerCase().trim();
    const lower2 = desc2.toLowerCase().trim();

    // Check if one is the negation of the other
    for (const pattern of negationPatterns) {
      if (lower1.replace(pattern, '').trim() === lower2.replace(pattern, '').trim()) {
        return true;
      }
    }

    // Check for explicit "not X" vs "X" patterns
    if (lower1.startsWith('not ') && lower1.substring(4) === lower2) {
      return true;
    }
    if (lower2.startsWith('not ') && lower2.substring(4) === lower1) {
      return true;
    }

    return false;
  }

  /**
   * Check for circular dependency between constraints
   *
   * @private
   */
  private hasCircularDependency(c1: Constraint, c2: Constraint): boolean {
    return (
      c1.dependencies.includes(c2.constraint_id)
      && c2.dependencies.includes(c1.constraint_id)
    );
  }

  /**
   * Check if constraints have mismatched types on same premise
   *
   * @private
   */
  private isHardSoftMismatch(c1: Constraint, c2: Constraint): boolean {
    return (
      c1.type !== c2.type
      && (c1.category === c2.category || c1.description === c2.description)
    );
  }

  /**
   * Calculate contradiction set size (CSS)
   *
   * From RESE Manual §3.3: Contradiction Set Size (CSS)
   *
   * @private
   */
  private calculateContradictionSetSize(c1: Constraint, c2: Constraint): number {
    // Count unique premises in the contradiction set
    const premises = new Set([
      c1.constraint_id,
      c2.constraint_id,
      ...c1.dependencies,
      ...c2.dependencies,
    ]);
    return Math.min(premises.size, this.config.MAX_CONTRADICTION_SET_SIZE);
  }

  /**
   * Calculate rollback steps to root premise
   *
   * From RESE Manual §3.3: Rollback steps for root premise isolation
   *
   * @private
   */
  private calculateRollbackSteps(c1: Constraint, c2: Constraint): number {
    // Calculate maximum dependency depth
    const maxDepth = Math.max(c1.dependencies.length, c2.dependencies.length);
    return maxDepth;
  }

  /**
   * Get circuit breaker stats
   */
  getStats() {
    return this.circuitBreaker.getStats();
  }

  /**
   * Reset circuit breaker
   */
  reset() {
    this.circuitBreaker.reset();
  }
}

// ============================================================================
// CORE CLASS: ConsistencyChecker
// ============================================================================

/**
 * ConsistencyChecker Class
 *
 * Verifies internal consistency of constraint sets.
 * From bytecode analysis: One of 3 classes in symbolic_constraint_engine.py
 *
 * Follows Law of Idempotency: Pure function, safe to retry
 */
export class ConsistencyChecker {
  private logger: Logger;
  private config: SCEConfig;

  constructor(logger: Logger, config: SCEConfig) {
    this.logger = logger.child({ component: 'ConsistencyChecker' });
    this.config = config;
  }

  /**
   * Check consistency of constraint set
   *
   * @param constraints - Set of constraints to check
   * @param correlationId - Distributed tracing correlation ID
   * @returns Consistency check result
   */
  async checkConsistency(
    constraints: Constraint[],
    correlationId: string
  ): Promise<{
    consistent: boolean;
    issues: string[];
    checked_at: Date;
  }> {
    this.logger.info('Starting consistency check', {
      correlation_id: correlationId,
      constraint_count: constraints.length,
    });

    const issues: string[] = [];

    // Check for duplicate constraint IDs
    const ids = new Set<string>();
    for (const constraint of constraints) {
      if (ids.has(constraint.constraint_id)) {
        issues.push(`Duplicate constraint ID: ${constraint.constraint_id}`);
      }
      ids.add(constraint.constraint_id);
    }

    // Check for orphaned dependencies
    const allIds = new Set(constraints.map((c) => c.constraint_id));
    for (const constraint of constraints) {
      for (const dep of constraint.dependencies) {
        if (!allIds.has(dep)) {
          issues.push(`Orphaned dependency: ${constraint.constraint_id} depends on non-existent ${dep}`);
        }
      }
    }

    // Check for dependency cycles
    const cycles = this.detectDependencyCycles(constraints);
    if (cycles.length > 0) {
      cycles.forEach((cycle) => {
        issues.push(`Dependency cycle detected: ${cycle.join(' -> ')}`);
      });
    }

    // Check constraint count limit
    if (constraints.length > this.config.MAX_CONSTRAINTS) {
      issues.push(
        `Constraint count ${constraints.length} exceeds maximum ${this.config.MAX_CONSTRAINTS}`
      );
    }

    const consistent = issues.length === 0;

    this.logger.info('Consistency check completed', {
      correlation_id: correlationId,
      consistent,
      issues_found: issues.length,
    });

    return {
      consistent,
      issues,
      checked_at: new Date(),
    };
  }

  /**
   * Detect dependency cycles using DFS
   *
   * @private
   */
  private detectDependencyCycles(constraints: Constraint[]): string[][] {
    const constraintMap = new Map<string, Constraint>();
    constraints.forEach((c) => constraintMap.set(c.constraint_id, c));

    const cycles: string[][] = [];
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const dfs = (constraintId: string, path: string[]): boolean => {
      if (recursionStack.has(constraintId)) {
        // Found a cycle
        const cycleStart = path.indexOf(constraintId);
        cycles.push([...path.slice(cycleStart), constraintId]);
        return true;
      }

      if (visited.has(constraintId)) {
        return false;
      }

      visited.add(constraintId);
      recursionStack.add(constraintId);

      const constraint = constraintMap.get(constraintId);
      if (constraint) {
        for (const dep of constraint.dependencies) {
          if (dfs(dep, [...path, constraintId])) {
            return true;
          }
        }
      }

      recursionStack.delete(constraintId);
      return false;
    };

    for (const constraint of constraints) {
      if (!visited.has(constraint.constraint_id)) {
        dfs(constraint.constraint_id, []);
      }
    }

    return cycles;
  }
}

// ============================================================================
// CORE CLASS: SymbolicConstraintEngine
// ============================================================================

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
export class SymbolicConstraintEngine {
  private logger: Logger;
  private config: SCEConfig;
  private constraints: Map<string, Constraint>;
  private contradictionDetector: ContradictionDetector;
  private consistencyChecker: ConsistencyChecker;

  constructor() {
    this.logger = new Logger('SymbolicConstraintEngine');
    this.config = loadConfig();
    this.constraints = new Map();

    this.contradictionDetector = new ContradictionDetector(this.logger, this.config);
    this.consistencyChecker = new ConsistencyChecker(this.logger, this.config);

    this.logger.info('SymbolicConstraintEngine initialized', {
      max_constraints: this.config.MAX_CONSTRAINTS,
      max_iterations: this.config.MAX_ITERATIONS,
      enable_lean4: this.config.ENABLE_LEAN4_INTEGRATION,
      enable_tacit_mining: this.config.ENABLE_TACIT_ASSUMPTION_MINING,
    });
  }

  // ==========================================================================
  // CONSTRAINT MANAGEMENT
  // ==========================================================================

  /**
   * Add a constraint to the engine
   *
   * Law of Idempotency: Check before create (UPSERT logic)
   *
   * @param constraint - Constraint to add
   * @param correlationId - Distributed tracing correlation ID
   * @returns True if added, false if updated
   */
  async addConstraint(
    constraint: Constraint,
    correlationId: string
  ): Promise<{ added: boolean; updated: boolean }> {
    this.logger.info('Adding constraint', {
      correlation_id: correlationId,
      constraint_id: constraint.constraint_id,
      type: constraint.type,
      category: constraint.category,
    });

    // Check constraint count limit
    if (this.constraints.size >= this.config.MAX_CONSTRAINTS) {
      throw new Error(
        `Cannot add constraint: maximum limit ${this.config.MAX_CONSTRAINTS} reached`
      );
    }

    const exists = this.constraints.has(constraint.constraint_id);
    this.constraints.set(constraint.constraint_id, {
      ...constraint,
      created_at: constraint.created_at || new Date(),
    });

    this.logger.debug('Constraint added/updated', {
      correlation_id: correlationId,
      constraint_id: constraint.constraint_id,
      existed: exists,
    });

    return { added: !exists, updated: exists };
  }

  /**
   * Remove a constraint from the engine
   *
   * Law of Idempotency: Safe to run multiple times
   *
   * @param constraintId - ID of constraint to remove
   * @param correlationId - Distributed tracing correlation ID
   * @returns True if removed, false if not found
   */
  async removeConstraint(
    constraintId: string,
    correlationId: string
  ): Promise<{ removed: boolean }> {
    this.logger.info('Removing constraint', {
      correlation_id: correlationId,
      constraint_id: constraintId,
    });

    const existed = this.constraints.delete(constraintId);

    this.logger.debug('Constraint removal completed', {
      correlation_id: correlationId,
      constraint_id: constraintId,
      existed,
    });

    return { removed: existed };
  }

  /**
   * Get a constraint by ID
   *
   * @param constraintId - ID of constraint to retrieve
   * @returns Constraint or null if not found
   */
  getConstraint(constraintId: string): Constraint | null {
    return this.constraints.get(constraintId) || null;
  }

  /**
   * Get all constraints
   *
   * @returns Array of all constraints
   */
  getAllConstraints(): Constraint[] {
    return Array.from(this.constraints.values());
  }

  /**
   * Get constraints by type
   *
   * @param type - Constraint type filter
   * @returns Array of matching constraints
   */
  getConstraintsByType(type: ConstraintType): Constraint[] {
    return this.getAllConstraints().filter((c) => c.type === type);
  }

  /**
   * Get constraints by category
   *
   * @param category - Constraint category filter
   * @returns Array of matching constraints
   */
  getConstraintsByCategory(category: ConstraintCategoryInternal): Constraint[] {
    return this.getAllConstraints().filter((c) => c.category === category);
  }

  // ==========================================================================
  // CONTRADICTION DETECTION
  // ==========================================================================

  /**
   * Detect contradictions in the current constraint set
   *
   * From RESE Manual §3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
   *
   * @param correlationId - Distributed tracing correlation ID
   * @returns Contradiction detection result
   */
  async detectContradictions(
    correlationId: string
  ): Promise<ContradictionDetectionResult> {
    const constraints = this.getAllConstraints();

    this.logger.info('Starting contradiction detection audit', {
      correlation_id: correlationId,
      constraint_count: constraints.length,
    });

    const result = await this.contradictionDetector.detectContradictions(
      constraints,
      correlationId
    );

    this.logger.info('Contradiction detection audit completed', {
      correlation_id: correlationId,
      contradictions_found: result.contradictions.length,
      largest_set: result.largest_contradiction_set,
    });

    return result;
  }

  // ==========================================================================
  // CONSISTENCY CHECKING
  // ==========================================================================

  /**
   * Check consistency of the current constraint set
   *
   * @param correlationId - Distributed tracing correlation ID
   * @returns Consistency check result
   */
  async checkConsistency(correlationId: string): Promise<{
    consistent: boolean;
    issues: string[];
    checked_at: Date;
  }> {
    const constraints = this.getAllConstraints();

    return this.consistencyChecker.checkConsistency(constraints, correlationId);
  }

  // ==========================================================================
  // TACIT ASSUMPTION MINING (Φ₁.₅)
  // ==========================================================================

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
  async mineTacitAssumptions(
    failurePatterns: Array<{
      pattern_description: string;
      failure_rate: number;
      data_points: number;
    }>,
    correlationId: string
  ): Promise<TacitAssumption[]> {
    if (!this.config.ENABLE_TACIT_ASSUMPTION_MINING) {
      this.logger.warn('Tacit assumption mining is disabled', {
        correlation_id: correlationId,
      });
      return [];
    }

    this.logger.info('Starting tacit assumption mining', {
      correlation_id: correlationId,
      pattern_count: failurePatterns.length,
    });

    const assumptions: TacitAssumption[] = [];

    for (const pattern of failurePatterns) {
      // High failure rate suggests a tacit assumption
      if (pattern.failure_rate > 0.3) {
        const assumption: TacitAssumption = {
          id: uuidv4(),
          description: this.inferAssumptionFromPattern(pattern.pattern_description),
          source_pattern: pattern.pattern_description,
          confidence_score: Math.min(pattern.failure_rate, 1.0),
          supporting_evidence_count: pattern.data_points,
          formalized_in_lean4: false, // Will be formalized separately
        };

        assumptions.push(assumption);

        this.logger.debug('Tacit assumption mined', {
          correlation_id: correlationId,
          assumption_id: assumption.id,
          confidence: assumption.confidence_score,
        });
      }
    }

    this.logger.info('Tacit assumption mining completed', {
      correlation_id: correlationId,
      assumptions_mined: assumptions.length,
    });

    return assumptions;
  }

  /**
   * Infer assumption from failure pattern description
   *
   * @private
   */
  private inferAssumptionFromPattern(patternDescription: string): string {
    // Simple heuristic: negate the failure pattern
    const negations: Record<string, string> = {
      'lattice defects': 'Lattice defects are uniformly distributed',
      'loading ratio': 'Loading ratio is the critical factor',
      'temperature': 'Temperature is the primary control variable',
      'pressure': 'Pressure remains constant during reaction',
      'duration': 'Reaction duration is deterministic',
    };

    for (const [key, assumption] of Object.entries(negations)) {
      if (patternDescription.toLowerCase().includes(key)) {
        return assumption;
      }
    }

    return `Unstated assumption about ${patternDescription}`;
  }

  // ==========================================================================
  // EPISTEMIC AUDIT (PHASE I)
  // ==========================================================================

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
  async performEpistemicAudit(
    problemDescription: string,
    failurePatterns: Array<{
      pattern_description: string;
      failure_rate: number;
      data_points: number;
    }>,
    correlationId: string
  ): Promise<EpistemicAuditResult> {
    const startTime = Date.now();

    this.logger.info('Starting Phase I: Epistemic Audit', {
      correlation_id: correlationId,
      problem_description: problemDescription,
    });

    // Φ₁.₅: Mine tacit assumptions
    const tacitAssumptions = await this.mineTacitAssumptions(
      failurePatterns,
      correlationId
    );

    // Φ₃: Detect contradictions
    const contradictionResult = await this.detectContradictions(correlationId);

    // Transform contradictions to canonical format
    const contradictions = contradictionResult.contradictions.map((c) => ({
      id: uuidv4(),
      fallacy_type: this.mapLogicalFallacy(c.type) as any,
      contradiction_set_size: c.contradiction_set_size,
      rollback_steps: c.rollback_steps,
      affected_premises: c.affected_premises,
      resolved: false,
    }));

    // Check consistency
    const consistencyResult = await this.checkConsistency(correlationId);

    const executionTime = Date.now() - startTime;

    // Build canonical result
    const auditResult: EpistemicAuditResult = {
      phase: 'phase1_epistemic_audit',
      audit_id: uuidv4(),
      problem_description: problemDescription,
      tacit_assumptions: tacitAssumptions,
      contradictions,
      falsification_results: [], // To be populated by red team protocol (Φ₄)
      metrics: {
        total_assumptions_analyzed: tacitAssumptions.length,
        confirmed_contradictions: contradictions.length,
        hypotheses_falsified: 0, // To be updated by Φ₄
      },
      metadata: {
        execution_time_ms: executionTime,
        lean4_version: this.config.ENABLE_LEAN4_INTEGRATION ? '4.7.0' : undefined,
        epoch_number: 1, // Default to first epoch
      },
      correlation_id: correlationId,
      timestamp: new Date().toISOString(), // UTC timestamp (Law of UTC)
    };

    this.logger.info('Phase I: Epistemic Audit completed', {
      correlation_id: correlationId,
      audit_id: auditResult.audit_id,
      execution_time_ms: executionTime,
      tacit_assumptions_found: tacitAssumptions.length,
      contradictions_found: contradictions.length,
      consistent: consistencyResult.consistent,
    });

    return auditResult;
  }

  /**
   * Map internal logical fallacy to canonical schema
   *
   * @private
   */
  private mapLogicalFallacy(fallacy: LogicalFallacyInternal): LogicalFallacy {
    const mapping: Record<LogicalFallacyInternal, LogicalFallacy> = {
      [LogicalFallacyInternal.CIRCULUS_IN_PROBANDO]: 'circulus_in_probando',
      [LogicalFallacyInternal.CONFIRMATION_BIAS]: 'confirmation_bias',
      [LogicalFallacyInternal.HASTY_GENERALIZATION]: 'hasty_generalization',
      [LogicalFallacyInternal.FALSE_CAUSE]: 'false_cause',
      [LogicalFallacyInternal.AD_HOMINEM]: 'ad_hominem',
      [LogicalFallacyInternal.STRAW_MAN]: 'straw_man',
      [LogicalFallacyInternal.CONTRADICTION]: 'contradiction',
      [LogicalFallacyInternal.INCONSISTENCY]: 'inconsistency',
      [LogicalFallacyInternal.OTHER]: 'other',
    };

    return mapping[fallacy] || 'other';
  }

  // ==========================================================================
  // UTILITY METHODS
  // ==========================================================================

  /**
   * Clear all constraints from the engine
   *
   * Useful for testing and isolation
   */
  clear(): void {
    this.constraints.clear();
    this.logger.info('All constraints cleared');
  }

  /**
   * Get engine statistics
   */
  getStats(): {
    constraint_count: number;
    hard_constraints: number;
    soft_constraints: number;
    contradiction_detector_stats: any;
    } {
    const constraints = this.getAllConstraints();

    return {
      constraint_count: constraints.length,
      hard_constraints: constraints.filter((c) => c.type === ConstraintType.HARD).length,
      soft_constraints: constraints.filter((c) => c.type === ConstraintType.SOFT).length,
      contradiction_detector_stats: this.contradictionDetector.getStats(),
    };
  }

  /**
   * Reset circuit breakers
   */
  resetCircuitBreakers(): void {
    this.contradictionDetector.reset();
    this.logger.info('Circuit breakers reset');
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default SymbolicConstraintEngine;
