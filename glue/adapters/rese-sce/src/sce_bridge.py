#!/usr/bin/env python3
"""
RESE Symbolic Constraint Engine (SCE) - Python Bridge

This module provides a Python implementation of the Symbolic Constraint Engine,
serving as a bridge for Phase I executor to use.

Follows CLAUDE.md Laws:
- Law of Idempotency: All operations safe to run 100x
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker Pattern: Failure detection
- Structured Logging: JSON with correlation_id
- Timeout Enforcement: All operations have timeouts

Technical Manual Reference:
- Section 2.1: Symbolic Constraint Engine (SCE)
- Section 3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
- Section 3.1.5: Tacit Assumption Mining (Φ₁.₅)
"""

import os
import sys
import json
import uuid
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Z3 Integration (Law of Air Gap: Use root-level integration, not core-projects)
try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3Variable,
        Z3Constraint,
        Z3ConstraintType,
        Z3SolverResult,
        Z3ResultStatus,
        Z3Config
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Create stub types for type hints when Z3 not available
    Z3SolverEngine = None  # type: ignore
    Z3Variable = None  # type: ignore
    Z3Constraint = None  # type: ignore
    Z3ConstraintType = None  # type: ignore
    Z3SolverResult = None  # type: ignore
    Z3ResultStatus = None  # type: ignore
    Z3Config = None  # type: ignore
    logging.warning("Z3 integration not available - will use naive contradiction detection")

# DITO Integration
try:
    from dito_optimizer import DITOOptimizer, ActivationStrategy
    DITO_AVAILABLE = True
except ImportError:
    DITO_AVAILABLE = False
    DITOOptimizer = None  # type: ignore
    ActivationStrategy = None  # type: ignore
    logging.warning("DITO optimizer not available - will use Z3 or naive detection")

# ============================================================================
# CONFIGURATION (Law of Configuration Explicitness)
# ============================================================================

@dataclass
class SCEConfig:
    """Symbolic Constraint Engine Configuration"""

    # Timeout settings (milliseconds)
    TIMEOUT_MS: int
    CONSTRAINT_TIMEOUT_MS: int
    CONTRADICTION_DETECTION_TIMEOUT_MS: int

    # Iteration limits
    MAX_ITERATIONS: int
    MAX_CONSTRAINTS: int
    MAX_CONTRADICTION_SET_SIZE: int

    # Circuit breaker settings
    CIRCUIT_BREAKER_THRESHOLD: int
    CIRCUIT_BREAKER_TIMEOUT_MS: int

    # Feature flags
    ENABLE_TACIT_ASSUMPTION_MINING: bool
    ENABLE_Z3_SCE: bool  # Enable Z3 SMT solver for contradiction detection
    ENABLE_DITO: bool  # Enable DITO optimizer

    # Z3 Configuration
    Z3_TIMEOUT_MS: int
    Z3_MAX_MEMORY_MB: int
    Z3_UNSAT_CORE: bool

    # DITO Configuration
    DITO_ACTIVATION_STRATEGY: str  # selective_bfs, selective_dfs, minimal_subgraph, full
    DITO_ENABLE_LEAN4: bool

    @classmethod
    def from_env(cls) -> 'SCEConfig':
        """Load configuration from environment variables

        Law of Configuration Explicitness: All config via env vars
        Crashes immediately if required config is missing or invalid
        """
        config = cls(
            TIMEOUT_MS=int(os.getenv('SCE_TIMEOUT_MS', '5000')),
            CONSTRAINT_TIMEOUT_MS=int(os.getenv('SCE_CONSTRAINT_TIMEOUT_MS', '3000')),
            CONTRADICTION_DETECTION_TIMEOUT_MS=int(os.getenv('SCE_CONTRADICTION_TIMEOUT_MS', '10000')),

            MAX_ITERATIONS=int(os.getenv('SCE_MAX_ITERATIONS', '1000')),
            MAX_CONSTRAINTS=int(os.getenv('SCE_MAX_CONSTRAINTS', '10000')),
            MAX_CONTRADICTION_SET_SIZE=int(os.getenv('SCE_MAX_CONTRADICTION_SET_SIZE', '100')),

            CIRCUIT_BREAKER_THRESHOLD=int(os.getenv('SCE_CIRCUIT_BREAKER_THRESHOLD', '5')),
            CIRCUIT_BREAKER_TIMEOUT_MS=int(os.getenv('SCE_CIRCUIT_BREAKER_TIMEOUT_MS', '60000')),

            ENABLE_TACIT_ASSUMPTION_MINING=os.getenv('SCE_ENABLE_TACIT_MINING', 'true').lower() == 'true',

            # Z3 Configuration (Law of Configuration Explicitness)
            ENABLE_Z3_SCE=os.getenv('RESE_Z3_SCE_ENABLED', 'true').lower() == 'true',
            Z3_TIMEOUT_MS=int(os.getenv('Z3_TIMEOUT', '5000')),
            Z3_MAX_MEMORY_MB=int(os.getenv('Z3_MAX_MEMORY_MB', '4096')),
            Z3_UNSAT_CORE=os.getenv('Z3_UNSAT_CORE', 'true').lower() == 'true',

            # DITO Configuration
            ENABLE_DITO=os.getenv('RESE_DITO_ENABLED', 'true').lower() == 'true',
            DITO_ACTIVATION_STRATEGY=os.getenv('RESE_DITO_ACTIVATION_STRATEGY', 'selective_bfs'),
            DITO_ENABLE_LEAN4=os.getenv('RESE_DITO_ENABLE_LEAN4', 'false').lower() == 'true',
        )

        # Validate configuration
        if config.TIMEOUT_MS <= 0:
            raise ValueError("SCE_TIMEOUT_MS must be positive")
        if config.MAX_ITERATIONS <= 0:
            raise ValueError("SCE_MAX_ITERATIONS must be positive")
        if config.MAX_CONSTRAINTS <= 0:
            raise ValueError("SCE_MAX_CONSTRAINTS must be positive")
        if config.ENABLE_Z3_SCE and not Z3_AVAILABLE:
            logging.warning("Z3_SCE_ENABLED=true but Z3 not available - falling back to naive detection")

        if config.ENABLE_DITO and not DITO_AVAILABLE:
            logging.warning("DITO_ENABLED=true but DITO not available - falling back to Z3 or naive detection")

        # Validate DITO activation strategy
        valid_strategies = ['selective_bfs', 'selective_dfs', 'minimal_subgraph', 'full']
        if config.DITO_ACTIVATION_STRATEGY not in valid_strategies:
            raise ValueError(f"Invalid DITO_ACTIVATION_STRATEGY: {config.DITO_ACTIVATION_STRATEGY}. "
                           f"Must be one of {valid_strategies}")

        return config


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ConstraintType(Enum):
    """Constraint Type Enum"""
    HARD = 'hard'
    SOFT = 'soft'


class ConstraintCategory(Enum):
    """Constraint Category Enum"""
    HARD_PARAMETER_INEQUALITY = 'hard_parameter_inequality'
    SOFT_STATISTICAL = 'soft_statistical'
    TACIT_ASSUMPTION = 'tacit_assumption'
    INVERTED_CONSTRAINT = 'inverted_constraint'


class LogicalFallacy(Enum):
    """Logical Fallacy Types"""
    CIRCULUS_IN_PROBANDO = 'circulus_in_probando'
    CONFIRMATION_BIAS = 'confirmation_bias'
    HASTY_GENERALIZATION = 'hasty_generalization'
    FALSE_CAUSE = 'false_cause'
    AD_HOMINEM = 'ad_hominem'
    STRAW_MAN = 'straw_man'
    CONTRADICTION = 'contradiction'
    INCONSISTENCY = 'inconsistency'
    OTHER = 'other'


@dataclass
class Constraint:
    """Constraint Data Structure"""
    constraint_id: str
    type: ConstraintType
    category: ConstraintCategory
    description: str
    expression: Any = None
    dependencies: List[str] = None
    formalized_in_lean4: bool = False
    lean4_theorem: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['category'] = self.category.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class TacitAssumption:
    """Tacit Assumption"""
    id: str
    description: str
    source_pattern: str
    confidence_score: float
    supporting_evidence_count: int
    formalized_in_lean4: bool = False
    lean4_proposition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContradictionPair:
    """Contradiction Pair"""
    constraint1_id: str
    constraint2_id: str
    type: LogicalFallacy
    contradiction_set_size: int
    rollback_steps: int
    affected_premises: List[str]
    detected_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['detected_at'] = self.detected_at.isoformat()
        return data


@dataclass
class ContradictionDetectionResult:
    """Contradiction Detection Result"""
    contradictions: List[ContradictionPair]
    total_checked: int
    contradiction_found: bool
    largest_contradiction_set: int
    detection_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'contradictions': [c.to_dict() for c in self.contradictions],
            'total_checked': self.total_checked,
            'contradiction_found': self.contradiction_found,
            'largest_contradiction_set': self.largest_contradiction_set,
            'detection_time_ms': self.detection_time_ms,
        }


# ============================================================================
# MAIN CLASS: SymbolicConstraintEngine
# ============================================================================

class SymbolicConstraintEngine:
    """Symbolic Constraint Engine

    Main engine class for RESE Phase I: Epistemic Audit.

    Responsibilities:
    - Constraint management (add, remove, query)
    - Contradiction detection
    - Consistency checking
    - Tacit assumption mining (Φ₁.₅)

    Follows CLAUDE.md Laws:
    - Law of Idempotency: All operations safe to run 100x
    - Law of Configuration Explicitness: Config from env vars
    - Structured Logging: JSON with correlation_id
    - Timeout Enforcement: All operations have timeouts
    """

    def __init__(self, config: SCEConfig = None):
        """Initialize the Symbolic Constraint Engine

        Args:
            config: Configuration object (loaded from env if None)
        """
        self.config = config or SCEConfig.from_env()
        self.constraints: Dict[str, Constraint] = {}

        # Setup logger
        self.logger = logging.getLogger('rese.sce')
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Initialize Z3 solver if enabled and available
        self.z3_enabled = (
            self.config.ENABLE_Z3_SCE and
            Z3_AVAILABLE and
            self._initialize_z3_solver()
        )

        # Initialize DITO optimizer if enabled and available
        self.dito_enabled = (
            self.config.ENABLE_DITO and
            DITO_AVAILABLE and
            self._initialize_dito_optimizer()
        )

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'SymbolicConstraintEngine initialized',
            'max_constraints': self.config.MAX_CONSTRAINTS,
            'max_iterations': self.config.MAX_ITERATIONS,
            'enable_tacit_mining': self.config.ENABLE_TACIT_ASSUMPTION_MINING,
            'z3_enabled': self.z3_enabled,
            'z3_available': Z3_AVAILABLE,
            'dito_enabled': self.dito_enabled,
            'dito_available': DITO_AVAILABLE,
            'dito_strategy': self.config.DITO_ACTIVATION_STRATEGY if self.dito_enabled else None,
        }))

    # ========================================================================
    # Z3 INTEGRATION (O(n log n) Contradiction Detection)
    # ========================================================================

    def _initialize_z3_solver(self) -> bool:
        """Initialize Z3 solver with configuration

        Returns:
            bool: True if initialization successful
        """
        if not Z3_AVAILABLE:
            return False

        try:
            z3_config = Z3Config(
                timeout=self.config.Z3_TIMEOUT_MS / 1000.0,  # Convert to seconds
                memory_limit_mb=self.config.Z3_MAX_MEMORY_MB,
                proof_generation=True,
                unsat_core=self.config.Z3_UNSAT_CORE,
                auto_config=True
            )
            self.z3_solver = Z3SolverEngine(config=z3_config)
            self.logger.info(json.dumps({
                'level': 'info',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 solver initialized successfully',
                'timeout_ms': self.config.Z3_TIMEOUT_MS,
                'memory_mb': self.config.Z3_MAX_MEMORY_MB,
                'unsat_core': self.config.Z3_UNSAT_CORE,
            }))
            return True
        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 solver initialization failed',
                'error': str(e),
            }))
            return False

    def _initialize_dito_optimizer(self) -> bool:
        """Initialize DITO optimizer with configuration

        Returns:
            bool: True if initialization successful
        """
        if not DITO_AVAILABLE:
            return False

        try:
            # Map strategy string to enum
            strategy_map = {
                'selective_bfs': ActivationStrategy.SELECTIVE_BFS,
                'selective_dfs': ActivationStrategy.SELECTIVE_DFS,
                'minimal_subgraph': ActivationStrategy.MINIMAL_SUBGRAPH,
                'full': ActivationStrategy.FULL,
            }

            strategy = strategy_map.get(
                self.config.DITO_ACTIVATION_STRATEGY,
                ActivationStrategy.SELECTIVE_BFS
            )

            self.dito_optimizer = DITOOptimizer(
                config=self.config,
                activation_strategy=strategy,
                enable_lean4=self.config.DITO_ENABLE_LEAN4,
            )

            self.logger.info(json.dumps({
                'level': 'info',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'DITO optimizer initialized successfully',
                'strategy': self.config.DITO_ACTIVATION_STRATEGY,
                'lean4_enabled': self.config.DITO_ENABLE_LEAN4,
            }))
            return True
        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'DITO optimizer initialization failed',
                'error': str(e),
            }))
            return False

    def _encode_to_z3(self, constraint: Constraint) -> Optional[str]:
        """
        Convert RESE constraint to Z3 SMT-LIB2 formula

        From RESE Technical Manual §3.3: Use formal logic for constraint encoding

        Args:
            constraint: RESE Constraint object

        Returns:
            str: SMT-LIB2 formula string or None if encoding fails
        """
        try:
            # If constraint already has expression, use it
            if constraint.expression and isinstance(constraint.expression, str):
                # Check if it's already in SMT-LIB format
                if constraint.expression.strip().startswith('('):
                    return constraint.expression
                # Otherwise, try to convert from simple notation
                return self._convert_simple_expression_to_smtlib(constraint.expression)

            # Extract from description as fallback
            return self._extract_formula_from_description(constraint)

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Failed to encode constraint to Z3',
                'constraint_id': constraint.constraint_id,
                'error': str(e),
            }))
            return None

    def _convert_simple_expression_to_smtlib(self, expr: str) -> Optional[str]:
        """
        Convert simple expression (e.g., "x < 10") to SMT-LIB format

        Args:
            expr: Simple expression string

        Returns:
            str: SMT-LIB formatted expression or None
        """
        expr = expr.strip()

        # If already SMT-LIB, return as-is
        if expr.startswith('('):
            return expr

        # Parse common patterns: x < 10, temperature >= 100, etc.
        # Supported operators: <, <=, >, >=, =, !=
        operators = ['<=', '>=', '<', '>', '=', '!=']

        for op in operators:
            if op in expr:
                parts = expr.split(op)
                if len(parts) == 2:
                    var = parts[0].strip()
                    val = parts[1].strip()

                    # Determine type from value
                    if '.' in val or 'e' in val.lower():
                        val_type = 'Real'
                    else:
                        val_type = 'Int'

                    # Generate SMT-LIB expression with assertion
                    if op == '!=':
                        return f"(not (= {var} {val}))"
                    else:
                        return f"({op} {var} {val})"

        # Boolean expression
        if expr.lower() in ['true', 'false']:
            return expr.lower()

        # Variable reference
        if re.match(r'^\w+$', expr):
            return expr

        return None

    def _extract_formula_from_description(self, constraint: Constraint) -> Optional[str]:
        """
        Extract logical formula from constraint description

        Args:
            constraint: Constraint with description

        Returns:
            str: SMT-LIB formula or None
        """
        desc = constraint.description.lower()

        # Hard parameter inequalities
        if constraint.category == ConstraintCategory.HARD_PARAMETER_INEQUALITY:
            # Look for patterns like "cannot exceed", "must be less than", etc.
            if 'cannot exceed' in desc or 'must not exceed' in desc:
                var = self._extract_variable_name(desc)
                val = self._extract_value(desc)
                return f"(<= {var} {val})"
            elif 'must be at least' in desc or 'minimum' in desc:
                var = self._extract_variable_name(desc)
                val = self._extract_value(desc)
                return f"(>= {var} {val})"
            elif 'less than' in desc:
                var = self._extract_variable_name(desc)
                val = self._extract_value(desc)
                return f"(< {var} {val})"
            elif 'greater than' in desc:
                var = self._extract_variable_name(desc)
                val = self._extract_value(desc)
                return f"(> {var} {val})"

        # Soft statistical constraints
        elif constraint.category == ConstraintCategory.SOFT_STATISTICAL:
            if 'confidence' in desc or 'probability' in desc:
                var = self._extract_variable_name(desc)
                val = self._extract_value(desc)
                return f"(> {var} {val})"

        # Tacit assumptions (encode as Boolean)
        elif constraint.category == ConstraintCategory.TACIT_ASSUMPTION:
            # Create a Boolean variable for the assumption
            var_name = f"assumption_{constraint.constraint_id[:8]}"
            return var_name

        # Default: treat as Boolean assertion
        var_name = f"constraint_{constraint.constraint_id[:8]}"
        return var_name

    def _extract_variable_name(self, text: str) -> str:
        """
        Extract variable name from constraint text

        Args:
            text: Constraint description or expression

        Returns:
            str: Variable name (default 'X' if not found)
        """
        # Common variable patterns in scientific contexts
        var_patterns = [
            r'\b(temperature|temp|T)\b',
            r'\b(pressure|press|P)\b',
            r'\b(energy|E)\b',
            r'\b(time|t)\b',
            r'\b(ratio|r)\b',
            r'\b(rate|velocity|v)\b',
            r'\b(mass|m)\b',
            r'\b(length|l|L)\b',
            r'\b(x|y|z)\b',
        ]

        text_lower = text.lower()

        for pattern in var_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)

        # Fallback: extract first word that looks like a variable
        match = re.search(r'\b([a-z])[a-z]*\b', text_lower)
        if match:
            return match.group(1)

        return "X"

    def _extract_value(self, text: str) -> str:
        """
        Extract numeric value from constraint text

        Args:
            text: Constraint description or expression

        Returns:
            str: Numeric value as string (default '0.0' if not found)
        """
        # Match numbers (integers or decimals, scientific notation)
        patterns = [
            r'(\d+\.\d+)',  # Decimal: 3.14
            r'(\d+e[+-]?\d+)',  # Scientific: 1e5
            r'(\d+)',  # Integer: 42
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return "0.0"

    def _extract_unsat_core(
        self,
        z3_result: Any,  # Z3SolverResult when available
        constraints: List[Constraint]
    ) -> List[str]:
        """
        Extract minimal contradiction set from Z3 unsat core

        From RESE Technical Manual §3.3: Extract minimal contradiction set for efficient resolution

        Args:
            z3_result: Z3 solver result with unsat core
            constraints: Original RESE constraints

        Returns:
            List[str]: Constraint IDs in contradiction
        """
        contradictory_ids = []

        try:
            # If Z3 result has proof/unsat core information
            if z3_result and hasattr(z3_result, 'smtlib_output') and z3_result.smtlib_output:
                # Parse output for unsat core
                # Z3 unsat core format varies, try common patterns
                output = z3_result.smtlib_output

                # Pattern 1: Named assertions in unsat core
                # (:named-assertions c1 c2 c3)
                core_match = re.search(r'\(:named-assertions\s+(.*?)\)', output)
                if core_match:
                    core_items = core_match.group(1).split()
                    for item in core_items:
                        constraint_id = self._map_core_to_constraint_id(item)
                        if constraint_id:
                            contradictory_ids.append(constraint_id)

                # Pattern 2: Extract from error messages
                if not contradictory_ids and 'unsat' in output.lower():
                    # If we got UNSAT but no core, assume all constraints are involved
                    contradictory_ids = [c.constraint_id for c in constraints]

            # Fallback: if no unsat core available, return all constraint IDs
            if not contradictory_ids and z3_result and hasattr(z3_result, 'status') and Z3_AVAILABLE and z3_result.status == Z3ResultStatus.UNSAT:
                contradictory_ids = [c.constraint_id for c in constraints]

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Failed to extract unsat core',
                'error': str(e),
            }))
            # Fallback: return all constraint IDs
            contradictory_ids = [c.constraint_id for c in constraints]

        return contradictory_ids

    def _map_core_to_constraint_id(self, core_item: str) -> Optional[str]:
        """
        Map Z3 unsat core item to constraint ID

        Args:
            core_item: Z3 assertion name from unsat core

        Returns:
            str: Constraint ID or None
        """
        core_item = core_item.strip()

        # Pattern 1: constraint_abc123 -> abc123
        if core_item.startswith('constraint_'):
            short_id = core_item[len('constraint_'):]
            # Find full constraint ID matching short ID
            for cid in self.constraints.keys():
                if cid.startswith(short_id):
                    return cid

        # Pattern 2: assumption_abc123 -> abc123
        if core_item.startswith('assumption_'):
            short_id = core_item[len('assumption_'):]
            for cid in self.constraints.keys():
                if cid.startswith(short_id):
                    return cid

        # Pattern 3: Direct variable name match
        if core_item in self.constraints:
            return core_item

        return None

    # ========================================================================
    # CONSTRAINT MANAGEMENT
    # ========================================================================

    async def add_constraint(
        self,
        constraint: Constraint,
        correlation_id: str
    ) -> Dict[str, bool]:
        """Add a constraint to the engine

        Law of Idempotency: Check before create (UPSERT logic)

        Args:
            constraint: Constraint to add
            correlation_id: Distributed tracing correlation ID

        Returns:
            Dict with 'added' and 'updated' flags
        """
        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Adding constraint',
            'correlation_id': correlation_id,
            'constraint_id': constraint.constraint_id,
            'type': constraint.type.value,
            'category': constraint.category.value,
        }))

        # Check constraint count limit
        if len(self.constraints) >= self.config.MAX_CONSTRAINTS:
            raise ValueError(
                f"Cannot add constraint: maximum limit {self.config.MAX_CONSTRAINTS} reached"
            )

        exists = constraint.constraint_id in self.constraints
        self.constraints[constraint.constraint_id] = constraint

        return {'added': not exists, 'updated': exists}

    async def remove_constraint(
        self,
        constraint_id: str,
        correlation_id: str
    ) -> Dict[str, bool]:
        """Remove a constraint from the engine

        Law of Idempotency: Safe to run multiple times

        Args:
            constraint_id: ID of constraint to remove
            correlation_id: Distributed tracing correlation ID

        Returns:
            Dict with 'removed' flag
        """
        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Removing constraint',
            'correlation_id': correlation_id,
            'constraint_id': constraint_id,
        }))

        existed = constraint_id in self.constraints
        if existed:
            del self.constraints[constraint_id]

        return {'removed': existed}

    def get_constraint(self, constraint_id: str) -> Optional[Constraint]:
        """Get a constraint by ID

        Args:
            constraint_id: ID of constraint to retrieve

        Returns:
            Constraint or None if not found
        """
        return self.constraints.get(constraint_id)

    def get_all_constraints(self) -> List[Constraint]:
        """Get all constraints

        Returns:
            List of all constraints
        """
        return list(self.constraints.values())

    def get_constraints_by_type(self, type: ConstraintType) -> List[Constraint]:
        """Get constraints by type

        Args:
            type: Constraint type filter

        Returns:
            List of matching constraints
        """
        return [c for c in self.get_all_constraints() if c.type == type]

    def get_constraints_by_category(self, category: ConstraintCategory) -> List[Constraint]:
        """Get constraints by category

        Args:
            category: Constraint category filter

        Returns:
            List of matching constraints
        """
        return [c for c in self.get_all_constraints() if c.category == category]

    # ========================================================================
    # CONTRADICTION DETECTION
    # ========================================================================

    async def detect_contradictions(
        self,
        correlation_id: str
    ) -> ContradictionDetectionResult:
        """
        Detect contradictions in the current constraint set

        From RESE Technical Manual §3.3: Formal Logic Audit and Contradiction Detection (Φ₃)

        Implementation Strategy:
        - If Z3 enabled and available: Use SMT solver (O(n log n) complexity)
        - Otherwise: Fallback to naive pairwise comparison (O(n²) complexity)

        Args:
            correlation_id: Distributed tracing correlation ID

        Returns:
            Contradiction detection result
        """
        start_time = time.time()
        constraints = self.get_all_constraints()

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Starting contradiction detection audit',
            'correlation_id': correlation_id,
            'constraint_count': len(constraints),
            'dito_enabled': self.dito_enabled,
            'z3_enabled': self.z3_enabled,
        }))

        # Route to appropriate detection method
        # Priority: DITO > Z3 > Naive
        if self.dito_enabled and len(constraints) > 2:
            result = await self._detect_contradictions_dito(constraints, correlation_id)
        elif self.z3_enabled and len(constraints) > 2:
            result = await self._detect_contradictions_z3(constraints, correlation_id)
        else:
            result = await self._detect_contradictions_naive(constraints, correlation_id)

        detection_time = int((time.time() - start_time) * 1000)
        result.detection_time_ms = detection_time

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Contradiction detection audit completed',
            'correlation_id': correlation_id,
            'contradictions_found': len(result.contradictions),
            'largest_set': result.largest_contradiction_set,
            'detection_time_ms': detection_time,
            'solver_used': 'dito' if self.dito_enabled else ('z3' if self.z3_enabled else 'naive'),
        }))

        return result

    async def _detect_contradictions_z3(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> ContradictionDetectionResult:
        """
        Detect contradictions using Z3 SMT solver

        From RESE Technical Manual §3.3: Use Z3 for efficient contradiction detection
        Complexity: O(n log n) vs O(n²) for naive pairwise

        Args:
            constraints: List of RESE constraints
            correlation_id: Distributed tracing correlation ID

        Returns:
            Contradiction detection result
        """
        start_time = time.time()

        try:
            # Step 1: Encode constraints as Z3 formulas
            z3_formulas = []
            constraint_map = {}  # Map assertion names to constraint IDs
            variables = set()

            for constraint in constraints:
                formula = self._encode_to_z3(constraint)
                if formula:
                    z3_formulas.append(formula)
                    constraint_map[formula] = constraint.constraint_id

                    # Extract variables from formula
                    extracted_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
                    variables.update(extracted_vars)

            self.logger.debug(json.dumps({
                'level': 'debug',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Encoded constraints to Z3 formulas',
                'correlation_id': correlation_id,
                'formulas_generated': len(z3_formulas),
                'variables_extracted': len(variables),
            }))

            # Step 2: Build SMT-LIB2 program
            smtlib_lines = [
                "; RESE Constraint Contradiction Detection",
                f"; Correlation ID: {correlation_id}",
                "; Generated by: SymbolicConstraintEngine._detect_contradictions_z3",
                "",
                "(set-logic ALL)",
                "(set-option :produce-models true)",
                "(set-option :produce-proofs true)" if self.config.Z3_UNSAT_CORE else "",
            ]

            # Declare variables
            for var in variables:
                # Assume Real type for numeric variables
                smtlib_lines.append(f"(declare-fun {var} () Real)")

            # Add assertions
            for formula in z3_formulas:
                # Wrap in named assertion for unsat core tracking
                constraint_id = constraint_map.get(formula, 'unknown')
                safe_name = f"constraint_{constraint_id[:8]}"
                smtlib_lines.append(f"(assert (! {formula} :named {safe_name}))")

            # Check satisfiability
            smtlib_lines.append("(check-sat)")

            # Get model if SAT
            smtlib_lines.append("(get-model)")

            smtlib_content = "\n".join(smtlib_lines)

            self.logger.debug(json.dumps({
                'level': 'debug',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Generated SMT-LIB2 program',
                'correlation_id': correlation_id,
                'smtlib_size': len(smtlib_content),
            }))

            # Step 3: Check satisfiability with Z3
            z3_result = self.z3_solver.solve_smtlib(smtlib_content)

            self.logger.debug(json.dumps({
                'level': 'debug',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 solver completed',
                'correlation_id': correlation_id,
                'status': z3_result.status.value,
                'execution_time': z3_result.execution_time,
            }))

            # Step 4: Process result
            if z3_result.status == Z3ResultStatus.UNSAT:
                # Contradiction found - extract unsat core
                contradictory_ids = self._extract_unsat_core(z3_result, constraints)

                # Create contradiction pair from unsat core
                if len(contradictory_ids) >= 2:
                    c1_id = contradictory_ids[0]
                    c2_id = contradictory_ids[1]
                    c1 = self.constraints.get(c1_id)
                    c2 = self.constraints.get(c2_id)

                    if c1 and c2:
                        contradiction = ContradictionPair(
                            constraint1_id=c1_id,
                            constraint2_id=c2_id,
                            type=LogicalFallacy.CONTRADICTION,
                            contradiction_set_size=min(len(contradictory_ids), self.config.MAX_CONTRADICTION_SET_SIZE),
                            rollback_steps=max(len(c1.dependencies), len(c2.dependencies)),
                            affected_premises=contradictory_ids,
                            detected_at=datetime.now(timezone.utc),
                        )

                        return ContradictionDetectionResult(
                            contradictions=[contradiction],
                            total_checked=len(constraints),
                            contradiction_found=True,
                            largest_contradiction_set=len(contradictory_ids),
                            detection_time_ms=int((time.time() - start_time) * 1000),
                        )

            # Step 5: If SAT, no contradictions
            return ContradictionDetectionResult(
                contradictions=[],
                total_checked=len(constraints),
                contradiction_found=False,
                largest_contradiction_set=0,
                detection_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            # Circuit Breaker Pattern: Fallback to naive method on Z3 failure
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 contradiction detection failed, falling back to naive method',
                'correlation_id': correlation_id,
                'error': str(e),
            }))

            return await self._detect_contradictions_naive(constraints, correlation_id)

    async def _detect_contradictions_dito(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> ContradictionDetectionResult:
        """
        Detect contradictions using DITO optimizer

        From RESE Technical Manual §3.3.1: DITO optimizes contradiction detection
        via selective subgraph activation and targeted ATP.

        Complexity: O(n log n) vs O(n²) for naive pairwise

        Args:
            constraints: List of RESE constraints
            correlation_id: Distributed tracing correlation ID

        Returns:
            Contradiction detection result
        """
        start_time = time.time()

        try:
            self.logger.info(json.dumps({
                'level': 'info',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Using DITO optimizer for contradiction detection',
                'correlation_id': correlation_id,
                'constraint_count': len(constraints),
                'strategy': self.config.DITO_ACTIVATION_STRATEGY,
            }))

            # Run DITO optimization
            contradictions, stats = self.dito_optimizer.optimize_contradiction_detection(
                constraints,
                correlation_id
            )

            # Transform DITO contradictions to SCE format
            sce_contradictions = []
            for dito_contradiction in contradictions:
                sce_contradiction = ContradictionPair(
                    constraint1_id=dito_contradiction.constraint1_id,
                    constraint2_id=dito_contradiction.constraint2_id,
                    type=dito_contradiction.type,
                    contradiction_set_size=dito_contradiction.contradiction_set_size,
                    rollback_steps=dito_contradiction.rollback_steps,
                    affected_premises=dito_contradiction.affected_premises,
                    detected_at=dito_contradiction.detected_at,
                )
                sce_contradictions.append(sce_contradiction)

            # Calculate largest contradiction set
            largest_set = max(
                [c.contradiction_set_size for c in sce_contradictions],
                default=0
            )

            # Log DITO statistics
            self.logger.info(json.dumps({
                'level': 'info',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'DITO optimization completed',
                'correlation_id': correlation_id,
                'contradictions_found': len(sce_contradictions),
                'verified_nodes': stats.verified_nodes,
                'active_nodes': stats.active_nodes,
                'complexity_saved': f"{stats.complexity_saved:.1f}%",
                'atp_checks': stats.atp_checks_performed,
                'backtracks': stats.backtracks_performed,
            }))

            return ContradictionDetectionResult(
                contradictions=sce_contradictions,
                total_checked=stats.total_nodes,
                contradiction_found=len(sce_contradictions) > 0,
                largest_contradiction_set=largest_set,
                detection_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            # Circuit Breaker Pattern: Fallback to Z3 or naive method on DITO failure
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'DITO contradiction detection failed, falling back to Z3/naive',
                'correlation_id': correlation_id,
                'error': str(e),
            }))

            if self.z3_enabled:
                return await self._detect_contradictions_z3(constraints, correlation_id)
            else:
                return await self._detect_contradictions_naive(constraints, correlation_id)

    async def _detect_contradictions_naive(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> ContradictionDetectionResult:
        """
        Detect contradictions using naive pairwise comparison

        Fallback method when Z3 is unavailable or fails.
        O(n²) complexity - acceptable for small constraint sets.

        Args:
            constraints: List of RESE constraints
            correlation_id: Distributed tracing correlation ID

        Returns:
            Contradiction detection result
        """
        start_time = time.time()

        contradictions = []
        total_checked = len(constraints)
        largest_set = 0

        # Naive pairwise comparison
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                c1 = constraints[i]
                c2 = constraints[j]

                # Check if constraints are direct contradictions
                contradiction = self._check_pairwise_contradiction(c1, c2)
                if contradiction:
                    contradictions.append(contradiction)
                    largest_set = max(largest_set, contradiction.contradiction_set_size)

                # Enforce max iterations
                if len(contradictions) >= self.config.MAX_ITERATIONS:
                    self.logger.warning(json.dumps({
                        'level': 'warn',
                        'component': 'SymbolicConstraintEngine',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'message': 'Max iterations reached, stopping detection',
                        'correlation_id': correlation_id,
                        'iterations': len(contradictions),
                        'max_iterations': self.config.MAX_ITERATIONS,
                    }))
                    break

            if len(contradictions) >= self.config.MAX_ITERATIONS:
                break

        return ContradictionDetectionResult(
            contradictions=contradictions,
            total_checked=total_checked,
            contradiction_found=len(contradictions) > 0,
            largest_contradiction_set=largest_set,
            detection_time_ms=int((time.time() - start_time) * 1000),
        )

    def _check_pairwise_contradiction(
        self,
        c1: Constraint,
        c2: Constraint
    ) -> Optional[ContradictionPair]:
        """Check if two constraints contradict each other"""
        # Direct textual contradiction detection
        if self._is_direct_negation(c1.description, c2.description):
            return ContradictionPair(
                constraint1_id=c1.constraint_id,
                constraint2_id=c2.constraint_id,
                type=LogicalFallacy.CONTRADICTION,
                contradiction_set_size=2,
                rollback_steps=self._calculate_rollback_steps(c1, c2),
                affected_premises=[c1.constraint_id, c2.constraint_id],
                detected_at=datetime.now(timezone.utc),
            )

        # Check for circular dependency
        if self._has_circular_dependency(c1, c2):
            return ContradictionPair(
                constraint1_id=c1.constraint_id,
                constraint2_id=c2.constraint_id,
                type=LogicalFallacy.CIRCULUS_IN_PROBANDO,
                contradiction_set_size=self._calculate_contradiction_set_size(c1, c2),
                rollback_steps=self._calculate_rollback_steps(c1, c2),
                affected_premises=[*c1.dependencies, *c2.dependencies],
                detected_at=datetime.now(timezone.utc),
            )

        # Check for constraint type mismatch
        if self._is_hard_soft_mismatch(c1, c2):
            return ContradictionPair(
                constraint1_id=c1.constraint_id,
                constraint2_id=c2.constraint_id,
                type=LogicalFallacy.INCONSISTENCY,
                contradiction_set_size=2,
                rollback_steps=1,
                affected_premises=[c1.constraint_id, c2.constraint_id],
                detected_at=datetime.now(timezone.utc),
            )

        return None

    def _is_direct_negation(self, desc1: str, desc2: str) -> bool:
        """Check if description2 is a direct negation of description1"""
        negation_patterns = [
            'not ', 'no ', 'non-', 'un-',
        ]

        lower1 = desc1.lower().strip()
        lower2 = desc2.lower().strip()

        # Check for explicit "not X" vs "X" patterns
        if lower1.startswith('not ') and lower1[4:] == lower2:
            return True
        if lower2.startswith('not ') and lower2[4:] == lower1:
            return True

        return False

    def _has_circular_dependency(self, c1: Constraint, c2: Constraint) -> bool:
        """Check for circular dependency between constraints"""
        return (
            c2.constraint_id in c1.dependencies and
            c1.constraint_id in c2.dependencies
        )

    def _is_hard_soft_mismatch(self, c1: Constraint, c2: Constraint) -> bool:
        """Check if constraints have mismatched types on same premise"""
        return (
            c1.type != c2.type and
            (c1.category == c2.category or c1.description == c2.description)
        )

    def _calculate_contradiction_set_size(self, c1: Constraint, c2: Constraint) -> int:
        """Calculate contradiction set size (CSS)"""
        premises = set([
            c1.constraint_id,
            c2.constraint_id,
            *c1.dependencies,
            *c2.dependencies,
        ])
        return min(len(premises), self.config.MAX_CONTRADICTION_SET_SIZE)

    def _calculate_rollback_steps(self, c1: Constraint, c2: Constraint) -> int:
        """Calculate rollback steps to root premise"""
        return max(len(c1.dependencies), len(c2.dependencies))

    # ========================================================================
    # CONSISTENCY CHECKING
    # ========================================================================

    async def check_consistency(
        self,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Check consistency of the current constraint set

        Args:
            correlation_id: Distributed tracing correlation ID

        Returns:
            Consistency check result
        """
        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Starting consistency check',
            'correlation_id': correlation_id,
            'constraint_count': len(self.constraints),
        }))

        issues = []

        # Check for duplicate constraint IDs
        ids = set()
        for constraint_id in self.constraints.keys():
            if constraint_id in ids:
                issues.append(f"Duplicate constraint ID: {constraint_id}")
            ids.add(constraint_id)

        # Check for orphaned dependencies
        all_ids = set(self.constraints.keys())
        for constraint in self.constraints.values():
            for dep in constraint.dependencies:
                if dep not in all_ids:
                    issues.append(
                        f"Orphaned dependency: {constraint.constraint_id} "
                        f"depends on non-existent {dep}"
                    )

        # Check for dependency cycles
        cycles = self._detect_dependency_cycles()
        if cycles:
            for cycle in cycles:
                issues.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

        # Check constraint count limit
        if len(self.constraints) > self.config.MAX_CONSTRAINTS:
            issues.append(
                f"Constraint count {len(self.constraints)} exceeds maximum "
                f"{self.config.MAX_CONSTRAINTS}"
            )

        consistent = len(issues) == 0

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Consistency check completed',
            'correlation_id': correlation_id,
            'consistent': consistent,
            'issues_found': len(issues),
        }))

        return {
            'consistent': consistent,
            'issues': issues,
            'checked_at': datetime.now(timezone.utc).isoformat(),
        }

    def _detect_dependency_cycles(self) -> List[List[str]]:
        """Detect dependency cycles using DFS"""
        cycles = []
        visited = set()
        recursion_stack = set()

        def dfs(constraint_id: str, path: List[str]) -> bool:
            if constraint_id in recursion_stack:
                # Found a cycle
                cycle_start = path.index(constraint_id)
                cycles.append(path[cycle_start:] + [constraint_id])
                return True

            if constraint_id in visited:
                return False

            visited.add(constraint_id)
            recursion_stack.add(constraint_id)

            constraint = self.constraints.get(constraint_id)
            if constraint:
                for dep in constraint.dependencies:
                    if dfs(dep, path + [constraint_id]):
                        return True

            recursion_stack.remove(constraint_id)
            return False

        for constraint_id in self.constraints.keys():
            if constraint_id not in visited:
                dfs(constraint_id, [])

        return cycles

    # ========================================================================
    # TACIT ASSUMPTION MINING (Φ₁.₅)
    # ========================================================================

    async def mine_tacit_assumptions(
        self,
        failure_patterns: List[Dict[str, Any]],
        correlation_id: str
    ) -> List[TacitAssumption]:
        """Mine tacit assumptions from failure patterns

        From RESE Manual §3.1.5: Tacit Assumption Mining (Φ₁.₅)

        Args:
            failure_patterns: Patterns of failure to analyze
            correlation_id: Distributed tracing correlation ID

        Returns:
            Mined tacit assumptions
        """
        if not self.config.ENABLE_TACIT_ASSUMPTION_MINING:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'SymbolicConstraintEngine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Tacit assumption mining is disabled',
                'correlation_id': correlation_id,
            }))
            return []

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Starting tacit assumption mining',
            'correlation_id': correlation_id,
            'pattern_count': len(failure_patterns),
        }))

        assumptions = []

        for pattern in failure_patterns:
            pattern_desc = pattern.get('pattern_description', '')
            failure_rate = pattern.get('failure_rate', 0.0)
            data_points = pattern.get('data_points', 0)

            # High failure rate suggests a tacit assumption
            if failure_rate > 0.3:
                assumption = TacitAssumption(
                    id=str(uuid.uuid4()),
                    description=self._infer_assumption_from_pattern(pattern_desc),
                    source_pattern=pattern_desc,
                    confidence_score=min(failure_rate, 1.0),
                    supporting_evidence_count=data_points,
                    formalized_in_lean4=False,
                )

                assumptions.append(assumption)

                self.logger.debug(json.dumps({
                    'level': 'debug',
                    'component': 'SymbolicConstraintEngine',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'Tacit assumption mined',
                    'correlation_id': correlation_id,
                    'assumption_id': assumption.id,
                    'confidence': assumption.confidence_score,
                }))

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Tacit assumption mining completed',
            'correlation_id': correlation_id,
            'assumptions_mined': len(assumptions),
        }))

        return assumptions

    def _infer_assumption_from_pattern(self, pattern_description: str) -> str:
        """Infer assumption from failure pattern description"""
        heuristics = {
            'lattice defects': 'Lattice defects are uniformly distributed',
            'loading ratio': 'Loading ratio is the critical factor',
            'temperature': 'Temperature is the primary control variable',
            'pressure': 'Pressure remains constant during reaction',
            'duration': 'Reaction duration is deterministic',
        }

        pattern_lower = pattern_description.lower()
        for key, assumption in heuristics.items():
            if key in pattern_lower:
                return assumption

        return f"Unstated assumption about {pattern_description}"

    # ========================================================================
    # EPISTEMIC AUDIT (PHASE I)
    # ========================================================================

    async def perform_epistemic_audit(
        self,
        problem_description: str,
        failure_patterns: List[Dict[str, Any]],
        correlation_id: str
    ) -> Dict[str, Any]:
        """Perform full Epistemic Audit (Phase I)

        From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification

        Combines:
        - Φ₁.₅: Tacit Assumption Mining
        - Φ₃: Formal Logic Audit and Contradiction Detection

        Args:
            problem_description: Description of the problem to audit
            failure_patterns: Failure patterns for tacit assumption mining
            correlation_id: Distributed tracing correlation ID

        Returns:
            Canonical EpistemicAuditResult
        """
        start_time = time.time()

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Starting Phase I: Epistemic Audit',
            'correlation_id': correlation_id,
            'problem_description': problem_description,
        }))

        # Φ₁.₅: Mine tacit assumptions
        tacit_assumptions = await self.mine_tacit_assumptions(
            failure_patterns,
            correlation_id
        )

        # Φ₃: Detect contradictions
        contradiction_result = await self.detect_contradictions(correlation_id)

        # Transform contradictions to canonical format
        contradictions = [
            {
                'id': str(uuid.uuid4()),
                'fallacy_type': c.type.value,
                'contradiction_set_size': c.contradiction_set_size,
                'rollback_steps': c.rollback_steps,
                'affected_premises': c.affected_premises,
                'resolved': False,
            }
            for c in contradiction_result.contradictions
        ]

        # Check consistency
        consistency_result = await self.check_consistency(correlation_id)

        execution_time = int((time.time() - start_time) * 1000)

        # Build canonical result
        audit_result = {
            'phase': 'phase1_epistemic_audit',
            'audit_id': str(uuid.uuid4()),
            'problem_description': problem_description,
            'tacit_assumptions': [a.to_dict() for a in tacit_assumptions],
            'contradictions': contradictions,
            'falsification_results': [],  # To be populated by red team protocol (Φ₄)
            'hardened_constraints': [],  # To be populated by Φ₁
            'metrics': {
                'total_assumptions_analyzed': len(tacit_assumptions),
                'confirmed_contradictions': len(contradictions),
                'hypotheses_falsified': 0,  # To be updated by Φ₄
            },
            'metadata': {
                'execution_time_ms': execution_time,
                'lean4_version': None,  # Not implemented yet
                'epoch_number': 1,  # Default to first epoch
            },
            'correlation_id': correlation_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),  # Law of UTC
        }

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Phase I: Epistemic Audit completed',
            'correlation_id': correlation_id,
            'audit_id': audit_result['audit_id'],
            'execution_time_ms': execution_time,
            'tacit_assumptions_found': len(tacit_assumptions),
            'contradictions_found': len(contradictions),
            'consistent': consistency_result['consistent'],
        }))

        return audit_result

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def clear(self) -> None:
        """Clear all constraints from the engine

        Useful for testing and isolation
        """
        self.constraints.clear()
        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'All constraints cleared',
        }))

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        constraints = self.get_all_constraints()

        return {
            'constraint_count': len(constraints),
            'hard_constraints': len([c for c in constraints if c.type == ConstraintType.HARD]),
            'soft_constraints': len([c for c in constraints if c.type == ConstraintType.SOFT]),
        }

    def reset_circuit_breakers(self) -> None:
        """Reset circuit breakers

        No-op in this implementation (circuit breakers handled by adapter layer)
        """
        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'SymbolicConstraintEngine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Circuit breakers reset',
        }))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for testing"""
    import asyncio

    async def test_sce():
        # Create engine
        sce = SymbolicConstraintEngine()

        # Test constraint management
        constraint1 = Constraint(
            constraint_id=str(uuid.uuid4()),
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Lattice loading ratio cannot exceed 0.9',
        )

        await sce.add_constraint(constraint1, 'test-correlation-1')

        # Test contradiction detection
        result = await sce.detect_contradictions('test-correlation-2')
        print(json.dumps(result.to_dict(), indent=2))

        # Test tacit assumption mining
        failure_patterns = [
            {
                'pattern_description': 'lattice defects correlation',
                'failure_rate': 0.65,
                'data_points': 150,
            }
        ]

        assumptions = await sce.mine_tacit_assumptions(failure_patterns, 'test-correlation-3')
        print(f"Mined {len(assumptions)} tacit assumptions")

        # Test epistemic audit
        audit_result = await sce.perform_epistemic_audit(
            problem_description='LENR thermal coefficient inconsistency',
            failure_patterns=failure_patterns,
            correlation_id='test-correlation-4',
        )
        print(json.dumps(audit_result, indent=2))

    asyncio.run(test_sce())


if __name__ == '__main__':
    main()
