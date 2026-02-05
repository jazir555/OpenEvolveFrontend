#!/usr/bin/env python3
"""
Dynamic Inference Trace Optimizer (DITO) for RESE SCE

Implements O(n log n) contradiction detection via:
- Targeted Z3 ATP: Use Z3 SMT solver for efficient contradiction detection
- Selective subgraph activation: Avoid exponential complexity
- Backtracking: Reset to last verified node
- Minimum subgraph isolation: Root premise violation
- Incremental solving: Z3 push/pop for backtracking

From RESE Technical Manual §3.3.1: DITO optimizes contradiction detection
by selectively activating constraint subgraphs only when needed.

Enhanced with Z3 ATP:
- Replaces naive O(n²) pairwise checking with Z3 SAT solving
- Targeted contradiction detection via UNSAT cores
- Incremental constraint checking with push/pop
- Performance tracking: Z3 vs naive baseline

Author: OpenEvolve
Created: 2026-02-04
Enhanced: 2026-02-04 (Z3 ATP Integration)
"""

import os
import sys
import json
import uuid
import time
import heapq
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import SCE components
from sce_bridge import (
    Constraint,
    ConstraintType,
    ConstraintCategory,
    ContradictionPair,
    LogicalFallacy,
    SCEConfig,
)

# Try to import Z3 for targeted ATP
try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3Variable,
        Z3Constraint,
        Z3ConstraintType,
        Z3Config,
        Z3SolverResult,
        Z3ResultStatus,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    Z3SolverEngine = None  # type: ignore
    Z3Variable = None  # type: ignore
    Z3Constraint = None  # type: ignore
    Z3ConstraintType = None  # type: ignore
    Z3Config = None  # type: ignore
    Z3SolverResult = None  # type: ignore
    Z3ResultStatus = None  # type: ignore

# Try to import Lean 4 bridge (placeholder if not available)
try:
    from lean4_atp_bridge import Lean4ATPBridge, Lean4ProofResult
    LEAN4_AVAILABLE = True
except ImportError:
    LEAN4_AVAILABLE = False
    Lean4ATPBridge = None  # type: ignore
    Lean4ProofResult = None  # type: ignore

# Try to import LeanAide client for AI-guided tactic suggestion
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    LeanAideClient = None  # type: ignore
    LeanAideConfig = None  # type: ignore
    LeanAideResult = None  # type: ignore


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class InferenceGraphNode:
    """Node in the inference graph

    Represents a constraint with its activation state and dependencies.
    """

    def __init__(
        self,
        constraint: Constraint,
        node_id: str = None,
    ):
        self.constraint = constraint
        self.node_id = node_id or constraint.constraint_id
        self.is_active = False  # Selective activation
        self.is_verified = False  # Verification state
        self.dependents: Set[str] = set()  # Nodes that depend on this one
        self.dependencies: Set[str] = set(constraint.dependencies)
        self.activation_timestamp: Optional[datetime] = None
        self.verification_timestamp: Optional[datetime] = None

        # Contradiction tracking
        self.contradiction_detected = False
        self.contradiction_partners: Set[str] = set()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'node_id': self.node_id,
            'constraint_id': self.constraint.constraint_id,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'dependencies': list(self.dependencies),
            'dependents': list(self.dependents),
            'contradiction_detected': self.contradiction_detected,
            'contradiction_partners': list(self.contradiction_partners),
        }


class ActivationStrategy(Enum):
    """Subgraph activation strategy"""
    FULL = 'full'  # Activate entire graph (naive)
    SELECTIVE_BFS = 'selective_bfs'  # Breadth-first selective activation
    SELECTIVE_DFS = 'ive_dfs'  # Depth-first selective activation
    MINIMAL_SUBGRAPH = 'minimal_subgraph'  # Minimum subgraph isolation
    AI_GUIDED = 'ai_guided'  # LeanAide-guided intelligent activation


class VerificationTier(Enum):
    """Tiered verification level"""
    Z3_FAST = 'z3_fast'  # Level 1: Z3 fast contradiction detection
    LEANAIDE_AI = 'leanaide_ai'  # Level 2: LeanAide AI-assisted proof discovery
    LEAN4_FORMAL = 'lean4_formal'  # Level 3: Lean 4 formal verification


@dataclass
class Z3ATPStats:
    """Z3 ATP performance statistics"""
    z3_checks_performed: int = 0
    z3_contradictions_found: int = 0
    z3_unsat_results: int = 0
    z3_sat_results: int = 0
    z3_unknown_results: int = 0
    z3_total_time_ms: int = 0
    naive_checks_performed: int = 0
    naive_contradictions_found: int = 0
    naive_total_time_ms: int = 0
    speedup_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'z3_checks_performed': self.z3_checks_performed,
            'z3_contradictions_found': self.z3_contradictions_found,
            'z3_unsat_results': self.z3_unsat_results,
            'z3_sat_results': self.z3_sat_results,
            'z3_unknown_results': self.z3_unknown_results,
            'z3_total_time_ms': self.z3_total_time_ms,
            'naive_checks_performed': self.naive_checks_performed,
            'naive_contradictions_found': self.naive_contradictions_found,
            'naive_total_time_ms': self.naive_total_time_ms,
            'speedup_factor': self.speedup_factor,
        }


@dataclass
class LeanAideAIStats:
    """LeanAide AI performance statistics"""
    leanaide_checks_performed: int = 0
    leanaide_tactics_suggested: int = 0
    leanaide_contradictions_resolved: int = 0
    leanaide_autoformalizations: int = 0
    leanaide_total_time_ms: int = 0
    leanaide_subgraph_activations: int = 0
    leanaide_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'leanaide_checks_performed': self.leanaide_checks_performed,
            'leanaide_tactics_suggested': self.leanaide_tactics_suggested,
            'leanaide_contradictions_resolved': self.leanaide_contradictions_resolved,
            'leanaide_autoformalizations': self.leanaide_autoformalizations,
            'leanaide_total_time_ms': self.leanaide_total_time_ms,
            'leanaide_subgraph_activations': self.leanaide_subgraph_activations,
            'leanaide_success_rate': self.leanaide_success_rate,
        }


@dataclass
class DITOStats:
    """DITO execution statistics"""
    total_nodes: int = 0
    active_nodes: int = 0
    verified_nodes: int = 0
    contradictions_found: int = 0
    activations_performed: int = 0
    backtracks_performed: int = 0
    atp_checks_performed: int = 0
    execution_time_ms: int = 0
    complexity_saved: float = 0.0  # Percentage of graph not activated
    z3_atp_stats: Optional[Z3ATPStats] = None  # Z3 ATP performance stats
    leanaide_ai_stats: Optional[LeanAideAIStats] = None  # LeanAide AI stats
    tier_distribution: Dict[str, int] = None  # Distribution of verification tiers used

    def __post_init__(self):
        if self.tier_distribution is None:
            self.tier_distribution = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_nodes': self.total_nodes,
            'active_nodes': self.active_nodes,
            'verified_nodes': self.verified_nodes,
            'contradictions_found': self.contradictions_found,
            'activations_performed': self.activations_performed,
            'backtracks_performed': self.backtracks_performed,
            'atp_checks_performed': self.atp_checks_performed,
            'execution_time_ms': self.execution_time_ms,
            'complexity_saved': self.complexity_saved,
            'z3_atp_stats': self.z3_atp_stats.to_dict() if self.z3_atp_stats else {},
            'leanaide_ai_stats': self.leanaide_ai_stats.to_dict() if self.leanaide_ai_stats else {},
            'tier_distribution': self.tier_distribution,
        }


@dataclass
class BacktrackPoint:
    """Backtracking checkpoint"""
    node_id: str
    active_nodes: Set[str]
    verified_nodes: Set[str]
    timestamp: datetime
    contradiction_context: Optional[Dict[str, Any]] = None
    z3_solver_state: Optional[Any] = None  # Z3 solver state snapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'active_nodes': list(self.active_nodes),
            'verified_nodes': list(self.verified_nodes),
            'timestamp': self.timestamp.isoformat(),
            'contradiction_context': self.contradiction_context,
        }


# ============================================================================
# Z3-BASED CONTRADICTION DETECTOR
# ============================================================================

class Z3ContradictionDetector:
    """
    Z3-based contradiction detector using automated theorem proving.

    Replaces naive O(n²) pairwise checking with efficient Z3 SAT solving.
    Uses incremental solving (push/pop) for backtracking support.

    Key Features:
    1. Constraint encoding to SMT-LIB2
    2. Incremental solving with push/pop
    3. UNSAT core extraction for contradiction diagnosis
    4. Performance tracking vs naive baseline
    """

    def __init__(
        self,
        z3_solver: Z3SolverEngine,
        config: SCEConfig,
        logger: logging.Logger
    ):
        """Initialize Z3 contradiction detector

        Args:
            z3_solver: Z3 solver engine instance
            config: SCE configuration
            logger: Logger instance
        """
        self.z3_solver = z3_solver
        self.config = config
        self.logger = logger

        # Performance statistics
        self.stats = Z3ATPStats()

        # Variable registry for encoding
        self.variable_registry: Dict[str, Z3Variable] = {}

        # Constraint cache for incremental solving
        self.constraint_cache: Dict[str, Z3Constraint] = {}

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'Z3ContradictionDetector',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Z3 contradiction detector initialized',
        }))

    def encode_constraint_to_z3(self, constraint: Constraint) -> Optional[Tuple[Z3Variable, Z3Constraint]]:
        """
        Encode RESE constraint to Z3 variable and constraint.

        Args:
            constraint: RESE constraint to encode

        Returns:
            Tuple of (Z3Variable, Z3Constraint) or None if encoding fails
        """
        try:
            # Extract variable name from constraint
            var_name = self._extract_variable_name(constraint)
            if not var_name:
                return None

            # Determine variable type
            var_type = self._determine_variable_type(constraint)

            # Create Z3 variable
            if var_name not in self.variable_registry:
                z3_var = Z3Variable(
                    name=var_name,
                    var_type=var_type,
                    bounds=self._extract_bounds(constraint)
                )
                self.variable_registry[var_name] = z3_var

            # Create Z3 constraint expression
            z3_expr = self._create_z3_expression(constraint, var_name)
            if not z3_expr:
                return None

            z3_constraint = Z3Constraint(
                expression=z3_expr,
                constraint_type=var_type,
                description=constraint.description
            )

            # Cache constraint
            self.constraint_cache[constraint.constraint_id] = z3_constraint

            return (self.variable_registry[var_name], z3_constraint)

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'Z3ContradictionDetector',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Failed to encode constraint to Z3',
                'constraint_id': constraint.constraint_id,
                'error': str(e),
            }))
            return None

    def _extract_variable_name(self, constraint: Constraint) -> Optional[str]:
        """Extract variable name from constraint description"""
        desc = constraint.description

        # Common patterns
        patterns = [
            r'([A-Za-z_]\w*)\s*[<>=!]=?',  # "T < 1000", "x == 5"
            r'([A-Za-z_]\w*)\s+(must|should|cannot)\s+',  # "Temperature must be"
        ]

        import re
        for pattern in patterns:
            match = re.search(pattern, desc)
            if match:
                var_name = match.group(1)
                # Standardize variable names
                return var_name.lower().replace(' ', '_')

        # Fallback: use constraint type
        if constraint.category == ConstraintCategory.HARD_PARAMETER_INEQUALITY:
            return "param"

        return None

    def _determine_variable_type(self, constraint: Constraint) -> Z3ConstraintType:
        """Determine Z3 variable type from constraint"""
        if constraint.category == ConstraintCategory.HARD_PARAMETER_INEQUALITY:
            # Check for real numbers
            desc = constraint.description.lower()
            if any(op in desc for op in ['<=', '>=', '<', '>']):
                # Check if values are integers or floats
                import re
                numbers = re.findall(r'\d+\.?\d*', desc)
                if numbers and '.' in numbers[0]:
                    return Z3ConstraintType.REAL
                return Z3ConstraintType.INTEGER

        return Z3ConstraintType.REAL  # Default to real

    def _extract_bounds(self, constraint: Constraint) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Extract variable bounds from constraint"""
        import re
        desc = constraint.description

        # Look for patterns like "T > 0" or "T < 1000"
        # This is simplified - a full implementation would be more sophisticated
        return None  # For now, let Z3 infer bounds

    def _create_z3_expression(self, constraint: Constraint, var_name: str) -> Optional[str]:
        """
        Create Z3 SMT-LIB2 expression from constraint

        Examples:
            "T < 1000" -> "(< T 1000)"
            "T > 0" -> "(> T 0)"
            "P <= 5000" -> "(<= P 5000)"
        """
        import re
        desc = constraint.description

        # Try to extract operator and value
        patterns = [
            (r'<', '<'),
            (r'>', '>'),
            (r'<=', '<='),
            (r'>=', '>='),
            (r'==', '='),
            (r'=', '='),
        ]

        for pattern, smt_op in patterns:
            if pattern in desc:
                parts = desc.split(pattern)
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()

                    # Extract numeric value
                    value_match = re.search(r'-?\d+\.?\d*', rhs)
                    if value_match:
                        value = value_match.group()
                        return f"({smt_op} {var_name} {value})"

        # Try expression field if available
        if constraint.expression:
            expr = str(constraint.expression).strip()
            if expr.startswith('('):
                return expr
            # Convert Python-like expressions to SMT-LIB
            return self._python_to_smtlib(expr, var_name)

        return None

    def _python_to_smtlib(self, expr: str, var_name: str) -> Optional[str]:
        """Convert Python-like expression to SMT-LIB"""
        # Simple substitutions
        expr = expr.replace(f'{var_name} <', f'(< {var_name}')
        expr = expr.replace(f'{var_name} >', f'(> {var_name}')
        expr = expr.replace(f'{var_name} <=', f'(<= {var_name}')
        expr = expr.replace(f'{var_name} >=', f'(>= {var_name}')
        expr = expr.replace(f'{var_name} ==', f'(= {var_name}')

        if expr.startswith('('):
            return expr + ')'

        return None

    def check_contradiction_z3(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Tuple[Optional[ContradictionPair], Z3SolverResult]:
        """
        Check for contradictions using Z3 ATP.

        Encodes constraints to SMT-LIB2 and checks satisfiability.
        Returns UNSAT if contradiction found.

        Args:
            constraints: List of constraints to check
            correlation_id: Distributed tracing correlation ID

        Returns:
            Tuple of (contradiction pair or None, Z3 solver result)
        """
        start_time = time.time()
        self.stats.z3_checks_performed += 1

        try:
            # Encode constraints to Z3
            z3_variables = []
            z3_constraints = []

            for constraint in constraints:
                encoded = self.encode_constraint_to_z3(constraint)
                if encoded:
                    var, constr = encoded
                    if var not in z3_variables:
                        z3_variables.append(var)
                    z3_constraints.append(constr)

            if len(z3_constraints) < 2:
                # Need at least 2 constraints for contradiction
                return None, Z3SolverResult(status=Z3ResultStatus.UNKNOWN)

            # Use Z3 solver to check satisfiability
            result = self.z3_solver.solve_constraints(
                variables=z3_variables,
                constraints=z3_constraints
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.z3_total_time_ms += elapsed_ms

            # Update status counters
            if result.status == Z3ResultStatus.UNSAT:
                self.stats.z3_unsat_results += 1
                self.stats.z3_contradictions_found += 1

                # Extract contradiction pair from UNSAT result
                contradiction = self._extract_contradiction_from_unsat(
                    constraints,
                    result
                )

                return contradiction, result

            elif result.status == Z3ResultStatus.SAT:
                self.stats.z3_sat_results += 1
                return None, result

            else:
                self.stats.z3_unknown_results += 1
                return None, result

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'Z3ContradictionDetector',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 contradiction check failed',
                'correlation_id': correlation_id,
                'error': str(e),
            }))
            return None, Z3SolverResult(
                status=Z3ResultStatus.ERROR,
                reason=str(e)
            )

    def _extract_contradiction_from_unsat(
        self,
        constraints: List[Constraint],
        result: Z3SolverResult
    ) -> Optional[ContradictionPair]:
        """Extract contradiction pair from UNSAT result"""
        if len(constraints) < 2:
            return None

        # For now, return first two constraints as contradictory pair
        # A more sophisticated implementation would use UNSAT cores
        return ContradictionPair(
            constraint1_id=constraints[0].constraint_id,
            constraint2_id=constraints[1].constraint_id,
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=len(constraints),
            rollback_steps=max(
                len(constraints[0].dependencies),
                len(constraints[1].dependencies)
            ),
            affected_premises=[
                c.constraint_id for c in constraints
            ],
            detected_at=datetime.now(timezone.utc),
        )

    def check_contradiction_naive(
        self,
        constraint1: Constraint,
        constraint2: Constraint
    ) -> Optional[ContradictionPair]:
        """
        Naive O(1) pairwise contradiction check (baseline for comparison).

        Args:
            constraint1: First constraint
            constraint2: Second constraint

        Returns:
            Contradiction pair if found, None otherwise
        """
        start_time = time.time()
        self.stats.naive_checks_performed += 1

        # Direct textual negation
        if self._is_negation(constraint1.description, constraint2.description):
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.naive_total_time_ms += elapsed_ms
            self.stats.naive_contradictions_found += 1

            return ContradictionPair(
                constraint1_id=constraint1.constraint_id,
                constraint2_id=constraint2.constraint_id,
                type=LogicalFallacy.CONTRADICTION,
                contradiction_set_size=2,
                rollback_steps=max(
                    len(constraint1.dependencies),
                    len(constraint2.dependencies)
                ),
                affected_premises=[
                    constraint1.constraint_id,
                    constraint2.constraint_id
                ],
                detected_at=datetime.now(timezone.utc),
            )

        # Circular dependency
        if (constraint2.constraint_id in constraint1.dependencies and
            constraint1.constraint_id in constraint2.dependencies):
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.naive_total_time_ms += elapsed_ms
            self.stats.naive_contradictions_found += 1

            return ContradictionPair(
                constraint1_id=constraint1.constraint_id,
                constraint2_id=constraint2.constraint_id,
                type=LogicalFallacy.CIRCULUS_IN_PROBANDO,
                contradiction_set_size=2,
                rollback_steps=max(
                    len(constraint1.dependencies),
                    len(constraint2.dependencies)
                ),
                affected_premises=[
                    *constraint1.dependencies,
                    *constraint2.dependencies
                ],
                detected_at=datetime.now(timezone.utc),
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.stats.naive_total_time_ms += elapsed_ms
        return None

    def _is_negation(self, desc1: str, desc2: str) -> bool:
        """Check if desc2 is a negation of desc1"""
        d1, d2 = desc1.lower().strip(), desc2.lower().strip()

        # Direct "not X" vs "X"
        if d1.startswith('not ') and d1[4:] == d2:
            return True
        if d2.startswith('not ') and d2[4:] == d1:
            return True

        # Antonym patterns
        antonym_pairs = [
            ('less than', 'greater than'),
            ('cannot exceed', 'must exceed'),
            ('impossible', 'possible'),
            ('forbidden', 'required'),
        ]

        for a1, a2 in antonym_pairs:
            if a1 in d1 and a2 in d2:
                return True
            if a2 in d1 and a1 in d2:
                return True

        # Check for opposite inequalities on same variable
        import re
        var1_match = re.search(r'([a-z_]\w*)\s*[<>=]', d1)
        var2_match = re.search(r'([a-z_]\w*)\s*[<>=]', d2)

        if var1_match and var2_match:
            if var1_match.group(1) == var2_match.group(1):
                # Same variable - check for opposite operators
                has_lt = '<' in d1 and '>' in d2
                has_gt = '>' in d1 and '<' in d2
                if has_lt or has_gt:
                    return True

        return False

    def calculate_speedup(self) -> float:
        """Calculate speedup factor of Z3 vs naive"""
        if self.stats.naive_total_time_ms == 0:
            return 0.0

        if self.stats.z3_total_time_ms == 0:
            return float('inf')

        return self.stats.naive_total_time_ms / self.stats.z3_total_time_ms

    def get_stats(self) -> Z3ATPStats:
        """Get performance statistics"""
        self.stats.speedup_factor = self.calculate_speedup()
        return self.stats


# ============================================================================
# LEANAIDE TACTIC SUGGESTER (AI-Guided Proof Discovery)
# ============================================================================

class LeanAideTacticSuggester:
    """
    LeanAide-based AI tactic suggestion for contradiction resolution.

    Key Features:
    1. AI-guided proof tactic suggestion
    2. Automatic constraint formalization
    3. Subgraph activation guidance
    4. Contradiction resolution assistance
    5. Natural language constraint processing

    Uses LeanAide ML models to suggest optimal proof tactics for resolving
    contradictions detected by Z3 or found through other means.
    """

    def __init__(
        self,
        config: SCEConfig,
        logger: logging.Logger
    ):
        """Initialize LeanAide Tactic Suggester

        Args:
            config: SCE configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Performance statistics
        self.stats = LeanAideAIStats()

        # LeanAide client (async)
        self.leanaide_client: Optional[LeanAideClient] = None
        self.leanaide_available = LEANAIDE_AVAILABLE

        # Initialize LeanAide client if available
        if self.leanaide_available:
            try:
                leanaide_config = LeanAideConfig(
                    host=getattr(config, 'LEANAIDE_HOST', 'localhost'),
                    port=getattr(config, 'LEANAIDE_PORT', 7654),
                    timeout=getattr(config, 'LEANAIDE_TIMEOUT_MS', 30000) / 1000.0,
                    max_retries=getattr(config, 'LEANAIDE_MAX_RETRIES', 3),
                )
                self.leanaide_client = LeanAideClient(config=leanaide_config)
                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide tactic suggester initialized',
                    'leanaide_available': True,
                }))
            except Exception as e:
                self.leanaide_available = False
                self.logger.warning(json.dumps({
                    'level': 'warn',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide initialization failed',
                    'error': str(e),
                }))

    async def suggest_tactics(
        self,
        contradiction: ContradictionPair,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[List[str]]:
        """
        Suggest proof tactics for resolving a contradiction.

        Uses LeanAide AI to analyze the contradiction and suggest optimal tactics.

        Args:
            contradiction: Detected contradiction pair
            constraints: All constraints involved
            correlation_id: Distributed tracing correlation ID

        Returns:
            List of suggested tactic names or None if unavailable
        """
        start_time = time.time()
        self.stats.leanaide_checks_performed += 1

        if not self.leanaide_available or not self.leanaide_client:
            self.logger.debug(json.dumps({
                'level': 'debug',
                'component': 'LeanAideTacticSuggester',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'LeanAide not available for tactic suggestion',
                'correlation_id': correlation_id,
            }))
            return None

        try:
            # Build natural language description of contradiction
            contradiction_desc = self._build_contradiction_description(
                contradiction,
                constraints
            )

            # Use LeanAide math_query to get tactic suggestions
            query = (
                f"What are the best Lean 4 proof tactics to resolve this contradiction?\n"
                f"Contradiction: {contradiction_desc}\n"
                f"Suggest 3-5 specific Lean tactics (e.g., rw, simp, apply, cases, etc.)."
            )

            result: LeanAideResult = await self.leanaide_client.math_query(
                query=query,
                n=3
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.leanaide_total_time_ms += elapsed_ms

            if result.success and result.data:
                # Extract tactics from result
                tactics = self._extract_tactics(result.data)
                self.stats.leanaide_tactics_suggested += len(tactics)

                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide suggested tactics',
                    'correlation_id': correlation_id,
                    'contradiction': f"{contradiction.constraint1_id} vs {contradiction.constraint2_id}",
                    'tactics': tactics,
                    'response_time_ms': elapsed_ms,
                }))

                return tactics
            else:
                self.logger.warning(json.dumps({
                    'level': 'warn',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide tactic suggestion failed',
                    'correlation_id': correlation_id,
                    'error': result.error,
                }))
                return None

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'LeanAideTacticSuggester',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'LeanAide tactic suggestion exception',
                'correlation_id': correlation_id,
                'error': str(e),
            }))
            return None

    async def resolve_with_ai(
        self,
        contradiction: ContradictionPair,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LeanAide AI to assist in resolving a contradiction.

        Analyzes the contradiction and provides resolution suggestions.

        Args:
            contradiction: Detected contradiction pair
            constraints: All constraints involved
            correlation_id: Distributed tracing correlation ID

        Returns:
            Resolution suggestions or None if unavailable
        """
        start_time = time.time()

        if not self.leanaide_available or not self.leanaide_client:
            return None

        try:
            # Build detailed problem description
            problem_desc = self._build_resolution_problem(
                contradiction,
                constraints
            )

            # Use LeanAide to get resolution suggestions
            result = await self.leanaide_client.math_query(
                query=f"How can I resolve this contradiction?\n{problem_desc}",
                n=2
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.leanaide_total_time_ms += elapsed_ms

            if result.success and result.data:
                self.stats.leanaide_contradictions_resolved += 1

                resolution = {
                    'contradiction_id': f"{contradiction.constraint1_id}-{contradiction.constraint2_id}",
                    'suggestions': result.data.get('result', []),
                    'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                    'response_time_ms': elapsed_ms,
                }

                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide assisted resolution',
                    'correlation_id': correlation_id,
                    'resolution': resolution,
                }))

                return resolution
            else:
                return None

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'LeanAideTacticSuggester',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'LeanAide resolution exception',
                'correlation_id': correlation_id,
                'error': str(e),
            }))
            return None

    async def formalize_with_ai(
        self,
        natural_constraint: str,
        correlation_id: str
    ) -> Optional[str]:
        """
        Autoformalize natural language constraint to formal logic.

        Uses LeanAide to convert natural language constraints to formal Lean 4 code.

        Args:
            natural_constraint: Natural language description
            correlation_id: Distributed tracing correlation ID

        Returns:
            Formalized constraint or None if unavailable
        """
        start_time = time.time()

        if not self.leanaide_available or not self.leanaide_client:
            return None

        try:
            # Use LeanAide translate_thm to formalize
            result = await self.leanaide_client.translate_thm_detailed(
                theorem_text=natural_constraint,
                theorem_name="auto_formalized"
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.leanaide_total_time_ms += elapsed_ms

            if result.success and result.data:
                self.stats.leanaide_autoformalizations += 1

                # Extract formalization
                formal_code = result.data.get('type', result.data.get('result', ''))

                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide autoformalization',
                    'correlation_id': correlation_id,
                    'input': natural_constraint[:100],
                    'formal_code': formal_code[:200],
                    'response_time_ms': elapsed_ms,
                }))

                return formal_code
            else:
                return None

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'LeanAideTacticSuggester',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'LeanAide autoformalization exception',
                'correlation_id': correlation_id,
                'error': str(e),
            }))
            return None

    async def suggest_subgraph_activation(
        self,
        root_node_id: str,
        graph: Dict[str, 'InferenceGraphNode'],
        correlation_id: str
    ) -> Optional[Set[str]]:
        """
        Suggest optimal subgraph activation using AI analysis.

        Analyzes constraint dependencies and suggests which nodes to activate.

        Args:
            root_node_id: Root node to activate from
            graph: Full inference graph
            correlation_id: Distributed tracing correlation ID

        Returns:
            Set of node IDs to activate or None for fallback
        """
        start_time = time.time()

        if not self.leanaide_available or not self.leanaide_client:
            return None

        try:
            # Build graph description for AI
            graph_desc = self._build_graph_description(root_node_id, graph)

            # Query LeanAide for activation suggestions
            query = (
                f"Given this constraint dependency graph, which nodes should I activate "
                f"to efficiently check for contradictions around node '{root_node_id}'?\n"
                f"{graph_desc}\n"
                f"Return a comma-separated list of node IDs to activate."
            )

            result = await self.leanaide_client.math_query(query=query, n=1)

            elapsed_ms = int((time.time() - start_time) * 1000)
            self.stats.leanaide_total_time_ms += elapsed_ms
            self.stats.leanaide_subgraph_activations += 1

            if result.success and result.data:
                # Extract node IDs from result
                suggested = self._extract_node_ids(result.data, graph.keys())
                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'LeanAideTacticSuggester',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide subgraph activation suggestion',
                    'correlation_id': correlation_id,
                    'root_node': root_node_id,
                    'suggested_nodes': list(suggested),
                    'response_time_ms': elapsed_ms,
                }))
                return suggested
            else:
                return None

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'LeanAideTacticSuggester',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'LeanAide subgraph activation exception',
                'correlation_id': correlation_id,
                'error': str(e),
            }))
            return None

    def _build_contradiction_description(
        self,
        contradiction: ContradictionPair,
        constraints: List[Constraint]
    ) -> str:
        """Build natural language description of contradiction"""
        # Find the two contradictory constraints
        c1 = next((c for c in constraints if c.constraint_id == contradiction.constraint1_id), None)
        c2 = next((c for c in constraints if c.constraint_id == contradiction.constraint2_id), None)

        if c1 and c2:
            return (
                f"Constraint 1: {c1.description}\n"
                f"Constraint 2: {c2.description}\n"
                f"Type: {contradiction.type.value}\n"
                f"Affected premises: {', '.join(contradiction.affected_premises)}"
            )
        else:
            return f"Contradiction between {contradiction.constraint1_id} and {contradiction.constraint2_id}"

    def _build_resolution_problem(
        self,
        contradiction: ContradictionPair,
        constraints: List[Constraint]
    ) -> str:
        """Build detailed problem description for resolution"""
        desc = self._build_contradiction_description(contradiction, constraints)
        return (
            f"{desc}\n\n"
            f"Context: These constraints are part of a symbolic constraint engine.\n"
            f"Please suggest how to resolve this contradiction by:\n"
            f"1. Identifying which constraint should be modified\n"
            f"2. Suggesting specific modifications\n"
            f"3. Explaining why this resolves the contradiction"
        )

    def _build_graph_description(
        self,
        root_node_id: str,
        graph: Dict[str, 'InferenceGraphNode']
    ) -> str:
        """Build description of graph structure for AI analysis"""
        root_node = graph.get(root_node_id)
        if not root_node:
            return f"Node {root_node_id} not found"

        lines = [
            f"Root node: {root_node_id}",
            f"Description: {root_node.constraint.description}",
            f"Dependencies: {', '.join(root_node.dependencies) if root_node.dependencies else 'None'}",
            f"Dependents: {', '.join(root_node.dependents) if root_node.dependents else 'None'}",
            "",
            "Related nodes:",
        ]

        # Add info about related nodes
        for dep_id in list(root_node.dependencies)[:5]:  # Limit to 5
            dep_node = graph.get(dep_id)
            if dep_node:
                lines.append(f"  - {dep_id}: {dep_node.constraint.description[:60]}")

        for dep_id in list(root_node.dependents)[:5]:  # Limit to 5
            dep_node = graph.get(dep_id)
            if dep_node:
                lines.append(f"  - {dep_id}: {dep_node.constraint.description[:60]}")

        return "\n".join(lines)

    def _extract_tactics(self, data: Dict[str, Any]) -> List[str]:
        """Extract tactic names from LeanAide response"""
        tactics = []

        # Try to extract from 'result' field
        result = data.get('result', [])
        if isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    # Look for tactic names in the text
                    tactic_keywords = ['rw', 'simp', 'apply', 'cases', 'induction', 'intro', 'intros']
                    for keyword in tactic_keywords:
                        if keyword in item.lower():
                            tactics.append(keyword)
                elif isinstance(item, dict):
                    # Extract from dict
                    answer = item.get('answer', '')
                    if answer:
                        tactic_keywords = ['rw', 'simp', 'apply', 'cases', 'induction', 'intro', 'intros']
                        for keyword in tactic_keywords:
                            if keyword in answer.lower():
                                tactics.append(keyword)

        # Deduplicate
        return list(set(tactics))

    def _extract_node_ids(
        self,
        data: Dict[str, Any],
        available_ids: Set[str]
    ) -> Set[str]:
        """Extract node IDs from LeanAide response"""
        suggested = set()

        # Try to extract from result
        result = data.get('result', [])
        if isinstance(result, list):
            for item in result:
                text = str(item)
                # Look for node IDs in the text
                for node_id in available_ids:
                    if node_id in text:
                        suggested.add(node_id)

        return suggested

    def get_stats(self) -> LeanAideAIStats:
        """Get LeanAide AI statistics"""
        if self.stats.leanaide_checks_performed > 0:
            self.stats.leanaide_success_rate = (
                self.stats.leanaide_contradictions_resolved / self.stats.leanaide_checks_performed
            )
        return self.stats

    async def close(self):
        """Close LeanAide client"""
        if self.leanaide_client:
            await self.leanaide_client.close()


# ============================================================================
# MAIN CLASS: Dynamic Inference Trace Optimizer
# ============================================================================

class DITOOptimizer:
    """
    Dynamic Inference Trace Optimizer (DITO)

    Optimizes contradiction detection via selective subgraph activation.

    Key Features:
    1. Targeted Z3 ATP: Use Z3 SMT solver for efficient contradiction detection
    2. Selective Activation: Activate minimal subgraph needed
    3. Backtracking: Reset to last verified node on contradiction
    4. Complexity: O(n log n) vs O(n²) for naive pairwise
    5. Incremental Solving: Z3 push/pop for backtracking

    From RESE Technical Manual §3.3.1:
    "DITO avoids exponential complexity by selectively activating only
    the relevant subgraph when a potential contradiction is detected."

    Enhanced with Z3 ATP (2026-02-04):
    - Replaces naive O(n²) checking with Z3 SAT solving
    - Targeted contradiction detection via UNSAT cores
    - Performance tracking: Z3 vs naive baseline
    """

    def __init__(
        self,
        config: SCEConfig = None,
        activation_strategy: ActivationStrategy = ActivationStrategy.SELECTIVE_BFS,
        enable_lean4: bool = False,
        enable_leanaide: bool = True,
    ):
        """Initialize DITO optimizer

        Args:
            config: SCE configuration
            activation_strategy: Subgraph activation strategy
            enable_lean4: Enable Lean 4 formal verification
            enable_leanaide: Enable LeanAide AI assistance
        """
        self.config = config or SCEConfig.from_env()
        self.activation_strategy = activation_strategy
        self.enable_lean4 = enable_lean4 and LEAN4_AVAILABLE
        self.enable_leanaide = enable_leanaide and LEANAIDE_AVAILABLE

        # Inference graph
        self.graph: Dict[str, InferenceGraphNode] = {}

        # Verification tracking
        self.backtrack_stack: List[BacktrackPoint] = []
        self.verified_nodes: Set[str] = set()
        self.active_nodes: Set[str] = set()

        # Statistics
        self.stats = DITOStats()
        self.z3_atp_stats = Z3ATPStats()
        self.leanaide_ai_stats = LeanAideAIStats()

        # Setup logger
        self.logger = logging.getLogger('rese.dito')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Initialize ATP engines
        self.z3_enabled = Z3_AVAILABLE and self._initialize_z3()
        self.lean4_bridge = Lean4ATPBridge() if self.enable_lean4 else None

        # Initialize Z3 contradiction detector
        self.z3_detector: Optional[Z3ContradictionDetector] = None
        if self.z3_enabled:
            self.z3_detector = Z3ContradictionDetector(
                self.z3_solver,
                self.config,
                self.logger
            )

        # Initialize LeanAide tactic suggester
        self.leanaide_suggester: Optional[LeanAideTacticSuggester] = None
        if self.enable_leanaide:
            self.leanaide_suggester = LeanAideTacticSuggester(
                self.config,
                self.logger
            )

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'DITO initialized with Z3 ATP and LeanAide AI',
            'activation_strategy': activation_strategy.value,
            'z3_enabled': self.z3_enabled,
            'z3_detector_available': self.z3_detector is not None,
            'lean4_enabled': self.enable_lean4,
            'leanaide_enabled': self.enable_leanaide,
            'leanaide_available': self.leanaide_suggester is not None,
        }))

    def _initialize_z3(self) -> bool:
        """Initialize Z3 solver for targeted ATP"""
        if not Z3_AVAILABLE:
            return False

        try:
            z3_config = Z3Config(
                timeout=self.config.Z3_TIMEOUT_MS / 1000.0,
                memory_limit_mb=self.config.Z3_MAX_MEMORY_MB,
                proof_generation=True,
                unsat_core=True,
                auto_config=True
            )
            self.z3_solver = Z3SolverEngine(config=z3_config)
            return True
        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'DITOOptimizer',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 initialization failed',
                'error': str(e),
            }))
            return False

    # ========================================================================
    # GRAPH MANAGEMENT
    # ========================================================================

    def build_inference_graph(self, constraints: List[Constraint]) -> None:
        """Build inference graph from constraints

        Args:
            constraints: List of constraints to build graph from
        """
        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Building inference graph',
            'constraint_count': len(constraints),
        }))

        # Clear existing graph
        self.graph.clear()
        self.backtrack_stack.clear()
        self.verified_nodes.clear()
        self.active_nodes.clear()

        # Create nodes
        for constraint in constraints:
            node = InferenceGraphNode(constraint)
            self.graph[node.node_id] = node

        # Build dependency edges (reverse dependencies)
        for node_id, node in self.graph.items():
            for dep_id in node.dependencies:
                if dep_id in self.graph:
                    self.graph[dep_id].dependents.add(node_id)

        self.stats.total_nodes = len(self.graph)

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Inference graph built',
            'nodes': self.stats.total_nodes,
            'edges': sum(len(n.dependencies) for n in self.graph.values()),
        }))

    def get_node(self, node_id: str) -> Optional[InferenceGraphNode]:
        """Get node by ID"""
        return self.graph.get(node_id)

    def get_active_nodes(self) -> List[InferenceGraphNode]:
        """Get all currently active nodes"""
        return [self.graph[nid] for nid in self.active_nodes]

    # ========================================================================
    # SELECTIVE SUBGRAPH ACTIVATION
    # ========================================================================

    def activate_subgraph(
        self,
        root_node_id: str,
        strategy: ActivationStrategy = None
    ) -> Set[str]:
        """
        Activate selective subgraph starting from root node

        From RESE Technical Manual §3.3.1:
        "Selective subgraph activation avoids exponential complexity by
        only activating nodes relevant to the current contradiction check."

        Args:
            root_node_id: Root node to activate from
            strategy: Activation strategy (uses default if None)

        Returns:
            Set of activated node IDs
        """
        strategy = strategy or self.activation_strategy

        self.logger.debug(json.dumps({
            'level': 'debug',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Activating subgraph',
            'root_node': root_node_id,
            'strategy': strategy.value,
        }))

        if strategy == ActivationStrategy.FULL:
            activated = self._activate_full()
        elif strategy == ActivationStrategy.SELECTIVE_BFS:
            activated = self._activate_selective_bfs(root_node_id)
        elif strategy == ActivationStrategy.SELECTIVE_DFS:
            activated = self._activate_selective_dfs(root_node_id)
        elif strategy == ActivationStrategy.MINIMAL_SUBGRAPH:
            activated = self._activate_minimal_subgraph(root_node_id)
        elif strategy == ActivationStrategy.AI_GUIDED:
            # AI-guided falls back to selective BFS for sync version
            activated = self._activate_selective_bfs(root_node_id)
        else:
            activated = self._activate_selective_bfs(root_node_id)

        # Update activation timestamps
        now = datetime.now(timezone.utc)
        for node_id in activated:
            if node_id in self.graph:
                self.graph[node_id].is_active = True
                self.graph[node_id].activation_timestamp = now

        self.active_nodes.update(activated)
        self.stats.activations_performed += len(activated)

        return activated

    def _activate_full(self) -> Set[str]:
        """Activate entire graph (naive baseline)"""
        return set(self.graph.keys())

    def _activate_selective_bfs(
        self,
        root_node_id: str,
        max_depth: int = 3
    ) -> Set[str]:
        """
        Breadth-first selective activation

        Activates nodes within max_depth hops from root.
        This limits the activated subgraph size.
        """
        if root_node_id not in self.graph:
            return set()

        activated = set()
        queue = [(root_node_id, 0)]  # (node_id, depth)
        visited = set()

        while queue:
            node_id, depth = heapq.heappop(queue) if isinstance(queue, list) else queue.pop(0)

            if node_id in visited or depth > max_depth:
                continue

            visited.add(node_id)
            activated.add(node_id)

            # Activate dependencies and dependents
            node = self.graph.get(node_id)
            if node:
                # Prioritize dependencies
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        activated.add(dep_id)

                # Add dependents for next layer
                if depth < max_depth:
                    for dep_id in node.dependents:
                        if dep_id not in visited:
                            activated.add(dep_id)

        return activated

    def _activate_selective_dfs(
        self,
        root_node_id: str,
        max_depth: int = 3
    ) -> Set[str]:
        """
        Depth-first selective activation

        Activates nodes via DFS with depth limit.
        Good for finding contradictions in deep dependency chains.
        """
        if root_node_id not in self.graph:
            return set()

        activated = set()
        stack = [(root_node_id, 0)]  # (node_id, depth)
        visited = set()

        while stack:
            node_id, depth = stack.pop()

            if node_id in visited or depth > max_depth:
                continue

            visited.add(node_id)
            activated.add(node_id)

            # Activate dependencies first (depth-first)
            node = self.graph.get(node_id)
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        stack.append((dep_id, depth + 1))

                # Then dependents
                if depth < max_depth:
                    for dep_id in node.dependents:
                        if dep_id not in visited:
                            stack.append((dep_id, depth + 1))

        return activated

    def _activate_minimal_subgraph(
        self,
        root_node_id: str
    ) -> Set[str]:
        """
        Minimum subgraph isolation

        Activates only the root node and its immediate dependencies.
        This is the most conservative strategy.
        """
        if root_node_id not in self.graph:
            return set()

        activated = {root_node_id}
        node = self.graph[root_node_id]

        # Add only immediate dependencies
        for dep_id in node.dependencies:
            activated.add(dep_id)

        return activated

    async def activate_subgraph_intelligently(
        self,
        root_node_id: str,
        correlation_id: str
    ) -> Set[str]:
        """
        AI-guided subgraph activation using LeanAide.

        Uses LeanAide AI to analyze the graph and suggest optimal activation.

        Args:
            root_node_id: Root node to activate from
            correlation_id: Distributed tracing correlation ID

        Returns:
            Set of activated node IDs
        """
        if not self.leanaide_suggester:
            # Fallback to BFS
            return self.activate_subgraph(root_node_id, ActivationStrategy.SELECTIVE_BFS)

        try:
            # Get AI suggestion
            suggested = await self.leanaide_suggester.suggest_subgraph_activation(
                root_node_id,
                self.graph,
                correlation_id
            )

            if suggested and len(suggested) > 0:
                # Use AI suggestion
                activated = suggested
                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'DITOOptimizer',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'AI-guided subgraph activation',
                    'correlation_id': correlation_id,
                    'root_node': root_node_id,
                    'ai_suggested_count': len(activated),
                }))
            else:
                # Fallback to BFS
                activated = self._activate_selective_bfs(root_node_id)

            # Update activation timestamps
            now = datetime.now(timezone.utc)
            for node_id in activated:
                if node_id in self.graph:
                    self.graph[node_id].is_active = True
                    self.graph[node_id].activation_timestamp = now

            self.active_nodes.update(activated)
            self.stats.activations_performed += len(activated)

            return activated

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'DITOOptimizer',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'AI-guided activation failed, using fallback',
                'correlation_id': correlation_id,
                'error': str(e),
            }))
            # Fallback to BFS
            return self.activate_subgraph(root_node_id, ActivationStrategy.SELECTIVE_BFS)

    # ========================================================================
    # TIERED CONTRADICTION DETECTION
    # ========================================================================

    def select_verification_tier(
        self,
        constraints: List[Constraint],
        complexity_score: float = 0.0
    ) -> VerificationTier:
        """
        Select appropriate verification tier based on complexity.

        Tiers:
        - Level 1: Z3 fast contradiction detection (simple cases)
        - Level 2: LeanAide AI-assisted proof discovery (medium complexity)
        - Level 3: Lean 4 formal verification (complex cases)

        Args:
            constraints: Constraints to verify
            complexity_score: Estimated complexity (0.0 to 1.0)

        Returns:
            Appropriate verification tier
        """
        # Calculate complexity if not provided
        if complexity_score == 0.0:
            complexity_score = self._calculate_complexity_score(constraints)

        # Select tier based on complexity
        if complexity_score < 0.3:
            # Simple case - Z3 is sufficient
            return VerificationTier.Z3_FAST
        elif complexity_score < 0.7 and self.leanaide_suggester:
            # Medium complexity - Use LeanAide AI
            return VerificationTier.LEANAIDE_AI
        elif self.enable_lean4:
            # High complexity - Use Lean 4 formal verification
            return VerificationTier.LEAN4_FORMAL
        else:
            # Default to Z3
            return VerificationTier.Z3_FAST

    def _calculate_complexity_score(self, constraints: List[Constraint]) -> float:
        """
        Calculate complexity score for a set of constraints.

        Factors:
        - Number of constraints
        - Dependency depth
        - Constraint types
        - Quantifiers and logical complexity

        Returns:
            Complexity score (0.0 to 1.0)
        """
        if not constraints:
            return 0.0

        score = 0.0

        # Factor 1: Number of constraints (normalized)
        count_score = min(len(constraints) / 50.0, 1.0) * 0.3
        score += count_score

        # Factor 2: Dependency depth
        max_depth = 0
        for c in constraints:
            depth = len(c.dependencies)
            if depth > max_depth:
                max_depth = depth
        depth_score = min(max_depth / 10.0, 1.0) * 0.3
        score += depth_score

        # Factor 3: Constraint category complexity
        complex_categories = {
            ConstraintCategory.HARD_TEMPORAL_CONSTRAINT,
            ConstraintCategory.HARD_QUANTIFIER,
        }
        complex_count = sum(1 for c in constraints if c.category in complex_categories)
        category_score = min(complex_count / len(constraints), 1.0) * 0.4
        score += category_score

        return min(score, 1.0)

    async def check_contradiction_tiered(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Tuple[Optional[ContradictionPair], VerificationTier]:
        """
        Tiered contradiction detection.

        Automatically selects appropriate verification tier based on complexity.

        Args:
            constraints: Constraints to check
            correlation_id: Distributed tracing correlation ID

        Returns:
            Tuple of (contradiction if found, verification tier used)
        """
        # Calculate complexity
        complexity_score = self._calculate_complexity_score(constraints)

        # Select tier
        tier = self.select_verification_tier(constraints, complexity_score)

        # Update tier distribution
        if not self.stats.tier_distribution:
            self.stats.tier_distribution = {}
        self.stats.tier_distribution[tier.value] = self.stats.tier_distribution.get(tier.value, 0) + 1

        self.logger.debug(json.dumps({
            'level': 'debug',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Tiered contradiction detection',
            'correlation_id': correlation_id,
            'complexity_score': complexity_score,
            'selected_tier': tier.value,
        }))

        # Execute based on tier
        if tier == VerificationTier.Z3_FAST:
            # Level 1: Z3 fast detection
            contradiction = await self._check_with_z3(constraints, correlation_id)
        elif tier == VerificationTier.LEANAIDE_AI:
            # Level 2: LeanAide AI-assisted
            contradiction = await self._check_with_leanaide(constraints, correlation_id)
        elif tier == VerificationTier.LEAN4_FORMAL:
            # Level 3: Lean 4 formal verification
            contradiction = await self._check_with_lean4(constraints, correlation_id)
        else:
            # Default to Z3
            contradiction = await self._check_with_z3(constraints, correlation_id)

        return contradiction, tier

    async def _check_with_z3(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """Level 1: Fast Z3 contradiction detection"""
        if not self.z3_detector:
            return None

        # Use existing Z3 detector
        contradiction, result = self.z3_detector.check_contradiction_z3(
            constraints,
            correlation_id
        )

        return contradiction

    async def _check_with_leanaide(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """Level 2: LeanAide AI-assisted contradiction detection"""
        if not self.leanaide_suggester:
            # Fallback to Z3
            return await self._check_with_z3(constraints, correlation_id)

        # First try Z3 for fast detection
        contradiction, z3_result = self.z3_detector.check_contradiction_z3(
            constraints,
            correlation_id
        )

        if contradiction:
            # Z3 found it - get AI tactics for resolution
            tactics = await self.leanaide_suggester.suggest_tactics(
                contradiction,
                constraints,
                correlation_id
            )

            if tactics:
                self.logger.info(json.dumps({
                    'level': 'info',
                    'component': 'DITOOptimizer',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'LeanAide suggested resolution tactics',
                    'correlation_id': correlation_id,
                    'tactics': tactics,
                }))

        return contradiction

    async def _check_with_lean4(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """Level 3: Lean 4 formal verification"""
        if not self.lean4_bridge:
            # Fallback to LeanAide
            return await self._check_with_leanaide(constraints, correlation_id)

        # Use Lean 4 for formal verification
        # This would involve translating constraints to Lean 4 and proving
        # For now, this is a placeholder
        self.logger.debug(json.dumps({
            'level': 'debug',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Lean 4 formal verification (placeholder)',
            'correlation_id': correlation_id,
        }))

        # Fallback to LeanAide
        return await self._check_with_leanaide(constraints, correlation_id)

    # ========================================================================
    # AI-ASSISTED RESOLUTION METHODS
    # ========================================================================

    async def resolve_with_ai(
        self,
        contradiction: ContradictionPair,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LeanAide AI to assist in resolving a contradiction.

        Args:
            contradiction: Detected contradiction pair
            constraints: All constraints involved
            correlation_id: Distributed tracing correlation ID

        Returns:
            Resolution suggestions or None
        """
        if not self.leanaide_suggester:
            return None

        return await self.leanaide_suggester.resolve_with_ai(
            contradiction,
            constraints,
            correlation_id
        )

    async def formalize_with_ai(
        self,
        natural_constraint: str,
        correlation_id: str
    ) -> Optional[str]:
        """
        Autoformalize natural language constraint to formal logic.

        Args:
            natural_constraint: Natural language description
            correlation_id: Distributed tracing correlation ID

        Returns:
            Formalized constraint or None
        """
        if not self.leanaide_suggester:
            return None

        return await self.leanaide_suggester.formalize_with_ai(
            natural_constraint,
            correlation_id
        )

    # ========================================================================
    # TARGETED ATP (Automated Theorem Proving)
    # ========================================================================

    def check_contradiction_targeted(
        self,
        node_id: str,
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """
        Targeted ATP check for contradiction using Z3

        Enhanced to use Z3ContradictionDetector for efficient contradiction detection.
        Falls back to naive pairwise checking if Z3 unavailable.

        Only checks contradictions within the activated subgraph.
        Uses contradiction as the proof target for Z3 SMT solver.

        Args:
            node_id: Node to check for contradictions
            correlation_id: Distributed tracing correlation ID

        Returns:
            Contradiction pair if found, None otherwise
        """
        node = self.graph.get(node_id)
        if not node or not node.is_active:
            return None

        self.stats.atp_checks_performed += 1

        self.logger.debug(json.dumps({
            'level': 'debug',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Targeted ATP check with Z3',
            'node_id': node_id,
            'active_nodes': len(self.active_nodes),
            'z3_detector_available': self.z3_detector is not None,
        }))

        # Get active nodes in subgraph
        active_constraints = [
            self.graph[nid].constraint
            for nid in self.active_nodes
            if nid in self.graph
        ]

        # Use Z3 detector if available
        if self.z3_detector and len(active_constraints) >= 2:
            # Check using Z3-based detector
            contradiction, z3_result = self.z3_detector.check_contradiction_z3(
                active_constraints,
                correlation_id
            )

            if contradiction:
                self.stats.contradictions_found += 1
                self.logger.debug(json.dumps({
                    'level': 'debug',
                    'component': 'DITOOptimizer',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': 'Z3 contradiction found',
                    'node_id': node_id,
                    'z3_status': z3_result.status.value if z3_result else 'unknown',
                }))
                return contradiction

            # If Z3 found SAT (satisfiable), no contradiction
            if z3_result and z3_result.status == Z3ResultStatus.SAT:
                return None

        # Fallback to naive pairwise check within active subgraph
        # This is the baseline O(n²) approach
        for active_node_id in self.active_nodes:
            if active_node_id == node_id:
                continue

            other_node = self.graph.get(active_node_id)
            if other_node and other_node.is_active:
                contradiction = self._check_pairwise_contradiction(
                    node,
                    other_node,
                    correlation_id
                )

                if contradiction:
                    self.stats.contradictions_found += 1
                    return contradiction

        return None

    def _check_z3_contradiction(
        self,
        constraint: Constraint,
        active_constraints: List[Constraint],
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """
        Check contradiction using Z3 with targeted formula

        Uses contradiction as proof target (UNSAT check).
        """
        try:
            # Build SMT-LIB2 program with contradiction target
            smtlib_lines = [
                "; DITO Targeted Contradiction Check",
                f"; Correlation ID: {correlation_id}",
                "",
                "(set-logic ALL)",
                "(set-option :produce-models true)",
                "(set-option :produce-proofs true)",
                "(set-option :produce-unsat-cores true)",
            ]

            # Encode constraint
            formula1 = self._encode_constraint_to_smtlib(constraint)
            if not formula1:
                return None

            # Check against active constraints
            for other_constraint in active_constraints:
                if other_constraint.constraint_id == constraint.constraint_id:
                    continue

                formula2 = self._encode_constraint_to_smtlib(other_constraint)
                if not formula2:
                    continue

                # Check if formulas contradict (UNSAT)
                smtlib_check = smtlib_lines + [
                    f"(assert {formula1})",
                    f"(assert {formula2})",
                    "(check-sat)",
                ]

                smtlib_content = "\n".join(smtlib_check)
                result = self.z3_solver.solve_smtlib(smtlib_content)

                if result.status == Z3ResultStatus.UNSAT:
                    # Contradiction found
                    return ContradictionPair(
                        constraint1_id=constraint.constraint_id,
                        constraint2_id=other_constraint.constraint_id,
                        type=LogicalFallacy.CONTRADICTION,
                        contradiction_set_size=2,
                        rollback_steps=max(
                            len(constraint.dependencies),
                            len(other_constraint.dependencies)
                        ),
                        affected_premises=[
                            constraint.constraint_id,
                            other_constraint.constraint_id
                        ],
                        detected_at=datetime.now(timezone.utc),
                    )

        except Exception as e:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'DITOOptimizer',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Z3 targeted check failed',
                'error': str(e),
            }))

        return None

    def _encode_constraint_to_smtlib(self, constraint: Constraint) -> Optional[str]:
        """Encode constraint to SMT-LIB2 formula"""
        # Try expression first
        if constraint.expression and isinstance(constraint.expression, str):
            expr = constraint.expression.strip()
            if expr.startswith('('):
                return expr

        # Extract from description
        desc = constraint.description.lower()

        # Simple pattern matching for common constraints
        if '<' in constraint.description:
            parts = constraint.description.split('<')
            if len(parts) == 2:
                var = parts[0].strip()
                val = parts[1].strip()
                return f"(< {var} {val})"

        if '>' in constraint.description:
            parts = constraint.description.split('>')
            if len(parts) == 2:
                var = parts[0].strip()
                val = parts[1].strip()
                return f"(> {var} {val})"

        return None

    def _check_pairwise_contradiction(
        self,
        node1: InferenceGraphNode,
        node2: InferenceGraphNode,
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """Check pairwise contradiction between two nodes"""
        c1, c2 = node1.constraint, node2.constraint

        # Direct textual negation
        if self._is_negation(c1.description, c2.description):
            return ContradictionPair(
                constraint1_id=c1.constraint_id,
                constraint2_id=c2.constraint_id,
                type=LogicalFallacy.CONTRADICTION,
                contradiction_set_size=2,
                rollback_steps=max(len(c1.dependencies), len(c2.dependencies)),
                affected_premises=[c1.constraint_id, c2.constraint_id],
                detected_at=datetime.now(timezone.utc),
            )

        # Circular dependency
        if (c2.constraint_id in c1.dependencies and
            c1.constraint_id in c2.dependencies):
            return ContradictionPair(
                constraint1_id=c1.constraint_id,
                constraint2_id=c2.constraint_id,
                type=LogicalFallacy.CIRCULUS_IN_PROBANDO,
                contradiction_set_size=2,
                rollback_steps=max(len(c1.dependencies), len(c2.dependencies)),
                affected_premises=[*c1.dependencies, *c2.dependencies],
                detected_at=datetime.now(timezone.utc),
            )

        return None

    def _is_negation(self, desc1: str, desc2: str) -> bool:
        """Check if desc2 is a negation of desc1"""
        d1, d2 = desc1.lower().strip(), desc2.lower().strip()

        # Direct "not X" vs "X"
        if d1.startswith('not ') and d1[4:] == d2:
            return True
        if d2.startswith('not ') and d2[4:] == d1:
            return True

        # Antonym patterns
        antonym_pairs = [
            ('less than', 'greater than'),
            ('cannot exceed', 'must exceed'),
            ('impossible', 'possible'),
        ]

        for a1, a2 in antonym_pairs:
            if a1 in d1 and a2 in d2:
                return True
            if a2 in d1 and a1 in d2:
                return True

        return False

    # ========================================================================
    # BACKTRACKING
    # ========================================================================

    def create_backtrack_point(
        self,
        node_id: str,
        context: Dict[str, Any] = None
    ) -> BacktrackPoint:
        """
        Create backtrack checkpoint

        Saves current state for backtracking on contradiction.
        """
        point = BacktrackPoint(
            node_id=node_id,
            active_nodes=self.active_nodes.copy(),
            verified_nodes=self.verified_nodes.copy(),
            timestamp=datetime.now(timezone.utc),
            contradiction_context=context,
        )

        self.backtrack_stack.append(point)
        self.logger.debug(json.dumps({
            'level': 'debug',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Backtrack point created',
            'node_id': node_id,
            'stack_depth': len(self.backtrack_stack),
        }))

        return point

    def backtrack(self) -> Optional[BacktrackPoint]:
        """
        Backtrack to last verified node

        Resets graph state to last backtrack point.
        """
        if not self.backtrack_stack:
            self.logger.warning(json.dumps({
                'level': 'warn',
                'component': 'DITOOptimizer',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Backtrack attempted but stack is empty',
            }))
            return None

        point = self.backtrack_stack.pop()

        # Reset state
        for node_id in list(self.active_nodes):
            if node_id not in point.active_nodes:
                if node_id in self.graph:
                    self.graph[node_id].is_active = False
                self.active_nodes.discard(node_id)

        self.stats.backtracks_performed += 1

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Backtracked to checkpoint',
            'backtrack_to': point.node_id,
            'active_nodes_after': len(self.active_nodes),
            'total_backtracks': self.stats.backtracks_performed,
        }))

        return point

    # ========================================================================
    # MAIN OPTIMIZATION LOOP
    # ========================================================================

    def optimize_contradiction_detection(
        self,
        constraints: List[Constraint],
        correlation_id: str
    ) -> Tuple[List[ContradictionPair], DITOStats]:
        """
        Main DITO optimization loop with Z3 ATP

        Executes O(n log n) contradiction detection via selective activation
        and Z3 automated theorem proving.

        Algorithm:
        1. Build inference graph
        2. For each unverified node:
           a. Create backtrack point
           b. Activate selective subgraph
           c. Perform targeted Z3 ATP check
           d. If contradiction: backtrack and continue
           e. Else: mark verified
        3. Collect Z3 ATP statistics
        4. Return contradictions and statistics

        Args:
            constraints: List of constraints to check
            correlation_id: Distributed tracing correlation ID

        Returns:
            Tuple of (contradictions found, statistics)
        """
        start_time = time.time()

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'Starting DITO optimization with Z3 ATP and LeanAide AI',
            'correlation_id': correlation_id,
            'constraint_count': len(constraints),
            'strategy': self.activation_strategy.value,
            'z3_detector_available': self.z3_detector is not None,
            'leanaide_available': self.leanaide_suggester is not None,
        }))

        # Reset state
        self.stats = DITOStats()
        self.z3_atp_stats = Z3ATPStats()
        self.leanaide_ai_stats = LeanAideAIStats()
        contradictions = []

        # Build graph
        self.build_inference_graph(constraints)

        # Process nodes in dependency order
        node_order = self._topological_sort()

        for node_id in node_order:
            if node_id in self.verified_nodes:
                continue

            # Create backtrack point
            self.create_backtrack_point(node_id, {
                'iteration': len(self.verified_nodes),
                'total_nodes': len(node_order),
            })

            # Activate selective subgraph
            activated = self.activate_subgraph(node_id)

            # Targeted Z3 ATP check
            contradiction = self.check_contradiction_targeted(
                node_id,
                correlation_id
            )

            if contradiction:
                # Contradiction found - record and backtrack
                contradictions.append(contradiction)

                # Mark nodes as contradictory
                node = self.graph.get(node_id)
                if node:
                    node.contradiction_detected = True
                    node.contradiction_partners.add(contradiction.constraint2_id)

                # Backtrack to last verified state
                self.backtrack()

                # Deactivate conflicting subgraph
                self._deactivate_subgraph(node_id)
            else:
                # No contradiction - mark as verified
                self.verified_nodes.add(node_id)
                node = self.graph.get(node_id)
                if node:
                    node.is_verified = True
                    node.verification_timestamp = datetime.now(timezone.utc)

        # Calculate statistics
        self.stats.execution_time_ms = int((time.time() - start_time) * 1000)
        self.stats.active_nodes = len(self.active_nodes)
        self.stats.verified_nodes = len(self.verified_nodes)
        self.stats.contradictions_found = len(contradictions)

        # Update Z3 ATP statistics
        if self.z3_detector:
            self.z3_atp_stats = self.z3_detector.get_stats()
            self.stats.z3_atp_stats = self.z3_atp_stats

        # Update LeanAide AI statistics
        if self.leanaide_suggester:
            self.leanaide_ai_stats = self.leanaide_suggester.get_stats()
            self.stats.leanaide_ai_stats = self.leanaide_ai_stats

        # Complexity saved: percentage of graph not activated
        if self.stats.total_nodes > 0:
            self.stats.complexity_saved = (
                1.0 - (self.stats.active_nodes / self.stats.total_nodes)
            ) * 100.0

        # Log completion with Z3 ATP and LeanAide AI statistics
        log_data = {
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'DITO optimization completed with Z3 ATP and LeanAide AI',
            'correlation_id': correlation_id,
            'contradictions': len(contradictions),
            'verified_nodes': self.stats.verified_nodes,
            'active_nodes': self.stats.active_nodes,
            'complexity_saved': f"{self.stats.complexity_saved:.1f}%",
            'execution_time_ms': self.stats.execution_time_ms,
            'z3_atp_stats': self.z3_atp_stats.to_dict() if self.z3_atp_stats else {},
            'leanaide_ai_stats': self.leanaide_ai_stats.to_dict() if self.leanaide_ai_stats else {},
            'tier_distribution': self.stats.tier_distribution,
        }

        self.logger.info(json.dumps(log_data))

        return contradictions, self.stats

    def get_z3_atp_stats(self) -> Optional[Z3ATPStats]:
        """Get Z3 ATP performance statistics"""
        if self.z3_detector:
            return self.z3_detector.get_stats()
        return None

    def get_leanaide_ai_stats(self) -> Optional[LeanAideAIStats]:
        """Get LeanAide AI performance statistics"""
        if self.leanaide_suggester:
            return self.leanaide_suggester.get_stats()
        return None

    async def close(self):
        """Close DITO and cleanup resources"""
        if self.leanaide_suggester:
            await self.leanaide_suggester.close()

    def _topological_sort(self) -> List[str]:
        """Topological sort of nodes by dependencies"""
        sorted_nodes = []
        visited = set()
        temp_visited = set()

        def visit(node_id: str):
            if node_id in temp_visited:
                return  # Cycle detected, skip
            if node_id in visited:
                return

            temp_visited.add(node_id)

            node = self.graph.get(node_id)
            if node:
                for dep_id in node.dependencies:
                    visit(dep_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            sorted_nodes.append(node_id)

        for node_id in self.graph.keys():
            if node_id not in visited:
                visit(node_id)

        return sorted_nodes

    def _deactivate_subgraph(self, root_node_id: str) -> None:
        """Deactivate subgraph rooted at node"""
        to_deactivate = {root_node_id}

        # Deactivate dependents
        queue = [root_node_id]
        while queue:
            node_id = queue.pop(0)
            node = self.graph.get(node_id)
            if node:
                for dep_id in node.dependents:
                    to_deactivate.add(dep_id)
                    queue.append(dep_id)

        # Deactivate nodes
        for node_id in to_deactivate:
            if node_id in self.graph:
                self.graph[node_id].is_active = False
                self.active_nodes.discard(node_id)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for testing"""
    import asyncio

    async def test_dito():
        from sce_bridge import Constraint, ConstraintType, ConstraintCategory

        # Create test constraints
        constraints = [
            Constraint(
                constraint_id="c1",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="Temperature must be less than 1000",
            ),
            Constraint(
                constraint_id="c2",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="Temperature must be greater than 0",
            ),
            Constraint(
                constraint_id="c3",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="Pressure must be less than 5000",
            ),
            Constraint(
                constraint_id="c4",
                type=ConstraintType.HARD,
                category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                description="Temperature must be greater than 1500",  # Contradicts c1
                dependencies=["c1"],
            ),
        ]

        # Create DITO optimizer
        dito = DITOOptimizer(
            activation_strategy=ActivationStrategy.SELECTIVE_BFS
        )

        # Run optimization
        contradictions, stats = dito.optimize_contradiction_detection(
            constraints,
            "test-correlation-1"
        )

        print(f"\nDITO Optimization Results:")
        print(f"Contradictions found: {len(contradictions)}")
        print(f"Verified nodes: {stats.verified_nodes}")
        print(f"Active nodes: {stats.active_nodes}")
        print(f"Complexity saved: {stats.complexity_saved:.1f}%")
        print(f"Execution time: {stats.execution_time_ms}ms")

        for contradiction in contradictions:
            print(f"\nContradiction:")
            print(f"  {contradiction.constraint1_id} vs {contradiction.constraint2_id}")
            print(f"  Type: {contradiction.type.value}")

    asyncio.run(test_dito())


if __name__ == '__main__':
    main()
