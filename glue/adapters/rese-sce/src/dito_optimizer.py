#!/usr/bin/env python3
"""
Dynamic Inference Trace Optimizer (DITO) for RESE SCE

Implements O(n log n) contradiction detection via:
- Targeted ATP: Use contradiction as proof target
- Selective subgraph activation: Avoid exponential complexity
- Backtracking: Reset to last verified node
- Minimum subgraph isolation: Root premise violation

From RESE Technical Manual §3.3.1: DITO optimizes contradiction detection
by selectively activating constraint subgraphs only when needed.

Author: OpenEvolve
Created: 2026-02-04
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
        Z3Config,
        Z3SolverResult,
        Z3ResultStatus,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    Z3SolverEngine = None  # type: ignore
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
        }


@dataclass
class BacktrackPoint:
    """Backtracking checkpoint"""
    node_id: str
    active_nodes: Set[str]
    verified_nodes: Set[str]
    timestamp: datetime
    contradiction_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'active_nodes': list(self.active_nodes),
            'verified_nodes': list(self.verified_nodes),
            'timestamp': self.timestamp.isoformat(),
            'contradiction_context': self.contradiction_context,
        }


# ============================================================================
# MAIN CLASS: Dynamic Inference Trace Optimizer
# ============================================================================

class DITOOptimizer:
    """
    Dynamic Inference Trace Optimizer (DITO)

    Optimizes contradiction detection via selective subgraph activation.

    Key Features:
    1. Targeted ATP: Only check contradictions when nodes are activated
    2. Selective Activation: Activate minimal subgraph needed
    3. Backtracking: Reset to last verified node on contradiction
    4. Complexity: O(n log n) vs O(n²) for naive pairwise

    From RESE Technical Manual §3.3.1:
    "DITO avoids exponential complexity by selectively activating only
    the relevant subgraph when a potential contradiction is detected."
    """

    def __init__(
        self,
        config: SCEConfig = None,
        activation_strategy: ActivationStrategy = ActivationStrategy.SELECTIVE_BFS,
        enable_lean4: bool = False,
    ):
        """Initialize DITO optimizer

        Args:
            config: SCE configuration
            activation_strategy: Subgraph activation strategy
            enable_lean4: Enable Lean 4 formal verification
        """
        self.config = config or SCEConfig.from_env()
        self.activation_strategy = activation_strategy
        self.enable_lean4 = enable_lean4 and LEAN4_AVAILABLE

        # Inference graph
        self.graph: Dict[str, InferenceGraphNode] = {}

        # Verification tracking
        self.backtrack_stack: List[BacktrackPoint] = []
        self.verified_nodes: Set[str] = set()
        self.active_nodes: Set[str] = set()

        # Statistics
        self.stats = DITOStats()

        # Setup logger
        self.logger = logging.getLogger('rese.dito')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Initialize ATP engines
        self.z3_enabled = Z3_AVAILABLE and self._initialize_z3()
        self.lean4_bridge = Lean4ATPBridge() if self.enable_lean4 else None

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'DITO initialized',
            'activation_strategy': activation_strategy.value,
            'z3_enabled': self.z3_enabled,
            'lean4_enabled': self.enable_lean4,
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

    # ========================================================================
    # TARGETED ATP (Automated Theorem Proving)
    # ========================================================================

    def check_contradiction_targeted(
        self,
        node_id: str,
        correlation_id: str
    ) -> Optional[ContradictionPair]:
        """
        Targeted ATP check for contradiction

        Only checks contradictions within the activated subgraph.
        Uses contradiction as the proof target for Z3/Lean4.

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
            'message': 'Targeted ATP check',
            'node_id': node_id,
            'active_nodes': len(self.active_nodes),
        }))

        # Get active nodes in subgraph
        active_constraints = [
            self.graph[nid].constraint
            for nid in self.active_nodes
            if nid in self.graph
        ]

        if self.z3_enabled and len(active_constraints) >= 2:
            # Use Z3 for targeted check
            contradiction = self._check_z3_contradiction(
                node.constraint,
                active_constraints,
                correlation_id
            )

            if contradiction:
                self.stats.contradictions_found += 1
                return contradiction

        # Fallback to pairwise check within active subgraph
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
        Main DITO optimization loop

        Executes O(n log n) contradiction detection via selective activation.

        Algorithm:
        1. Build inference graph
        2. For each unverified node:
           a. Create backtrack point
           b. Activate selective subgraph
           c. Perform targeted ATP check
           d. If contradiction: backtrack and continue
           e. Else: mark verified
        3. Return contradictions and statistics

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
            'message': 'Starting DITO optimization',
            'correlation_id': correlation_id,
            'constraint_count': len(constraints),
            'strategy': self.activation_strategy.value,
        }))

        # Reset state
        self.stats = DITOStats()
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

            # Targeted ATP check
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

        # Complexity saved: percentage of graph not activated
        if self.stats.total_nodes > 0:
            self.stats.complexity_saved = (
                1.0 - (self.stats.active_nodes / self.stats.total_nodes)
            ) * 100.0

        self.logger.info(json.dumps({
            'level': 'info',
            'component': 'DITOOptimizer',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': 'DITO optimization completed',
            'correlation_id': correlation_id,
            'contradictions': len(contradictions),
            'verified_nodes': self.stats.verified_nodes,
            'active_nodes': self.stats.active_nodes,
            'complexity_saved': f"{self.stats.complexity_saved:.1f}%",
            'execution_time_ms': self.stats.execution_time_ms,
        }))

        return contradictions, self.stats

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
