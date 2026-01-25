"""
Γ₁ Solvability Index Engine (S)

Implements solvability prediction for CSP instances.
Higher solvability = more tractable = easier to solve.

Components:
- Phase Distance: Distance from phase transition (hardest region)
- Propagation Effectiveness: How much constraints reduce search space
- Structure Quality: Constraint topology quality (tree-width, etc.)
- Heuristic Effectiveness: How well heuristics will perform
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from collections import deque
import math
import networkx as nx
import numpy as np
from gamma1.core.csp_models import CSPInstance


@dataclass
class SolvabilityComponents:
    """
    Solvability Index (S) Components

    Attributes:
        phase_distance: Distance from phase transition
        propagation: Propagation effectiveness
        structure: Constraint structure quality
        heuristic: Heuristic effectiveness prediction
    """
    phase_distance: float = 0.0
    propagation: float = 0.0
    structure: float = 0.0
    heuristic: float = 0.0

    def total(self, weights: tuple = (0.3, 0.3, 0.2, 0.2)) -> float:
        """
        Calculate total S with given weights

        Args:
            weights: (w_phase, w_propagation, w_structure, w_heuristic)

        Returns:
            Total solvability S ∈ [0, 1]
        """
        S = (weights[0] * self.phase_distance +
             weights[1] * self.propagation +
             weights[2] * self.structure +
             weights[3] * self.heuristic)
        return max(0.0, min(1.0, S))


class SolvabilityIndex:
    """
    Calculate solvability index for CSP instances

    S = w_phase * S_phase + w_prop * S_prop +
        w_struct * S_struct + w_heur * S_heur

    Higher S = more solvable = easier to solve
    """

    def __init__(self, weights: tuple = (0.3, 0.3, 0.2, 0.2)):
        """
        Initialize solvability calculator

        Args:
            weights: (w_phase, w_propagation, w_structure, w_heuristic)
        """
        if len(weights) != 4 or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.weights = weights

    def calculate(self, csp: CSPInstance) -> SolvabilityComponents:
        """
        Calculate solvability index for CSP

        Args:
            csp: CSP instance

        Returns:
            SolvabilityComponents with all solvability measures
        """
        components = SolvabilityComponents()

        # Calculate each component
        components.phase_distance = self._phase_transition_distance(csp)
        components.propagation = self._propagation_effectiveness(csp)
        components.structure = self._constraint_structure_quality(csp)
        components.heuristic = self._heuristic_effectiveness(csp)

        return components

    def _phase_transition_distance(self, csp: CSPInstance) -> float:
        """
        Calculate distance from phase transition

        Problems near phase transition are hardest.
        Farther from transition = easier to solve.

        Args:
            csp: CSP instance

        Returns:
            Normalized distance ∈ [0, 1], higher = farther = easier
        """
        if not csp.constraints:
            return 1.0  # No constraints = trivial

        # Calculate constraint tightness
        tightness_values = []

        for constraint in csp.constraints:
            total_tuples = 1
            for var_name in constraint.variables:
                var = csp.get_variable(var_name)
                if var:
                    total_tuples *= var.domain_size()

            if total_tuples > 0:
                t = 1.0 - (len(constraint.allowed_tuples) / total_tuples)
                tightness_values.append(t)

        if not tightness_values:
            avg_tightness = 0.5
        else:
            avg_tightness = np.mean(tightness_values)

        # Calculate constraint density
        n = csp.num_variables()
        m = csp.num_constraints()

        if n < 2:
            density = 0.0
        else:
            max_binary_constraints = n * (n - 1) / 2
            density = m / max_binary_constraints

        # Phase transition point (empirical for random CSP)
        # Critical tightness ≈ 0.5, critical density ≈ 0.5
        critical_tightness = 0.5
        critical_density = 0.5

        # Euclidean distance in (tightness, density) space
        distance = math.sqrt(
            (avg_tightness - critical_tightness) ** 2 +
            (density - critical_density) ** 2
        )

        # Normalize to [0, 1]
        # Maximum distance ≈ sqrt(0.5² + 0.5²) ≈ 0.707
        max_distance = 0.707
        S_phase = distance / max_distance

        return min(1.0, S_phase)

    def _propagation_effectiveness(self, csp: CSPInstance) -> float:
        """
        Calculate propagation effectiveness

        Simulates AC-3 to estimate domain reduction.

        Args:
            csp: CSP instance

        Returns:
            Propagation effectiveness ∈ [0, 1]
        """
        if not csp.variables:
            return 0.0

        # Initial domain sizes
        initial_size = sum(v.domain_size() for v in csp.variables)

        if initial_size == 0:
            return 0.0

        # Simulate AC-3 (simplified)
        reduced_domains = {v.name: list(v.domain) for v in csp.variables}
        queue = deque(csp.constraints)

        reductions = 0
        iterations = 0
        max_iterations = 100

        while queue and iterations < max_iterations:
            constraint = queue.popleft()

            for var_name in constraint.variables:
                if var_name not in reduced_domains:
                    continue

                old_size = len(reduced_domains[var_name])

                # Remove values that violate constraint
                valid_values = []
                for value in reduced_domains[var_name]:
                    # Check if this value appears in any allowed tuple
                    if self._value_allowed(constraint, var_name, value, reduced_domains):
                        valid_values.append(value)

                reduced_domains[var_name] = valid_values
                new_size = len(reduced_domains[var_name])

                if new_size < old_size:
                    reductions += (old_size - new_size)
                    # Add neighboring constraints to queue
                    for other_constraint in csp.constraints:
                        if other_constraint != constraint:
                            if var_name in other_constraint.variables:
                                queue.append(other_constraint)

            iterations += 1

        # Final domain sizes
        final_size = sum(len(dom) for dom in reduced_domains.values())

        # Effectiveness: fraction of domain values removed
        if initial_size > 0:
            S_prop = (initial_size - final_size) / initial_size
        else:
            S_prop = 0.0

        return max(0.0, min(1.0, S_prop))

    def _value_allowed(self, constraint, var_name: str, value, domains: Dict) -> bool:
        """Check if value is allowed in any tuple"""
        var_idx = constraint.variables.index(var_name)

        for tuple_values in constraint.allowed_tuples:
            # Check if this tuple allows the value
            if tuple_values[var_idx] == value:
                # Check if other variables' values are in their domains
                tuple_allowed = True
                for i, other_var_name in enumerate(constraint.variables):
                    if i != var_idx:
                        if other_var_name in domains:
                            if tuple_values[i] not in domains[other_var_name]:
                                tuple_allowed = False
                                break

                if tuple_allowed:
                    return True

        return False

    def _constraint_structure_quality(self, csp: CSPInstance) -> float:
        """
        Calculate constraint structure quality

        Measures quality of constraint topology.
        Higher = better structure = more solvable.

        Args:
            csp: CSP instance

        Returns:
            Normalized structure quality ∈ [0, 1]
        """
        G = csp.constraint_graph

        if G.number_of_nodes() == 0:
            return 0.0

        # Quality 1: Tree-width approximation (lower = easier)
        treewidth = csp.tree_width_approximation()
        n = csp.num_variables()

        if n > 0:
            treewidth_score = 1.0 - (treewidth / n)
        else:
            treewidth_score = 0.0

        # Quality 2: Constraint consistency
        consistency_score = 1.0
        for constraint in csp.constraints:
            if len(constraint.allowed_tuples) == 0:
                consistency_score = 0.0
                break

        # Quality 3: Domain-to-constraint ratio
        if n > 0:
            ratio = csp.num_constraints() / n
            # Optimal ratio ≈ 2
            ratio_score = 1.0 - abs(ratio - 2.0) / 5.0
            ratio_score = max(0.0, ratio_score)
        else:
            ratio_score = 0.0

        # Combine
        S_struct = (0.4 * treewidth_score +
                   0.3 * consistency_score +
                   0.3 * ratio_score)

        return max(0.0, min(1.0, S_struct))

    def _heuristic_effectiveness(self, csp: CSPInstance) -> float:
        """
        Predict heuristic effectiveness

        Predicts how well standard heuristics will perform.

        Args:
            csp: CSP instance

        Returns:
            Predicted heuristic effectiveness ∈ [0, 1]
        """
        # Empty CSP has no heuristic effectiveness needed
        if not csp.variables:
            return 0.0

        # Effectiveness 1: MRV (Minimum Remaining Values)
        domain_sizes = [v.domain_size() for v in csp.variables]

        if domain_sizes:
            if len(domain_sizes) > 1:
                domain_cv = np.std(domain_sizes) / (np.mean(domain_sizes) + 1e-9)
                mrv_effectiveness = 1.0 / (1.0 + domain_cv)
            else:
                mrv_effectiveness = 1.0
        else:
            mrv_effectiveness = 0.0

        # Effectiveness 2: LCV (Least Constraining Value)
        tightness_values = []

        for constraint in csp.constraints:
            total = 1
            for var_name in constraint.variables:
                var = csp.get_variable(var_name)
                if var:
                    total *= var.domain_size()

            if total > 0:
                t = 1.0 - (len(constraint.allowed_tuples) / total)
                tightness_values.append(t)

        if tightness_values:
            tightness_range = max(tightness_values) - min(tightness_values)
            lcv_effectiveness = 1.0 - tightness_range
        else:
            lcv_effectiveness = 0.5

        # Effectiveness 3: Decomposability
        if csp.is_connected():
            decomposability = 0.0
        else:
            n_components = csp.num_connected_components()
            n = csp.num_variables()
            decomposability = n_components / n if n > 0 else 0.0

        # Combine
        S_heur = (0.4 * mrv_effectiveness +
                 0.3 * lcv_effectiveness +
                 0.3 * decomposability)

        return max(0.0, min(1.0, S_heur))


if __name__ == "__main__":
    print("=" * 70)
    print("Solvability Index Engine - Demonstration")
    print("=" * 70)

    from gamma1.core.csp_models import create_test_csp, create_tree_csp, create_dense_csp

    # Test on different CSP types
    calculator = SolvabilityIndex()

    # Test CSP
    test_csp = create_test_csp(n_variables=10, domain_size=5)
    test_solvability = calculator.calculate(test_csp)
    print(f"\n[OK] Test CSP solvability: {test_solvability.total():.3f}")
    print(f"  Phase distance: {test_solvability.phase_distance:.3f}")
    print(f"  Propagation: {test_solvability.propagation:.3f}")
    print(f"  Structure: {test_solvability.structure:.3f}")
    print(f"  Heuristic: {test_solvability.heuristic:.3f}")

    # Tree CSP (should have higher solvability - more structured)
    tree_csp = create_tree_csp(n_variables=10, domain_size=5)
    tree_solvability = calculator.calculate(tree_csp)
    print(f"\n[OK] Tree CSP solvability: {tree_solvability.total():.3f}")
    print(f"  Phase distance: {tree_solvability.phase_distance:.3f}")
    print(f"  Propagation: {tree_solvability.propagation:.3f}")
    print(f"  Structure: {tree_solvability.structure:.3f}")
    print(f"  Heuristic: {tree_solvability.heuristic:.3f}")

    # Dense CSP (should have lower solvability - near phase transition)
    dense_csp = create_dense_csp(n_variables=10, domain_size=5)
    dense_solvability = calculator.calculate(dense_csp)
    print(f"\n[OK] Dense CSP solvability: {dense_solvability.total():.3f}")
    print(f"  Phase distance: {dense_solvability.phase_distance:.3f}")
    print(f"  Propagation: {dense_solvability.propagation:.3f}")
    print(f"  Structure: {dense_solvability.structure:.3f}")
    print(f"  Heuristic: {dense_solvability.heuristic:.3f}")

    # Comparison
    print(f"\n[OK] Solvability comparison:")
    print(f"  Tree > Test: {tree_solvability.total() > test_solvability.total()}")
    print(f"  Test > Dense: {test_solvability.total() > dense_solvability.total()}")

    print("\n" + "=" * 70)
    print("[OK] Solvability index engine demonstration complete")
    print("=" * 70)
