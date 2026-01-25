"""
Intervention Simulator for Causal Equivalence Testing

Simulate do-calculus interventions to test mechanistic equivalence.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, List, Set
import numpy as np
import networkx as nx
from ..core.fdg import FunctionalDependencyGraph, EdgeType


class InterventionSimulator:
    """
    Simulate interventions on FDGs to test causal equivalence

    Implements do-calculus for intervention testing
    """

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples

    def compare_intervention_responses(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compare responses to interventions across both FDGs

        Args:
            fdg1: First FDG
            fdg2: Second FDG
            mapping: Node mapping from fdg1 to fdg2

        Returns:
            Similarity score in [0, 1]
        """
        if not mapping:
            return 0.0

        # Sample nodes to intervene on
        nodes_to_test = min(10, len(mapping))
        sample_nodes = list(mapping.keys())[:nodes_to_test]

        similarity_sum = 0.0
        valid_tests = 0

        for node in sample_nodes:
            if node not in mapping:
                continue

            mapped_node = mapping[node]

            # Simulate intervention on both
            effect1 = self._simulate_intervention(fdg1, node)
            effect2 = self._simulate_intervention(fdg2, mapped_node)

            # Compare effects
            if effect1 and effect2:
                similarity = self._compare_effects(effect1, effect2)
                similarity_sum += similarity
                valid_tests += 1

        return similarity_sum / valid_tests if valid_tests > 0 else 0.5

    def _simulate_intervention(
        self,
        fdg: FunctionalDependencyGraph,
        node_id: str,
        value: float = 1.0
    ) -> Dict[str, float]:
        """
        Simulate intervention do(node_id = value)

        Propagates effects along causal edges
        """
        if node_id not in fdg.nodes:
            return {}

        effect = {node_id: value}
        queue = [node_id]
        visited = {node_id}

        while queue:
            current = queue.pop(0)

            # Propagate to descendants via causal edges
            for neighbor in fdg.graph.successors(current):
                if neighbor in visited:
                    continue

                # Check if edge is causal
                edge = fdg.get_edge(current, neighbor)
                if edge and edge.edge_type == EdgeType.CAUSAL:
                    # Simple propagation: multiply by edge weight
                    effect[neighbor] = effect[current] * edge.weight
                    queue.append(neighbor)
                    visited.add(neighbor)

        return effect

    def _compare_effects(
        self,
        effect1: Dict[str, float],
        effect2: Dict[str, float],
        tolerance: float = 0.1
    ) -> float:
        """
        Compare two intervention effects

        Returns similarity in [0, 1]
        """
        if not effect1 or not effect2:
            return 0.0

        # Get all affected nodes
        all_nodes = set(effect1.keys()) | set(effect2.keys())

        if not all_nodes:
            return 1.0

        # Compare effects
        matches = 0
        for node in all_nodes:
            val1 = effect1.get(node, 0.0)
            val2 = effect2.get(node, 0.0)

            if abs(val1 - val2) <= tolerance:
                matches += 1

        return matches / len(all_nodes)

    def compute_causal_effect(
        self,
        fdg: FunctionalDependencyGraph,
        source: str,
        target: str
    ) -> float:
        """
        Compute causal effect of source on target

        Uses path-based analysis
        """
        # Find all causal paths from source to target
        try:
            paths = list(nx.all_simple_paths(
                fdg.get_causal_subgraph(),
                source,
                target,
                cutoff=5  # Max path length
            ))

            if not paths:
                return 0.0

            # Aggregate effects along all paths
            total_effect = 0.0
            for path in paths:
                path_effect = 1.0
                for i in range(len(path) - 1):
                    edge = fdg.get_edge(path[i], path[i+1])
                    if edge:
                        path_effect *= edge.weight
                total_effect += path_effect

            return total_effect

        except nx.NetworkXNoPath:
            return 0.0

    def test_interventional_equivalence(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str],
        num_tests: int = 20
    ) -> float:
        """
        Test interventional equivalence across multiple random interventions

        Returns:
            Fraction of equivalent interventions
        """
        if not mapping:
            return 0.0

        equivalent_count = 0
        total_tests = 0

        # Test random pairs
        for _ in range(num_tests):
            # Sample source-target pairs
            nodes1 = list(mapping.keys())
            if len(nodes1) < 2:
                continue

            # Random pair
            idx1 = np.random.randint(0, len(nodes1))
            idx2 = np.random.randint(0, len(nodes1))
            while idx2 == idx1:
                idx2 = np.random.randint(0, len(nodes1))

            source1 = nodes1[idx1]
            target1 = nodes1[idx2]

            if source1 not in mapping or target1 not in mapping:
                continue

            source2 = mapping[source1]
            target2 = mapping[target1]

            # Compute causal effects
            effect1 = self.compute_causal_effect(fdg1, source1, target1)
            effect2 = self.compute_causal_effect(fdg2, source2, target2)

            # Check equivalence
            if abs(effect1 - effect2) < 0.1:
                equivalent_count += 1

            total_tests += 1

        return equivalent_count / total_tests if total_tests > 0 else 0.0
