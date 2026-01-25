"""
Causal Similarity Analyzer

Analyzes causal structure similarity between FDGs.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, List, Set, Tuple
import networkx as nx
from .fdg import FunctionalDependencyGraph, EdgeType
from .result import SimilarityResult
from ..algorithms.intervention import InterventionSimulator


class CausalSimilarityAnalyzer:
    """
    Analyze causal similarity between FDGs
    """

    def __init__(self):
        self.intervention_sim = InterventionSimulator()

    def analyze(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compute causal similarity score

        Args:
            fdg1: First FDG
            fdg2: Second FDG
            mapping: Node mapping from fdg1 to fdg2

        Returns:
            Causal similarity score in [0, 1]
        """
        if not mapping:
            return 0.0

        # Compare causal graph structures
        graph_score = self._compare_causal_graphs(fdg1, fdg2, mapping)

        # Compare intervention responses
        intervention_score = self.intervention_sim.compare_intervention_responses(
            fdg1, fdg2, mapping
        )

        # Compare mechanistic patterns
        pattern_score = self._compare_mechanistic_patterns(fdg1, fdg2, mapping)

        # Weighted combination
        return 0.3 * graph_score + 0.5 * intervention_score + 0.2 * pattern_score

    def _compare_causal_graphs(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compare causal graph structures under mapping
        """
        # Extract causal subgraphs
        causal1 = fdg1.get_causal_subgraph()
        causal2 = fdg2.get_causal_subgraph()

        # Count mapped causal edges
        edges_matched = 0
        edges_total = 0

        for u, v in causal1.edges():
            if u in mapping and v in mapping:
                edges_total += 1
                if (mapping[u], mapping[v]) in causal2.edges():
                    edges_matched += 1

        return edges_matched / edges_total if edges_total > 0 else 0.0

    def _compare_mechanistic_patterns(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compare mechanistic patterns (feedback loops, chains, etc.)
        """
        # Detect feedback loops
        loops1 = fdg1.get_feedback_loops()
        loops2 = fdg2.get_feedback_loops()

        # Match loops
        matched_loops = 0
        for loop1 in loops1:
            mapped_loop = [mapping.get(n) for n in loop1]
            if None not in mapped_loop:
                # Check if mapped loop exists
                for loop2 in loops2:
                    if set(mapped_loop) == set(loop2):
                        matched_loops += 1
                        break

        # Score: fraction of loops matched
        loop_score = matched_loops / max(len(loops1), 1)

        # Detect causal chains
        chains1 = self._detect_causal_chains(fdg1)
        chains2 = self._detect_causal_chains(fdg2)

        # Match chains
        matched_chains = 0
        for chain1 in chains1:
            mapped_chain = [mapping.get(n) for n in chain1]
            if None not in mapped_chain:
                for chain2 in chains2:
                    if mapped_chain == chain2:
                        matched_chains += 1
                        break

        chain_score = matched_chains / max(len(chains1), 1)

        # Combine scores
        return 0.5 * loop_score + 0.5 * chain_score

    def _detect_causal_chains(
        self,
        fdg: FunctionalDependencyGraph,
        min_length: int = 3
    ) -> List[List[str]]:
        """
        Detect causal chains (sequences of causal edges)
        """
        chains = []
        causal_graph = fdg.get_causal_subgraph()

        # Find all simple paths of length >= min_length
        for source in causal_graph.nodes():
            for target in causal_graph.nodes():
                if source == target:
                    continue

                try:
                    paths = list(nx.all_simple_paths(
                        causal_graph,
                        source,
                        target,
                        cutoff=min_length
                    ))

                    for path in paths:
                        if len(path) >= min_length:
                            chains.append(path)

                except nx.NetworkXNoPath:
                    continue

        return chains

    def compare_interventions(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compare intervention responses

        Wrapper for InterventionSimulator
        """
        return self.intervention_sim.compare_intervention_responses(
            fdg1, fdg2, mapping
        )

    def compute_interventional_equivalence(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compute interventional equivalence score

        Tests multiple random interventions
        """
        return self.intervention_sim.test_interventional_equivalence(
            fdg1, fdg2, mapping
        )
