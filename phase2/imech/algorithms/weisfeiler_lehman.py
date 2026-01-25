"""
Weisfeiler-Lehman Graph Isomorphism Algorithm

1-WL color refinement for fast graph similarity detection.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, Tuple
from collections import Counter
import networkx as nx
from ..core.fdg import FunctionalDependencyGraph


class WeisfeilerLehman:
    """
    Weisfeiler-Lehman 1-WL color refinement algorithm

    Fast approximate graph isomorphism detection
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    def compute_similarity(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> float:
        """
        Compute structural similarity using WL color refinement

        Args:
            fdg1: First FDG
            fdg2: Second FDG

        Returns:
            Similarity score in [0, 1]
        """
        # Quick size check
        if len(fdg1) == 0 or len(fdg2) == 0:
            return 0.0

        # Initialize colors
        colors1 = self._init_colors(fdg1)
        colors2 = self._init_colors(fdg2)

        # Refine colors
        for iteration in range(self.max_iterations):
            new_colors1 = self._refine_colors(fdg1, colors1)
            new_colors2 = self._refine_colors(fdg2, colors2)

            # Check convergence
            if new_colors1 == colors1 and new_colors2 == colors2:
                break

            colors1, colors2 = new_colors1, new_colors2

        # Compute similarity
        similarity = self._compare_color_distributions(colors1, colors2)
        return similarity

    def _init_colors(self, fdg: FunctionalDependencyGraph) -> Dict[str, int]:
        """
        Initialize colors based on degree and constraint type
        """
        colors = {}
        for node_id in fdg.nodes:
            degree = fdg.graph.degree(node_id)
            label = fdg.nodes[node_id].constraint_type
            # Hash degree + label for initial color
            colors[node_id] = hash((degree, label))
        return colors

    def _refine_colors(
        self,
        fdg: FunctionalDependencyGraph,
        colors: Dict[str, int]
    ) -> Dict[str, int]:
        """
        One iteration of color refinement
        """
        new_colors = {}
        for node_id in fdg.nodes:
            # Get sorted neighbor colors as multiset
            neighbor_colors = sorted([
                colors[n] for n in fdg.graph.neighbors(node_id)
            ])

            # New color = hash(old color, neighbor multiset)
            new_colors[node_id] = hash((
                colors[node_id],
                tuple(neighbor_colors)
            ))

        return new_colors

    def _compare_color_distributions(
        self,
        colors1: Dict[str, int],
        colors2: Dict[str, int]
    ) -> float:
        """
        Compare color distributions using Jaccard similarity
        """
        freq1 = Counter(colors1.values())
        freq2 = Counter(colors2.values())

        # Jaccard similarity
        intersection = sum((freq1 & freq2).values())
        union = sum((freq1 | freq2).values())

        return intersection / union if union > 0 else 0.0

    def are_isomorphic(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> bool:
        """
        Quick check if graphs might be isomorphic

        Returns False if definitely not isomorphic, True if possibly isomorphic
        """
        # Size check
        if len(fdg1) != len(fdg2):
            return False

        # Degree sequence check
        degrees1 = sorted([int(fdg1.graph.degree(n)) for n in fdg1.nodes])
        degrees2 = sorted([int(fdg2.graph.degree(n)) for n in fdg2.nodes])

        if degrees1 != degrees2:
            return False

        # WL similarity check
        similarity = self.compute_similarity(fdg1, fdg2)
        return similarity > 0.95
