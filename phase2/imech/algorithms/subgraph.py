"""
Subgraph Isomorphism Algorithm

Find partial matches when complete isomorphism doesn't exist.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Optional, Dict, Tuple
import networkx as nx
from ..core.fdg import FunctionalDependencyGraph


class SubgraphMatcher:
    """
    Subgraph isomorphism for partial mechanistic matches
    """

    def __init__(self):
        pass

    def find_best_match(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Find best subgraph isomorphism

        Args:
            fdg1: Pattern FDG (typically smaller)
            fdg2: Target FDG (typically larger)

        Returns:
            Tuple of (mapping, score) where score is in [0, 1]
        """
        # Ensure fdg1 is the smaller graph
        if len(fdg1) > len(fdg2):
            # Swap
            fdg1, fdg2 = fdg2, fdg1
            swapped = True
        else:
            swapped = False

        try:
            matcher = nx.isomorphism.DiGraphMatcher(
                fdg2.graph,
                fdg1.graph,
                node_match=self._node_match,
                edge_match=self._edge_match
            )

            best_match = None
            best_size = 0

            # Find all subgraph isomorphisms
            for match in matcher.subgraph_isomorphisms_iter():
                if len(match) > best_size:
                    best_match = match
                    best_size = len(match)

            if best_match:
                # Score = fraction of nodes matched
                score = best_size / max(len(fdg1), 1)

                if swapped:
                    # Reverse mapping to get fdg1 -> fdg2
                    best_match = {v: k for k, v in best_match.items()}

                return best_match, score

        except Exception as e:
            print(f"Subgraph isomorphism error: {e}")

        return None, 0.0

    def _node_match(self, n1_attrs: Dict, n2_attrs: Dict) -> bool:
        """Node matching criterion"""
        type1 = n1_attrs.get('constraint_type')
        type2 = n2_attrs.get('constraint_type')
        return type1 == type2

    def _edge_match(self, e1_attrs: Dict, e2_attrs: Dict) -> bool:
        """Edge matching criterion"""
        type1 = e1_attrs.get('type')
        type2 = e2_attrs.get('type')
        return type1 == type2

    def find_maximum_common_subgraph(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> Tuple[Dict[str, str], float]:
        """
        Find maximum common induced subgraph

        Returns:
            Tuple of (mapping, similarity_score)
        """
        # Use NetworkX's isomorphism for MCS
        try:
            # Try matching fdg1 as subgraph of fdg2
            matcher1 = nx.isomorphism.GraphMatcher(
                fdg2.graph,
                fdg1.graph,
                node_match=self._node_match,
                edge_match=self._edge_match
            )

            best_match1 = None
            best_size1 = 0

            for match in matcher1.subgraph_isomorphisms_iter():
                if len(match) > best_size1:
                    best_match1 = match
                    best_size1 = len(match)

            # Try matching fdg2 as subgraph of fdg1
            matcher2 = nx.isomorphism.DiGraphMatcher(
                fdg1.graph,
                fdg2.graph,
                node_match=self._node_match,
                edge_match=self._edge_match
            )

            best_match2 = None
            best_size2 = 0

            for match in matcher2.subgraph_isomorphisms_iter():
                if len(match) > best_size2:
                    best_match2 = match
                    best_size2 = len(match)

            # Return best match
            if best_size1 >= best_size2:
                if best_match1:
                    # Reverse mapping
                    mapping = {v: k for k, v in best_match1.items()}
                    score = best_size1 / max(len(fdg1), len(fdg2))
                    return mapping, score

            if best_match2:
                mapping = dict(best_match2)
                score = best_size2 / max(len(fdg1), len(fdg2))
                return mapping, score

        except Exception as e:
            print(f"MCS error: {e}")

        return {}, 0.0
