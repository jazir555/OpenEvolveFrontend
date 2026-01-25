"""
VF2 Exact Graph Isomorphism Algorithm

Uses NetworkX's VF2 implementation for exact matching.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Optional, Dict
import networkx as nx
from ..core.fdg import FunctionalDependencyGraph


class VF2Matcher:
    """
    VF2 algorithm for exact graph isomorphism

    Uses NetworkX's optimized implementation
    """

    def __init__(self):
        pass

    def find_isomorphism(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> Optional[Dict[str, str]]:
        """
        Find exact isomorphism between two FDGs

        Args:
            fdg1: First FDG (source)
            fdg2: Second FDG (target)

        Returns:
            Node mapping if isomorphic, None otherwise
        """
        # Size check
        if len(fdg1) != len(fdg2):
            return None

        # Degree sequence check
        degrees1 = sorted([fdg1.graph.degree(n) for n in fdg1.nodes])
        degrees2 = sorted([fdg2.graph.degree(n) for n in fdg2.nodes])

        if degrees1 != degrees2:
            return None

        try:
            # Create matcher (use DiGraphMatcher for directed graphs)
            matcher = nx.isomorphism.DiGraphMatcher(
                fdg1.graph,
                fdg2.graph,
                node_match=self._node_match,
                edge_match=self._edge_match
            )

            # Check isomorphism
            if matcher.is_isomorphic():
                return dict(matcher.mapping)

        except Exception as e:
            print(f"VF2 matching error: {e}")

        return None

    def _node_match(self, n1_attrs: Dict, n2_attrs: Dict) -> bool:
        """
        Node matching criterion

        Nodes must have same constraint type
        """
        # Get constraint types
        type1 = n1_attrs.get('constraint_type')
        type2 = n2_attrs.get('constraint_type')

        return type1 == type2

    def _edge_match(self, e1_attrs: Dict, e2_attrs: Dict) -> bool:
        """
        Edge matching criterion

        Edges must have same type
        """
        # Get edge types
        type1 = e1_attrs.get('type')
        type2 = e2_attrs.get('type')

        return type1 == type2

    def find_all_isomorphisms(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> list:
        """
        Find all isomorphisms between two FDGs

        Returns:
            List of node mappings
        """
        isomorphisms = []

        try:
            matcher = nx.isomorphism.DiGraphMatcher(
                fdg1.graph,
                fdg2.graph,
                node_match=self._node_match,
                edge_match=self._edge_match
            )

            for mapping in matcher.isomorphisms_iter():
                isomorphisms.append(dict(mapping))

        except Exception as e:
            print(f"VF2 multiple matching error: {e}")

        return isomorphisms
