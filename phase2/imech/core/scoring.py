"""
Similarity Scoring System

Multi-factor similarity scoring for I_mech.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, Tuple
from .fdg import FunctionalDependencyGraph


class SimilarityScorer:
    """
    Compute multi-factor similarity scores
    """

    def __init__(
        self,
        weight_structural: float = 0.3,
        weight_causal: float = 0.3,
        weight_semantic: float = 0.2,
        weight_intervention: float = 0.2
    ):
        """
        Initialize scorer with weights

        Args:
            weight_structural: Weight for structural similarity
            weight_causal: Weight for causal similarity
            weight_semantic: Weight for semantic similarity
            weight_intervention: Weight for interventional similarity
        """
        # Normalize weights
        total = weight_structural + weight_causal + weight_semantic + weight_intervention
        self.weight_structural = weight_structural / total
        self.weight_causal = weight_causal / total
        self.weight_semantic = weight_semantic / total
        self.weight_intervention = weight_intervention / total

    def compute_total_score(
        self,
        structural_score: float,
        causal_score: float,
        semantic_score: float,
        intervention_score: float
    ) -> float:
        """
        Compute weighted total similarity score

        Args:
            structural_score: Graph isomorphism score
            causal_score: Causal mechanism score
            semantic_score: Semantic label score
            intervention_score: Interventional equivalence score

        Returns:
            Total score in [0, 1]
        """
        total = (
            self.weight_structural * structural_score +
            self.weight_causal * causal_score +
            self.weight_semantic * semantic_score +
            self.weight_intervention * intervention_score
        )

        return max(0.0, min(1.0, total))

    def compute_semantic_similarity(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compute semantic similarity of labels

        Args:
            fdg1: First FDG
            fdg2: Second FDG
            mapping: Node mapping

        Returns:
            Semantic similarity in [0, 1]
        """
        if not mapping:
            return 0.0

        # Node label similarity
        node_sim = 0.0
        for node1, node2 in mapping.items():
            if node1 in fdg1.nodes and node2 in fdg2.nodes:
                label1 = fdg1.nodes[node1].constraint_type
                label2 = fdg2.nodes[node2].constraint_type

                # Exact match
                if label1 == label2:
                    sim = 1.0
                else:
                    # Hierarchical or embedding similarity (simplified)
                    sim = self._label_similarity(label1, label2)

                node_sim += sim

        node_sim /= len(mapping)

        # Edge type similarity
        edge_sim = self._compute_edge_similarity(fdg1, fdg2, mapping)

        # Combine
        return 0.6 * node_sim + 0.4 * edge_sim

    def _label_similarity(self, label1: str, label2: str) -> float:
        """
        Compute similarity between two labels

        Simplified version - can be enhanced with embeddings
        """
        # Hierarchical categories (simplified)
        numeric_types = {'integer', 'float', 'continuous', 'discrete'}
        string_types = {'string', 'text', 'categorical'}
        logical_types = {'boolean', 'binary', 'logical'}

        # Same category
        if label1 in numeric_types and label2 in numeric_types:
            return 0.7
        if label1 in string_types and label2 in string_types:
            return 0.7
        if label1 in logical_types and label2 in logical_types:
            return 0.7

        # No similarity
        return 0.0

    def _compute_edge_similarity(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph,
        mapping: Dict[str, str]
    ) -> float:
        """
        Compute edge type similarity under mapping
        """
        if not mapping:
            return 0.0

        edges_matched = 0
        edges_total = 0

        for (u1, v1), edge1 in fdg1.edges.items():
            if u1 in mapping and v1 in mapping:
                edges_total += 1
                u2, v2 = mapping[u1], mapping[v1]

                edge2 = fdg2.get_edge(u2, v2)
                if edge2 and edge1.edge_type == edge2.edge_type:
                    edges_matched += 1

        return edges_matched / edges_total if edges_total > 0 else 0.0

    def compute_confidence(
        self,
        total_score: float,
        structural_score: float,
        causal_score: float
    ) -> float:
        """
        Compute confidence in similarity score

        Higher confidence when structural and causal scores agree
        """
        # Variance as inverse confidence
        scores = [structural_score, causal_score, total_score]
        variance = sum((s - total_score)**2 for s in scores) / len(scores)

        # Lower variance = higher confidence
        confidence = 1.0 - min(variance, 1.0)

        return confidence
