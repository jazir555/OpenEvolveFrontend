"""
Ontology Mapper - I_mech Stage 2 Integration

Integration script for real-time ontology mapping in I_mech.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from .ontology_mapper import OntologyMapper, MappingResult

try:
    from ..imech.core.domain import Domain
    from ..imech.core.fdg import FunctionalDependencyGraph
except ImportError:
    # For standalone usage
    Domain = None
    FunctionalDependencyGraph = None


class I_mechOntologyIntegrator:
    """
    Integration layer for Ontology Mapper with I_mech Stage 2.

    Provides real-time semantic mapping for isomorphism detection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize I_mech ontology integrator

        Args:
            config: Configuration for ontology mapper
        """
        self.mapper = OntologyMapper(config)
        self.mapping_cache: Dict[Tuple[str, str], MappingResult] = {}

    def get_semantic_mapping(
        self,
        domain_a: Domain,
        domain_b: Domain,
        use_cache: bool = True
    ) -> MappingResult:
        """
        Get semantic mapping between two domains (with caching)

        Args:
            domain_a: First domain
            domain_b: Second domain
            use_cache: Use cached mappings if available

        Returns:
            MappingResult
        """
        cache_key = (domain_a.id, domain_b.id)

        # Check cache
        if use_cache and cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]

        # Compute mapping
        result = self.mapper.map_ontologies(
            domain_a,
            domain_b,
            use_stages=['lexical', 'semantic', 'graph', 'aggregate']
        )

        # Cache result
        if use_cache:
            self.mapping_cache[cache_key] = result

        return result

    def compute_similarity_score(
        self,
        domain_a: Domain,
        domain_b: Domain
    ) -> float:
        """
        Compute semantic similarity score for domain pair

        Used by I_mech to rank candidate domain pairs for solution transfer.

        Args:
            domain_a: First domain
            domain_b: Second domain

        Returns:
            Similarity score [0, 1]
        """
        mapping = self.get_semantic_mapping(domain_a, domain_b)

        if not mapping.confidence:
            return 0.0

        # Average confidence
        avg_confidence = sum(mapping.confidence.values()) / len(mapping.confidence)

        # Adjust by coverage
        graph_a = domain_a.fdg.to_networkx() if domain_a.fdg else nx.DiGraph()
        graph_b = domain_b.fdg.to_networkx() if domain_b.fdg else nx.DiGraph()

        max_nodes = max(len(graph_a.nodes()), len(graph_b.nodes()))
        coverage = len(mapping.concept_mapping) / max_nodes if max_nodes > 0 else 0.0

        # Combined score
        similarity = avg_confidence * coverage

        return similarity

    def find_best_transfer_candidates(
        self,
        source_domain: Domain,
        target_domains: List[Domain],
        top_k: int = 5
    ) -> List[Tuple[Domain, float, MappingResult]]:
        """
        Find best domains for solution transfer from source

        Args:
            source_domain: Source domain with solution
            target_domains: List of candidate target domains
            top_k: Number of top candidates to return

        Returns:
            List of (domain, similarity, mapping) tuples, sorted by similarity
        """
        candidates = []

        for target_domain in target_domains:
            # Skip same domain
            if target_domain.id == source_domain.id:
                continue

            # Compute similarity
            similarity = self.compute_similarity_score(source_domain, target_domain)

            # Get mapping
            mapping = self.get_semantic_mapping(source_domain, target_domain)

            candidates.append((target_domain, similarity, mapping))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return candidates[:top_k]

    def validate_isomorphic_mapping(
        self,
        domain_a: Domain,
        domain_b: Domain,
        structural_isomorphism: bool
    ) -> Tuple[bool, float, str]:
        """
        Validate isomorphic mapping combining semantic and structural evidence

        Args:
            domain_a: First domain
            domain_b: Second domain
            structural_isomorphism: Result from structural isomorphism test

        Returns:
            Tuple of (is_isomorphic, confidence, reason)
        """
        # Get semantic mapping
        mapping = self.get_semantic_mapping(domain_a, domain_b)

        # Compute semantic confidence
        if mapping.confidence:
            semantic_confidence = sum(mapping.confidence.values()) / len(mapping.confidence)
        else:
            semantic_confidence = 0.0

        # Combine with structural evidence
        if structural_isomorphism:
            # Structural isomorphism exists
            # High semantic confidence -> strong isomorphism
            # Low semantic confidence -> weak isomorphism (structural only)

            if semantic_confidence > 0.7:
                final_confidence = 0.9
                reason = "Strong structural and semantic isomorphism"
            elif semantic_confidence > 0.5:
                final_confidence = 0.7
                reason = "Moderate structural and semantic isomorphism"
            else:
                final_confidence = 0.5
                reason = "Structural isomorphism with weak semantic correspondence"

            return (True, final_confidence, reason)

        else:
            # No structural isomorphism
            # High semantic confidence -> partial isomorphism (candidates for transfer)
            # Low semantic confidence -> no isomorphism

            if semantic_confidence > 0.6:
                final_confidence = 0.4
                reason = "Partial isomorphism (strong semantic correspondence)"
            else:
                final_confidence = 0.1
                reason = "No isomorphism detected"

            return (False, final_confidence, reason)

    def suggest_transfer_strategy(
        self,
        source_domain: Domain,
        target_domain: Domain,
        has_solution: bool = True
    ) -> Dict[str, Any]:
        """
        Suggest solution transfer strategy based on mapping analysis

        Args:
            source_domain: Source domain (with solution)
            target_domain: Target domain (needs solution)
            has_solution: Whether source has a solution

        Returns:
            Dictionary with transfer strategy recommendations
        """
        # Get semantic mapping
        mapping = self.get_semantic_mapping(source_domain, target_domain)

        # Analyze mapping
        if not mapping.confidence:
            return {
                'recommendation': 'NO_TRANSFER',
                'confidence': 0.0,
                'reason': 'No semantic correspondence found'
            }

        avg_confidence = sum(mapping.confidence.values()) / len(mapping.confidence)
        coverage = len(mapping.concept_mapping) / max(
            len(source_domain.fdg.to_networkx().nodes()) if source_domain.fdg else 1,
            len(target_domain.fdg.to_networkx().nodes()) if target_domain.fdg else 1
        )

        # Generate recommendation
        if avg_confidence > 0.7 and coverage > 0.7:
            return {
                'recommendation': 'DIRECT_TRANSFER',
                'confidence': avg_confidence,
                'reason': 'Strong semantic correspondence with high coverage',
                'mapping': mapping.concept_mapping,
                'requires_adaptation': False
            }

        elif avg_confidence > 0.5 and coverage > 0.5:
            return {
                'recommendation': 'ADAPTIVE_TRANSFER',
                'confidence': avg_confidence,
                'reason': 'Moderate semantic correspondence, may require adaptation',
                'mapping': mapping.concept_mapping,
                'requires_adaptation': True,
                'adaptations_needed': self._identify_adaptations(mapping)
            }

        else:
            return {
                'recommendation': 'LIMITED_TRANSFER',
                'confidence': avg_confidence,
                'reason': 'Weak semantic correspondence, limited transfer potential',
                'mapping': mapping.concept_mapping,
                'requires_adaptation': True,
                'adaptations_needed': 'Full reconstruction required'
            }

    def _identify_adaptations(self, mapping: MappingResult) -> List[str]:
        """
        Identify required adaptations for transfer

        Args:
            mapping: Semantic mapping

        Returns:
            List of required adaptations
        """
        adaptations = []

        # Check for low-confidence mappings
        for source, score in mapping.confidence.items():
            if score < 0.5:
                adaptations.append(f"Verify mapping: {source} → {mapping.concept_mapping.get(source, 'unknown')}")

        # Check for unmapped concepts
        # (This requires access to domain concepts - simplified here)

        return adaptations

    def batch_similarity_matrix(
        self,
        domains: List[Domain]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute similarity matrix for all domain pairs

        Args:
            domains: List of domains

        Returns:
            Dictionary mapping (domain_id_a, domain_id_b) → similarity
        """
        similarity_matrix = {}

        for i, domain_a in enumerate(domains):
            for j, domain_b in enumerate(domains):
                if i < j:  # Avoid duplicates and self-comparison
                    similarity = self.compute_similarity_score(domain_a, domain_b)
                    similarity_matrix[(domain_a.id, domain_b.id)] = similarity

        return similarity_matrix


# Convenience functions

def create_imech_integrator(config: Optional[Dict[str, Any]] = None) -> I_mechOntologyIntegrator:
    """
    Create I_mech ontology integrator

    Args:
        config: Optional configuration

    Returns:
        I_mechOntologyIntegrator instance
    """
    return I_mechOntologyIntegrator(config)


def find_isomorphic_domains(
    source_domain: Domain,
    domain_database: List[Domain],
    threshold: float = 0.5,
    top_k: int = 5
) -> List[Tuple[Domain, float, MappingResult]]:
    """
    Find domains isomorphic to source domain

    Convenience function for I_mech Stage 2.

    Args:
        source_domain: Source domain
        domain_database: Database of domains to search
        threshold: Minimum similarity threshold
        top_k: Number of results to return

    Returns:
        List of (domain, similarity, mapping) tuples
    """
    integrator = create_imech_integrator()
    candidates = integrator.find_best_transfer_candidates(
        source_domain,
        domain_database,
        top_k=top_k * 2  # Get more, filter later
    )

    # Filter by threshold
    filtered = [(d, s, m) for d, s, m in candidates if s >= threshold]

    return filtered[:top_k]


if __name__ == "__main__":
    # Demo
    print("I_mech Ontology Integration")
    print("=" * 60)

    # Create test domains
    domain_a = Domain(
        id="fluid",
        name="Fluid Dynamics",
        description="Fluid flow in pipes"
    )

    domain_b = Domain(
        id="electricity",
        name="Electrical Circuits",
        description="Electrical circuits"
    )

    # Create simple FDGs
    fdg_a = nx.DiGraph()
    fdg_a.add_nodes_from(['flow', 'pressure', 'resistance'])
    fdg_a.add_edges_from([('pressure', 'flow'), ('resistance', 'flow')])

    fdg_b = nx.DiGraph()
    fdg_b.add_nodes_from(['current', 'voltage', 'resistance'])
    fdg_b.add_edges_from([('voltage', 'current'), ('resistance', 'current')])

    domain_a.fdg = type('FDG', (), {'to_networkx': lambda: fdg_a})()
    domain_b.fdg = type('FDG', (), {'to_networkx': lambda: fdg_b})()

    # Create integrator
    integrator = create_imech_integrator()

    # Test 1: Semantic mapping
    print("\n1. Semantic Mapping:")
    mapping = integrator.get_semantic_mapping(domain_a, domain_b)
    print(f"   Concepts mapped: {len(mapping.concept_mapping)}")
    for source, target in list(mapping.concept_mapping.items())[:3]:
        score = mapping.confidence.get(source, 0.0)
        print(f"   {source:15} → {target:15}: {score:.3f}")

    # Test 2: Similarity score
    print("\n2. Similarity Score:")
    similarity = integrator.compute_similarity_score(domain_a, domain_b)
    print(f"   Similarity: {similarity:.3f}")

    # Test 3: Transfer strategy
    print("\n3. Transfer Strategy:")
    strategy = integrator.suggest_transfer_strategy(domain_a, domain_b)
    print(f"   Recommendation: {strategy['recommendation']}")
    print(f"   Confidence: {strategy['confidence']:.3f}")
    print(f"   Reason: {strategy['reason']}")

    # Test 4: Isomorphism validation
    print("\n4. Isomorphism Validation:")
    is_isomorphic, confidence, reason = integrator.validate_isomorphic_mapping(
        domain_a, domain_b, structural_isomorphism=True
    )
    print(f"   Is Isomorphic: {is_isomorphic}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Reason: {reason}")

    print("\n" + "=" * 60)
    print("✅ I_mech Ontology Integration working!")
