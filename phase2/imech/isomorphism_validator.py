"""
I_mech: Mechanistic Isomorphism Validator

Main interface for mechanistic isomorphism detection and solution transfer.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
Status: Production Implementation
"""

from typing import Optional, Dict
import time
from .core import (
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType,
    Domain,
    SimilarityResult
)
from .core.fdg_extractor import FDGExtractor
from .algorithms import (
    WeisfeilerLehman,
    VF2Matcher,
    SubgraphMatcher,
    InterventionSimulator
)
from .core.causality import CausalSimilarityAnalyzer
from .core.scoring import SimilarityScorer
from .lean4.proof_generator import ProofGenerator
from .transfer import SolutionMapper, SolutionValidator


class IMechValidator:
    """
    Mechanistic Isomorphism Validator

    Main interface for detecting mechanistic isomorphisms and transferring solutions.

    Usage:
        validator = IMechValidator()
        result = validator.compare(domain1, domain2)
        if result.total_score > 0.7:
            transferred = result.transferred_solution
    """

    def __init__(
        self,
        use_exact_isomorphism: bool = False,
        enable_proofs: bool = False,
        cache_enabled: bool = True
    ):
        """
        Initialize I_mech validator

        Args:
            use_exact_isomorphism: Use VF2 for exact matching (slower)
            enable_proofs: Generate Lean 4 proofs (requires Lean 4)
            cache_enabled: Enable similarity caching
        """
        # Core components
        self.fdg_extractor = FDGExtractor(use_causal_discovery=True)
        self.wl = WeisfeilerLehman(max_iterations=10)
        self.vf2 = VF2Matcher()
        self.subgraph = SubgraphMatcher()
        self.causal_analyzer = CausalSimilarityAnalyzer()
        self.scorer = SimilarityScorer()
        self.mapper = SolutionMapper()
        self.validator = SolutionValidator()

        # Proof generation (optional)
        self.proof_generator = ProofGenerator() if enable_proofs else None
        self.enable_proofs = enable_proofs

        # Caching
        self.cache_enabled = cache_enabled
        self._cache = {}

    def compare(
        self,
        domain1: Domain,
        domain2: Domain
    ) -> SimilarityResult:
        """
        Compare two domains for mechanistic isomorphism

        Args:
            domain1: Source domain (typically with solution)
            domain2: Target domain

        Returns:
            SimilarityResult with scores, mapping, and transferred solution
        """
        start_time = time.time()

        # Check cache
        cache_key = (domain1.id, domain2.id)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Stage 1: Extract FDGs (if not already done)
        if domain1.fdg is None:
            domain1.fdg = self.fdg_extractor.extract(domain1)
        if domain2.fdg is None:
            domain2.fdg = self.fdg_extractor.extract(domain2)

        # Stage 2: Structural similarity (WL)
        struct_score = self.wl.compute_similarity(domain1.fdg, domain2.fdg)

        # Early termination if clearly not isomorphic
        if struct_score < 0.3:
            return SimilarityResult(
                total_score=0.0,
                structural_score=struct_score,
                causal_score=0.0,
                semantic_score=0.0,
                intervention_score=0.0,
                node_mapping={},
                computation_time=time.time() - start_time
            )

        # Stage 3: Find mapping (VF2 or subgraph)
        mapping = None
        if len(domain1.fdg) == len(domain2.fdg):
            # Try exact isomorphism
            mapping = self.vf2.find_isomorphism(domain1.fdg, domain2.fdg)

        if not mapping:
            # Try subgraph isomorphism
            mapping, score = self.subgraph.find_best_match(domain1.fdg, domain2.fdg)

        if not mapping:
            # Generate heuristic mapping
            mapping = self._generate_mapping(domain1.fdg, domain2.fdg)

        # Stage 4: Mechanistic similarity
        causal_score = self.causal_analyzer.analyze(
            domain1.fdg,
            domain2.fdg,
            mapping
        )

        # Stage 5: Semantic similarity
        semantic_score = self.scorer.compute_semantic_similarity(
            domain1.fdg,
            domain2.fdg,
            mapping
        )

        # Stage 6: Intervention similarity
        intervention_score = self.causal_analyzer.compare_interventions(
            domain1.fdg,
            domain2.fdg,
            mapping
        )

        # Stage 7: Total score
        total_score = self.scorer.compute_total_score(
            struct_score,
            causal_score,
            semantic_score,
            intervention_score
        )

        # Stage 8: Generate proof (if enabled and score high)
        proof = None
        proof_verified = False
        if self.proof_generator and total_score > 0.7:
            proof = self.proof_generator.generate(
                domain1.fdg,
                domain2.fdg,
                mapping
            )
            if proof:
                proof_verified = self.proof_generator.verify(proof)

        # Stage 9: Transfer solution (if available)
        transferred_solution = None
        validation_result = None

        if domain1.has_solution() and mapping:
            solution = domain1.get_primary_solution()
            transferred_solution = self.mapper.transfer(
                solution,
                mapping,
                domain1,
                domain2
            )

            # Validate transferred solution
            if transferred_solution:
                validation_result = self.validator.validate(
                    transferred_solution,
                    domain2
                )

        # Create result
        result = SimilarityResult(
            total_score=total_score,
            structural_score=struct_score,
            causal_score=causal_score,
            semantic_score=semantic_score,
            intervention_score=intervention_score,
            node_mapping=mapping,
            proof=proof,
            proof_verified=proof_verified,
            transferred_solution=transferred_solution,
            validation_result=validation_result,
            computation_time=time.time() - start_time
        )

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = result

        return result

    def _generate_mapping(
        self,
        fdg1: FunctionalDependencyGraph,
        fdg2: FunctionalDependencyGraph
    ) -> Dict[str, str]:
        """
        Generate best-effort mapping using heuristics
        """
        mapping = {}

        # Map nodes with same constraint types
        for node1_id, node1 in fdg1.nodes.items():
            candidates = [
                node2_id for node2_id, node2 in fdg2.nodes.items()
                if node1.constraint_type == node2.constraint_type
            ]

            if candidates:
                # Pick best candidate based on degree
                degree1 = fdg1.graph.degree(node1_id)
                candidates.sort(key=lambda n: abs(fdg2.graph.degree(n) - degree1))

                # Select best unmatched candidate
                for candidate in candidates:
                    if candidate not in mapping.values():
                        mapping[node1_id] = candidate
                        break

        return mapping

    def find_analogous_domains(
        self,
        target_domain: Domain,
        candidate_domains: list,
        threshold: float = 0.7
    ) -> list:
        """
        Find domains analogous to target from a list of candidates

        Args:
            target_domain: Domain to find analogies for
            candidate_domains: List of candidate domains (with solutions)
            threshold: Minimum similarity score

        Returns:
            List of (domain, similarity_result) tuples sorted by score
        """
        results = []

        for candidate in candidate_domains:
            result = self.compare(candidate, target_domain)

            if result.total_score >= threshold:
                results.append((candidate, result))

        # Sort by score
        results.sort(key=lambda x: x[1].total_score, reverse=True)

        return results

    def validate_transfer_success(
        self,
        result: SimilarityResult
    ) -> bool:
        """
        Check if solution transfer was successful

        Args:
            result: Similarity result from compare()

        Returns:
            True if transfer successful
        """
        if result.validation_result is None:
            return False

        return result.validation_result.get('is_valid', False)


# Convenience function
def compare_domains(domain1: Domain, domain2: Domain) -> SimilarityResult:
    """
    Convenience function to compare two domains

    Usage:
        result = compare_domains(domain1, domain2)
        if result.total_score > 0.7:
            print(f"Isomorphic! Score: {result.total_score}")
    """
    validator = IMechValidator()
    return validator.compare(domain1, domain2)
