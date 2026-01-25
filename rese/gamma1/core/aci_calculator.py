"""
Γ₁ ACI Calculator

Main entry point for Algorithmic Complexity Index calculation.

ACI = α·(1-H) + β·C + γ·S

Where:
- H = Disorder Entropy ∈ [0, 1] (higher = more disordered)
- C = Causal Coherence ∈ [0, 1] (higher = more coherent)
- S = Solvability Index ∈ [0, 1] (higher = more solvable)
- α, β, γ = Learned weights (α + β + γ = 1)

Higher ACI = more solvable = easier to solve
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time
from gamma1.core.csp_models import CSPInstance
from gamma1.core.entropy_engine import DisorderEntropy
from gamma1.core.coherence_engine import CausalCoherence
from gamma1.core.solvability_engine import SolvabilityIndex


@dataclass
class ACIResult:
    """
    Complete ACI Calculation Result

    Attributes:
        ACI: Final ACI score ∈ [0, 1]
        components: Component breakdown (H, C, S)
        confidence: Confidence in ACI score ∈ [0, 1]
        interpretation: Human-readable interpretation
        recommendation: Search strategy recommendation
        computation_time: Time taken for calculation (seconds)
        cached: Whether result was retrieved from cache
    """
    ACI: float
    components: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    interpretation: Dict = field(default_factory=dict)
    recommendation: Dict = field(default_factory=dict)
    computation_time: float = 0.0
    cached: bool = False

    def __str__(self) -> str:
        return (f"ACI={self.ACI:.3f} "
                f"(confidence={self.confidence:.2f}, "
                f"H={self.components.get('disorder_entropy', 0):.3f}, "
                f"C={self.components.get('causal_coherence', 0):.3f}, "
                f"S={self.components.get('solvability_index', 0):.3f})")


class ACICalculator:
    """
    Calculate Algorithmic Complexity Index (ACI) for CSP instances

    ACI = α·(1-H) + β·C + γ·S

    Where:
    - H = Disorder Entropy (lower = better)
    - C = Causal Coherence (higher = better)
    - S = Solvability Index (higher = better)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 0.45,
        gamma: float = 0.30,
        use_cache: bool = True
    ):
        """
        Initialize ACI calculator

        Args:
            alpha: Weight for (1-H) component (reduced)
            beta: Weight for C component (increased - coherence is key)
            gamma: Weight for S component
            use_cache: Whether to use caching
        """
        # Validate weights
        if not (abs(alpha + beta + gamma - 1.0) < 1e-6):
            raise ValueError("Weights must sum to 1.0")
        if not (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1):
            raise ValueError("Weights must be in [0, 1]")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_cache = use_cache

        # Initialize engines
        self.entropy_engine = DisorderEntropy()
        self.coherence_engine = CausalCoherence()
        self.solvability_engine = SolvabilityIndex()

        # Cache
        self._cache = {}

    def calculate(self, csp: CSPInstance) -> ACIResult:
        """
        Calculate ACI for CSP instance

        Args:
            csp: CSP instance

        Returns:
            ACIResult with ACI score and detailed breakdown
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(csp)
        if self.use_cache and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            cached_result.cached = True
            return cached_result

        # Calculate components
        entropy_components = self.entropy_engine.calculate(csp)
        coherence_components = self.coherence_engine.calculate(csp)
        solvability_components = self.solvability_engine.calculate(csp)

        # Extract component totals
        H = entropy_components.total()
        C = coherence_components.total()
        S = solvability_components.total()

        # Calculate ACI using formula
        # Note: (1-H) because low entropy = ordered = solvable
        ACI = (self.alpha * (1.0 - H) +
               self.beta * C +
               self.gamma * S)

        # Ensure bounds
        ACI = max(0.0, min(1.0, ACI))

        # Calculate confidence
        confidence = self._calculate_confidence(H, C, S, csp)

        # Generate interpretation and recommendation
        interpretation = self._interpret_aci(ACI)
        recommendation = self._generate_recommendation(ACI, H, C, S)

        # Build result
        result = ACIResult(
            ACI=ACI,
            components={
                'disorder_entropy': H,
                'causal_coherence': C,
                'solvability_index': S,
                # Also include sub-components for analysis
                'entropy_local': entropy_components.local,
                'entropy_constraint': entropy_components.constraint,
                'entropy_structural': entropy_components.structural,
                'coherence_graph': coherence_components.graph,
                'coherence_flow': coherence_components.flow,
                'coherence_stability': coherence_components.stability,
                'solvability_phase': solvability_components.phase_distance,
                'solvability_propagation': solvability_components.propagation,
                'solvability_structure': solvability_components.structure,
                'solvability_heuristic': solvability_components.heuristic,
            },
            confidence=confidence,
            interpretation=interpretation,
            recommendation=recommendation,
            computation_time=time.time() - start_time,
            cached=False
        )

        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result

        return result

    def _get_cache_key(self, csp: CSPInstance) -> str:
        """Generate cache key for CSP instance"""
        # Simple hash based on structure
        var_info = tuple((v.name, len(v.domain)) for v in csp.variables)
        constraint_info = tuple(
            (tuple(c.variables), len(c.allowed_tuples))
            for c in csp.constraints
        )
        return hash((var_info, constraint_info))

    def _calculate_confidence(
        self,
        H: float,
        C: float,
        S: float,
        csp: CSPInstance
    ) -> float:
        """
        Calculate confidence in ACI score

        Factors:
        1. Component agreement (do all components agree?)
        2. Problem size (larger = more reliable)
        3. Constraint density (more constraints = more structure)
        4. Domain size consistency

        Args:
            H: Disorder entropy
            C: Causal coherence
            S: Solvability index
            csp: CSP instance

        Returns:
            Confidence ∈ [0, 1]
        """
        # Factor 1: Component agreement
        components = [1-H, C, S]  # All oriented: higher = more solvable
        if len(components) > 1:
            component_variance = float(max(components) - min(components))
            agreement = 1.0 - component_variance
        else:
            agreement = 1.0

        # Factor 2: Problem size
        n = csp.num_variables()
        size_factor = min(1.0, n / 100)  # Normalize: 100 variables = max

        # Factor 3: Constraint density
        m = csp.num_constraints()
        if n > 1:
            density = m / (n * (n - 1) / 2)
            density_factor = min(1.0, density * 10)
        else:
            density_factor = 0.0

        # Factor 4: Domain size consistency
        domain_sizes = [v.domain_size() for v in csp.variables]
        if domain_sizes:
            domain_mean = sum(domain_sizes) / len(domain_sizes)
            if domain_mean > 0:
                domain_cv = (max(domain_sizes) - min(domain_sizes)) / domain_mean
                domain_consistency = 1.0 / (1.0 + domain_cv)
            else:
                domain_consistency = 0.0
        else:
            domain_consistency = 0.0

        # Combine factors
        confidence = (0.3 * agreement +
                     0.3 * size_factor +
                     0.2 * density_factor +
                     0.2 * domain_consistency)

        return max(0.0, min(1.0, confidence))

    def _interpret_aci(self, aci: float) -> Dict:
        """
        Generate human-readable interpretation of ACI score

        Args:
            aci: ACI score ∈ [0, 1]

        Returns:
            Interpretation dictionary
        """
        if aci >= 0.8:
            return {
                'category': 'HIGHLY_TRACTABLE',
                'description': 'Problem has high regularity and strong causal structure. Expected to be easily solvable with standard methods.',
                'estimated_difficulty': 'Easy',
                'success_probability': '>0.95',
                'color': 'green'
            }
        elif aci >= 0.6:
            return {
                'category': 'TRACTABLE',
                'description': 'Problem shows good structure and moderate regularity. Should be solvable with appropriate heuristics.',
                'estimated_difficulty': 'Medium',
                'success_probability': '0.7-0.95',
                'color': 'lightgreen'
            }
        elif aci >= 0.4:
            return {
                'category': 'CHALLENGING',
                'description': 'Problem has mixed characteristics. May require sophisticated search strategies and significant computational resources.',
                'estimated_difficulty': 'Hard',
                'success_probability': '0.3-0.7',
                'color': 'yellow'
            }
        elif aci >= 0.2:
            return {
                'category': 'HIGHLY_INTRACTABLE',
                'description': 'Problem exhibits high disorder and weak causal structure. Likely requires exponential time or may be unsolvable.',
                'estimated_difficulty': 'Very Hard',
                'success_probability': '0.05-0.3',
                'color': 'orange'
            }
        else:
            return {
                'category': 'PROVABLY_INTRACTABLE',
                'description': 'Problem shows maximum disorder and no coherent structure. High probability of being unsolvable or requiring exponential resources.',
                'estimated_difficulty': 'Extreme',
                'success_probability': '<0.05',
                'color': 'red'
            }

    def _generate_recommendation(
        self,
        aci: float,
        H: float,
        C: float,
        S: float
    ) -> Dict:
        """
        Generate search strategy recommendation

        Args:
            aci: ACI score
            H: Disorder entropy
            C: Causal coherence
            S: Solvability index

        Returns:
            Recommendation dictionary
        """
        if aci > 0.8:
            return {
                'solver': 'BACKTRACKING_WITH_FORWARD_CHECKING',
                'heuristic': 'MRV_LCV',
                'propagation': 'AC-3',
                'reasoning': 'Highly tractable. Simple backtracking sufficient.',
                'expected_time': 'Fast (<1s for n<100)',
                'priority': 'LOW'
            }
        elif aci > 0.6:
            return {
                'solver': 'CONSTRAINT_PROPAGATION',
                'heuristic': 'DOM_WDEG',
                'propagation': 'AC-4 or PC-5',
                'reasoning': 'Tractable. Use stronger propagation.',
                'expected_time': 'Moderate (<10s for n<100)',
                'priority': 'MEDIUM'
            }
        elif aci > 0.4:
            return {
                'solver': 'MONTE_CARLO_TREE_SEARCH',
                'heuristic': 'ADAPTIVE_MCTS',
                'propagation': 'DYNAMIC',
                'reasoning': 'Challenging. MCTS with adaptive exploration.',
                'expected_time': 'Slow (minutes to hours)',
                'priority': 'HIGH'
            }
        else:
            return {
                'solver': 'SPECIALIZED_OR_APPROXIMATION',
                'heuristic': 'NONE',
                'propagation': 'NONE',
                'reasoning': 'Highly intractable. Consider reformulation or approximation.',
                'expected_time': 'Very slow or impossible',
                'priority': 'CRITICAL'
            }

    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size': len(self._cache),
            'cache_enabled': self.use_cache
        }


if __name__ == "__main__":
    print("=" * 70)
    print("ACI Calculator - Demonstration")
    print("=" * 70)

    from gamma1.core.csp_models import create_test_csp, create_tree_csp, create_dense_csp

    # Create calculator
    calculator = ACICalculator(alpha=0.35, beta=0.35, gamma=0.30)

    # Test on different CSP types
    print("\n" + "=" * 70)
    print("Test 1: Random CSP")
    print("=" * 70)

    test_csp = create_test_csp(n_variables=15, domain_size=5, n_constraints=10)
    print(f"\nCSP: {test_csp}")

    test_result = calculator.calculate(test_csp)
    print(f"\n{test_result}")
    print(f"\nInterpretation:")
    print(f"  Category: {test_result.interpretation['category']}")
    print(f"  Description: {test_result.interpretation['description']}")
    print(f"  Difficulty: {test_result.interpretation['estimated_difficulty']}")
    print(f"\nRecommendation:")
    print(f"  Solver: {test_result.recommendation['solver']}")
    print(f"  Reasoning: {test_result.recommendation['reasoning']}")

    print("\n" + "=" * 70)
    print("Test 2: Tree-Structured CSP (should have high ACI)")
    print("=" * 70)

    tree_csp = create_tree_csp(n_variables=15, domain_size=5)
    print(f"\nCSP: {tree_csp}")

    tree_result = calculator.calculate(tree_csp)
    print(f"\n{tree_result}")
    print(f"\nCategory: {tree_result.interpretation['category']}")

    print("\n" + "=" * 70)
    print("Test 3: Dense CSP (should have low ACI)")
    print("=" * 70)

    dense_csp = create_dense_csp(n_variables=15, domain_size=5, constraint_density=0.8)
    print(f"\nCSP: {dense_csp}")

    dense_result = calculator.calculate(dense_csp)
    print(f"\n{dense_result}")
    print(f"\nCategory: {dense_result.interpretation['category']}")

    # Comparison
    print("\n" + "=" * 70)
    print("ACI Comparison")
    print("=" * 70)
    print(f"Tree CSP ACI:      {tree_result.ACI:.3f} (High)")
    print(f"Test CSP ACI:      {test_result.ACI:.3f} (Medium)")
    print(f"Dense CSP ACI:     {dense_result.ACI:.3f} (Low)")

    # Verify ordering
    print(f"\nOrdering correct: {tree_result.ACI > test_result.ACI > dense_result.ACI}")

    # Computation time
    print(f"\nComputation times:")
    print(f"  Tree:   {tree_result.computation_time*1000:.2f}ms")
    print(f"  Test:   {test_result.computation_time*1000:.2f}ms")
    print(f"  Dense:  {dense_result.computation_time*1000:.2f}ms")

    # Cache stats
    print(f"\nCache stats: {calculator.get_cache_stats()}")

    print("\n" + "=" * 70)
    print("[OK] ACI calculator demonstration complete")
    print("=" * 70)
