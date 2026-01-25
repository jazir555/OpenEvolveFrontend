"""
Γ₁ Disorder Entropy Engine (H)

Implements multi-scale entropy measurement for CSP instances.
Higher entropy = more disorder = less solvable.

Components:
- Local Entropy: Per-variable domain entropy
- Constraint Entropy: Information reduction from constraints
- Structural Entropy: Graph topology randomness
- Kolmogorov Approximation: Compression-based complexity
"""

from dataclasses import dataclass
from typing import List, Dict
import math
import networkx as nx
import numpy as np
from gamma1.core.csp_models import CSPInstance


@dataclass
class EntropyComponents:
    """
    Disorder Entropy (H) Components

    Attributes:
        local: Local domain entropy (per-variable)
        constraint: Constraint entropy (information reduction)
        structural: Structural entropy (graph topology)
        kolmogorov: Kolmogorov complexity approximation
    """
    local: float = 0.0
    constraint: float = 0.0
    structural: float = 0.0
    kolmogorov: float = 0.0

    def total(self, weights: tuple = (0.3, 0.4, 0.2, 0.1)) -> float:
        """
        Calculate total H with given weights

        Args:
            weights: (w_local, w_constraint, w_structural, w_kolmogorov)

        Returns:
            Total entropy H ∈ [0, 1]
        """
        H = (weights[0] * self.local +
             weights[1] * self.constraint +
             weights[2] * self.structural +
             weights[3] * self.kolmogorov)
        return max(0.0, min(1.0, H))


class DisorderEntropy:
    """
    Calculate disorder entropy for CSP instances

    H = w_local * H_local + w_constraint * H_constraint +
        w_structural * H_structural + w_kolmogorov * H_kolmogorov

    Higher H = more disordered = less solvable
    """

    def __init__(self, weights: tuple = (0.3, 0.4, 0.2, 0.1)):
        """
        Initialize entropy calculator

        Args:
            weights: (w_local, w_constraint, w_structural, w_kolmogorov)
        """
        if len(weights) != 4 or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.weights = weights

    def calculate(self, csp: CSPInstance) -> EntropyComponents:
        """
        Calculate disorder entropy for CSP

        Args:
            csp: CSP instance

        Returns:
            EntropyComponents with all entropy measures
        """
        components = EntropyComponents()

        # Calculate each component
        components.local = self._local_domain_entropy(csp)
        components.constraint = self._constraint_entropy(csp)
        components.structural = self._structural_entropy(csp)
        components.kolmogorov = self._kolmogorov_approximation(csp)

        return components

    def _local_domain_entropy(self, csp: CSPInstance) -> float:
        """
        Calculate local domain entropy

        Measures uncertainty in variable assignments.
        Higher = more uncertain about values.

        Args:
            csp: CSP instance

        Returns:
            Normalized local entropy ∈ [0, 1]
        """
        if not csp.variables:
            return 0.0

        entropies = []
        max_domain_size = max(v.domain_size() for v in csp.variables)

        for var in csp.variables:
            domain_size = var.domain_size()

            if domain_size == 0:
                H_var = 0.0
            elif domain_size == 1:
                H_var = 0.0  # Deterministic
            else:
                # Assume uniform distribution over domain
                # H = -sum(p * log2(p))
                p = 1.0 / domain_size
                H_var = -domain_size * p * math.log2(p)

            # Normalize by maximum possible entropy for THIS variable
            # This gives better discrimination between different domain sizes
            max_entropy = math.log2(domain_size) if domain_size > 1 else 1.0
            H_norm = H_var / max_entropy if max_entropy > 0 else 0.0
            entropies.append(H_norm)

        # Average across variables
        H_local = np.mean(entropies) if entropies else 0.0
        return max(0.0, min(1.0, H_local))

    def _constraint_entropy(self, csp: CSPInstance) -> float:
        """
        Calculate constraint entropy

        Measures information reduction from constraints.
        Higher = constraints are less restrictive.

        Args:
            csp: CSP instance

        Returns:
            Normalized constraint entropy ∈ [0, 1]
        """
        if not csp.constraints:
            return 1.0  # No constraints = maximum entropy

        constraint_entropies = []

        for constraint in csp.constraints:
            # Calculate constraint tightness
            total_tuples = 1
            for var_name in constraint.variables:
                var = csp.get_variable(var_name)
                if var:
                    total_tuples *= var.domain_size()

            if total_tuples == 0:
                H_con = 0.0
            else:
                allowed = len(constraint.allowed_tuples)
                p_allowed = allowed / total_tuples
                p_forbidden = 1.0 - p_allowed

                # Entropy of binary distribution (allowed vs forbidden)
                if p_allowed > 0 and p_forbidden > 0:
                    H_con = -(p_allowed * math.log2(p_allowed) +
                             p_forbidden * math.log2(p_forbidden))
                elif p_allowed == 0:
                    H_con = 0.0  # All forbidden
                else:  # p_forbidden == 0
                    H_con = 0.0  # All allowed

                # Maximum constraint entropy is 1 bit
                constraint_entropies.append(H_con)

        # Average across constraints
        H_constraint = np.mean(constraint_entropies) if constraint_entropies else 0.0
        return max(0.0, min(1.0, H_constraint))

    def _structural_entropy(self, csp: CSPInstance) -> float:
        """
        Calculate structural entropy

        Measures randomness in constraint graph topology.
        Higher = more random structure.

        Args:
            csp: CSP instance

        Returns:
            Normalized structural entropy ∈ [0, 1]
        """
        G = csp.constraint_graph

        if G.number_of_nodes() == 0:
            return 0.0

        # Component 1: Degree distribution entropy
        degrees = [G.degree(n) for n in G.nodes()]
        if sum(degrees) == 0:
            H_degree = 0.0
        else:
            degree_probs = [d / sum(degrees) for d in degrees if d > 0]
            if sum(degree_probs) > 0:
                H_degree = -sum(p * math.log2(p) for p in degree_probs if p > 0)
                # Normalize by max possible degree entropy
                max_H = math.log2(len(degrees)) if len(degrees) > 1 else 1.0
                H_degree_norm = H_degree / max_H if max_H > 0 else 0.0
            else:
                H_degree_norm = 0.0

        # Component 2: Clustering coefficient (inverse of entropy)
        # High clustering = low entropy = ordered
        try:
            clustering = nx.average_clustering(G)
            H_clustering = 1.0 - clustering
        except:
            H_clustering = 0.5

        # Component 3: Path regularity
        try:
            if nx.is_connected(G):
                avg_path = nx.average_shortest_path_length(G)
                n = G.number_of_nodes()
                # Normalize: n = max path length (chain)
                path_regularity = 1.0 - (avg_path / n) if n > 0 else 0.0
            else:
                # Penalize disconnected graphs
                path_regularity = 1.0 / nx.number_connected_components(G)
        except:
            path_regularity = 0.5

        # Combine components
        H_structural = (0.4 * H_degree_norm +
                       0.3 * H_clustering +
                       0.3 * path_regularity)

        return max(0.0, min(1.0, H_structural))

    def _kolmogorov_approximation(self, csp: CSPInstance) -> float:
        """
        Approximate Kolmogorov complexity using compression

        Uses a simple compression-based approximation.
        Higher = more complex (less regular).

        Args:
            csp: CSP instance

        Returns:
            Normalized Kolmogorov complexity ∈ [0, 1]
        """
        try:
            import zlib
            import pickle

            # Serialize CSP
            csp_dict = {
                'vars': [(v.name, tuple(v.domain)) for v in csp.variables],
                'constraints': [
                    (c.variables, len(c.allowed_tuples))
                    for c in csp.constraints
                ]
            }

            csp_bytes = pickle.dumps(csp_dict)
            original_length = len(csp_bytes)

            if original_length == 0:
                return 0.0

            # Compress
            compressed = zlib.compress(csp_bytes, level=9)
            compressed_length = len(compressed)

            # Complexity ratio
            complexity = compressed_length / original_length

            # Normalize to [0, 1]
            # Typically compression gives 0.3-0.8 for structured data
            return max(0.0, min(1.0, complexity))

        except Exception:
            # Fallback if compression fails
            return 0.5


# ============================================================================
# Utility Functions
# ============================================================================

def shannon_entropy(probabilities: List[float]) -> float:
    """
    Calculate Shannon entropy

    H = -sum(p * log2(p))

    Args:
        probabilities: List of probabilities (must sum to 1)

    Returns:
        Entropy in bits
    """
    H = 0.0
    for p in probabilities:
        if p > 0:
            H -= p * math.log2(p)
    return H


def differential_entropy(data: List[float], bandwidth: float = None) -> float:
    """
    Calculate differential entropy for continuous variables

    h = -integral f(x) * log(f(x)) dx

    Uses kernel density estimation for probability density.

    Args:
        data: List of continuous samples
        bandwidth: KDE bandwidth (None = auto)

    Returns:
        Differential entropy in nats
    """
    # Convert to list if numpy array
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if not data or len(data) < 2:
        return 0.0

    try:
        from scipy.stats import gaussian_kde

        data_array = np.array(data)
        kde = gaussian_kde(data_array, bw_method=bandwidth)

        # Evaluate log-likelihood at data points
        log_densities = kde.logpdf(data_array)

        # Differential entropy (in nats)
        h = -np.mean(log_densities)

        return float(h)
    except ImportError:
        # Fallback: use histogram-based estimation
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        # Remove zero probabilities
        hist = hist[hist > 0]
        h = -np.sum(hist * np.log(hist + 1e-10)) * (bin_edges[1] - bin_edges[0])
        return float(h)


def sample_entropy(data: List[float], m: int = 2, r: float = 0.2) -> float:
    """
    Calculate sample entropy (regularity metric)

    Measures the regularity of time series data.
    Lower = more regular = less complex.

    Args:
        data: Time series data
        m: Template length
        r: Tolerance (as fraction of std)

    Returns:
        Sample entropy (lower = more regular)
    """
    N = len(data)
    if N < m + 1:
        return 0.0

    data = np.array(data)
    r_scaled = r * np.std(data)

    def _maxdist(x, y):
        return abs(x - y)

    def _phi(m):
        patterns = []
        for i in range(N - m + 1):
            patterns.append(data[i:i+m])

        matches = 0
        total = 0

        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                # Check if all components within tolerance
                match = True
                for k in range(m):
                    if _maxdist(patterns[i][k], patterns[j][k]) > r_scaled:
                        match = False
                        break
                if match:
                    matches += 1
                total += 1

        if total == 0:
            return 0.0
        return matches / total

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return 0.0

    return -math.log(phi_m1 / phi_m)


def approximate_entropy(data: List[float], m: int = 2, r: float = 0.2) -> float:
    """
    Calculate approximate entropy (ApEn)

    Similar to sample entropy but includes self-matches.
    Lower = more regular = less complex.

    Args:
        data: Time series data
        m: Template length
        r: Tolerance (as fraction of std)

    Returns:
        Approximate entropy
    """
    N = len(data)
    if N < m + 1:
        return 0.0

    data = np.array(data)
    r_scaled = r * np.std(data)

    def _phi(m):
        phi_val = 0.0
        for i in range(N - m + 1):
            template = data[i:i+m]
            matches = 0
            for j in range(N - m + 1):
                comparison = data[j:j+m]
                if max(abs(template - comparison)) <= r_scaled:
                    matches += 1

            if matches > 0:
                phi_val += math.log(matches / (N - m + 1))

        return phi_val / (N - m + 1)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return phi_m - phi_m1


def normalized_entropy(domain_size: int) -> float:
    """
    Calculate normalized entropy for uniform distribution

    Args:
        domain_size: Size of domain

    Returns:
        Normalized entropy ∈ [0, 1]
    """
    if domain_size <= 1:
        return 0.0

    # Uniform distribution entropy
    H = math.log2(domain_size)

    # Normalize by log2 of domain_size (which is the maximum)
    return H / math.log2(domain_size)


if __name__ == "__main__":
    print("=" * 70)
    print("Disorder Entropy Engine - Demonstration")
    print("=" * 70)

    from gamma1.core.csp_models import create_test_csp, create_tree_csp, create_dense_csp

    # Test on different CSP types
    calculator = DisorderEntropy()

    # Test CSP
    test_csp = create_test_csp(n_variables=10, domain_size=5)
    test_entropy = calculator.calculate(test_csp)
    print(f"\n[OK] Test CSP entropy: {test_entropy.total():.3f}")
    print(f"  Local: {test_entropy.local:.3f}")
    print(f"  Constraint: {test_entropy.constraint:.3f}")
    print(f"  Structural: {test_entropy.structural:.3f}")
    print(f"  Kolmogorov: {test_entropy.kolmogorov:.3f}")

    # Tree CSP (should have lower entropy - more structured)
    tree_csp = create_tree_csp(n_variables=10, domain_size=5)
    tree_entropy = calculator.calculate(tree_csp)
    print(f"\n[OK] Tree CSP entropy: {tree_entropy.total():.3f}")
    print(f"  Local: {tree_entropy.local:.3f}")
    print(f"  Constraint: {tree_entropy.constraint:.3f}")
    print(f"  Structural: {tree_entropy.structural:.3f}")

    # Dense CSP (should have higher entropy - more chaotic)
    dense_csp = create_dense_csp(n_variables=10, domain_size=5)
    dense_entropy = calculator.calculate(dense_csp)
    print(f"\n[OK] Dense CSP entropy: {dense_entropy.total():.3f}")
    print(f"  Local: {dense_entropy.local:.3f}")
    print(f"  Constraint: {dense_entropy.constraint:.3f}")
    print(f"  Structural: {dense_entropy.structural:.3f}")

    # Comparison
    print(f"\n[OK] Entropy comparison:")
    print(f"  Tree < Test: {tree_entropy.total() < test_entropy.total()}")
    print(f"  Test < Dense: {test_entropy.total() < dense_entropy.total()}")

    print("\n" + "=" * 70)
    print("[OK] Disorder entropy engine demonstration complete")
    print("=" * 70)
