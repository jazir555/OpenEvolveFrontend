"""
Γ₁ Causal Coherence Engine (C)

Implements causal coherence measurement for CSP instances.
Higher coherence = more structured causal relationships = more solvable.

Components:
- Graph Coherence: Topological regularity of constraint graph
- Flow Coherence: Information flow regularity (transfer entropy approximation)
- Stability Coherence: Intervention stability
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import networkx as nx
import numpy as np
from gamma1.core.csp_models import CSPInstance


@dataclass
class CoherenceComponents:
    """
    Causal Coherence (C) Components

    Attributes:
        graph: Graph structure coherence (topology regularity)
        flow: Information flow regularity
        stability: Intervention stability
    """
    graph: float = 0.0
    flow: float = 0.0
    stability: float = 0.0

    def total(self, weights: tuple = (0.4, 0.3, 0.3)) -> float:
        """
        Calculate total C with given weights

        Args:
            weights: (w_graph, w_flow, w_stability)

        Returns:
            Total coherence C ∈ [0, 1]
        """
        C = (weights[0] * self.graph +
             weights[1] * self.flow +
             weights[2] * self.stability)
        return max(0.0, min(1.0, C))


class CausalCoherence:
    """
    Calculate causal coherence for CSP instances

    C = w_graph * C_graph + w_flow * C_flow + w_stability * C_stab

    Higher C = more coherent causal structure = more solvable
    """

    def __init__(self, weights: tuple = (0.4, 0.3, 0.3)):
        """
        Initialize coherence calculator

        Args:
            weights: (w_graph, w_flow, w_stability)
        """
        if len(weights) != 3 or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.weights = weights

    def calculate(self, csp: CSPInstance) -> CoherenceComponents:
        """
        Calculate causal coherence for CSP

        Args:
            csp: CSP instance

        Returns:
            CoherenceComponents with all coherence measures
        """
        components = CoherenceComponents()

        # Calculate each component
        components.graph = self._graph_coherence(csp)
        components.flow = self._flow_coherence(csp)
        components.stability = self._stability_coherence(csp)

        return components

    def _graph_coherence(self, csp: CSPInstance) -> float:
        """
        Calculate graph structure coherence

        Measures topological regularity of constraint graph.
        Higher = more regular structure = more coherent.

        Args:
            csp: CSP instance

        Returns:
            Normalized graph coherence ∈ [0, 1]
        """
        G = csp.constraint_graph

        if G.number_of_nodes() == 0:
            return 0.0

        # Metric 1: Average path length (shorter = more coherent)
        try:
            if nx.is_connected(G):
                avg_path = nx.average_shortest_path_length(G)
                n = G.number_of_nodes()
                # Normalize: max path ≈ n (in chain)
                path_score = 1.0 - (avg_path / n) if n > 0 else 0.0
            else:
                # Penalize disconnected graphs
                n_components = nx.number_connected_components(G)
                path_score = 1.0 / n_components if n_components > 0 else 0.0
        except:
            path_score = 0.5

        # Metric 2: Clustering coefficient (higher = local coherence)
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0.0

        # Metric 3: Degree balance (balanced = regular structure)
        degrees = [G.degree(n) for n in G.nodes()]
        if len(degrees) > 1:
            degree_variance = np.var(degrees)
            max_degree = max(degrees) if degrees else 1
            # Normalize variance
            max_variance = (max_degree ** 2) / 4  # Max variance for given range
            balance_score = 1.0 - (degree_variance / (max_variance + 1e-9))
        else:
            balance_score = 1.0

        # Metric 4: Tree-like structure (trees are most coherent)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        # For tree: m = n - 1
        if n > 0:
            tree_score = 1.0 - abs(m - (n - 1)) / n
        else:
            tree_score = 0.0

        # Combine metrics
        C_graph = (0.25 * path_score +
                  0.25 * clustering +
                  0.25 * balance_score +
                  0.25 * tree_score)

        return max(0.0, min(1.0, C_graph))

    def _flow_coherence(self, csp: CSPInstance) -> float:
        """
        Calculate information flow regularity

        Measures regularity of information propagation through constraints.
        Higher = more regular flow = more coherent.

        Args:
            csp: CSP instance

        Returns:
            Normalized flow coherence ∈ [0, 1]
        """
        G = csp.constraint_graph

        # No nodes or no edges means no flow possible
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return 0.0

        # Approximation 1: Edge betweenness centrality (flow balance)
        try:
            betweenness = nx.edge_betweenness_centrality(G, normalized=True)
            if betweenness:
                betweenness_values = list(betweenness.values())
                # Regular flow = balanced betweenness
                betweenness_mean = np.mean(betweenness_values)
                betweenness_std = np.std(betweenness_values)
                cv = betweenness_std / (betweenness_mean + 1e-9)  # Coefficient of variation
                flow_balance = 1.0 / (1.0 + cv)  # Lower CV = higher score
            else:
                flow_balance = 0.5
        except:
            flow_balance = 0.5

        # Approximation 2: Propagation effectiveness
        propagation_power = self._estimate_propagation_power(csp)

        # Combine
        C_flow = 0.5 * flow_balance + 0.5 * propagation_power

        return max(0.0, min(1.0, C_flow))

    def _estimate_propagation_power(self, csp: CSPInstance) -> float:
        """
        Estimate constraint propagation effectiveness

        Simulates arc consistency to estimate how much constraints reduce search space.

        Args:
            csp: CSP instance

        Returns:
            Propagation effectiveness ∈ [0, 1]
        """
        if not csp.constraints:
            return 0.0

        initial_total = sum(v.domain_size() for v in csp.variables)

        if initial_total == 0:
            return 0.0

        # Estimate reduction from constraint tightness
        total_reduction = 0

        for constraint in csp.constraints:
            # Calculate constraint tightness
            total_tuples = 1
            for var_name in constraint.variables:
                var = csp.get_variable(var_name)
                if var:
                    total_tuples *= var.domain_size()

            if total_tuples > 0:
                tightness = 1.0 - (len(constraint.allowed_tuples) / total_tuples)

                # Estimate domain reduction
                # Rough approximation: each constraint reduces affected domains
                for var_name in constraint.variables:
                    var = csp.get_variable(var_name)
                    if var:
                        reduction = var.domain_size() * tightness * 0.3  # 30% estimate
                        total_reduction += reduction

        # Effectiveness: fraction of domain values removed
        effectiveness = total_reduction / (initial_total + 1e-9)

        return max(0.0, min(1.0, effectiveness))

    def _stability_coherence(self, csp: CSPInstance) -> float:
        """
        Calculate intervention stability

        Measures how stable variable assignments are.
        Stable interventions = coherent causal structure.

        Args:
            csp: CSP instance

        Returns:
            Normalized stability coherence ∈ [0, 1]
        """
        if not csp.variables:
            return 0.0

        G = csp.constraint_graph
        n_vars = csp.num_variables()

        # Sample variables to test
        sample_size = min(10, n_vars)
        var_names = [v.name for v in csp.variables]
        sample_vars = var_names[:sample_size]

        stability_scores = []

        for var_name in sample_vars:
            # Count affected variables (reachable in constraint graph)
            try:
                if var_name in G:
                    # Use BFS to find reachable nodes
                    reachable = nx.single_source_shortest_path_length(G, var_name, cutoff=3)
                    affected = len(reachable) - 1  # Exclude the variable itself
                else:
                    affected = 0
            except:
                affected = 0

            # Stability: Moderate affected is best
            # Too few: disconnected (incoherent)
            # Too many: chaotic propagation
            optimal_affected = n_vars * 0.3  # 30% of variables
            deviation = abs(affected - optimal_affected) / (n_vars + 1e-9)
            stability = 1.0 - deviation

            stability_scores.append(stability)

        # Average stability
        C_stab = np.mean(stability_scores) if stability_scores else 0.5

        return max(0.0, min(1.0, C_stab))


# ============================================================================
# Advanced Causal Coherence Methods
# ============================================================================

def granger_causality_test(
    x: List[float],
    y: List[float],
    max_lag: int = 5,
    significance: float = 0.05
) -> Tuple[float, bool]:
    """
    Perform Granger causality test

    Tests if x Granger-causes y (x predicts y better than y's past alone).

    Args:
        x: Time series data (causal variable)
        y: Time series data (effect variable)
        max_lag: Maximum lag to test
        significance: Significance threshold

    Returns:
        (f_statistic, is_significant)
    """
    try:
        from scipy.stats import f
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        if len(x) < max_lag + 2 or len(y) < max_lag + 2:
            return (0.0, False)

        x = np.array(x)
        y = np.array(y)

        # Create lagged features
        def create_lagged_features(series, lag):
            features = []
            for i in range(lag, len(series)):
                features.append(series[i-lag:i][::-1])  # Past values
            return np.array(features)

        # Prepare data
        y_lagged = create_lagged_features(y, max_lag)
        x_lagged = create_lagged_features(x, max_lag)

        min_len = min(len(y_lagged), len(x_lagged))
        y_target = y[max_lag:max_lag+min_len]
        y_lagged = y_lagged[:min_len]
        x_lagged = x_lagged[:min_len]

        # Model 1: y predicted by its own past (restricted)
        model_restricted = LinearRegression()
        model_restricted.fit(y_lagged, y_target)
        y_pred_restricted = model_restricted.predict(y_lagged)
        mse_restricted = mean_squared_error(y_target, y_pred_restricted)

        # Model 2: y predicted by its own past + x's past (unrestricted)
        combined_features = np.hstack([y_lagged, x_lagged])
        model_unrestricted = LinearRegression()
        model_unrestricted.fit(combined_features, y_target)
        y_pred_unrestricted = model_unrestricted.predict(combined_features)
        mse_unrestricted = mean_squared_error(y_target, y_pred_unrestricted)

        # F-statistic
        if mse_unrestricted == 0:
            return (0.0, False)

        num_params = max_lag  # Number of added parameters
        den_params = len(y_target) - 2 * max_lag

        if den_params <= 0:
            return (0.0, False)

        f_stat = ((mse_restricted - mse_unrestricted) / num_params) / (mse_unrestricted / den_params)

        # Test significance
        # Critical value at alpha=0.05 (approximate)
        critical_value = 3.0  # Approximate for typical sample sizes
        is_significant = f_stat > critical_value

        return (f_stat, is_significant)

    except Exception:
        return (0.0, False)


def transfer_entropy(
    source: List[float],
    target: List[float],
    n_bins: int = 10,
    k: int = 1
) -> float:
    """
    Calculate transfer entropy from source to target

    TE = sum p(x_t+1, x_t, y_t) * log(p(x_t+1|x_t, y_t) / p(x_t+1|x_t))

    Measures information flow from source to target.

    Args:
        source: Source time series (y)
        target: Target time series (x)
        n_bins: Number of bins for discretization
        k: History length

    Returns:
        Transfer entropy (in bits)
    """
    try:
        from collections import defaultdict

        if len(source) < k + 2 or len(target) < k + 2:
            return 0.0

        # Discretize
        def discretize(series, bins):
            min_val, max_val = min(series), max(series)
            if max_val == min_val:
                return [0] * len(series)
            range_val = max_val - min_val
            return [int((x - min_val) / range_val * (bins - 1)) for x in series]

        source_disc = discretize(source, n_bins)
        target_disc = discretize(target, n_bins)

        # Count patterns
        te = 0.0
        n_samples = len(target_disc) - k - 1

        if n_samples <= 0:
            return 0.0

        # Joint counts: p(x_t+1, x_t, y_t)
        joint_counts = defaultdict(int)
        # Conditional counts: p(x_t+1|x_t)
        cond_x_counts = defaultdict(int)
        # Marginal counts: p(x_t)
        marginal_x_counts = defaultdict(int)

        for i in range(k, len(target_disc) - 1):
            x_next = target_disc[i + 1]
            x_hist = tuple(target_disc[i-k+1:i+1])
            y_hist = source_disc[i]

            joint_counts[(x_next, x_hist, y_hist)] += 1
            cond_x_counts[(x_next, x_hist)] += 1
            marginal_x_counts[x_hist] += 1

        total = n_samples

        # Calculate TE
        for (x_next, x_hist, y_hist), count in joint_counts.items():
            p_joint = count / total

            # p(x_next | x_hist, y_hist)
            p_cond_joint = count / joint_counts.get((x_next, x_hist, y_hist), count)

            # p(x_next | x_hist)
            p_cond_x = cond_x_counts.get((x_next, x_hist), 0) / marginal_x_counts.get(x_hist, 1)

            if p_cond_joint > 0 and p_cond_x > 0:
                te += p_joint * math.log2(p_cond_joint / p_cond_x)

        return max(0.0, te)

    except Exception:
        return 0.0


def bayesian_network_score(csp: CSPInstance) -> float:
    """
    Calculate Bayesian network structure score

    Measures how well CSP fits a Bayesian network structure.
    Higher = more coherent causal structure.

    Args:
        csp: CSP instance

    Returns:
        Bayesian network score ∈ [0, 1]
    """
    G = csp.constraint_graph

    if G.number_of_nodes() == 0:
        return 0.0

    try:
        # Score 1: DAG-friendliness (can edges be oriented?)
        # Check if graph has many cycles
        try:
            is_dag = nx.is_directed_acyclic_graph(G.to_directed())
        except:
            is_dag = False

        # For undirected, approximate DAG-ness
        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()
        max_edges_before_cycle = n_nodes - 1

        dag_score = min(1.0, max_edges_before_cycle / (n_edges + 1))

        # Score 2: Conditional independence structure
        # More sparse = better Bayesian network
        density = G.number_of_edges() / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        sparsity_score = 1.0 - density

        # Score 3: Local Markov property satisfaction
        # Variables should be independent of non-descendants given parents
        # Approximate using clustering
        try:
            clustering = nx.average_clustering(G)
            markov_score = clustering  # High clustering = local structure
        except:
            markov_score = 0.5

        # Combine
        bayesian_score = 0.4 * dag_score + 0.3 * sparsity_score + 0.3 * markov_score

        return max(0.0, min(1.0, bayesian_score))

    except Exception:
        return 0.5


if __name__ == "__main__":
    print("=" * 70)
    print("Causal Coherence Engine - Demonstration")
    print("=" * 70)

    from gamma1.core.csp_models import create_test_csp, create_tree_csp, create_dense_csp

    # Test on different CSP types
    calculator = CausalCoherence()

    # Test CSP
    test_csp = create_test_csp(n_variables=10, domain_size=5)
    test_coherence = calculator.calculate(test_csp)
    print(f"\n[OK] Test CSP coherence: {test_coherence.total():.3f}")
    print(f"  Graph: {test_coherence.graph:.3f}")
    print(f"  Flow: {test_coherence.flow:.3f}")
    print(f"  Stability: {test_coherence.stability:.3f}")

    # Tree CSP (should have higher coherence - more structured)
    tree_csp = create_tree_csp(n_variables=10, domain_size=5)
    tree_coherence = calculator.calculate(tree_csp)
    print(f"\n[OK] Tree CSP coherence: {tree_coherence.total():.3f}")
    print(f"  Graph: {tree_coherence.graph:.3f}")
    print(f"  Flow: {tree_coherence.flow:.3f}")
    print(f"  Stability: {tree_coherence.stability:.3f}")

    # Dense CSP (should have lower coherence - more chaotic)
    dense_csp = create_dense_csp(n_variables=10, domain_size=5)
    dense_coherence = calculator.calculate(dense_csp)
    print(f"\n[OK] Dense CSP coherence: {dense_coherence.total():.3f}")
    print(f"  Graph: {dense_coherence.graph:.3f}")
    print(f"  Flow: {dense_coherence.flow:.3f}")
    print(f"  Stability: {dense_coherence.stability:.3f}")

    # Comparison
    print(f"\n[OK] Coherence comparison:")
    print(f"  Tree > Test: {tree_coherence.total() > test_coherence.total()}")
    print(f"  Test > Dense: {test_coherence.total() > dense_coherence.total()}")

    print("\n" + "=" * 70)
    print("[OK] Causal coherence engine demonstration complete")
    print("=" * 70)
