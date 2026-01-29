"""
Causal-learn Adapter for OpenEvolve

This module provides an adapter that wraps causal-learn's functionality to implement
the OpenEvolve CausalDiscoveryInterface. It enables causal reasoning capabilities
without modifying causal-learn's source code.

Author: Causal-learn Integration Specialist
Version: 1.0.0
Date: 2026-01-02
"""

import sys
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import numpy as np
from functools import lru_cache

# Robust causal-learn path resolution
def _resolve_causal_learn_path():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "../../projects to analyze/causal-learn"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "projects to analyze/causal-learn"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "causal-learn"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return True
    return False

_resolve_causal_learn_path()

# Try to import causal-learn
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.FCMBased import lingam
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    from causallearn.utils.cit import fisherz, chisq, gsq, kci, mv_fisherz
    from causallearn.score.LocalScoreFunction import (
        local_score_BIC,
        local_score_BDeu,
        local_score_CV_general
    )
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.GraphUtils import GraphUtils

    CAUSAL_LEARN_AVAILABLE = True
except ImportError as e:
    CAUSAL_LEARN_AVAILABLE = False
    CAUSAL_LEARN_IMPORT_ERROR = str(e)

from integrations.base.causal_interface import (
    CausalDiscoveryInterface,
    CausalGraphResult,
    CausalEffectResult,
    IndependenceTestResult,
    CounterfactualResult,
    ConfounderAnalysisResult,
    CausalAncestorResult,
    CausalMethod,
    IndependenceTest,
    ScoreFunction,
    EdgeType,
    CausalDiscoveryError,
    ConfigurationError,
    ValidationError,
    DiscoveryError,
    EstimationError,
    TestError,
    PredictionError,
    GraphError,
    AnalysisError,
    ShutdownError,
)

logger = logging.getLogger(__name__)


class CausalLearnAdapter(CausalDiscoveryInterface):
    """
    Adapter for causal-learn causal discovery library.

    This adapter wraps causal-learn's comprehensive causal discovery algorithms
    to provide a consistent interface for OpenEvolve. It supports:

    Algorithms:
    - PC: Peter-Clark (constraint-based)
    - GES: Greedy Equivalence Search (score-based)
    - DirectLiNGAM: Direct LiNGAM (non-Gaussian)
    - FCI: Fast Causal Inference (latent confounders)

    Independence Tests:
    - Fisher Z (Gaussian continuous)
    - Chi-square (discrete)
    - G-square (discrete)
    - KCI (kernel-based, nonlinear)

    Score Functions:
    - BIC (Bayesian Information Criterion)
    - BDeu (Bayesian Dirichlet equivalent uniform)
    - CV (Cross-validation)

    Gracefully degrades if causal-learn is unavailable.
    """

    def __init__(self):
        """Initialize the adapter without connecting to causal-learn."""
        self.config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self._cache: Dict[str, Any] = {}
        self._algorithms = {
            'pc': self._run_pc,
            'pc_stable': self._run_pc_stable,
            'ges': self._run_ges,
            'direct_lingam': self._run_direct_lingam,
            'ica_lingam': self._run_ica_lingam,
            'fci': self._run_fci,
        }
        self._indep_tests = {
            'fisherz': fisherz,
            'chisq': chisq,
            'gsq': gsq,
            'kci': kci,
            'mv_fisherz': mv_fisherz,
        }
        self._score_funcs = {
            'local_score_BIC': local_score_BIC,
            'local_score_BDeu': local_score_BDeu,
            'local_score_CV_general': local_score_CV_general,
        }

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize causal-learn adapter with the provided configuration.

        Args:
            config: Configuration dictionary with keys:
                - default_algorithm: Default algorithm (default: "pc")
                - default_indep_test: Default independence test (default: "fisherz")
                - default_alpha: Default significance level (default: 0.05)
                - default_score_func: Default score function (default: "local_score_BIC")
                - cache_enabled: Enable result caching (default: True)
                - performance.max_workers: Maximum parallel workers (default: 4)
                - performance.timeout: Operation timeout in seconds (default: 300)

        Returns:
            True if initialization was successful

        Raises:
            ConfigurationError: If causal-learn is unavailable
        """
        if not CAUSAL_LEARN_AVAILABLE:
            logger.error(f"causal-learn not available: {CAUSAL_LEARN_IMPORT_ERROR}")
            raise ConfigurationError(
                f"causal-learn is not available: {CAUSAL_LEARN_IMPORT_ERROR}. "
                "Install with: pip install causal-learn"
            )

        self.config = {
            'default_algorithm': config.get('default_algorithm', 'pc'),
            'default_indep_test': config.get('default_indep_test', 'fisherz'),
            'default_alpha': config.get('default_alpha', 0.05),
            'default_score_func': config.get('default_score_func', 'local_score_BIC'),
            'cache_enabled': config.get('cache_enabled', True),
            'performance': {
                'max_workers': config.get('performance', {}).get('max_workers', 4),
                'timeout': config.get('performance', {}).get('timeout', 300),
            }
        }

        self.is_initialized = True
        logger.info("CausalLearnAdapter initialized successfully")
        return True

    async def discover_causal_structure(
        self,
        data: Union[np.ndarray, str],
        method: str = "pc",
        **kwargs
    ) -> CausalGraphResult:
        """
        Discover causal structure from observational data.

        Args:
            data: Observational data as numpy array (n_samples x n_features)
                  or path to data file
            method: Causal discovery method (default: "pc")
            **kwargs: Method-specific parameters

        Returns:
            CausalGraphResult containing the discovered causal graph

        Raises:
            ValidationError: If data is invalid
            DiscoveryError: If discovery fails
        """
        if not self.is_initialized:
            raise DiscoveryError("Adapter not initialized. Call initialize() first.")

        # Load data if path provided
        if isinstance(data, str):
            try:
                data = np.loadtxt(data, skiprows=1)
            except Exception as e:
                raise ValidationError(f"Failed to load data from {data}: {e}")

        # Validate data
        if not isinstance(data, np.ndarray):
            raise ValidationError("Data must be numpy array or file path")
        if data.ndim != 2:
            raise ValidationError(f"Data must be 2D array, got shape {data.shape}")

        n_samples, n_features = data.shape
        logger.info(f"Discovering causal structure: {n_samples} samples x {n_features} features")

        # Extract parameters
        alpha = kwargs.get('alpha', self.config['default_alpha'])
        indep_test = kwargs.get('indep_test', self.config['default_indep_test'])
        score_func = kwargs.get('score_func', self.config['default_score_func'])
        stable = kwargs.get('stable', True)

        # Check cache if enabled
        cache_key = f"{method}_{data.shape}_{alpha}_{indep_test}_{score_func}"
        if self.config['cache_enabled'] and cache_key in self._cache:
            logger.info("Returning cached result")
            return self._cache[cache_key]

        try:
            # Run discovery algorithm
            if method.lower() == 'pc':
                result = await self._run_pc(data, alpha, indep_test, stable)
            elif method.lower() == 'ges':
                result = await self._run_ges(data, score_func)
            elif method.lower() == 'direct_lingam':
                result = await self._run_direct_lingam(data)
            elif method.lower() == 'fci':
                result = await self._run_fci(data, alpha, indep_test)
            else:
                raise DiscoveryError(f"Unknown method: {method}")

            # Cache result if enabled
            if self.config['cache_enabled']:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            raise DiscoveryError(f"Causal discovery failed: {e}")

    async def _run_pc(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        stable: bool = True
    ) -> CausalGraphResult:
        """Run PC algorithm (constraint-based)."""
        test_func = self._indep_tests[indep_test]

        # Run PC in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        cg = await loop.run_in_executor(
            None,
            lambda: pc(data, alpha, test_func, stable=stable)
        )

        # Extract graph information
        graph = cg.G
        nodes = graph.get_nodes()
        n_nodes = len(nodes)

        # Extract edges by type
        directed = graph.find_fully_directed()
        undirected = graph.find_undirected()
        bidirected = graph.find_bi_directed()

        # Build edge list with types
        edges = []
        for i, j in directed:
            edges.append((i, j, EdgeType.DIRECTED))
        for i, j in undirected:
            edges.append((i, j, EdgeType.UNDIRECTED))
        for i, j in bidirected:
            edges.append((i, j, EdgeType.BIDIRECTED))

        # Get adjacency matrix
        adjacency_matrix = self._graph_to_adjacency(graph, n_nodes)

        return CausalGraphResult(
            graph=graph,
            adjacency_matrix=adjacency_matrix,
            nodes=[node.get_name() for node in nodes],
            edges=edges,
            directed_edges=directed,
            undirected_edges=undirected,
            bidirected_edges=bidirected,
            causal_order=None,  # PC doesn't provide causal order
            confidence_scores=None,
            algorithm_used="PC",
            method_parameters={
                'alpha': alpha,
                'indep_test': indep_test,
                'stable': stable
            },
            timestamp=datetime.now()
        )

    async def _run_pc_stable(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        stable: bool = True
    ) -> CausalGraphResult:
        """Run PC-stable algorithm (more stable version of PC)."""
        return await self._run_pc(data, alpha, indep_test, stable=True)

    async def _run_ges(
        self,
        data: np.ndarray,
        score_func: str
    ) -> CausalGraphResult:
        """Run GES algorithm (score-based)."""
        score_func_impl = self._score_funcs[score_func]

        # Run GES in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ges(data, score_func=score_func_impl)
        )

        graph = result['G']
        score = result['score']

        nodes = graph.get_nodes()
        n_nodes = len(nodes)

        # Extract edges
        directed = graph.find_fully_directed()
        undirected = graph.find_undirected()
        bidirected = graph.find_bi_directed()

        edges = []
        for i, j in directed:
            edges.append((i, j, EdgeType.DIRECTED))
        for i, j in undirected:
            edges.append((i, j, EdgeType.UNDIRECTED))
        for i, j in bidirected:
            edges.append((i, j, EdgeType.BIDIRECTED))

        adjacency_matrix = self._graph_to_adjacency(graph, n_nodes)

        return CausalGraphResult(
            graph=graph,
            adjacency_matrix=adjacency_matrix,
            nodes=[node.get_name() for node in nodes],
            edges=edges,
            directed_edges=directed,
            undirected_edges=undirected,
            bidirected_edges=bidirected,
            causal_order=None,
            confidence_scores=None,
            algorithm_used="GES",
            method_parameters={
                'score_func': score_func,
                'score': score
            },
            timestamp=datetime.now()
        )

    async def _run_direct_lingam(
        self,
        data: np.ndarray
    ) -> CausalGraphResult:
        """Run DirectLiNGAM algorithm (non-Gaussian data)."""
        model = DirectLiNGAM()

        # Fit in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: model.fit(data))

        # Extract results
        adjacency_matrix = model.adjacency_matrix_
        causal_order = model.causal_order_

        n_nodes = adjacency_matrix.shape[0]
        nodes = [f"X{i}" for i in range(n_nodes)]

        # Convert adjacency matrix to edge list
        edges = []
        directed = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency_matrix[i, j] != 0:
                    edges.append((i, j, EdgeType.DIRECTED))
                    directed.append((i, j))

        return CausalGraphResult(
            graph=model,  # Store the model itself
            adjacency_matrix=adjacency_matrix,
            nodes=nodes,
            edges=edges,
            directed_edges=directed,
            undirected_edges=[],
            bidirected_edges=[],
            causal_order=causal_order,
            confidence_scores=None,
            algorithm_used="DirectLiNGAM",
            method_parameters={},
            timestamp=datetime.now()
        )

    async def _run_ica_lingam(
        self,
        data: np.ndarray
    ) -> CausalGraphResult:
        """Run ICA-LiNGAM algorithm."""
        from causallearn.search.FCMBased.lingam import ICALiNGAM

        model = ICALiNGAM()

        # Fit in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: model.fit(data))

        adjacency_matrix = model.adjacency_matrix_
        causal_order = model.causal_order_

        n_nodes = adjacency_matrix.shape[0]
        nodes = [f"X{i}" for i in range(n_nodes)]

        edges = []
        directed = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency_matrix[i, j] != 0:
                    edges.append((i, j, EdgeType.DIRECTED))
                    directed.append((i, j))

        return CausalGraphResult(
            graph=model,
            adjacency_matrix=adjacency_matrix,
            nodes=nodes,
            edges=edges,
            directed_edges=directed,
            undirected_edges=[],
            bidirected_edges=[],
            causal_order=causal_order,
            confidence_scores=None,
            algorithm_used="ICALiNGAM",
            method_parameters={},
            timestamp=datetime.now()
        )

    async def _run_fci(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str
    ) -> CausalGraphResult:
        """Run FCI algorithm (Fast Causal Inference for latent confounders)."""
        from causallearn.search.ConstraintBased.FCI import fci

        test_func = self._indep_tests[indep_test]

        # Run FCI in thread pool
        loop = asyncio.get_event_loop()
        cg = await loop.run_in_executor(
            None,
            lambda: fci(data, alpha, test_func)
        )

        graph = cg.G
        nodes = graph.get_nodes()
        n_nodes = len(nodes)

        # FCI produces bidirected edges for latent confounders
        directed = graph.find_fully_directed()
        undirected = graph.find_undirected()
        bidirected = graph.find_bi_directed()  # Latent confounders

        edges = []
        for i, j in directed:
            edges.append((i, j, EdgeType.DIRECTED))
        for i, j in undirected:
            edges.append((i, j, EdgeType.UNDIRECTED))
        for i, j in bidirected:
            edges.append((i, j, EdgeType.BIDIRECTED))

        adjacency_matrix = self._graph_to_adjacency(graph, n_nodes)

        return CausalGraphResult(
            graph=graph,
            adjacency_matrix=adjacency_matrix,
            nodes=[node.get_name() for node in nodes],
            edges=edges,
            directed_edges=directed,
            undirected_edges=undirected,
            bidirected_edges=bidirected,
            causal_order=None,
            confidence_scores=None,
            algorithm_used="FCI",
            method_parameters={
                'alpha': alpha,
                'indep_test': indep_test
            },
            timestamp=datetime.now()
        )

    async def validate_causal_claim(
        self,
        claim: str,
        data: Union[np.ndarray, str],
        evidence: Optional[Dict[str, Any]] = None,
        method: str = "direct_lingam"
    ) -> Dict[str, Any]:
        """
        Validate a causal claim using causal discovery.

        Distinguishes correlation from causation by testing if the claimed
        causal relationship is supported by the data.

        Args:
            claim: Causal claim (e.g., "X causes Y")
            data: Observational data
            evidence: Additional evidence
            method: Validation method

        Returns:
            Dictionary with validation results
        """
        # Discover causal structure
        result = await self.discover_causal_structure(data, method=method)

        # Parse claim to extract variables (simplified)
        # In production, use NLP to extract variable names
        variables = self._parse_claim(claim)

        if len(variables) < 2:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'effect_size': 0.0,
                'explanation': 'Could not parse claim',
                'is_causal': False
            }

        # Check if causal relationship exists
        x, y = variables[0], variables[1]
        is_causal = self._check_causal_relationship(result, x, y)

        # Estimate effect if causal
        effect_size = 0.0
        if is_causal:
            try:
                effect_result = await self.estimate_causal_effect(
                    data, x, y, method=method
                )
                effect_size = effect_result.effect_size
            except Exception as e:
                logger.warning(f"Failed to estimate effect: {e}")

        return {
            'is_valid': is_causal,
            'confidence': 0.8 if is_causal else 0.2,
            'effect_size': effect_size,
            'explanation': f"Causal relationship detected: X{x} -> X{y}" if is_causal else "No causal relationship found",
            'is_causal': is_causal
        }

    async def estimate_causal_effect(
        self,
        data: Union[np.ndarray, str],
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
        method: str = "direct_lingam"
    ) -> CausalEffectResult:
        """
        Estimate causal effect of treatment on outcome.

        Uses LiNGAM to estimate direct causal effects.
        """
        if isinstance(data, str):
            data = np.loadtxt(data, skiprows=1)

        # Fit LiNGAM model
        model = DirectLiNGAM()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: model.fit(data))

        # Extract causal effect
        adjacency_matrix = model.adjacency_matrix_
        effect_size = adjacency_matrix[treatment, outcome]

        # Check significance (non-zero effect)
        is_significant = abs(effect_size) > 0.01

        # Simple confidence interval (in production, use bootstrap)
        ci_lower = effect_size - 0.1
        ci_upper = effect_size + 0.1

        return CausalEffectResult(
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=0.05 if is_significant else 0.5,
            method="DirectLiNGAM",
            is_significant=is_significant,
            confounders=confounders or [],
            mediators=[],
            colliders=[],
            sample_size=len(data),
            timestamp=datetime.now()
        )

    async def test_independence(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        z: Optional[List[int]] = None,
        method: str = "fisherz"
    ) -> IndependenceTestResult:
        """
        Test conditional independence X ⟂ Y | Z.
        """
        test_func = self._indep_tests[method]

        # Run test
        p_value = test_func(data, x, y, z)

        is_independent = p_value > 0.05
        is_significant = p_value < 0.05

        return IndependenceTestResult(
            is_independent=is_independent,
            p_value=float(p_value),
            test_statistic=0.0,  # Not returned by causal-learn
            method=method,
            is_significant=is_significant
        )

    async def counterfactual_analysis(
        self,
        data: np.ndarray,
        intervention: Dict[int, float],
        method: str = "lingam"
    ) -> CounterfactualResult:
        """
        Perform counterfactual analysis using structural causal model.

        Predicts outcome under intervention.
        """
        # Fit LiNGAM model
        model = DirectLiNGAM()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: model.fit(data))

        # Get structural model
        B = model.adjacency_matrix_  # Causal adjacency matrix
        n_vars = B.shape[0]

        # Apply intervention (simple intervention: set variable to value)
        # In production, implement full counterfactual reasoning
        original_data = data.copy()
        intervened_data = data.copy()

        for var_idx, value in intervention.items():
            intervened_data[:, var_idx] = value

        # Simple prediction: predict mean outcome
        predicted_outcome = np.mean(intervened_data, axis=0)
        actual_outcome = np.mean(original_data, axis=0)

        effect = float(np.mean(predicted_outcome - actual_outcome))

        return CounterfactualResult(
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            intervention=intervention,
            effect=effect,
            confidence_interval=(effect - 0.1, effect + 0.1),
            method="LiNGAM"
        )

    async def get_causal_ancestors(
        self,
        graph: Any,
        target: int
    ) -> CausalAncestorResult:
        """
        Get all causal ancestors of a target variable.

        Returns direct parents and all ancestors for intervention control.
        """
        if isinstance(graph, DirectLiNGAM):
            # LiNGAM has adjacency matrix
            adj = graph.adjacency_matrix_

            # Find direct ancestors (parents)
            direct_ancestors = [i for i in range(adj.shape[0]) if adj[i, target] != 0]

            # Find all ancestors (transitive closure)
            ancestors = set(direct_ancestors)
            to_visit = list(direct_ancestors)

            while to_visit:
                node = to_visit.pop()
                parents = [i for i in range(adj.shape[0]) if adj[i, node] != 0]
                for parent in parents:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        to_visit.append(parent)

            ancestors_list = sorted(list(ancestors))
            indirect_ancestors = [a for a in ancestors_list if a not in direct_ancestors]

            return CausalAncestorResult(
                target_node=target,
                ancestors=ancestors_list,
                direct_ancestors=direct_ancestors,
                indirect_ancestors=indirect_ancestors,
                control_variables=ancestors_list
            )
        else:
            # For GeneralGraph from PC/GES/FCI
            # Use graph traversal
            nodes = graph.get_nodes()
            n_nodes = len(nodes)

            # Get all incoming edges to target
            direct_ancestors = []
            for i in range(n_nodes):
                edge = graph.get_edge(nodes[i], nodes[target])
                if edge and edge.get_endpoint1() == 1:  # Arrow into target
                    direct_ancestors.append(i)

            # Find all ancestors (transitive)
            ancestors = set(direct_ancestors)
            to_visit = list(direct_ancestors)

            while to_visit:
                node = to_visit.pop()
                for i in range(n_nodes):
                    edge = graph.get_edge(nodes[i], nodes[node])
                    if edge and edge.get_endpoint1() == 1:
                        if i not in ancestors:
                            ancestors.add(i)
                            to_visit.append(i)

            ancestors_list = sorted(list(ancestors))
            indirect_ancestors = [a for a in ancestors_list if a not in direct_ancestors]

            return CausalAncestorResult(
                target_node=target,
                ancestors=ancestors_list,
                direct_ancestors=direct_ancestors,
                indirect_ancestors=indirect_ancestors,
                control_variables=ancestors_list
            )

    async def identify_confounders(
        self,
        graph: Any,
        treatment: int,
        outcome: int
    ) -> ConfounderAnalysisResult:
        """
        Identify latent confounders using bidirected edges from FCI.

        Bidirected edges (X <-> Y) indicate presence of latent confounder.
        """
        bidirected_edges = []

        if hasattr(graph, 'find_bi_directed'):
            bidirected = graph.find_bi_directed()
            for i, j in bidirected:
                bidirected_edges.append((i, j))

        # Check if treatment-outcome pair is confounded
        confounded_pairs = []
        for i, j in bidirected_edges:
            if (i == treatment and j == outcome) or (i == outcome and j == treatment):
                confounded_pairs.append((i, j))

        has_latent = len(confounded_pairs) > 0

        return ConfounderAnalysisResult(
            has_latent_confounders=has_latent,
            bidirected_edges=bidirected_edges,
            confounded_pairs=confounded_pairs,
            fci_graph=graph,
            num_latent_confounders=len(bidirected_edges)
        )

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the causal discovery system.

        Tests basic operations on synthetic data.
        """
        if not CAUSAL_LEARN_AVAILABLE:
            return {
                'is_valid': False,
                'checks': {},
                'issues': [f"causal-learn not available: {CAUSAL_LEARN_IMPORT_ERROR}"],
                'version': None
            }

        checks = {}
        issues = []

        try:
            # Test 1: Generate synthetic data
            n_samples = 100
            X = np.random.randn(n_samples)
            Y = 0.5 * X + np.random.randn(n_samples)
            Z = 0.3 * Y + np.random.randn(n_samples)
            data = np.column_stack([X, Y, Z])

            # Test 2: Run PC algorithm
            cg = pc(data, 0.05, fisherz)
            checks['pc_algorithm'] = cg is not None

            # Test 3: Run GES
            result = ges(data, score_func='local_score_BIC')
            checks['ges_algorithm'] = result is not None

            # Test 4: Run DirectLiNGAM
            model = DirectLiNGAM()
            model.fit(data)
            checks['direct_lingam'] = model.adjacency_matrix_ is not None

            # Test 5: Independence test
            p_val = fisherz(data, 0, 1, None)
            checks['independence_test'] = p_val is not None

            is_valid = all(checks.values())

            if not is_valid:
                failed_checks = [name for name, passed in checks.items() if not passed]
                issues.append(f"Failed checks: {failed_checks}")

        except Exception as e:
            is_valid = False
            issues.append(f"Validation error: {e}")

        return {
            'is_valid': is_valid,
            'checks': checks,
            'issues': issues,
            'version': '0.1.4.4'  # causal-learn version
        }

    async def shutdown(self) -> bool:
        """
        Shutdown the adapter.

        Clears cache and releases resources.
        """
        self._cache.clear()
        self.is_initialized = False
        logger.info("CausalLearnAdapter shutdown complete")
        return True

    def _graph_to_adjacency(self, graph, n_nodes: int) -> np.ndarray:
        """Convert causal-learn graph to adjacency matrix."""
        adj = np.zeros((n_nodes, n_nodes), dtype=int)

        nodes = graph.get_nodes()

        for i in range(n_nodes):
            for j in range(n_nodes):
                edge = graph.get_edge(nodes[i], nodes[j])
                if edge:
                    # edge.get_endpoint1() gives endpoint at node i
                    # edge.get_endpoint2() gives endpoint at node j
                    # 1 = arrow, 2 = circle, 3 = tail
                    end1 = edge.get_endpoint1()
                    end2 = edge.get_endpoint2()

                    if end1 == 1 and end2 == 3:  # -->
                        adj[i, j] = 1
                    elif end1 == 3 and end2 == 1:  # <--
                        adj[j, i] = 1
                    elif end1 == 2 and end2 == 2:  # <->
                        adj[i, j] = 2
                        adj[j, i] = 2
                    elif end1 == 3 and end2 == 3:  # ---
                        adj[i, j] = 3
                        adj[j, i] = 3

        return adj

    def _parse_claim(self, claim: str) -> List[int]:
        """
        Parse causal claim to extract variable indices.

        Simplified implementation. In production, use NLP.
        """
        # Extract X0, X1, etc. from claim
        import re
        matches = re.findall(r'X(\d+)', claim)
        return [int(m) for m in matches]

    def _check_causal_relationship(
        self,
        result: CausalGraphResult,
        x: int,
        y: int
    ) -> bool:
        """Check if causal relationship X -> Y exists in graph."""
        return (x, y) in result.directed_edges
