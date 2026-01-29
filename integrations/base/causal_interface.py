"""
Base Causal Discovery Interface for OpenEvolve

This module defines the abstract interface that all causal discovery implementations must follow.
It provides a consistent API for causal reasoning and discovery across different backends.

The interface supports:
- Causal structure discovery from observational data
- Causal effect estimation
- Independence testing
- Counterfactual analysis
- Latent confounder identification
- Pre-experiment validation for SOP Generator

Author: Causal-learn Integration Specialist
Version: 1.0.0
Date: 2026-01-02
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime


class CausalMethod(Enum):
    """Supported causal discovery methods."""
    PC = "pc"  # Peter-Clark (constraint-based)
    PC_STABLE = "pc_stable"  # Stable version of PC
    GES = "ges"  # Greedy Equivalence Search (score-based)
    FCI = "fci"  # Fast Causal Inference (latent confounders)
    DIRECT_LINGAM = "direct_lingam"  # DirectLiNGAM (non-Gaussian)
    ICA_LINGAM = "ica_lingam"  # ICA-based LiNGAM
    VAR_LINGAM = "var_lingam"  # Vector Autoregressive LiNGAM (time series)


class IndependenceTest(Enum):
    """Supported conditional independence tests."""
    FISHER_Z = "fisherz"  # Fisher's Z test (Gaussian continuous)
    CHI_SQ = "chisq"  # Chi-square test (discrete)
    G_SQ = "gsq"  # G-square test (discrete)
    KCI = "kci"  # Kernel-based conditional independence (nonlinear)
    MV_FISHER_Z = "mv_fisherz"  # Fisher's Z with missing values


class ScoreFunction(Enum):
    """Supported score functions for score-based methods."""
    BIC = "local_score_BIC"  # Bayesian Information Criterion
    BDEU = "local_score_BDeu"  # BDeu score (discrete)
    CV_GENERAL = "local_score_CV_general"  # Cross-validation score


class EdgeType(Enum):
    """Types of edges in causal graphs."""
    DIRECTED = "-->"  # X --> Y (directed cause)
    UNDIRECTED = "---"  # X --- Y (unknown direction)
    BIDIRECTED = "<->"  # X <-> Y (latent confounder)
    CIRCLE = "-o"  # X -o Y (circled, unknown endpoint)


@dataclass
class CausalGraphResult:
    """
    Result from causal structure discovery.

    Attributes:
        graph: The discovered causal graph (backend-specific format)
        adjacency_matrix: Adjacency matrix representation of the graph
        nodes: List of node names/IDs
        edges: List of edges with their types
        directed_edges: List of directed edges (X -> Y)
        undirected_edges: List of undirected edges (X -- Y)
        bidirected_edges: List of bidirected edges (X <-> Y) indicating latent confounders
        causal_order: Causal ordering (if available)
        confidence_scores: Optional confidence scores for edges
        algorithm_used: Which algorithm was used
        method_parameters: Parameters used in discovery
        timestamp: When the discovery was performed
    """
    graph: Any
    adjacency_matrix: np.ndarray
    nodes: List[str]
    edges: List[Tuple[int, int, EdgeType]]
    directed_edges: List[Tuple[int, int]]
    undirected_edges: List[Tuple[int, int]]
    bidirected_edges: List[Tuple[int, int]]
    causal_order: Optional[List[int]]
    confidence_scores: Optional[Dict[str, float]]
    algorithm_used: str
    method_parameters: Dict[str, Any]
    timestamp: datetime


@dataclass
class CausalEffectResult:
    """
    Result from causal effect estimation.

    Attributes:
        effect_size: Estimated causal effect (treatment on outcome)
        confidence_interval: 95% confidence interval (lower, upper)
        p_value: Statistical significance
        method: Method used for estimation
        is_significant: Whether effect is statistically significant (alpha=0.05)
        confounders: List of identified confounders
        mediators: List of identified mediators
        colliders: List of identified colliders
        sample_size: Number of samples used
        timestamp: When the estimation was performed
    """
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    is_significant: bool
    confounders: List[int]
    mediators: List[int]
    colliders: List[int]
    sample_size: int
    timestamp: datetime


@dataclass
class IndependenceTestResult:
    """
    Result from conditional independence testing.

    Attributes:
        is_independent: Whether X is independent of Y given Z
        p_value: P-value from the test
        test_statistic: Test statistic value
        method: Test method used
        is_significant: Whether independence is significant (alpha=0.05)
    """
    is_independent: bool
    p_value: float
    test_statistic: float
    method: str
    is_significant: bool


@dataclass
class CounterfactualResult:
    """
    Result from counterfactual analysis.

    Attributes:
        predicted_outcome: Predicted outcome under intervention
        actual_outcome: Actual outcome (for validation)
        intervention: Intervention applied
        effect: Difference between counterfactual and actual
        confidence_interval: Confidence interval for prediction
        method: Method used
    """
    predicted_outcome: np.ndarray
    actual_outcome: Optional[np.ndarray]
    intervention: Dict[int, float]
    effect: float
    confidence_interval: Tuple[float, float]
    method: str


@dataclass
class ConfounderAnalysisResult:
    """
    Result from latent confounder analysis.

    Attributes:
        has_latent_confounders: Whether latent confounders were detected
        bidirected_edges: List of bidirected edges (X <-> Y)
        confounded_pairs: List of variable pairs with latent confounders
        fci_graph: FCI output graph (if applicable)
        num_latent_confounders: Estimated number of latent confounders
    """
    has_latent_confounders: bool
    bidirected_edges: List[Tuple[int, int]]
    confounded_pairs: List[Tuple[int, int]]
    fci_graph: Any
    num_latent_confounders: int


@dataclass
class CausalAncestorResult:
    """
    Result from causal ancestor analysis.

    Attributes:
        target_node: The target variable
        ancestors: List of ancestor node indices
        direct_ancestors: Direct causes (parents)
        indirect_ancestors: Indirect causes (grandparents, etc.)
        control_variables: Variables to control for (all ancestors)
    """
    target_node: int
    ancestors: List[int]
    direct_ancestors: List[int]
    indirect_ancestors: List[int]
    control_variables: List[int]


class CausalDiscoveryInterface(ABC):
    """
    Abstract base class for causal discovery implementations.

    This interface defines the contract that all causal discovery adapters must implement,
    ensuring consistency across different backend technologies (causal-learn, etc.).

    The interface provides comprehensive causal reasoning capabilities:
    1. Causal structure discovery from observational data
    2. Causal effect estimation
    3. Conditional independence testing
    4. Counterfactual analysis
    5. Latent confounder identification
    6. Causal ancestor analysis for intervention design
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the causal discovery system with the given configuration.

        Args:
            config: Configuration dictionary containing:
                - default_algorithm: Default algorithm to use
                - default_indep_test: Default independence test
                - default_alpha: Default significance level (default: 0.05)
                - default_score_func: Default score function for score-based methods
                - cache_enabled: Enable result caching
                - performance: Performance settings (timeout, max_workers)

        Returns:
            True if initialization was successful, False otherwise

        Raises:
            ConfigurationError: If configuration is invalid
            ImportError: If causal-learn is not available
        """
        pass

    @abstractmethod
    async def discover_causal_structure(
        self,
        data: Union[np.ndarray, str],
        method: str = "pc",
        **kwargs
    ) -> CausalGraphResult:
        """
        Discover causal structure from observational data.

        This is the main causal discovery method. It supports multiple algorithms
        for different scenarios:
        - PC: Constraint-based, good for continuous Gaussian data
        - GES: Score-based, faster for large datasets
        - FCI: Handles latent confounders
        - LiNGAM family: For non-Gaussian data

        Args:
            data: Observational data as numpy array (n_samples x n_features)
                  or path to data file
            method: Causal discovery method (default: "pc")
            **kwargs: Method-specific parameters:
                - alpha: Significance level (default: 0.05)
                - indep_test: Independence test method (default: "fisherz")
                - score_func: Score function for GES (default: "local_score_BIC")
                - stable: Use stable PC version (default: True)
                - uc_rule: Unshielded collider orientation rule
                - uc_priority: Unshielded collider orientation priority

        Returns:
            CausalGraphResult containing the discovered causal graph

        Raises:
            ValidationError: If data is invalid
            DiscoveryError: If discovery fails
        """
        pass

    @abstractmethod
    async def validate_causal_claim(
        self,
        claim: str,
        data: Union[np.ndarray, str],
        evidence: Optional[Dict[str, Any]] = None,
        method: str = "direct_lingam"
    ) -> Dict[str, Any]:
        """
        Validate a causal claim using causal discovery.

        This method distinguishes correlation from causation by testing whether
        the claimed causal relationship is supported by the data.

        Args:
            claim: Causal claim to validate (e.g., "X causes Y")
            data: Observational data to test claim against
            evidence: Optional additional evidence
            method: Method to use for validation

        Returns:
            Dictionary containing:
                - is_valid: Whether the claim is supported
                - confidence: Confidence in the validation
                - effect_size: Estimated causal effect
                - explanation: Explanation of the validation
                - is_causal: True if causal, False if just correlation

        Raises:
            ValidationError: If claim or data is invalid
        """
        pass

    @abstractmethod
    async def estimate_causal_effect(
        self,
        data: Union[np.ndarray, str],
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
        method: str = "direct_lingam"
    ) -> CausalEffectResult:
        """
        Estimate the causal effect of treatment on outcome.

        Args:
            data: Observational data
            treatment: Treatment variable index
            outcome: Outcome variable index
            confounders: List of confounder variable indices
            method: Estimation method

        Returns:
            CausalEffectResult with effect size, confidence interval, etc.

        Raises:
            ValidationError: If variables are invalid
            EstimationError: If estimation fails
        """
        pass

    @abstractmethod
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

        Args:
            data: Data array (n_samples x n_features)
            x: Variable X index
            y: Variable Y index
            z: Conditioning set Z (list of variable indices)
            method: Independence test method

        Returns:
            IndependenceTestResult with test results

        Raises:
            ValidationError: If variables are invalid
            TestError: If test fails
        """
        pass

    @abstractmethod
    async def counterfactual_analysis(
        self,
        data: np.ndarray,
        intervention: Dict[int, float],
        method: str = "lingam"
    ) -> CounterfactualResult:
        """
        Perform counterfactual analysis: predict outcome under intervention.

        Args:
            data: Observational data
            intervention: Intervention as dict {variable_index: new_value}
            method: Method for counterfactual prediction

        Returns:
            CounterfactualResult with predicted outcome

        Raises:
            ValidationError: If intervention is invalid
            PredictionError: If prediction fails
        """
        pass

    @abstractmethod
    async def get_causal_ancestors(
        self,
        graph: Any,
        target: int
    ) -> CausalAncestorResult:
        """
        Get all causal ancestors of a target variable.

        This is used for intervention design: to control for all ancestors
        when estimating effects or designing experiments.

        Args:
            graph: Causal graph to analyze
            target: Target variable index

        Returns:
            CausalAncestorResult with ancestor information

        Raises:
            GraphError: If graph analysis fails
        """
        pass

    @abstractmethod
    async def identify_confounders(
        self,
        graph: Any,
        treatment: int,
        outcome: int
    ) -> ConfounderAnalysisResult:
        """
        Identify latent confounders using FCI algorithm.

        Latent confounders appear as bidirected edges (X <-> Y) in FCI output.

        Args:
            graph: Causal graph to analyze
            treatment: Treatment variable index
            outcome: Outcome variable index

        Returns:
            ConfounderAnalysisResult with confounder information

        Raises:
            AnalysisError: If confounder identification fails
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the causal discovery system state.

        Performs health checks:
        1. Check causal-learn is available
        2. Test basic operations on synthetic data
        3. Verify algorithms are working

        Returns:
            Dictionary containing:
                - is_valid: Overall validation status
                - checks: Individual check results
                - issues: List of any issues found
                - version: causal-learn version

        Raises:
            ValidationError: If validation itself fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the causal discovery system.

        Performs cleanup and releases resources.

        Returns:
            True if shutdown was successful, False otherwise

        Raises:
            ShutdownError: If shutdown fails
        """
        pass


class CausalDiscoveryError(Exception):
    """Base exception for causal discovery operations."""
    pass


class ConfigurationError(CausalDiscoveryError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(CausalDiscoveryError):
    """Raised when data validation fails."""
    pass


class DiscoveryError(CausalDiscoveryError):
    """Raised when causal discovery fails."""
    pass


class EstimationError(CausalDiscoveryError):
    """Raised when causal effect estimation fails."""
    pass


class TestError(CausalDiscoveryError):
    """Raised when independence testing fails."""
    pass


class PredictionError(CausalDiscoveryError):
    """Raised when counterfactual prediction fails."""
    pass


class GraphError(CausalDiscoveryError):
    """Raised when graph operations fail."""
    pass


class AnalysisError(CausalDiscoveryError):
    """Raised when confounder analysis fails."""
    pass


class ShutdownError(CausalDiscoveryError):
    """Raised when shutdown operations fail."""
    pass
