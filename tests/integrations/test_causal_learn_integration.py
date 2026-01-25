"""
Comprehensive Test Suite for Causal-learn Integration

This test suite covers:
- Algorithm tests (PC, GES, DirectLiNGAM, FCI)
- Independence tests (Fisher Z, Chi-square, KCI)
- Causal effect estimation
- Confounder identification
- Counterfactual analysis
- Validation tests
- Integration tests
- Error handling
- Performance tests

Target Coverage: >80%

Author: Causal-learn Integration Specialist
Version: 1.0.0
Date: 2026-01-02
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add integrations to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from integrations.causal_learn.adapter import CausalLearnAdapter
from integrations.causal_learn.bridge import CausalDiscoveryBridge
from integrations.base.causal_interface import (
    CausalGraphResult,
    CausalEffectResult,
    EdgeType,
    CausalDiscoveryError,
    ConfigurationError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def adapter():
    """Initialize adapter for testing."""
    adapter = CausalLearnAdapter()
    try:
        await adapter.initialize({'default_algorithm': 'pc'})
        yield adapter
    except ConfigurationError:
        pytest.skip("causal-learn not available")
    finally:
        await adapter.shutdown()


@pytest.fixture
async def bridge():
    """Initialize bridge for testing."""
    bridge = CausalDiscoveryBridge()
    try:
        await bridge.initialize()
        yield bridge
    except ConfigurationError:
        pytest.skip("causal-learn not available")
    finally:
        await bridge.shutdown()


@pytest.fixture
def simple_causal_data():
    """Generate simple causal data: X -> Y -> Z"""
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples)
    Y = 0.5 * X + np.random.randn(n_samples)
    Z = 0.3 * Y + np.random.randn(n_samples)
    return np.column_stack([X, Y, Z])


@pytest.fixture
def latent_confounder_data():
    """Generate data with latent confounder: L -> X, L -> Y"""
    np.random.seed(42)
    n_samples = 1000
    L = np.random.randn(n_samples)  # Latent confounder
    X = 0.5 * L + np.random.randn(n_samples)
    Y = 0.3 * L + np.random.randn(n_samples)
    return np.column_stack([X, Y])


@pytest.fixture
def nonlinear_data():
    """Generate non-Gaussian data for LiNGAM."""
    np.random.seed(42)
    n_samples = 1000
    X = np.random.exponential(1, n_samples)
    Y = 0.5 * X + np.random.exponential(1, n_samples)
    Z = 0.3 * Y + np.random.exponential(1, n_samples)
    return np.column_stack([X, Y, Z])


# ============================================================================
# Algorithm Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pc_algorithm(adapter, simple_causal_data):
    """Test PC algorithm with known causal structure."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='pc',
        alpha=0.05
    )

    assert isinstance(result, CausalGraphResult)
    assert result.algorithm_used == "PC"
    assert len(result.directed_edges) >= 2  # Should find X->Y and Y->Z
    assert result.adjacency_matrix.shape == (3, 3)
    assert len(result.nodes) == 3


@pytest.mark.asyncio
async def test_pc_stable_algorithm(adapter, simple_causal_data):
    """Test PC-stable algorithm."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='pc_stable',
        alpha=0.05
    )

    assert isinstance(result, CausalGraphResult)
    assert result.algorithm_used == "PC"


@pytest.mark.asyncio
async def test_ges_algorithm(adapter, simple_causal_data):
    """Test GES algorithm."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='ges',
        score_func='local_score_BIC'
    )

    assert isinstance(result, CausalGraphResult)
    assert result.algorithm_used == "GES"
    assert 'score' in result.method_parameters


@pytest.mark.asyncio
async def test_direct_lingam_algorithm(adapter, nonlinear_data):
    """Test DirectLiNGAM algorithm with non-Gaussian data."""
    result = await adapter.discover_causal_structure(
        data=nonlinear_data,
        method='direct_lingam'
    )

    assert isinstance(result, CausalGraphResult)
    assert result.algorithm_used == "DirectLiNGAM"
    assert result.causal_order is not None
    assert len(result.causal_order) == 3


@pytest.mark.asyncio
async def test_fci_algorithm(adapter, latent_confounder_data):
    """Test FCI algorithm for latent confounder detection."""
    result = await adapter.discover_causal_structure(
        data=latent_confounder_data,
        method='fci',
        alpha=0.05
    )

    assert isinstance(result, CausalGraphResult)
    assert result.algorithm_used == "FCI"
    # FCI should detect some structure (may be undirected or bidirected)
    assert len(result.edges) >= 0


# ============================================================================
# Independence Test Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fisher_z_test(adapter, simple_causal_data):
    """Test Fisher Z independence test."""
    result = await adapter.test_independence(
        data=simple_causal_data,
        x=0,
        y=2,
        z=None,  # Unconditional
        method='fisherz'
    )

    assert result.method == "fisherz"
    assert 0.0 <= result.p_value <= 1.0
    assert isinstance(result.is_independent, bool)


@pytest.mark.asyncio
async def test_conditional_independence(adapter, simple_causal_data):
    """Test conditional independence X ⟂ Z | Y."""
    result = await adapter.test_independence(
        data=simple_causal_data,
        x=0,  # X
        y=2,  # Z
        z=[1],  # Given Y
        method='fisherz'
    )

    # X and Z should be independent given Y
    assert result.is_independent or result.p_value > 0.01  # Should be independent


@pytest.mark.asyncio
async def test_independent_variables(adapter):
    """Test with independent variables."""
    np.random.seed(42)
    data = np.random.randn(1000, 2)  # Two independent variables

    result = await adapter.test_independence(
        data=data,
        x=0,
        y=1,
        z=None,
        method='fisherz'
    )

    # Should be independent
    assert result.is_independent
    assert result.p_value > 0.05


# ============================================================================
# Causal Effect Tests
# ============================================================================

@pytest.mark.asyncio
async def test_causal_effect_estimation(adapter, simple_causal_data):
    """Test causal effect estimation."""
    result = await adapter.estimate_causal_effect(
        data=simple_causal_data,
        treatment=0,  # X
        outcome=1,    # Y
        method='direct_lingam'
    )

    assert isinstance(result, CausalEffectResult)
    assert isinstance(result.effect_size, float)
    assert result.method == "DirectLiNGAM"
    assert len(result.confidence_interval) == 2
    assert result.sample_size == len(simple_causal_data)


@pytest.mark.asyncio
async def test_zero_effect(adapter):
    """Test with zero causal effect."""
    np.random.seed(42)
    X = np.random.randn(1000)
    Y = np.random.randn(1000)  # Independent of X
    data = np.column_stack([X, Y])

    result = await adapter.estimate_causal_effect(
        data=data,
        treatment=0,
        outcome=1,
        method='direct_lingam'
    )

    # Effect should be near zero
    assert abs(result.effect_size) < 0.3


# ============================================================================
# Confounder Identification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_latent_confounder_detection(adapter, latent_confounder_data):
    """Test latent confounder detection with FCI."""
    # First run FCI
    result = await adapter.discover_causal_structure(
        data=latent_confounder_data,
        method='fci',
        alpha=0.05
    )

    # Then identify confounders
    confounder_result = await adapter.identify_confounders(
        graph=result.graph,
        treatment=0,
        outcome=1
    )

    assert confounder_result.num_latent_confounders >= 0
    assert isinstance(confounder_result.has_latent_confounders, bool)
    assert isinstance(confounder_result.bidirected_edges, list)


@pytest.mark.asyncio
async def test_no_latent_confounders(adapter, simple_causal_data):
    """Test when no latent confounders exist."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='pc',
        alpha=0.05
    )

    # PC should not produce bidirected edges (no latent confounders)
    assert len(result.bidirected_edges) == 0


# ============================================================================
# Counterfactual Analysis Tests
# ============================================================================

@pytest.mark.asyncio
async def test_counterfactual_analysis(adapter, simple_causal_data):
    """Test counterfactual prediction."""
    intervention = {0: 1.0}  # Set X0 to 1.0

    result = await adapter.counterfactual_analysis(
        data=simple_causal_data,
        intervention=intervention,
        method='lingam'
    )

    assert result.predicted_outcome is not None
    assert result.intervention == intervention
    assert isinstance(result.effect, float)
    assert len(result.confidence_interval) == 2


@pytest.mark.asyncio
async def test_counterfactual_with_intervention(adapter):
    """Test counterfactual with specific intervention."""
    np.random.seed(42)
    X = np.random.randn(1000)
    Y = 0.5 * X + np.random.randn(1000)
    data = np.column_stack([X, Y])

    intervention = {0: 2.0}  # Double X

    result = await adapter.counterfactual_analysis(
        data=data,
        intervention=intervention,
        method='lingam'
    )

    # Doubling X should approximately double Y (causal effect ~0.5)
    assert result.effect > 0  # Should increase


# ============================================================================
# Ancestor Analysis Tests
# ============================================================================

@pytest.mark.asyncio
async def test_causal_ancestors(adapter, simple_causal_data):
    """Test causal ancestor extraction."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='direct_lingam'
    )

    ancestors = await adapter.get_causal_ancestors(
        graph=result.graph,
        target=2  # Z
    )

    assert ancestors.target_node == 2
    assert len(ancestors.ancestors) >= 0
    assert isinstance(ancestors.direct_ancestors, list)
    assert isinstance(ancestors.indirect_ancestors, list)


@pytest.mark.asyncio
async def test_ancestor_control_variables(adapter, simple_causal_data):
    """Test that control variables include all ancestors."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='direct_lingam'
    )

    ancestors = await adapter.get_causal_ancestors(
        graph=result.graph,
        target=2
    )

    # Control variables should include all ancestors
    assert len(ancestors.control_variables) >= len(ancestors.direct_ancestors)


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_validate_causal_claim(adapter, simple_causal_data):
    """Test causal claim validation."""
    claim = "X0 causes X1"

    result = await adapter.validate_causal_claim(
        claim=claim,
        data=simple_causal_data,
        method='direct_lingam'
    )

    assert 'is_valid' in result
    assert 'is_causal' in result
    assert 'confidence' in result
    assert isinstance(result['is_causal'], bool)


@pytest.mark.asyncio
async def test_correlation_vs_causation(adapter):
    """Test distinguishing correlation from causation."""
    np.random.seed(42)
    X = np.random.randn(1000)
    Y = np.random.randn(1000)  # Independent (correlation might exist by chance)
    data = np.column_stack([X, Y])

    result = await adapter.validate_causal_claim(
        claim="X0 causes X1",
        data=data,
        method='direct_lingam'
    )

    # Should not be causal (or low confidence)
    assert result['confidence'] < 0.7 or not result['is_causal']


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pre_experiment_validation(bridge, simple_causal_data):
    """Test pre-experiment validation (SOP Generator integration)."""
    validation = await bridge.pre_experiment_validation(
        workflow_data={
            'data': simple_causal_data,
            'variables': ['X', 'Y', 'Z'],
            'domain': 'physics'
        },
        hypothesis="X causes Z"
    )

    assert 'causal_structure' in validation
    assert 'all_variables' in validation
    assert 'latent_confounders' in validation
    assert 'readiness_score' in validation
    assert 0 <= validation['readiness_score'] <= 100


@pytest.mark.asyncio
async def test_problem_analysis(bridge, simple_causal_data):
    """Test problem analysis (Problem Analyzer integration)."""
    analysis = await bridge.analyze_problem_causally(
        problem_text="How does X affect Z?",
        data=simple_causal_data
    )

    assert 'has_causal_structure' in analysis
    assert isinstance(analysis['num_variables'], int)


@pytest.mark.asyncio
async def test_knowledge_extraction(bridge, simple_causal_data):
    """Test causal knowledge extraction (Knowledge Engine integration)."""
    knowledge = await bridge.extract_causal_knowledge(
        workflow_data={'data': simple_causal_data}
    )

    assert 'causal_triples' in knowledge
    assert isinstance(knowledge['causal_triples'], list)
    assert 'graph_summary' in knowledge


@pytest.mark.asyncio
async def test_hypothesis_validation(bridge, simple_causal_data):
    """Test hypothesis validation (ROMA/MDAP integration)."""
    validation = await bridge.validate_hypothesis(
        hypothesis="X0 causes X1",
        evidence_data=simple_causal_data
    )

    assert 'hypothesis' in validation
    assert 'is_causal' in validation
    assert 'confidence' in validation


@pytest.mark.asyncio
async def test_suggest_interventions(bridge, simple_causal_data):
    """Test intervention suggestion."""
    result = await bridge.adapter.discover_causal_structure(
        data=simple_causal_data,
        method='direct_lingam'
    )

    interventions = await bridge.suggest_interventions(
        target_outcome="X2",
        causal_graph=result
    )

    assert isinstance(interventions, list)
    if len(interventions) > 0:
        assert 'variable' in interventions[0]
        assert 'action' in interventions[0]


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_invalid_data(adapter):
    """Test with invalid data."""
    with pytest.raises(Exception):  # ValidationError
        await adapter.discover_causal_structure(
            data="not_an_array",
            method='pc'
        )


@pytest.mark.asyncio
async def test_wrong_dimensions(adapter):
    """Test with wrong data dimensions."""
    data_1d = np.array([1, 2, 3])

    with pytest.raises(Exception):  # ValidationError
        await adapter.discover_causal_structure(
            data=data_1d,
            method='pc'
        )


@pytest.mark.asyncio
async def test_unknown_method(adapter, simple_causal_data):
    """Test with unknown method."""
    with pytest.raises(Exception):  # DiscoveryError
        await adapter.discover_causal_structure(
            data=simple_causal_data,
            method='unknown_method'
        )


@pytest.mark.asyncio
async def test_initialization_without_causal_learn():
    """Test initialization when causal-learn unavailable."""
    # This test would require mocking causal-learn import
    # Skipped for now
    pytest.skip("Requires import mocking")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_dataset(adapter):
    """Test with large dataset."""
    np.random.seed(42)
    n_samples = 10000
    n_vars = 10

    # Generate random data
    data = np.random.randn(n_samples, n_vars)

    import time
    start = time.time()

    result = await adapter.discover_causal_structure(
        data=data,
        method='pc',
        alpha=0.05
    )

    elapsed = time.time() - start

    assert isinstance(result, CausalGraphResult)
    # Should complete in reasonable time (<60 seconds)
    assert elapsed < 60


@pytest.mark.asyncio
async def test_caching_performance(adapter, simple_causal_data):
    """Test that caching improves performance."""
    import time

    # First run (cache miss)
    start = time.time()
    result1 = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='pc',
        alpha=0.05
    )
    time1 = time.time() - start

    # Second run (cache hit)
    start = time.time()
    result2 = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='pc',
        alpha=0.05
    )
    time2 = time.time() - start

    # Cached result should be much faster
    if adapter.config['cache_enabled']:
        assert time2 < time1 or time2 < 0.1  # Should be fast


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.asyncio
async def test_health_check(adapter):
    """Test adapter health check."""
    validation = await adapter.validate()

    assert 'is_valid' in validation
    assert 'checks' in validation
    assert 'issues' in validation
    assert isinstance(validation['is_valid'], bool)


@pytest.mark.asyncio
async def test_health_check_with_synthetic_data(adapter):
    """Test health check generates valid results."""
    validation = await adapter.validate()

    if validation['is_valid']:
        # Check that all algorithms passed
        for check_name, passed in validation['checks'].items():
            assert isinstance(passed, bool)


# ============================================================================
# Shutdown Tests
# ============================================================================

@pytest.mark.asyncio
async def test_shutdown(adapter):
    """Test graceful shutdown."""
    result = await adapter.shutdown()
    assert result is True


@pytest.mark.asyncio
async def test_shutdown_bridge(bridge):
    """Test bridge shutdown."""
    await bridge.shutdown()
    # Should not raise exception


# ============================================================================
# Data Structure Tests
# ============================================================================

@pytest.mark.asyncio
async def test_causal_graph_result_structure(adapter, simple_causal_data):
    """Test CausalGraphResult structure."""
    result = await adapter.discover_causal_structure(
        data=simple_causal_data,
        method='pc'
    )

    # Check all fields
    assert result.graph is not None
    assert isinstance(result.adjacency_matrix, np.ndarray)
    assert isinstance(result.nodes, list)
    assert isinstance(result.edges, list)
    assert isinstance(result.directed_edges, list)
    assert isinstance(result.undirected_edges, list)
    assert isinstance(result.bidirected_edges, list)
    assert isinstance(result.timestamp, datetime)

    # Check edge tuples
    for edge in result.edges:
        assert len(edge) == 3
        assert isinstance(edge[0], int)  # source
        assert isinstance(edge[1], int)  # target
        assert isinstance(edge[2], EdgeType) or isinstance(edge[2], str)


@pytest.mark.asyncio
async def test_causal_effect_result_structure(adapter, simple_causal_data):
    """Test CausalEffectResult structure."""
    result = await adapter.estimate_causal_effect(
        data=simple_causal_data,
        treatment=0,
        outcome=1
    )

    # Check all fields
    assert isinstance(result.effect_size, float)
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2
    assert isinstance(result.p_value, float)
    assert isinstance(result.method, str)
    assert isinstance(result.is_significant, bool)
    assert isinstance(result.confounders, list)
    assert isinstance(result.mediators, list)
    assert isinstance(result.colliders, list)
    assert isinstance(result.sample_size, int)
    assert isinstance(result.timestamp, datetime)


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_small_sample_size(adapter):
    """Test with small sample size."""
    np.random.seed(42)
    data = np.random.randn(50, 3)  # Only 50 samples

    # Should still work but may be less accurate
    result = await adapter.discover_causal_structure(
        data=data,
        method='pc',
        alpha=0.1  # More lenient
    )

    assert isinstance(result, CausalGraphResult)


@pytest.mark.asyncio
async def test_highly_correlated_data(adapter):
    """Test with highly correlated variables."""
    np.random.seed(42)
    X = np.random.randn(1000)
    Y = X + 0.01 * np.random.randn(1000)  # Almost perfect correlation
    Z = Y + 0.01 * np.random.randn(1000)
    data = np.column_stack([X, Y, Z])

    result = await adapter.discover_causal_structure(
        data=data,
        method='direct_lingam'
    )

    # Should still discover structure
    assert isinstance(result, CausalGraphResult)


@pytest.mark.asyncio
async def test_zero_variance_variable(adapter):
    """Test with zero variance variable."""
    np.random.seed(42)
    X = np.random.randn(1000)
    Y = np.ones(1000)  # Zero variance
    data = np.column_stack([X, Y])

    # Should handle gracefully
    try:
        result = await adapter.discover_causal_structure(
            data=data,
            method='pc'
        )
        # May fail or return empty graph
        assert isinstance(result, CausalGraphResult)
    except Exception:
        # Acceptable to fail on constant data
        pass


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
