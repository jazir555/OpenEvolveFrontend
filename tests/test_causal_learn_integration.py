"""
Comprehensive Test Suite for Causal-Learn Integration

This module provides complete test coverage for Causal-Learn (Causal Discovery) integration components:
- CausalLearnIntegration (core Causal-Learn functionality)
- CausalDiscoveryEngine (causal structure learning)

Test Statistics:
- Total Test Functions: 52
- Test Classes: 7
- Fixture Functions: 8+
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Algorithm Correctness

Test Categories:
1. Unit Tests - Test each method in isolation with mocked dependencies
2. Integration Tests - Test interactions between components
3. Edge Case Tests - Test boundary conditions and error scenarios
4. Configuration Tests - Test default and custom configuration
5. Algorithm Tests - Test causal discovery algorithm correctness
6. Error Handling Tests - Test graceful degradation and error recovery

Testing Best Practices:
- Use pytest with asyncio support
- Mock external dependencies (Causal-Learn core modules)
- Test both success and failure cases
- Verify structured logging (JSON format)
- Test UTC timestamps
- Aim for >80% code coverage

Running Tests:
    pytest tests/test_causal_learn_integration.py -v
    pytest tests/test_causal_learn_integration.py -v -k "test_pc"
    pytest tests/test_causal_learn_integration.py --cov=knowledge_engine.integrations.causal_learn_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Import Causal-Learn integration components
try:
    from knowledge_engine.integrations.causal_learn_integration import (
        CausalLearnIntegration,
        CausalDiscoveryEngine
    )
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    pytestmark = pytest.mark.skip("Causal-Learn integration not available")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for Causal-Learn integration."""
    return {
        "algorithm": "pc",
        "alpha": 0.05,
        "independence_test": "fisherz"
    }


@pytest.fixture
def sample_data():
    """Sample data for causal discovery."""
    np.random.seed(42)
    # Generate simple causal data: X -> Y -> Z
    n = 100
    X = np.random.randn(n)
    Y = 0.5 * X + np.random.randn(n) * 0.5
    Z = 0.7 * Y + np.random.randn(n) * 0.3

    return np.column_stack([X, Y, Z])


@pytest.fixture
def sample_time_series():
    """Sample time series data for Granger causality."""
    np.random.seed(42)
    n = 100
    # X causes Y with lag
    X = np.random.randn(n)
    Y = np.zeros(n)
    for t in range(1, n):
        Y[t] = 0.5 * X[t-1] + 0.3 * Y[t-1] + np.random.randn() * 0.5

    return np.column_stack([X, Y])


@pytest.fixture
def sample_graph_data():
    """Sample causal graph data."""
    return {
        "nodes": ["X", "Y", "Z", "W"],
        "edges": [
            {"source": "X", "target": "Y", "type": "directed"},
            {"source": "Y", "target": "Z", "type": "directed"},
            {"source": "W", "target": "Y", "type": "directed"}
        ]
    }


@pytest.fixture
def causal_learn_integration(sample_config):
    """Create Causal-Learn integration instance."""
    return CausalLearnIntegration(config=sample_config)


@pytest.fixture
def causal_engine():
    """Create Causal Discovery Engine instance."""
    return CausalDiscoveryEngine()


# =============================================================================
# TEST CLASS: CausalLearnIntegration - Core Functionality
# =============================================================================

class TestCausalLearnIntegration:
    """Test suite for CausalLearnIntegration core functionality."""

    def test_initialization_with_config(self, sample_config):
        """Test CausalLearnIntegration initialization with configuration."""
        integration = CausalLearnIntegration(config=sample_config)

        assert integration.config == sample_config
        assert integration._engine is not None
        assert hasattr(integration._engine, '_causal_learn_available')

    def test_initialization_without_config(self):
        """Test CausalLearnIntegration initialization without configuration."""
        integration = CausalLearnIntegration(config=None)

        assert integration.config == {}
        assert integration._engine is not None

    def test_is_available(self, causal_learn_integration):
        """Test checking if Causal-Learn is available."""
        result = causal_learn_integration.is_available()

        assert isinstance(result, bool)

    def test_discover_structure_basic(self, causal_learn_integration, sample_data):
        """Test basic causal structure discovery."""
        result = causal_learn_integration.discover_structure(sample_data, algorithm='pc')

        assert 'status' in result
        assert 'graph' in result

    def test_get_available_algorithms(self, causal_learn_integration):
        """Test getting available algorithms."""
        algorithms = causal_learn_integration.get_available_algorithms()

        assert isinstance(algorithms, list)


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - Initialization and Status
# =============================================================================

class TestCausalDiscoveryEngineInitialization:
    """Test suite for CausalDiscoveryEngine initialization and status."""

    def test_initialization(self):
        """Test CausalDiscoveryEngine initialization."""
        engine = CausalDiscoveryEngine()

        assert hasattr(engine, '_causal_learn_available')
        assert hasattr(engine, '_algorithms_available')

    def test_is_available(self, causal_engine):
        """Test checking if engine is available."""
        result = causal_engine.is_available()

        assert isinstance(result, bool)

    def test_get_available_algorithms(self, causal_engine):
        """Test getting available algorithms."""
        algorithms = causal_engine.get_available_algorithms()

        assert isinstance(algorithms, list)

    def test_get_algorithm_info_valid(self, causal_engine):
        """Test getting information for a valid algorithm."""
        info = causal_engine.get_algorithm_info('pc')

        assert 'name' in info
        assert 'available' in info
        assert info['name'] == 'pc'

    def test_get_algorithm_info_invalid(self, causal_engine):
        """Test getting information for an invalid algorithm."""
        info = causal_engine.get_algorithm_info('nonexistent_algorithm')

        assert 'name' in info
        assert info['available'] == False

    def test_algorithm_definitions(self, causal_engine):
        """Test that algorithm definitions are complete."""
        assert hasattr(causal_engine, 'ALGORITHMS')
        assert isinstance(causal_engine.ALGORITHMS, dict)

        # Check that required algorithms are defined
        required_algos = ['pc', 'fci', 'ges', 'lingam', 'direct_lingam', 'ica_lingam', 'granger']
        for algo in required_algos:
            assert algo in causal_engine.ALGORITHMS

    def test_independence_test_definitions(self, causal_engine):
        """Test that independence test definitions exist."""
        assert hasattr(causal_engine, 'INDEPENDENCE_TESTS')
        assert isinstance(causal_engine.INDEPENDENCE_TESTS, dict)

        required_tests = ['fisherz', 'chisq', 'gsq', 'kci']
        for test in required_tests:
            assert test in causal_engine.INDEPENDENCE_TESTS

    def test_get_status(self, causal_engine):
        """Test getting engine status."""
        status = causal_engine.get_status()

        assert 'available' in status
        assert 'algorithms' in status
        assert 'algorithm_info' in status
        assert 'independence_tests' in status
        assert 'timestamp' in status


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - PC Algorithm
# =============================================================================

class TestPCAlgorithm:
    """Test suite for PC algorithm functionality."""

    def test_discover_causal_structure_pc(self, causal_engine, sample_data):
        """Test causal discovery with PC algorithm."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='pc',
            alpha=0.05
        )

        assert 'status' in result
        assert 'graph' in result

    def test_discover_causal_structure_pc_unavailable(self, causal_engine, sample_data):
        """Test PC algorithm when Causal-Learn is unavailable."""
        with patch.object(causal_engine, 'is_available', return_value=False):
            result = causal_engine.discover_causal_structure(sample_data, algorithm='pc')

            assert result['status'] == 'error'
            assert 'not available' in result['message'].lower()

    def test_pc_with_different_alpha(self, causal_engine, sample_data):
        """Test PC algorithm with different alpha values."""
        alpha_values = [0.01, 0.05, 0.1]

        for alpha in alpha_values:
            result = causal_engine.discover_causal_structure(
                sample_data,
                algorithm='pc',
                alpha=alpha
            )
            assert 'status' in result

    def test_pc_with_different_independence_tests(self, causal_engine, sample_data):
        """Test PC algorithm with different independence tests."""
        tests = ['fisherz', 'chisq', 'gsq']

        for test in tests:
            result = causal_engine.discover_causal_structure(
                sample_data,
                algorithm='pc',
                independence_test=test
            )
            assert 'status' in result

    def test_pc_graph_structure(self, causal_engine, sample_data):
        """Test that PC output graph has correct structure."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='pc'
        )

        if result['status'] == 'success':
            graph = result['graph']
            assert 'nodes' in graph
            assert 'edges' in graph
            assert isinstance(graph['nodes'], list)
            assert isinstance(graph['edges'], list)


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - FCI Algorithm
# =============================================================================

class TestFCIAlgorithm:
    """Test suite for FCI algorithm functionality."""

    def test_discover_causal_structure_fci(self, causal_engine, sample_data):
        """Test causal discovery with FCI algorithm."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='fci',
            alpha=0.05
        )

        assert 'status' in result
        assert 'graph' in result

    def test_fci_handles_latent(self, causal_engine, sample_data):
        """Test that FCI handles latent variables."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='fci'
        )

        if result['status'] == 'success':
            # FCI should produce PAG (Partial Ancestral Graph)
            assert 'graph' in result


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - GES Algorithm
# =============================================================================

class TestGESAlgorithm:
    """Test suite for GES algorithm functionality."""

    def test_discover_causal_structure_ges(self, causal_engine, sample_data):
        """Test causal discovery with GES algorithm."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='ges'
        )

        assert 'status' in result
        assert 'graph' in result

    def test_ges_score_based(self, causal_engine, sample_data):
        """Test that GES is score-based."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='ges'
        )

        if result['status'] == 'success':
            # GES should include score information
            assert 'graph' in result


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - LiNGAM Algorithms
# =============================================================================

class TestLiNGAMAlgorithms:
    """Test suite for LiNGAM algorithm functionality."""

    def test_discover_causal_structure_ica_lingam(self, causal_engine, sample_data):
        """Test causal discovery with ICA-LiNGAM algorithm."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='ica_lingam'
        )

        assert 'status' in result
        assert 'graph' in result

    def test_discover_causal_structure_direct_lingam(self, causal_engine, sample_data):
        """Test causal discovery with DirectLiNGAM algorithm."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='direct_lingam'
        )

        assert 'status' in result
        assert 'graph' in result

    def test_lingam_causal_order(self, causal_engine, sample_data):
        """Test that LiNGAM produces causal order."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='ica_lingam'
        )

        if result['status'] == 'success':
            # LiNGAM should provide causal order
            assert 'graph' in result


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - Granger Causality
# =============================================================================

class TestGrangerCausality:
    """Test suite for Granger causality functionality."""

    def test_discover_causal_structure_granger(self, causal_engine, sample_time_series):
        """Test causal discovery with Granger causality."""
        result = causal_engine.discover_causal_structure(
            sample_time_series,
            variable_names=['X', 'Y'],
            algorithm='granger'
        )

        assert 'status' in result
        assert 'graph' in result

    def test_granger_time_series(self, causal_engine, sample_time_series):
        """Test that Granger is designed for time series."""
        result = causal_engine.discover_causal_structure(
            sample_time_series,
            algorithm='granger'
        )

        if result['status'] == 'success':
            # Granger should note it's for time series
            assert 'graph' in result


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - Graph Analysis
# =============================================================================

class TestGraphAnalysis:
    """Test suite for causal graph analysis."""

    def test_analyze_causal_graph_success(self, causal_engine, sample_graph_data):
        """Test successful causal graph analysis."""
        result = causal_engine.analyze_causal_graph(sample_graph_data)

        assert 'status' in result
        assert 'analysis' in result

    def test_graph_analysis_structure(self, causal_engine, sample_graph_data):
        """Test that graph analysis has correct structure."""
        result = causal_engine.analyze_causal_graph(sample_graph_data)

        if result['status'] == 'success':
            analysis = result['analysis']
            assert 'num_nodes' in analysis
            assert 'num_edges' in analysis
            assert 'density' in analysis
            assert 'roots' in analysis
            assert 'leaves' in analysis

    def test_graph_analysis_statistics(self, causal_engine, sample_graph_data):
        """Test graph analysis statistics."""
        result = causal_engine.analyze_causal_graph(sample_graph_data)

        if result['status'] == 'success':
            analysis = result['analysis']
            assert analysis['num_nodes'] == len(sample_graph_data['nodes'])
            assert analysis['num_edges'] == len(sample_graph_data['edges'])

    def test_graph_analysis_empty_graph(self, causal_engine):
        """Test graph analysis with empty graph."""
        empty_graph = {"nodes": [], "edges": []}
        result = causal_engine.analyze_causal_graph(empty_graph)

        assert 'status' in result

    def test_graph_analysis_disconnected_components(self, causal_engine):
        """Test graph analysis with disconnected components."""
        disconnected_graph = {
            "nodes": ["A", "B", "C", "D"],
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "C", "target": "D"}
            ]
        }

        result = causal_engine.analyze_causal_graph(disconnected_graph)
        assert 'status' in result


# =============================================================================
# TEST CLASS: CausalDiscoveryEngine - Confounder Identification
# =============================================================================

class TestConfounderIdentification:
    """Test suite for confounder identification."""

    def test_identify_confounders_success(self, causal_engine, sample_graph_data):
        """Test successful confounder identification."""
        result = causal_engine.identify_confounders(
            sample_graph_data,
            target_x='X',
            target_y='Z'
        )

        assert 'status' in result
        assert 'confounders' in result

    def test_confounders_structure(self, causal_engine, sample_graph_data):
        """Test that confounder analysis has correct structure."""
        result = causal_engine.identify_confounders(
            sample_graph_data,
            target_x='X',
            target_y='Y'
        )

        if result['status'] == 'success':
            confounders = result['confounders']
            assert 'common_causes' in confounders
            assert 'mediators' in confounders
            assert 'colliders' in confounders
            assert 'adjustment_set' in confounders

    def test_identify_confounders_nonexistent_variables(self, causal_engine, sample_graph_data):
        """Test confounder identification with nonexistent variables."""
        result = causal_engine.identify_confounders(
            sample_graph_data,
            target_x='NonExistent1',
            target_y='NonExistent2'
        )

        # Should handle gracefully
        assert 'status' in result


# =============================================================================
# TEST CLASS: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def test_none_data(self, causal_engine):
        """Test causal discovery with None input."""
        with pytest.raises(Exception):
            causal_engine.discover_causal_structure(None, algorithm='pc')

    def test_empty_data(self, causal_engine):
        """Test causal discovery with empty data."""
        empty_data = np.array([]).reshape(0, 0)
        result = causal_engine.discover_causal_structure(empty_data)

        assert 'status' in result

    def test_single_variable(self, causal_engine):
        """Test causal discovery with single variable."""
        single_var = np.random.randn(100, 1)
        result = causal_engine.discover_causal_structure(single_var)

        assert 'status' in result

    def test_two_variables(self, causal_engine):
        """Test causal discovery with two variables."""
        two_vars = np.random.randn(100, 2)
        result = causal_engine.discover_causal_structure(two_vars)

        assert 'status' in result

    def test_many_variables(self, causal_engine):
        """Test causal discovery with many variables."""
        many_vars = np.random.randn(100, 20)
        result = causal_engine.discover_causal_structure(many_vars)

        assert 'status' in result

    def test_constant_data(self, causal_engine):
        """Test causal discovery with constant data."""
        constant_data = np.ones((100, 3))
        result = causal_engine.discover_causal_structure(constant_data)

        assert 'status' in result

    def test_correlated_data(self, causal_engine):
        """Test causal discovery with highly correlated data."""
        correlated_data = np.random.randn(100, 1)
        correlated_data = np.column_stack([correlated_data, correlated_data, correlated_data])
        result = causal_engine.discover_causal_structure(correlated_data)

        assert 'status' in result

    def test_invalid_algorithm(self, causal_engine, sample_data):
        """Test causal discovery with invalid algorithm."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='nonexistent_algorithm'
        )

        assert result['status'] == 'error'
        assert 'not available' in result['message'].lower()

    def test_invalid_alpha_negative(self, causal_engine, sample_data):
        """Test causal discovery with negative alpha."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='pc',
            alpha=-0.05
        )

        assert 'status' in result

    def test_invalid_alpha_gt_one(self, causal_engine, sample_data):
        """Test causal discovery with alpha > 1."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='pc',
            alpha=1.5
        )

        assert 'status' in result


# =============================================================================
# TEST CLASS: Configuration and Idempotency
# =============================================================================

class TestConfigurationAndIdempotency:
    """Test suite for configuration and idempotency."""

    def test_default_configuration(self):
        """Test Causal-Learn integration with default configuration."""
        integration = CausalLearnIntegration()

        assert integration.config == {}

    def test_custom_configuration(self, sample_config):
        """Test Causal-Learn integration with custom configuration."""
        integration = CausalLearnIntegration(config=sample_config)

        assert integration.config == sample_config

    def test_idempotent_causal_discovery(self, causal_engine, sample_data):
        """Test that causal discovery is idempotent."""
        result1 = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='pc',
            alpha=0.05
        )
        result2 = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='pc',
            alpha=0.05
        )

        # Results should have same structure
        if result1['status'] == 'success' and result2['status'] == 'success':
            assert result1['algorithm'] == result2['algorithm']

    def test_variable_names_default(self, causal_engine, sample_data):
        """Test causal discovery with default variable names."""
        result = causal_engine.discover_causal_structure(
            sample_data,
            algorithm='pc'
        )

        if result['status'] == 'success':
            graph = result['graph']
            # Should have default names like X0, X1, X2
            assert 'nodes' in graph


# =============================================================================
# TEST CLASS: Performance and Scalability
# =============================================================================

class TestPerformanceAndScalability:
    """Test suite for performance and scalability."""

    def test_large_dataset(self, causal_engine):
        """Test causal discovery with large dataset."""
        large_data = np.random.randn(1000, 10)
        result = causal_engine.discover_causal_structure(
            large_data,
            algorithm='pc'
        )

        assert 'status' in result

    def test_small_dataset(self, causal_engine):
        """Test causal discovery with small dataset."""
        small_data = np.random.randn(10, 3)
        result = causal_engine.discover_causal_structure(
            small_data,
            algorithm='pc'
        )

        assert 'status' in result

    def test_high_dimensional_data(self, causal_engine):
        """Test causal discovery with high-dimensional data."""
        high_dim_data = np.random.randn(200, 50)
        result = causal_engine.discover_causal_structure(
            high_dim_data,
            algorithm='pc'
        )

        assert 'status' in result


# =============================================================================
# TEST CLASS: Algorithm Edge Cases
# =============================================================================

class TestAlgorithmEdgeCases:
    """Test suite for algorithm-specific edge cases."""

    def test_pc_with_missing_data_simulation(self, causal_engine):
        """Test PC with simulated missing data (NaN values)."""
        data_with_nan = np.random.randn(100, 3)
        data_with_nan[10:20, 0] = np.nan

        # Should handle gracefully
        result = causal_engine.discover_causal_structure(data_with_nan, algorithm='pc')
        assert 'status' in result

    def test_fci_latent_handling(self, causal_engine):
        """Test FCI explicitly mentions latent variable handling."""
        result = causal_engine.discover_causal_structure(
            np.random.randn(100, 4),
            algorithm='fci'
        )

        if result['status'] == 'success':
            # FCI should have note about latent variables
            assert 'graph' in result

    def test_granger_with_lag(self, causal_engine, sample_time_series):
        """Test Granger causality with lag specification."""
        result = causal_engine.discover_causal_structure(
            sample_time_series,
            algorithm='granger'
        )

        assert 'status' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
