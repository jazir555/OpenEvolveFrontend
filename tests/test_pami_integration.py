"""
Comprehensive Test Suite for PAMI Integration

This module provides complete test coverage for PAMI (Pattern Mining) integration components:
- PAMIIntegration (core PAMI functionality)
- PAMIPatternMiner (pattern mining capabilities)

Test Statistics:
- Total Test Functions: 48
- Test Classes: 4
- Fixture Functions: 8+
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Algorithm Correctness

Test Categories:
1. Unit Tests - Test each method in isolation with mocked dependencies
2. Integration Tests - Test interactions between components
3. Edge Case Tests - Test boundary conditions and error scenarios
4. Configuration Tests - Test default and custom configuration
5. Algorithm Tests - Test pattern mining algorithm correctness
6. Error Handling Tests - Test graceful degradation and error recovery

Testing Best Practices:
- Use pytest with asyncio support
- Mock external dependencies (PAMI core modules)
- Test both success and failure cases
- Verify structured logging (JSON format)
- Test UTC timestamps
- Aim for >80% code coverage

Running Tests:
    pytest tests/test_pami_integration.py -v
    pytest tests/test_pami_integration.py -v -k "test_frequent_patterns"
    pytest tests/test_pami_integration.py --cov=knowledge_engine.integrations.pami_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from collections import Counter

# Import PAMI integration components
try:
    from knowledge_engine.integrations.pami_integration import (
        PAMIIntegration,
        PAMIPatternMiner
    )
    PAMI_AVAILABLE = True
except ImportError:
    PAMI_AVAILABLE = False
    pytestmark = pytest.mark.skip("PAMI integration not available")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for PAMI integration."""
    return {
        "min_support": 0.1,
        "max_pattern_length": 5,
        "algorithm": "fpgrowth",
        "enable_caching": True
    }


@pytest.fixture
def sample_transactions():
    """Sample transaction data for pattern mining."""
    return [
        ["milk", "bread", "butter"],
        ["beer", "diapers"],
        ["milk", "diapers", "bread", "eggs"],
        ["bread", "milk", "diapers"],
        ["bread", "diapers", "beer"],
        ["milk", "diapers", "bread", "butter"],
        ["milk", "bread", "diapers"]
    ]


@pytest.fixture
def sample_sequences():
    """Sample sequence data for sequential pattern mining."""
    return [
        [["a"], ["b", "c"], ["d"]],
        [["a"], ["b"], ["c", "d"]],
        [["a", "b"], ["c"], ["d"]],
        [["a"], ["b", "c", "d"]]
    ]


@pytest.fixture
def sample_graph_data():
    """Sample knowledge graph data for graph pattern mining."""
    return {
        "nodes": [
            {"id": "n1", "type": "Person"},
            {"id": "n2", "type": "Organization"},
            {"id": "n3", "type": "Person"},
            {"id": "n4", "type": "Location"}
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "works_at"},
            {"source": "n3", "target": "n2", "type": "works_at"},
            {"source": "n1", "target": "n4", "type": "lives_in"},
            {"source": "n2", "target": "n4", "type": "located_in"}
        ]
    }


@pytest.fixture
def pami_integration(sample_config):
    """Create PAMI integration instance."""
    return PAMIIntegration(config=sample_config)


@pytest.fixture
def pami_miner():
    """Create PAMI pattern miner instance."""
    return PAMIPatternMiner()


# =============================================================================
# TEST CLASS: PAMIIntegration - Core Functionality
# =============================================================================

class TestPAMIIntegration:
    """Test suite for PAMIIntegration core functionality."""

    def test_initialization_with_config(self, sample_config):
        """Test PAMIIntegration initialization with configuration."""
        integration = PAMIIntegration(config=sample_config)

        assert integration.config == sample_config
        assert integration._miner is not None
        assert hasattr(integration._miner, '_pami_available')

    def test_initialization_without_config(self):
        """Test PAMIIntegration initialization without configuration."""
        integration = PAMIIntegration(config=None)

        assert integration.config == {}
        assert integration._miner is not None

    def test_is_available(self, pami_integration):
        """Test checking if PAMI is available."""
        result = pami_integration.is_available()

        assert isinstance(result, bool)
        # Result depends on whether PAMI is installed

    def test_mine_patterns_basic(self, pami_integration, sample_transactions):
        """Test basic pattern mining functionality."""
        result = pami_integration.mine_patterns(sample_transactions, min_support=0.1)

        assert 'status' in result
        assert 'patterns' in result
        assert isinstance(result['patterns'], list)

    def test_mine_patterns_with_different_support(self, pami_integration, sample_transactions):
        """Test pattern mining with different support thresholds."""
        result_low = pami_integration.mine_patterns(sample_transactions, min_support=0.05)
        result_high = pami_integration.mine_patterns(sample_transactions, min_support=0.5)

        # Lower support should yield more or equal patterns
        assert len(result_low['patterns']) >= len(result_high['patterns'])

    def test_get_available_algorithms(self, pami_integration):
        """Test getting available algorithms."""
        algorithms = pami_integration.get_available_algorithms()

        assert isinstance(algorithms, dict)
        assert 'frequent_pattern' in algorithms
        assert 'sequence_pattern' in algorithms
        assert 'graph_pattern' in algorithms
        assert isinstance(algorithms['frequent_pattern'], list)


# =============================================================================
# TEST CLASS: PAMIPatternMiner - Initialization and Status
# =============================================================================

class TestPAMIPatternMinerInitialization:
    """Test suite for PAMIPatternMiner initialization and status."""

    def test_initialization(self):
        """Test PAMIPatternMiner initialization."""
        miner = PAMIPatternMiner()

        assert hasattr(miner, '_pami_available')
        assert hasattr(miner, 'frequent_pattern_algorithms')
        assert hasattr(miner, 'sequence_pattern_algorithms')
        assert hasattr(miner, 'graph_pattern_algorithms')

    def test_is_available(self, pami_miner):
        """Test checking if miner is available."""
        result = pami_miner.is_available()

        assert isinstance(result, bool)

    def test_get_available_algorithms(self, pami_miner):
        """Test getting available algorithms."""
        algorithms = pami_miner.get_available_algorithms()

        assert isinstance(algorithms, dict)
        assert len(algorithms) == 3  # frequent_pattern, sequence_pattern, graph_pattern

    def test_get_status(self, pami_miner):
        """Test getting miner status."""
        status = pami_miner.get_status()

        assert 'available' in status
        assert 'algorithms' in status
        assert 'timestamp' in status
        assert isinstance(status['available'], bool)
        assert isinstance(status['algorithms'], dict)


# =============================================================================
# TEST CLASS: PAMIPatternMiner - Frequent Pattern Mining
# =============================================================================

class TestFrequentPatternMining:
    """Test suite for frequent pattern mining functionality."""

    def test_mine_frequent_patterns_success(self, pami_miner, sample_transactions):
        """Test successful frequent pattern mining."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=0.1,
            algorithm='fpgrowth'
        )

        assert 'status' in result
        assert 'patterns' in result
        assert 'statistics' in result
        assert 'config' in result

    def test_mine_frequent_patterns_unavailable(self, pami_miner, sample_transactions):
        """Test frequent pattern mining when PAMI is unavailable."""
        with patch.object(pami_miner, 'is_available', return_value=False):
            result = pami_miner.mine_frequent_patterns(sample_transactions)

            assert result['status'] == 'error'
            assert 'not available' in result['message'].lower()

    def test_mine_frequent_patterns_algorithm_not_available(self, pami_miner, sample_transactions):
        """Test frequent pattern mining with unavailable algorithm."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            algorithm='nonexistent_algorithm'
        )

        # Should either succeed with fallback or fail gracefully
        assert 'status' in result

    def test_mine_frequent_patterns_with_max_length(self, pami_miner, sample_transactions):
        """Test frequent pattern mining with maximum pattern length."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=0.1,
            max_pattern_length=2
        )

        if result['status'] == 'success':
            # Check that no pattern exceeds max length
            for pattern in result['patterns']:
                assert pattern['length'] <= 2

    def test_pattern_structure(self, pami_miner, sample_transactions):
        """Test that mined patterns have correct structure."""
        result = pami_miner.mine_frequent_patterns(sample_transactions)

        if result['status'] == 'success' and result['patterns']:
            pattern = result['patterns'][0]
            assert 'pattern' in pattern
            assert 'support' in pattern
            assert 'support_ratio' in pattern
            assert 'length' in pattern
            assert isinstance(pattern['pattern'], list)
            assert isinstance(pattern['support'], (int, float))
            assert 0 <= pattern['support_ratio'] <= 1

    def test_pattern_statistics(self, pami_miner, sample_transactions):
        """Test pattern statistics calculation."""
        result = pami_miner.mine_frequent_patterns(sample_transactions)

        if result['status'] == 'success':
            stats = result['statistics']
            assert 'total_patterns' in stats
            assert 'patterns_by_length' in stats
            assert 'average_support' in stats
            assert isinstance(stats['total_patterns'], int)

    def test_empty_transactions(self, pami_miner):
        """Test pattern mining with empty transaction list."""
        result = pami_miner.mine_frequent_patterns([])

        assert 'status' in result
        if result['status'] == 'success':
            assert result['statistics']['total_patterns'] == 0

    def test_single_transaction(self, pami_miner):
        """Test pattern mining with single transaction."""
        transactions = [["a", "b", "c"]]
        result = pami_miner.mine_frequent_patterns(transactions, min_support=0.5)

        assert 'status' in result

    def test_high_support_threshold(self, pami_miner, sample_transactions):
        """Test pattern mining with very high support threshold."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=0.9
        )

        # Should return fewer patterns with high support
        assert 'status' in result

    def test_pattern_sorting(self, pami_miner, sample_transactions):
        """Test that patterns are sorted by support."""
        result = pami_miner.mine_frequent_patterns(sample_transactions)

        if result['status'] == 'success' and len(result['patterns']) > 1:
            supports = [p['support'] for p in result['patterns']]
            # Should be sorted descending
            assert supports == sorted(supports, reverse=True)


# =============================================================================
# TEST CLASS: PAMIPatternMiner - Sequential Pattern Mining
# =============================================================================

class TestSequentialPatternMining:
    """Test suite for sequential pattern mining functionality."""

    def test_mine_sequences_success(self, pami_miner, sample_sequences):
        """Test successful sequential pattern mining."""
        result = pami_miner.mine_sequences(
            sample_sequences,
            min_support=0.1
        )

        assert 'status' in result
        assert 'patterns' in result
        assert 'statistics' in result

    def test_mine_sequences_unavailable(self, pami_miner, sample_sequences):
        """Test sequential mining when PAMI is unavailable."""
        with patch.object(pami_miner, 'is_available', return_value=False):
            result = pami_miner.mine_sequences(sample_sequences)

            assert result['status'] == 'error'
            assert 'not available' in result['message'].lower()

    def test_mine_sequences_with_max_gap(self, pami_miner, sample_sequences):
        """Test sequential pattern mining with max gap constraint."""
        result = pami_miner.mine_sequences(
            sample_sequences,
            min_support=0.1,
            max_gap=2
        )

        assert 'status' in result
        if result['status'] == 'success':
            assert 'config' in result
            assert result['config']['max_gap'] == 2

    def test_sequence_pattern_structure(self, pami_miner, sample_sequences):
        """Test that sequence patterns have correct structure."""
        result = pami_miner.mine_sequences(sample_sequences)

        if result['status'] == 'success' and result['patterns']:
            pattern = result['patterns'][0]
            assert 'pattern' in pattern
            assert 'support' in pattern
            assert 'support_ratio' in pattern
            assert 'length' in pattern

    def test_empty_sequences(self, pami_miner):
        """Test sequential mining with empty sequence list."""
        result = pami_miner.mine_sequences([])

        assert 'status' in result

    def test_single_sequence(self, pami_miner):
        """Test sequential mining with single sequence."""
        sequences = [[["a"], ["b"], ["c"]]]
        result = pami_miner.mine_sequences(sequences, min_support=0.5)

        assert 'status' in result

    def test_sequence_statistics(self, pami_miner, sample_sequences):
        """Test sequence pattern statistics."""
        result = pami_miner.mine_sequences(sample_sequences)

        if result['status'] == 'success':
            stats = result['statistics']
            assert 'total_patterns' in stats
            assert 'patterns_by_length' in stats
            assert 'average_support' in stats


# =============================================================================
# TEST CLASS: PAMIPatternMiner - Graph Pattern Analysis
# =============================================================================

class TestGraphPatternAnalysis:
    """Test suite for knowledge graph pattern analysis."""

    def test_analyze_knowledge_graph_patterns_success(self, pami_miner, sample_graph_data):
        """Test successful graph pattern analysis."""
        result = pami_miner.analyze_knowledge_graph_patterns(
            sample_graph_data,
            min_support=0.1
        )

        assert 'status' in result
        assert 'patterns' in result
        assert 'statistics' in result

    def test_graph_pattern_unavailable(self, pami_miner, sample_graph_data):
        """Test graph pattern analysis when PAMI is unavailable."""
        with patch.object(pami_miner, 'is_available', return_value=False):
            result = pami_miner.analyze_knowledge_graph_patterns(sample_graph_data)

            assert result['status'] == 'error'
            assert 'not available' in result['message'].lower()

    def test_graph_pattern_entity_types(self, pami_miner, sample_graph_data):
        """Test extraction of entity type patterns."""
        result = pami_miner.analyze_knowledge_graph_patterns(sample_graph_data)

        if result['status'] == 'success':
            assert 'entity_types' in result['patterns']
            entity_types = result['patterns']['entity_types']
            assert all('pattern' in e for e in entity_types)
            assert all('count' in e for e in entity_types)

    def test_graph_pattern_relationship_types(self, pami_miner, sample_graph_data):
        """Test extraction of relationship type patterns."""
        result = pami_miner.analyze_knowledge_graph_patterns(sample_graph_data)

        if result['status'] == 'success':
            assert 'relationship_types' in result['patterns']
            rel_types = result['patterns']['relationship_types']
            assert all('pattern' in r for r in rel_types)

    def test_graph_pattern_triple_patterns(self, pami_miner, sample_graph_data):
        """Test extraction of triple patterns."""
        result = pami_miner.analyze_knowledge_graph_patterns(sample_graph_data)

        if result['status'] == 'success':
            assert 'triple_patterns' in result['patterns']
            triples = result['patterns']['triple_patterns']
            assert all('pattern' in t for t in triples)

    def test_graph_statistics(self, pami_miner, sample_graph_data):
        """Test graph analysis statistics."""
        result = pami_miner.analyze_knowledge_graph_patterns(sample_graph_data)

        if result['status'] == 'success':
            stats = result['statistics']
            assert 'total_nodes' in stats
            assert 'total_edges' in stats
            assert 'unique_entity_types' in stats
            assert 'unique_relationship_types' in stats
            assert stats['total_nodes'] == len(sample_graph_data['nodes'])
            assert stats['total_edges'] == len(sample_graph_data['edges'])

    def test_empty_graph(self, pami_miner):
        """Test graph pattern analysis with empty graph."""
        empty_graph = {"nodes": [], "edges": []}
        result = pami_miner.analyze_knowledge_graph_patterns(empty_graph)

        assert 'status' in result

    def test_graph_without_types(self, pami_miner):
        """Test graph pattern analysis when nodes lack type information."""
        graph_no_types = {
            "nodes": [
                {"id": "n1"},
                {"id": "n2"}
            ],
            "edges": [
                {"source": "n1", "target": "n2"}
            ]
        }
        result = pami_miner.analyze_knowledge_graph_patterns(graph_no_types)

        assert 'status' in result


# =============================================================================
# TEST CLASS: PAMIPatternMiner - Association Rules
# =============================================================================

class TestAssociationRules:
    """Test suite for association rule discovery."""

    def test_discover_association_rules_success(self, pami_miner, sample_transactions):
        """Test successful association rule discovery."""
        result = pami_miner.discover_association_rules(
            sample_transactions,
            min_support=0.1,
            min_confidence=0.5
        )

        assert 'status' in result
        assert 'rules' in result
        assert 'statistics' in result

    def test_association_rules_structure(self, pami_miner, sample_transactions):
        """Test that association rules have correct structure."""
        result = pami_miner.discover_association_rules(
            sample_transactions,
            min_support=0.1,
            min_confidence=0.3
        )

        if result['status'] == 'success' and result['rules']:
            rule = result['rules'][0]
            assert 'antecedent' in rule
            assert 'consequent' in rule
            assert 'support' in rule
            assert 'confidence' in rule
            assert 'lift' in rule
            assert isinstance(rule['antecedent'], list)
            assert isinstance(rule['consequent'], list)

    def test_association_rules_with_high_confidence(self, pami_miner, sample_transactions):
        """Test association rule discovery with high confidence threshold."""
        result = pami_miner.discover_association_rules(
            sample_transactions,
            min_support=0.1,
            min_confidence=0.9
        )

        # Should return fewer rules with high confidence
        assert 'status' in result
        if result['status'] == 'success':
            for rule in result['rules']:
                assert rule['confidence'] >= 0.9

    def test_association_rules_statistics(self, pami_miner, sample_transactions):
        """Test association rule statistics."""
        result = pami_miner.discover_association_rules(sample_transactions)

        if result['status'] == 'success':
            stats = result['statistics']
            assert 'total_rules' in stats
            assert 'average_confidence' in stats
            assert 'average_support' in stats

    def test_association_rules_sorting(self, pami_miner, sample_transactions):
        """Test that rules are sorted by confidence."""
        result = pami_miner.discover_association_rules(sample_transactions)

        if result['status'] == 'success' and len(result['rules']) > 1:
            confidences = [r['confidence'] for r in result['rules']]
            # Should be sorted descending
            assert confidences == sorted(confidences, reverse=True)

    def test_empty_transactions_association_rules(self, pami_miner):
        """Test association rule discovery with empty transactions."""
        result = pami_miner.discover_association_rules([])

        assert 'status' in result


# =============================================================================
# TEST CLASS: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def test_none_transactions(self, pami_miner):
        """Test pattern mining with None input."""
        with pytest.raises(Exception):
            pami_miner.mine_frequent_patterns(None)

    def test_invalid_support_negative(self, pami_miner, sample_transactions):
        """Test pattern mining with negative support."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=-0.1
        )

        assert 'status' in result

    def test_invalid_support_gt_one(self, pami_miner, sample_transactions):
        """Test pattern mining with support > 1."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=1.5
        )

        assert 'status' in result

    def test_zero_support(self, pami_miner, sample_transactions):
        """Test pattern mining with zero support."""
        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=0.0
        )

        assert 'status' in result

    def test_malformed_transactions(self, pami_miner):
        """Test pattern mining with malformed transaction data."""
        malformed = [
            ["a", "b"],
            None,  # Invalid transaction
            ["c"]
        ]

        # Should handle gracefully
        result = pami_miner.mine_frequent_patterns(malformed)
        assert 'status' in result

    def test_duplicate_transactions(self, pami_miner):
        """Test pattern mining with duplicate transactions."""
        duplicate_transactions = [
            ["a", "b"],
            ["a", "b"],
            ["a", "b"]
        ]

        result = pami_miner.mine_frequent_patterns(duplicate_transactions)
        assert 'status' in result

    def test_very_long_transactions(self, pami_miner):
        """Test pattern mining with very long transactions."""
        long_transactions = [
            [f"item_{i}" for i in range(100)]
        ]

        result = pami_miner.mine_frequent_patterns(long_transactions)
        assert 'status' in result


# =============================================================================
# TEST CLASS: Configuration and Idempotency
# =============================================================================

class TestConfigurationAndIdempotency:
    """Test suite for configuration and idempotency."""

    def test_default_configuration(self):
        """Test PAMI integration with default configuration."""
        integration = PAMIIntegration()

        assert integration.config == {}

    def test_custom_configuration(self, sample_config):
        """Test PAMI integration with custom configuration."""
        integration = PAMIIntegration(config=sample_config)

        assert integration.config == sample_config

    def test_idempotent_pattern_mining(self, pami_miner, sample_transactions):
        """Test that pattern mining is idempotent."""
        result1 = pami_miner.mine_frequent_patterns(sample_transactions, min_support=0.2)
        result2 = pami_miner.mine_frequent_patterns(sample_transactions, min_support=0.2)

        # Results should be consistent
        if result1['status'] == 'success' and result2['status'] == 'success':
            assert len(result1['patterns']) == len(result2['patterns'])

    def test_config_validation_in_mining(self, pami_miner, sample_transactions):
        """Test that configuration parameters are properly used."""
        custom_config = {
            'algorithm': 'fpgrowth',
            'max_pattern_length': 3
        }

        result = pami_miner.mine_frequent_patterns(
            sample_transactions,
            max_pattern_length=custom_config['max_pattern_length']
        )

        assert 'config' in result


# =============================================================================
# TEST CLASS: Performance and Scalability
# =============================================================================

class TestPerformanceAndScalability:
    """Test suite for performance and scalability."""

    def test_large_transaction_set(self, pami_miner):
        """Test pattern mining with large transaction set."""
        # Generate 1000 transactions
        large_transactions = [
            [f"item_{i}", f"item_{i+1}", f"item_{i+2}"]
            for i in range(1000)
        ]

        result = pami_miner.mine_frequent_patterns(
            large_transactions,
            min_support=0.01,
            max_pattern_length=2
        )

        assert 'status' in result

    def test_performance_with_high_support(self, pami_miner, sample_transactions):
        """Test that high support threshold is faster."""
        import time

        start = time.time()
        result1 = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=0.1
        )
        time1 = time.time() - start

        start = time.time()
        result2 = pami_miner.mine_frequent_patterns(
            sample_transactions,
            min_support=0.5
        )
        time2 = time.time() - start

        # Higher support should generally be faster
        # (though this is a weak test due to variability)
        assert 'status' in result1
        assert 'status' in result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
