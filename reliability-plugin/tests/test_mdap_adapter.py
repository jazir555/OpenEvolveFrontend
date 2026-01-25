"""
Comprehensive test suite for MDAP Reliability Adapter.

This test suite covers all functionality of the MDAP adapter including:
- Adapter initialization and dual-mode operation
- Core integration mode testing
- Vote validation functionality
- Solve with validation functionality
- Statistics tracking
- Health checks
- Error handling and graceful degradation
- Integration with other reliability components
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import classes to test (with proper error handling for missing modules)
try:
    from reliability.mdap_adapter import (
        MDAPReliabilityAdapter,
        VoteValidationResult,
        MDAPSolveResult,
        RemediationStrategy,
        create_mdap_adapter,
        get_default_adapter
    )
    MDAP_ADAPTER_AVAILABLE = True
except ImportError:
    MDAP_ADAPTER_AVAILABLE = False
    # Create mock classes for testing when MDAP is not available
    class VoteValidationResult:
        def __init__(self, is_valid: bool, vote: Any, failures: List[str] = None,
                     remediated: bool = False, original_vote: Any = None,
                     validator_name: str = None, timestamp: str = None):
            self.is_valid = is_valid
            self.vote = vote
            self.failures = failures or []
            self.remediated = remediated
            self.original_vote = original_vote
            self.validator_name = validator_name
            self.timestamp = timestamp or datetime.utcnow().isoformat()

    class MDAPSolveResult:
        def __init__(self, success: bool, result: Dict = None, error: str = None,
                     statistics: Dict = None, validation_failures: List[str] = None,
                     correlation_id: str = None, timestamp: str = None,
                     layers_used: List[str] = None, method: str = None):
            self.success = success
            self.result = result
            self.error = error
            self.statistics = statistics or {}
            self.validation_failures = validation_failures or []
            self.correlation_id = correlation_id
            self.timestamp = timestamp or datetime.utcnow().isoformat()
            self.layers_used = layers_used or []
            self.method = method

    class RemediationStrategy:
        REASK = "reask"
        FIX = "fix"
        FILTER = "filter"
        REFRAIN = "refrain"
        EXCEPTION = "exception"

    class MDAPReliabilityAdapter:
        def __init__(self, config: Optional[Dict] = None):
            pass

    def create_mdap_adapter(config: Optional[Dict] = None):
        return MDAPReliabilityAdapter(config)

    def get_default_adapter():
        return MDAPReliabilityAdapter()


# Mock vote for testing
@dataclass
class MockVote:
    content: str
    score: float
    metadata: Dict = None


class TestMDAPAdapterInitialization:
    """Test suite for MDAP adapter initialization."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_initialization_with_config(self, mock_config):
        """Test adapter initialization with configuration."""
        config = mock_config
        adapter = MDAPReliabilityAdapter(config)

        assert adapter is not None
        assert adapter.config == config
        # Check that adapters are initialized
        assert hasattr(adapter, 'lmql_adapter')
        assert hasattr(adapter, 'guardrails_adapter')

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_initialization_without_config(self):
        """Test adapter initialization without configuration."""
        adapter = MDAPReliabilityAdapter()

        assert adapter is not None
        assert adapter.config is None  # Should handle missing config

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_initialization_with_custom_adapters(self, mock_lmql_adapter, mock_guardrails_adapter):
        """Test adapter initialization with custom adapters."""
        adapter = MDAPReliabilityAdapter(
            lmql_adapter=mock_lmql_adapter,
            guardrails_adapter=mock_guardrails_adapter
        )

        assert adapter is not None
        assert adapter.lmql_adapter == mock_lmql_adapter
        assert adapter.guardrails_adapter == mock_guardrails_adapter

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_initialization_status_attributes(self, mock_config):
        """Test initialization of status attributes."""
        adapter = MDAPReliabilityAdapter(mock_config)

        # Check that all status attributes are initialized
        assert hasattr(adapter, 'mdap_core_available')
        assert hasattr(adapter, 'mdap_mcp_available')
        assert hasattr(adapter, 'mdap_available')
        assert hasattr(adapter, 'lmql_enabled')
        assert hasattr(adapter, 'guardrails_enabled')

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_initialization_logging(self, mock_config):
        """Test initialization logs appropriate events."""
        with patch('reliability.mdap_adapter.logger') as mock_logger:
            adapter = MDAPReliabilityAdapter(mock_config)

            # Check that initialization was logged
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "mdap_reliability_adapter_initialized" in call_args.get("event", "")

    def test_initialization_when_mdap_unavailable(self):
        """Test adapter initialization when MDAP is completely unavailable."""
        with patch.dict('sys.modules', {'mdap_core': None, 'mdap_mcp_tools': None}):
            adapter = MDAPReliabilityAdapter()

            assert adapter is not None
            assert adapter.mdap_available is False

    @pytest.mark.parametrize("lmql_enabled", [True, False])
    @pytest.mark.parametrize("guardrails_enabled", [True, False])
    def test_initialization_with_different_layer_combinations(self, lmql_enabled, guardrails_enabled):
        """Test initialization with different combinations of layers."""
        with patch('reliability.mdap_adapter.MDAP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            assert adapter is not None
            # Layers are initialized based on actual availability


class TestDualModeOperation:
    """Test suite for dual-mode operation (core integration vs MCP fallback)."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_core_integration_mode(self, mock_mdap_core):
        """Test core integration mode when available."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            assert adapter.mdap_core_available is True
            assert adapter.mdap_mcp_available is True
            assert adapter.mdap_available is True

            # Should prefer core integration
            status = adapter.get_status()
            assert "core_preferred_with_mcp_fallback" in status.get("execution_mode", "")

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_mcp_fallback_mode(self, mock_mdap_core):
        """Test MCP fallback mode when core is unavailable."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', False), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            assert adapter.mdap_core_available is False
            assert adapter.mdap_mcp_available is True
            assert adapter.mdap_available is True

            # Should use MCP mode
            status = adapter.get_status()
            assert "mcp_only" in status.get("execution_mode", "")

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_unavailable_mode(self):
        """Test when both core and MCP are unavailable."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', False), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', False):
            adapter = MDAPReliabilityAdapter()

            assert adapter.mdap_core_available is False
            assert adapter.mdap_mcp_available is False
            assert adapter.mdap_available is False

            # Should be unavailable
            status = adapter.get_status()
            assert status.get("execution_mode") == "unavailable"

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_mode_switching(self, mock_mdap_core):
        """Test adapter mode switching based on availability."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Initially should prefer core
            status = adapter.get_status()
            assert "core_preferred_with_mcp_fallback" in status.get("execution_mode", "")

            # Simulate core becoming unavailable
            adapter.mdap_core_available = False
            status = adapter.get_status()
            assert "mcp_only" in status.get("execution_mode", "")


class TestVoteValidation:
    """Test suite for vote validation functionality."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_success(self, mock_mdap_core):
        """Test successful vote validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            vote = MockVote("Good vote content", 0.9)
            validation_result = adapter._validate_vote(vote, ["toxic_language"])

            assert validation_result.is_valid is True
            assert validation_result.vote == vote
            assert len(validation_result.failures) == 0
            assert validation_result.remediated is False

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_with_failures(self, mock_mdap_core):
        """Test vote validation with failures."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            vote = MockVote("Bad vote content", 0.9)
            validation_result = adapter._validate_vote(vote, ["toxic_language"])

            assert validation_result.is_valid is False
            assert validation_result.vote == vote
            assert len(validation_result.failures) > 0
            assert validation_result.remediated is False

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_with_remediation(self, mock_mdap_core):
        """Test vote validation with remediation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            vote = MockVote("Bad vote content", 0.9)
            validation_result = adapter._validate_vote(
                vote,
                ["toxic_language"],
                remediate=True
            )

            assert validation_result.is_valid is True  # After remediation
            assert validation_result.remediated is True
            assert validation_result.original_vote == vote

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_score_threshold(self, mock_mdap_core):
        """Test vote validation with score threshold."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Test vote above threshold
            high_vote = MockVote("Good vote", 0.9)
            result_high = adapter._validate_vote(high_vote, [], score_threshold=0.8)
            assert result_high.is_valid is True

            # Test vote below threshold
            low_vote = MockVote("Low score vote", 0.7)
            result_low = adapter._validate_vote(low_vote, [], score_threshold=0.8)
            assert result_low.is_valid is False

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_multiple_validators(self, mock_mdap_core):
        """Test vote validation with multiple validators."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            vote = MockVote("Vote content", 0.9)
            validators = ["toxic_language", "pii_detection", "length"]

            validation_result = adapter._validate_vote(vote, validators)

            assert validation_result is not None

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_exception_handling(self, mock_mdap_core):
        """Test vote validation exception handling."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure validator to raise exception
            adapter.mdap_core.VoteValidator.return_value.validate_vote.side_effect = Exception("Validation error")

            vote = MockVote("Test vote", 0.9)
            validation_result = adapter._validate_vote(vote, ["toxic_language"])

            # Should handle exception gracefully and return invalid result
            assert validation_result.is_valid is False
            assert "Validation error" in str(validation_result.failures)

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_validate_vote_edge_cases(self, mock_mdap_core):
        """Test vote validation edge cases."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Test with empty vote content
            empty_vote = MockVote("", 0.9)
            result_empty = adapter._validate_vote(empty_vote, ["length"])
            assert result_empty is not None

            # Test with None vote
            result_none = adapter._validate_vote(None, ["toxic_language"])
            assert result_none is not None

            # Test with high score
            high_score_vote = MockVote("High score content", 1.0)
            result_high = adapter._validate_vote(high_score_vote, ["toxic_language"])
            assert result_high.is_valid is True

            # Test with low score
            low_score_vote = MockVote("Low score content", 0.1)
            result_low = adapter._validate_vote(low_score_vote, [], score_threshold=0.5)
            assert result_low.is_valid is False


class TestCoreIntegration:
    """Test suite for core integration functionality."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_core_integration(self, mock_mdap_core):
        """Test solving with core integration."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[
                    MockVote("Vote 1", 0.9),
                    MockVote("Vote 2", 0.8)
                ],
                final_decision="Test decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                correlation_id="test_corr_id"
            )

            # Should succeed and use core integration
            assert result["success"] is True
            assert "mdap_core" in result.get("layers_used", [])
            assert len(result.get("statistics", {})) > 0

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_core_integration_with_vote_validation(self, mock_mdap_core):
        """Test core integration with vote validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[
                    MockVote("Good vote", 0.9),
                    MockVote("Bad vote", 0.7)  # Will be filtered out
                ],
                final_decision="Valid decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Configure vote validator
            mock_validator = Mock()
            mock_validator.validate_vote.side_effect = [
                Mock(is_valid=True, vote=MockVote("Good vote", 0.9), failures=[]),
                Mock(is_valid=False, vote=MockVote("Bad vote", 0.7),
                     failures=[{"validator": "score_threshold"}])
            ]
            mock_mdap_core.VoteValidator.return_value = mock_validator

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                correlation_id="test_corr_id"
            )

            # Should succeed with only valid votes
            assert result["success"] is True
            assert "vote_validation" in result.get("layers_used", [])

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_core_integration_with_lmql_constraints(self, mock_lmql_adapter, mock_mdap_core):
        """Test core integration with LMQL constraints."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Constrained vote", 0.9)],
                final_decision="Constrained decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                correlation_id="test_corr_id",
                use_lmql_constraints=True
            )

            # Should use LMQL constraints
            assert result["success"] is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_core_integration_with_guardrails_validation(self, mock_guardrails_adapter, mock_mdap_core):
        """Test core integration with Guardrails validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Safe vote", 0.9)],
                final_decision="Safe decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Configure guardrails validation
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True, failures=[]
            )

            result = adapter._solve_with_core_integration(
                task="Safe task",
                max_votes=100,
                validation_threshold=0.8,
                correlation_id="test_corr_id",
                enable_input_validation=True
            )

            # Should use input validation
            assert result["success"] is True
            assert "guardrails_input" in result.get("layers_used", [])

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_core_integration_output_validation(self, mock_guardrails_adapter, mock_mdap_core):
        """Test core integration with output validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Valid vote", 0.9)],
                final_decision="Valid decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Configure guardrails output validation
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True, failures=[], remediation_applied=None, output=None
            )

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                correlation_id="test_corr_id",
                enable_output_validation=True
            )

            # Should use output validation
            assert result["success"] is True
            assert "guardrails_output" in result.get("layers_used", [])


class TestMCPFallback:
    """Test suite for MCP fallback functionality."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_mcp_tools(self):
        """Test solving with MCP tools."""
        with patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MCP result
            mock_result = {
                "solution": "MCP solution",
                "votes": [
                    {"content": "Vote 1", "score": 0.9},
                    {"content": "Vote 2", "score": 0.8}
                ],
                "statistics": {"total_votes": 2, "valid_votes": 2}
            }

            with patch('reliability.mdap_adapter.solve_with_mdap') as mock_solve:
                mock_solve.return_value = mock_result

                result = adapter._solve_with_mcp_tools(
                    task="Solve: 2 + 2",
                    max_votes=100,
                    validation_threshold=0.8,
                    correlation_id="test_corr_id"
                )

                # Should succeed
                assert result["success"] is True
                assert result["result"]["solution"] == "MCP solution"
                assert "mdap_mcp" in result.get("layers_used", [])
                assert "vote_validation" in result.get("layers_used", [])

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_mcp_tools_error_handling(self):
        """Test MCP tools error handling."""
        with patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MCP to return error
            with patch('reliability.mdap_adapter.solve_with_mdap') as mock_solve:
                mock_solve.return_value = {"error": "MCP tool error"}

                result = adapter._solve_with_mcp_tools(
                    task="Solve: 2 + 2",
                    max_votes=100,
                    validation_threshold=0.8,
                    correlation_id="test_corr_id"
                )

                # Should fail gracefully
                assert result["success"] is False
                assert result["error"] == "MCP tool error"

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_mcp_tools_exception_handling(self):
        """Test MCP tools exception handling."""
        with patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MCP to raise exception
            with patch('reliability.mdap_adapter.solve_with_mdap') as mock_solve:
                mock_solve.side_effect = Exception("Network error")

                result = adapter._solve_with_mcp_tools(
                    task="Solve: 2 + 2",
                    max_votes=100,
                    validation_threshold=0.8,
                    correlation_id="test_corr_id"
                )

                # Should fail gracefully
                assert result["success"] is False
                assert "Network error" in result["error"]


class TestSolveWithValidation:
    """Test suite for solve_with_validation functionality."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_validation_success(self, mock_mdap_core):
        """Test successful solve with validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Valid vote", 0.9)],
                final_decision="Valid decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8
            )

            # Should succeed
            assert result.success is True
            assert result.result["final_decision"] == "Valid decision"
            assert "mdap_core" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_input_validation(self, mock_guardrails_adapter, mock_mdap_core):
        """Test solve with input validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure input validation to pass
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True, failures=[]
            )

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Valid vote", 0.9)],
                final_decision="Valid decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Safe task",
                max_votes=100,
                validation_threshold=0.8,
                enable_input_validation=True
            )

            # Should succeed and use input validation
            assert result.success is True
            assert "guardrails_input" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_input_validation_failure(self, mock_guardrails_adapter):
        """Test solve with input validation failure."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure input validation to fail
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=False,
                failures=[{"validator": "length", "message": "Too long"}]
            )

            result = adapter.solve_with_validation(
                task="This is a very long task that exceeds the maximum allowed length",
                max_votes=100,
                validation_threshold=0.8,
                enable_input_validation=True
            )

            # Should fail at input validation
            assert result.success is False
            assert "Input validation failed" in result.error
            assert "guardrails_input" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_output_validation(self, mock_guardrails_adapter, mock_mdap_core):
        """Test solve with output validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure output validation
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True,
                failures=[],
                remediation_applied=None,
                output=None
            )

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Valid vote", 0.9)],
                final_decision="Valid decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                enable_output_validation=True
            )

            # Should succeed and use output validation
            assert result.success is True
            assert "guardrails_output" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_vote_filtering(self, mock_mdap_core):
        """Test solve with vote filtering based on threshold."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver with mixed scores
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[
                    MockVote("High score vote", 0.9),  # Should pass
                    MockVote("Low score vote", 0.6),   # Should be filtered
                    MockVote("Medium score vote", 0.8)  # Should pass
                ],
                final_decision="Filtered decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.7,  # Filter below 0.7
                enable_vote_filtering=True
            )

            # Should succeed with filtered votes
            assert result.success is True
            assert "vote_filtering" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_batch_validation(self, mock_mdap_core):
        """Test solve with batch validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver with many votes
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote(f"Vote {i}", 0.8 + i * 0.02) for i in range(20)],
                final_decision="Batch validated decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve complex problem",
                max_votes=100,
                validation_threshold=0.7,
                enable_batch_validation=True
            )

            # Should succeed with batch validation
            assert result.success is True
            assert "batch_validation" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_statistics_tracking(self, mock_mdap_core):
        """Test solve with statistics tracking."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Vote", 0.9)],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Configure statistics tracker
            mock_stats = Mock()
            mock_stats.get_statistics.return_value = {
                "total_votes": 10,
                "valid_votes": 9,
                "invalid_votes": 1,
                "average_score": 0.85
            }
            mock_mdap_core.StatisticsTracker.return_value = mock_stats

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                enable_statistics=True
            )

            # Should include statistics
            assert result.success is True
            assert len(result.statistics) > 0
            assert "total_votes" in result.statistics


class TestStatisticsTracking:
    """Test suite for statistics tracking functionality."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_initialization_statistics(self, mock_mdap_core):
        """Test statistics initialization."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            stats = adapter.get_statistics()

            assert isinstance(stats, dict)
            assert "total_validations" in stats
            assert "successful_validations" in stats
            assert "failed_validations" in stats
            assert "vote_validation_success_rate" in stats
            assert "average_validation_time" in stats

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_statistics_update_on_validation(self, mock_mdap_core):
        """Test statistics update on validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Get initial stats
            initial_stats = adapter.get_statistics()

            # Perform validation
            adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8
            )

            # Check that statistics were updated
            updated_stats = adapter.get_statistics()
            assert updated_stats["total_validations"] > initial_stats["total_validations"]

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_vote_statistics_tracking(self, mock_mdap_core):
        """Test vote-specific statistics tracking."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Vote", 0.9)],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Perform validation
            adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8
            )

            # Check vote statistics
            stats = adapter.get_vote_statistics()
            assert isinstance(stats, dict)
            assert "total_votes_validated" in stats
            assert "vote_validation_success_rate" in stats

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_statistics_reset(self, mock_mdap_core):
        """Test statistics reset functionality."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Perform some validations
            adapter.solve_with_validation("Task 1", 100, 0.8)
            adapter.solve_with_validation("Task 2", 100, 0.8)

            # Reset statistics
            adapter.reset_statistics()

            # Statistics should be reset
            stats = adapter.get_statistics()
            assert stats["total_validations"] == 0
            assert stats["successful_validations"] == 0
            assert stats["failed_validations"] == 0

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_statistics_persistence(self, mock_mdap_core):
        """Test statistics persistence across multiple operations."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Perform several operations
            for i in range(5):
                adapter.solve_with_validation(f"Task {i}", 100, 0.8)

            # Statistics should accumulate
            stats = adapter.get_statistics()
            assert stats["total_validations"] == 5

            # Perform more operations
            for i in range(3):
                adapter.solve_with_validation(f"Task {i}", 100, 0.8)

            # Statistics should continue to accumulate
            updated_stats = adapter.get_statistics()
            assert updated_stats["total_validations"] == 8


class TestHealthChecks:
    """Test suite for health checking functionality."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_get_status(self, mock_mdap_core):
        """Test getting adapter status."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            status = adapter.get_status()

            assert isinstance(status, dict)
            assert "mdap_available" in status
            assert "mdap_core_available" in status
            assert "mdap_mcp_available" in status
            assert "lmql_available" in status
            assert "guardrails_available" in status
            assert "config" in status

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_health_check(self, mock_mdap_core):
        """Test comprehensive health check."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            health = adapter.health_check()

            assert isinstance(health, dict)
            assert "adapter_healthy" in health
            assert "execution_mode" in health
            assert "components" in health
            assert health["components"]["mdap_core"]["healthy"] is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_health_check_unavailable_components(self):
        """Test health check with unavailable components."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', False), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', False):
            adapter = MDAPReliabilityAdapter()

            health = adapter.health_check()

            assert health["adapter_healthy"] is False
            assert health["execution_mode"] == "unavailable"
            assert health["components"]["mdap_core"]["healthy"] is False

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_is_available(self, mock_mdap_core):
        """Test is_available method."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            assert adapter.is_available() is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_is_available_when_unavailable(self):
        """Test is_available method when MDAP is unavailable."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', False), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', False):
            adapter = MDAPReliabilityAdapter()

            assert adapter.is_available() is False


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_empty_task(self, mock_mdap_core):
        """Test solve with empty task."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            result = adapter.solve_with_validation(
                task="",
                max_votes=100,
                validation_threshold=0.8
            )

            # Should handle empty task gracefully
            assert result is not None

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_solve_with_none_parameters(self, mock_mdap_core):
        """Test solve with None parameters."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=None,
                validation_threshold=None
            )

            # Should handle None parameters gracefully
            assert result is not None

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_core_integration_exception_handling(self, mock_mdap_core):
        """Test core integration exception handling."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure core to raise exception
            mock_mdap_core.MDAPSolver.side_effect = Exception("Core exception")

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8
            )

            # Should fail gracefully
            assert result.success is False
            assert "Core exception" in result.error

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_mcp_exception_handling(self):
        """Test MCP exception handling."""
        with patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MCP to raise exception
            with patch('reliability.mdap_adapter.solve_with_mdap') as mock_solve:
                mock_solve.side_effect = Exception("MCP exception")

                result = adapter.solve_with_validation(
                    task="Solve: 2 + 2",
                    max_votes=100,
                    validation_threshold=0.8
                )

                # Should fail gracefully
                assert result.success is False
                assert "MCP exception" in result.error

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_unavailable_mdap_handling(self):
        """Test handling when MDAP is completely unavailable."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', False), \
             patch('reliability.mdap_adapter.MDAP_MCP_AVAILABLE', False):
            adapter = MDAPReliabilityAdapter()

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8
            )

            # Should fail gracefully with appropriate error
            assert result.success is False
            assert "MDAP not available" in result.error

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_vote_validation_failure_handling(self, mock_mdap_core):
        """Test vote validation failure handling."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[
                    MockVote("Valid vote", 0.9),
                    MockVote("Invalid vote", 0.3)
                ],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Configure validator to fail
            adapter.mdap_core.VoteValidator.return_value.validate_vote.return_value = Mock(
                is_valid=False,
                vote=MockVote("Invalid vote", 0.3),
                failures=[{"validator": "score_threshold"}]
            )

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                enable_vote_validation=True
            )

            # Should handle validation failures gracefully
            assert result.success is True


class TestIntegration:
    """Test suite for integration with other reliability components."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_integration_with_lmql_adapter(self, mock_lmql_adapter, mock_mdap_core):
        """Test integration with LMQL adapter."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Configure constraints
            mock_lmql_adapter.create_constraint.return_value = Mock(
                type="max_tokens",
                value=1000
            )

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Constrained vote", 0.9)],
                final_decision="Constrained decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Generate with constraints",
                max_votes=100,
                validation_threshold=0.8,
                enable_lmql_constraints=True
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_integration_with_guardrails_adapter(self, mock_guardrails_adapter, mock_mdap_core):
        """Test integration with Guardrails adapter."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure validation
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True, failures=[]
            )
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True, failures=[], remediation_applied=None, output=None
            )

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Safe vote", 0.9)],
                final_decision="Safe decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Safe task",
                max_votes=100,
                validation_threshold=0.8,
                enable_input_validation=True,
                enable_output_validation=True
            )

            assert result.success is True
            assert "guardrails_input" in result.layers_used
            assert "guardrails_output" in result.layers_used

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_unified_bridge_workflow(self, mock_lmql_adapter, mock_guardrails_adapter, mock_mdap_core):
        """Test integration with Unified Bridge workflow."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter(
                lmql_adapter=mock_lmql_adapter,
                guardrails_adapter=mock_guardrails_adapter
            )

            # Configure all components
            mock_lmql_adapter.create_constraint.return_value = Mock(
                type="max_tokens",
                value=1000
            )

            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True, failures=[]
            )

            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True, failures=[], remediation_applied=None, output=None
            )

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Unified vote", 0.9)],
                final_decision="Unified decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Unified bridge task",
                max_votes=100,
                validation_threshold=0.8,
                enable_input_validation=True,
                enable_output_validation=True,
                enable_lmql_constraints=True
            )

            # Should use all layers
            assert result.success is True
            assert "guardrails_input" in result.layers_used
            assert "lmql_constraints" in result.layers_used
            assert "mdap_core" in result.layers_used
            assert "guardrails_output" in result.layers_used


class TestPerformance:
    """Test suite for performance and optimization."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_batch_generation(self, mock_mdap_core):
        """Test batch generation for performance."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Batch vote", 0.9)],
                final_decision="Batch decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            tasks = [f"Solve: {i} + {i}" for i in range(5)]
            results = []

            for task in tasks:
                result = adapter.solve_with_validation(
                    task=task,
                    max_votes=100,
                    validation_threshold=0.8
                )
                results.append(result)

            # Should process all tasks
            assert len(results) == 5
            assert all(result.success for result in results)

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_caching_mechanism(self, mock_mdap_core):
        """Test caching mechanism for performance."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Cached vote", 0.9)],
                final_decision="Cached decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Same task multiple times
            for _ in range(3):
                result = adapter.solve_with_validation(
                    task="Same task",
                    max_votes=100,
                    validation_threshold=0.8
                )

            # Should cache results (implementation dependent)
            stats = adapter.get_performance_stats()
            assert isinstance(stats, dict)

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_concurrent_vote_validation(self, mock_mdap_core):
        """Test concurrent vote validation."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver with many votes
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote(f"Vote {i}", 0.8 + i * 0.02) for i in range(100)],
                final_decision="Concurrent decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve complex problem",
                max_votes=100,
                validation_threshold=0.7,
                enable_concurrent_validation=True
            )

            # Should succeed with concurrent validation
            assert result.success is True
            assert "concurrent_validation" in result.layers_used


class TestParameterizedTests:
    """Parameterized tests for different configurations."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    @pytest.mark.parametrize("max_votes", [10, 50, 100, 500])
    def test_different_max_votes(self, max_votes, mock_mdap_core):
        """Test solve with different max votes."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote(f"Vote", 0.9) for _ in range(max_votes)],
                final_decision=f"Decision with {max_votes} votes"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=max_votes,
                validation_threshold=0.8
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    @pytest.mark.parametrize("validation_threshold", [0.5, 0.7, 0.8, 0.9])
    def test_different_validation_thresholds(self, validation_threshold, mock_mdap_core):
        """Test solve with different validation thresholds."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote(f"Vote", 0.9)],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=validation_threshold
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    @pytest.mark.parametrize("enable_input_validation", [True, False])
    @pytest.mark.parametrize("enable_output_validation", [True, False])
    def test_different_validation_combinations(self, enable_input_validation, enable_output_validation, mock_mdap_core):
        """Test with different validation combinations."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Valid vote", 0.9)],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.8,
                enable_input_validation=enable_input_validation,
                enable_output_validation=enable_output_validation
            )

            assert result.success is True
            if enable_input_validation:
                assert "guardrails_input" in result.layers_used
            if enable_output_validation:
                assert "guardrails_output" in result.layers_used


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_extremely_long_task(self, mock_mdap_core):
        """Test with extremely long task."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Solution for long task", 0.9)],
                final_decision="Solution for long task"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            very_long_task = "x" * 10000  # Very long task

            result = adapter.solve_with_validation(
                task=very_long_task,
                max_votes=100,
                validation_threshold=0.8
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_unicode_task(self, mock_mdap_core):
        """Test with unicode task."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Solution", 0.9)],
                final_decision="Solution"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            unicode_task = "测试 task with 🚀 emojis and ñáéíóú"

            result = adapter.solve_with_validation(
                task=unicode_task,
                max_votes=100,
                validation_threshold=0.8
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_special_characters_task(self, mock_mdap_core):
        """Test with special characters."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Solution", 0.9)],
                final_decision="Solution"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            special_task = "Task with !@#$%^&*()_+-=[]{}|;':\",./<>?`~"

            result = adapter.solve_with_validation(
                task=special_task,
                max_votes=100,
                validation_threshold=0.8
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_zero_threshold_validation(self, mock_mdap_core):
        """Test with zero validation threshold."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Low score vote", 0.1)],
                final_decision="Zero threshold decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=0.0,  # Accept all votes
                enable_vote_validation=True
            )

            assert result.success is True

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_maximum_threshold_validation(self, mock_mdap_core):
        """Test with maximum validation threshold."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Perfect vote", 1.0)],
                final_decision="Perfect decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            result = adapter.solve_with_validation(
                task="Solve: 2 + 2",
                max_votes=100,
                validation_threshold=1.0,  # Only perfect votes
                enable_vote_validation=True
            )

            assert result.success is True


# =============================================================================
# UTILITIES
# =============================================================================

class TestUtilities:
    """Test suite for utility functions."""

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_result_conversion_to_dict(self):
        """Test result conversion to dictionary."""
        result = MDAPSolveResult(
            success=True,
            result={"test": "data"},
            statistics={"total_votes": 10},
            correlation_id="test_corr"
        )

        result_dict = result.__dict__

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["result"] == {"test": "data"}
        assert result_dict["statistics"] == {"total_votes": 10}

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_vote_validation_result_creation(self):
        """Test vote validation result creation."""
        vote = MockVote("Test vote", 0.9)
        result = VoteValidationResult(
            is_valid=True,
            vote=vote,
            failures=[],
            remediated=False
        )

        assert result.is_valid is True
        assert result.vote == vote
        assert len(result.failures) == 0
        assert result.remediated is False

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_remediation_strategy_enumeration(self):
        """Test remediation strategy enumeration."""
        assert hasattr(RemediationStrategy, 'REASK')
        assert hasattr(RemediationStrategy, 'FIX')
        assert hasattr(RemediationStrategy, 'FILTER')
        assert hasattr(RemediationStrategy, 'REFRAIN')
        assert hasattr(RemediationStrategy, 'EXCEPTION')

        assert RemediationStrategy.REASK == "reask"
        assert RemediationStrategy.FIX == "fix"
        assert RemediationStrategy.FILTER == "filter"
        assert RemediationStrategy.REFRAIN == "refrain"
        assert RemediationStrategy.EXCEPTION == "exception"

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_convenience_functions(self, mock_mdap_core):
        """Test convenience functions."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Valid vote", 0.9)],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Test solve_with_validation function
            result = solve_with_validation("Solve: 2 + 2", max_votes=100, validation_threshold=0.8)

            assert result.success is True

            # Test get_default_adapter function
            default_adapter = get_default_adapter()
            assert default_adapter is not None

    @pytest.mark.skipif(not MDAP_ADAPTER_AVAILABLE, reason="MDAP adapter not available")
    def test_performance_profiling(self, mock_mdap_core):
        """Test performance profiling utilities."""
        with patch('reliability.mdap_adapter.MDAP_CORE_AVAILABLE', True):
            adapter = MDAPReliabilityAdapter()

            # Configure MDAP solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                votes=[MockVote("Vote", 0.9)],
                final_decision="Decision"
            )
            mock_mdap_core.MDAPSolver.return_value = mock_solver

            # Perform some operations
            for _ in range(5):
                adapter.solve_with_validation("Task", 100, 0.8)

            # Get performance profile
            profile = adapter.get_performance_profile()

            assert isinstance(profile, dict)
            assert "total_operations" in profile
            assert "average_operation_time" in profile
            assert "success_rate" in profile


# =============================================================================
# SETUP/TEARDOWN
# =============================================================================

@pytest.fixture(scope="class", autouse=True)
def setup_class():
    """Setup for test class."""
    yield
    # Cleanup after all tests in class


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Clear any state between tests