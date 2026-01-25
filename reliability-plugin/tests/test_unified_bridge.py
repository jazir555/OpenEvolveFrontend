"""
Comprehensive integration test suite for Unified Bridge reliability component.

This module tests the UnifiedBridge class which coordinates across all reliability layers:
- LMQL constraints (layer 1)
- Guardrails validation (layer 2)
- ROMA/MDAP execution (layer 3)
- Output validation (layer 4)
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the actual modules with fallbacks
try:
    from reliability.unified_bridge import UnifiedBridge, LayerOrder, CoordinationResult
    UNIFIED_BRIDGE_AVAILABLE = True
except ImportError:
    UNIFIED_BRIDGE_AVAILABLE = False
    # Create mock classes for testing
    class MockLayerOrder:
        LMQL_CONSTRAINTS = "lmql_constraints"
        GUARDRAILS_VALIDATION = "guardrails_validation"
        ROMA_MDAP_EXECUTION = "roma_mdap_execution"
        OUTPUT_VALIDATION = "output_validation"

    class MockCoordinationResult:
        def __init__(self, success=True, data=None, error=None, layer_results=None, statistics=None):
            self.success = success
            self.data = data
            self.error = error
            self.layer_results = layer_results or {}
            self.statistics = statistics or {}

    # Mock UnifiedBridge class
    class MockUnifiedBridge:
        def __init__(self, config=None):
            self.config = config or {}
            self.statistics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "layer_stats": {
                    MockLayerOrder.LMQL_CONSTRAINTS: {"attempts": 0, "successes": 0, "failures": 0},
                    MockLayerOrder.GUARDRAILS_VALIDATION: {"attempts": 0, "successes": 0, "failures": 0},
                    MockLayerOrder.ROMA_MDAP_EXECUTION: {"attempts": 0, "successes": 0, "failures": 0},
                    MockLayerOrder.OUTPUT_VALIDATION: {"attempts": 0, "successes": 0, "failures": 0}
                }
            }

        async def coordinate_generation(self, task: str, constraints: Dict[str, Any] = None):
            return MockCoordinationResult(success=True, data={"result": f"Generated for: {task}"})

        async def coordinate_validation(self, output: str, validators: List[str] = None):
            return MockCoordinationResult(success=True, data={"is_valid": True})

        def get_layer_statistics(self, layer_name: str = None):
            return self.statistics

        async def health_check(self):
            return {"status": "healthy", "layers": {"all": True}}


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestUnifiedBridgeInitialization:
    """Test UnifiedBridge initialization and configuration."""

    def test_initialization_with_default_config(self, mock_config):
        """Test initialization with default configuration."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        assert bridge.config == mock_config["unified_bridge"]
        assert bridge.statistics["total_operations"] == 0
        assert bridge.config["enabled"] is True

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "enabled": True,
            "max_retries": 5,
            "retry_delay": 2.0,
            "batch_size": 10,
            "enable_coordination": True,
            "layer_order": [
                LayerOrder.LMQL_CONSTRAINTS,
                LayerOrder.GUARDRAILS_VALIDATION,
                LayerOrder.ROMA_MDAP_EXECUTION,
                LayerOrder.OUTPUT_VALIDATION
            ]
        }

        bridge = UnifiedBridge(config=custom_config)
        assert bridge.config == custom_config
        assert bridge.config["max_retries"] == 5
        assert bridge.config["retry_delay"] == 2.0

    def test_initialization_with_empty_config(self):
        """Test initialization with empty configuration."""
        bridge = UnifiedBridge(config={})
        # Should use default values
        assert bridge.config["enabled"] is True
        assert bridge.config["max_retries"] == 3

    def test_initialization_with_disabled_bridge(self, mock_config):
        """Test initialization when bridge is disabled."""
        disabled_config = mock_config["unified_bridge"].copy()
        disabled_config["enabled"] = False

        bridge = UnifiedBridge(config=disabled_config)
        assert bridge.config["enabled"] is False

    @pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
    def test_custom_adapters_initialization(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test initialization with custom adapters."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        assert bridge._lmql_adapter is mock_lmql_adapter
        assert bridge._guardrails_adapter is mock_guardrails_adapter
        assert bridge._roma_core is mock_roma_core
        assert bridge._mdap_core is mock_mdap_core

    def test_different_layer_combinations_initialization(self, mock_config):
        """Test initialization with different layer combinations."""
        # Test with only some layers enabled
        partial_config = mock_config["unified_bridge"].copy()
        partial_config["layer_order"] = [
            LayerOrder.LMQL_CONSTRAINTS,
            LayerOrder.OUTPUT_VALIDATION
        ]

        bridge = UnifiedBridge(config=partial_config)
        assert len(bridge.layer_order) == 2

    def test_statistics_initialization(self, mock_config):
        """Test statistics initialization."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        assert bridge.statistics["total_operations"] == 0
        assert bridge.statistics["successful_operations"] == 0
        assert bridge.statistics["failed_operations"] == 0

        # Check layer statistics initialization
        for layer_stats in bridge.statistics["layer_stats"].values():
            assert layer_stats["attempts"] == 0
            assert layer_stats["successes"] == 0
            assert layer_stats["failures"] == 0


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestLayerOrderValidation:
    """Test layer order validation and configuration."""

    def test_valid_layer_order(self):
        """Test valid layer order configuration."""
        valid_orders = [
            # Default order
            [
                LayerOrder.LMQL_CONSTRAINTS,
                LayerOrder.GUARDRAILS_VALIDATION,
                LayerOrder.ROMA_MDAP_EXECUTION,
                LayerOrder.OUTPUT_VALIDATION
            ],
            # Custom order
            [
                LayerOrder.GUARDRAILS_VALIDATION,
                LayerOrder.LMQL_CONSTRAINTS,
                LayerOrder.OUTPUT_VALIDATION,
                LayerOrder.ROMA_MDAP_EXECUTION
            ],
            # Subset of layers
            [
                LayerOrder.LMQL_CONSTRAINTS,
                LayerOrder.OUTPUT_VALIDATION
            ]
        ]

        for order in valid_orders:
            bridge = UnifiedBridge(config={"enabled": True, "layer_order": order})
            assert bridge.layer_order == order

    def test_invalid_layer_order(self):
        """Test invalid layer order raises appropriate errors."""
        invalid_orders = [
            [],  # Empty order
            ["invalid_layer"],  # Invalid layer name
            [123, "valid_layer"],  # Non-string layer
            LayerOrder.LMQL_CONSTRAINTS  # Single layer instead of list
        ]

        for order in invalid_orders:
            with pytest.raises((ValueError, TypeError)):
                UnifiedBridge(config={"enabled": True, "layer_order": order})

    def test_duplicate_layer_order(self):
        """Test layer order with duplicate layers."""
        duplicate_order = [
            LayerOrder.LMQL_CONSTRAINTS,
            LayerOrder.LMQL_CONSTRAINTS,
            LayerOrder.GUARDRAILS_VALIDATION
        ]

        bridge = UnifiedBridge(config={"enabled": True, "layer_order": duplicate_order})
        assert bridge.layer_order == duplicate_order  # Should allow duplicates


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestCoordinationGeneration:
    """Test generation coordination across layers."""

    @pytest.mark.asyncio
    async def test_successful_generation_coordination(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test successful generation coordination across all layers."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL generation
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"
        mock_lmql_adapter.constrained_generation.return_value.tokens_used = 100

        # Mock Guardrails validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        # Mock ROMA generation
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        # Mock MDAP validation
        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"
        constraints = {"max_depth": 3, "max_subtasks": 5}

        result = await bridge.coordinate_generation(task, constraints)

        assert result.success is True
        assert result.data is not None
        assert result.layer_results["lmql_constraints"]["success"] is True
        assert result.layer_results["guardrails_validation"]["success"] is True
        assert result.layer_results["roma_mdap_execution"]["success"] is True
        assert result.layer_results["output_validation"]["success"] is True

        # Check statistics
        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1
        assert stats["failed_operations"] == 0

        # Check layer statistics
        for layer_name in ["lmql_constraints", "guardrails_validation", "roma_mdap_execution", "output_validation"]:
            layer_stats = stats["layer_stats"][layer_name]
            assert layer_stats["attempts"] == 1
            assert layer_stats["successes"] == 1
            assert layer_stats["failures"] == 0

    @pytest.mark.asyncio
    async def test_generation_coordination_with_failures(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with layer failures."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL failure
        mock_lmql_adapter.constrained_generation.return_value.success = False
        mock_lmql_adapter.constrained_generation.return_value.error = "LMQL constraint violation"

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is False
        assert result.error is not None
        assert result.layer_results["lmql_constraints"]["success"] is False
        assert result.layer_results["guardrails_validation"]["success"] is False  # Should not be executed
        assert result.layer_results["roma_mdap_execution"]["success"] is False  # Should not be executed
        assert result.layer_results["output_validation"]["success"] is False  # Should not be executed

        # Check statistics
        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 0
        assert stats["failed_operations"] == 1

    @pytest.mark.asyncio
    async def test_generation_coordination_with_partial_failures(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with partial failures."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL success
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        # Mock Guardrails failure
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.failures = ["toxic_language detected"]
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is False
        assert result.layer_results["lmql_constraints"]["success"] is True
        assert result.layer_results["guardrails_validation"]["success"] is False
        assert result.layer_results["roma_mdap_execution"]["success"] is False  # Should not be executed
        assert result.layer_results["output_validation"]["success"] is False  # Should not be executed

    @pytest.mark.asyncio
    async def test_generation_coordination_with_roma_mcp_fallback(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter):
        """Test generation coordination with ROMA MCP fallback."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter

        # Mock ROMA core as unavailable
        bridge._roma_core = None

        # Mock LMQL success
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        # Mock Guardrails success
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        # Mock ROMA MCP tools
        mock_roma_mcp_tools = Mock()
        mock_roma_mcp_tools.solve_with_roma.return_value = {
            "result": "ROMA MCP enhanced text",
            "status": "completed",
            "token_usage": {"input": 100, "output": 200},
            "execution_time": 2.5
        }
        bridge._roma_mcp_tools = mock_roma_mcp_tools

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        assert result.layer_results["roma_mdap_execution"]["success"] is True
        mock_roma_mcp_tools.solve_with_roma.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_coordination_with_mdap_mcp_fallback(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_mcp_tools):
        """Test generation coordination with MDAP MCP fallback."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core

        # Mock MDAP core as unavailable
        bridge._mdap_core = None

        # Mock LMQL success
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        # Mock Guardrails success
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        # Mock ROMA success
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        # Mock MDAP MCP tools
        mock_mdap_mcp_tools.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP MCP validated decision"
        )
        bridge._mdap_mcp_tools = mock_mdap_mcp_tools

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        assert result.layer_results["roma_mdap_execution"]["success"] is True
        mock_mdap_mcp_tools.solve.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_coordination_with_retry_logic(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with retry logic."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL with transient failure then success
        mock_result1 = Mock()
        mock_result1.success = False
        mock_result1.error = "Transient error"
        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.text = "LMQL constrained text"
        mock_result2.tokens_used = 100
        mock_lmql_adapter.constrained_generation.side_effect = [mock_result1, mock_result2]

        # Mock other layers as successful
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        # LMQL should have been called twice (retry)
        assert mock_lmql_adapter.constrained_generation.call_count == 2
        # Other layers should only have been called once
        assert mock_guardrails_adapter.validate_output.call_count == 1
        assert mock_solver.solve.call_count == 1
        assert mock_mdap_solver.solve.call_count == 1


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestCoordinationValidation:
    """Test output coordination across layers."""

    @pytest.mark.asyncio
    async def test_successful_validation_coordination(self, mock_config, mock_guardrails_adapter, mock_mdap_core):
        """Test successful validation coordination."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._mdap_core = mock_mdap_core

        # Mock Guardrails validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        # Mock MDAP validation
        mock_validator = mock_mdap_core.VoteValidator.return_value
        mock_validator.validate_vote.return_value = Mock(
            is_valid=True,
            failures=[],
            remediated=False
        )

        output = "This is a safe and appropriate output"
        validators = ["toxic_language", "pii_detection"]

        result = await bridge.coordinate_validation(output, validators)

        assert result.success is True
        assert result.data is not None
        assert result.layer_results["guardrails_validation"]["success"] is True
        assert result.layer_results["roma_mdap_execution"]["success"] is True

    @pytest.mark.asyncio
    async def test_validation_coordination_with_failures(self, mock_config, mock_guardrails_adapter, mock_mdap_core):
        """Test validation coordination with failures."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._mdap_core = mock_mdap_core

        # Mock Guardrails failure
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.failures = ["toxic_language detected"]
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        output = "This contains toxic content"
        validators = ["toxic_language", "pii_detection"]

        result = await bridge.coordinate_validation(output, validators)

        assert result.success is False
        assert result.layer_results["guardrails_validation"]["success"] is False
        assert result.layer_results["roma_mdap_execution"]["success"] is False  # Should not be executed

    @pytest.mark.asyncio
    async def test_validation_coordination_with_partial_success(self, mock_config, mock_guardrails_adapter, mock_mdap_core):
        """Test validation coordination with partial success."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._mdap_core = mock_mdap_core

        # Mock Guardrails success
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        # Mock MDAP failure
        mock_validator = mock_mdap_core.VoteValidator.return_value
        mock_validator.validate_vote.return_value = Mock(
            is_valid=False,
            failures=["Low quality content"],
            remediated=False
        )

        output = "This is safe but low quality content"
        validators = ["toxic_language", "quality_check"]

        result = await bridge.coordinate_validation(output, validators)

        assert result.success is False
        assert result.layer_results["guardrails_validation"]["success"] is True
        assert result.layer_results["roma_mdap_execution"]["success"] is False


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestBatchGeneration:
    """Test batch generation capabilities."""

    @pytest.mark.asyncio
    async def test_successful_batch_generation(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test successful batch generation."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL generation for all tasks
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        # Mock Guardrails validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        # Mock ROMA generation
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        # Mock MDAP validation
        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = [
            "Write a Python function to calculate fibonacci numbers",
            "Write a Python function to sort a list",
            "Write a Python function to reverse a string"
        ]

        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == 3
        assert all(result.success for result in results)

        # Check statistics
        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 3
        assert stats["failed_operations"] == 0

        # Check that LMQL was called for each task
        assert mock_lmql_adapter.constrained_generation.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_generation_with_failures(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation with some failures."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL to fail on first task
        mock_lmql_adapter.constrained_generation.side_effect = [
            Mock(success=False, error="LMQL constraint violation"),
            Mock(success=True, text="LMQL constrained text", tokens_used=100),
            Mock(success=True, text="LMQL constrained text", tokens_used=100)
        ]

        # Mock other layers
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = [
            "Write a Python function to calculate fibonacci numbers",  # Will fail
            "Write a Python function to sort a list",  # Will succeed
            "Write a Python function to reverse a string"  # Will succeed
        ]

        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == 3
        assert results[0].success is False
        assert results[1].success is True
        assert results[2].success is True

        # Check statistics
        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 2
        assert stats["failed_operations"] == 1

    @pytest.mark.asyncio
    async def test_batch_generation_with_concurrent_execution(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation with concurrent execution."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock slow LMQL generation
        async def slow_generation(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate delay
            return Mock(success=True, text="LMQL constrained text", tokens_used=100)

        mock_lmql_adapter.constrained_generation.side_effect = slow_generation

        # Mock other layers
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = [
            "Write a Python function to calculate fibonacci numbers",
            "Write a Python function to sort a list",
            "Write a Python function to reverse a string"
        ]

        start_time = asyncio.get_event_loop().time()
        results = await bridge.coordinate_batch_generation(tasks)
        end_time = asyncio.get_event_loop().time()

        assert len(results) == 3
        assert all(result.success for result in results)

        # With concurrent execution, should take less than sequential time
        execution_time = end_time - start_time
        assert execution_time < 0.3  # Should be much less than 3 * 0.1 = 0.3s


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestGracefulDegradation:
    """Test graceful degradation when components are unavailable."""

    @pytest.mark.asyncio
    async def test_generation_coordination_without_lmql(self, mock_config, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination without LMQL adapter."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = None  # LMQL not available
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock other layers
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        # Should skip LMQL layer
        assert "lmql_constraints" not in result.layer_results or result.layer_results["lmql_constraints"]["success"] is False
        assert result.layer_results["guardrails_validation"]["success"] is True
        assert result.layer_results["roma_mdap_execution"]["success"] is True
        assert result.layer_results["output_validation"]["success"] is True

    @pytest.mark.asyncio
    async def test_generation_coordination_without_guardrails(self, mock_config, mock_lmql_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination without Guardrails adapter."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = None  # Guardrails not available
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock other layers
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        assert result.layer_results["lmql_constraints"]["success"] is True
        # Should skip Guardrails layer
        assert "guardrails_validation" not in result.layer_results or result.layer_results["guardrails_validation"]["success"] is False
        assert result.layer_results["roma_mdap_execution"]["success"] is True
        assert result.layer_results["output_validation"]["success"] is True

    @pytest.mark.asyncio
    async def test_generation_coordination_without_roma_and_mdap(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter):
        """Test generation coordination without ROMA and MDAP adapters."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = None  # ROMA not available
        bridge._mdap_core = None  # MDAP not available

        # Mock other layers
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        assert result.layer_results["lmql_constraints"]["success"] is True
        assert result.layer_results["guardrails_validation"]["success"] is True
        # Should skip ROMA/MDAP layer
        assert "roma_mdap_execution" not in result.layer_results or result.layer_results["roma_mdap_execution"]["success"] is False
        assert result.layer_results["output_validation"]["success"] is True

    @pytest.mark.asyncio
    async def test_generation_coordination_with_all_layers_disabled(self, mock_config):
        """Test generation coordination when all layers are disabled."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = None
        bridge._guardrails_adapter = None
        bridge._roma_core = None
        bridge._mdap_core = None

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is False
        assert result.error is not None
        assert "No reliability adapters available" in result.error

    @pytest.mark.asyncio
    async def test_validation_coordination_without_guardrails(self, mock_config, mock_mdap_core):
        """Test validation coordination without Guardrails adapter."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._guardrails_adapter = None  # Guardrails not available
        bridge._mdap_core = mock_mdap_core

        # Mock MDAP validation
        mock_validator = mock_mdap_core.VoteValidator.return_value
        mock_validator.validate_vote.return_value = Mock(
            is_valid=True,
            failures=[],
            remediated=False
        )

        output = "This is a safe and appropriate output"
        validators = ["toxic_language", "pii_detection"]

        result = await bridge.coordinate_validation(output, validators)

        assert result.success is True
        # Should skip Guardrails layer
        assert "guardrails_validation" not in result.layer_results or result.layer_results["guardrails_validation"]["success"] is False
        assert result.layer_results["roma_mdap_execution"]["success"] is True

    @pytest.mark.asyncio
    async def test_validation_coordination_without_mdap(self, mock_config, mock_guardrails_adapter):
        """Test validation coordination without MDAP adapter."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._mdap_core = None  # MDAP not available

        # Mock Guardrails validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        output = "This is a safe and appropriate output"
        validators = ["toxic_language", "pii_detection"]

        result = await bridge.coordinate_validation(output, validators)

        assert result.success is True
        assert result.layer_results["guardrails_validation"]["success"] is True
        # Should skip MDAP layer
        assert "roma_mdap_execution" not in result.layer_results or result.layer_results["roma_mdap_execution"]["success"] is False


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestStatisticsAndHealth:
    """Test statistics tracking and health checks."""

    def test_layer_statistics_tracking(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test layer statistics tracking."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Simulate some operations
        bridge.statistics["total_operations"] = 10
        bridge.statistics["successful_operations"] = 8
        bridge.statistics["failed_operations"] = 2

        # Update layer statistics
        bridge.statistics["layer_stats"]["lmql_constraints"] = {"attempts": 10, "successes": 8, "failures": 2}
        bridge.statistics["layer_stats"]["guardrails_validation"] = {"attempts": 9, "successes": 7, "failures": 2}
        bridge.statistics["layer_stats"]["roma_mdap_execution"] = {"attempts": 8, "successes": 6, "failures": 2}
        bridge.statistics["layer_stats"]["output_validation"] = {"attempts": 7, "successes": 5, "failures": 2}

        stats = bridge.get_layer_statistics()

        assert stats["total_operations"] == 10
        assert stats["successful_operations"] == 8
        assert stats["failed_operations"] == 2

        for layer_name in ["lmql_constraints", "guardrails_validation", "roma_mdap_execution", "output_validation"]:
            layer_stats = stats["layer_stats"][layer_name]
            assert isinstance(layer_stats["attempts"], int)
            assert isinstance(layer_stats["successes"], int)
            assert isinstance(layer_stats["failures"], int)

    def test_get_layer_statistics_for_specific_layer(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test getting statistics for a specific layer."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Simulate LMQL layer operations
        bridge.statistics["layer_stats"]["lmql_constraints"] = {"attempts": 5, "successes": 4, "failures": 1}

        # Get specific layer statistics
        lmql_stats = bridge.get_layer_statistics("lmql_constraints")
        assert lmql_stats == {"attempts": 5, "successes": 4, "failures": 1}

    def test_get_layer_statistics_for_nonexistent_layer(self, mock_config):
        """Test getting statistics for non-existent layer."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])

        stats = bridge.get_layer_statistics("nonexistent_layer")
        assert stats is None

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test health check when all layers are healthy."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock all adapters as available
        mock_lmql_adapter.is_available.return_value = True
        mock_lmql_adapter.get_status.return_value = {"available": True, "model": "gpt-4"}

        mock_guardrails_adapter.is_available.return_value = True
        mock_guardrails_adapter.get_status.return_value = {"enabled": True, "validators": ["toxic_language"]}

        health = await bridge.health_check()

        assert health["status"] == "healthy"
        assert health["layers"]["all"] is True
        assert health["layers"]["lmql_constraints"] is True
        assert health["layers"]["guardrails_validation"] is True
        assert health["layers"]["roma_mdap_execution"] is True
        assert health["layers"]["output_validation"] is True

    @pytest.mark.asyncio
    async def test_health_check_partial_failures(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test health check with partial layer failures."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock some adapters as unavailable
        mock_lmql_adapter.is_available.return_value = False
        mock_lmql_adapter.get_status.return_value = {"available": False, "error": "Service unavailable"}

        mock_guardrails_adapter.is_available.return_value = True
        mock_guardrails_adapter.get_status.return_value = {"enabled": True, "validators": ["toxic_language"]}

        # Mock ROMA as available
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        health = await bridge.health_check()

        assert health["status"] == "degraded"
        assert health["layers"]["all"] is False
        assert health["layers"]["lmql_constraints"] is False
        assert health["layers"]["guardrails_validation"] is True
        assert health["layers"]["roma_mdap_execution"] is True
        assert health["layers"]["output_validation"] is True

    @pytest.mark.asyncio
    async def test_health_check_all_unavailable(self, mock_config):
        """Test health check when all layers are unavailable."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = None
        bridge._guardrails_adapter = None
        bridge._roma_core = None
        bridge._mdap_core = None

        health = await bridge.health_check()

        assert health["status"] == "unhealthy"
        assert health["layers"]["all"] is False
        assert health["layers"]["lmql_constraints"] is False
        assert health["layers"]["guardrails_validation"] is False
        assert health["layers"]["roma_mdap_execution"] is False
        assert health["layers"]["output_validation"] is False

    def test_statistics_reset(self, mock_config):
        """Test statistics reset functionality."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])

        # Simulate some operations
        bridge.statistics["total_operations"] = 10
        bridge.statistics["successful_operations"] = 8
        bridge.statistics["failed_operations"] = 2

        # Update layer statistics
        bridge.statistics["layer_stats"]["lmql_constraints"] = {"attempts": 10, "successes": 8, "failures": 2}

        # Reset statistics
        bridge.statistics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "layer_stats": {
                "lmql_constraints": {"attempts": 0, "successes": 0, "failures": 0},
                "guardrails_validation": {"attempts": 0, "successes": 0, "failures": 0},
                "roma_mdap_execution": {"attempts": 0, "successes": 0, "failures": 0},
                "output_validation": {"attempts": 0, "successes": 0, "failures": 0}
            }
        }

        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 0
        assert stats["successful_operations"] == 0
        assert stats["failed_operations"] == 0

        for layer_name in ["lmql_constraints", "guardrails_validation", "roma_mdap_execution", "output_validation"]:
            layer_stats = stats["layer_stats"][layer_name]
            assert layer_stats["attempts"] == 0
            assert layer_stats["successes"] == 0
            assert layer_stats["failures"] == 0


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestErrorHandling:
    """Test error handling and exception management."""

    @pytest.mark.asyncio
    async def test_generation_coordination_with_exception(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with exceptions."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL to raise exception
        mock_lmql_adapter.constrained_generation.side_effect = Exception("LMQL service error")

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is False
        assert result.error is not None
        assert "LMQL service error" in result.error
        assert result.layer_results["lmql_constraints"]["success"] is False
        assert "exception" in result.layer_results["lmql_constraints"]

    @pytest.mark.asyncio
    async def test_generation_coordination_with_retry_exhaustion(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with retry exhaustion."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Configure max retries
        bridge.config["max_retries"] = 2

        # Mock LMQL to always fail
        mock_lmql_adapter.constrained_generation.return_value.success = False
        mock_lmql_adapter.constrained_generation.return_value.error = "Persistent failure"

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is False
        assert "Retry exhausted" in result.error or "Persistent failure" in result.error

        # Check that LMQL was called max_retries + 1 times
        assert mock_lmql_adapter.constrained_generation.call_count == 3  # 2 retries + 1 initial

    @pytest.mark.asyncio
    async def test_generation_coordination_with_timeout(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with timeout handling."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock timeout exception
        mock_lmql_adapter.constrained_generation.side_effect = asyncio.TimeoutError("Operation timed out")

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_batch_generation_with_mixed_results(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation with mixed success/failure results."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock mixed results
        mock_lmql_adapter.constrained_generation.side_effect = [
            Mock(success=True, text="LMQL constrained text 1", tokens_used=100),
            Mock(success=False, error="LMQL constraint violation"),
            Mock(success=True, text="LMQL constrained text 3", tokens_used=100)
        ]

        # Mock other layers
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = [
            "Write a Python function to calculate fibonacci numbers",
            "Write a Python function to sort a list",
            "Write a Python function to reverse a string"
        ]

        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

        # Check statistics reflect mixed results
        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 2
        assert stats["failed_operations"] == 1


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_generation_coordination_performance(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination performance."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock fast operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        # Measure performance
        start_time = asyncio.get_event_loop().time()
        result = await bridge.coordinate_generation(task)
        end_time = asyncio.get_event_loop().time()

        assert result.success is True
        execution_time = end_time - start_time

        # Should complete quickly with mocked operations
        assert execution_time < 1.0  # Less than 1 second

    @pytest.mark.asyncio
    async def test_batch_generation_performance(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation performance."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock fast operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = ["Task " + str(i) for i in range(10)]  # 10 tasks

        # Measure performance
        start_time = asyncio.get_event_loop().time()
        results = await bridge.coordinate_batch_generation(tasks)
        end_time = asyncio.get_event_loop().time()

        assert len(results) == 10
        assert all(result.success for result in results)
        execution_time = end_time - start_time

        # Batch should be significantly faster than sequential
        assert execution_time < 2.0  # Less than 2 seconds for 10 tasks

    @pytest.mark.asyncio
    async def test_large_batch_generation(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation with large batch sizes."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Configure batch size
        bridge.config["batch_size"] = 5

        # Mock fast operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = ["Task " + str(i) for i in range(20)]  # 20 tasks

        # Should handle large batch without issues
        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == 20
        assert all(result.success for result in results)


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestParameterizedTests:
    """Parameterized tests for different configurations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retry_count", [0, 1, 3, 5])
    async def test_generation_coordination_with_different_retry_counts(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core, retry_count):
        """Test generation coordination with different retry counts."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Configure retry count
        bridge.config["max_retries"] = retry_count

        # Mock LMQL to fail initially then succeed
        mock_result1 = Mock()
        mock_result1.success = False
        mock_result1.error = "Transient error"
        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.text = "LMQL constrained text"
        mock_result2.tokens_used = 100

        # Create side effect that fails once then succeeds
        def side_effect(*args, **kwargs):
            if mock_lmql_adapter.constrained_generation.call_count == 1:
                return mock_result1
            return mock_result2

        mock_lmql_adapter.constrained_generation.side_effect = side_effect

        # Mock other layers
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True

        # Should be called 1 (initial) + min(retry_count, 1) times
        expected_calls = 1 + min(retry_count, 1)
        assert mock_lmql_adapter.constrained_generation.call_count == expected_calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    async def test_batch_generation_with_different_batch_sizes(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core, batch_size):
        """Test batch generation with different batch sizes."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Configure batch size
        bridge.config["batch_size"] = batch_size

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        tasks = ["Task " + str(i) for i in range(batch_size * 2)]  # 2x batch size

        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == len(tasks)
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("layer_order", [
        [LayerOrder.LMQL_CONSTRAINTS, LayerOrder.GUARDRAILS_VALIDATION, LayerOrder.ROMA_MDAP_EXECUTION, LayerOrder.OUTPUT_VALIDATION],
        [LayerOrder.GUARDRAILS_VALIDATION, LayerOrder.LMQL_CONSTRAINTS, LayerOrder.OUTPUT_VALIDATION, LayerOrder.ROMA_MDAP_EXECUTION],
        [LayerOrder.ROMA_MDAP_EXECUTION, LayerOrder.LMQL_CONSTRAINTS, LayerOrder.GUARDRAILS_VALIDATION, LayerOrder.OUTPUT_VALIDATION],
        [LayerOrder.OUTPUT_VALIDATION, LayerOrder.ROMA_MDAP_EXECUTION, LayerOrder.GUARDRAILS_VALIDATION, LayerOrder.LMQL_CONSTRAINTS],
    ])
    async def test_generation_coordination_with_different_layer_orders(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core, layer_order):
        """Test generation coordination with different layer orders."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Configure custom layer order
        bridge.layer_order = layer_order

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True

        # Check that all layers were executed in the specified order
        executed_layers = []
        for layer_name, layer_result in result.layer_results.items():
            if layer_result["success"]:
                executed_layers.append(layer_name)

        # Verify layers were executed according to custom order
        custom_order_names = [
            LayerOrder.LMQL_CONSTRAINTS,
            LayerOrder.GUARDRAILS_VALIDATION,
            LayerOrder.ROMA_MDAP_EXECUTION,
            LayerOrder.OUTPUT_VALIDATION
        ]

        # Filter based on enabled layers in custom order
        expected_order = [layer for layer in layer_order if layer in executed_layers]
        assert executed_layers == expected_order


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    @pytest.mark.asyncio
    async def test_generation_coordination_with_empty_task(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with empty task."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "Generated text for empty task"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        # Empty task
        result = await bridge.coordinate_generation("")

        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_generation_coordination_with_extremely_long_task(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with extremely long task."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Create very long task
        long_task = "Write a Python function to calculate fibonacci numbers. " * 1000

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "Generated text for long task"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        result = await bridge.coordinate_generation(long_task)

        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_generation_coordination_with_unicode_and_special_characters(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with unicode and special characters."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Task with unicode and special characters
        unicode_task = "Write a Python function with: 中文, 日本語, 한국어, русский, العربية, emojis 🚀🎉🔥"

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "Generated text with unicode"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        result = await bridge.coordinate_generation(unicode_task)

        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_generation_coordination_with_none_constraints(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with None constraints."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "Generated text with None constraints"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        # Pass None constraints
        result = await bridge.coordinate_generation(task, None)

        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_batch_generation_with_empty_task_list(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation with empty task list."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Empty task list
        results = await bridge.coordinate_batch_generation([])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_generation_with_duplicate_tasks(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation with duplicate tasks."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock successful operations
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "Generated text"

        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        # Duplicate tasks
        duplicate_task = "Write a Python function to calculate fibonacci numbers"
        tasks = [duplicate_task] * 5  # 5 identical tasks

        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == 5
        assert all(result.success for result in results)
        assert mock_lmql_adapter.constrained_generation.call_count == 5

    @pytest.mark.asyncio
    async def test_generation_coordination_with_threshold_boundaries(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with threshold boundary values."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL with boundary values
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"
        mock_lmql_adapter.constrained_generation.return_value.tokens_used = 1000  # Boundary value

        # Mock other layers
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_validation_result

        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        # Use boundary constraints
        constraints = {
            "max_depth": 0,  # Minimum boundary
            "max_subtasks": 1000,  # Maximum boundary
            "subtask_token_limit": 0  # Minimum boundary
        }

        result = await bridge.coordinate_generation(task, constraints)

        assert result.success is True
        assert result.data is not None


@pytest.mark.skipif(not UNIFIED_BRIDGE_AVAILABLE, reason="Unified Bridge not available")
class TestIntegration:
    """Test integration with other reliability components."""

    @pytest.mark.asyncio
    async def test_generation_coordination_with_all_reliability_layers(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test generation coordination with all reliability layers integrated."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL constraints
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "LMQL constrained text"
        mock_lmql_adapter.constrained_generation.return_value.tokens_used = 100

        # Mock Guardrails input validation
        mock_input_result = Mock()
        mock_input_result.is_valid = True
        mock_input_result.failures = []
        mock_guardrails_adapter.validate_input.return_value = mock_input_result

        # Mock Guardrails output validation
        mock_output_result = Mock()
        mock_output_result.is_valid = True
        mock_output_result.failures = []
        mock_guardrails_adapter.validate_output.return_value = mock_output_result

        # Mock ROMA analysis
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        # Mock MDAP validation
        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        task = "Write a Python function to calculate fibonacci numbers"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        assert result.data is not None

        # Verify all layers were called
        assert mock_lmql_adapter.constrained_generation.call_count == 1
        assert mock_guardrails_adapter.validate_input.call_count == 1
        assert mock_guardrails_adapter.validate_output.call_count == 1
        assert mock_solver.solve.call_count == 1
        assert mock_mdap_solver.solve.call_count == 1

    @pytest.mark.asyncio
    async def test_generation_coordination_with_error_propagation(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test error propagation through the reliability layers."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock LMQL to generate with specific error
        mock_lmql_adapter.constrained_generation.return_value.success = True
        mock_lmql_adapter.constrained_generation.return_value.text = "Text with potential issue"

        # Mock Guardrails to detect and remediate issues
        mock_input_result = Mock()
        mock_input_result.is_valid = True
        mock_input_result.failures = []
        mock_guardrails_adapter.validate_input.return_value = mock_input_result

        mock_output_result = Mock()
        mock_output_result.is_valid = False
        mock_output_result.failures = ["Potential security risk detected"]
        mock_output_result.remediation_applied = "Security filtering applied"
        mock_output_result.output = "Filtered and safe text"
        mock_guardrails_adapter.validate_output.return_value = mock_output_result

        # Mock ROMA to handle the filtered text
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced filtered text", status=Mock(value="completed"))

        # Mock MDAP validation
        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.95), Mock(content="vote2", score=0.9)],
            final_decision="MDAP validated filtered decision"
        )

        task = "Write a Python function with potential security concerns"

        result = await bridge.coordinate_generation(task)

        assert result.success is True
        assert result.data is not None

        # Verify error handling and remediation
        assert "guardrails_validation" in result.layer_results
        assert result.layer_results["guardrails_validation"]["success"] is True
        assert "remediation_applied" in result.layer_results["guardrails_validation"]

    @pytest.mark.asyncio
    async def test_batch_generation_with_error_handling(self, mock_config, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core, mock_mdap_core):
        """Test batch generation with comprehensive error handling."""
        bridge = UnifiedBridge(config=mock_config["unified_bridge"])
        bridge._lmql_adapter = mock_lmql_adapter
        bridge._guardrails_adapter = mock_guardrails_adapter
        bridge._roma_core = mock_roma_core
        bridge._mdap_core = mock_mdap_core

        # Mock mixed results for batch processing
        tasks = [
            "Write a Python function to calculate fibonacci numbers",  # Success
            "Write code with security vulnerability",  # Guardrails will catch this
            "Write a Python function to sort a list"  # Success
        ]

        # Configure different responses for different tasks
        call_count = 0

        def mock_lmql_generation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second task (security vulnerability)
                return Mock(success=True, text="Potentially vulnerable code", tokens_used=100)
            return Mock(success=True, text="Safe code", tokens_used=100)

        mock_lmql_adapter.constrained_generation.side_effect = mock_lmql_generation

        # Mock Guardrails to catch security issues
        def mock_guardrails_validation(*args, **kwargs):
            if call_count == 2:  # Second task
                mock_result = Mock()
                mock_result.is_valid = False
                mock_result.failures = ["Security vulnerability detected"]
                mock_result.remediation_applied = "Code sanitized"
                mock_result.output = "Sanitized code"
                return mock_result
            mock_result = Mock()
            mock_result.is_valid = True
            mock_result.failures = []
            return mock_result

        mock_guardrails_adapter.validate_output.side_effect = mock_guardrails_validation

        # Mock ROMA and MDAP
        mock_solver = mock_roma_core.RecursiveSolver.return_value
        mock_solver.solve.return_value = Mock(result="ROMA enhanced text", status=Mock(value="completed"))

        mock_mdap_solver = mock_mdap_core.MDAPSolver.return_value
        mock_mdap_solver.solve.return_value = Mock(
            votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
            final_decision="MDAP validated decision"
        )

        results = await bridge.coordinate_batch_generation(tasks)

        assert len(results) == 3

        # First task should succeed
        assert results[0].success is True
        assert results[0].layer_results["lmql_constraints"]["success"] is True
        assert results[0].layer_results["guardrails_validation"]["success"] is True

        # Second task should be caught and remediated by Guardrails
        assert results[1].success is True  # Should succeed after remediation
        assert results[1].layer_results["guardrails_validation"]["success"] is True
        assert "remediation_applied" in results[1].layer_results["guardrails_validation"]

        # Third task should succeed
        assert results[2].success is True
        assert results[2].layer_results["lmql_constraints"]["success"] is True
        assert results[2].layer_results["guardrails_validation"]["success"] is True

        # Verify statistics reflect mixed but successful outcomes
        stats = bridge.get_layer_statistics()
        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 3
        assert stats["failed_operations"] == 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_unified_bridge_result(success=True, data=None, error=None, layer_results=None, statistics=None):
    """Create a mock unified bridge result for testing."""
    result = Mock()
    result.success = success
    result.data = data
    result.error = error
    result.layer_results = layer_results or {}
    result.statistics = statistics or {}
    return result


def assert_unified_bridge_result(result, expected_success=True, expected_data=None, expected_error=None):
    """Assert the structure of a unified bridge result."""
    assert hasattr(result, 'success')
    assert result.success == expected_success
    if hasattr(result, 'error') and expected_success:
        assert result.error is None or expected_error is not None
    if hasattr(result, 'data') and expected_success:
        assert result.data is not None or expected_data is not None
    if hasattr(result, 'layer_results'):
        assert isinstance(result.layer_results, dict)


# =============================================================================
# PARAMETERIZED FIXTURES
# =============================================================================

@pytest.fixture(params=[True, False])
def enable_coordination(request):
    """Parameterized fixture for coordination enablement."""
    return request.param


@pytest.fixture(params=[0, 1, 3, 5])
def max_retries(request):
    """Parameterized fixture for maximum retry attempts."""
    return request.param


@pytest.fixture(params=[1, 3, 5, 10])
def batch_size(request):
    """Parameterized fixture for batch size."""
    return request.param


@pytest.fixture(params=[
    LayerOrder.LMQL_CONSTRAINTS,
    LayerOrder.GUARDRAILS_VALIDATION,
    LayerOrder.ROMA_MDAP_EXECUTION,
    LayerOrder.OUTPUT_VALIDATION
])
def single_layer_order(request):
    """Parameterized fixture for single layer orders."""
    return request.param


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@pytest.fixture
def patch_unavailable_bridge():
    """Context manager to patch unified bridge as unavailable."""
    with patch.dict('sys.modules', {
        'reliability.unified_bridge': None
    }):
        yield


# =============================================================================
# SETUP/TEARDOWN
# =============================================================================

@pytest.fixture(scope="function", autouse=True)
def setup_unified_bridge_logging():
    """Setup logging for unified bridge tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    yield
    # Cleanup


@pytest.fixture(autouse=True)
def clear_unified_bridge_imports():
    """Clear imports between tests to prevent module caching issues."""
    modules_to_remove = [k for k in sys.modules.keys()
                        if k.startswith('reliability.unified_bridge')]
    for module in modules_to_remove:
        del sys.modules[module]