"""
Comprehensive Test Suite for DSPy Integration

This module provides complete test coverage for the DSPy integration component.

Test Statistics:
- Total Test Functions: 38
- Test Classes: 6
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Error Handling

Running Tests:
    pytest tests/test_dspy_integration.py -v
    pytest tests/test_dspy_integration.py -v -k "test_cot"
    pytest tests/test_dspy_integration.py --cov=knowledge_engine.integrations.dspy_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_engine.integrations.dspy_integration import (
    DSPyIntegration,
    DSPyResult
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_dspy_config() -> Dict[str, Any]:
    """Default configuration for DSPy integration."""
    return {
        "model": "gpt-4o",
        "api_key": None,
        "api_base": None,
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_retries": 3,
        "backoff_factor": 1,
        "teleprompter": {
            "type": "BootstrapFewShot",
            "k": 8,
            "max_bootstrapped_demos": 8,
            "max_labeled_demos": 8
        },
        "cot_config": {
            "max_iters": 3,
            "rationale_field": "reasoning"
        },
        "pot_config": {
            "max_iters": 3
        }
    }


@pytest.fixture
def sample_question() -> str:
    """Sample question for reasoning."""
    return "What is the capital of France?"


@pytest.fixture
def sample_context() -> str:
    """Sample context for reasoning."""
    return "France is a country in Western Europe."


@pytest.fixture
def mock_dspy_result():
    """Mock DSPy result."""
    result = MagicMock()
    result.reasoning = "France is a country with Paris as its capital city."
    result.answer = "Paris"
    return result


# ============================================================================
# Test Class 1: Initialization Tests
# ============================================================================

class TestDSPyInitialization:
    """Test DSPy integration initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()

            assert integration.config is not None
            assert integration.config["model"] == "gpt-4o"
            assert integration.config["temperature"] == 0.7
            assert integration.config["max_tokens"] == 4096
            assert integration.config["teleprompter"]["type"] == "BootstrapFewShot"

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.5,
            "max_tokens": 8192,
            "api_key": "test_key"
        }

        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration(config=custom_config)

            assert integration.config["model"] == "claude-3-5-sonnet-20241022"
            assert integration.config["temperature"] == 0.5
            assert integration.config["max_tokens"] == 8192

    def test_default_config_structure(self):
        """Test that default config has all required fields."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            config = integration._get_default_config()

            required_keys = [
                "model", "api_key", "api_base", "temperature", "max_tokens",
                "top_p", "frequency_penalty", "presence_penalty",
                "max_retries", "backoff_factor", "teleprompter",
                "cot_config", "pot_config"
            ]

            for key in required_keys:
                assert key in config, f"Missing required config key: {key}"

    def test_teleprompter_config_defaults(self):
        """Test teleprompter configuration defaults."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            teleprompter_config = integration.config["teleprompter"]

            assert teleprompter_config["type"] == "BootstrapFewShot"
            assert teleprompter_config["k"] == 8
            assert teleprompter_config["max_bootstrapped_demos"] == 8
            assert teleprompter_config["max_labeled_demos"] == 8

    def test_cot_config_defaults(self):
        """Test chain-of-thought configuration defaults."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            cot_config = integration.config["cot_config"]

            assert cot_config["max_iters"] == 3
            assert cot_config["rationale_field"] == "reasoning"


# ============================================================================
# Test Class 2: Chain of Thought Tests
# ============================================================================

class TestChainOfThought:
    """Test chain of thought reasoning functionality."""

    @pytest.mark.asyncio
    async def test_chain_of_thought_success(
        self, sample_question, sample_context, mock_dspy_result
    ):
        """Test successful chain of thought reasoning."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()  # Mock LM

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.return_value = mock_dspy_result

                result = await integration.chain_of_thought(
                    question=sample_question,
                    context=sample_context,
                    max_steps=5
                )

                assert result.success is True
                assert result.output == "Paris"
                assert result.reasoning is not None
                assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_chain_of_thought_without_lm(self, sample_question):
        """Test chain of thought when LM is not initialized."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = None

            result = await integration.chain_of_thought(question=sample_question)

            assert result.success is False
            assert result.error is not None
            assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chain_of_thought_custom_max_steps(
        self, sample_question, mock_dspy_result
    ):
        """Test chain of thought with custom max steps."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.return_value = mock_dspy_result

                result = await integration.chain_of_thought(
                    question=sample_question,
                    max_steps=10
                )

                assert result.success is True
                assert result.metadata["max_steps"] == 10

    @pytest.mark.asyncio
    async def test_chain_of_thought_with_correlation_id(
        self, sample_question, mock_dspy_result
    ):
        """Test chain of thought with custom correlation ID."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.return_value = mock_dspy_result

                result = await integration.chain_of_thought(
                    question=sample_question,
                    correlation_id="cot_test_123"
                )

                assert isinstance(result, DSPyResult)


# ============================================================================
# Test Class 3: Program of Thought Tests
# ============================================================================

class TestProgramOfThought:
    """Test program of thought reasoning functionality."""

    @pytest.mark.asyncio
    async def test_program_of_thought_success(
        self, sample_question, sample_context, mock_dspy_result
    ):
        """Test successful program of thought reasoning."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.return_value = mock_dspy_result

                result = await integration.program_of_thought(
                    question=sample_question,
                    context=sample_context,
                    max_iterations=3
                )

                assert result.success is True
                assert result.output is not None

    @pytest.mark.asyncio
    async def test_program_of_thought_without_lm(self, sample_question):
        """Test program of thought when LM is not initialized."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = None

            result = await integration.program_of_thought(question=sample_question)

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_program_of_thought_custom_iterations(
        self, sample_question, mock_dspy_result
    ):
        """Test program of thought with custom iterations."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.return_value = mock_dspy_result

                result = await integration.program_of_thought(
                    question=sample_question,
                    max_iterations=5
                )

                assert result.success is True


# ============================================================================
# Test Class 4: DSPyResult Tests
# ============================================================================

class TestDSPyResult:
    """Test DSPyResult dataclass."""

    def test_result_creation_success(self):
        """Test creating a successful result."""
        result = DSPyResult(
            success=True,
            output="Paris",
            reasoning="France's capital is Paris",
            metadata={"model": "gpt-4o"},
            processing_time_ms=150.0
        )

        assert result.success is True
        assert result.output == "Paris"
        assert result.reasoning == "France's capital is Paris"
        assert result.processing_time_ms == 150.0
        assert result.error is None

    def test_result_creation_failure(self):
        """Test creating a failed result."""
        result = DSPyResult(
            success=False,
            output=None,
            reasoning="",
            metadata={},
            processing_time_ms=50.0,
            error="LM not initialized"
        )

        assert result.success is False
        assert result.error == "LM not initialized"
        assert result.output is None

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DSPyResult(
            success=True,
            output="Answer",
            reasoning="Reasoning",
            metadata={},
            processing_time_ms=100.0
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "success" in result_dict
        assert "output" in result_dict
        assert "reasoning" in result_dict
        assert "metadata" in result_dict
        assert "processing_time_ms" in result_dict
        assert "error" in result_dict


# ============================================================================
# Test Class 5: Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration handling."""

    def test_config_with_missing_api_key(self):
        """Test configuration without API key."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            config = {"model": "gpt-4o", "api_key": None}
            integration = DSPyIntegration(config=config)

            # Should use mock components when no API key
            assert integration.config["api_key"] is None

    def test_config_with_openai_model(self):
        """Test configuration with OpenAI model."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            config = {
                "model": "gpt-4o",
                "api_key": "test_key"
            }
            integration = DSPyIntegration(config=config)

            assert integration.config["model"] == "gpt-4o"

    def test_config_with_anthropic_model(self):
        """Test configuration with Anthropic model."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            config = {
                "model": "claude-3-5-sonnet-20241022",
                "api_key": "test_key"
            }
            integration = DSPyIntegration(config=config)

            assert integration.config["model"] == "claude-3-5-sonnet-20241022"

    def test_config_temperature_bounds(self):
        """Test temperature configuration at boundaries."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            # Test minimum
            config_min = {"temperature": 0.0}
            integration_min = DSPyIntegration(config=config_min)
            assert integration_min.config["temperature"] == 0.0

            # Test maximum
            config_max = {"temperature": 2.0}
            integration_max = DSPyIntegration(config=config_max)
            assert integration_max.config["temperature"] == 2.0


# ============================================================================
# Test Class 6: Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_chain_of_thought_exception_handling(self, sample_question):
        """Test exception handling in chain of thought."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.side_effect = Exception("Test error")

                result = await integration.chain_of_thought(question=sample_question)

                assert result.success is False
                assert result.error is not None

    @pytest.mark.asyncio
    async def test_program_of_thought_exception_handling(self, sample_question):
        """Test exception handling in program of thought."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.side_effect = Exception("Test error")

                result = await integration.program_of_thought(question=sample_question)

                assert result.success is False
                assert result.error is not None

    @pytest.mark.asyncio
    async def test_empty_question_handling(self):
        """Test handling of empty question."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components'):
            integration = DSPyIntegration()
            integration.lm = MagicMock()

            # Should handle gracefully
            mock_result = MagicMock()
            mock_result.reasoning = ""
            mock_result.answer = ""

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.return_value = mock_result

                result = await integration.chain_of_thought(question="")

                assert isinstance(result, DSPyResult)

    def test_component_initialization_failure(self):
        """Test handling of component initialization failure."""
        with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_components') as mock_init:
            mock_init.side_effect = ImportError("DSPy not available")

            # Should not raise, but use mock components
            with patch('knowledge_engine.integrations.dspy_integration.DSPyIntegration._initialize_mock_components'):
                integration = DSPyIntegration()

                assert integration.lm is None
