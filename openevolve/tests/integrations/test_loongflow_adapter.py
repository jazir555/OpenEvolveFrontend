"""
Integration tests for LoongFlow adapter

Tests the integration between OpenEvolve and LoongFlow's PES system.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from openevolve.integrations import LoongFlowAdapter


class TestLoongFlowAdapter:
    """Test suite for LoongFlowAdapter"""

    def test_adapter_initialization(self):
        """Test that adapter initializes correctly"""
        config = {
            "max_iterations": 50,
            "population_size": 10,
            "enable_planning": True,
            "enable_memory": True,
        }

        adapter = LoongFlowAdapter(config)

        # Verify adapter was created
        assert adapter is not None
        assert adapter.config == config
        assert hasattr(adapter, 'pes_agent')
        assert hasattr(adapter, 'available')

    def test_adapter_without_loongflow(self):
        """Test adapter behaves correctly when LoongFlow is not installed"""
        config = {"max_iterations": 10}

        # Mock the import to fail
        with patch('openevolve.integrations.loongflow_adapter.LoongFlowAdapter._initialize_pes_agent'):
            adapter = LoongFlowAdapter(config)
            adapter.available = False
            adapter.pes_agent = None

        # Verify adapter reports unavailable
        assert not adapter.is_available()

    def test_config_mapping(self):
        """Test that OpenEvolve config maps to LoongFlow format correctly"""
        config = {
            "max_iterations": 100,
            "population_size": 20,
            "timeout": 300,
            "enable_planning": True,
            "enable_memory": True,
            "llm_config": {
                "model": "gpt-4",
                "temperature": 0.7
            }
        }

        adapter = LoongFlowAdapter(config)
        mapped_config = adapter._map_config(config)

        # Verify all parameters are mapped
        assert mapped_config["max_iterations"] == 100
        assert mapped_config["population_size"] == 20
        assert mapped_config["timeout"] == 300
        assert mapped_config["enable_planning"] is True
        assert mapped_config["enable_memory"] is True
        assert mapped_config["llm_config"]["model"] == "gpt-4"

    def test_config_mapping_defaults(self):
        """Test that config mapping uses sensible defaults"""
        config = {}  # Empty config

        adapter = LoongFlowAdapter(config)
        mapped_config = adapter._map_config(config)

        # Verify defaults are applied
        assert mapped_config["max_iterations"] == 100
        assert mapped_config["population_size"] == 20
        assert mapped_config["timeout"] == 300
        assert mapped_config["enable_planning"] is True
        assert mapped_config["enable_memory"] is True

    @pytest.mark.asyncio
    async def test_evolve_without_loongflow(self):
        """Test evolve returns fallback result when LoongFlow unavailable"""
        config = {"max_iterations": 10}

        adapter = LoongFlowAdapter(config)
        adapter.available = False
        adapter.pes_agent = None

        result = await adapter.evolve(
            problem="Test problem",
            domain="general"
        )

        # Verify fallback result structure
        assert result["best_solution"] is None
        assert result["best_fitness"] == 0.0
        assert result["total_evaluations"] == 0
        assert result["improvement_rate"] == 0.0
        assert result["iterations_performed"] == 0
        assert "error" in result
        assert result["strategy_used"] == "pes"
        assert result["source"] == "loongflow_pes"

    @pytest.mark.asyncio
    async def test_evolve_with_loongflow_mock(self):
        """Test evolve with mocked LoongFlow agent"""
        config = {"max_iterations": 10}

        # Create adapter
        adapter = LoongFlowAdapter(config)
        adapter.available = True

        # Mock the PES agent
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value={
            "best_solution": "def optimized_func(): pass",
            "best_fitness": 0.95,
            "total_evaluations": 50,
            "improvement_rate": 0.15,
            "iterations_performed": 10
        })
        adapter.pes_agent = mock_agent

        # Run evolution
        result = await adapter.evolve(
            problem="Optimize function",
            domain="code"
        )

        # Verify result
        assert result["best_solution"] == "def optimized_func(): pass"
        assert result["best_fitness"] == 0.95
        assert result["total_evaluations"] == 50
        assert result["improvement_rate"] == 0.15
        assert result["iterations_performed"] == 10
        assert result["strategy_used"] == "pes"
        assert result["source"] == "loongflow_pes"

    @pytest.mark.asyncio
    async def test_evolve_with_error_handling(self):
        """Test that evolution errors are handled gracefully"""
        config = {"max_iterations": 10}

        adapter = LoongFlowAdapter(config)
        adapter.available = True

        # Mock agent that raises an error
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("Evolution failed"))
        adapter.pes_agent = mock_agent

        # Run evolution
        result = await adapter.evolve(
            problem="Test problem",
            domain="general"
        )

        # Verify error is handled
        assert result["best_solution"] is None
        assert result["best_fitness"] == 0.0
        assert "error" in result
        assert "Evolution failed" in result["error"]

    def test_is_available(self):
        """Test is_available method"""
        adapter = LoongFlowAdapter({})

        # When unavailable
        adapter.available = False
        assert not adapter.is_available()

        # When available
        adapter.available = True
        assert adapter.is_available()

    def test_get_capabilities(self):
        """Test get_capabilities method"""
        config = {
            "enable_planning": True,
            "enable_memory": True,
        }

        adapter = LoongFlowAdapter(config)
        adapter.available = True

        capabilities = adapter.get_capabilities()

        # Verify capabilities structure
        assert "available" in capabilities
        assert "supports_planning" in capabilities
        assert "supports_memory" in capabilities
        assert "supported_domains" in capabilities

        assert capabilities["available"] is True
        assert capabilities["supports_planning"] is True
        assert capabilities["supports_memory"] is True
        assert "general" in capabilities["supported_domains"]
        assert "math" in capabilities["supported_domains"]
        assert "code" in capabilities["supported_domains"]

    def test_repr(self):
        """Test string representation"""
        config = {"max_iterations": 100}
        adapter = LoongFlowAdapter(config)
        adapter.available = True

        repr_str = repr(adapter)

        assert "LoongFlowAdapter" in repr_str
        assert "available" in repr_str

    def test_convert_result(self):
        """Test result conversion from LoongFlow to OpenEvolve format"""
        config = {}
        adapter = LoongFlowAdapter(config)

        loongflow_result = {
            "best_solution": "solution code",
            "best_fitness": 0.92,
            "total_evaluations": 100,
            "improvement_rate": 0.25,
            "iterations_performed": 50,
            "metadata": {"extra": "data"}
        }

        converted = adapter._convert_result(loongflow_result)

        assert converted["best_solution"] == "solution code"
        assert converted["best_fitness"] == 0.92
        assert converted["total_evaluations"] == 100
        assert converted["improvement_rate"] == 0.25
        assert converted["iterations_performed"] == 50
        assert converted["strategy_used"] == "pes"
        assert converted["source"] == "loongflow_pes"
        assert converted["metadata"]["extra"] == "data"

    @pytest.mark.asyncio
    async def test_evolve_with_all_parameters(self):
        """Test evolve with all optional parameters"""
        config = {"max_iterations": 10}
        adapter = LoongFlowAdapter(config)
        adapter.available = True

        # Mock agent
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value={
            "best_solution": "optimized code",
            "best_fitness": 0.9,
            "total_evaluations": 30,
            "improvement_rate": 0.2,
            "iterations_performed": 10
        })
        adapter.pes_agent = mock_agent

        # Run with all parameters
        result = await adapter.evolve(
            problem="Complex optimization problem",
            domain="code",
            initial_code="def func(): pass",
            custom_param="value"
        )

        # Verify call was made with correct data
        assert mock_agent.run.called
        call_args = mock_agent.run.call_args
        problem_data = call_args[0][0]

        assert problem_data["description"] == "Complex optimization problem"
        assert problem_data["domain"] == "code"
        assert problem_data["initial_code"] == "def func(): pass"
        assert problem_data["custom_param"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
