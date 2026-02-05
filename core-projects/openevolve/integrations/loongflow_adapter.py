"""
LoongFlow PES Integration Wrapper

Adapts LoongFlow's PES system to work with OpenEvolve's evolutionary framework.
This wrapper provides a clean interface for using LoongFlow's Plan-Execute-Summarize
agent within the OpenEvolve ecosystem.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LoongFlowAdapter:
    """
    Adapter for LoongFlow's PES (Plan-Execute-Summarize) system.

    This adapter wraps LoongFlow's PESAgent and adapts it to OpenEvolve's interface
    and configuration format. It provides seamless integration between the two systems
    while handling errors gracefully and providing fallback modes.

    Attributes:
        config: OpenEvolve-style configuration dictionary
        pes_agent: LoongFlow PES agent instance (None if not available)
        available: Whether LoongFlow is successfully initialized

    Example:
        >>> config = {
        ...     "max_iterations": 50,
        ...     "population_size": 10,
        ...     "enable_planning": True
        ... }
        >>> adapter = LoongFlowAdapter(config)
        >>> result = await adapter.evolve(
        ...     problem="Optimize function: f(x) = x^2",
        ...     domain="math"
        ... )
        >>> print(f"Best fitness: {result['best_fitness']}")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LoongFlow adapter.

        Args:
            config: OpenEvolve-style configuration dictionary with keys:
                - max_iterations: Maximum number of evolution iterations
                - population_size: Size of population for evolution
                - enable_planning: Whether to enable planning phase
                - enable_memory: Whether to enable memory system
                - llm_config: LLM configuration dict
                - timeout: Timeout in seconds for evolution

        The adapter will attempt to initialize LoongFlow's PES agent.
        If LoongFlow is not available, it will operate in fallback mode.
        """
        self.config = config
        self.pes_agent = None
        self.available = False

        # Initialize PES agent
        self._initialize_pes_agent()

    def _initialize_pes_agent(self):
        """
        Initialize LoongFlow PES agent with mapped config.

        Attempts to import and initialize LoongFlow's GeneralPESAgent.
        If LoongFlow is not installed or initialization fails, sets available=False.
        """
        try:
            from loongflow.agents.general_agent import GeneralEvolveAgent

            # Map OpenEvolve config to LoongFlow format
            pes_config = self._map_config(self.config)

            # Initialize PES agent
            self.pes_agent = GeneralEvolveAgent(config=pes_config)
            self.available = True

            logger.info("[OK] LoongFlow PES agent initialized successfully")

        except ImportError as e:
            logger.warning(f"[WARN]  LoongFlow not available: {e}")
            logger.info("   Will use fallback mode if needed")
            self.available = False

        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize LoongFlow: {e}")
            logger.info("   Will use fallback mode if needed")
            self.available = False

    def _map_config(self, oe_config: Dict) -> Dict:
        """
        Map OpenEvolve configuration to LoongFlow format.

        Converts between OpenEvolve's configuration schema and LoongFlow's
        expected configuration format.

        OpenEvolve params -> LoongFlow params:
            - max_iterations -> max_iterations
            - population_size -> population_size
            - enable_planning -> enable_planning
            - enable_memory -> enable_memory
            - llm_config -> llm_config
            - timeout -> timeout

        Args:
            oe_config: OpenEvolve configuration dictionary

        Returns:
            LoongFlow-format configuration dictionary
        """
        pes_config = {
            # Core parameters
            "max_iterations": oe_config.get("max_iterations", 100),
            "population_size": oe_config.get("population_size", 20),
            "timeout": oe_config.get("timeout", 300),

            # PES-specific features
            "enable_planning": oe_config.get("enable_planning", True),
            "enable_memory": oe_config.get("enable_memory", True),

            # LLM configuration
            "llm_config": oe_config.get("llm_config", {}),

            # Evolution configuration
            "evolve": oe_config.get("evolve", {}),
        }

        return pes_config

    async def evolve(
        self,
        problem: str,
        domain: str = "general",
        initial_code: Optional[str] = None,
        evaluator: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run PES evolution using LoongFlow.

        Executes the Plan-Execute-Summarize evolutionary process using
        the LoongFlow agent. Returns results in OpenEvolve format.

        Args:
            problem: Problem description to solve
            domain: Problem domain (math, code, general, etc.)
            initial_code: Optional initial code solution
            evaluator: Optional evaluator function/object
            **kwargs: Additional parameters for evolution

        Returns:
            Dictionary with keys:
                - best_solution: Best solution found (code/program)
                - best_fitness: Fitness score of best solution
                - total_evaluations: Total number of evaluations performed
                - improvement_rate: Rate of improvement over iterations
                - iterations_performed: Number of iterations completed
                - strategy_used: Strategy identifier ("pes")
                - source: Source identifier ("loongflow_pes")
                - error: Error message if evolution failed

        Example:
            >>> result = await adapter.evolve(
            ...     problem="Optimize sorting algorithm",
            ...     domain="code",
            ...     max_iterations=50
            ... )
            >>> print(f"Fitness: {result['best_fitness']}")
        """

        if not self.available or self.pes_agent is None:
            # Fallback: Return mock result
            logger.warning("LoongFlow not available, returning fallback result")
            return {
                "best_solution": None,
                "best_fitness": 0.0,
                "total_evaluations": 0,
                "improvement_rate": 0.0,
                "iterations_performed": 0,
                "strategy_used": "pes",
                "source": "loongflow_pes",
                "error": "LoongFlow not initialized"
            }

        # Prepare problem for LoongFlow
        problem_data = {
            "description": problem,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "initial_code": initial_code,
        }

        # Add any additional kwargs to problem data
        problem_data.update(kwargs)

        # Run PES evolution
        try:
            logger.info(f"Starting LoongFlow evolution for problem: {problem[:50]}...")

            # Call LoongFlow's run method
            # Note: Adapt this based on actual LoongFlow API
            result = await self._run_loongflow_evolution(problem_data, evaluator)

            # Convert LoongFlow result to OpenEvolve format
            return self._convert_result(result)

        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            # Handle errors gracefully
            return {
                "best_solution": None,
                "best_fitness": 0.0,
                "total_evaluations": 0,
                "improvement_rate": 0.0,
                "iterations_performed": 0,
                "strategy_used": "pes",
                "source": "loongflow_pes",
                "error": str(e)
            }

    async def _run_loongflow_evolution(
        self,
        problem_data: Dict[str, Any],
        evaluator: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Execute LoongFlow evolution with proper error handling.

        Args:
            problem_data: Problem description and metadata
            evaluator: Optional evaluator

        Returns:
            Raw result from LoongFlow

        Raises:
            Exception: If evolution fails
        """
        # This is a placeholder implementation
        # The actual implementation will depend on LoongFlow's API

        # If LoongFlow has an async run method:
        if hasattr(self.pes_agent, 'run'):
            result = await self.pes_agent.run(problem_data)
            return result

        # If LoongFlow has a sync run method:
        elif hasattr(self.pes_agent, 'run_sync'):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.pes_agent.run_sync,
                problem_data
            )
            return result

        # Otherwise, try to adapt the call
        else:
            raise NotImplementedError(
                "Unable to determine LoongFlow's execution interface. "
                "Please check LoongFlow documentation for the correct method."
            )

    def _convert_result(self, loongflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert LoongFlow result format to OpenEvolve format.

        Args:
            loongflow_result: Result from LoongFlow

        Returns:
            OpenEvolve-format result dictionary
        """
        return {
            "best_solution": loongflow_result.get("best_solution"),
            "best_fitness": loongflow_result.get("best_fitness", 0.0),
            "total_evaluations": loongflow_result.get("total_evaluations", 0),
            "improvement_rate": loongflow_result.get("improvement_rate", 0.0),
            "iterations_performed": loongflow_result.get("iterations_performed", 0),
            "strategy_used": "pes",
            "source": "loongflow_pes",
            "metadata": loongflow_result.get("metadata", {})
        }

    def is_available(self) -> bool:
        """
        Check if LoongFlow is available and initialized.

        Returns:
            True if LoongFlow is available, False otherwise
        """
        return self.available

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the LoongFlow adapter.

        Returns:
            Dictionary with capabilities:
                - available: Whether LoongFlow is available
                - supports_planning: Whether planning phase is supported
                - supports_memory: Whether memory system is supported
                - supported_domains: List of supported domains
        """
        return {
            "available": self.available,
            "supports_planning": self.config.get("enable_planning", True),
            "supports_memory": self.config.get("enable_memory", True),
            "supported_domains": [
                "general",
                "math",
                "code",
                "ml"
            ]
        }

    def __repr__(self) -> str:
        """String representation of the adapter."""
        status = "available" if self.available else "unavailable"
        return f"LoongFlowAdapter(status={status}, config={self.config})"
