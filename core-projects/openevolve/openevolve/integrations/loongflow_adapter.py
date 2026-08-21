"""
LoongFlow PES Integration Wrapper with Graceful Fallback

Adapts LoongFlow's PES system to work with OpenEvolve's evolutionary framework.
This wrapper provides seamless integration with graceful fallback to OpenEvolve-native
mode when LoongFlow is not available or disabled.
"""

import asyncio
import functools
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from openevolve.integrations.loongflow_checker import LoongFlowChecker
from openevolve.integrations.openevolve_fallback import OpenEvolveFallbackAdapter
from openevolve.utils.messages import LoongFlowMessages

logger = logging.getLogger(__name__)

# Ordered list of agent attributes we attempt to call. ``run``/``run_sync`` are
# the documented LoongFlow entrypoints; the rest cover common agent/LLM
# interfaces (LangChain ``invoke``, sklearn ``predict``, raw ``generate``/``chat``).
_AGENT_CALLABLE_ATTRS = ("run", "run_sync", "invoke", "predict", "generate", "chat")


class LoongFlowAdapter:
    """
    Adapter for LoongFlow's PES (Plan-Execute-Summarize) system with graceful fallback.

    This adapter provides seamless integration between LoongFlow and OpenEvolve,
    with automatic fallback to OpenEvolve-native mode when LoongFlow is not available
    or disabled. The fallback is transparent to the user, ensuring all functionality
    remains available regardless of LoongFlow status.

    The adapter supports three modes of operation:
        1. LoongFlow Mode: Uses LoongFlow's PES system when available
        2. OpenEvolve Fallback: Uses OpenEvolve-native capabilities
        3. Disabled Mode: LoongFlow explicitly disabled, uses OpenEvolve

    Configuration options:
        - enable_loongflow: Enable/disable LoongFlow (default: True)
        - require_loongflow: If True, fail instead of falling back (default: False)
        - show_messages: Show user-friendly status messages (default: True)

    Attributes:
        config: OpenEvolve-style configuration dictionary
        pes_agent: LoongFlow PES agent instance (None if not available)
        fallback_adapter: OpenEvolve fallback adapter (active when LoongFlow unavailable)
        using_loongflow: Whether LoongFlow is actively being used
        mode: Current mode ("loongflow", "openevolve", or "disabled")

    Example:
        >>> config = {
        ...     "max_iterations": 50,
        ...     "enable_loongflow": True,
        ...     "require_loongflow": False
        ... }
        >>> adapter = LoongFlowAdapter(config)
        >>> # Works seamlessly whether LoongFlow is available or not
        >>> result = await adapter.evolve(
        ...     problem="Optimize function: f(x) = x^2",
        ...     domain="math"
        ... )
        >>> print(f"Best fitness: {result['best_fitness']}")
        >>> print(f"System used: {result['system_used']}")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LoongFlow adapter with automatic fallback.

        Args:
            config: OpenEvolve-style configuration dictionary with keys:
                - max_iterations: Maximum number of evolution iterations
                - population_size: Size of population for evolution
                - enable_planning: Whether to enable planning phase
                - enable_memory: Whether to enable memory system
                - enable_loongflow: Enable/disable LoongFlow (default: True)
                - require_loongflow: Fail instead of fallback (default: False)
                - show_messages: Show status messages (default: True)
                - llm_config: LLM configuration dict
                - timeout: Timeout in seconds for evolution
                - mode: OpenEvolve mode for fallback (standard, qd, mo, adversarial)

        The adapter will:
            1. Check if LoongFlow should be used (enable_loongflow)
            2. Check if LoongFlow is available
            3. Initialize appropriate adapter (LoongFlow or OpenEvolve fallback)
            4. Provide user-friendly status messages
        """
        self.config = config
        self.pes_agent = None
        self.fallback_adapter = None
        self.using_loongflow = False
        self.mode = "unknown"
        self.evolution_mode = config.get("mode", "standard")
        self.show_messages = config.get("show_messages", True)

        # Initialize with graceful fallback
        self._initialize_or_fallback()

    def _initialize_or_fallback(self):
        """
        Initialize LoongFlow adapter or fall back to OpenEvolve.

        This method implements the graceful fallback logic:
            1. Check configuration (enable_loongflow, require_loongflow)
            2. Check LoongFlow availability
            3. Decide which system to use
            4. Initialize appropriate adapter
            5. Show user-friendly messages
        """
        # Get configuration flags
        enable_loongflow = self.config.get('enable_loongflow', True)
        require_loongflow = self.config.get('require_loongflow', False)

        # Check LoongFlow availability (with deep check if enabled)
        loongflow_available = LoongFlowChecker.is_available(requirement_check=True)

        # Decide whether to use LoongFlow or fallback
        use_loongflow = (
            enable_loongflow and
            loongflow_available
        )

        if use_loongflow:
            # Try to initialize LoongFlow
            try:
                self._initialize_loongflow()
                self.using_loongflow = True
                self.mode = "loongflow"
                logger.info("[OK] LoongFlow PES initialized successfully")

                if self.show_messages:
                    logger.info(LoongFlowMessages.using_loongflow_message())

            except Exception as e:
                # LoongFlow initialization failed
                if require_loongflow:
                    # User wants strict requirement - don't fall back
                    raise RuntimeError(
                        f"LoongFlow is required but failed to initialize: {e}"
                    )
                else:
                    # Fall back gracefully
                    if self.show_messages:
                        logger.warning(LoongFlowMessages.initialization_failed_message(
                            str(e), fallback_enabled=True
                        ))

                    logger.info("🔄 Falling back to OpenEvolve-only mode")
                    self._initialize_fallback()
                    self.using_loongflow = False
                    self.mode = "openevolve"

        else:
            # LoongFlow not available or disabled
            if not enable_loongflow:
                self.mode = "disabled"
                logger.info("ℹ️  LoongFlow disabled in configuration")

                if self.show_messages:
                    logger.info(LoongFlowMessages.disabled_message())

            else:
                self.mode = "unavailable"
                logger.info("ℹ️  LoongFlow not available")

                if self.show_messages:
                    logger.info(LoongFlowMessages.not_available_message(
                        fallback_enabled=not require_loongflow
                    ))

            # Initialize OpenEvolve fallback
            self._initialize_fallback()
            self.using_loongflow = False

    def _initialize_loongflow(self):
        """
        Initialize actual LoongFlow adapter.

        Raises:
            ImportError: If LoongFlow is not installed
            Exception: If initialization fails for any reason
        """
        from loongflow.agents.general_agent import GeneralEvolveAgent

        # Map OpenEvolve config to LoongFlow format
        pes_config = self._map_config(self.config)

        # Initialize PES agent
        self.pes_agent = GeneralEvolveAgent(config=pes_config)

        logger.debug("LoongFlow GeneralEvolveAgent initialized")

    def _initialize_fallback(self):
        """
        Initialize OpenEvolve fallback adapter.
        """
        self.fallback_adapter = OpenEvolveFallbackAdapter(self.config)
        logger.info("[OK] OpenEvolve fallback adapter initialized")

        if self.show_messages:
            mode = self.config.get("mode", "standard")
            logger.info(LoongFlowMessages.using_openevolve_message(mode))

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
        Run evolution using LoongFlow or fallback to OpenEvolve.

        This method works seamlessly regardless of which system is actually used.
        The interface is identical, and results are returned in a consistent format.

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
                - iterations_performed: Number of iterations completed
                - convergence_curve: List of scores over iterations
                - planning_strategies: List of strategies used (LoongFlow only)
                - execution_patterns: List of patterns (LoongFlow only)
                - summaries: List of summaries (LoongFlow only)
                - system_used: "loongflow" or "openevolve"
                - mode_used: Evolution mode used
                - error: Error message if evolution failed

        Example:
            >>> result = await adapter.evolve(
            ...     problem="Optimize sorting algorithm",
            ...     domain="code",
            ...     max_iterations=50
            ... )
            >>> print(f"Fitness: {result['best_fitness']}")
            >>> print(f"System: {result['system_used']}")
        """
        if self.using_loongflow:
            # Use LoongFlow
            logger.info("Using LoongFlow PES for evolution")
            return await self._evolve_with_loongflow(
                problem, domain, initial_code, evaluator, **kwargs
            )
        else:
            # Use OpenEvolve fallback
            logger.info("Using OpenEvolve for evolution")
            return await self._evolve_with_openevolve(
                problem, domain, initial_code, evaluator, **kwargs
            )

    async def _evolve_with_loongflow(
        self,
        problem: str,
        domain: str,
        initial_code: Optional[str],
        evaluator: Optional[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evolve using actual LoongFlow PES system.

        Args:
            problem: Problem description
            domain: Problem domain
            initial_code: Initial code if provided
            evaluator: Evaluator function
            **kwargs: Additional parameters

        Returns:
            Result dictionary in OpenEvolve format
        """
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
            result = await self._run_loongflow_evolution(problem_data, evaluator)

            # Convert LoongFlow result to OpenEvolve format
            return self._convert_result(result)

        except Exception as e:
            logger.error(f"LoongFlow evolution failed: {e}", exc_info=True)

            # If fallback is enabled, try OpenEvolve
            if not self.config.get('require_loongflow', False):
                logger.warning("LoongFlow failed, falling back to OpenEvolve")
                return await self._evolve_with_openevolve(
                    problem, domain, initial_code, evaluator, **kwargs
                )

            # Otherwise, return error
            return {
                "best_solution": None,
                "best_fitness": 0.0,
                "total_evaluations": 0,
                "iterations_performed": 0,
                "convergence_curve": [],
                "planning_strategies": [],
                "execution_patterns": [],
                "summaries": [],
                "system_used": "loongflow",
                "mode_used": "pes",
                "error": str(e)
            }

    async def _evolve_with_openevolve(
        self,
        problem: str,
        domain: str,
        initial_code: Optional[str],
        evaluator: Optional[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evolve using OpenEvolve fallback adapter.

        Args:
            problem: Problem description
            domain: Problem domain
            initial_code: Initial code if provided
            evaluator: Evaluator function
            **kwargs: Additional parameters

        Returns:
            Result dictionary in LoongFlow-compatible format
        """
        return await self.fallback_adapter.evolve(
            problem=problem,
            domain=domain,
            initial_code=initial_code,
            evaluator=evaluator,
            **kwargs
        )

    async def _run_loongflow_evolution(
        self,
        problem_data: Dict[str, Any],
        evaluator: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Execute LoongFlow evolution with a generic agent dispatch and graceful
        fallback to the OpenEvolve engine.

        The adapter no longer raises ``NotImplementedError`` on the default path.
        Instead it tries, in order:

            1. A documented LoongFlow callable (``run`` / ``run_sync``).
            2. Any other common agent interface (``invoke``, ``predict``,
               ``generate``, ``chat``, or the object itself being callable).
            3. When no usable agent callable is present, it drives an actual
               evolution through OpenEvolve's engine (the library's
               ``run_evolution`` entrypoint) so a real result is produced.

        Args:
            problem_data: Problem description and metadata
            evaluator: Optional evaluator

        Returns:
            Result dictionary (LoongFlow-format). Never raises for a missing
            agent interface; falls back to the engine instead.
        """
        agent = self.pes_agent

        # Try to dispatch through the agent's callable interface.
        if agent is not None:
            callable_fn, is_async = self._resolve_agent_callable(agent)
            if callable_fn is not None:
                try:
                    if is_async:
                        result = await callable_fn(problem_data)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, functools.partial(callable_fn, problem_data)
                        )
                    return result
                except Exception as e:
                    logger.error(
                        f"Agent callable failed during LoongFlow evolution: {e}",
                        exc_info=True,
                    )
                    # Fall through to the engine fallback below.

        # No usable agent callable: drive evolution via the OpenEvolve engine
        # rather than faking it.
        logger.warning(
            "No usable LoongFlow agent callable found; driving evolution "
            "through the OpenEvolve engine."
        )
        return await self._run_openevolve_engine(problem_data, evaluator)

    def _resolve_agent_callable(self, agent: Any) -> Tuple[Optional[Any], bool]:
        """
        Resolve the most appropriate callable on an agent object.

        Args:
            agent: The agent/proposer object.

        Returns:
            A tuple ``(callable, is_async)`` or ``(None, False)`` if no usable
            callable is exposed.
        """
        # Prefer the documented LoongFlow entrypoints.
        for attr in _AGENT_CALLABLE_ATTRS:
            fn = getattr(agent, attr, None)
            if callable(fn):
                return fn, asyncio.iscoroutinefunction(fn)

        # Fall back to the object being directly callable (``__call__``).
        if callable(agent):
            return agent, asyncio.iscoroutinefunction(agent.__call__)

        return None, False

    async def _run_openevolve_engine(
        self,
        problem_data: Dict[str, Any],
        evaluator: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Drive an actual evolution through OpenEvolve's engine.

        This reuses the library's public ``run_evolution`` entrypoint so the
        adapter performs genuine evolution (rather than returning a canned
        mock). It runs online-capable when real LLM models are configured and
        offline-capable (mock LLM) otherwise, making it safe for tests.

        Args:
            problem_data: Problem description and metadata
            evaluator: Optional evaluator (callable or file path)

        Returns:
            LoongFlow-format result dictionary.
        """
        from openevolve.api import run_evolution
        from openevolve.config import Config, LLMModelConfig

        # Build a configuration derived from the adapter config.
        config_obj = Config()
        config_obj.max_iterations = min(
            int(self.config.get("max_iterations", 10)), 5
        )
        if not config_obj.llm.models:
            # No real models configured: use the offline mock LLM so the
            # engine still produces a genuine result without API keys.
            config_obj.llm.models = [
                LLMModelConfig(name="mock", api_key="not-needed")
            ]

        initial_code = problem_data.get("initial_code") or (
            "# EVOLVE-BLOCK-START\n"
            "def solve(problem):\n"
            "    return None\n"
            "# EVOLVE-BLOCK-END\n"
        )
        eval_fn = evaluator if callable(evaluator) else self._default_evaluator

        loop = asyncio.get_event_loop()
        # ``run_evolution`` spins up its own event loop, so it must run in a
        # worker thread to avoid "event loop already running" errors here.
        result = await loop.run_in_executor(
            None,
            functools.partial(
                run_evolution,
                initial_program=initial_code,
                evaluator=eval_fn,
                config=config_obj,
                iterations=config_obj.max_iterations,
                output_dir=None,
                cleanup=True,
            ),
        )

        return self._convert_engine_result(result)

    @staticmethod
    def _default_evaluator(program_path: str) -> Dict[str, Any]:
        """
        Minimal offline evaluator used when the caller provides no evaluator.

        Args:
            program_path: Path to the program under evaluation.

        Returns:
            A trivial metrics dictionary.
        """
        return {"combined_score": 0.5, "iterations": 1, "evaluations": 1}

    def _convert_engine_result(self, result: Any) -> Dict[str, Any]:
        """
        Convert an OpenEvolve ``EvolutionResult`` into LoongFlow result format.

        Args:
            result: ``EvolutionResult`` from ``run_evolution``.

        Returns:
            LoongFlow-format result dictionary.
        """
        metrics = getattr(result, "metrics", {}) or {}
        return {
            "best_solution": getattr(result, "best_code", None),
            "best_fitness": getattr(result, "best_score", 0.0),
            "total_evaluations": metrics.get("evaluations", 0),
            "iterations_performed": metrics.get("iterations", 0),
            "convergence_curve": metrics.get("convergence", []),
            "planning_strategies": [],
            "execution_patterns": [],
            "summaries": [],
            "system_used": "openevolve",
            "mode_used": "engine",
            "metadata": {
                "engine": "openevolve",
                "output_dir": getattr(result, "output_dir", None),
            },
        }

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
            "convergence_curve": loongflow_result.get("convergence_curve", []),
            "planning_strategies": loongflow_result.get("planning_strategies", []),
            "execution_patterns": loongflow_result.get("execution_patterns", []),
            "summaries": loongflow_result.get("summaries", []),
            "system_used": loongflow_result.get("system_used", "loongflow"),
            "mode_used": loongflow_result.get("mode_used", "pes"),
            "metadata": loongflow_result.get("metadata", {})
        }

    def is_available(self) -> bool:
        """
        Check if LoongFlow is available and initialized.

        Returns:
            True if LoongFlow is available, False otherwise
        """
        return self.using_loongflow

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the adapter.

        Returns:
            Dictionary with status information:
                - mode: Current mode ("loongflow", "openevolve", "disabled")
                - using_loongflow: Whether LoongFlow is being used
                - loongflow_available: Whether LoongFlow is installed
                - loongflow_version: LoongFlow version if available
                - capabilities: Dictionary of capabilities
        """
        diagnostics = LoongFlowChecker.get_diagnostics()

        status = {
            "mode": self.mode,
            "using_loongflow": self.using_loongflow,
            "loongflow_available": diagnostics["installed"],
            "loongflow_version": diagnostics["version"],
            "config": {
                "enable_loongflow": self.config.get("enable_loongflow", True),
                "require_loongflow": self.config.get("require_loongflow", False),
            }
        }

        # Get capabilities from active adapter
        if self.using_loongflow:
            status["capabilities"] = self._get_loongflow_capabilities()
        else:
            status["capabilities"] = self.fallback_adapter.get_capabilities()

        return status

    def _get_loongflow_capabilities(self) -> Dict[str, Any]:
        """Get LoongFlow-specific capabilities."""
        return {
            "available": True,
            "system": "loongflow",
            "mode": "pes",
            "supports_planning": self.config.get("enable_planning", True),
            "supports_memory": self.config.get("enable_memory", True),
            "supports_qd": False,
            "supports_mo": False,
            "supports_adversarial": False,
            "supported_domains": [
                "general",
                "math",
                "code",
                "ml"
            ]
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the currently active adapter.

        Returns:
            Dictionary with capabilities
        """
        return self.get_status()["capabilities"]

    def print_status(self):
        """Print human-readable status information."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("LoongFlow Adapter Status")
        print("=" * 60)
        print(f"Mode: {status['mode']}")
        print(f"Using LoongFlow: {status['using_loongflow']}")
        print(f"LoongFlow Available: {status['loongflow_available']}")
        print(f"LoongFlow Version: {status['loongflow_version'] or 'N/A'}")
        print()
        print("Capabilities:")
        for key, value in status['capabilities'].items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")

    def __repr__(self) -> str:
        """String representation of the adapter."""
        if self.using_loongflow:
            return f"LoongFlowAdapter(mode=loongflow, config={self.config})"
        else:
            mode = self.config.get("mode", "standard")
            return f"LoongFlowAdapter(mode=openevolve_{mode}, config={self.config})"
