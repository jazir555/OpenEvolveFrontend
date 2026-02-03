"""
Base Financial Evolution Agent for DeFi Vertical

This module provides the base class for financial evolution agents when the
core FinancialEvolutionAgent is not available.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class MemorySystem:
    """Simple memory system for evolution agents"""

    def __init__(self):
        self._storage = {}

    def get_defi_exploits(self) -> Dict[str, Any]:
        """Get stored DeFi exploits"""
        from openevolve.finance.verticals.defi.historical_exploits import HISTORICAL_EXPLOITS
        return HISTORICAL_EXPLOITS

    def store(self, key: str, value: Any):
        """Store value in memory"""
        self._storage[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from memory"""
        return self._storage.get(key)


class FinancialEvolutionAgent:
    """
    Base class for financial evolution agents.

    Provides common functionality for evolution algorithms including:
    - Configuration management
    - Logging
    - Memory system
    - Optional LoongFlow integration
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize financial evolution agent.

        Args:
            config: Configuration dictionary with optional keys:
                - population_size: Size of population (default: 100)
                - generations: Number of generations (default: 50)
                - mutation_rate: Mutation rate (default: 0.2)
                - elitism_rate: Elitism rate (default: 0.1)
                - use_loongflow: Whether to use LoongFlow (default: False)
                - loongflow_config: LoongFlow configuration (optional)
        """
        self.config = config
        self.population_size = config.get("population_size", 100)
        self.generations = config.get("generations", 50)
        self.mutation_rate = config.get("mutation_rate", 0.2)
        self.elitism_rate = config.get("elitism_rate", 0.1)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Setup memory
        self.memory = MemorySystem()

        # Setup LoongFlow (optional)
        self.loongflow = None
        if config.get("use_loongflow", False):
            self._setup_loongflow(config.get("loongflow_config", {}))

    def _setup_loongflow(self, loongflow_config: Dict[str, Any]):
        """
        Setup LoongFlow integration.

        Args:
            loongflow_config: Configuration for LoongFlow
        """
        try:
            # Try to import and setup LoongFlow
            from openevolve.integration.loongflow_adapter import LoongFlowAdapter

            self.loongflow = LoongFlowAdapter(loongflow_config)
            self.logger.info("LoongFlow integration enabled")
        except ImportError:
            self.logger.warning("LoongFlow not available, using fallback")
            self.loongflow = None
        except Exception as e:
            self.logger.error(f"Failed to setup LoongFlow: {e}")
            self.loongflow = None

    async def plan(self, task: str, prompt: str) -> Any:
        """
        Plan using LoongFlow or fallback.

        Args:
            task: Task description
            prompt: Planning prompt

        Returns:
            Planning result
        """
        if self.loongflow:
            try:
                return await self.loongflow.plan(task=task, prompt=prompt)
            except Exception as e:
                self.logger.error(f"LoongFlow planning failed: {e}")

        # Fallback: return simple result
        return type('PlanningResult', (), {
            'scenarios': [],
            'plan': f"Fallback plan for {task}"
        })()

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
