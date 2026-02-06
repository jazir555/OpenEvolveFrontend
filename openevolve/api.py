"""
OpenEvolve API Stub Module

This module provides stub implementations for OpenEvolve API functions.
When the actual OpenEvolve package is available, it will be used instead.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Iterator, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Flag to indicate if this is a stub implementation
IS_STUB = True


@dataclass
class EvolutionResult:
    """Result of an evolution operation."""
    success: bool = False
    best_score: float = 0.0
    iterations: int = 0
    best_individual: Optional[Any] = None
    population: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class Config:
    """Configuration for OpenEvolve."""
    model_configs: List[Any] = field(default_factory=list)
    database_config: Optional[Any] = None
    evaluator_config: Optional[Any] = None
    prompt_config: Optional[Any] = None
    evolution_trace_config: Optional[Any] = None


def run_evolution(
    code: str,
    test_cases: List[Dict[str, Any]],
    config: Optional[Config] = None,
    **kwargs
) -> EvolutionResult:
    """
    Run evolution on code with test cases.
    
    This is a stub implementation that logs a warning and returns a default result.
    """
    logger.warning("OpenEvolve run_evolution is using stub implementation. "
                   "Install the full OpenEvolve package for actual functionality.")
    return EvolutionResult(
        success=False,
        error_message="OpenEvolve not installed - using stub implementation"
    )


def evolve_function(
    func: Callable,
    test_cases: List[Dict[str, Any]],
    config: Optional[Config] = None,
    **kwargs
) -> EvolutionResult:
    """Evolve a function with test cases."""
    logger.warning("OpenEvolve evolve_function is using stub implementation.")
    return EvolutionResult(
        success=False,
        error_message="OpenEvolve not installed - using stub implementation"
    )


def evolve_algorithm(
    algorithm_description: str,
    test_cases: List[Dict[str, Any]],
    config: Optional[Config] = None,
    **kwargs
) -> EvolutionResult:
    """Evolve an algorithm from description."""
    logger.warning("OpenEvolve evolve_algorithm is using stub implementation.")
    return EvolutionResult(
        success=False,
        error_message="OpenEvolve not installed - using stub implementation"
    )


def evolve_code(
    code: str,
    specification: str,
    config: Optional[Config] = None,
    **kwargs
) -> EvolutionResult:
    """Evolve code based on specification."""
    logger.warning("OpenEvolve evolve_code is using stub implementation.")
    return EvolutionResult(
        success=False,
        error_message="OpenEvolve not installed - using stub implementation"
    )
