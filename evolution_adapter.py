"""
Evolution Adapter - Clean Interface for Evolution Module

This module provides the EvolutionAdapter class which serves as a clean
interface between UnifiedConfiguration and the evolution module.

The adapter pattern allows us to:
- Separate configuration management from execution logic
- Provide a simple, consistent interface for running evolution
- Support all evolution modes through a single entry point
- Enable easy testing and mocking
- Maintain backward compatibility with existing code

Usage:
    unified_config = create_unified_config({
        'evolution_mode': 'standard',
        'max_iterations': 10,
        'temperature': 0.7
    })

    adapter = EvolutionAdapter(unified_config)
    result = adapter.run_evolution("Initial content to evolve")
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from unified_configuration import UnifiedConfiguration, create_unified_config, merge_configs

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """
    Result from evolution execution.

    Attributes:
        success: Whether evolution completed successfully
        final_content: The evolved content
        original_content: The original input content
        iterations_completed: Number of iterations actually completed
        best_fitness: Best fitness score achieved
        final_fitness: Fitness of final content
        improvement_ratio: Ratio of improvement
        convergence_iteration: Iteration where convergence occurred (if any)
        total_evaluations: Total number of evaluations performed
        duration_seconds: Total execution time in seconds
        evolution_mode: Mode of evolution used
        metrics: Additional metrics from evolution
        error: Error message if execution failed
    """
    success: bool
    final_content: str
    original_content: str
    iterations_completed: int = 0
    best_fitness: float = 0.0
    final_fitness: float = 0.0
    improvement_ratio: float = 0.0
    convergence_iteration: Optional[int] = None
    total_evaluations: int = 0
    duration_seconds: float = 0.0
    evolution_mode: str = "standard"
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class EvolutionAdapter:
    """
    Adapter for evolution module using unified configuration.

    This class provides a clean interface for running evolution with
    UnifiedConfiguration, handling all the complexity of parameter
    management and module execution.

    Attributes:
        config: The UnifiedConfiguration instance
        _evaluator: Optional custom evaluator function
        _status_callback: Optional callback for status updates
    """

    def __init__(
        self,
        config: UnifiedConfiguration,
        evaluator: Optional[Callable] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize EvolutionAdapter.

        Args:
            config: UnifiedConfiguration with all evolution parameters
            evaluator: Optional custom evaluator function
            status_callback: Optional callback for status updates during execution
        """
        self.config = config
        self._evaluator = evaluator
        self._status_callback = status_callback

        # Validate configuration has required parameters
        validation = config.validate()
        if not validation.valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")

        logger.info(f"EvolutionAdapter initialized with mode={config.evolution_mode}")

    def _update_status(self, message: str) -> None:
        """Update status if callback is provided"""
        if self._status_callback:
            self._status_callback(message)
        logger.debug(f"Evolution status: {message}")

    def run_evolution(
        self,
        initial_content: str,
        content_type: str = "document_general",
        **kwargs
    ) -> EvolutionResult:
        """
        Run evolution with the configured parameters.

        Args:
            initial_content: The content to evolve
            content_type: Type of content (e.g., 'code_python', 'document_general')
            **kwargs: Additional parameters to override config temporarily

        Returns:
            EvolutionResult with execution results

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If evolution execution fails critically
        """
        start_time = time.time()
        evolution_mode = self.config.evolution_mode

        self._update_status(f"🚀 Starting {evolution_mode} evolution...")

        # Merge kwargs with config (kwargs take precedence)
        if kwargs:
            effective_config = self.config.merge(kwargs, validate=True)
        else:
            effective_config = self.config

        try:
            # Import evolution module
            from evolution import run_evolution_loop

            # Run evolution loop
            self._update_status(f"🔄 Running {evolution_mode} evolution...")

            final_content = run_evolution_loop(
                current_content=initial_content,
                content_type=content_type,
                config=effective_config.to_evolution_config(),
                evaluator=self._evaluator,
                **kwargs
            )

            # Calculate metrics
            duration = time.time() - start_time

            # Extract metrics from session state if available (Streamlit)
            metrics = self._extract_metrics()

            # Create result
            result = EvolutionResult(
                success=True,
                final_content=final_content,
                original_content=initial_content,
                iterations_completed=metrics.get('iterations_completed', 0),
                best_fitness=metrics.get('best_score', 0.0),
                final_fitness=metrics.get('final_fitness', 0.0),
                improvement_ratio=metrics.get('improvement_ratio', 0.0),
                convergence_iteration=metrics.get('convergence_iteration'),
                total_evaluations=metrics.get('total_evaluations', 0),
                duration_seconds=duration,
                evolution_mode=evolution_mode,
                metrics=metrics
            )

            self._update_status(
                f"✅ Evolution completed! "
                f"Fitness: {result.best_fitness:.4f}, "
                f"Iterations: {result.iterations_completed}, "
                f"Duration: {duration:.2f}s"
            )

            return result

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            duration = time.time() - start_time
            error_msg = f"Evolution failed: {str(e)}"

            self._update_status(f"💥 {error_msg}")
            logger.error(error_msg, exc_info=True)

            return EvolutionResult(
                success=False,
                final_content=initial_content,  # Return original on failure
                original_content=initial_content,
                duration_seconds=duration,
                evolution_mode=evolution_mode,
                error=error_msg
            )

    def _extract_metrics(self) -> Dict[str, Any]:
        """
        Extract metrics from session state or other sources.

        Returns:
            Dictionary of evolution metrics
        """
        metrics = {}

        # Try to extract from Streamlit session state if available
        try:
            import streamlit as st

            if 'evolution_metrics' in st.session_state:
                metrics = st.session_state.evolution_metrics.copy()
        except (ImportError, AttributeError):
            # Streamlit not available or no session state
            pass

        return metrics

    # =========================================================================
    # CONVENIENCE METHODS FOR SPECIFIC EVOLUTION MODES
    # =========================================================================

    def run_standard_evolution(
        self,
        initial_content: str,
        content_type: str = "document_general",
        **kwargs
    ) -> EvolutionResult:
        """
        Run standard evolution with simplified interface.

        Args:
            initial_content: Content to evolve
            content_type: Type of content
            **kwargs: Additional parameters

        Returns:
            EvolutionResult
        """
        # Force standard mode
        kwargs['evolution_mode'] = 'standard'
        return self.run_evolution(initial_content, content_type, **kwargs)

    def run_quality_diversity_evolution(
        self,
        initial_content: str,
        content_type: str = "document_general",
        **kwargs
    ) -> EvolutionResult:
        """
        Run quality diversity evolution (MAP-Elites).

        Args:
            initial_content: Content to evolve
            content_type: Type of content
            **kwargs: Additional parameters (e.g., archive_size, feature_bins)

        Returns:
            EvolutionResult
        """
        kwargs['evolution_mode'] = 'quality_diversity'
        return self.run_evolution(initial_content, content_type, **kwargs)

    def run_multi_objective_evolution(
        self,
        initial_content: str,
        content_type: str = "document_general",
        **kwargs
    ) -> EvolutionResult:
        """
        Run multi-objective evolution.

        Args:
            initial_content: Content to evolve
            content_type: Type of content
            **kwargs: Additional parameters (e.g., objectives, pareto_front_size)

        Returns:
            EvolutionResult
        """
        kwargs['evolution_mode'] = 'multi_objective'
        return self.run_evolution(initial_content, content_type, **kwargs)

    def run_problem_decomposition(
        self,
        initial_content: str,
        content_type: str = "document_general",
        **kwargs
    ) -> EvolutionResult:
        """
        Run problem decomposition evolution.

        Args:
            initial_content: Content to evolve
            content_type: Type of content
            **kwargs: Additional parameters

        Returns:
            EvolutionResult
        """
        kwargs['evolution_mode'] = 'problem_decomposition'
        return self.run_evolution(initial_content, content_type, **kwargs)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_evolution_adapter(
    parameters: Optional[Dict[str, Any]] = None,
    mode: str = "standard",
    evaluator: Optional[Callable] = None,
    status_callback: Optional[Callable[[str], None]] = None
) -> EvolutionAdapter:
    """
    Factory function to create EvolutionAdapter with common presets.

    Args:
        parameters: Custom parameters (uses mode defaults if None)
        mode: Evolution mode preset ('standard', 'quality_diversity', etc.)
        evaluator: Optional custom evaluator
        status_callback: Optional status update callback

    Returns:
        Configured EvolutionAdapter

    Example:
        adapter = create_evolution_adapter(
            mode='standard',
            max_iterations=20,
            temperature=0.8
        )
        result = adapter.run_evolution("My content")
    """
    # Start with mode-specific base parameters
    if mode == 'standard':
        base_params = {
            'evolution_mode': 'standard',
            'max_iterations': 10,
            'population_size': 20,
            'temperature': 0.7
        }
    elif mode == 'quality_diversity':
        base_params = {
            'evolution_mode': 'quality_diversity',
            'archive_size': 100,
            'feature_bins': 10,
            'diversity_weight': 0.5
        }
    elif mode == 'multi_objective':
        base_params = {
            'evolution_mode': 'multi_objective',
            'pareto_front_size': 50,
            'dominance_metric': 'pareto'
        }
    elif mode == 'adversarial':
        base_params = {
            'evolution_mode': 'adversarial',
            'adversarial_rounds': 5,
            'attack_strength': 0.5
        }
    elif mode == 'problem_decomposition':
        base_params = {
            'evolution_mode': 'problem_decomposition',
            'max_iterations': 10
        }
    else:
        base_params = {'evolution_mode': mode}

    # Merge with custom parameters
    if parameters:
        base_params.update(parameters)

    # Create config and adapter
    config = create_unified_config(base_params)

    return EvolutionAdapter(
        config=config,
        evaluator=evaluator,
        status_callback=status_callback
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def run_batch_evolution(
    contents: List[str],
    config: UnifiedConfiguration,
    content_type: str = "document_general",
    status_callback: Optional[Callable[[str], None]] = None
) -> List[EvolutionResult]:
    """
    Run evolution on multiple contents with the same configuration.

    Args:
        contents: List of contents to evolve
        config: UnifiedConfiguration to use for all
        content_type: Type of all contents
        status_callback: Optional status callback

    Returns:
        List of EvolutionResult (one per input)

    Example:
        contents = ["Content 1", "Content 2", "Content 3"]
        config = create_standard_evolution_config(max_iterations=5)
        results = run_batch_evolution(contents, config)
    """
    adapter = EvolutionAdapter(config, status_callback=status_callback)
    results = []

    for i, content in enumerate(contents, 1):
        status_callback(f"Processing item {i}/{len(contents)}...") if status_callback else None
        result = adapter.run_evolution(content, content_type)
        results.append(result)

    return results


# =============================================================================
# ASYNC SUPPORT (Future Enhancement)
# =============================================================================

async def run_evolution_async(
    config: UnifiedConfiguration,
    initial_content: str,
    content_type: str = "document_general",
    **kwargs
) -> EvolutionResult:
    """
    Async version of evolution execution (for future async support).

    Args:
        config: UnifiedConfiguration
        initial_content: Content to evolve
        content_type: Type of content
        **kwargs: Additional parameters

    Returns:
        EvolutionResult

    Note:
        This is a placeholder for future async implementation.
        Currently runs synchronously.
    """
    adapter = EvolutionAdapter(config)
    return adapter.run_evolution(initial_content, content_type, **kwargs)
