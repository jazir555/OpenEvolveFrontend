"""
OpenEvolve Client - Unified interface for all OpenEvolve operations
Provides a clean API for all files to interact with OpenEvolve backend
"""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Import OpenEvolve components
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    logging.warning("OpenEvolve backend not available - fallback mode enabled")


@dataclass
class EvolutionResult:
    """Result from an evolution operation"""
    success: bool
    best_code: str
    best_score: float
    iterations_completed: int
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Result from parameter validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OpenEvolveClient:
    """
    Unified client for OpenEvolve operations.
    Provides a single interface for all files to interact with OpenEvolve.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenEvolve client.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.parameter_manager = None  # Will be set by ParameterManager
        self.metrics_collector = None  # Will be set by MetricsCollector
        self.fallback_handler = None  # Will be set by FallbackHandler
        self.logger = logging.getLogger(__name__)
        
        # Check OpenEvolve availability
        self.available = OPENEVOLVE_AVAILABLE
        if not self.available:
            self.logger.warning("OpenEvolve backend not available - using fallback mode")
    
    def evolve(
        self,
        content: str,
        evolution_mode: str = "standard",
        content_type: str = "general",
        evaluator: Optional[Callable] = None,
        **kwargs
    ) -> EvolutionResult:
        """
        Execute evolution with full parameter support.
        
        Args:
            content: Content to evolve
            evolution_mode: Evolution mode (standard, quality_diversity, multi_objective, adversarial)
            content_type: Type of content being evolved
            evaluator: Custom evaluator function
            **kwargs: Additional OpenEvolve parameters
            
        Returns:
            EvolutionResult with evolved content and metrics
        """
        start_time = time.time()
        operation_id = f"evolve_{int(start_time)}"
        
        self.logger.info(f"Starting evolution operation {operation_id} with mode {evolution_mode}")
        
        # Check availability
        if not self.available:
            self.logger.warning("OpenEvolve not available, using fallback")
            if self.fallback_handler:
                return self.fallback_handler.get_fallback_result("evolution", {
                    "content": content,
                    "evolution_mode": evolution_mode,
                    "content_type": content_type
                })
            else:
                return EvolutionResult(
                    success=False,
                    best_code=content,
                    best_score=0.0,
                    iterations_completed=0,
                    metrics={},
                    error="OpenEvolve not available and no fallback handler configured"
                )
        
        try:
            # Validate parameters
            if self.parameter_manager:
                validation = self.parameter_manager.validate(kwargs)
                if not validation.valid:
                    self.logger.error(f"Parameter validation failed: {validation.errors}")
                    return EvolutionResult(
                        success=False,
                        best_code=content,
                        best_score=0.0,
                        iterations_completed=0,
                        metrics={},
                        error=f"Parameter validation failed: {', '.join(validation.errors)}"
                    )
            
            # Prepare OpenEvolve configuration
            config = self._prepare_config(evolution_mode, content_type, evaluator, **kwargs)
            
            # Create temporary file for content
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            try:
                # Run evolution
                result = openevolve_run_evolution(
                    initial_program=temp_path,
                    evaluator=evaluator if evaluator else self._default_evaluator,
                    config=config,
                    iterations=kwargs.get('max_iterations', 10),
                    output_dir=kwargs.get('output_dir'),
                    cleanup=kwargs.get('cleanup', True)
                )
                
                # Extract results
                best_code = result.best_code if hasattr(result, 'best_code') else content
                best_score = result.best_fitness if hasattr(result, 'best_fitness') else 0.0
                iterations = result.generation if hasattr(result, 'generation') else 0
                
                # Collect metrics
                metrics = self._extract_metrics(result, start_time)
                
                # Store metrics
                if self.metrics_collector:
                    self.metrics_collector.collect(operation_id, metrics)
                
                self.logger.info(f"Evolution completed successfully: {iterations} iterations, score {best_score:.4f}")
                
                return EvolutionResult(
                    success=True,
                    best_code=best_code,
                    best_score=best_score,
                    iterations_completed=iterations,
                    metrics=metrics
                )
                
            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            self.logger.error(f"Evolution failed: {e}", exc_info=True)
            
            # Try fallback
            if self.fallback_handler:
                return self.fallback_handler.get_fallback_result("evolution", {
                    "content": content,
                    "evolution_mode": evolution_mode,
                    "content_type": content_type,
                    "error": str(e)
                })
            
            return EvolutionResult(
                success=False,
                best_code=content,
                best_score=0.0,
                iterations_completed=0,
                metrics={},
                error=str(e)
            )
    
    def get_metrics(self, operation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collected metrics.
        
        Args:
            operation_id: Optional operation ID to get specific metrics
            
        Returns:
            Dictionary of metrics
        """
        if self.metrics_collector:
            if operation_id:
                return self.metrics_collector.get_operation_metrics(operation_id)
            else:
                return self.metrics_collector.get_all_metrics()
        return {}
    
    def validate_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameter configuration.
        
        Args:
            params: Parameters to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        if self.parameter_manager:
            return self.parameter_manager.validate(params)
        
        # Basic validation if no parameter manager
        return ValidationResult(valid=True)
    
    def _prepare_config(
        self,
        evolution_mode: str,
        content_type: str,
        evaluator: Optional[Callable],
        **kwargs
    ) -> Config:
        """Prepare OpenEvolve configuration"""
        config = Config()
        
        # Set evolution mode
        config.evolution_mode = evolution_mode
        
        # Configure LLM
        if 'api_key' in kwargs:
            llm_config = LLMModelConfig(
                name=kwargs.get('model_name', 'gpt-4'),
                api_key=kwargs['api_key'],
                api_base=kwargs.get('api_base', 'https://api.openai.com/v1'),
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            config.llm.models = [llm_config]
        
        # Set basic parameters
        config.max_iterations = kwargs.get('max_iterations', 10)
        config.database.population_size = kwargs.get('population_size', 20)
        
        # Set mode-specific parameters
        if evolution_mode == 'quality_diversity':
            config.database.archive_size = kwargs.get('archive_size', 100)
            config.database.feature_dimensions = kwargs.get('feature_dimensions', [])
            config.database.feature_bins = kwargs.get('feature_bins', 10)
        
        elif evolution_mode == 'multi_objective':
            # Multi-objective specific config
            pass
        
        elif evolution_mode == 'adversarial':
            # Adversarial specific config
            pass
        
        return config
    
    def _default_evaluator(self, program_path: str) -> Dict[str, Any]:
        """Default evaluator if none provided"""
        try:
            with open(program_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple length-based scoring
            score = min(1.0, len(content) / 1000.0)
            
            return {
                "score": score,
                "timestamp": time.time(),
                "length": len(content)
            }
        except Exception as e:
            return {
                "score": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _extract_metrics(self, result: Any, start_time: float) -> Dict[str, Any]:
        """Extract metrics from evolution result"""
        end_time = time.time()
        
        metrics = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "iterations_completed": getattr(result, 'generation', 0),
            "best_fitness": getattr(result, 'best_fitness', 0.0),
            "population_size": getattr(result, 'population_size', 0),
        }
        
        # Add mode-specific metrics
        if hasattr(result, 'archive'):
            metrics["archive_size"] = len(result.archive)
        
        if hasattr(result, 'pareto_front'):
            metrics["pareto_front_size"] = len(result.pareto_front)
        
        return metrics


# Global client instance
_global_client: Optional[OpenEvolveClient] = None


def get_client(config: Optional[Dict[str, Any]] = None) -> OpenEvolveClient:
    """
    Get global OpenEvolve client instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        OpenEvolveClient instance
    """
    global _global_client
    if _global_client is None:
        _global_client = OpenEvolveClient(config)
    return _global_client


def set_client(client: OpenEvolveClient):
    """
    Set global OpenEvolve client instance.
    
    Args:
        client: OpenEvolveClient instance to set as global
    """
    global _global_client
    _global_client = client
