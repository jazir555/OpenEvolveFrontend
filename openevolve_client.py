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
        
        # Initialize parameter manager for 272 parameter support
        try:
            from parameter_manager import ParameterManager
            self.parameter_manager = ParameterManager()
            self.logger.info(f"Parameter manager initialized with {len(self.parameter_manager.schema.parameters)} parameters")
        except ImportError:
            self.logger.warning("Parameter manager not available - limited parameter validation")
        
        # Initialize metrics collector
        try:
            from metrics_collector import MetricsCollector
            self.metrics_collector = MetricsCollector()
        except ImportError:
            self.logger.warning("Metrics collector not available - limited metrics collection")
        
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
            # Comprehensive parameter validation for all 272 parameters
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
                
                # Log parameter usage statistics
                used_params = len([k for k in kwargs.keys() if k in self.parameter_manager.schema.parameters])
                total_params = len(self.parameter_manager.schema.parameters)
                self.logger.info(f"Using {used_params}/{total_params} available parameters ({used_params/total_params*100:.1f}%)")
                
                # Log warnings for unused parameters
                if validation.warnings:
                    for warning in validation.warnings:
                        self.logger.warning(f"Parameter warning: {warning}")
            else:
                # Basic validation without parameter manager
                self.logger.warning("No parameter manager available - skipping comprehensive validation")
            
            # Prepare OpenEvolve configuration
            config = self._prepare_config(evolution_mode, content_type, evaluator, **kwargs)
            
            # Create temporary file for content
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            try:
                # Filter parameters for OpenEvolve API call
                filtered_kwargs = self._filter_openevolve_parameters(kwargs)
                
                # Run evolution
                result = openevolve_run_evolution(
                    initial_program=temp_path,
                    evaluator=evaluator if evaluator else self._default_evaluator,
                    config=config,
                    **filtered_kwargs
                )
                
                # Extract results
                best_code = result.best_code if hasattr(result, 'best_code') else content
                best_score = result.best_fitness if hasattr(result, 'best_fitness') else 0.0
                iterations = result.generation if hasattr(result, 'generation') else 0
                
                # Collect comprehensive metrics
                metrics = self._extract_metrics(result, start_time, evolution_mode, kwargs)
                
                # Store metrics with enhanced tracking
                if self.metrics_collector:
                    self.metrics_collector.collect(operation_id, metrics)
                    self.metrics_collector.track_evolution_mode(evolution_mode)
                    self.metrics_collector.track_parameter_usage(kwargs)
                
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
            # Use comprehensive error handling
            from error_handler import handle_error, ErrorSeverity, ErrorCategory
            
            error_info = handle_error(
                error=e,
                context={
                    "function": "OpenEvolveClient.evolve",
                    "evolution_mode": evolution_mode,
                    "content_type": content_type,
                    "content_length": len(content)
                },
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.API_ERROR if 'api' in str(e).lower() else ErrorCategory.PROCESSING_ERROR
            )
            
            self.logger.error(f"Evolution failed: {error_info.message}")
            
            # Try fallback
            if self.fallback_handler:
                return self.fallback_handler.get_fallback_result("evolution", {
                    "content": content,
                    "evolution_mode": evolution_mode,
                    "content_type": content_type,
                    "error": error_info.message,
                    "error_details": error_info.__dict__
                })
            
            return EvolutionResult(
                success=False,
                best_code=content,
                best_score=0.0,
                iterations_completed=0,
                metrics={},
                error=error_info.message
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
        
        # Configure LLM - Always provide at least one model configuration
        api_key = kwargs.get('api_key')
        if not api_key:
            # Try to get from environment or config
            import os
            api_key = os.getenv('OPENAI_API_KEY') or self.config.get('api_key')
        
        if api_key:
            llm_config = LLMModelConfig(
                name=kwargs.get('model_name', 'gpt-4'),
                api_key=api_key,
                api_base=kwargs.get('api_base', 'https://api.openai.com/v1'),
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            config.llm.models = [llm_config]
        else:
            # Create a fallback configuration for testing
            self.logger.warning("No API key provided, creating fallback configuration")
            fallback_config = LLMModelConfig(
                name='fallback-model',
                api_key='fallback-key',
                api_base='http://localhost:8000/v1',  # Local fallback
                temperature=0.7,
                max_tokens=2048
            )
            config.llm.models = [fallback_config]
        
        # Set basic parameters
        config.max_iterations = kwargs.get('max_iterations', 10)
        config.database.population_size = kwargs.get('population_size', 20)
        
        # Set mode-specific parameters with safe attribute checking
        if evolution_mode == 'quality_diversity':
            if hasattr(config.database, 'archive_size'):
                config.database.archive_size = kwargs.get('archive_size', 100)
            if hasattr(config.database, 'feature_dimensions'):
                config.database.feature_dimensions = kwargs.get('feature_dimensions', [])
            if hasattr(config.database, 'feature_bins'):
                config.database.feature_bins = kwargs.get('feature_bins', 10)
        
        elif evolution_mode == 'multi_objective':
            # Multi-objective specific config
            if hasattr(config, 'multi_objective'):
                if hasattr(config.multi_objective, 'objectives'):
                    config.multi_objective.objectives = kwargs.get('objectives', ['fitness'])
                if hasattr(config.multi_objective, 'weights'):
                    config.multi_objective.weights = kwargs.get('objective_weights', [1.0])
        
        elif evolution_mode == 'adversarial':
            # Adversarial specific config
            if hasattr(config, 'adversarial'):
                if hasattr(config.adversarial, 'attack_types'):
                    config.adversarial.attack_types = kwargs.get('attack_types', ['mutation'])
                if hasattr(config.adversarial, 'defense_strategies'):
                    config.adversarial.defense_strategies = kwargs.get('defense_strategies', ['validation'])
        
        # Additional core parameters with safe attribute checking
        if hasattr(config, 'seed'):
            config.seed = kwargs.get('seed') or kwargs.get('random_seed', 42)
        
        # Selection and reproduction parameters
        if hasattr(config, 'selection'):
            if hasattr(config.selection, 'tournament_size'):
                config.selection.tournament_size = kwargs.get('tournament_size', 3)
            if hasattr(config.selection, 'selection_pressure'):
                config.selection.selection_pressure = kwargs.get('selection_pressure', 2.0)
        
        # Evaluation parameters
        if hasattr(config, 'evaluation'):
            if hasattr(config.evaluation, 'parallel_evaluations'):
                config.evaluation.parallel_evaluations = kwargs.get('parallel_evaluations', 4)
            if hasattr(config.evaluation, 'evaluator_timeout'):
                config.evaluation.evaluator_timeout = kwargs.get('evaluator_timeout', 300)
        
        # Validate configuration
        if not config.llm.models:
            raise ValueError("No LLM models configured. Please provide an API key or configure fallback models.")
        
        return config
    
    def _filter_openevolve_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to only include those supported by OpenEvolve API.
        This prevents 'unexpected keyword argument' errors.
        """
        # Define parameters that are safe to pass to openevolve_run_evolution
        # Based on actual OpenEvolve API signature
        supported_params = {
            'iterations', 'output_dir', 'verbose', 'log_level', 
            'save_intermediate', 'resume_from'
        }
        
        filtered = {}
        for key, value in kwargs.items():
            if key in supported_params:
                filtered[key] = value
        
        # Always include iterations if max_iterations is provided
        if 'max_iterations' in kwargs and 'iterations' not in filtered:
            filtered['iterations'] = kwargs['max_iterations']
        
        # Set safe defaults
        filtered.setdefault('cleanup', True)
        filtered.setdefault('iterations', 10)
        
        return filtered
    
    def create_config_with_validation(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gpt-4',
        evolution_mode: str = 'standard',
        **kwargs
    ) -> Config:
        """
        Create a validated OpenEvolve configuration.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name to use
            evolution_mode: Evolution mode
            **kwargs: Additional configuration parameters
            
        Returns:
            Validated Config object
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not api_key:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            
        if not api_key:
            raise ValueError(
                "No API key provided. Please provide api_key parameter or set OPENAI_API_KEY environment variable."
            )
        
        config = Config()
        config.evolution_mode = evolution_mode
        
        # Configure LLM
        llm_config = LLMModelConfig(
            name=model_name,
            api_key=api_key,
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
            config.multi_objective.objectives = kwargs.get('objectives', ['fitness'])
            config.multi_objective.weights = kwargs.get('objective_weights', [1.0])
        
        elif evolution_mode == 'adversarial':
            config.adversarial.attack_types = kwargs.get('attack_types', ['mutation'])
            config.adversarial.defense_strategies = kwargs.get('defense_strategies', ['validation'])
        
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
    
    def _extract_metrics(self, result: Any, start_time: float, evolution_mode: str = None, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract comprehensive metrics from evolution result"""
        end_time = time.time()
        
        metrics = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "iterations_completed": getattr(result, 'generation', 0),
            "evolution_mode": evolution_mode,
            "parameters_used": len(kwargs) if kwargs else 0,
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
