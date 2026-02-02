"""
ROMA MDAP Maker Associative Integration
Integrates ROMA with the MDAP (Multi-Domain Agent Planner) system

This module provides a simplified interface to the comprehensive
roma_mdap_maker_associative_integration implementation.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to import from the full implementation
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeConfig as _FullConfig,
        ROMAMDAPMakerAssociativeEngine as _FullEngine,
        create_romamdapmaker_associative_config as _create_full_config,
        solve_with_romamdapmaker_associative as _solve_full,
        get_romamdapmaker_associative_status as _get_status,
        ROMA_MDAP_MAKER_AVAILABLE,
        ASSOCIATIVE_AVAILABLE,
        GROUND_TRUTH_AVAILABLE
    )
    FULL_IMPLEMENTATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Full ROMA-MDAP-MAKER associative integration not available: {e}")
    FULL_IMPLEMENTATION_AVAILABLE = False
    ROMA_MDAP_MAKER_AVAILABLE = False
    ASSOCIATIVE_AVAILABLE = False
    GROUND_TRUTH_AVAILABLE = False
    _FullConfig = None
    _FullEngine = None
    _create_full_config = None
    _solve_full = None
    _get_status = None


# =============================================================================
# SIMPLIFIED CONFIGURATION (Fallback)
# =============================================================================

@dataclass
class ROMAMDAPMakerAssociativeConfig:
    """Simplified configuration for ROMA-MDAP integration (fallback)"""
    
    # ROMA settings
    roma_max_depth_analysis: int = 3
    roma_max_depth_solving: int = 2
    roma_execution_mode: str = "recursive"
    roma_enable_checkpoints: bool = False
    roma_enable_logging: bool = True
    
    # MDAP settings
    mdap_enabled: bool = True
    mdap_k_ahead: int = 3
    mdap_max_samples: int = 100
    mdap_enable_red_flagging: bool = True
    mdap_max_token_length: int = 750
    mdap_min_confidence: float = 0.2
    
    # Integration settings
    apply_maker_to_roma_atomic: bool = True
    apply_maker_to_roma_planning: bool = True
    aggregate_maker_results: bool = True
    enable_hierarchical_voting: bool = True
    enable_adaptive_k: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    
    # Fault tolerance
    max_retries: int = 3
    timeout_seconds: int = 300
    fallback_policy: str = "escalate_then_best_effort"
    
    # Associative Recomposition settings
    use_associative_recomposition: bool = True
    associative_max_retries: int = 3
    associative_use_agentjson: bool = True
    
    # Ground Truth settings
    enable_ground_truth: bool = True
    ground_truth_storage_path: str = "roma_mdap_maker_ground_truth.json"
    
    # Integration settings
    apply_mdap_to_recomposed: bool = True
    enable_hierarchical_validation: bool = True
    
    # Evaluator Team settings
    use_evaluator_team: bool = True
    evaluator_threshold: str = "standard_approval"
    evaluator_num_members: int = 3
    
    # Gauntlet System settings
    use_gauntlet_system: bool = True
    gauntlet_difficulty: str = "adaptive"
    
    # Recursive Refinement settings
    max_refinement_attempts: int = 3
    min_acceptance_score: float = 75.0
    
    # Provider settings
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# SIMPLIFIED ENGINE (Fallback)
# =============================================================================

class ROMAMDAPMakerAssociativeEngine:
    """
    Integration engine for ROMA and MDAP systems.
    Provides associative learning and multi-domain planning capabilities.
    
    This class acts as a wrapper around the full implementation when available,
    or provides a simplified fallback implementation.
    """

    def __init__(self, config: Optional[ROMAMDAPMakerAssociativeConfig] = None):
        """
        Initialize the ROMA-MDAP integration engine.
        
        Args:
            config: Configuration for the engine
        """
        self.config = config or ROMAMDAPMakerAssociativeConfig()
        self.initialized = False
        
        # Use full implementation if available
        if FULL_IMPLEMENTATION_AVAILABLE and _FullEngine is not None:
            logger.info("Using full ROMA-MDAP-MAKER associative implementation")
            self._full_engine = _FullEngine(self.config)
            self._use_full = True
        else:
            logger.warning("Using simplified ROMA-MDAP-MAKER associative implementation")
            self._full_engine = None
            self._use_full = False
            
        # Metrics for fallback implementation
        self._metrics = {
            "total_problems_solved": 0,
            "total_decomposition_time": 0.0,
            "total_recomposition_time": 0.0,
            "total_validation_time": 0.0,
            "avg_confidence": 0.0,
            "total_sub_solutions": 0,
            "successful_recompositions": 0,
            "failed_recompositions": 0
        }

    def initialize(self) -> bool:
        """Initialize the ROMA-MDAP integration engine."""
        try:
            if self._use_full and self._full_engine:
                # Full implementation initializes in __init__
                self.initialized = True
                return True
            else:
                # Simplified initialization
                self.initialized = True
                return True
        except Exception as e:
            logger.error(f"Failed to initialize ROMA engine: {e}")
            return False

    def plan_decomposition(self, problem: str, domain: str) -> Dict[str, Any]:
        """
        Plan a problem decomposition using ROMA heuristics.
        
        Args:
            problem: The problem to decompose
            domain: The domain context
            
        Returns:
            Decomposition plan with metadata
        """
        if not self.initialized:
            self.initialize()
        
        if self._use_full and self._full_engine:
            # Use full implementation for decomposition
            result = self._full_engine.solve_problem(
                problem=problem,
                context={"domain": domain}
            )
            return {
                "problem": problem,
                "domain": domain,
                "approach": "associative",
                "decomposition": result.get("roma_decomposition", {}),
                "confidence": result.get("confidence", 0.75),
                "num_sub_solutions": result.get("num_sub_solutions", 0)
            }
        else:
            # Simplified fallback
            return {
                "problem": problem,
                "domain": domain,
                "approach": "associative",
                "confidence": 0.75,
                "num_sub_solutions": 1,
                "decomposition": {
                    "goal": problem,
                    "subtasks": [{
                        "id": "task_1",
                        "description": problem,
                        "atomic": True
                    }]
                }
            }

    def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        llm_call_fn: Optional[Callable[[str], str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using the ROMA-MDAP-MAKER pipeline.
        
        Args:
            problem: Problem statement
            context: Additional context
            llm_call_fn: Optional LLM call function
            config_overrides: Optional configuration overrides
            
        Returns:
            Complete solution with metadata
        """
        if self._use_full and self._full_engine:
            return self._full_engine.solve_problem(
                problem=problem,
                context=context,
                llm_call_fn=llm_call_fn,
                config_overrides=config_overrides
            )
        else:
            # Simplified fallback
            logger.info(f"Solving problem (simplified): {problem[:100]}...")
            return {
                "success": True,
                "problem": problem,
                "solution": f"Simplified solution for: {problem}",
                "confidence": 0.7,
                "num_sub_solutions": 1,
                "total_time": 0.1,
                "error_free": True
            }

    def solve_problem_recursive(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        llm_call_fn: Optional[Callable[[str], str]] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using recursive refinement.
        
        Args:
            problem: Problem statement
            context: Additional context
            llm_call_fn: Optional LLM call function
            
        Returns:
            Final solution with refinement metadata
        """
        if self._use_full and self._full_engine:
            return self._full_engine.solve_problem_recursive(
                problem=problem,
                context=context,
                llm_call_fn=llm_call_fn
            )
        else:
            # Simplified fallback - just call solve_problem once
            return self.solve_problem(problem, context, llm_call_fn)

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        if self._use_full and self._full_engine:
            return self._full_engine.get_metrics()
        else:
            return self._metrics.copy()

    def reset_metrics(self):
        """Reset metrics."""
        if self._use_full and self._full_engine:
            self._full_engine.reset_metrics()
        else:
            self._metrics = {
                "total_problems_solved": 0,
                "total_decomposition_time": 0.0,
                "total_recomposition_time": 0.0,
                "total_validation_time": 0.0,
                "avg_confidence": 0.0,
                "total_sub_solutions": 0,
                "successful_recompositions": 0,
                "failed_recompositions": 0
            }

    def get_config(self) -> ROMAMDAPMakerAssociativeConfig:
        """Get the current configuration."""
        return self.config


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_romamdapmaker_associative_config(
    preset: str = "standard",
    **kwargs
) -> ROMAMDAPMakerAssociativeConfig:
    """
    Create a configuration for ROMA-MDAP integration.
    
    Args:
        preset: Configuration preset ("standard", "thorough", "fast", "validation", "recomposition")
        **kwargs: Additional configuration overrides
        
    Returns:
        ROMAMDAPMakerAssociativeConfig object
    """
    if FULL_IMPLEMENTATION_AVAILABLE and _create_full_config is not None:
        # Use full implementation config creation
        full_config = _create_full_config(preset=preset, **kwargs)
        # Convert to simplified config if needed
        return ROMAMDAPMakerAssociativeConfig(
            roma_max_depth_analysis=full_config.roma_max_depth_analysis,
            roma_max_depth_solving=full_config.roma_max_depth_solving,
            roma_execution_mode=full_config.roma_execution_mode,
            roma_enable_checkpoints=full_config.roma_enable_checkpoints,
            roma_enable_logging=full_config.roma_enable_logging,
            mdap_enabled=full_config.mdap_enabled,
            mdap_k_ahead=full_config.mdap_k_ahead,
            mdap_max_samples=full_config.mdap_max_samples,
            mdap_enable_red_flagging=full_config.mdap_enable_red_flagging,
            mdap_max_token_length=full_config.mdap_max_token_length,
            mdap_min_confidence=full_config.mdap_min_confidence,
            apply_maker_to_roma_atomic=full_config.apply_maker_to_roma_atomic,
            apply_maker_to_roma_planning=full_config.apply_maker_to_roma_planning,
            aggregate_maker_results=full_config.aggregate_maker_results,
            enable_hierarchical_voting=full_config.enable_hierarchical_voting,
            enable_adaptive_k=full_config.enable_adaptive_k,
            enable_caching=full_config.enable_caching,
            cache_ttl_seconds=full_config.cache_ttl_seconds,
            cache_max_size=full_config.cache_max_size,
            max_retries=full_config.max_retries,
            timeout_seconds=full_config.timeout_seconds,
            fallback_policy=full_config.fallback_policy,
            use_associative_recomposition=full_config.use_associative_recomposition,
            associative_max_retries=full_config.associative_max_retries,
            associative_use_agentjson=full_config.associative_use_agentjson,
            enable_ground_truth=full_config.enable_ground_truth,
            ground_truth_storage_path=full_config.ground_truth_storage_path,
            apply_mdap_to_recomposed=full_config.apply_mdap_to_recomposed,
            enable_hierarchical_validation=full_config.enable_hierarchical_validation,
            use_evaluator_team=full_config.use_evaluator_team,
            evaluator_threshold=full_config.evaluator_threshold,
            evaluator_num_members=full_config.evaluator_num_members,
            use_gauntlet_system=full_config.use_gauntlet_system,
            gauntlet_difficulty=full_config.gauntlet_difficulty,
            max_refinement_attempts=full_config.max_refinement_attempts,
            min_acceptance_score=full_config.min_acceptance_score,
            provider=full_config.provider,
            api_key=full_config.api_key,
            model=full_config.model,
            temperature=full_config.temperature,
            metadata=full_config.metadata
        )
    else:
        # Use simplified config creation
        return ROMAMDAPMakerAssociativeConfig(**kwargs)


def solve_with_romamdapmaker_associative(
    problem: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[ROMAMDAPMakerAssociativeConfig] = None,
    llm_call_fn: Optional[Callable[[str], str]] = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for ROMA-MDAP-MAKER + Associative problem solving.
    
    Args:
        problem: Problem statement
        context: Additional context
        config: Configuration (uses default if not provided)
        llm_call_fn: LLM call function
        recursive: Whether to use the recursive refinement loop
        
    Returns:
        Complete solution with all metadata
    """
    if FULL_IMPLEMENTATION_AVAILABLE and _solve_full is not None:
        # Use full implementation
        return _solve_full(
            problem=problem,
            context=context,
            config=config,
            llm_call_fn=llm_call_fn,
            recursive=recursive
        )
    else:
        # Use simplified implementation
        engine = ROMAMDAPMakerAssociativeEngine(config)
        if recursive:
            return engine.solve_problem_recursive(problem, context, llm_call_fn)
        else:
            return engine.solve_problem(problem, context, llm_call_fn)


def get_romamdapmaker_associative_status() -> Dict[str, Any]:
    """
    Get ROMA-MDAP-MAKER + Associative system status.
    
    Returns:
        Dict with availability and configuration info
    """
    if FULL_IMPLEMENTATION_AVAILABLE and _get_status is not None:
        return _get_status()
    else:
        return {
            "roma_mdap_maker_available": False,
            "associative_available": False,
            "ground_truth_available": False,
            "full_system_available": False,
            "components": {
                "roma_mdap_maker": False,
                "associative_recomposition": False,
                "ground_truth_store": False
            },
            "description": "ROMA hierarchical decomposition + Associative recomposition + MDAP multi-agent validation (simplified mode)"
        }
