"""
Decomposition Engine MDAP Integration

This module provides integration points between the DecompositionEngine
and enhanced MDAP components (MDAPCacheManager, MDAPLoadBalancer, AdaptiveThresholdManager).

Usage:
    from decomposition_engine import DecompositionEngine
    from decomposition_mdap_integration import create_mdap_enhanced_decomposition_engine

    # Create engine with MDAP enhancements
    engine = create_mdap_enhanced_decomposition_engine()

    # Or manually
    from mdap_engine import MDAPCacheManager, MDAPLoadBalancer, AdaptiveThresholdManager
    from workflow_structures import Team

    cache_manager = MDAPCacheManager(max_size=10000, ttl_seconds=3600)
    load_balancer = MDAPLoadBalancer(available_agents=team.members)
    threshold_manager = AdaptiveThresholdManager(initial_k=3, min_k=1, max_k=10)

    engine = DecompositionEngine(
        mdap_cache_manager=cache_manager,
        mdap_load_balancer=load_balancer,
        adaptive_threshold_manager=threshold_manager
    )
"""
from __future__ import annotations


import logging
from typing import Optional, List, Dict, Any

# Try to import MDAP components
try:
    from mdap_engine import MDAPCacheManager, MDAPLoadBalancer, AdaptiveThresholdManager
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    MDAPCacheManager = None
    MDAPLoadBalancer = None
    AdaptiveThresholdManager = None

# Try to import workflow structures
try:
    from workflow_structures import Team, ModelConfig
    WORKFLOW_STRUCTURES_AVAILABLE = True
except ImportError:
    WORKFLOW_STRUCTURES_AVAILABLE = False
    Team = None
    ModelConfig = None

# Try to import decomposition engine
try:
    from decomposition_engine import DecompositionEngine
    DECOMPOSITION_ENGINE_AVAILABLE = True
except ImportError:
    DECOMPOSITION_ENGINE_AVAILABLE = False
    DecompositionEngine = None

logger = logging.getLogger(__name__)


def create_mdap_enhanced_decomposition_engine(
    problem_analyzer=None,
    knowledge_manager=None,
    use_intelligent_selection: bool = True,
    team_assignment_engine=None,
    cache_max_size: int = 10000,
    cache_ttl_seconds: int = 3600,
    cache_storage_path: str = "mdap_decomposition_cache.json",
    initial_k: int = 3,
    min_k: int = 1,
    max_k: int = 10,
    team: Optional['Team'] = None
) -> 'DecompositionEngine':
    """
    Create a DecompositionEngine with MDAP enhancements enabled.

    Args:
        problem_analyzer: Optional ProblemAnalyzer instance
        knowledge_manager: Optional KnowledgeManager instance
        use_intelligent_selection: Whether to use intelligent strategy selection
        team_assignment_engine: Optional TeamAssignmentEngine
        cache_max_size: Maximum cache size (default: 10000)
        cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
        cache_storage_path: Path for persistent cache storage
        initial_k: Initial voting threshold
        min_k: Minimum voting threshold
        max_k: Maximum voting threshold
        team: Optional Team for load balancer initialization

    Returns:
        DecompositionEngine with MDAP enhancements

    Raises:
        ImportError: If MDAP components are not available
    """
    if not MDAP_AVAILABLE:
        raise ImportError("MDAP components not available. Install mdap_engine.")

    if not DECOMPOSITION_ENGINE_AVAILABLE:
        raise ImportError("DecompositionEngine not available.")

    # Create MDAP components
    cache_manager = MDAPCacheManager(
        max_size=cache_max_size,
        ttl_seconds=cache_ttl_seconds,
        storage_path=cache_storage_path
    )

    load_balancer = None
    if team and WORKFLOW_STRUCTURES_AVAILABLE:
        load_balancer = MDAPLoadBalancer(available_agents=team.members)
        logger.info(f"MDAP Load Balancer initialized with {len(team.members)} agents")
    else:
        logger.warning("No team provided for MDAP Load Balancer. Load balancing will be limited.")

    threshold_manager = AdaptiveThresholdManager(
        initial_k=initial_k,
        min_k=min_k,
        max_k=max_k
    )

    # Create enhanced decomposition engine
    engine = DecompositionEngine(
        problem_analyzer=problem_analyzer,
        knowledge_manager=knowledge_manager,
        use_intelligent_selection=use_intelligent_selection,
        team_assignment_engine=team_assignment_engine,
        mdap_cache_manager=cache_manager,
        mdap_load_balancer=load_balancer,
        adaptive_threshold_manager=threshold_manager
    )

    logger.info("Created MDAP-enhanced DecompositionEngine")

    return engine


def get_mdap_statistics(engine: 'DecompositionEngine') -> Dict[str, Any]:
    """
    Extract MDAP statistics from a DecompositionEngine.

    Args:
        engine: DecompositionEngine instance with MDAP enhancements

    Returns:
        Dict with MDAP component statistics
    """
    stats = {
        "mdap_enabled": hasattr(engine, 'mdap_available') and engine.mdap_available,
        "cache_stats": None,
        "load_balancer_stats": None,
        "adaptive_threshold_stats": None
    }

    if hasattr(engine, 'mdap_cache_manager') and engine.mdap_cache_manager:
        stats["cache_stats"] = engine.mdap_cache_manager.get_cache_stats()

    if hasattr(engine, 'mdap_load_balancer') and engine.mdap_load_balancer:
        stats["load_balancer_stats"] = engine.mdap_load_balancer.get_agent_statistics()

    if hasattr(engine, 'adaptive_threshold_manager') and engine.adaptive_threshold_manager:
        stats["adaptive_threshold_stats"] = engine.adaptive_threshold_manager.get_statistics()

    return stats


def cleanup_mdap_resources(engine: 'DecompositionEngine') -> None:
    """
    Cleanup MDAP resources in a DecompositionEngine.

    Call this before shutdown to ensure cache is saved and counters are reset.

    Args:
        engine: DecompositionEngine instance with MDAP enhancements
    """
    logger.info("Cleaning up MDAP resources...")

    if hasattr(engine, 'mdap_cache_manager') and engine.mdap_cache_manager:
        engine.mdap_cache_manager._save_cache_to_storage()
        logger.info("Saved MDAP cache to storage")

    if hasattr(engine, 'mdap_load_balancer') and engine.mdap_load_balancer:
        engine.mdap_load_balancer.reset_load_counters()
        logger.info("Reset load balancer counters")

    logger.info("MDAP resource cleanup complete")


class MDAPDecompositionConfig:
    """
    Configuration for MDAP-enhanced decomposition.

    This class provides a convenient way to configure all MDAP enhancements
    for decomposition operations.
    """

    def __init__(self,
                 enable_cache: bool = True,
                 cache_max_size: int = 10000,
                 cache_ttl_seconds: int = 3600,
                 cache_storage_path: str = "mdap_decomposition_cache.json",
                 enable_load_balancing: bool = True,
                 enable_adaptive_thresholds: bool = True,
                 initial_k: int = 3,
                 min_k: int = 1,
                 max_k: int = 10,
                 target_success_rate: float = 0.95):
        """
        Initialize MDAP decomposition configuration.

        Args:
            enable_cache: Enable caching of decomposition results
            cache_max_size: Maximum cache entries
            cache_ttl_seconds: Cache time-to-live in seconds
            cache_storage_path: Persistent cache storage path
            enable_load_balancing: Enable intelligent agent load balancing
            enable_adaptive_thresholds: Enable adaptive k-value calculation
            initial_k: Initial voting threshold
            min_k: Minimum voting threshold
            max_k: Maximum voting threshold
            target_success_rate: Target success rate for adaptation
        """
        self.enable_cache = enable_cache
        self.cache_max_size = cache_max_size
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_storage_path = cache_storage_path
        self.enable_load_balancing = enable_load_balancing
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.initial_k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.target_success_rate = target_success_rate

    def create_components(self, team: Optional['Team'] = None) -> tuple:
        """
        Create MDAP components based on configuration.

        Args:
            team: Optional team for load balancer initialization

        Returns:
            Tuple of (cache_manager, load_balancer, threshold_manager)
        """
        if not MDAP_AVAILABLE:
            raise ImportError("MDAP components not available")

        cache_manager = None
        if self.enable_cache:
            cache_manager = MDAPCacheManager(
                max_size=self.cache_max_size,
                ttl_seconds=self.cache_ttl_seconds,
                storage_path=self.cache_storage_path
            )

        load_balancer = None
        if self.enable_load_balancing and team and WORKFLOW_STRUCTURES_AVAILABLE:
            load_balancer = MDAPLoadBalancer(available_agents=team.members)

        threshold_manager = None
        if self.enable_adaptive_thresholds:
            threshold_manager = AdaptiveThresholdManager(
                initial_k=self.initial_k,
                min_k=self.min_k,
                max_k=self.max_k
            )
            threshold_manager.target_success_rate = self.target_success_rate

        return cache_manager, load_balancer, threshold_manager

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enable_cache": self.enable_cache,
            "cache_max_size": self.cache_max_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_storage_path": self.cache_storage_path,
            "enable_load_balancing": self.enable_load_balancing,
            "enable_adaptive_thresholds": self.enable_adaptive_thresholds,
            "initial_k": self.initial_k,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "target_success_rate": self.target_success_rate
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MDAPDecompositionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Convenience functions for common use cases

def create_high_throughput_config() -> MDAPDecompositionConfig:
    """
    Create configuration optimized for high throughput (more caching, lower k).
    """
    return MDAPDecompositionConfig(
        enable_cache=True,
        cache_max_size=50000,  # Larger cache
        cache_ttl_seconds=7200,  # 2 hours
        enable_load_balancing=True,
        enable_adaptive_thresholds=True,
        initial_k=2,  # Lower k for speed
        min_k=1,
        max_k=5
    )


def create_high_reliability_config() -> MDAPDecompositionConfig:
    """
    Create configuration optimized for high reliability (adaptive thresholds, higher k).
    """
    return MDAPDecompositionConfig(
        enable_cache=True,
        cache_max_size=10000,
        cache_ttl_seconds=3600,
        enable_load_balancing=True,
        enable_adaptive_thresholds=True,
        initial_k=5,  # Higher k for reliability
        min_k=3,
        max_k=15,
        target_success_rate=0.98  # Higher target
    )


def create_balanced_config() -> MDAPDecompositionConfig:
    """
    Create balanced configuration (default settings).
    """
    return MDAPDecompositionConfig()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if MDAP_AVAILABLE and DECOMPOSITION_ENGINE_AVAILABLE:
        # Create engine with default config
        engine = create_mdap_enhanced_decomposition_engine()

        # Get statistics
        stats = get_mdap_statistics(engine)
        print("MDAP Statistics:", stats)

        # Cleanup when done
        cleanup_mdap_resources(engine)
    else:
        print("MDAP or DecompositionEngine not available")
