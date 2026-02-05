"""
PES Enhanced Configuration Integration

Integration with OpenEvolve configuration system.
Provides seamless integration between PES Enhanced and the standard config system.

Author: Agent Z1 (Integration Specialist)
Created: 2026-02-04
"""

from typing import Optional, TYPE_CHECKING

# Import PES Enhanced config
from .config import PESEnhancedConfig as PESLocalConfig

# Import RESE config system
from config import RESEConfig, PESEnhancedConfig as RESEPESEnhancedConfig
from config_loader import Config as OpenEvolveConfig, PESEnhancedConfig as LoaderPESEnhancedConfig

if TYPE_CHECKING:
    from parameter_manager import ParameterManager


def integrate_pes_config_into_rese(rese_config: RESEConfig) -> RESEConfig:
    """
    Add PES Enhanced config to RESE config if not present.
    
    This function ensures the PES Enhanced configuration is properly
    integrated into the RESE configuration structure.
    
    Args:
        rese_config: Existing RESE configuration
        
    Returns:
        RESEConfig with PES Enhanced configuration added/updated
    """
    if rese_config.pes_enhanced is None:
        rese_config.pes_enhanced = RESEPESEnhancedConfig()
    return rese_config


def get_pes_config_from_parameters(param_manager: "ParameterManager") -> RESEPESEnhancedConfig:
    """
    Build PES Enhanced config from parameter manager.
    
    Extracts PES Enhanced related parameters from the parameter manager
    and constructs a PESEnhancedConfig dataclass.
    
    Args:
        param_manager: ParameterManager instance with loaded parameters
        
    Returns:
        PESEnhancedConfig populated from parameters
    """
    return RESEPESEnhancedConfig(
        enable_cost_optimization=param_manager.get("enable_cost_optimization"),
        max_cost_usd=param_manager.get("max_cost_usd"),
        cost_warning_threshold=param_manager.get("cost_warning_threshold"),
        cost_critical_threshold=param_manager.get("cost_critical_threshold"),
        prompt_token_price=param_manager.get("prompt_token_price", 0.00001),
        completion_token_price=param_manager.get("completion_token_price", 0.00003),
        enable_early_stopping=param_manager.get("enable_early_stopping"),
        early_stopping_patience=param_manager.get("early_stopping_patience"),
        early_stopping_min_improvement=param_manager.get("early_stopping_min_improvement", 0.001),
        early_stopping_plateau_threshold=param_manager.get("early_stopping_plateau_threshold", 0.001),
        pes_planning_enabled=param_manager.get("pes_planning_enabled"),
        pes_summarization_enabled=param_manager.get("pes_summarization_enabled"),
        pes_auto_select_strategy=param_manager.get("pes_auto_select_strategy"),
        use_cheap_models_for_execution=param_manager.get("use_cheap_models_for_execution"),
        cheap_model=param_manager.get("pes_cheap_model", "gpt-3.5-turbo"),
        expensive_model=param_manager.get("pes_expensive_model", "gpt-4o"),
    )


def sync_pes_config_to_parameters(
    pes_config: RESEPESEnhancedConfig,
    param_manager: "ParameterManager"
) -> None:
    """
    Sync PES Enhanced config to parameter manager.
    
    Updates the parameter manager with values from a PESEnhancedConfig.
    
    Args:
        pes_config: PESEnhancedConfig to sync from
        param_manager: ParameterManager to update
    """
    # Note: This is a conceptual method - actual implementation would
    # depend on whether ParameterManager supports setting parameters
    # For now, this documents the intended interface
    pass


def load_pes_enhanced_config_from_env() -> RESEPESEnhancedConfig:
    """
    Load PES Enhanced configuration from environment variables.
    
    Uses the same environment variable names as defined in config_loader.py
    and .env.example for consistency.
    
    Returns:
        PESEnhancedConfig loaded from environment
    """
    import os
    
    def get_env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def get_env_float(name: str, default: float = 0.0) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    def get_env_int(name: str, default: int = 0) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_env_str(name: str, default: str = "") -> str:
        return os.getenv(name, default)
    
    return RESEPESEnhancedConfig(
        enable_cost_optimization=get_env_bool("PES_COST_OPTIMIZATION", False),
        max_cost_usd=get_env_float("PES_MAX_COST_USD", 10.0),
        cost_warning_threshold=get_env_float("PES_COST_WARNING", 0.7),
        cost_critical_threshold=get_env_float("PES_COST_CRITICAL", 0.9),
        prompt_token_price=get_env_float("PES_PROMPT_TOKEN_PRICE", 0.00001),
        completion_token_price=get_env_float("PES_COMPLETION_TOKEN_PRICE", 0.00003),
        enable_early_stopping=get_env_bool("PES_EARLY_STOPPING", True),
        early_stopping_patience=get_env_int("PES_STOPPING_PATIENCE", 5),
        early_stopping_min_improvement=get_env_float("PES_MIN_IMPROVEMENT", 0.001),
        early_stopping_plateau_threshold=get_env_float("PES_PLATEAU_THRESHOLD", 0.001),
        pes_planning_enabled=get_env_bool("PES_PLANNING", True),
        pes_summarization_enabled=get_env_bool("PES_SUMMARIZATION", True),
        pes_auto_select_strategy=get_env_bool("PES_AUTO_SELECT", True),
        use_cheap_models_for_execution=get_env_bool("PES_USE_CHEAP_MODELS", True),
        cheap_model=get_env_str("PES_CHEAP_MODEL", "gpt-3.5-turbo"),
        expensive_model=get_env_str("PES_EXPENSIVE_MODEL", "gpt-4o"),
    )


def convert_local_to_rese_config(local_config: PESLocalConfig) -> RESEPESEnhancedConfig:
    """
    Convert local PES Enhanced config to RESE format.
    
    Converts the openevolve_pes_enhanced local config format to the
    RESE config format used by the main configuration system.
    
    Args:
        local_config: PESLocalConfig from openevolve_pes_enhanced.config
        
    Returns:
        RESEPESEnhancedConfig for use with RESEConfig
    """
    return RESEPESEnhancedConfig(
        enable_cost_optimization=local_config.enable_cost_optimization,
        max_cost_usd=local_config.cost.max_cost_usd,
        cost_warning_threshold=local_config.cost.warning_threshold,
        cost_critical_threshold=local_config.cost.critical_threshold,
        prompt_token_price=local_config.cost.prompt_token_price,
        completion_token_price=local_config.cost.completion_token_price,
        enable_early_stopping=local_config.enable_early_stopping,
        early_stopping_patience=local_config.early_stopping.patience,
        early_stopping_min_improvement=local_config.early_stopping.min_improvement,
        early_stopping_plateau_threshold=local_config.early_stopping.plateau_threshold,
        pes_planning_enabled=local_config.enable_planning,
        pes_summarization_enabled=local_config.enable_summarization,
        pes_auto_select_strategy=local_config.planning.auto_select_strategy,
        use_cheap_models_for_execution=local_config.cost.use_cheap_models_for_execution,
        cheap_model=local_config.cost.cheap_model,
        expensive_model=local_config.cost.expensive_model,
    )


def convert_rese_to_local_config(rese_config: RESEPESEnhancedConfig) -> PESLocalConfig:
    """
    Convert RESE PES Enhanced config to local format.
    
    Converts the RESE config format to the openevolve_pes_enhanced local
    config format.
    
    Args:
        rese_config: RESEPESEnhancedConfig from RESEConfig
        
    Returns:
        PESLocalConfig for use with openevolve_pes_enhanced
    """
    from .config import CostOptimizationConfig, EarlyStoppingConfig, PlanningConfig, SummarizationConfig
    
    local_config = PESLocalConfig()
    local_config.enable_cost_optimization = rese_config.enable_cost_optimization
    local_config.enable_early_stopping = rese_config.enable_early_stopping
    local_config.enable_planning = rese_config.pes_planning_enabled
    local_config.enable_summarization = rese_config.pes_summarization_enabled
    
    # Cost config
    local_config.cost = CostOptimizationConfig(
        max_cost_usd=rese_config.max_cost_usd,
        warning_threshold=rese_config.cost_warning_threshold,
        critical_threshold=rese_config.cost_critical_threshold,
        prompt_token_price=rese_config.prompt_token_price,
        completion_token_price=rese_config.completion_token_price,
        use_cheap_models_for_execution=rese_config.use_cheap_models_for_execution,
        cheap_model=rese_config.cheap_model,
        expensive_model=rese_config.expensive_model,
    )
    
    # Early stopping config
    local_config.early_stopping = EarlyStoppingConfig(
        enabled=rese_config.enable_early_stopping,
        patience=rese_config.early_stopping_patience,
        min_improvement=rese_config.early_stopping_min_improvement,
        plateau_threshold=rese_config.early_stopping_plateau_threshold,
    )
    
    # Planning config
    local_config.planning = PlanningConfig(
        enabled=rese_config.pes_planning_enabled,
        auto_select_strategy=rese_config.pes_auto_select_strategy,
    )
    
    # Summarization config
    local_config.summarization = SummarizationConfig(
        enabled=rese_config.pes_summarization_enabled,
    )
    
    return local_config


def apply_pes_config_to_openevolve(config: OpenEvolveConfig) -> OpenEvolveConfig:
    """
    Apply PES Enhanced configuration to OpenEvolve config.
    
    This function integrates PES Enhanced settings into the main
    OpenEvolve configuration system.
    
    Args:
        config: OpenEvolveConfig to update
        
    Returns:
        Updated OpenEvolveConfig with PES Enhanced settings
    """
    # The pes_enhanced field is already part of Config via the dataclass
    # This function serves as a hook for any additional integration logic
    return config


# Convenience function for getting a fully integrated config
def get_integrated_config(
    use_env: bool = True,
    rese_config: Optional[RESEConfig] = None,
    param_manager: Optional["ParameterManager"] = None,
) -> RESEConfig:
    """
    Get a fully integrated configuration with PES Enhanced.
    
    This is a convenience function that handles the integration of
    PES Enhanced configuration from multiple sources.
    
    Args:
        use_env: Whether to load from environment variables
        rese_config: Optional existing RESE config to extend
        param_manager: Optional parameter manager to extract settings from
        
    Returns:
        RESEConfig with fully integrated PES Enhanced configuration
    """
    if rese_config is None:
        rese_config = RESEConfig()
    
    # Priority: explicit param_manager > environment > existing config > defaults
    if param_manager is not None:
        rese_config.pes_enhanced = get_pes_config_from_parameters(param_manager)
    elif use_env:
        rese_config.pes_enhanced = load_pes_enhanced_config_from_env()
    elif rese_config.pes_enhanced is None:
        rese_config.pes_enhanced = RESEPESEnhancedConfig()
    
    return rese_config


__all__ = [
    "integrate_pes_config_into_rese",
    "get_pes_config_from_parameters",
    "sync_pes_config_to_parameters",
    "load_pes_enhanced_config_from_env",
    "convert_local_to_rese_config",
    "convert_rese_to_local_config",
    "apply_pes_config_to_openevolve",
    "get_integrated_config",
]
