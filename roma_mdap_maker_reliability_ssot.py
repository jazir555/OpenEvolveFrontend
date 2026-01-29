"""
ROMA-MDAP-MAKER Reliability Single Source of Truth (SSOT)

Provides standardized, high-reliability configuration presets for the 
ROMA-MDAP-MAKER + Associative Recomposition system.

This module ensures 100% parity across all integration points while 
eliminating code bloat from 40+ parameter blocks.
"""

import os
from typing import Dict, Any, Optional
from roma_mdap_maker_associative_integration import (
    ROMAMDAPMakerAssociativeConfig,
    create_romamdapmaker_associative_config
)

def get_reliability_config(
    preset: str = "standard", 
    **overrides
) -> ROMAMDAPMakerAssociativeConfig:
    """
    Get a standardized reliability configuration.
    
    Presets:
        - "standard": Balanced reliability for most tasks.
        - "thorough": Maximum rigor for mission-critical tasks (deeper, higher k).
        - "fast": Quick execution with basic safeguards.
        - "validation": Optimized for gauntlets and evaluators (shallow depth, high threshold).
        - "recomposition": Optimized for synthesis and assembly tasks.
    
    Args:
        preset: Name of the configuration preset.
        **overrides: Individual parameter overrides.
        
    Returns:
        ROMAMDAPMakerAssociativeConfig object.
    """
    
    # Base defaults (The "Single Source of Truth" for all 27+ master parameters)
    base_params = {
        # ROMA settings
        "roma_max_depth_analysis": 3,
        "roma_max_depth_solving": 2,
        "roma_execution_mode": "recursive",
        "roma_enable_checkpoints": False,
        "roma_enable_logging": True,
        
        # MDAP/MAKER settings
        "mdap_enabled": True,
        "mdap_k_ahead": 3,
        "mdap_max_samples": 100,
        "mdap_enable_red_flagging": True,
        "mdap_max_token_length": 1000,
        "mdap_min_confidence": 0.2,
        
        # Integration settings
        "apply_maker_to_roma_atomic": True,
        "apply_maker_to_roma_planning": True,
        "aggregate_maker_results": True,
        "enable_hierarchical_voting": True,
        "enable_adaptive_k": True,
        
        # Caching
        "enable_caching": True,
        "cache_ttl_seconds": 3600,
        "cache_max_size": 10000,
        
        # Fault tolerance
        "max_retries": 3,
        "timeout_seconds": 300,
        "fallback_policy": "escalate_then_best_effort",
        
        # Associative settings
        "use_associative_recomposition": True,
        "associative_max_retries": 3,
        "associative_use_agentjson": True,
        "enable_ground_truth": True,
        "ground_truth_storage_path": "roma_mdap_maker_ground_truth.json",
        "apply_mdap_to_recomposed": True,
        "enable_hierarchical_validation": True,
        
        # Orchestration settings
        "use_evaluator_team": True,
        "evaluator_threshold": "standard_approval",
        "evaluator_num_members": 3,
        "use_gauntlet_system": True,
        "gauntlet_difficulty": "adaptive",
        "max_refinement_attempts": 3,
        "min_acceptance_score": 75.0,
        
        # Provider settings
        "provider": os.getenv("ROMA_PROVIDER", "openai"),
        "model": os.getenv("ROMA_MODEL", "gpt-4o-mini"),
        "temperature": 0.1,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "metadata": {}
    }
    
    # Apply Preset Modifications
    if preset == "thorough":
        base_params.update({
            "roma_max_depth_analysis": 5,
            "roma_max_depth_solving": 3,
            "mdap_k_ahead": 5,
            "mdap_max_samples": 200,
            "mdap_min_confidence": 0.4,
            "max_refinement_attempts": 5,
            "min_acceptance_score": 85.0,
            "temperature": 0.2 # Slightly more exploration in voting
        })
    elif preset == "fast":
        base_params.update({
            "roma_max_depth_analysis": 2,
            "roma_max_depth_solving": 1,
            "mdap_k_ahead": 2,
            "mdap_max_samples": 50,
            "use_evaluator_team": False,
            "use_gauntlet_system": False,
            "temperature": 0.0
        })
    elif preset == "validation":
        base_params.update({
            "roma_max_depth_analysis": 1,
            "roma_max_depth_solving": 1,
            "mdap_min_confidence": 0.5,
            "use_associative_recomposition": False,
            "temperature": 0.0,
            "max_retries": 1
        })
    elif preset == "recomposition":
        base_params.update({
            "roma_max_depth_analysis": 1,
            "roma_max_depth_solving": 1,
            "mdap_k_ahead": 3,
            "use_associative_recomposition": True,
            "temperature": 0.1
        })
        
    # Apply User Overrides
    base_params.update(overrides)
    
    # Return constructed config
    return create_romamdapmaker_associative_config(**base_params)

# Global helper for even easier access
def get_standard_config(**kwargs): return get_reliability_config("standard", **kwargs)
def get_thorough_config(**kwargs): return get_reliability_config("thorough", **kwargs)
def get_fast_config(**kwargs): return get_reliability_config("fast", **kwargs)
def get_validation_config(**kwargs): return get_reliability_config("validation", **kwargs)
def get_recomposition_config(**kwargs): return get_reliability_config("recomposition", **kwargs)
