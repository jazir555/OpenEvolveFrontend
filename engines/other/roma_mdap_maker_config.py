"""
ROMA-MDAP-MAKER Configuration Models

This module defines the configuration structures used across the 
ROMA-MDAP-MAKER + Associative Recomposition system.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ROMAMDAPMakerAssociativeConfig:
    """Configuration for ROMA-MDAP-MAKER + Associative integration"""

    # ROMA-MDAP-MAKER settings
    roma_max_depth_analysis: int = 3
    roma_max_depth_solving: int = 2
    roma_execution_mode: str = "recursive"
    roma_enable_checkpoints: bool = False
    roma_enable_logging: bool = True

    # MDAP/MAKER settings
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
    apply_mdap_to_recomposed: bool = True  # Apply MDAP validation after recomposition
    enable_hierarchical_validation: bool = True
    
    # Evaluator Team settings
    use_evaluator_team: bool = True
    evaluator_threshold: str = "standard_approval"  # "minimal_acceptance", "standard_approval", "high_quality"
    evaluator_num_members: int = 3
    
    # Gauntlet System settings
    use_gauntlet_system: bool = True
    gauntlet_difficulty: str = "adaptive"  # "easy", "medium", "hard", "adaptive"
    
    # Recursive Refinement settings
    max_refinement_attempts: int = 3
    min_acceptance_score: float = 75.0  # Out of 100 from Evaluator Team

    # Provider settings
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

def create_romamdapmaker_associative_config(**kwargs) -> ROMAMDAPMakerAssociativeConfig:
    """Factory function to create config with overrides."""
    # Filter kwargs to only include valid fields
    valid_fields = ROMAMDAPMakerAssociativeConfig.__dataclass_fields__.keys()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    return ROMAMDAPMakerAssociativeConfig(**filtered_kwargs)
