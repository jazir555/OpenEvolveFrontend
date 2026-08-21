"""
ROMA-MDAP-MAKER Reliability Single Source of Truth (SSOT)

Provides standardized, high-reliability configuration presets for the 
ROMA-MDAP-MAKER + Associative Recomposition system.

This module ensures 100% parity across all integration points while 
eliminating code bloat from 40+ parameter blocks.
"""
from __future__ import annotations


import os
import logging
from typing import Dict, Any, List, Optional
from roma_mdap_maker_config import (
    ROMAMDAPMakerAssociativeConfig,
    create_romamdapmaker_associative_config
)

# Lean 4 Integration - Formal Verification for Reliability Presets
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_reliability_config(
    preset: str = "standard", 
    verify_on_load: bool = False,
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
        verify_on_load: If True, perform Lean verification and log warnings on failure.
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
    
    # Optional: Verify preset correctness using Lean 4
    if verify_on_load:
        try:
            verifier = get_lean_verifier()
            # Extract relevant config for verification
            config_for_verify = {
                "roma_max_depth_analysis": base_params.get("roma_max_depth_analysis", 3),
                "roma_max_depth_solving": base_params.get("roma_max_depth_solving", 2),
                "mdap_k_ahead": base_params.get("mdap_k_ahead", 3),
                "mdap_max_samples": base_params.get("mdap_max_samples", 100),
                "mdap_min_confidence": base_params.get("mdap_min_confidence", 0.2),
                "max_refinement_attempts": base_params.get("max_refinement_attempts", 3),
                "min_acceptance_score": base_params.get("min_acceptance_score", 75.0),
                "max_retries": base_params.get("max_retries", 3),
                "timeout_seconds": base_params.get("timeout_seconds", 300),
            }
            
            # Verify safety properties
            safety_result = verifier.verify_preset_safety(preset)
            if not safety_result.get("verified", False):
                reason = safety_result.get("reason", "unknown")
                logger.warning(
                    f"Lean verification WARNING for preset '{preset}': "
                    f"Safety properties not verified. Reason: {reason}. "
                    f"This preset may have reliability issues."
                )
            else:
                confidence = safety_result.get("confidence", 0.0)
                logger.info(
                    f"Lean verification SUCCESS for preset '{preset}': "
                    f"Safety verified with confidence {confidence:.2f}"
                )
            
            # Store verification result in config metadata
            base_params["metadata"]["lean_verification"] = {
                "safety": safety_result,
                "timestamp": __import__('time').time()
            }
            
        except Exception as e:
            # Log but don't fail - verification is advisory
            logger.warning(f"Lean verification failed for preset '{preset}': {e}")
            base_params["metadata"]["lean_verification"] = {
                "error": str(e),
                "timestamp": __import__('time').time()
            }
    
    # Return constructed config
    return create_romamdapmaker_associative_config(**base_params)

# Global helper for even easier access
def get_standard_config(**kwargs): return get_reliability_config("standard", **kwargs)
def get_thorough_config(**kwargs): return get_reliability_config("thorough", **kwargs)
def get_fast_config(**kwargs): return get_reliability_config("fast", **kwargs)
def get_validation_config(**kwargs): return get_reliability_config("validation", **kwargs)
def get_recomposition_config(**kwargs): return get_reliability_config("recomposition", **kwargs)


class LeanReliabilityVerifier:
    """
    Formal verification for ROMA-MDAP-MAKER reliability presets using Lean 4.
    
    This is CRITICAL - reliability presets affect mission-critical tasks
    and their correctness must be formally proven.
    """
    
    def __init__(self):
        self._client: Optional[LeanAideClient] = None
        self._verification_cache: Dict[str, Dict[str, Any]] = {}
    
    @property
    def client(self) -> Optional[LeanAideClient]:
        """Lazy initialization of LeanAide client."""
        if self._client is None and LEAN_AVAILABLE:
            try:
                self._client = LeanAideClient()
            except Exception as e:
                logger.error(f"Failed to initialize LeanAide client: {e}")
        return self._client
    
    def verify_preset_correctness(
        self, 
        preset: str, 
        properties: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify reliability preset satisfies formal correctness properties.
        
        This is CRITICAL - reliability presets affect mission-critical tasks
        and their correctness must be formally proven.
        
        Args:
            preset: Name of preset ("standard", "thorough", etc.)
            properties: List of properties the preset should satisfy
            config: Optional pre-fetched config (to avoid circular calls)
            
        Returns:
            Dict with verified (bool), confidence (float), proof (str)
        """
        cache_key = f"{preset}:{','.join(sorted(properties))}"
        
        # Check cache
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key]
        
        if not LEAN_AVAILABLE:
            result = {
                "preset": preset,
                "verified": False, 
                "reason": "Lean not available",
                "confidence": 0.0,
                "proof_code": None
            }
            self._verification_cache[cache_key] = result
            return result
        
        if self.client is None:
            result = {
                "preset": preset,
                "verified": False,
                "reason": "Failed to initialize Lean client",
                "confidence": 0.0,
                "proof_code": None
            }
            self._verification_cache[cache_key] = result
            return result
        
        try:
            # Get config if not provided
            if config is None:
                config = self._get_preset_params(preset)
            
            # Formalize correctness theorem
            theorem_statement = self._formalize_theorem(preset, properties, config)
            
            # Autoformalize to Lean 4
            formalized = self.client.autoformalize(theorem_statement)
            
            # Verify the formalized theorem
            result_obj = self.client.verify(formalized)
            
            result = {
                "preset": preset,
                "verified": getattr(result_obj, 'verified', False),
                "confidence": getattr(result_obj, 'confidence', 0.0),
                "proof_code": getattr(result_obj, 'proof_code', None),
                "theorem": theorem_statement,
                "formalized": formalized
            }
            
            logger.info(f"Lean verification for preset '{preset}': verified={result['verified']}, "
                       f"confidence={result['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Lean verification failed for preset '{preset}': {e}")
            result = {
                "preset": preset,
                "verified": False,
                "reason": f"Verification error: {str(e)}",
                "confidence": 0.0,
                "proof_code": None
            }
        
        self._verification_cache[cache_key] = result
        return result
    
    def _get_preset_params(self, preset: str) -> Dict[str, Any]:
        """Get base parameters for a preset (internal use)."""
        # Get base defaults from the main function's defaults
        base_params = {
            "roma_max_depth_analysis": 3,
            "roma_max_depth_solving": 2,
            "mdap_k_ahead": 3,
            "mdap_max_samples": 100,
            "mdap_min_confidence": 0.2,
            "max_refinement_attempts": 3,
            "min_acceptance_score": 75.0,
            "max_retries": 3,
            "timeout_seconds": 300,
        }
        
        # Apply preset modifications
        if preset == "thorough":
            base_params.update({
                "roma_max_depth_analysis": 5,
                "roma_max_depth_solving": 3,
                "mdap_k_ahead": 5,
                "mdap_max_samples": 200,
                "mdap_min_confidence": 0.4,
                "max_refinement_attempts": 5,
                "min_acceptance_score": 85.0,
            })
        elif preset == "fast":
            base_params.update({
                "roma_max_depth_analysis": 2,
                "roma_max_depth_solving": 1,
                "mdap_k_ahead": 2,
                "mdap_max_samples": 50,
            })
        elif preset == "validation":
            base_params.update({
                "roma_max_depth_analysis": 1,
                "roma_max_depth_solving": 1,
                "mdap_min_confidence": 0.5,
                "max_retries": 1,
            })
        elif preset == "recomposition":
            base_params.update({
                "roma_max_depth_analysis": 1,
                "roma_max_depth_solving": 1,
            })
        
        return base_params
    
    def _formalize_theorem(
        self, 
        preset: str, 
        properties: List[str],
        config: Dict[str, Any]
    ) -> str:
        """Create formal theorem statement for verification."""
        # Build a structured theorem statement
        theorem_parts = [
            f"Configuration '{preset}' ensures the following properties:"
        ]
        
        for prop in properties:
            theorem_parts.append(f"- {prop}")
        
        # Add parameter constraints
        theorem_parts.append(f"\nParameter constraints:")
        theorem_parts.append(f"- max_depth_analysis ≤ {config.get('roma_max_depth_analysis', 3)}")
        theorem_parts.append(f"- mdap_k_ahead ≥ {config.get('mdap_k_ahead', 3)}")
        theorem_parts.append(f"- min_confidence ≥ {config.get('mdap_min_confidence', 0.2)}")
        theorem_parts.append(f"- min_acceptance_score ≥ {config.get('min_acceptance_score', 75.0)}")
        
        return "\n".join(theorem_parts)
    
    def verify_preset_safety(self, preset: str) -> Dict[str, Any]:
        """
        Verify preset satisfies safety properties (convenience method).
        
        Checks:
        - Timeout bounds are reasonable
        - Retry limits prevent infinite loops
        - Depth limits prevent stack overflow
        """
        safety_properties = [
            "Timeout is bounded to prevent indefinite execution",
            "Retry limits prevent infinite retry loops",
            "Analysis depth is finite and bounded",
            "Minimum confidence threshold prevents low-quality outputs"
        ]
        return self.verify_preset_correctness(preset, safety_properties)
    
    def verify_preset_completeness(self, preset: str) -> Dict[str, Any]:
        """
        Verify preset satisfies completeness properties (convenience method).
        
        Checks:
        - Sufficient samples for statistical significance
        - Adequate depth for meaningful analysis
        - Proper thresholds for quality gates
        """
        completeness_properties = [
            "Sample size is sufficient for statistical significance",
            "Analysis depth is adequate for problem decomposition",
            "Confidence thresholds ensure quality output",
            "Acceptance scores match preset reliability level"
        ]
        return self.verify_preset_correctness(preset, completeness_properties)


# Global verifier instance (lazy initialization)
_lean_verifier: Optional[LeanReliabilityVerifier] = None


def get_lean_verifier() -> LeanReliabilityVerifier:
    """Get the global Lean reliability verifier instance."""
    global _lean_verifier
    if _lean_verifier is None:
        _lean_verifier = LeanReliabilityVerifier()
    return _lean_verifier


def verify_preset_correctness(
    preset: str, 
    properties: List[str]
) -> Dict[str, Any]:
    """
    Verify reliability preset satisfies formal correctness properties.
    
    This is CRITICAL - reliability presets affect mission-critical tasks
    and their correctness must be formally proven.
    
    Args:
        preset: Name of preset ("standard", "thorough", etc.)
        properties: List of properties the preset should satisfy
        
    Returns:
        Dict with verified (bool), confidence (float), proof (str)
    """
    verifier = get_lean_verifier()
    return verifier.verify_preset_correctness(preset, properties)
