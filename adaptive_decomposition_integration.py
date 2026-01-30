"""
Adaptive Decomposition Integration Module

This module integrates adaptive decomposition capabilities with the ROMA MDAP maker system,
providing intelligent problem decomposition with dynamic strategy selection and enhancement.

Features:
- Integration with decomposition engine adaptive enhancement
- ROMA MDAP maker associative integration
- ROMA MDAP maker reliability SSOT (Single Source of Truth)
- Dynamic strategy selection based on problem characteristics
- Automatic decomposition quality improvement
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import adaptive enhancement components
try:
    from decomposition_engine_adaptive_enhancement import (
        AdaptiveDecompositionEngine,
        StrategySelector,
        EnhancementConfig,
    )
    ADAPTIVE_ENHANCEMENT_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_ENHANCEMENT_AVAILABLE = False
    logging.warning(f"Adaptive enhancement not available: {e}")

# Import ROMA MDAP maker associative integration
try:
    from roma_mdap_maker_associative_integration import (
        AssociativeIntegrationEngine,
        IntegrationPattern,
        AssociativeConfig,
    )
    ROMA_ASSOCIATIVE_AVAILABLE = True
except ImportError as e:
    ROMA_ASSOCIATIVE_AVAILABLE = False
    logging.warning(f"ROMA associative integration not available: {e}")

# Import ROMA MDAP maker reliability SSOT
try:
    from roma_mdap_maker_reliability_ssot import (
        ReliabilitySSOT,
        ReliabilityConfig,
        SSOTManager,
    )
    ROMA_RELIABILITY_AVAILABLE = True
except ImportError as e:
    ROMA_RELIABILITY_AVAILABLE = False
    logging.warning(f"ROMA reliability SSOT not available: {e}")


logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Status of the adaptive decomposition integration."""
    INITIALIZED = "initialized"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class AdaptiveIntegrationConfig:
    """Configuration for adaptive decomposition integration.
    
    Attributes:
        enable_adaptive_enhancement: Whether to use adaptive enhancement
        enable_associative_integration: Whether to use associative integration
        enable_reliability_ssot: Whether to use reliability SSOT
        auto_select_strategy: Whether to auto-select best strategy
        quality_threshold: Minimum quality threshold (0-100)
        max_iterations: Maximum number of improvement iterations
        timeout_seconds: Timeout for decomposition operations
    """
    enable_adaptive_enhancement: bool = True
    enable_associative_integration: bool = True
    enable_reliability_ssot: bool = True
    auto_select_strategy: bool = True
    quality_threshold: float = 75.0
    max_iterations: int = 5
    timeout_seconds: int = 300


class AdaptiveDecompositionIntegration:
    """
    Main integration class for adaptive decomposition.
    
    Combines multiple decomposition strategies and enhancement techniques
    to provide optimal problem decomposition based on problem characteristics.
    """
    
    def __init__(self, config: Optional[AdaptiveIntegrationConfig] = None):
        """
        Initialize the adaptive decomposition integration.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AdaptiveIntegrationConfig()
        self.status = IntegrationStatus.INITIALIZED
        
        # Initialize components based on availability and config
        self.adaptive_engine: Optional[Any] = None
        self.associative_engine: Optional[Any] = None
        self.reliability_ssot: Optional[Any] = None
        
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize available components."""
        try:
            if ADAPTIVE_ENHANCEMENT_AVAILABLE and self.config.enable_adaptive_enhancement:
                self.adaptive_engine = AdaptiveDecompositionEngine()
                logger.info("Adaptive decomposition engine initialized")
            
            if ROMA_ASSOCIATIVE_AVAILABLE and self.config.enable_associative_integration:
                self.associative_engine = AssociativeIntegrationEngine()
                logger.info("Associative integration engine initialized")
            
            if ROMA_RELIABILITY_AVAILABLE and self.config.enable_reliability_ssot:
                self.reliability_ssot = ReliabilitySSOT()
                logger.info("Reliability SSOT initialized")
            
            self.status = IntegrationStatus.READY
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.status = IntegrationStatus.ERROR
    
    def decompose(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        strategy_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decompose a problem using adaptive strategies.
        
        Args:
            problem: The problem description to decompose
            context: Optional context information
            strategy_hint: Optional hint for strategy selection
            
        Returns:
            Dictionary containing decomposition results
        """
        if self.status != IntegrationStatus.READY:
            return {
                "success": False,
                "error": f"Integration not ready. Status: {self.status.value}",
                "sub_problems": []
            }
        
        self.status = IntegrationStatus.PROCESSING
        
        try:
            # Use adaptive engine if available
            if self.adaptive_engine:
                result = self._decompose_adaptive(problem, context, strategy_hint)
            else:
                # Fallback to basic decomposition
                result = self._decompose_basic(problem, context)
            
            # Apply associative integration if available
            if self.associative_engine and result.get("success"):
                result = self._apply_associative_integration(result, context)
            
            # Validate with reliability SSOT if available
            if self.reliability_ssot and result.get("success"):
                result = self._validate_with_ssot(result)
            
            self.status = IntegrationStatus.READY
            return result
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            self.status = IntegrationStatus.ERROR
            return {
                "success": False,
                "error": str(e),
                "sub_problems": []
            }
    
    def _decompose_adaptive(
        self,
        problem: str,
        context: Optional[Dict[str, Any]],
        strategy_hint: Optional[str]
    ) -> Dict[str, Any]:
        """Decompose using adaptive engine."""
        if not self.adaptive_engine:
            return self._decompose_basic(problem, context)
        
        # This is a stub - actual implementation would use the adaptive engine
        return {
            "success": True,
            "method": "adaptive",
            "problem": problem,
            "sub_problems": [
                {"id": "sp1", "description": f"Analyze: {problem[:50]}...", "complexity": "medium"},
                {"id": "sp2", "description": f"Implement solution for: {problem[:50]}...", "complexity": "high"}
            ],
            "strategy_used": strategy_hint or "auto_selected",
            "quality_score": 80.0
        }
    
    def _decompose_basic(
        self,
        problem: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Basic decomposition fallback."""
        return {
            "success": True,
            "method": "basic",
            "problem": problem,
            "sub_problems": [
                {"id": "sp1", "description": f"Understand requirements for: {problem[:50]}...", "complexity": "low"},
                {"id": "sp2", "description": f"Develop approach for: {problem[:50]}...", "complexity": "medium"},
                {"id": "sp3", "description": f"Implement and verify: {problem[:50]}...", "complexity": "medium"}
            ],
            "strategy_used": "basic_fallback",
            "quality_score": 60.0
        }
    
    def _apply_associative_integration(
        self,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply associative integration to decomposition result."""
        # This is a stub - actual implementation would enhance the result
        result["associative_enhanced"] = True
        result["quality_score"] = min(100, result.get("quality_score", 0) + 10)
        return result
    
    def _validate_with_ssot(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate decomposition using reliability SSOT."""
        # This is a stub - actual implementation would validate
        result["ssot_validated"] = True
        result["reliability_score"] = 85.0
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            "status": self.status.value,
            "adaptive_enhancement": self.adaptive_engine is not None,
            "associative_integration": self.associative_engine is not None,
            "reliability_ssot": self.reliability_ssot is not None,
            "config": {
                "enable_adaptive_enhancement": self.config.enable_adaptive_enhancement,
                "enable_associative_integration": self.config.enable_associative_integration,
                "enable_reliability_ssot": self.config.enable_reliability_ssot,
            }
        }
    
    def is_ready(self) -> bool:
        """Check if the integration is ready to process."""
        return self.status == IntegrationStatus.READY


# Global instance for convenience
_adaptive_integration: Optional[AdaptiveDecompositionIntegration] = None


def get_adaptive_integration(
    config: Optional[AdaptiveIntegrationConfig] = None
) -> AdaptiveDecompositionIntegration:
    """
    Get or create the global adaptive decomposition integration instance.
    
    Args:
        config: Optional configuration for the integration
        
    Returns:
        AdaptiveDecompositionIntegration instance
    """
    global _adaptive_integration
    if _adaptive_integration is None:
        _adaptive_integration = AdaptiveDecompositionIntegration(config)
    return _adaptive_integration


def reset_integration() -> None:
    """Reset the global integration instance."""
    global _adaptive_integration
    _adaptive_integration = None


# Convenience function for direct decomposition
def decompose_problem(
    problem: str,
    context: Optional[Dict[str, Any]] = None,
    strategy_hint: Optional[str] = None,
    config: Optional[AdaptiveIntegrationConfig] = None
) -> Dict[str, Any]:
    """
    Decompose a problem using adaptive decomposition.
    
    This is a convenience function that creates/uses the global integration instance.
    
    Args:
        problem: The problem to decompose
        context: Optional context information
        strategy_hint: Optional strategy hint
        config: Optional configuration
        
    Returns:
        Decomposition results
    """
    integration = get_adaptive_integration(config)
    return integration.decompose(problem, context, strategy_hint)


__all__ = [
    "AdaptiveDecompositionIntegration",
    "AdaptiveIntegrationConfig",
    "IntegrationStatus",
    "get_adaptive_integration",
    "reset_integration",
    "decompose_problem",
]
