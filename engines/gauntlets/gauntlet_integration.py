"""
Gauntlet Integration Module

Provides gauntlet integration for OpenEvolve, bridging the core Gauntlet System
with Decomposition Engines and other OpenEvolve components.

Author: OpenEvolve Team
Date: 2026-02-06
"""
from __future__ import annotations


import logging
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

from gauntlet_system import GauntletSystem, GauntletSystemConfig
from gauntlet_decomposition_integration import GauntletDecompositionMixin

logger = logging.getLogger(__name__)


@dataclass
class GauntletIntegrationConfig:
    """Configuration for gauntlet integration"""
    mode: str = "standard"
    auto_assign_gauntlets: bool = True
    default_template: str = "standard"


class GauntletIntegration:
    """
    Gauntlet Integration Manager.
    
    Acts as the bridge between OpenEvolve core systems and the Gauntlet System.
    """
    
    def __init__(self, config: Optional[GauntletIntegrationConfig] = None):
        self.config = config or GauntletIntegrationConfig()
        
        # Initialize the core Gauntlet System
        system_config = GauntletSystemConfig(
            orchestration_mode="hierarchical" if self.config.mode == "strict" else "adaptive"
        )
        self.gauntlet_system = GauntletSystem(system_config)
        logger.info(f"Gauntlet Integration initialized in {self.config.mode} mode")
    
    def integrate(self, app: Any) -> Dict[str, Any]:
        """
        Integrate gauntlet system with an OpenEvolve application instance.
        
        Args:
            app: The OpenEvolve application instance (e.g., Flask app, CLI runner)
            
        Returns:
            Integration status
        """
        logger.info("Integrating Gauntlet System with OpenEvolve app")
        
        # Register services
        if hasattr(app, 'services'):
            app.services['gauntlet_system'] = self.gauntlet_system
            
        # Register hooks (example)
        if hasattr(app, 'register_hook'):
            app.register_hook('after_solution_generation', self.validate_solution)
            
        return {"integrated": True, "system": self.gauntlet_system}
    
    def create_enhanced_decomposition_engine(self, base_engine_cls: Type) -> Type:
        """
        Create a DecompositionEngine class enhanced with Gauntlet capabilities.
        
        Args:
            base_engine_cls: The base DecompositionEngine class
            
        Returns:
            A new class inheriting from both the base engine and GauntletDecompositionMixin
        """
        class EnhancedDecompositionEngine(GauntletDecompositionMixin, base_engine_cls):
            """Decomposition Engine with integrated Gauntlet System."""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.gauntlet_integration_config = self.config if hasattr(self, 'config') else {}
                
            def decompose(self, problem, *args, **kwargs):
                # Intercept decompose call to add gauntlet processing
                if kwargs.get('use_gauntlets', True):
                    return self.decompose_with_gauntlets(problem, *args, **kwargs)
                return super().decompose(problem, *args, **kwargs)
                
        return EnhancedDecompositionEngine
    
    def validate_solution(self, solution: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate a solution using the configured Gauntlet System.
        
        Args:
            solution: Solution data
            context: Execution context
            
        Returns:
            Validation results
        """
        logger.info(f"Validating solution: {solution.get('id', 'unknown')}")
        return self.gauntlet_system.evaluate(solution)
    
    def configure(self, options: Dict[str, Any]) -> None:
        """Configure integration options."""
        self.config.mode = options.get("mode", self.config.mode)
        self.config.auto_assign_gauntlets = options.get("auto_assign_gauntlets", self.config.auto_assign_gauntlets)


def create_gauntlet_integration(config: Optional[GauntletIntegrationConfig] = None) -> GauntletIntegration:
    """Factory function to create gauntlet integration instance"""
    return GauntletIntegration(config)
