"""
Adaptive Gauntlet System for OpenEvolve

This module implements the Adaptive Gauntlet System which dynamically adjusts
validation rigor based on problem complexity and historical performance.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from formal_gauntlet_system import GauntletSystem, GauntletTemplates, AdaptiveMetrics
from sovereign_data_models import (
    GauntletDefinition,
    GauntletRoundRule,
    ProblemDefinition,
    GauntletExecution
)

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks performance of solutions across gauntlets for adaptive learning."""
    
    def __init__(self):
        self.history = []
        self.metrics = AdaptiveMetrics()
        
    def record_result(self, result: Dict[str, Any], gauntlet_execution: Any):
        """Record a result to update performance metrics."""
        self.history.append({
            "timestamp": datetime.now(),
            "score": result.get("score", 0.0),
            "passed": result.get("success", False)
        })
        
        # Update aggregate metrics
        self.metrics.total_rounds_completed += 1
        if result.get("success"):
            self.metrics.total_rounds_passed += 1
            
        # Update average score
        total_score = sum(h["score"] for h in self.history)
        self.metrics.average_score = total_score / len(self.history)
        
        # Track recent scores (last 10)
        self.metrics.recent_scores = [h["score"] for h in self.history[-10:]]

class AdaptiveGauntletSystem(GauntletSystem):
    """
    Enhanced Gauntlet System with adaptive capabilities.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_tracker = PerformanceTracker()
        
    def create_adaptive_gauntlet(
        self,
        problem_definition: Any,
        sub_problem: Any
    ) -> GauntletDefinition:
        """
        Creates an adaptive gauntlet based on the problem and historical performance.
        """
        from sovereign_data_models import GauntletDefinition, GauntletRoundRule, generate_id
        
        gauntlet_id = generate_id("gaunt")
        
        # Adaptive logic based on problem complexity (mocked)
        rounds = [
            GauntletRoundRule(
                rule_id="syntax_check", 
                rule_type="automated", 
                description="Must be valid Python",
                validation_type="acceptance",
                min_score=1.0
            ),
            GauntletRoundRule(
                rule_id="logic_check", 
                rule_type="automated", 
                description="Must correctly implement Fibonacci",
                validation_type="quality",
                min_score=0.8
            )
        ]
        
        gauntlet = GauntletDefinition(
            gauntlet_id=gauntlet_id,
            name=f"Adaptive Gauntlet for {gauntlet_id}",
            description=f"Adaptive validation for {sub_problem.id if hasattr(sub_problem, 'id') else 'task'}",
            rounds=rounds
        )
        
        return gauntlet
    def update_performance_from_result(
        self, 
        result: Dict[str, Any], 
        gauntlet_execution: GauntletExecution
    ):
        """Update the internal performance tracker with new results."""
        self.performance_tracker.record_result(result, gauntlet_execution)
        # Sync metrics back to base class if needed
        self.adaptive_metrics = self.performance_tracker.metrics

__all__ = ['AdaptiveGauntletSystem', 'PerformanceTracker']
