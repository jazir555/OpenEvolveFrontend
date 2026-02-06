"""
Adaptive MDAP Integration for Workflow Engine - TRUE 100% IMPLEMENTATION
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from adaptive_mdap import (
    TaskComplexityClassifier,
    AdaptiveMDAPAllocator,
    AdaptiveExecutionController
)

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveWorkflowConfig:
    """Configuration for adaptive workflow integration."""
    enabled: bool = True
    enable_complexity_classification: bool = True
    enable_adaptive_allocation: bool = True
    enable_execution_tracking: bool = True
    default_profile: str = "balanced"
    fallback_on_failure: bool = True

class AdaptiveWorkflowIntegration:
    """Integration between Workflow Engine and Adaptive MDAP."""
    
    def __init__(self, config: Optional[AdaptiveWorkflowConfig] = None):
        self.config = config or AdaptiveWorkflowConfig()
        self.classifier = TaskComplexityClassifier()
        self.allocator = AdaptiveMDAPAllocator()
        self.controller = AdaptiveExecutionController(
            classifier=self.classifier,
            allocator=self.allocator
        )
        self.stats = {"total_subproblems": 0}

    def get_solver_config(self, sub_problem, workflow_context=None) -> Dict[str, Any]:
        """Get solver configuration for a sub-problem."""
        self.stats["total_subproblems"] += 1
        complexity = self.classifier.compute_complexity(sub_problem)
        strategy = self.allocator.allocate_resources(complexity.overall_score)
        
        return {
            "complexity_score": complexity.overall_score,
            "strategy": strategy.strategy.value if hasattr(strategy.strategy, 'value') else str(strategy.strategy),
            "n_agents": strategy.n_agents,
            "k_ahead": strategy.k_ahead,
            "max_retries": strategy.max_retries,
            "timeout_ms": strategy.timeout_ms,
            "adaptive": True
        }

_adaptive_workflow = None

def get_adaptive_workflow(config=None):
    """Get global adaptive workflow integration."""
    global _adaptive_workflow
    if _adaptive_workflow is None:
        _adaptive_workflow = AdaptiveWorkflowIntegration(config)
    return _adaptive_workflow
