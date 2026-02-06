"""
Gauntlet System Module

Provides a unified facade for the OpenEvolve gauntlet system, orchestrating
validation, red teaming, and formal verification.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from gauntlet_manager import GauntletManager
from gauntlet_orchestrator import GauntletOrchestrator, OrchestrationMode, run_comprehensive_gauntlet_validation
from gauntlet_types import GauntletType

logger = logging.getLogger(__name__)


@dataclass
class GauntletSystemConfig:
    """Configuration for gauntlet system"""
    num_rounds: int = 3
    timeout: int = 300
    orchestration_mode: str = "hierarchical"  # sequential, parallel, hierarchical, adaptive
    use_red_team: bool = True
    use_gold_team: bool = True
    enable_formal_verification: bool = True


class GauntletSystem:
    """
    Gauntlet System Facade.
    
    Unified entry point for accessing:
    - Gauntlet Manager (CRUD, persistence)
    - Gauntlet Orchestrator (Execution flow)
    - Advanced Gauntlet Types (Red/Gold teams, Z3, etc.)
    """
    
    def __init__(self, config: Optional[GauntletSystemConfig] = None):
        self.config = config or GauntletSystemConfig()
        self.manager = GauntletManager()
        self.orchestrator = GauntletOrchestrator()
        logger.info(f"Gauntlet System initialized with mode: {self.config.orchestration_mode}")
    
    def run(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a comprehensive gauntlet validation on a problem definition.
        
        Args:
            problem: Problem definition dictionary
            
        Returns:
            Validation results
        """
        logger.info(f"Running gauntlet for problem: {problem.get('title', 'unknown')}")
        
        # Create a mock solution from the problem description for initial validation
        # In a real workflow, this would be the generated solution
        solution_content = problem.get('description', '') or problem.get('content', '')
        
        # Determine orchestration mode
        mode_map = {
            "sequential": OrchestrationMode.SEQUENTIAL,
            "parallel": OrchestrationMode.PARALLEL,
            "hierarchical": OrchestrationMode.HIERARCHICAL,
            "adaptive": OrchestrationMode.ADAPTIVE
        }
        mode = mode_map.get(self.config.orchestration_mode, OrchestrationMode.HIERARCHICAL)
        
        # Run comprehensive validation
        context = {
            "problem_type": problem.get("domain", "general"),
            "constraints": problem.get("constraints", []),
            "timeout": self.config.timeout
        }
        
        result = run_comprehensive_gauntlet_validation(
            solution=solution_content,
            context=context,
            mode=mode
        )
        
        return result.to_dict()
    
    def evaluate(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a specific submission/solution.
        
        Args:
            submission: Submission dictionary containing 'content' or 'code'
            
        Returns:
            Evaluation results with score and feedback
        """
        content = submission.get('content') or submission.get('code') or str(submission)
        domain = submission.get('domain', 'general')
        
        logger.info(f"Evaluating submission in domain: {domain}")
        
        # Use GauntletManager to create and execute an appropriate gauntlet
        gauntlet_def = self.manager.create_adaptive_gauntlet(
            name=f"eval_{int(logging.time.time())}",
            content=content,
            content_type=domain
        )
        
        if gauntlet_def:
            # Execute configured gauntlet
            result = self.manager.execute_gauntlet(
                gauntlet=gauntlet_def,
                solution_content=content,
                context={"domain": domain}
            )
            return result
        else:
            # Fallback to orchestration if adaptive creation fails
            context = {"domain": domain}
            
            # Re-creating gauntlets for orchestrator manually as fallback
            from gauntlet_orchestrator import create_all_gauntlets
            gauntlets = create_all_gauntlets()
            
            result = self.orchestrator.orchestrate(
                mode=OrchestrationMode.ADAPTIVE,
                gauntlets=gauntlets,
                solution=content,
                context=context
            )
            
            return result.to_dict()


def create_gauntlet_system(config: Optional[GauntletSystemConfig] = None) -> GauntletSystem:
    """Factory function to create gauntlet system instance"""
    return GauntletSystem(config)
