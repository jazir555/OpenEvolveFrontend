"""
Advanced OpenEvolve Integration for Adaptive MDAP/MAKER Adapter

This module provides advanced OpenEvolve workflow integration including:
- All workflow types (evolution, adversarial, sovereign, web3, rag)
- Advanced decomposition with MDAP-guided sub-problem creation
- Team selection based on complexity analysis
- Resource optimization for each workflow stage
- Workflow state persistence and recovery
- Multi-stage workflow orchestration

Federation Constitution Compliant.
"""

import os
import sys
import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .openevolve_integration import (
    get_openevolve_integration,
    OpenEvolveIntegrationConfig,
    WorkflowComplexityAnalysis,
    MAKERWorkflowDecision
)

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """All OpenEvolve workflow stages."""
    CONTENT_INPUT = "content_input"
    CONTENT_ANALYSIS = "content_analysis"
    PLANNING = "planning"
    DECOMPOSITION = "decomposition"
    SOLVING = "solving"
    VERIFICATION = "verification"
    ASSEMBLY = "assembly"
    EVALUATION = "evaluation"
    REFINEMENT = "refinement"


class TeamRole(Enum):
    """Team roles in OpenEvolve workflows."""
    CONTENT_ANALYZER = "content_analyzer"
    PLANNER = "planner"
    SOLVER = "solver"
    VERIFIER = "verifier"
    ASSEMBLER = "assembler"
    EVALUATOR = "evaluator"
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    PATCHER = "patcher"


@dataclass
class SubProblemDecomposition:
    """Result of decomposing a problem into sub-problems."""
    parent_problem_id: str
    sub_problems: List[Dict[str, Any]]
    decomposition_strategy: str
    estimated_total_complexity: float
    recommended_parallelization: int
    dependencies: Dict[str, List[str]]
    timestamp: str


@dataclass
class TeamSelectionResult:
    """Result of selecting teams for workflow."""
    workflow_id: str
    stage: str
    recommended_teams: Dict[str, str]
    team_sizes: Dict[str, int]
    reasoning: str
    estimated_cost: float
    timestamp: str


@dataclass
class ResourceOptimization:
    """Optimized resource allocation for workflow stage."""
    workflow_id: str
    stage: str
    cpu_allocation: float
    memory_allocation_mb: int
    timeout_ms: int
    max_parallelism: int
    retry_strategy: Dict[str, Any]
    estimated_cost_savings: float
    timestamp: str


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""
    workflow_id: str
    checkpoint_id: str
    stage: str
    state: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedOpenEvolveIntegration:
    """
    Advanced OpenEvolve integration with decomposition, team selection,
    and resource optimization.
    """

    def __init__(self, config: Optional[OpenEvolveIntegrationConfig] = None):
        """Initialize advanced integration."""
        self.base_integration = get_openevolve_integration()
        self.config = config

        # Workflow state persistence
        self.checkpoints: Dict[str, List[WorkflowCheckpoint]] = {}
        self.checkpoint_lock = threading.Lock()

        # Team registry (simulated, in production would load from database)
        self.team_registry = self._initialize_team_registry()

        logger.info("Advanced OpenEvolve Integration initialized")

    def _initialize_team_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize team registry with available teams."""
        return {
            "content_analyzer_general": {"role": "content_analyzer", "domain": "general", "capacity": 10},
            "content_analyzer_code": {"role": "content_analyzer", "domain": "code", "capacity": 8},
            "planner_strategic": {"role": "planner", "domain": "strategic", "capacity": 5},
            "planner_tactical": {"role": "planner", "domain": "tactical", "capacity": 8},
            "solver_general": {"role": "solver", "domain": "general", "capacity": 20},
            "solver_code": {"role": "solver", "domain": "code", "capacity": 15},
            "solver_math": {"role": "solver", "domain": "math", "capacity": 10},
            "verifier_strict": {"role": "verifier", "strictness": "high", "capacity": 5},
            "verifier_balanced": {"role": "verifier", "strictness": "medium", "capacity": 10},
            "assembler_general": {"role": "assembler", "domain": "general", "capacity": 8},
            "red_team_aggressive": {"role": "red_team", "style": "aggressive", "capacity": 5},
            "red_team_systematic": {"role": "red_team", "style": "systematic", "capacity": 8},
            "blue_team_defensive": {"role": "blue_team", "style": "defensive", "capacity": 8},
            "patcher_automated": {"role": "patcher", "type": "automated", "capacity": 10}
        }

    def decompose_problem(
        self,
        workflow_id: str,
        problem_statement: str,
        workflow_type: str,
        max_depth: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> SubProblemDecomposition:
        """
        Decompose problem into sub-problems using MDAP complexity analysis.

        Args:
            workflow_id: OpenEvolve workflow ID
            problem_statement: Problem to decompose
            workflow_type: Type of workflow
            max_depth: Maximum decomposition depth
            context: Additional context

        Returns:
            SubProblemDecomposition with sub-problems and strategy
        """
        logger.info(f"Decomposing problem for workflow {workflow_id}")

        # First, analyze overall complexity
        overall_analysis = self.base_integration.analyze_workflow_complexity(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
            workflow_type=workflow_type,
            context=context
        )

        # Determine decomposition strategy based on complexity
        if overall_analysis.overall_complexity > 0.8:
            strategy = "hierarchical_deep"
            n_sub_problems = min(8, int(overall_analysis.overall_complexity * 10))
        elif overall_analysis.overall_complexity > 0.5:
            strategy = "hierarchical_balanced"
            n_sub_problems = min(5, int(overall_analysis.overall_complexity * 8))
        else:
            strategy = "flat_shallow"
            n_sub_problems = max(2, int(overall_analysis.overall_complexity * 5))

        # Generate sub-problems
        sub_problems = []
        dependencies = {}

        for i in range(n_sub_problems):
            sub_problem_id = f"{workflow_id}_sub_{i}"
            sub_complexity = overall_analysis.overall_complexity * (0.5 + (i / n_sub_problems) * 0.5)

            sub_problems.append({
                "id": sub_problem_id,
                "description": f"Sub-problem {i+1}: {self._generate_sub_problem_description(problem_statement, i, n_sub_problems)}",
                "domain": context.get("domain", workflow_type) if context else workflow_type,
                "depth": min(max_depth, 1 + (i // 3)),
                "complexity": sub_complexity,
                "priority": "high" if i < 2 else "medium",
                "estimated_duration_ms": 30000 * (1 + sub_complexity)
            })

            # Create dependencies
            if i > 0:
                dependencies[sub_problem_id] = [f"{workflow_id}_sub_{max(0, i-2)}"]

        decomposition = SubProblemDecomposition(
            parent_problem_id=workflow_id,
            sub_problems=sub_problems,
            decomposition_strategy=strategy,
            estimated_total_complexity=overall_analysis.overall_complexity,
            recommended_parallelization=min(4, n_sub_problems // 2 + 1),
            dependencies=dependencies,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Save checkpoint
        self._save_checkpoint(workflow_id, "decomposition", {
            "sub_problems": sub_problems,
            "strategy": strategy
        })

        logger.info(
            f"Decomposition complete: {n_sub_problems} sub-problems, "
            f"strategy={strategy}, parallelization={decomposition.recommended_parallelization}"
        )

        return decomposition

    def select_teams_for_stage(
        self,
        workflow_id: str,
        stage: str,
        workflow_type: str,
        complexity_score: float,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> TeamSelectionResult:
        """
        Select optimal teams for a workflow stage based on complexity and requirements.

        Args:
            workflow_id: OpenEvolve workflow ID
            stage: Current workflow stage
            workflow_type: Type of workflow
            complexity_score: Overall complexity score
            domain: Problem domain
            context: Additional context

        Returns:
            TeamSelectionResult with recommended teams
        """
        logger.info(f"Selecting teams for {workflow_id} stage {stage}")

        context = context or {}
        recommended_teams = {}
        team_sizes = {}
        reasoning_parts = []

        # Determine required roles based on stage
        if stage == WorkflowStage.CONTENT_ANALYSIS.value:
            role = TeamRole.CONTENT_ANALYZER
            if domain == "code":
                recommended_teams["content_analyzer"] = "content_analyzer_code"
                team_sizes["content_analyzer"] = 1
                reasoning_parts.append("Code domain requires specialized code analyzer")
            else:
                recommended_teams["content_analyzer"] = "content_analyzer_general"
                team_sizes["content_analyzer"] = 1
                reasoning_parts.append("General domain uses general analyzer")

        elif stage == WorkflowStage.PLANNING.value or stage == WorkflowStage.DECOMPOSITION.value:
            if complexity_score > 0.7:
                recommended_teams["planner"] = "planner_strategic"
                team_sizes["planner"] = 2
                reasoning_parts.append("High complexity requires strategic planning")
            else:
                recommended_teams["planner"] = "planner_tactical"
                team_sizes["planner"] = 1
                reasoning_parts.append("Lower complexity allows tactical planning")

        elif stage == WorkflowStage.SOLVING.value:
            # Solver selection based on domain and complexity
            if domain == "math":
                recommended_teams["solver"] = "solver_math"
            elif domain == "code":
                recommended_teams["solver"] = "solver_code"
            else:
                recommended_teams["solver"] = "solver_general"

            # Team size based on complexity
            if complexity_score > 0.8:
                team_sizes["solver"] = 5
            elif complexity_score > 0.5:
                team_sizes["solver"] = 3
            else:
                team_sizes["solver"] = 1

            reasoning_parts.append(f"Solver team size {team_sizes['solver']} based on complexity {complexity_score:.2f}")

        elif stage == WorkflowStage.VERIFICATION.value:
            if complexity_score > 0.7:
                recommended_teams["verifier"] = "verifier_strict"
                team_sizes["verifier"] = 3
                reasoning_parts.append("High complexity requires strict verification")
            else:
                recommended_teams["verifier"] = "verifier_balanced"
                team_sizes["verifier"] = 1
                reasoning_parts.append("Standard verification sufficient")

        elif stage == WorkflowStage.ASSEMBLY.value:
            recommended_teams["assembler"] = "assembler_general"
            team_sizes["assembler"] = 1
            reasoning_parts.append("Standard assembly team")

        # Adversarial workflow special handling
        if workflow_type == "adversarial":
            recommended_teams["red_team"] = "red_team_aggressive" if complexity_score > 0.6 else "red_team_systematic"
            recommended_teams["blue_team"] = "blue_team_defensive"
            team_sizes["red_team"] = 2 if complexity_score > 0.6 else 1
            team_sizes["blue_team"] = 2
            reasoning_parts.append("Adversarial workflow requires red/blue teams")

        # Calculate estimated cost (simplified)
        estimated_cost = sum(
            self.team_registry.get(team_id, {}).get("capacity", 1) * team_sizes.get(role, 1)
            for role, team_id in recommended_teams.items()
        ) * 0.1

        selection = TeamSelectionResult(
            workflow_id=workflow_id,
            stage=stage,
            recommended_teams=recommended_teams,
            team_sizes=team_sizes,
            reasoning=". ".join(reasoning_parts),
            estimated_cost=estimated_cost,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        logger.info(f"Team selection complete: {len(recommended_teams)} teams, cost=${estimated_cost:.2f}")

        return selection

    def optimize_resources(
        self,
        workflow_id: str,
        stage: str,
        complexity_score: float,
        estimated_duration_ms: float,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ResourceOptimization:
        """
        Optimize resource allocation for a workflow stage.

        Args:
            workflow_id: OpenEvolve workflow ID
            stage: Current workflow stage
            complexity_score: Overall complexity score
            estimated_duration_ms: Estimated duration in milliseconds
            constraints: Resource constraints (max_cpu, max_memory, etc.)

        Returns:
            ResourceOptimization with optimal resource allocation
        """
        logger.info(f"Optimizing resources for {workflow_id} stage {stage}")

        constraints = constraints or {}

        # Calculate base allocation
        if complexity_score > 0.8:
            cpu_base = 2.0
            memory_base = 512
            parallelism_base = 4
        elif complexity_score > 0.5:
            cpu_base = 1.5
            memory_base = 256
            parallelism_base = 2
        else:
            cpu_base = 0.5
            memory_base = 128
            parallelism_base = 1

        # Apply constraints
        max_cpu = constraints.get("max_cpu", 4.0)
        max_memory = constraints.get("max_memory", 1024)
        max_parallelism = constraints.get("max_parallelism", 8)

        cpu_allocation = min(cpu_base, max_cpu)
        memory_allocation_mb = min(memory_base, max_memory)
        max_parallelism = min(parallelism_base, max_parallelism)

        # Calculate timeout with buffer
        timeout_ms = int(estimated_duration_ms * 1.5)

        # Retry strategy based on complexity
        if complexity_score > 0.7:
            max_retries = 5
            retry_delay_ms = 1000
            backoff_multiplier = 2.0
        elif complexity_score > 0.4:
            max_retries = 3
            retry_delay_ms = 500
            backoff_multiplier = 1.5
        else:
            max_retries = 1
            retry_delay_ms = 200
            backoff_multiplier = 1.0

        # Estimate cost savings (compared to over-provisioning)
        estimated_cost_savings = (1.0 - (cpu_allocation / max_cpu)) * 0.3 + \
                                  (1.0 - (memory_allocation_mb / max_memory)) * 0.2

        optimization = ResourceOptimization(
            workflow_id=workflow_id,
            stage=stage,
            cpu_allocation=cpu_allocation,
            memory_allocation_mb=memory_allocation_mb,
            timeout_ms=timeout_ms,
            max_parallelism=max_parallelism,
            retry_strategy={
                "max_retries": max_retries,
                "initial_delay_ms": retry_delay_ms,
                "backoff_multiplier": backoff_multiplier,
                "jitter": True
            },
            estimated_cost_savings=estimated_cost_savings,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        logger.info(
            f"Resource optimization complete: CPU={cpu_allocation}, "
            f"Memory={memory_allocation_mb}MB, Savings={estimated_cost_savings:.1%}"
        )

        return optimization

    def save_checkpoint(
        self,
        workflow_id: str,
        stage: str,
        state: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> WorkflowCheckpoint:
        """
        Save a workflow checkpoint for recovery.

        Args:
            workflow_id: OpenEvolve workflow ID
            stage: Current stage
            state: Current workflow state
            metrics: Optional metrics

        Returns:
            WorkflowCheckpoint
        """
        return self._save_checkpoint(workflow_id, stage, state, metrics)

    def _save_checkpoint(
        self,
        workflow_id: str,
        stage: str,
        state: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> WorkflowCheckpoint:
        """Internal checkpoint saving."""
        checkpoint_id = f"ckpt_{workflow_id}_{stage}_{int(time.time() * 1000)}"

        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            checkpoint_id=checkpoint_id,
            stage=stage,
            state=state,
            metrics=metrics or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"version": "1.0"}
        )

        with self.checkpoint_lock:
            if workflow_id not in self.checkpoints:
                self.checkpoints[workflow_id] = []
            self.checkpoints[workflow_id].append(checkpoint)

            # Keep only last 10 checkpoints per workflow
            if len(self.checkpoints[workflow_id]) > 10:
                self.checkpoints[workflow_id] = self.checkpoints[workflow_id][-10:]

        logger.info(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint

    def load_latest_checkpoint(
        self,
        workflow_id: str,
        stage: Optional[str] = None
    ) -> Optional[WorkflowCheckpoint]:
        """
        Load latest checkpoint for workflow.

        Args:
            workflow_id: OpenEvolve workflow ID
            stage: Optional stage filter

        Returns:
            Latest WorkflowCheckpoint or None
        """
        with self.checkpoint_lock:
            if workflow_id not in self.checkpoints:
                return None

            checkpoints = self.checkpoints[workflow_id]

            # Filter by stage if specified
            if stage:
                checkpoints = [cp for cp in checkpoints if cp.stage == stage]

            if not checkpoints:
                return None

            return checkpoints[-1]  # Return latest

    def export_checkpoints(self, workflow_id: str, filepath: str):
        """
        Export all checkpoints for a workflow to JSON.

        Args:
            workflow_id: OpenEvolve workflow ID
            filepath: Output file path
        """
        with self.checkpoint_lock:
            if workflow_id not in self.checkpoints:
                logger.warning(f"No checkpoints found for workflow {workflow_id}")
                return

            checkpoints_data = [
                asdict(cp) for cp in self.checkpoints[workflow_id]
            ]

            with open(filepath, 'w') as f:
                json.dump(checkpoints_data, f, indent=2)

            logger.info(f"Exported {len(checkpoints_data)} checkpoints to {filepath}")

    def _generate_sub_problem_description(self, parent_description: str, index: int, total: int) -> str:
        """Generate description for sub-problem."""
        # Split description into parts (simplified)
        words = parent_description.split()
        chunk_size = len(words) // total

        start_idx = index * chunk_size
        end_idx = start_idx + chunk_size if index < total - 1 else len(words)

        sub_words = words[start_idx:end_idx]
        return " ".join(sub_words)


# Global instance
_advanced_integration: Optional[AdvancedOpenEvolveIntegration] = None


def get_advanced_openevolve_integration() -> AdvancedOpenEvolveIntegration:
    """Get or create global advanced integration instance."""
    global _advanced_integration
    if _advanced_integration is None:
        _advanced_integration = AdvancedOpenEvolveIntegration()
    return _advanced_integration


__all__ = [
    "WorkflowStage",
    "TeamRole",
    "SubProblemDecomposition",
    "TeamSelectionResult",
    "ResourceOptimization",
    "WorkflowCheckpoint",
    "AdvancedOpenEvolveIntegration",
    "get_advanced_openevolve_integration"
]
