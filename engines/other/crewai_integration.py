"""
CrewAI Integration Module

This module provides integration between CrewAI and OpenEvolve systems,
replacing the legacy orchestration integration.

Key Features:
- CrewAI team management
- Task execution through CrewAI
- Integration with OpenEvolve workflow system
"""
from __future__ import annotations


import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from utils.entanglement_utils import normalize_entanglement_matrix, serialize_entanglement_matrix
except ImportError:
    def normalize_entanglement_matrix(matrix, allowed_ids=None, enforce_symmetry=True, strict=False):
        if not matrix:
            return {}
        norm = {}
        for src, targets in matrix.items():
            if allowed_ids is not None and src not in allowed_ids:
                continue
            norm[src] = list(targets) if targets else []
        if enforce_symmetry:
            for src, targets in list(norm.items()):
                for t in targets:
                    norm.setdefault(t, [])
                    if src not in norm[t]:
                        norm[t].append(src)
        return norm

    def serialize_entanglement_matrix(normalized):
        return {k: list(v) for k, v in normalized.items()}

logger = logging.getLogger(__name__)


class TicketStatus(Enum):
    """Ticket status for CrewAI integration tasks.
    
    Maps to workflow states for tracking task progression.
    """
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    BLOCKED = "blocked"


class TicketType(Enum):
    """Ticket type classification for CrewAI tasks."""
    TASK = "task"
    BUG = "bug"
    FEATURE = "feature"
    EPIC = "epic"


@dataclass
class CrewAIConfig:
    """CrewAI configuration"""
    enable_crewai: bool = True
    verbose_logging: bool = False
    max_execution_time: int = 300  # 5 minutes default
    enable_caching: bool = True


class CrewAIIntegrationManager:
    """
    Manager for CrewAI integration with OpenEvolve

    This class provides methods to execute CrewAI workflows and teams
    within the OpenEvolve ecosystem.
    """

    def __init__(
        self,
        config: Optional[CrewAIConfig] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initialize CrewAI integration manager"""
        if isinstance(config, str):
            api_base, api_key, project_id = config, api_base, api_key
            config = None
        self.config = config or CrewAIConfig()
        self.api_base = api_base
        self.api_key = api_key
        self.project_id = project_id
        self.active_teams: Dict[str, Any] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        logger.info("CrewAI Integration Manager initialized")

    def _normalize_entanglement_matrix(
        self,
        matrix: Optional[Dict[str, Any]],
        allowed_ids: Optional[List[str]] = None,
        strict: bool = False,
    ) -> Dict[str, List[str]]:
        normalized = normalize_entanglement_matrix(
            matrix,
            allowed_ids=allowed_ids,
            enforce_symmetry=True,
            strict=strict,
        )
        return serialize_entanglement_matrix(normalized)

    def initialize_workflow_sync(self, workflow_state: Any) -> bool:
        """Initialize a CrewAI workflow mirror and attach entanglement metadata."""
        if not workflow_state or not getattr(workflow_state, "decomposition_plan", None):
            logger.warning("CrewAI init skipped: missing decomposition plan")
            return False

        workflow_id = getattr(workflow_state, "crewai_workflow_id", None)
        if not workflow_id:
            workflow_id = f"crewai_{uuid.uuid4().hex[:10]}"
            workflow_state.crewai_workflow_id = workflow_id

        entanglement = getattr(workflow_state, "entanglement_matrix", None)
        if not entanglement:
            plan_ctx = getattr(workflow_state.decomposition_plan, "analyzed_context", {}) or {}
            entanglement = plan_ctx.get("entanglement_matrix", {})

        allowed_ids = [sp.id for sp in workflow_state.decomposition_plan.sub_problems]
        entanglement = self._normalize_entanglement_matrix(
            entanglement,
            allowed_ids=allowed_ids,
            strict=bool(getattr(workflow_state, "entanglement_strict_mode", False)),
        )

        workflow_record = {
            "workflow_id": workflow_id,
            "created_at": datetime.utcnow().isoformat(),
            "entanglement_matrix": entanglement,
            "tasks": [],
        }

        for sub_problem in workflow_state.decomposition_plan.sub_problems:
            task_id = f"{workflow_id}:{sub_problem.id}"
            entangled_with = entanglement.get(sub_problem.id, [])
            task_record = {
                "task_id": task_id,
                "workflow_id": workflow_id,
                "sub_problem_id": sub_problem.id,
                "description": sub_problem.description,
                "dependencies": list(sub_problem.dependencies or []),
                "entangled_with": entangled_with,
                "status": "pending",
                "metadata": {
                    "entangled_with": entangled_with,
                    "entanglement_source": "decomposition_system",
                },
            }
            self.active_tasks[task_id] = task_record
            workflow_record["tasks"].append(task_id)
            workflow_state.id_to_ticket_id_map[sub_problem.id] = task_id
            workflow_state.ticket_id_to_subproblem_id_map[task_id] = sub_problem.id

        self.active_workflows[workflow_id] = workflow_record
        logger.info("CrewAI workflow mirror initialized: %s", workflow_id)
        return True

    def update_subproblem_status(
        self,
        workflow_state: Any,
        sub_problem_id: str,
        new_status: str,
        solution_content: Optional[str] = None,
    ) -> bool:
        task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id) if workflow_state else None
        if not task_id or task_id not in self.active_tasks:
            logger.warning("CrewAI status sync skipped: task missing for %s", sub_problem_id)
            return False
        task = self.active_tasks[task_id]
        task["status"] = new_status
        if solution_content:
            task.setdefault("metadata", {})["solution_content"] = solution_content
        task["updated_at"] = datetime.utcnow().isoformat()
        return True

    def sync_solution_to_ticket(self, workflow_state: Any, sub_problem_id: str, solution: Any) -> bool:
        task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id) if workflow_state else None
        if not task_id or task_id not in self.active_tasks:
            logger.warning("CrewAI solution sync skipped: task missing for %s", sub_problem_id)
            return False
        task = self.active_tasks[task_id]
        solution_content = getattr(solution, "solution_content", None) or getattr(solution, "content", None)
        task.setdefault("metadata", {})["solution"] = solution_content
        task["updated_at"] = datetime.utcnow().isoformat()
        return True

    def sync_critique_to_ticket(self, workflow_state: Any, sub_problem_id: str, critique: Any) -> bool:
        task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id) if workflow_state else None
        if not task_id or task_id not in self.active_tasks:
            logger.warning("CrewAI critique sync skipped: task missing for %s", sub_problem_id)
            return False
        task = self.active_tasks[task_id]
        task.setdefault("metadata", {})["critique"] = getattr(critique, "summary", None) or str(critique)
        task["updated_at"] = datetime.utcnow().isoformat()
        return True

    def sync_verification_to_ticket(self, workflow_state: Any, sub_problem_id: str, verification: Any) -> bool:
        task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id) if workflow_state else None
        if not task_id or task_id not in self.active_tasks:
            logger.warning("CrewAI verification sync skipped: task missing for %s", sub_problem_id)
            return False
        task = self.active_tasks[task_id]
        task.setdefault("metadata", {})["verification"] = getattr(verification, "summary", None) or str(verification)
        task["updated_at"] = datetime.utcnow().isoformat()
        return True

    def execute_task(
        self,
        task_name: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        team_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a CrewAI task

        Args:
            task_name: Name of the task to execute
            task_description: Description of what the task should do
            context: Additional context for the task
            team_name: Optional team name to execute the task with

        Returns:
            Dict containing task execution results
        """
        logger.info(f"Executing CrewAI task: {task_name}")

        # Placeholder for actual CrewAI execution
        # This would integrate with the actual CrewAI framework
        result = {
            "task_name": task_name,
            "status": "completed",
            "result": f"Task '{task_name}' executed successfully",
            "context": context or {}
        }

        return result

    def create_team(
        self,
        team_name: str,
        agents: List[Dict[str, Any]],
        processes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a CrewAI team

        Args:
            team_name: Name for the team
            agents: List of agent configurations
            processes: List of process configurations

        Returns:
            Dict containing team creation results
        """
        logger.info(f"Creating CrewAI team: {team_name}")

        team_info = {
            "team_name": team_name,
            "num_agents": len(agents),
            "num_processes": len(processes),
            "status": "created"
        }

        self.active_teams[team_name] = team_info
        return team_info

    def get_team_status(self, team_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a team

        Args:
            team_name: Name of the team

        Returns:
            Team status information or None if not found
        """
        return self.active_teams.get(team_name)

    def list_teams(self) -> List[str]:
        """List all active teams"""
        return list(self.active_teams.keys())

    def shutdown(self):
        """Shutdown the CrewAI integration manager"""
        logger.info("Shutting down CrewAI Integration Manager")
        self.active_teams.clear()


def setup_crewai_integration(
    enable_crewai: bool = True,
    verbose_logging: bool = False,
    max_execution_time: int = 300
) -> CrewAIIntegrationManager:
    """
    Setup and initialize CrewAI integration

    Args:
        enable_crewai: Whether to enable CrewAI features
        verbose_logging: Enable verbose logging
        max_execution_time: Maximum time for task execution

    Returns:
        Initialized CrewAIIntegrationManager instance
    """
    config = CrewAIConfig(
        enable_crewai=enable_crewai,
        verbose_logging=verbose_logging,
        max_execution_time=max_execution_time
    )

    manager = CrewAIIntegrationManager(config)
    logger.info("CrewAI integration setup complete")

    return manager


# Convenience function for quick task execution
def execute_crewai_task(
    task_name: str,
    task_description: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute a CrewAI task without explicit manager

    Args:
        task_name: Name of the task
        task_description: Description of the task
        context: Additional context

    Returns:
        Task execution results
    """
    manager = setup_crewai_integration()
    result = manager.execute_task(task_name, task_description, context)
    manager.shutdown()
    return result


# Re-export CrewAIClient from crewai_client for backward compatibility
try:
    from crewai_client import CrewAIClient, create_crewai_client
    __all__ = ['CrewAIIntegrationManager', 'CrewAIConfig', 'TicketStatus', 'TicketType',
               'setup_crewai_integration', 'execute_crewai_task', 'CrewAIClient', 'create_crewai_client']
except ImportError:
    __all__ = ['CrewAIIntegrationManager', 'CrewAIConfig', 'TicketStatus', 'TicketType',
               'setup_crewai_integration', 'execute_crewai_task']
    CrewAIClient = None
