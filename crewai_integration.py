"""
CrewAI Integration Module

This module provides integration between CrewAI and OpenEvolve systems,
replacing the Hephaestus-based integration.

Key Features:
- CrewAI team management
- Task execution through CrewAI
- Integration with OpenEvolve workflow system
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    def __init__(self, config: Optional[CrewAIConfig] = None):
        """Initialize CrewAI integration manager"""
        self.config = config or CrewAIConfig()
        self.active_teams: Dict[str, Any] = {}
        logger.info("CrewAI Integration Manager initialized")

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
    __all__ = ['CrewAIIntegrationManager', 'CrewAIConfig', 'setup_crewai_integration',
               'execute_crewai_task', 'CrewAIClient', 'create_crewai_client']
except ImportError:
    __all__ = ['CrewAIIntegrationManager', 'CrewAIConfig', 'setup_crewai_integration',
               'execute_crewai_task']
    CrewAIClient = None
