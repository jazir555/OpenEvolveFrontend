import json
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from openevolve_structures import Team, ModelConfig

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for team operations
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

TEAMS_FILE = "teams.json" # Name of the file used for persisting team data.
logger = logging.getLogger(__name__)

class TeamManager:
    """
    Manages the creation, retrieval, updating, and deletion of Team objects.
    Persists team data to a JSON file.
    """
    def __init__(self, teams_file: str = TEAMS_FILE):
        """Initializes the TeamManager.

        Args:
            teams_file (str): The name of the JSON file to use for persisting team data.
        """
        self.teams_file = teams_file
        self.teams: Dict[str, Team] = self._load_teams()

    def _load_teams(self) -> Dict[str, Team]:
        """Loads teams from the JSON file and deserializes them into Team objects.
        Handles deserialization of nested `ModelConfig` objects and optional `description` field.
        """
        if os.path.exists(self.teams_file):
            with open(self.teams_file, "r") as f:
                data = json.load(f)
                loaded_teams = {}
                for team_name, team_data in data.items():
                    # Deserialize ModelConfig objects first
                    members = [ModelConfig(**mc) for mc in team_data['members']]
                    # Then deserialize the Team object
                    loaded_teams[team_name] = Team(
                        name=team_data['name'],
                        tenant_id=team_data.get('tenant_id'),
                        role=team_data['role'],
                        members=members,
                        description=team_data.get('description')
                    )
                return loaded_teams
        return {}

    def _save_teams(self):
        """Serializes Team objects, including nested `ModelConfig` objects, and saves them to the JSON file."""
        data = {}
        for name, team in self.teams.items():
            # Convert Team object to a dictionary
            team_dict = team.__dict__.copy()
            # Convert ModelConfig objects within the team's members to dictionaries
            team_dict['members'] = [member.__dict__ for member in team.members]
            data[name] = team_dict

        with open(self.teams_file, "w") as f:
            json.dump(data, f, indent=4)

    def create_team(self, team: Team) -> bool:
        """Adds a new team to the manager and saves the changes."""
        if team.name in self.teams:
            return False # Team with this name already exists
        self.teams[team.name] = team
        self._save_teams()
        return True

    def get_team(self, name: str) -> Optional[Team]:
        """Retrieves a team by its name."""
        return self.teams.get(name)

    def get_all_teams(self) -> List[Team]:
        """Retrieves all managed teams."""
        return list(self.teams.values())

    def update_team(self, team: Team) -> bool:
        """Updates an existing team and saves the changes."""
        if team.name not in self.teams:
            return False # Team does not exist
        self.teams[team.name] = team
        self._save_teams()
        return True

    def delete_team(self, name: str) -> bool:
        """Deletes a team by its name and saves the changes."""
        if name in self.teams:
            del self.teams[name]
            self._save_teams()
            return True
        return False

    def get_teams_by_role(self, role: str) -> List[Team]:
        """Retrieves all teams assigned to a specific role.

        Args:
            role (str): The role to filter teams by (e.g., "Blue", "Red", "Gold").
        """
        return [team for team in self.teams.values() if team.role == role]

    def add_openevolve_metrics(self, team_name: str, metrics: Dict[str, Any]) -> bool:
        """
        Add OpenEvolve metrics to a team

        Args:
            team_name: Name of the team
            metrics: OpenEvolve metrics to add

        Returns:
            True if successful, False if team not found
        """
        team = self.get_team(team_name)
        if not team:
            return False

        # Initialize openevolve_metrics if not present
        if not hasattr(team, 'openevolve_metrics'):
            team.openevolve_metrics = []

        # Add metrics with timestamp
        metrics_entry = {
            'timestamp': metrics.get('timestamp', None),
            'metrics': metrics
        }
        team.openevolve_metrics.append(metrics_entry)

        # Update team
        self.update_team(team)
        return True

    def get_openevolve_metrics(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Get OpenEvolve metrics for a team

        Args:
            team_name: Name of the team

        Returns:
            List of metrics entries
        """
        team = self.get_team(team_name)
        if not team or not hasattr(team, 'openevolve_metrics'):
            return []

        return team.openevolve_metrics

    def aggregate_team_metrics(self, team_name: str) -> Dict[str, Any]:
        """
        Aggregate OpenEvolve metrics for a team

        Args:
            team_name: Name of the team

        Returns:
            Aggregated metrics
        """
        metrics_list = self.get_openevolve_metrics(team_name)

        if not metrics_list:
            return {
                'total_operations': 0,
                'avg_fitness': 0.0,
                'total_iterations': 0,
                'total_cost': 0.0
            }

        # Aggregate metrics
        total_operations = len(metrics_list)
        total_fitness = 0.0
        total_iterations = 0
        total_cost = 0.0

        for entry in metrics_list:
            metrics = entry.get('metrics', {})
            total_fitness += metrics.get('best_fitness', 0.0)
            total_iterations += metrics.get('iterations_completed', 0)
            total_cost += metrics.get('cost_usd', 0.0)

        return {
            'total_operations': total_operations,
            'avg_fitness': total_fitness / total_operations if total_operations > 0 else 0.0,
            'total_iterations': total_iterations,
            'avg_iterations': total_iterations / total_operations if total_operations > 0 else 0.0,
            'total_cost': total_cost,
            'avg_cost': total_cost / total_operations if total_operations > 0 else 0.0
        }

    def get_all_teams_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated metrics for all teams

        Returns:
            Dictionary mapping team names to their aggregated metrics
        """
        all_metrics = {}

        for team_name in self.teams.keys():
            all_metrics[team_name] = self.aggregate_team_metrics(team_name)

        return all_metrics

    def clear_team_metrics(self, team_name: str) -> bool:
        """
        Clear OpenEvolve metrics for a team

        Args:
            team_name: Name of the team

        Returns:
            True if successful, False if team not found
        """
        team = self.get_team(team_name)
        if not team:
            return False

        if hasattr(team, 'openevolve_metrics'):
            team.openevolve_metrics = []
            self.update_team(team)

        return True

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting and knowledge for team operations
    # =========================================================================

    def _trigger_team_alerts(
        self,
        team_name: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for team operation failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Team Operation Failed: {team_name}",
                    description=f"Team operation failed for '{team_name}'. " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="team_manager",
                    component="team",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger team alert: {e}")

    def _extract_team_knowledge(
        self,
        team_name: str,
        operation: str,
        result: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract team operation knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"team_{team_name}_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="team_operation",
                source_component="team_manager",
                title=f"Team Operation: {team_name} - {operation}",
                content={
                    "team_name": team_name,
                    "operation": operation,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "success": result.get("success", True)
                },
                tags=["team", operation]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted team knowledge for {team_name} - {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract team knowledge: {e}")
            return False

    def _track_team_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        team_name: str
    ):
        """**ACTUAL INTEGRATION**: Track team operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"team_{operation}_{team_name}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "team_name": team_name
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked team performance for {team_name} - {operation}")

        except Exception as e:
            logger.error(f"Failed to track team performance: {e}")
