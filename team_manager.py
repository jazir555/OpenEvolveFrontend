import json
import os
from typing import List, Optional, Dict, Any
from workflow_structures import Team, ModelConfig # Assuming workflow_structures is in the same directory

TEAMS_FILE = "teams.json" # Name of the file used for persisting team data.

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
                    loaded_teams[team_name] = Team(name=team_data['name'], role=team_data['role'], members=members, description=team_data.get('description'))
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
