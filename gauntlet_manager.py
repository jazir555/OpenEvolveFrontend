import json
import os
import time
from typing import List, Optional, Dict, Any
from openevolve_structures import GauntletDefinition, GauntletRoundRule

GAUNTLETS_FILE = "gauntlets.json" # Name of the file used for persisting gauntlet data.

class GauntletManager:
    """
    Manages the creation, retrieval, updating, and deletion of GauntletDefinition objects.
    Persists gauntlet data to a JSON file.
    """
    def __init__(self, gauntlets_file: str = GAUNTLETS_FILE):
        """Initializes the GauntletManager.

        Args:
            gauntlets_file (str): The name of the JSON file to use for persisting gauntlet data.
        """
        self.gauntlets_file = gauntlets_file
        self.gauntlets: Dict[str, GauntletDefinition] = self._load_gauntlets()

    def _load_gauntlets(self) -> Dict[str, GauntletDefinition]:
        """Loads gauntlets from the JSON file and deserializes them into GauntletDefinition objects.
        Handles deserialization of nested `GauntletRoundRule` objects, and optional fields like `description`,
        `attack_modes`, and `generation_mode`.
        """
        if os.path.exists(self.gauntlets_file):
            with open(self.gauntlets_file, "r") as f:
                data = json.load(f)
                loaded_gauntlets = {}
                for gauntlet_name, gauntlet_data in data.items():
                    rounds = []
                    for round_data in gauntlet_data['rounds']:
                        # Deserialize GauntletRoundRule objects
                        rounds.append(GauntletRoundRule(**round_data))
                    # Deserialize the GauntletDefinition object
                    loaded_gauntlets[gauntlet_name] = GauntletDefinition(
                        name=gauntlet_data['name'],
                        tenant_id=gauntlet_data.get('tenant_id'),
                        team_name=gauntlet_data['team_name'],
                        rounds=rounds,
                        description=gauntlet_data.get('description'),
                        attack_modes=gauntlet_data.get('attack_modes', []),
                        generation_mode=gauntlet_data.get('generation_mode', 'single_candidate')
                    )
                return loaded_gauntlets
        return {}

    def _save_gauntlets(self):
        """Serializes GauntletDefinition objects, including nested `GauntletRoundRule` objects, and saves them to the JSON file."""
        data = {}
        for name, gauntlet in self.gauntlets.items():
            # Convert GauntletDefinition object to a dictionary
            gauntlet_dict = gauntlet.__dict__.copy()
            # Convert GauntletRoundRule objects within the gauntlet's rounds to dictionaries
            gauntlet_dict['rounds'] = [r.__dict__ for r in gauntlet.rounds]
            data[name] = gauntlet_dict

        with open(self.gauntlets_file, "w") as f:
            json.dump(data, f, indent=4)

    def create_gauntlet(self, gauntlet: GauntletDefinition) -> bool:
        """Adds a new gauntlet to the manager and saves the changes."""
        if gauntlet.name in self.gauntlets:
            return False # Gauntlet with this name already exists
        self.gauntlets[gauntlet.name] = gauntlet
        self._save_gauntlets()
        return True

    def get_gauntlet(self, name: str) -> Optional[GauntletDefinition]:
        """Retrieves a gauntlet by its name."""
        return self.gauntlets.get(name)

    def get_all_gauntlets(self) -> List[GauntletDefinition]:
        """Retrieves all managed gauntlets."""
        return list(self.gauntlets.values())

    def update_gauntlet(self, gauntlet: GauntletDefinition) -> bool:
        """Updates an existing gauntlet and saves the changes."""
        if gauntlet.name not in self.gauntlets:
            return False # Gauntlet does not exist
        self.gauntlets[gauntlet.name] = gauntlet
        self._save_gauntlets()
        return True

    def delete_gauntlet(self, name: str) -> bool:
        """Deletes a gauntlet by its name and saves the changes."""
        if name in self.gauntlets:
            del self.gauntlets[name]
            self._save_gauntlets()
            return True
        return False


    def adapt_gauntlet_with_openevolve(
        self,
        gauntlet_name: str,
        performance_data: Dict[str, Any],
        api_key: str,
        max_iterations: int = 5
    ) -> bool:
        """
        Adapt gauntlet configuration using OpenEvolve based on performance data

        Args:
            gauntlet_name: Name of gauntlet to adapt
            performance_data: Historical performance data
            api_key: API key for OpenEvolve
            max_iterations: Number of evolution iterations

        Returns:
            True if adaptation successful
        """
        gauntlet = self.get_gauntlet(gauntlet_name)
        if not gauntlet:
            return False

        try:
            from openevolve_client import OpenEvolveClient
            import json

            client = OpenEvolveClient(api_key=api_key)

            # Create adaptation prompt
            current_config = {
                'name': gauntlet.name,
                'role': gauntlet.role,
                'num_rounds': len(gauntlet.rounds) if gauntlet.rounds else 0
            }

            adaptation_prompt = f"""Adapt this gauntlet configuration based on performance data:

Current Configuration:
{json.dumps(current_config, indent=2)}

Performance Data:
{json.dumps(performance_data, indent=2)}

Suggest improvements to make the gauntlet more effective. Return JSON with suggested changes."""

            # Run evolution
            result = client.evolve(
                content=adaptation_prompt,
                evolution_mode="standard",
                max_iterations=max_iterations,
                population_size=10,
                temperature=0.7,
                content_type="text_general"
            )

            # Parse suggestions
            suggestions = result.get('best_code', '{}')
            try:
                suggested_changes = json.loads(suggestions)

                # Track metrics
                if not hasattr(gauntlet, 'openevolve_metrics'):
                    gauntlet.openevolve_metrics = []

                gauntlet.openevolve_metrics.append({
                    'timestamp': time.time(),
                    'adaptation_metrics': result.get('metrics', {}),
                    'suggested_changes': suggested_changes
                })

                # Update gauntlet
                self.update_gauntlet(gauntlet)
                return True

            except json.JSONDecodeError:
                return False

        except Exception as e:
            print(f"Error adapting gauntlet with OpenEvolve: {e}")
            return False

    def track_openevolve_metrics(
        self,
        gauntlet_name: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Track OpenEvolve metrics for a gauntlet

        Args:
            gauntlet_name: Name of gauntlet
            metrics: Metrics to track

        Returns:
            True if successful
        """
        gauntlet = self.get_gauntlet(gauntlet_name)
        if not gauntlet:
            return False

        if not hasattr(gauntlet, 'openevolve_metrics'):
            gauntlet.openevolve_metrics = []

        gauntlet.openevolve_metrics.append({
            'timestamp': time.time(),
            'metrics': metrics
        })

        self.update_gauntlet(gauntlet)
        return True

    def get_gauntlet_effectiveness(
        self,
        gauntlet_name: str
    ) -> Dict[str, Any]:
        """
        Analyze gauntlet effectiveness from metrics

        Args:
            gauntlet_name: Name of gauntlet

        Returns:
            Dictionary with effectiveness analysis
        """
        gauntlet = self.get_gauntlet(gauntlet_name)
        if not gauntlet or not hasattr(gauntlet, 'openevolve_metrics'):
            return {
                'total_uses': 0,
                'avg_effectiveness': 0.0,
                'trend': 'unknown'
            }

        metrics_list = gauntlet.openevolve_metrics
        if not metrics_list:
            return {
                'total_uses': 0,
                'avg_effectiveness': 0.0,
                'trend': 'unknown'
            }

        # Calculate effectiveness
        total_uses = len(metrics_list)
        effectiveness_scores = []

        for entry in metrics_list:
            metrics = entry.get('metrics', {})
            score = metrics.get('best_fitness', 0.0)
            effectiveness_scores.append(score)

        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0

        # Determine trend
        if len(effectiveness_scores) >= 2:
            recent_avg = sum(effectiveness_scores[-5:]) / min(5, len(effectiveness_scores[-5:]))
            older_avg = sum(effectiveness_scores[:-5]) / len(effectiveness_scores[:-5]) if len(effectiveness_scores) > 5 else avg_effectiveness

            if recent_avg > older_avg * 1.1:
                trend = 'improving'
            elif recent_avg < older_avg * 0.9:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'total_uses': total_uses,
            'avg_effectiveness': avg_effectiveness,
            'trend': trend,
            'recent_scores': effectiveness_scores[-10:]
        }
