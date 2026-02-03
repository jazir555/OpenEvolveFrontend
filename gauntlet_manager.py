import json
import os
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from openevolve_structures import GauntletDefinition, GauntletRoundRule

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for gauntlet operations
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

GAUNTLETS_FILE = "gauntlets.json" # Name of the file used for persisting gauntlet data.
logger = logging.getLogger(__name__)

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

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting and knowledge for gauntlet operations
    # =========================================================================

    def _trigger_gauntlet_alerts(
        self,
        gauntlet_name: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for gauntlet failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.HIGH

                alert_manager.create_alert(
                    title=f"Gauntlet Failed: {gauntlet_name}",
                    description=f"Gauntlet '{gauntlet_name}' failed. " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="gauntlet_manager",
                    component="gauntlet",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger gauntlet alert: {e}")

    def _extract_gauntlet_knowledge(
        self,
        gauntlet_name: str,
        execution_result: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract gauntlet execution knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"gauntlet_{gauntlet_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="gauntlet_execution",
                source_component="gauntlet_manager",
                title=f"Gauntlet Execution: {gauntlet_name}",
                content={
                    "gauntlet_name": gauntlet_name,
                    "execution_result": execution_result,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "passed": execution_result.get("passed", False),
                    "score": execution_result.get("score", 0.0)
                },
                tags=["gauntlet", "testing", "adversarial"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted gauntlet knowledge for {gauntlet_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract gauntlet knowledge: {e}")
            return False

    def _track_gauntlet_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        gauntlet_name: str,
        score: float = 0.0
    ):
        """**ACTUAL INTEGRATION**: Track gauntlet performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = score if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"gauntlet_{operation}_{gauntlet_name}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "gauntlet_name": gauntlet_name,
                    "score": score
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked gauntlet performance for {gauntlet_name}")

        except Exception as e:
            logger.error(f"Failed to track gauntlet performance: {e}")


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

    def execute_gauntlet(
        self,
        gauntlet: GauntletDefinition,
        solution_content: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes a gauntlet against a solution.
        For now, this is a simulated execution that interfaces with the data models.
        """
        from sovereign_data_models import GauntletExecution, SolutionAttempt, generate_id
        from datetime import datetime

        start_time = time.time()
        execution_id = generate_id("exec")
        solution_id = generate_id("sol")

        # Create a mock solution attempt for the execution record
        solution = SolutionAttempt(
            id=solution_id,
            sub_problem_id=context.get("sub_problem_id", "root"),
            approach="automated_generation",
            solution_content=solution_content,
            team_id="default_team",
            confidence_score=0.8
        )

        execution = GauntletExecution(
            execution_id=execution_id,
            gauntlet_definition=gauntlet,
            sub_problem_id=context.get("sub_problem_id", "root"),
            solution_attempt=solution,
            start_time=datetime.now()
        )

        # Simple simulated pass/fail logic
        passed_rounds = 0
        for round_rule in gauntlet.rounds:
            passed_rounds += 1 # Simulation always passes for now

        execution.rounds_passed = passed_rounds
        execution.overall_passed = True
        execution.final_score = 1.0
        execution.end_time = datetime.now()

        duration = time.time() - start_time

        result = {
            "execution_id": execution_id,
            "passed": execution.overall_passed,
            "score": execution.final_score,
            "final_score": execution.final_score,
            "rounds_passed": execution.rounds_passed,
            "total_rounds": len(gauntlet.rounds),
            "rounds": [{"name": r.rule_id, "passed": True} for r in gauntlet.rounds],
            "feedback": ["Simulated gauntlet pass"]
        }

        # **ACTUAL INTEGRATION**: Extract knowledge, track performance, and trigger alerts
        self._extract_gauntlet_knowledge(gauntlet.name, result)
        self._track_gauntlet_performance("execute_gauntlet", result["passed"], duration, gauntlet.name, result["score"])

        if not result["passed"]:
            self._trigger_gauntlet_alerts(gauntlet.name, False, "Gauntlet execution failed")

        return result
