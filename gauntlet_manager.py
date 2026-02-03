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

# Adaptive MDAP not available
ADAPTIVE_MDAP_AVAILABLE = False

# **BUBBLELABS INTEGRATION**: BubbleLab workflow visualization for gauntlets
try:
    from bubblelabs_gauntlet_bubbles import (
        create_gauntlet_execution_bubble,
        create_gauntlet_round_bubble,
        create_gauntlet_result_bubble,
        create_red_team_bubble,
        create_blue_team_bubble,
        create_gold_team_bubble,
        create_loongeval_bubble,
        create_bubble_edge,
        create_3_round_gauntlet_workflow,
        update_bubble_status,
        add_bubble_result,
        GauntletBubbleConfig,
    )
    BUBBLELABS_AVAILABLE = True
except ImportError:
    BUBBLELABS_AVAILABLE = False
    create_gauntlet_execution_bubble = None
    create_gauntlet_round_bubble = None
    create_gauntlet_result_bubble = None
    create_red_team_bubble = None
    create_blue_team_bubble = None
    create_gold_team_bubble = None
    create_loongeval_bubble = None
    create_bubble_edge = None
    create_3_round_gauntlet_workflow = None
    update_bubble_status = None
    add_bubble_result = None
    GauntletBubbleConfig = None

GAUNTLETS_FILE = "gauntlets.json" # Name of the file used for persisting gauntlet data.
logger = logging.getLogger(__name__)

class GauntletManager:
    """
    Manages the creation, retrieval, updating, and deletion of GauntletDefinition objects.
    Persists gauntlet data to a JSON file.
    Also manages BubbleLab workflow visualization for gauntlets.
    """
    def __init__(self, gauntlets_file: str = GAUNTLETS_FILE):
        """Initializes the GauntletManager.

        Args:
            gauntlets_file (str): The name of the JSON file to use for persisting gauntlet data.
        """
        self.gauntlets_file = gauntlets_file
        self.gauntlets: Dict[str, GauntletDefinition] = self._load_gauntlets()
        
        # **BUBBLELABS INTEGRATION**: Storage for BubbleLab workflow visualizations
        self.bubble_workflows: Dict[str, Dict[str, Any]] = {}
        self.bubble_nodes: Dict[str, Dict[str, Any]] = {}
        self.execution_to_bubble_map: Dict[str, str] = {}  # Maps execution_id to bubble_id

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

        # **BUBBLELABS INTEGRATION**: Update bubble nodes with execution results
        if BUBBLELABS_AVAILABLE:
            try:
                # Find and update result bubble
                workflows = self.get_bubble_workflows_for_gauntlet(gauntlet.name)
                for workflow in workflows:
                    for node in workflow.get("nodes", []):
                        if node.get("type") == "gauntlet_result":
                            status = "passed" if result["passed"] else "failed"
                            self.update_bubble_node_status(node["id"], status, {
                                "score": result.get("score", 0.0),
                                "feedback": result.get("feedback", []),
                                "execution_id": execution_id
                            })
                            break
            except Exception as e:
                logger.error(f"Failed to update bubble status: {e}")

        if not result["passed"]:
            self._trigger_gauntlet_alerts(gauntlet.name, False, "Gauntlet execution failed")

        return result
    
    # =========================================================================
    # BUBBLELABS INTEGRATION METHODS - BubbleLab workflow visualization
    # =========================================================================
    
    def create_bubble_workflow_from_gauntlet(
        self,
        gauntlet: GauntletDefinition,
        problem_statement: str = ""
    ) -> Optional[Dict[str, Any]]:
        """**BUBBLELABS INTEGRATION**: Create a BubbleLab workflow from a gauntlet definition.
        
        Args:
            gauntlet: The GauntletDefinition to create workflow for
            problem_statement: Optional problem context
            
        Returns:
            Dict with workflow nodes and edges, or None if BubbleLabs unavailable
        """
        if not BUBBLELABS_AVAILABLE or not create_3_round_gauntlet_workflow:
            logger.warning("BubbleLabs integration not available")
            return None
        
        try:
            # Determine team names from gauntlet configuration
            team_config = {
                "red_team": getattr(gauntlet, 'red_team_name', "Red Team"),
                "blue_team": getattr(gauntlet, 'blue_team_name', "Blue Team"),
                "gold_team": gauntlet.team_name or "Gold Team"
            }
            
            # Create the 3-round gauntlet workflow
            workflow = create_3_round_gauntlet_workflow(
                problem_statement=problem_statement or f"Gauntlet: {gauntlet.name}",
                gauntlet_name=gauntlet.name,
                team_config=team_config
            )
            
            # Store the workflow
            workflow_id = workflow["id"]
            self.bubble_workflows[workflow_id] = workflow
            
            # Store individual nodes for tracking
            for node in workflow["nodes"]:
                self.bubble_nodes[node["id"]] = {
                    "node": node,
                    "workflow_id": workflow_id,
                    "gauntlet_name": gauntlet.name,
                    "status": "pending"
                }
            
            logger.info(f"Created BubbleLab workflow {workflow_id} for gauntlet {gauntlet.name}")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to create bubble workflow for gauntlet {gauntlet.name}: {e}")
            return None
    
    def get_bubble_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a BubbleLab workflow by ID.
        
        Args:
            workflow_id: The workflow ID
            
        Returns:
            Workflow dict or None
        """
        return self.bubble_workflows.get(workflow_id)
    
    def get_bubble_workflows_for_gauntlet(self, gauntlet_name: str) -> List[Dict[str, Any]]:
        """Get all BubbleLab workflows for a gauntlet.
        
        Args:
            gauntlet_name: Name of the gauntlet
            
        Returns:
            List of workflow dicts
        """
        workflows = []
        for workflow in self.bubble_workflows.values():
            if workflow.get("metadata", {}).get("gauntlet_name") == gauntlet_name:
                workflows.append(workflow)
        return workflows
    
    def update_bubble_node_status(
        self,
        node_id: str,
        status: str,
        additional_data: Dict[str, Any] = None
    ) -> bool:
        """**BUBBLELABS INTEGRATION**: Update the status of a bubble node.
        
        Args:
            node_id: ID of the node to update
            status: New status (pending, running, passed, failed, partial)
            additional_data: Optional additional data to merge
            
        Returns:
            True if update successful
        """
        if node_id not in self.bubble_nodes:
            return False
        
        node_info = self.bubble_nodes[node_id]
        bubble = node_info["node"]
        
        if update_bubble_status:
            updated_bubble = update_bubble_status(bubble, status, additional_data)
            node_info["node"] = updated_bubble
            node_info["status"] = status
            
            # Update in workflow
            workflow_id = node_info["workflow_id"]
            if workflow_id in self.bubble_workflows:
                workflow = self.bubble_workflows[workflow_id]
                for i, node in enumerate(workflow["nodes"]):
                    if node["id"] == node_id:
                        workflow["nodes"][i] = updated_bubble
                        break
            
            return True
        
        return False
    
    def add_result_to_bubble(
        self,
        node_id: str,
        score: float,
        feedback: str,
        improvements: List[str] = None
    ) -> bool:
        """**BUBBLELABS INTEGRATION**: Add execution result to a bubble node.
        
        Args:
            node_id: ID of the node to update
            score: Execution score (0.0 to 1.0)
            feedback: Feedback message
            improvements: List of improvement suggestions
            
        Returns:
            True if update successful
        """
        if node_id not in self.bubble_nodes:
            return False
        
        node_info = self.bubble_nodes[node_id]
        bubble = node_info["node"]
        
        if add_bubble_result:
            updated_bubble = add_bubble_result(bubble, score, feedback, improvements)
            node_info["node"] = updated_bubble
            node_info["status"] = "passed" if score >= 0.7 else "failed"
            
            # Update in workflow
            workflow_id = node_info["workflow_id"]
            if workflow_id in self.bubble_workflows:
                workflow = self.bubble_workflows[workflow_id]
                for i, node in enumerate(workflow["nodes"]):
                    if node["id"] == node_id:
                        workflow["nodes"][i] = updated_bubble
                        break
            
            return True
        
        return False
    
    def map_execution_to_bubble(
        self,
        execution_id: str,
        bubble_id: str
    ) -> None:
        """Map a gauntlet execution ID to a bubble node ID for tracking.
        
        Args:
            execution_id: The gauntlet execution ID
            bubble_id: The bubble node ID
        """
        self.execution_to_bubble_map[execution_id] = bubble_id
    
    def get_bubble_for_execution(self, execution_id: str) -> Optional[str]:
        """Get the bubble node ID for a gauntlet execution.
        
        Args:
            execution_id: The gauntlet execution ID
            
        Returns:
            Bubble node ID or None
        """
        return self.execution_to_bubble_map.get(execution_id)
    
    def execute_gauntlet_with_bubbles(
        self,
        gauntlet: GauntletDefinition,
        solution_content: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute gauntlet with full BubbleLab visualization integration.
        
        Args:
            gauntlet: The gauntlet to execute
            solution_content: The solution to evaluate
            context: Execution context
            
        Returns:
            Execution result with bubble updates
        """
        # Create bubble workflow if not exists
        workflow_id = None
        problem_statement = context.get("problem_statement", "")
        
        if BUBBLELABS_AVAILABLE:
            existing_workflows = self.get_bubble_workflows_for_gauntlet(gauntlet.name)
            if not existing_workflows:
                workflow = self.create_bubble_workflow_from_gauntlet(gauntlet, problem_statement)
                if workflow:
                    workflow_id = workflow["id"]
            else:
                workflow_id = existing_workflows[0]["id"]
        
        # Update input node status
        if workflow_id and BUBBLELABS_AVAILABLE:
            workflow = self.get_bubble_workflow(workflow_id)
            if workflow:
                for node in workflow["nodes"]:
                    if node["data"].get("label", "").startswith("📥"):
                        self.update_bubble_node_status(node["id"], "running", {
                            "problem_statement": problem_statement
                        })
                        break
        
        # Execute the gauntlet
        result = self.execute_gauntlet(gauntlet, solution_content, context)
        
        # Update bubbles with results
        if workflow_id and BUBBLELABS_AVAILABLE:
            workflow = self.get_bubble_workflow(workflow_id)
            if workflow:
                execution_id = result.get("execution_id")
                
                # Find and update the result bubble
                for node in workflow["nodes"]:
                    if node["type"] == "gauntlet_result":
                        status = "passed" if result.get("passed") else "failed"
                        self.update_bubble_node_status(node["id"], status, {
                            "score": result.get("score", 0.0),
                            "feedback": result.get("feedback", []),
                            "execution_id": execution_id
                        })
                        self.map_execution_to_bubble(execution_id, node["id"])
                        break
        
        return result
    
    def get_bubble_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all bubble statuses.
        
        Returns:
            Dict with workflow and node status counts
        """
        status_counts = {
            "pending": 0,
            "running": 0,
            "passed": 0,
            "failed": 0,
            "partial": 0
        }
        
        for node_info in self.bubble_nodes.values():
            status = node_info.get("status", "pending")
            if status in status_counts:
                status_counts[status] += 1
        
        return {
            "total_workflows": len(self.bubble_workflows),
            "total_nodes": len(self.bubble_nodes),
            "status_counts": status_counts,
            "bubblelabs_available": BUBBLELABS_AVAILABLE
        }
    
    # =========================================================================
    # ADAPTIVE MDAP INTEGRATION - Complexity-based gauntlet configuration
    # =========================================================================
    
    def create_adaptive_gauntlet(
        self,
        name: str,
        content: str,
        content_type: str = "general",
        base_config: Optional[Dict[str, Any]] = None
    ) -> Optional[GauntletDefinition]:
        """
        Create a gauntlet with adaptive configuration based on content complexity.
        
        Uses Adaptive MDAP to analyze content complexity and configure:
        - Number of rounds
        - Evaluator models
        - Round rules
        
        Args:
            name: Gauntlet name
            content: Content to be evaluated
            content_type: Type of content
            base_config: Base configuration to extend
            
        Returns:
            GauntletDefinition or None if creation fails
        """
        if not ADAPTIVE_MDAP_AVAILABLE:
            logging.warning("Adaptive MDAP not available - using default gauntlet config")
            return None
        
        try:
            # Create sub-problem for complexity analysis
            sp = SubProblem(
                id=f"gauntlet-{name}",
                description=content[:500],  # First 500 chars
                domain=content_type,
                depth=1,
                dependencies=[],
                metadata={"content_length": len(content), "gauntlet_name": name}
            )
            
            # Classify complexity
            from adaptive_mdap import TaskComplexityClassifier
            classifier = TaskComplexityClassifier()
            score = classifier.compute_complexity(sp)
            complexity = score.overall_score
            
            # Configure gauntlet based on complexity
            if complexity <= 0.3:
                # Simple content - minimal gauntlet
                num_rounds = 2
                models = ["gpt-4o-mini"]
            elif complexity <= 0.6:
                # Medium complexity - standard gauntlet
                num_rounds = 3
                models = ["gpt-4o-mini", "gpt-4o"]
            else:
                # High complexity - comprehensive gauntlet
                num_rounds = 4
                models = ["gpt-4o", "claude-3-5-sonnet"]
            
            # Create rounds
            from openevolve_structures import GauntletRoundRule
            rounds = []
            for i in range(num_rounds):
                rounds.append(GauntletRoundRule(
                    round_number=i + 1,
                    models=models,
                    aggregation_method="majority_vote" if complexity > 0.5 else "average"
                ))
            
            # Create gauntlet
            gauntlet = GauntletDefinition(
                name=name,
                rounds=rounds,
                description=f"Adaptive gauntlet for {content_type} content (complexity: {complexity:.3f})",
                generation_mode="multi_candidate_peer_review" if complexity > 0.5 else "single_candidate"
            )
            
            # Store complexity metadata
            gauntlet.metadata = {
                "complexity_score": complexity,
                "adaptive_config": True,
                "num_rounds": num_rounds,
                "models": models
            }
            
            logging.info(
                f"Created adaptive gauntlet '{name}' with complexity {complexity:.3f}, "
                f"{num_rounds} rounds"
            )
            
            return gauntlet
            
        except Exception as e:
            logging.error(f"Failed to create adaptive gauntlet: {e}")
            return None
    
    def get_complexity_for_gauntlet(
        self,
        content: str,
        content_type: str = "general"
    ) -> Optional[float]:
        """
        Get complexity score for gauntlet content.
        
        Args:
            content: Content to analyze
            content_type: Type of content
            
        Returns:
            Complexity score (0.0-1.0) or None
        """
        if not ADAPTIVE_MDAP_AVAILABLE:
            return None
        
        try:
            sp = SubProblem(
                id="gauntlet-complexity-check",
                description=content[:500],
                domain=content_type,
                depth=1,
                dependencies=[],
                metadata={}
            )
            
            from adaptive_mdap import TaskComplexityClassifier
            classifier = TaskComplexityClassifier()
            score = classifier.compute_complexity(sp)
            
            return score.overall_score
            
        except Exception as e:
            logging.warning(f"Failed to compute gauntlet complexity: {e}")
            return None
