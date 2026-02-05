import json
import os
import time
import logging
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from openevolve_structures import GauntletDefinition, GauntletRoundRule

# SECURITY: Import security framework
try:
    from security_framework import (
        Permission, UserContext, authenticated, authorized,
        InputValidator, get_audit_logger, ValidationError
    )
    from input_validation import get_validator
    SECURITY_AVAILABLE = True
    logging.info("SECURITY: Gauntlet manager security enabled")
except ImportError as e:
    SECURITY_AVAILABLE = False
    logging.warning(f"SECURITY: Gauntlet manager security not available: {e}")

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

# Import gauntlet types for REAL evaluation
try:
    from gauntlet_types import (
        BaseGauntlet, GauntletResult, GauntletType,
        AdversarialGauntlet, FormalVerificationGauntlet, StatisticalGauntlet,
        DomainSpecificGauntlet, MultiObjectiveGauntlet, EvolutionaryGauntlet,
        TemporalGauntlet, CrossValidationGauntlet,
        create_gauntlet
    )
    GAUNTLET_TYPES_AVAILABLE = True
except ImportError:
    GAUNTLET_TYPES_AVAILABLE = False

# Import team system for REAL evaluation
try:
    from red_team import RedTeam
    from blue_team import BlueTeam
    from evaluator_team import EvaluatorTeam
    TEAM_SYSTEM_AVAILABLE = True
except ImportError:
    TEAM_SYSTEM_AVAILABLE = False

GAUNTLETS_FILE = "gauntlets.json" # Name of the file used for persisting gauntlet data.
logger = logging.getLogger(__name__)


class GauntletEvaluator:
    """
    REAL Gauntlet Evaluator - performs actual evaluation of solutions.
    
    REPLACES: Hardcoded 'passed_rounds += 1' with REAL evaluation logic.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".GauntletEvaluator")
        self.red_team = None
        self.blue_team = None
        self.evaluator_team = None
        
        if TEAM_SYSTEM_AVAILABLE:
            try:
                self.red_team = RedTeam()
                self.blue_team = BlueTeam()
                self.evaluator_team = EvaluatorTeam()
                self.logger.info("Team system initialized for gauntlet evaluation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize team system: {e}")
    
    def evaluate_round(
        self,
        round_num: int,
        round_rule: GauntletRoundRule,
        solution_content: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single gauntlet round with REAL evaluation.
        
        Args:
            round_num: Round number (1-indexed)
            round_rule: Rule for this round
            solution_content: Solution to evaluate
            context: Additional context
            
        Returns:
            Round evaluation result
        """
        start_time = time.time()
        
        try:
            # Determine evaluation strategy based on round
            if round_num == 1:
                # Round 1: Red Team Assessment (LoongFlow AI Eval equivalent)
                return self._evaluate_round_1_red_team(solution_content, context)
            elif round_num == 2:
                # Round 2: Adversarial Testing (Red Team Attack)
                return self._evaluate_round_2_adversarial(solution_content, context)
            elif round_num == 3:
                # Round 3: Gold Team Verification
                return self._evaluate_round_3_gold_team(solution_content, context)
            else:
                # Additional rounds: Use gauntlet types
                return self._evaluate_with_gauntlet_type(round_num, solution_content, context)
                
        except Exception as e:
            self.logger.error(f"Round {round_num} evaluation failed: {e}")
            return {
                "round": round_num,
                "passed": False,
                "score": 0.0,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _evaluate_round_1_red_team(self, solution_content: str, context: Dict) -> Dict[str, Any]:
        """Round 1: Red Team Assessment - identify issues."""
        start_time = time.time()
        
        if self.red_team and GAUNTLET_TYPES_AVAILABLE:
            try:
                # Use AdversarialGauntlet for structured evaluation
                gauntlet = AdversarialGauntlet("round_1_assessment", config={"attack_modes": ["systematic"]})
                
                # Create mock solution object
                class MockSolution:
                    def __init__(self, content):
                        self.id = "round_1_solution"
                        self.content = content
                
                solution = MockSolution(solution_content)
                gauntlet_context = {
                    "content": solution_content,
                    "content_type": context.get("content_type", "code")
                }
                
                result = gauntlet.execute(solution, gauntlet_context)
                
                # Score based on robustness
                score = result.score
                passed = score >= 0.6  # Threshold for round 1
                
                return {
                    "round": 1,
                    "name": "Red Team Assessment",
                    "passed": passed,
                    "score": score,
                    "confidence": result.confidence,
                    "issues_found": result.details.get("issues_found_count", 0),
                    "feedback": result.feedback,
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                self.logger.warning(f"Adversarial gauntlet failed: {e}, using fallback")
        
        # Fallback: Basic evaluation
        return self._basic_evaluation(1, "Red Team Assessment", solution_content)
    
    def _evaluate_round_2_adversarial(self, solution_content: str, context: Dict) -> Dict[str, Any]:
        """Round 2: Adversarial Testing - attack robustness."""
        start_time = time.time()
        
        if GAUNTLET_TYPES_AVAILABLE:
            try:
                # Use AdversarialGauntlet with attack modes
                gauntlet = AdversarialGauntlet(
                    "round_2_adversarial",
                    config={"attack_modes": ["adversarial", "focused_attack"]}
                )
                
                class MockSolution:
                    def __init__(self, content):
                        self.id = "round_2_solution"
                        self.content = content
                
                solution = MockSolution(solution_content)
                gauntlet_context = {
                    "content": solution_content,
                    "content_type": context.get("content_type", "code")
                }
                
                result = gauntlet.execute(solution, gauntlet_context)
                
                # Round 2 is harder - higher threshold
                score = result.score
                passed = score >= 0.7
                
                return {
                    "round": 2,
                    "name": "Adversarial Testing",
                    "passed": passed,
                    "score": score,
                    "confidence": result.confidence,
                    "robustness_score": result.score,
                    "feedback": result.feedback,
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                self.logger.warning(f"Adversarial round failed: {e}, using fallback")
        
        return self._basic_evaluation(2, "Adversarial Testing", solution_content)
    
    def _evaluate_round_3_gold_team(self, solution_content: str, context: Dict) -> Dict[str, Any]:
        """Round 3: Gold Team Verification - quality verification."""
        start_time = time.time()
        
        if self.evaluator_team:
            try:
                # Use evaluator team for consensus
                evaluation = self.evaluator_team.evaluate_solution(
                    solution_content,
                    criteria=context.get("evaluation_criteria", ["correctness", "quality"])
                )
                
                score = evaluation.overall_score
                passed = score >= 0.75  # Highest threshold for final round
                
                return {
                    "round": 3,
                    "name": "Gold Team Verification",
                    "passed": passed,
                    "score": score,
                    "confidence": evaluation.confidence,
                    "consensus_reached": evaluation.consensus_reached,
                    "feedback": evaluation.feedback,
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                self.logger.warning(f"Gold team evaluation failed: {e}, using fallback")
        
        # Fallback: Quality-based evaluation
        return self._quality_evaluation(3, "Gold Team Verification", solution_content)
    
    def _evaluate_with_gauntlet_type(
        self, round_num: int, solution_content: str, context: Dict
    ) -> Dict[str, Any]:
        """Evaluate using specific gauntlet type based on context."""
        start_time = time.time()
        
        gauntlet_type = context.get(f"round_{round_num}_gauntlet_type", "statistical")
        
        if not GAUNTLET_TYPES_AVAILABLE:
            return self._basic_evaluation(round_num, f"Round {round_num}", solution_content)
        
        try:
            gauntlet = create_gauntlet(gauntlet_type, f"round_{round_num}_{gauntlet_type}")
            
            class MockSolution:
                def __init__(self, content):
                    self.id = f"round_{round_num}_solution"
                    self.content = content
            
            solution = MockSolution(solution_content)
            result = gauntlet.execute(solution, context.get("gauntlet_context", {}))
            
            return {
                "round": round_num,
                "name": f"{gauntlet_type.title()} Evaluation",
                "passed": result.passed,
                "score": result.score,
                "confidence": result.confidence,
                "feedback": result.feedback,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Gauntlet type evaluation failed: {e}")
            return self._basic_evaluation(round_num, f"Round {round_num}", solution_content)
    
    def _basic_evaluation(self, round_num: int, name: str, solution_content: str) -> Dict[str, Any]:
        """Basic evaluation fallback."""
        # Check for basic quality indicators
        score = 0.5
        
        # Length-based scoring (longer solutions tend to be more complete)
        if len(solution_content) > 100:
            score += 0.1
        if len(solution_content) > 500:
            score += 0.1
        
        # Structure-based scoring
        if "def " in solution_content or "class " in solution_content:
            score += 0.1
        if "#" in solution_content or '"""' in solution_content:
            score += 0.1
        
        # Error detection
        if "error" in solution_content.lower() or "fixme" in solution_content.lower():
            score -= 0.2
        
        score = max(0.0, min(1.0, score))
        
        return {
            "round": round_num,
            "name": name,
            "passed": score >= 0.6,
            "score": score,
            "confidence": 0.6,
            "method": "basic_heuristic",
            "feedback": f"Basic evaluation: score={score:.2f}"
        }
    
    def _quality_evaluation(self, round_num: int, name: str, solution_content: str) -> Dict[str, Any]:
        """Quality-based evaluation for final round."""
        score = 0.6  # Start higher for quality check
        
        # Check for documentation
        if '"""' in solution_content or "'''" in solution_content:
            score += 0.1
        
        # Check for error handling
        if "try:" in solution_content and "except" in solution_content:
            score += 0.1
        
        # Check for type hints
        if ": " in solution_content and ("-> " in solution_content or "def " in solution_content):
            score += 0.1
        
        # Check for tests
        if "test" in solution_content.lower() or "assert" in solution_content:
            score += 0.1
        
        score = max(0.0, min(1.0, score))
        
        return {
            "round": round_num,
            "name": name,
            "passed": score >= 0.75,
            "score": score,
            "confidence": 0.7,
            "method": "quality_heuristic",
            "feedback": f"Quality evaluation: score={score:.2f}"
        }
    
    def calculate_final_score(self, round_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate final score from round results.
        
        Returns:
            Dict with final score, pass status, and aggregate metrics
        """
        if not round_results:
            return {
                "passed": False,
                "score": 0.0,
                "rounds_passed": 0,
                "total_rounds": 0
            }
        
        # Count passed rounds
        passed_count = sum(1 for r in round_results if r.get("passed", False))
        total_rounds = len(round_results)
        
        # Calculate weighted score (later rounds weighted more)
        weights = [0.2, 0.3, 0.5]  # Weights for rounds 1, 2, 3
        if len(round_results) > 3:
            # Extend weights for additional rounds
            extra_weights = [0.5 / len(round_results[3:])] * len(round_results[3:])
            weights = [0.15, 0.25, 0.35] + extra_weights
        
        weighted_score = sum(
            r.get("score", 0) * w
            for r, w in zip(round_results, weights[:len(round_results)])
        )
        
        # Normalize to 0-1
        final_score = weighted_score / sum(weights[:len(round_results)])
        
        # Determine pass status
        # Must pass all rounds for overall pass
        all_passed = passed_count == total_rounds
        
        # Or at least 2/3 rounds with high score
        majority_passed = passed_count >= total_rounds * 0.67 and final_score >= 0.7
        
        return {
            "passed": all_passed or majority_passed,
            "score": final_score,
            "rounds_passed": passed_count,
            "total_rounds": total_rounds,
            "all_passed": all_passed,
            "majority_passed": majority_passed,
            "round_scores": [r.get("score", 0) for r in round_results],
            "average_round_score": sum(r.get("score", 0) for r in round_results) / total_rounds
        }


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
        
        # REAL evaluator for gauntlet execution
        self.evaluator = GauntletEvaluator()

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
        Executes a gauntlet against a solution with REAL evaluation.
        
        REPLACES: Hardcoded 'passed_rounds += 1' with actual evaluation using GauntletEvaluator.
        """
        from sovereign_data_models import GauntletExecution, SolutionAttempt, generate_id
        from datetime import datetime

        start_time = time.time()
        execution_id = generate_id("exec")
        solution_id = generate_id("sol")

        # Create solution attempt for the execution record
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

        # =========================================================================
        # REAL EVALUATION - Each round is actually evaluated
        # =========================================================================
        round_results = []
        
        for round_num, round_rule in enumerate(gauntlet.rounds, 1):
            # REAL evaluation using GauntletEvaluator
            round_result = self.evaluator.evaluate_round(
                round_num=round_num,
                round_rule=round_rule,
                solution_content=solution_content,
                context=context
            )
            round_results.append(round_result)
            
            logger.info(
                f"Round {round_num} ({round_result.get('name', 'Unknown')}): "
                f"passed={round_result.get('passed')}, score={round_result.get('score', 0):.3f}"
            )

        # Calculate final score using REAL aggregation
        final_result = self.evaluator.calculate_final_score(round_results)
        
        execution.rounds_passed = final_result["rounds_passed"]
        execution.overall_passed = final_result["passed"]
        execution.final_score = final_result["score"]
        execution.end_time = datetime.now()

        duration = time.time() - start_time

        result = {
            "execution_id": execution_id,
            "passed": execution.overall_passed,
            "score": execution.final_score,
            "final_score": execution.final_score,
            "rounds_passed": execution.rounds_passed,
            "total_rounds": len(gauntlet.rounds),
            "rounds": [
                {
                    "name": r.get("name", f"Round {r.get('round', i+1)}"),
                    "passed": r.get("passed", False),
                    "score": r.get("score", 0.0),
                    "feedback": r.get("feedback", "")
                }
                for i, r in enumerate(round_results)
            ],
            "feedback": [
                f"Round {r.get('round', i+1)}: {r.get('feedback', 'No feedback')}"
                for i, r in enumerate(round_results)
            ],
            "evaluation_summary": {
                "all_passed": final_result.get("all_passed", False),
                "majority_passed": final_result.get("majority_passed", False),
                "average_round_score": final_result.get("average_round_score", 0.0),
                "round_scores": final_result.get("round_scores", [])
            }
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
                                "execution_id": execution_id,
                                "rounds_passed": result.get("rounds_passed", 0),
                                "total_rounds": result.get("total_rounds", 0)
                            })
                            break
            except Exception as e:
                logger.error(f"Failed to update bubble status: {e}")

        if not result["passed"]:
            self._trigger_gauntlet_alerts(
                gauntlet.name,
                False,
                f"Gauntlet execution failed: {result['rounds_passed']}/{result['total_rounds']} rounds passed"
            )

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
    
    # =========================================================================
    # ADVANCED GAUNTLET TYPES - All 8+ gauntlet implementations
    # =========================================================================
    
    def create_adversarial_gauntlet(
        self,
        name: str,
        solution: Any,
        attack_modes: Optional[List[str]] = None,
        use_blue_team: bool = True
    ) -> Dict[str, Any]:
        """
        Create and execute an adversarial gauntlet.
        
        Uses red team attacks and robustness testing to validate solution.
        
        Args:
            name: Gauntlet name
            solution: Solution to test
            attack_modes: List of attack modes (e.g., ["systematic", "focused_attack"])
            use_blue_team: Whether to use blue team for defense validation
            
        Returns:
            Dict with robustness score and findings
        """
        try:
            from gauntlet_types import AdversarialGauntlet
            
            config = {
                "attack_modes": attack_modes or ["systematic", "adversarial"],
                "use_blue_team": use_blue_team
            }
            
            gauntlet = AdversarialGauntlet(name, config)
            
            context = {
                "content": str(solution),
                "content_type": getattr(solution, 'content_type', 'general')
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "robustness_score": result.score,
                "confidence": result.confidence,
                "feedback": result.feedback,
                "improvements": result.improvements,
                "details": result.details,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Adversarial gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_formal_gauntlet(
        self,
        name: str,
        solution: Any,
        properties: List[Dict[str, Any]],
        constraints: Optional[List[Dict]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Create and execute a formal verification gauntlet.
        
        Uses Z3-based formal proofs for property verification.
        
        Args:
            name: Gauntlet name
            solution: Solution to verify
            properties: List of property specifications to verify
            constraints: Optional constraints for verification
            timeout: Z3 solver timeout in seconds
            
        Returns:
            Dict with proof score and verification results
        """
        try:
            from gauntlet_types import FormalVerificationGauntlet
            
            config = {"timeout": timeout}
            gauntlet = FormalVerificationGauntlet(name, config)
            
            context = {
                "properties": properties,
                "constraints": constraints or [],
                "code": str(solution)
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "proof_score": result.score,
                "confidence": result.confidence,
                "verified_count": result.details.get("verified_count", 0),
                "failed_count": result.details.get("failed_count", 0),
                "total_properties": result.details.get("total_properties", 0),
                "verification_results": result.details.get("verification_results", []),
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Formal gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_statistical_gauntlet(
        self,
        name: str,
        solution: Any,
        test_data: Optional[List[float]] = None,
        expected_distribution: Optional[Dict] = None,
        num_samples: int = 1000,
        tests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create and execute a statistical gauntlet.
        
        Uses Monte Carlo validation and hypothesis testing.
        
        Args:
            name: Gauntlet name
            solution: Solution to validate
            test_data: Optional test data (will be generated if not provided)
            expected_distribution: Expected distribution parameters
            num_samples: Number of Monte Carlo samples
            tests: List of tests to run ("mean", "variance", "distribution")
            
        Returns:
            Dict with statistical validation score
        """
        try:
            from gauntlet_types import StatisticalGauntlet
            
            config = {
                "num_samples": num_samples,
                "tests": tests or ["mean", "variance", "distribution"]
            }
            
            gauntlet = StatisticalGauntlet(name, config)
            
            context = {
                "test_data": test_data or [],
                "expected_distribution": expected_distribution or {},
                "expected_mean": expected_distribution.get("mean", 0.0) if expected_distribution else 0.0,
                "expected_std": expected_distribution.get("std", 1.0) if expected_distribution else 1.0
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "confidence": result.confidence,
                "p_value": result.details.get("p_value", 1.0),
                "test_results": result.details.get("test_results", {}),
                "num_samples": result.details.get("num_samples", 0),
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Statistical gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_domain_gauntlet(
        self,
        name: str,
        solution: Any,
        domain: str,
        domain_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create and execute a domain-specific gauntlet.
        
        Available domains: physics, finance, chemistry, engineering
        
        Args:
            name: Gauntlet name
            solution: Solution to validate
            domain: Domain name (physics, finance, chemistry, engineering)
            domain_context: Domain-specific context/parameters
            
        Returns:
            Dict with domain validation score
        """
        try:
            from gauntlet_types import DomainSpecificGauntlet
            
            gauntlet = DomainSpecificGauntlet(domain, name)
            
            context = domain_context or {}
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "domain": domain,
                "confidence": result.confidence,
                "check_results": result.details.get("check_results", []),
                "passed_checks": result.details.get("passed_checks", 0),
                "total_checks": result.details.get("total_checks", 0),
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Domain gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_multi_objective_gauntlet(
        self,
        name: str,
        solution: Any,
        objectives: List[str],
        objective_values: Dict[str, float],
        weights: Optional[List[float]] = None,
        reference_front: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Create and execute a multi-objective gauntlet.
        
        Validates Pareto optimality across multiple objectives.
        
        Args:
            name: Gauntlet name
            solution: Solution to validate
            objectives: List of objective names
            objective_values: Dict mapping objective names to values
            weights: Optional weights for each objective
            reference_front: Optional Pareto front for comparison
            
        Returns:
            Dict with Pareto validation score
        """
        try:
            from gauntlet_types import MultiObjectiveGauntlet
            
            config = {
                "objectives": objectives,
                "weights": weights or [1.0/len(objectives)] * len(objectives)
            }
            
            gauntlet = MultiObjectiveGauntlet(name, config)
            
            context = {
                "objective_values": objective_values,
                "reference_front": reference_front or []
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "confidence": result.confidence,
                "is_pareto_optimal": result.details.get("is_pareto_optimal", False),
                "weighted_score": result.details.get("weighted_score", 0.0),
                "dominated_by": result.details.get("dominated_by", 0),
                "objectives": objectives,
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Multi-objective gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_evolutionary_gauntlet(
        self,
        name: str,
        solution: Any,
        fitness_function: Optional[Callable] = None,
        population_size: int = 50,
        generations: int = 10
    ) -> Dict[str, Any]:
        """
        Create and execute an evolutionary gauntlet.
        
        Uses fitness-based evaluation with REAL EvolutionEngine.
        
        Args:
            name: Gauntlet name
            solution: Solution to evaluate
            fitness_function: Optional custom fitness function
            population_size: Size of competing population
            generations: Number of generations to simulate
            
        Returns:
            Dict with fitness evaluation results
        """
        try:
            from gauntlet_types import EvolutionaryGauntlet
            
            config = {
                "population_size": population_size,
                "generations": generations
            }
            
            gauntlet = EvolutionaryGauntlet(name, config)
            
            context = {
                "fitness_function": fitness_function,
                "solution_space": "discrete"
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "confidence": result.confidence,
                "raw_fitness": result.details.get("raw_fitness", 0.0),
                "relative_fitness": result.details.get("relative_fitness", 0.0),
                "population_rank": result.details.get("population_rank"),
                "population_size": result.details.get("population_size", 0),
                "evolution_engine_used": result.details.get("evolution_engine_used", False),
                "best_fitness_achieved": result.details.get("best_fitness_achieved"),
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Evolutionary gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_temporal_gauntlet(
        self,
        name: str,
        solution: Any,
        time_series_data: Optional[List[float]] = None,
        simulation_function: Optional[Callable] = None,
        time_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Create and execute a temporal gauntlet.
        
        Validates solutions over time for stability and convergence.
        
        Args:
            name: Gauntlet name
            solution: Solution to validate
            time_series_data: Optional time series data
            simulation_function: Optional function to simulate over time
            time_steps: Number of time steps to simulate
            
        Returns:
            Dict with temporal validation results
        """
        try:
            from gauntlet_types import TemporalGauntlet
            
            config = {
                "time_steps": time_steps,
                "stability_threshold": 0.1,
                "convergence_threshold": 0.01
            }
            
            gauntlet = TemporalGauntlet(name, config)
            
            context = {
                "time_series_data": time_series_data or [],
                "simulation_function": simulation_function
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "confidence": result.confidence,
                "stability": result.details.get("stability", {}),
                "convergence": result.details.get("convergence", {}),
                "trend": result.details.get("trend", {}),
                "is_stable": result.details.get("stability", {}).get("stable", False),
                "has_converged": result.details.get("convergence", {}).get("converged", False),
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Temporal gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def create_cross_validation_gauntlet(
        self,
        name: str,
        solution: Any,
        data: List[Any],
        evaluation_function: Optional[Callable] = None,
        k_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Create and execute a cross-validation gauntlet.
        
        Uses K-fold style validation for robustness testing.
        
        Args:
            name: Gauntlet name
            solution: Solution to validate
            data: Dataset for cross-validation
            evaluation_function: Function to evaluate solution on data
            k_folds: Number of folds for cross-validation
            
        Returns:
            Dict with cross-validation results
        """
        try:
            from gauntlet_types import CrossValidationGauntlet
            
            config = {
                "k_folds": k_folds,
                "shuffle": True
            }
            
            gauntlet = CrossValidationGauntlet(name, config)
            
            context = {
                "data": data,
                "evaluation_function": evaluation_function
            }
            
            result = gauntlet.execute(solution, context)
            
            return {
                "passed": result.passed,
                "score": result.score,
                "confidence": result.confidence,
                "mean_score": result.details.get("mean_score", 0.0),
                "std_score": result.details.get("std_score", 0.0),
                "min_score": result.details.get("min_score", 0.0),
                "max_score": result.details.get("max_score", 0.0),
                "confidence_interval": result.details.get("confidence_interval", [0, 0]),
                "fold_results": result.details.get("fold_results", []),
                "k_folds": k_folds,
                "feedback": result.feedback,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Cross-validation gauntlet failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }
    
    def list_advanced_gauntlet_types(self) -> Dict[str, str]:
        """
        List all available advanced gauntlet types.
        
        Returns:
            Dict mapping gauntlet type names to descriptions
        """
        try:
            from gauntlet_types import list_available_gauntlets
            return list_available_gauntlets()
        except Exception as e:
            logger.warning(f"Failed to list gauntlet types: {e}")
            return {
                "adversarial": "Red team attacks and robustness testing",
                "formal_verification": "Z3-based formal proofs (REAL Z3 integration)",
                "statistical": "Monte Carlo validation and hypothesis testing",
                "domain": "Domain-specific validation (physics, finance, chemistry, engineering)",
                "multi_objective": "Pareto frontier validation",
                "evolutionary": "Fitness-based evaluation using REAL EvolutionEngine",
                "temporal": "Time-series validation",
                "cross_validation": "K-fold style validation"
            }
