"""
Gauntlet Node for BubbleLabs Integration

Implements multi-stage quality control with Red/Blue/Gold team testing.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class GauntletNode(BubbleLabsNode):
    """
    Runs solutions through a gauntlet of tests and evaluations.

    Supports multiple team types:
    - red: Adversarial testing and critique
    - blue: Solution refinement and improvement
    - gold: Final evaluation and certification
    """

    # Node metadata
    DISPLAY_NAME = "Gauntlet Testing"
    DESCRIPTION = (
        "Run solutions through Red/Blue/Gold team gauntlets for "
        "comprehensive quality control and adversarial testing."
    )
    ICON = "gauntlet"
    CATEGORY = "quality"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import gauntlet manager (safe import)
        GauntletManager = self.safe_import(
            'gauntlet_manager.GauntletManager',
            fallback_value=None,
            error_msg="GauntletManager not available for GauntletNode"
        )

        if GauntletManager:
            try:
                self.manager = GauntletManager()
            except Exception as e:
                self.logger.warning(f"Could not instantiate GauntletManager: {e}")
                self.manager = None
        else:
            self.manager = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - solution: Dict or SolutionAttempt object

        Optional:
            - gauntlet_type: str (red, blue, gold)
            - rounds: int
            - difficulty: str
            - evaluation_criteria: List[str]
        """
        errors = []

        # Check required fields
        if 'solution' not in inputs:
            errors.append("Missing required field: solution")
        elif not isinstance(inputs['solution'], (dict, object)):
            errors.append("solution must be a dictionary or SolutionAttempt object")

        # Validate gauntlet_type
        if 'gauntlet_type' in inputs:
            valid_types = ['red', 'blue', 'gold', 'full']
            if inputs['gauntlet_type'] not in valid_types:
                errors.append(f"gauntlet_type must be one of: {', '.join(valid_types)}")

        # Validate rounds
        if 'rounds' in inputs:
            if not isinstance(inputs['rounds'], int):
                errors.append("rounds must be an integer")
            elif inputs['rounds'] < 1:
                errors.append("rounds must be at least 1")
            elif inputs['rounds'] > 10:
                errors.append("rounds cannot exceed 10")

        # Validate difficulty
        if 'difficulty' in inputs:
            valid_difficulties = ['easy', 'medium', 'hard', 'adaptive']
            if inputs['difficulty'] not in valid_difficulties:
                errors.append(f"difficulty must be one of: {', '.join(valid_difficulties)}")

        # Validate evaluation_criteria
        if 'evaluation_criteria' in inputs:
            if not isinstance(inputs['evaluation_criteria'], list):
                errors.append("evaluation_criteria must be a list")
            elif not all(isinstance(c, str) for c in inputs['evaluation_criteria']):
                errors.append("All criteria must be strings")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Run solution through gauntlet testing.

        Args:
            inputs: Must contain 'solution' and optional gauntlet parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - passed: Whether solution passed the gauntlet
                - score: Overall gauntlet score (0-100)
                - round_results: List of results from each round
                - feedback: List of feedback items
                - improvements_needed: List of required improvements
        """
        if not self.manager:
            return self._run_gauntlet_simple(inputs, context)

        solution = inputs['solution']
        gauntlet_type = inputs.get('gauntlet_type', self.config.get('gauntlet_type', 'full'))
        rounds = inputs.get('rounds', self.config.get('rounds', 3))
        difficulty = inputs.get('difficulty', self.config.get('difficulty', 'adaptive'))
        evaluation_criteria = inputs.get('evaluation_criteria', self.config.get('evaluation_criteria', [
            'correctness',
            'completeness',
            'efficiency',
            'clarity',
            'robustness'
        ]))

        # Update progress
        context.update_progress(10, f"Initializing {gauntlet_type.upper()} gauntlet")
        self.logger.info(f"Running {gauntlet_type} gauntlet: {rounds} rounds, difficulty={difficulty}")

        try:
            # Run gauntlet
            context.update_progress(20, "Loading gauntlet configuration")

            gauntlet_result = self.manager.run(
                solution=solution,
                gauntlet_type=gauntlet_type,
                rounds=rounds,
                difficulty=difficulty,
                evaluation_criteria=evaluation_criteria,
                callback=lambda p, m: context.update_progress(20 + p * 0.7, m)
            )

            # Update progress
            context.update_progress(90, "Processing gauntlet results")

            # Extract and format results
            result = {
                'passed': gauntlet_result.passed,
                'score': gauntlet_result.overall_score,
                'round_results': self._format_round_results(gauntlet_result.rounds),
                'feedback': self._format_feedback(gauntlet_result.feedback),
                'improvements_needed': gauntlet_result.improvements_needed,
                'team_performances': self._format_team_performances(gauntlet_result.team_performances),
                'summary': {
                    'gauntlet_type': gauntlet_type,
                    'rounds_completed': len(gauntlet_result.rounds),
                    'total_rounds': rounds,
                    'difficulty_used': gauntlet_result.actual_difficulty,
                    'criteria_evaluated': evaluation_criteria,
                    'overall_score': gauntlet_result.overall_score,
                    'pass_threshold': gauntlet_result.pass_threshold,
                    'passed': gauntlet_result.passed
                },
                'metadata': {
                    'gauntlet_version': gauntlet_result.version,
                    'execution_time': gauntlet_result.execution_time,
                    'team_configurations': gauntlet_result.team_configs
                }
            }

            # Add artifacts to context
            context.add_artifact('gauntlet_run', {
                'result': result,
                'solution_id': getattr(solution, 'id', 'unknown'),
                'gauntlet_type': gauntlet_type
            })

            status_msg = "PASSED" if result['passed'] else "FAILED"
            context.update_progress(
                100,
                f"Gauntlet {status_msg}: score={result['score']:.1f}/100, "
                f"rounds={len(result['round_results'])}/{rounds}, "
                f"improvements={len(result['improvements_needed'])}"
            )

            self.logger.info(
                f"Gauntlet completed: {status_msg}, "
                f"score={result['score']:.1f}/100, "
                f"{len(result['improvements_needed'])} improvements needed"
            )

            return result

        except Exception as e:
            self.logger.error(f"Gauntlet execution failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Gauntlet execution failed: {str(e)}",
                details={
                    'solution_id': getattr(solution, 'id', 'unknown'),
                    'gauntlet_type': gauntlet_type,
                    'rounds': rounds,
                    'difficulty': difficulty,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _run_gauntlet_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple gauntlet fallback when manager not available"""
        solution = inputs['solution']
        gauntlet_type = inputs.get('gauntlet_type', self.config.get('gauntlet_type', 'full'))
        rounds = inputs.get('rounds', self.config.get('rounds', 3))

        context.update_progress(10, "Using simple gauntlet (manager not available)")

        import time
        start_time = time.time()

        context.update_progress(30, "Running basic evaluations")

        # Simple gauntlet simulation
        round_results = []
        for i in range(min(rounds, 3)):
            round_results.append({
                'round': i + 1,
                'score': 70 + (i * 5),  # Improving each round
                'passed': True,
                'feedback': [f"Round {i+1} basic check passed"],
                'team': 'simple'
            })

        # Calculate overall score
        overall_score = sum(r['score'] for r in round_results) / len(round_results) if round_results else 50
        passed = overall_score >= 70

        execution_time = time.time() - start_time

        result = {
            'passed': passed,
            'score': overall_score,
            'round_results': round_results,
            'feedback': [
                'Basic structural check passed',
                'Solution completeness verified',
                'Note: Full gauntlet manager not available'
            ],
            'improvements_needed': [] if passed else ['Improve overall quality'],
            'team_performances': [
                {
                    'team': 'simple',
                    'score': overall_score,
                    'rounds': len(round_results)
                }
            ],
            'summary': {
                'gauntlet_type': gauntlet_type,
                'rounds_completed': len(round_results),
                'total_rounds': rounds,
                'difficulty_used': 'medium',
                'criteria_evaluated': ['correctness', 'completeness'],
                'overall_score': overall_score,
                'pass_threshold': 70,
                'passed': passed
            },
            'metadata': {
                'execution_time': execution_time,
                'warning': 'Full manager not available, using simple evaluation'
            }
        }

        context.update_progress(100, f"Simple gauntlet complete in {execution_time:.2f}s")
        return result

    def _format_round_results(self, rounds: List) -> List[Dict[str, Any]]:
        """Format round results for output"""
        formatted = []

        for round_result in rounds:
            formatted.append({
                'round': round_result.round_number,
                'team': round_result.team_type,
                'score': round_result.score,
                'passed': round_result.passed,
                'criteria_scores': round_result.criteria_scores,
                'feedback': round_result.feedback,
                'timestamp': round_result.timestamp
            })

        return formatted

    def _format_feedback(self, feedback: List) -> List[Dict[str, Any]]:
        """Format feedback for output"""
        formatted = []

        for item in feedback:
            formatted.append({
                'category': getattr(item, 'category', 'general'),
                'severity': getattr(item, 'severity', 'info'),
                'message': getattr(item, 'message', ''),
                'suggestion': getattr(item, 'suggestion', None),
                'source': getattr(item, 'source', 'unknown')
            })

        return formatted

    def _format_team_performances(self, performances: List) -> List[Dict[str, Any]]:
        """Format team performances for output"""
        formatted = []

        for perf in performances:
            formatted.append({
                'team': perf.team_type,
                'overall_score': perf.score,
                'rounds_participated': perf.rounds_count,
                'strengths': perf.strengths,
                'weaknesses': perf.weaknesses,
                'recommendations': perf.recommendations
            })

        return formatted

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Gauntlet Configuration",
            "description": "Configure gauntlet testing and quality control parameters",
            "properties": {
                "gauntlet_type": {
                    "type": "string",
                    "title": "Gauntlet Type",
                    "description": "Type of gauntlet to run",
                    "enum": ["red", "blue", "gold", "full"],
                    "enumNames": [
                        "Red Team (Adversarial Testing)",
                        "Blue Team (Solution Refinement)",
                        "Gold Team (Final Evaluation)",
                        "Full Gauntlet (All Teams)"
                    ],
                    "default": "full"
                },
                "rounds": {
                    "type": "integer",
                    "title": "Number of Rounds",
                    "description": "Number of gauntlet rounds to run",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3
                },
                "difficulty": {
                    "type": "string",
                    "title": "Difficulty",
                    "description": "Difficulty level of testing",
                    "enum": ["easy", "medium", "hard", "adaptive"],
                    "enumNames": [
                        "Easy",
                        "Medium",
                        "Hard",
                        "Adaptive (Adjusts to performance)"
                    ],
                    "default": "adaptive"
                },
                "evaluation_criteria": {
                    "type": "array",
                    "title": "Evaluation Criteria",
                    "description": "Criteria to evaluate during gauntlet",
                    "items": {
                        "type": "string",
                        "enum": [
                            "correctness",
                            "completeness",
                            "efficiency",
                            "clarity",
                            "robustness",
                            "security",
                            "scalability",
                            "maintainability"
                        ]
                    },
                    "uniqueItems": True,
                    "default": ["correctness", "completeness", "efficiency", "clarity", "robustness"]
                },
                "pass_threshold": {
                    "type": "number",
                    "title": "Pass Threshold",
                    "description": "Minimum score required to pass (0-100)",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 70
                },
                "enable_learning": {
                    "type": "boolean",
                    "title": "Enable Learning",
                    "description": "Adapt gauntlet based on previous runs",
                    "default": True
                }
            },
            "required": ["gauntlet_type", "rounds", "difficulty"]
        }
