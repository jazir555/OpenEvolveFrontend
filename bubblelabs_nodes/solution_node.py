"""
Solution Node for BubbleLabs Integration

Implements multi-strategy solution generation using MAKER, MCTS, Evolutionary, and Hybrid approaches.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class SolutionNode(BubbleLabsNode):
    """
    Generates solutions using various AI strategies.

    Supports multiple solution strategies:
    - maker: MAKER v2 system
    - mcts: Monte Carlo Tree Search
    - evolutionary: Evolutionary algorithms
    - hybrid: Combination of multiple strategies
    """

    # Node metadata
    DISPLAY_NAME = "Solution Generation"
    DESCRIPTION = (
        "Generate solutions using MAKER, MCTS, Evolutionary, "
        "or Hybrid strategies."
    )
    ICON = "solution"
    CATEGORY = "generation"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import solution orchestrator (safe import)
        SolutionOrchestrator = self.safe_import(
            'solution_orchestration.SolutionOrchestrator',
            fallback_value=None,
            error_msg="SolutionOrchestrator not available for SolutionNode"
        )

        if SolutionOrchestrator:
            try:
                self.orchestrator = SolutionOrchestrator()
            except Exception as e:
                self.logger.warning(f"Could not instantiate SolutionOrchestrator: {e}")
                self.orchestrator = None
        else:
            self.orchestrator = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - problem: str or SubProblem object

        Optional:
            - strategy: str (maker, mcts, evolutionary, hybrid)
            - model: str
            - iterations: int
            - quality_threshold: float
            - context: Dict
        """
        errors = []

        # Check required fields
        if 'problem' not in inputs:
            errors.append("Missing required field: problem")
        elif not isinstance(inputs['problem'], (str, dict, object)):
            errors.append("problem must be a string, dictionary, or SubProblem object")

        # Validate strategy
        if 'strategy' in inputs:
            valid_strategies = ['maker', 'mcts', 'evolutionary', 'hybrid']
            if inputs['strategy'] not in valid_strategies:
                errors.append(f"strategy must be one of: {', '.join(valid_strategies)}")

        # Validate model
        if 'model' in inputs:
            if not isinstance(inputs['model'], str):
                errors.append("model must be a string")

        # Validate iterations
        if 'iterations' in inputs:
            if not isinstance(inputs['iterations'], int):
                errors.append("iterations must be an integer")
            elif inputs['iterations'] < 1:
                errors.append("iterations must be at least 1")
            elif inputs['iterations'] > 10000:
                errors.append("iterations cannot exceed 10000")

        # Validate quality_threshold
        if 'quality_threshold' in inputs:
            if not isinstance(inputs['quality_threshold'], (int, float)):
                errors.append("quality_threshold must be a number")
            elif inputs['quality_threshold'] < 0 or inputs['quality_threshold'] > 1:
                errors.append("quality_threshold must be between 0 and 1")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Generate a solution using the specified strategy.

        Args:
            inputs: Must contain 'problem' and optional strategy parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - solution: The generated solution
                - confidence: Solution confidence score (0-1)
                - quality_score: Quality assessment (0-1)
                - generation_method: Strategy used
                - iterations_used: Number of iterations performed
                - alternative_solutions: List of alternative solutions
        """
        if not self.orchestrator:
            return self._generate_solution_simple(inputs, context)

        problem = inputs['problem']
        strategy = inputs.get('strategy', self.config.get('strategy', 'hybrid'))
        model = inputs.get('model', self.config.get('model', 'gpt-4o'))
        iterations = inputs.get('iterations', self.config.get('iterations', 100))
        quality_threshold = inputs.get('quality_threshold', self.config.get('quality_threshold', 0.8))
        problem_context = inputs.get('context', {})

        # Update progress
        context.update_progress(10, f"Initializing {strategy.upper()} solution generator")
        self.logger.info(f"Generating solution using {strategy} strategy with {model}")

        try:
            # Normalize problem to string
            if hasattr(problem, 'title') and hasattr(problem, 'description'):
                problem_text = f"{problem.title}: {problem.description}"
            elif isinstance(problem, dict):
                problem_text = problem.get('description', problem.get('title', str(problem)))
            else:
                problem_text = str(problem)

            # Generate solution
            context.update_progress(20, "Analyzing problem structure")

            solution_result = self.orchestrator.generate(
                problem=problem_text,
                strategy=strategy,
                model=model,
                iterations=iterations,
                quality_threshold=quality_threshold,
                context=problem_context,
                callback=lambda p, m: context.update_progress(20 + p * 0.7, m)
            )

            # Update progress
            context.update_progress(90, "Processing generated solution")

            # Extract and format results
            result = {
                'solution': solution_result.solution,
                'confidence': solution_result.confidence,
                'quality_score': solution_result.quality_score,
                'generation_method': strategy,
                'iterations_used': solution_result.iterations_performed,
                'alternative_solutions': solution_result.alternatives[:5],  # Top 5 alternatives
                'metadata': {
                    'model': model,
                    'quality_threshold': quality_threshold,
                    'generation_time': solution_result.generation_time,
                    'strategy_params': solution_result.strategy_parameters,
                    'convergence_info': solution_result.convergence_info
                },
                'problem_hash': hash(problem_text)
            }

            # Add artifacts to context
            context.add_artifact('solution_generation', {
                'result': result,
                'problem': problem_text,
                'strategy': strategy
            })

            # Check if quality threshold was met
            if result['quality_score'] < quality_threshold:
                self.logger.warning(
                    f"Solution quality ({result['quality_score']:.2f}) below threshold ({quality_threshold:.2f})"
                )

            context.update_progress(
                100,
                f"Solution generation complete: quality={result['quality_score']:.2f}, "
                f"confidence={result['confidence']:.2f}, "
                f"iterations={result['iterations_used']}"
            )

            self.logger.info(
                f"Solution generated using {strategy}: quality={result['quality_score']:.2f}, "
                f"confidence={result['confidence']:.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Solution generation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Solution generation failed: {str(e)}",
                details={
                    'problem': str(problem)[:100],  # First 100 chars
                    'strategy': strategy,
                    'model': model,
                    'iterations': iterations,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _generate_solution_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple solution generation fallback when orchestrator not available"""
        problem = inputs['problem']
        strategy = inputs.get('strategy', self.config.get('strategy', 'hybrid'))

        context.update_progress(10, "Using simple solution generator (orchestrator not available)")

        import time
        start_time = time.time()

        # Normalize problem to string
        if hasattr(problem, 'title'):
            problem_text = f"{problem.title}: {problem.description}"
        elif isinstance(problem, dict):
            problem_text = problem.get('description', str(problem))
        else:
            problem_text = str(problem)

        context.update_progress(30, "Analyzing problem")

        # Generate simple solution structure
        solution = {
            'problem': problem_text,
            'approach': 'simple_fallback',
            'strategy': strategy,
            'description': f"Solution for: {problem_text}",
            'steps': [
                "Analyze problem requirements",
                "Design solution approach",
                "Implement solution",
                "Verify and validate"
            ],
            'note': 'Full orchestrator not available, using simple solution structure'
        }

        generation_time = time.time() - start_time

        result = {
            'solution': solution,
            'confidence': 0.5,
            'quality_score': 0.5,
            'generation_method': f"{strategy}_simple",
            'iterations_used': 1,
            'alternative_solutions': [],
            'metadata': {
                'generation_time': generation_time,
                'warning': 'Full orchestrator not available'
            },
            'problem_hash': hash(problem_text)
        }

        context.update_progress(100, f"Simple solution generation complete in {generation_time:.2f}s")
        return result

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Solution Generation Configuration",
            "description": "Configure solution generation strategy and parameters",
            "properties": {
                "strategy": {
                    "type": "string",
                    "title": "Generation Strategy",
                    "description": "AI strategy to use for solution generation",
                    "enum": ["maker", "mcts", "evolutionary", "hybrid"],
                    "enumNames": [
                        "MAKER v2 (Multi-Agent)",
                        "MCTS (Monte Carlo Tree Search)",
                        "Evolutionary Algorithm",
                        "Hybrid (Best of All)"
                    ],
                    "default": "hybrid"
                },
                "model": {
                    "type": "string",
                    "title": "Language Model",
                    "description": "LLM to use for solution generation",
                    "enum": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "claude-3-haiku"],
                    "default": "gpt-4o"
                },
                "iterations": {
                    "type": "integer",
                    "title": "Iterations",
                    "description": "Number of iterations/generations to perform",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 100
                },
                "quality_threshold": {
                    "type": "number",
                    "title": "Quality Threshold",
                    "description": "Minimum quality threshold for accepting solutions (0-1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8
                },
                "enable_caching": {
                    "type": "boolean",
                    "title": "Enable Solution Caching",
                    "description": "Cache and reuse similar solutions",
                    "default": True
                },
                "diversity_factor": {
                    "type": "number",
                    "title": "Diversity Factor",
                    "description": "Encourage solution diversity (0-1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3
                }
            },
            "required": ["strategy", "model"]
        }
