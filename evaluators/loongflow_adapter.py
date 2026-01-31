"""
LoongFlow Evaluator Adapter for OpenEvolve Gauntlets
Wraps LoongFlow's AI evaluation as a gauntlet round for quick quality screening.

This adapter enables using LoongFlow's sophisticated evaluation as Round 1
of gauntlets, providing fast AI-based quality assessment before more expensive
red team and gold team evaluations.
"""

import asyncio
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GauntletRoundResult:
    """
    Result from a single gauntlet round evaluation.

    Attributes:
        rule_id: ID of the gauntlet round rule
        passed: Whether the solution passed this round
        score: Score achieved (0.0-1.0+)
        feedback: Human-readable feedback
        details: Additional evaluation details
        execution_time: Time taken for evaluation in seconds
        timestamp: When the evaluation was performed
    """
    rule_id: str
    passed: bool
    score: float
    feedback: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class LoongFlowEvaluatorAdapter:
    """
    Adapts LoongFlow's evaluation to work as a gauntlet round.

    This enables quick AI-based evaluation as Round 1 of gauntlets, providing:
    - Fast quality screening (typically 10-30 seconds)
    - Consistent scoring (0-1+ scale)
    - Structured feedback for improvement
    - Low computational cost compared to full red/gold team evaluation

    The adapter handles two modes:
    1. AI Agent Mode: Uses LoongFlow's AI agent for evaluation
    2. Fallback Mode: Simple keyword-based evaluation if LoongFlow unavailable

    Example:
        ```python
        adapter = LoongFlowEvaluatorAdapter(
            llm_config={
                'model': 'claude-3-5-sonnet-20241022',
                'api_key': 'sk-...',
                'url': 'http://localhost:8001'
            },
            timeout=60
        )

        result = await adapter.evaluate_round(
            solution=solution_attempt,
            round_rule=gauntlet_round,
            context={'problem': 'Solve X', 'criteria': ['correctness', 'clarity']}
        )
        ```
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        timeout: int = 60,
        enable_loongflow: bool = True
    ):
        """
        Initialize the LoongFlow evaluator adapter.

        Args:
            llm_config: Configuration for LLM (model, api_key, url, etc.)
            timeout: Maximum time for evaluation in seconds
            enable_loongflow: If False, use fallback mode even if LoongFlow available
        """
        self.llm_config = llm_config
        self.timeout = timeout
        self.enable_loongflow = enable_loongflow
        self.evaluator = None
        self.use_fallback = False

        self._initialize_evaluator()

    def _initialize_evaluator(self):
        """Initialize LoongFlow evaluator with fallback handling."""
        if not self.enable_loongflow:
            logger.info("LoongFlow disabled, using fallback evaluator")
            self.use_fallback = True
            return

        try:
            # Add LoongFlow to path if needed
            loongflow_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'LoongFlow'
            )
            if os.path.exists(loongflow_path) and loongflow_path not in sys.path:
                sys.path.insert(0, loongflow_path)

            # Import LoongFlow components
            from loongflow.agents.general_agent.evaluator import GeneralEvaluator
            from loongflow.framework.pes.context import EvaluatorConfig, LLMConfig

            # Create LLM config
            llm_cfg = LLMConfig(
                model=self.llm_config.get('model', 'claude-3-5-sonnet-20241022'),
                api_key=self.llm_config.get('api_key', ''),
                url=self.llm_config.get('url', 'http://localhost:8001'),
                temperature=self.llm_config.get('temperature', 0.3),
                max_tokens=self.llm_config.get('max_tokens', 4096)
            )

            # Create evaluator config
            evaluator_cfg = EvaluatorConfig(
                llm_config=llm_cfg,
                timeout=self.timeout,
                agent=self.llm_config.get('agent_config', {})
            )

            # Initialize evaluator
            self.evaluator = GeneralEvaluator(config=evaluator_cfg)
            logger.info("LoongFlow evaluator initialized successfully")

        except ImportError as e:
            logger.warning(f"LoongFlow not available: {e}. Using fallback evaluator.")
            self.use_fallback = True
        except Exception as e:
            logger.error(f"Failed to initialize LoongFlow: {e}. Using fallback evaluator.")
            self.use_fallback = True

    async def evaluate_round(
        self,
        solution: Any,
        round_rule: Any,
        context: Dict[str, Any]
    ) -> GauntletRoundResult:
        """
        Evaluate solution using LoongFlow's AI evaluator.

        Args:
            solution: SolutionAttempt object or dict with solution content
            round_rule: GauntletRoundRule with evaluation criteria
            context: Additional context (problem, constraints, criteria, etc.)

        Returns:
            GauntletRoundResult with score, feedback, passed/failed
        """
        start_time = time.time()

        # Extract rule_id from per_judge_requirements or use default
        rule_id = "unknown"
        if hasattr(round_rule, 'per_judge_requirements'):
            rule_id = round_rule.per_judge_requirements.get("rule_id", "unknown")
        elif hasattr(round_rule, 'rule_id'):
            rule_id = round_rule.rule_id

        # Extract solution content
        solution_content = self._extract_solution_content(solution)

        try:
            if self.use_fallback or not self.evaluator:
                result = await self._evaluate_with_fallback(
                    solution_content, round_rule, context, rule_id
                )
            else:
                result = await self._evaluate_with_loongflow(
                    solution_content, round_rule, context, rule_id
                )

            # Add execution time
            result.execution_time = time.time() - start_time
            result.timestamp = time.time()

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return GauntletRoundResult(
                rule_id=rule_id,
                passed=False,
                score=0.0,
                feedback=f"Evaluation failed: {str(e)}",
                details={
                    "error": str(e),
                    "evaluator_type": "loongflow_adapter",
                    "fallback_used": self.use_fallback
                },
                execution_time=time.time() - start_time
            )

    async def _evaluate_with_loongflow(
        self,
        solution_content: str,
        round_rule: Any,
        context: Dict[str, Any],
        rule_id: str
    ) -> GauntletRoundResult:
        """
        Evaluate using LoongFlow's AI evaluator.

        This method uses LoongFlow's GeneralEvaluator which provides
        sophisticated AI-based evaluation with scoring and feedback.
        """
        try:
            # Import LoongFlow message types
            from loongflow.agentsdk.message import Message, ContentElement
            from loongflow.framework.pes.context import Context

            # Create message with solution
            message = Message.from_elements([
                ContentElement(
                    mime_type="text/plain",
                    data=solution_content
                )
            ])

            # Add context if available
            if "problem" in context:
                message.add_element(ContentElement(
                    mime_type="text/plain",
                    data=f"Problem: {context['problem']}"
                ))

            if "criteria" in context:
                criteria_text = "\n".join(f"- {c}" for c in context["criteria"])
                message.add_element(ContentElement(
                    mime_type="text/plain",
                    data=f"Evaluation Criteria:\n{criteria_text}"
                ))

            # Create evaluation context
            eval_context = Context(
                trace_id=context.get("trace_id", "gauntlet_eval"),
                workspace_dir=context.get("workspace_dir", "/tmp/gauntlet_eval")
            )

            # Run evaluation
            logger.info(f"Starting LoongFlow evaluation for {round_rule.rule_id}")
            eval_result = await self.evaluator.evaluate(
                message=message,
                context=eval_context
            )

            # Extract score and feedback
            score = float(getattr(eval_result, 'score', 0.0))
            summary = getattr(eval_result, 'summary', '')
            metrics = getattr(eval_result, 'metrics', {})

            # Determine if passed
            min_score = getattr(round_rule, 'min_overall_confidence', 0.7)
            passed = score >= min_score

            # Build result
            return GauntletRoundResult(
                rule_id=rule_id,
                passed=passed,
                score=score,
                feedback=summary,
                details={
                    "evaluation_type": "loongflow_ai",
                    "metrics": metrics,
                    "status": getattr(eval_result, 'status', 'unknown'),
                    "min_score_required": min_score
                },
                execution_time=0.0  # Will be set by caller
            )

        except Exception as e:
            logger.error(f"LoongFlow evaluation error: {e}", exc_info=True)
            # Fall back to simple evaluation
            return await self._evaluate_with_fallback(
                solution_content, round_rule, context, rule_id
            )

    async def _evaluate_with_fallback(
        self,
        solution_content: str,
        round_rule: Any,
        context: Dict[str, Any],
        rule_id: str
    ) -> GauntletRoundResult:
        """
        Fallback evaluation using simple keyword and pattern matching.

        This provides basic quality assessment when LoongFlow is unavailable.
        It checks for:
        - Solution completeness (length)
        - Code quality indicators
        - Documentation presence
        - Basic patterns
        """
        logger.info("Using fallback evaluation")

        # Extract criteria
        criteria = context.get("criteria", ["correctness", "completeness"])
        problem = context.get("problem", "")

        # Basic metrics
        length_score = min(1.0, len(solution_content) / 1000)  # Up to 1000 chars
        has_code = "```" in solution_content or "def " in solution_content or "class " in solution_content
        has_explanation = any(word in solution_content.lower() for word in
                             ["because", "therefore", "approach", "solution", "method"])

        # Calculate base score
        score = length_score * 0.5
        if has_code:
            score += 0.3
        if has_explanation:
            score += 0.2

        # Problem-specific boosts
        if problem and any(word in solution_content.lower() for word in problem.lower().split()[:5]):
            score += 0.1

        # Cap at 1.0
        score = min(1.0, score)

        # Generate feedback
        feedback_parts = []
        if length_score < 0.5:
            feedback_parts.append("Solution seems brief - consider adding more detail")
        if has_code:
            feedback_parts.append("Includes code implementation")
        if has_explanation:
            feedback_parts.append("Provides explanation of approach")
        if problem and problem.lower() not in solution_content.lower():
            feedback_parts.append("Could better reference the specific problem")

        feedback = "Fallback evaluation: " + "; ".join(feedback_parts) if feedback_parts else "Basic solution structure detected"

        # Determine if passed
        min_score = getattr(round_rule, 'min_overall_confidence', 0.7)
        passed = score >= min_score

        return GauntletRoundResult(
            rule_id=rule_id,
            passed=passed,
            score=score,
            feedback=feedback,
            details={
                "evaluation_type": "fallback",
                "length_score": length_score,
                "has_code": has_code,
                "has_explanation": has_explanation,
                "criteria_evaluated": criteria,
                "min_score_required": min_score
            },
            execution_time=0.0
        )

    def _extract_solution_content(self, solution: Any) -> str:
        """Extract solution content from various input types."""
        if isinstance(solution, str):
            return solution
        elif hasattr(solution, 'content'):
            return solution.content
        elif hasattr(solution, 'solution_content'):
            return solution.solution_content
        elif isinstance(solution, dict):
            return solution.get('content', solution.get('solution', ''))
        else:
            return str(solution)

    async def batch_evaluate(
        self,
        solutions: List[Any],
        round_rule: Any,
        context: Dict[str, Any]
    ) -> List[GauntletRoundResult]:
        """
        Evaluate multiple solutions in parallel.

        Args:
            solutions: List of solutions to evaluate
            round_rule: Gauntlet round configuration
            context: Additional context

        Returns:
            List of GauntletRoundResult (same order as input)
        """
        tasks = [
            self.evaluate_round(solution, round_rule, context)
            for solution in solutions
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(GauntletRoundResult(
                    rule_id=round_rule.rule_id,
                    passed=False,
                    score=0.0,
                    feedback=f"Evaluation failed: {str(result)}",
                    details={"error": str(result), "solution_index": i},
                    execution_time=0.0
                ))
            else:
                processed_results.append(result)

        return processed_results


def create_loongflow_evaluator(
    llm_config: Dict[str, Any],
    timeout: int = 60,
    enable_loongflow: bool = True
) -> LoongFlowEvaluatorAdapter:
    """
    Factory function to create a LoongFlow evaluator adapter.

    Args:
        llm_config: LLM configuration
        timeout: Evaluation timeout in seconds
        enable_loongflow: Whether to use LoongFlow (False = fallback only)

    Returns:
        Configured LoongFlowEvaluatorAdapter instance

    Example:
        ```python
        adapter = create_loongflow_evaluator(
            llm_config={
                'model': 'claude-3-5-sonnet-20241022',
                'api_key': 'sk-...',
                'url': 'http://localhost:8001'
            },
            timeout=60
        )
        ```
    """
    return LoongFlowEvaluatorAdapter(
        llm_config=llm_config,
        timeout=timeout,
        enable_loongflow=enable_loongflow
    )
