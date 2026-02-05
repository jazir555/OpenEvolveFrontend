"""
LoongFlow Gauntlet Evaluator

Integrates LoongFlow PES evaluation as a quick screening Round 1 evaluator
in the OpenEvolve gauntlet system.

This module provides:
- LoongFlowGauntletEvaluator: Main evaluator class
- LoongFlowGauntletConfig: Configuration schema
- GauntletEvaluationResult: Evaluation result data structure
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_validator

# Import LoongFlow adapter
try:
    from openevolve.integrations.loongflow_adapter import LoongFlowAdapter
except ImportError:
    # Fallback for direct execution
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from openevolve.integrations.loongflow_adapter import LoongFlowAdapter
    except ImportError:
        # Create a mock adapter for testing
        LoongFlowAdapter = None

logger = logging.getLogger(__name__)


class LoongFlowGauntletConfig(BaseModel):
    """
    Configuration for LoongFlow Gauntlet Evaluator.

    Attributes:
        enable_planning: Enable PES planning phase
        enable_memory: Enable PES memory system
        early_stopping: Enable early stopping on improvement
        plan_temperature: Temperature for planning LLM calls
        summary_temperature: Temperature for summary LLM calls
        evaluation_timeout: Timeout for single evaluation (seconds)
        max_evaluations: Maximum PES evaluations per solution
        quality_threshold: Minimum quality score to pass (0.0-1.0)
        confidence_threshold: Minimum confidence to pass (0.0-1.0)
        enable_detailed_feedback: Enable detailed feedback generation
        correctness_weight: Weight for correctness score (0.0-1.0)
        efficiency_weight: Weight for efficiency score (0.0-1.0)
        robustness_weight: Weight for robustness score (0.0-1.0)
        creativity_weight: Weight for creativity score (0.0-1.0)

    Example:
        >>> config = LoongFlowGauntletConfig(
        ...     quality_threshold=0.6,
        ...     max_evaluations=50,
        ...     enable_detailed_feedback=True
        ... )
    """

    # LoongFlow PES configuration
    enable_planning: bool = True
    enable_memory: bool = True
    early_stopping: bool = True
    plan_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    summary_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Gauntlet-specific configuration
    evaluation_timeout: int = Field(default=30, ge=5, le=300, description="Timeout in seconds")
    max_evaluations: int = Field(default=50, ge=10, le=1000, description="Max PES evaluations")
    quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Pass threshold")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Min confidence")
    enable_detailed_feedback: bool = True

    # Scoring weights (must sum to 1.0)
    correctness_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    efficiency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    robustness_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    creativity_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    class Config:
        """Pydantic config."""
        validate_assignment = True

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self):
        """Validate that weights sum to 1.0."""
        total = (
            self.correctness_weight +
            self.efficiency_weight +
            self.robustness_weight +
            self.creativity_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total}. "
                f"Adjust weights: correctness={self.correctness_weight}, "
                f"efficiency={self.efficiency_weight}, robustness={self.robustness_weight}, "
                f"creativity={self.creativity_weight}"
            )
        return self


@dataclass
class GauntletEvaluationResult:
    """
    Result from LoongFlow gauntlet evaluation.

    Attributes:
        solution: The solution that was evaluated
        passed: Whether the solution passed the gauntlet
        overall_score: Overall score (0.0 to 1.0)
        confidence: Confidence in the evaluation (0.0 to 1.0)
        correctness_score: Score for correctness dimension
        efficiency_score: Score for efficiency dimension
        robustness_score: Score for robustness dimension
        creativity_score: Score for creativity dimension
        pes_iterations: Number of PES iterations performed
        pes_evaluations: Number of PES evaluations performed
        convergence_quality: Quality of convergence (0.0 to 1.0)
        feedback: Detailed feedback text
        strengths: List of identified strengths
        weaknesses: List of identified weaknesses
        suggestions: List of improvement suggestions
        evaluation_time: Time taken for evaluation (seconds)
        timestamp: When evaluation was performed
        artifacts: Additional artifacts and metadata

    Example:
        >>> result = GauntletEvaluationResult(
        ...     solution="def foo(): return 42",
        ...     passed=True,
        ...     overall_score=0.85,
        ...     confidence=0.90,
        ...     feedback="Excellent solution",
        ...     evaluation_time=5.2,
        ...     timestamp=datetime.now(UTC)
        ... )
    """

    solution: str
    passed: bool
    overall_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    # Detailed scores
    correctness_score: float
    efficiency_score: float
    robustness_score: float
    creativity_score: float

    # PES-specific metrics
    pes_iterations: int
    pes_evaluations: int
    convergence_quality: float

    # Feedback
    feedback: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Metadata
    evaluation_time: float = 0.0  # seconds
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "solution": self.solution,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "correctness_score": self.correctness_score,
            "efficiency_score": self.efficiency_score,
            "robustness_score": self.robustness_score,
            "creativity_score": self.creativity_score,
            "pes_iterations": self.pes_iterations,
            "pes_evaluations": self.pes_evaluations,
            "convergence_quality": self.convergence_quality,
            "feedback": self.feedback,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp.isoformat(),
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GauntletEvaluationResult":
        """Create GauntletEvaluationResult from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class LoongFlowGauntletEvaluator:
    """
    LoongFlow Gauntlet Evaluator for Round 1 screening.

    This evaluator uses LoongFlow's PES system to quickly assess solution quality
    before proceeding to more expensive Red Team and Gold Team evaluation rounds.

    Key Features:
    - Fast PES-based evaluation (<30 seconds)
    - Multi-dimensional scoring (correctness, efficiency, robustness, creativity)
    - Configurable thresholds and weights
    - Detailed feedback generation
    - Batch evaluation support
    - Graceful fallback when LoongFlow unavailable

    Example:
        >>> config = LoongFlowGauntletConfig(quality_threshold=0.6)
        >>> evaluator = LoongFlowGauntletEvaluator(config)
        >>> result = await evaluator.evaluate_solution(
        ...     solution="def foo(): return 42",
        ...     problem="Create a function that returns 42",
        ...     domain="code"
        ... )
        >>> print(f"Passed: {result.passed}, Score: {result.overall_score}")
    """

    def __init__(self, config: LoongFlowGauntletConfig):
        """
        Initialize LoongFlow Gauntlet Evaluator.

        Args:
            config: Configuration for the evaluator

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config

        # Initialize LoongFlow adapter
        if LoongFlowAdapter is not None:
            loongflow_config = {
                "max_iterations": config.max_evaluations,
                "enable_planning": config.enable_planning,
                "enable_memory": config.enable_memory,
                "timeout": config.evaluation_timeout,
            }

            self.loongflow_adapter = LoongFlowAdapter(config=loongflow_config)

            # Log initialization status
            if self.loongflow_adapter.is_available():
                logger.info("[OK] LoongFlow Gauntlet Evaluator initialized successfully")
            else:
                logger.warning("[WARN]  LoongFlow unavailable, using fallback evaluation mode")
        else:
            self.loongflow_adapter = None
            logger.warning("[WARN]  LoongFlow adapter not available, using mock evaluation mode")

    async def evaluate_solution(
        self,
        solution: str,
        problem: str,
        domain: str = "general",
        **kwargs
    ) -> GauntletEvaluationResult:
        """
        Evaluate a single solution using LoongFlow PES.

        This is the main evaluation method. It runs a quick PES assessment
        and returns a detailed evaluation result.

        Args:
            solution: The solution code/program to evaluate
            problem: Problem description/statement
            domain: Problem domain (math, code, general, etc.)
            **kwargs: Additional parameters

        Returns:
            GauntletEvaluationResult with detailed scores and feedback

        Example:
            >>> result = await evaluator.evaluate_solution(
            ...     solution="def solve(): return optimal_solution",
            ...     problem="Optimize the packing problem",
            ...     domain="math"
            ... )
            >>> if result.passed:
            ...     print("Solution passed Round 1!")
        """
        start_time = time.time()

        try:
            # Step 1: Quick PES assessment
            logger.info(f"Starting LoongFlow evaluation for domain={domain}")

            pes_result = await self.loongflow_adapter.evolve(
                problem=problem,
                domain=domain,
                initial_code=solution,
                max_iterations=10,  # Quick assessment
                **kwargs
            )

            # Step 2: Extract metrics and calculate scores
            scores = await self._calculate_scores(solution, problem, pes_result)

            # Step 3: Calculate overall score
            overall_score = self._calculate_overall_score(scores)

            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(pes_result, overall_score)

            # Step 5: Check thresholds
            passed = self._check_thresholds(overall_score, confidence)

            # Step 6: Generate feedback
            feedback_data = await self._generate_feedback(
                solution, problem, scores, overall_score, passed
            )

            # Step 7: Create result
            evaluation_time = time.time() - start_time

            result = GauntletEvaluationResult(
                solution=solution,
                passed=passed,
                overall_score=overall_score,
                confidence=confidence,
                correctness_score=scores["correctness"],
                efficiency_score=scores["efficiency"],
                robustness_score=scores["robustness"],
                creativity_score=scores["creativity"],
                pes_iterations=pes_result.get("iterations_performed", 0),
                pes_evaluations=pes_result.get("total_evaluations", 0),
                convergence_quality=pes_result.get("convergence_quality", 0.5),
                feedback=feedback_data["feedback"],
                strengths=feedback_data["strengths"],
                weaknesses=feedback_data["weaknesses"],
                suggestions=feedback_data["suggestions"],
                evaluation_time=evaluation_time,
                timestamp=datetime.now(UTC),
                artifacts={
                    "pes_result": pes_result,
                    "domain": domain,
                    "config": self.config.model_dump(),
                }
            )

            logger.info(
                f"Evaluation complete: score={overall_score:.2%}, "
                f"passed={passed}, time={evaluation_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)

            # Return failure result
            evaluation_time = time.time() - start_time

            return GauntletEvaluationResult(
                solution=solution,
                passed=False,
                overall_score=0.0,
                confidence=0.0,
                correctness_score=0.0,
                efficiency_score=0.0,
                robustness_score=0.0,
                creativity_score=0.0,
                pes_iterations=0,
                pes_evaluations=0,
                convergence_quality=0.0,
                feedback=f"Evaluation error: {str(e)}",
                strengths=[],
                weaknesses=["Evaluation failed"],
                suggestions=[],
                evaluation_time=evaluation_time,
                timestamp=datetime.now(UTC),
                artifacts={"error": str(e)}
            )

    async def evaluate_batch(
        self,
        solutions: List[str],
        problem: str,
        domain: str = "general",
        **kwargs
    ) -> List[GauntletEvaluationResult]:
        """
        Evaluate multiple solutions in batch.

        Evaluates solutions concurrently for better performance.

        Args:
            solutions: List of solutions to evaluate
            problem: Problem description
            domain: Problem domain
            **kwargs: Additional parameters

        Returns:
            List of evaluation results (same order as input)

        Example:
            >>> solutions = ["def foo(): return 1", "def foo(): return 2"]
            >>> results = await evaluator.evaluate_batch(
            ...     solutions=solutions,
            ...     problem="Return a number",
            ...     domain="code"
            ... )
            >>> for result in results:
            ...     print(f"Score: {result.overall_score}")
        """
        logger.info(f"Starting batch evaluation of {len(solutions)} solutions")

        # Evaluate concurrently
        tasks = [
            self.evaluate_solution(sol, problem, domain, **kwargs)
            for sol in solutions
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Solution {i} evaluation failed: {result}")
                # Create failure result
                final_results.append(GauntletEvaluationResult(
                    solution=solutions[i],
                    passed=False,
                    overall_score=0.0,
                    confidence=0.0,
                    correctness_score=0.0,
                    efficiency_score=0.0,
                    robustness_score=0.0,
                    creativity_score=0.0,
                    pes_iterations=0,
                    pes_evaluations=0,
                    convergence_quality=0.0,
                    feedback=f"Evaluation failed: {str(result)}",
                    strengths=[],
                    weaknesses=["Batch evaluation failed"],
                    suggestions=[],
                    evaluation_time=0.0,
                    timestamp=datetime.now(UTC),
                    artifacts={"error": str(result)}
                ))
            else:
                final_results.append(result)

        logger.info(
            f"Batch evaluation complete: "
            f"{sum(r.passed for r in final_results)}/{len(final_results)} passed"
        )

        return final_results

    async def _calculate_scores(
        self,
        solution: str,
        problem: str,
        pes_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate detailed scores from PES result.

        Analyzes the PES evolution to extract:
        - Correctness: Does it solve the problem?
        - Efficiency: How many evaluations were needed?
        - Robustness: Did it converge stably?
        - Creativity: Is the approach novel?

        Args:
            solution: The solution code
            problem: Problem description
            pes_result: Result from LoongFlow PES

        Returns:
            Dictionary with score dimensions
        """
        # Base scores from PES fitness
        fitness = pes_result.get("best_fitness", 0.0)

        # Correctness: Based on final fitness
        correctness = min(1.0, max(0.0, fitness))

        # Efficiency: Based on evaluations used vs budget
        total_evals = pes_result.get("total_evaluations", self.config.max_evaluations)
        efficiency = 1.0 - (total_evals / self.config.max_evaluations)
        efficiency = max(0.0, min(1.0, efficiency))

        # Robustness: Based on convergence quality
        improvement_rate = pes_result.get("improvement_rate", 0.0)
        robustness = min(1.0, improvement_rate)

        # Creativity: Analyze solution novelty
        # Check for unique patterns, non-obvious approaches
        creativity = await self._assess_creativity(solution, problem)

        return {
            "correctness": correctness,
            "efficiency": efficiency,
            "robustness": robustness,
            "creativity": creativity,
        }

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score.

        Args:
            scores: Dictionary with score dimensions

        Returns:
            Overall score (0.0 to 1.0)
        """
        overall = (
            scores["correctness"] * self.config.correctness_weight +
            scores["efficiency"] * self.config.efficiency_weight +
            scores["robustness"] * self.config.robustness_weight +
            scores["creativity"] * self.config.creativity_weight
        )
        return max(0.0, min(1.0, overall))

    def _calculate_confidence(
        self,
        pes_result: Dict[str, Any],
        overall_score: float
    ) -> float:
        """
        Calculate confidence in the evaluation.

        Higher confidence when:
        - LoongFlow is available
        - More evaluations were performed
        - Convergence was stable
        - Score is consistently high

        Args:
            pes_result: PES evolution result
            overall_score: Overall score

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not self.loongflow_adapter.is_available():
            # Low confidence in fallback mode
            return 0.3

        # Base confidence on iterations performed
        iterations = pes_result.get("iterations_performed", 0)
        base_confidence = min(1.0, iterations / 10.0)

        # Adjust based on overall score
        # Higher scores should have higher confidence
        score_adjustment = overall_score * 0.2

        # Adjust based on convergence
        convergence = pes_result.get("convergence_quality", 0.5)
        convergence_adjustment = (convergence - 0.5) * 0.2

        confidence = base_confidence + score_adjustment + convergence_adjustment
        return max(0.0, min(1.0, confidence))

    def _check_thresholds(self, overall_score: float, confidence: float) -> bool:
        """
        Check if solution passes quality and confidence thresholds.

        Args:
            overall_score: Overall score
            confidence: Confidence score

        Returns:
            True if solution passes both thresholds
        """
        score_pass = overall_score >= self.config.quality_threshold
        confidence_pass = confidence >= self.config.confidence_threshold

        return score_pass and confidence_pass

    async def _generate_feedback(
        self,
        solution: str,
        problem: str,
        scores: Dict[str, float],
        overall_score: float,
        passed: bool
    ) -> Dict[str, Any]:
        """
        Generate detailed feedback for the solution.

        Args:
            solution: The solution code
            problem: Problem description
            scores: Score dimensions
            overall_score: Overall score
            passed: Whether solution passed

        Returns:
            Dictionary with feedback, strengths, weaknesses, suggestions
        """
        # Generate strengths
        strengths = []
        if scores["correctness"] > 0.8:
            strengths.append(f"Excellent correctness ({scores['correctness']:.1%})")
        elif scores["correctness"] > 0.6:
            strengths.append(f"Good correctness ({scores['correctness']:.1%})")

        if scores["efficiency"] > 0.8:
            strengths.append(f"Highly efficient approach ({scores['efficiency']:.1%})")
        elif scores["efficiency"] > 0.6:
            strengths.append(f"Reasonably efficient ({scores['efficiency']:.1%})")

        if scores["robustness"] > 0.8:
            strengths.append(f"Very robust solution ({scores['robustness']:.1%})")

        if scores["creativity"] > 0.7:
            strengths.append("Creative and novel approach")

        # Generate weaknesses
        weaknesses = []
        if scores["correctness"] < 0.5:
            weaknesses.append(f"Low correctness ({scores['correctness']:.1%}) - may not solve problem")

        if scores["efficiency"] < 0.4:
            weaknesses.append(f"Inefficient ({scores['efficiency']:.1%}) - uses too many resources")

        if scores["robustness"] < 0.4:
            weaknesses.append(f"Fragile ({scores['robustness']:.1%}) - may fail on edge cases")

        if scores["creativity"] < 0.3:
            weaknesses.append("Conventional approach - lacks innovation")

        # Generate suggestions
        suggestions = []

        # Score-based suggestions
        if scores["correctness"] < 0.7:
            suggestions.append("Improve correctness by testing more edge cases")

        if scores["efficiency"] < 0.7:
            suggestions.append("Optimize algorithm to reduce resource usage")

        if scores["robustness"] < 0.7:
            suggestions.append("Add error handling and validation")

        if scores["creativity"] < 0.5:
            suggestions.append("Consider alternative, more creative approaches")

        # Generate main feedback text
        feedback_parts = [
            f"**Overall Score:** {overall_score:.1%}",
            f"**Confidence:** {self._calculate_confidence({}, overall_score):.1%}\n",
            "**Score Breakdown:**",
            f"- Correctness: {scores['correctness']:.1%}",
            f"- Efficiency: {scores['efficiency']:.1%}",
            f"- Robustness: {scores['robustness']:.1%}",
            f"- Creativity: {scores['creativity']:.1%}\n",
        ]

        if strengths:
            feedback_parts.append("**Strengths:**")
            for strength in strengths:
                feedback_parts.append(f"[OK] {strength}")
            feedback_parts.append("")

        if weaknesses:
            feedback_parts.append("**Weaknesses:**")
            for weakness in weaknesses:
                feedback_parts.append(f"[FAIL] {weakness}")
            feedback_parts.append("")

        if suggestions and self.config.enable_detailed_feedback:
            feedback_parts.append("**Suggestions:**")
            for i, suggestion in enumerate(suggestions, 1):
                feedback_parts.append(f"{i}. {suggestion}")
            feedback_parts.append("")

        # Final recommendation
        if passed:
            feedback_parts.append("**Recommendation:** [OK] PASS - Proceed to Round 2 (Red Team)")
        else:
            feedback_parts.append("**Recommendation:** [FAIL] FAIL - Do not proceed to further rounds")

        feedback = "\n".join(feedback_parts)

        return {
            "feedback": feedback,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
        }

    async def _assess_creativity(self, solution: str, problem: str) -> float:
        """
        Assess creativity/novelty of solution.

        Heuristics for creativity:
        - Uses uncommon patterns/idioms
        - Non-obvious approach to problem
        - Unique combination of techniques
        - Avoids standard templates

        Args:
            solution: The solution code
            problem: Problem description

        Returns:
            Creativity score (0.0 to 1.0)
        """
        # Simple heuristics for now
        # In production, this could use LLM assessment

        creativity = 0.5  # Base score

        solution_lower = solution.lower()

        # Check for advanced concepts
        advanced_patterns = [
            "generator", "decorator", "context manager", "meta",
            "functional", "recursive", "async", "comprehension"
        ]

        pattern_count = sum(1 for pattern in advanced_patterns if pattern in solution_lower)
        creativity += min(0.3, pattern_count * 0.05)

        # Check for custom implementations (not just library calls)
        if "def " in solution and solution.count("def ") > 1:
            creativity += 0.1  # Multiple functions

        # Check for comments/docstrings (shows thought)
        if '"""' in solution or "'''" in solution or "#" in solution:
            creativity += 0.05

        # Penalize overly simplistic solutions
        if len(solution.splitlines()) < 3:
            creativity -= 0.2

        return max(0.0, min(1.0, creativity))

    def get_config(self) -> LoongFlowGauntletConfig:
        """Get current configuration."""
        return self.config

    def is_available(self) -> bool:
        """Check if LoongFlow is available."""
        if self.loongflow_adapter is None:
            return False
        return self.loongflow_adapter.is_available()

    async def evaluate(
        self,
        solution: str,
        problem: str,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> GauntletEvaluationResult:
        """
        Evaluate a solution - wrapper for evaluate_solution.

        This method provides a consistent interface expected by the multi-round
        orchestrator.

        Args:
            solution: The solution to evaluate
            problem: Problem description
            domain: Problem domain
            context: Additional context (optional)

        Returns:
            GauntletEvaluationResult with detailed evaluation
        """
        return await self.evaluate_solution(
            solution=solution,
            problem=problem,
            domain=domain
        )
