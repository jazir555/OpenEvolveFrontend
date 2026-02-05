"""Quality judgment and evaluation for ICR.

Evaluates if outputs meet quality thresholds through comprehensive
scoring across multiple criteria dimensions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of an evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    UNDETERMINED = "undetermined"


@dataclass
class Criteria:
    """Quality criteria configuration.
    
    Attributes:
        accuracy: Weight for accuracy (0-1)
        completeness: Weight for completeness (0-1)
        clarity: Weight for clarity (0-1)
        conciseness: Weight for conciseness (0-1)
        correctness: Weight for correctness (0-1)
        consistency: Weight for consistency (0-1)
        custom: Dictionary of custom criteria and weights
    """
    accuracy: float = 0.25
    completeness: float = 0.25
    clarity: float = 0.20
    conciseness: float = 0.10
    correctness: float = 0.15
    consistency: float = 0.05
    custom: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        all_weights = [
            self.accuracy, self.completeness, self.clarity,
            self.conciseness, self.correctness, self.consistency,
            *self.custom.values(),
        ]
        total = sum(all_weights)
        if total > 0 and abs(total - 1.0) > 0.01:
            # Normalize
            factor = 1.0 / total
            self.accuracy *= factor
            self.completeness *= factor
            self.clarity *= factor
            self.conciseness *= factor
            self.correctness *= factor
            self.consistency *= factor
            self.custom = {k: v * factor for k, v in self.custom.items()}
    
    def get_all_criteria(self) -> Dict[str, float]:
        """Get all criteria as a flat dictionary."""
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "conciseness": self.conciseness,
            "correctness": self.correctness,
            "consistency": self.consistency,
            **self.custom,
        }
    
    @classmethod
    def strict(cls) -> "Criteria":
        """Create strict criteria emphasizing correctness and accuracy."""
        return cls(
            accuracy=0.30,
            completeness=0.20,
            clarity=0.15,
            conciseness=0.05,
            correctness=0.25,
            consistency=0.05,
        )
    
    @classmethod
    def balanced(cls) -> "Criteria":
        """Create balanced criteria."""
        return cls(
            accuracy=0.20,
            completeness=0.20,
            clarity=0.20,
            conciseness=0.10,
            correctness=0.20,
            consistency=0.10,
        )
    
    @classmethod
    def creative(cls) -> "Criteria":
        """Create criteria emphasizing clarity and completeness."""
        return cls(
            accuracy=0.15,
            completeness=0.25,
            clarity=0.30,
            conciseness=0.10,
            correctness=0.15,
            consistency=0.05,
        )


@dataclass
class EvaluationResult:
    """Result of quality evaluation.
    
    Attributes:
        score: Overall quality score (0-1)
        passed: Whether the evaluation passed
        criteria_scores: Individual scores per criterion
        feedback: Human-readable feedback
        status: Evaluation status
        metadata: Additional evaluation metadata
    """
    score: float
    passed: bool
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    status: EvaluationStatus = EvaluationStatus.UNDETERMINED
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Validate and set status."""
        self.score = max(0.0, min(1.0, self.score))
        
        if self.status == EvaluationStatus.UNDETERMINED:
            if self.passed:
                self.status = EvaluationStatus.PASSED
            elif self.score > 0.5:
                self.status = EvaluationStatus.PARTIAL
            else:
                self.status = EvaluationStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "passed": self.passed,
            "criteria_scores": self.criteria_scores,
            "feedback": self.feedback,
            "status": self.status.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two outputs.
    
    Attributes:
        winner: Which output is better ("original", "refined", or "tie")
        score_delta: Difference in scores
        improvements: List of improvements in refined version
        regressions: List of regressions in refined version
        recommendation: Action recommendation
    """
    winner: str
    score_delta: float
    improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    @property
    def is_improvement(self) -> bool:
        """Check if refined is better than original."""
        return self.winner == "refined" and self.score_delta > 0.05
    
    @property
    def is_regression(self) -> bool:
        """Check if refined is worse than original."""
        return self.winner == "original" and self.score_delta < -0.05


class Judge:
    """Quality evaluator for outputs.
    
    The Judge evaluates content quality against criteria and determines
    if it meets the required threshold for acceptance.
    
    Example:
        >>> judge = Judge(default_criteria=Criteria.strict())
        >>> result = judge.evaluate(
        ...     output="The refined content...",
        ...     threshold=0.9
        ... )
        >>> if result.passed:
        ...     print("Quality threshold met!")
        ... else:
        ...     print(f"Score {result.score:.2f} below threshold")
    """
    
    def __init__(
        self,
        default_criteria: Optional[Criteria] = None,
        default_threshold: float = 0.85,
        use_critique_integration: bool = True,
    ):
        """Initialize the judge.
        
        Args:
            default_criteria: Default criteria for evaluation
            default_threshold: Default quality threshold
            use_critique_integration: Whether to integrate with critic
        """
        self.default_criteria = default_criteria or Criteria.balanced()
        self.default_threshold = default_threshold
        self.use_critique_integration = use_critique_integration
        self._evaluation_count = 0
        self._backend: Optional[Callable] = None
        
        logger.info(f"Initialized Judge with threshold={default_threshold}")
    
    def set_backend(self, backend: Callable[[str, Criteria], Dict[str, float]]) -> None:
        """Set backend for criteria evaluation.
        
        Args:
            backend: Callable that takes (content, criteria) and returns score dict
        """
        self._backend = backend
        logger.debug("Evaluation backend registered")
    
    def evaluate(
        self,
        output: str,
        criteria: Optional[Criteria] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate output quality.
        
        Args:
            output: Content to evaluate
            criteria: Criteria to use (default if not provided)
            context: Additional evaluation context
            
        Returns:
            EvaluationResult with scores and feedback
        """
        criteria = criteria or self.default_criteria
        context = context or {}
        
        logger.info("Starting evaluation", extra={
            "correlation_id": context.get("correlation_id"),
            "output_length": len(output),
        })
        
        # Get criteria scores
        if self._backend:
            criteria_scores = self._backend(output, criteria)
        else:
            criteria_scores = self._heuristic_evaluation(output, criteria)
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(criteria_scores, criteria)
        
        # Generate feedback
        feedback = self._generate_feedback(criteria_scores, criteria)
        
        # Determine if passed (use context threshold or default)
        threshold = context.get("threshold", self.default_threshold)
        passed = overall_score >= threshold
        
        self._evaluation_count += 1
        
        result = EvaluationResult(
            score=overall_score,
            passed=passed,
            criteria_scores=criteria_scores,
            feedback=feedback,
            metadata={
                "evaluation_number": self._evaluation_count,
                "threshold_used": threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        logger.debug(f"Evaluation complete: score={overall_score:.3f}, passed={passed}")
        return result
    
    def compare(
        self,
        original: str,
        refined: str,
        criteria: Optional[Criteria] = None,
    ) -> ComparisonResult:
        """Compare original and refined outputs.
        
        Args:
            original: Original content
            refined: Refined content
            criteria: Criteria for comparison
            
        Returns:
            ComparisonResult with winner and analysis
        """
        original_eval = self.evaluate(original, criteria)
        refined_eval = self.evaluate(refined, criteria)
        
        score_delta = refined_eval.score - original_eval.score
        
        # Determine winner
        if abs(score_delta) < 0.05:
            winner = "tie"
        elif score_delta > 0:
            winner = "refined"
        else:
            winner = "original"
        
        # Identify improvements and regressions
        improvements = []
        regressions = []
        
        for criterion in original_eval.criteria_scores:
            orig_score = original_eval.criteria_scores.get(criterion, 0)
            refined_score = refined_eval.criteria_scores.get(criterion, 0)
            diff = refined_score - orig_score
            
            if diff > 0.1:
                improvements.append(f"{criterion}: +{diff:.2f}")
            elif diff < -0.1:
                regressions.append(f"{criterion}: {diff:.2f}")
        
        # Generate recommendation
        if winner == "refined":
            recommendation = "Accept refined version"
        elif winner == "original":
            recommendation = "Keep original version"
        else:
            recommendation = "Versions are comparable; consider other factors"
        
        return ComparisonResult(
            winner=winner,
            score_delta=score_delta,
            improvements=improvements,
            regressions=regressions,
            recommendation=recommendation,
        )
    
    def meets_threshold(
        self,
        output: str,
        threshold: Optional[float] = None,
        criteria: Optional[Criteria] = None,
    ) -> bool:
        """Quick check if output meets quality threshold.
        
        Args:
            output: Content to check
            threshold: Quality threshold (default if not provided)
            criteria: Criteria to use
            
        Returns:
            True if threshold is met
        """
        threshold = threshold or self.default_threshold
        result = self.evaluate(output, criteria)
        return result.score >= threshold
    
    def _heuristic_evaluation(self, output: str, criteria: Criteria) -> Dict[str, float]:
        """Heuristic-based evaluation when no backend is available."""
        scores = {}
        
        # Accuracy heuristic: penalize placeholders
        placeholders = ["TODO", "FIXME", "XXX", "[Generated]", "placeholder"]
        placeholder_count = sum(1 for p in placeholders if p in output)
        scores["accuracy"] = max(0.0, 1.0 - placeholder_count * 0.3)
        
        # Completeness heuristic: length-based with limits
        if len(output) < 50:
            scores["completeness"] = 0.5
        elif len(output) < 200:
            scores["completeness"] = 0.75
        elif len(output) < 1000:
            scores["completeness"] = 0.9
        else:
            scores["completeness"] = 0.85  # Very long might be verbose
        
        # Clarity heuristic: sentence structure
        sentences = [s.strip() for s in output.split('.') if s.strip()]
        if not sentences:
            scores["clarity"] = 0.3
        else:
            avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
            if avg_sentence_len < 50:
                scores["clarity"] = 0.9
            elif avg_sentence_len < 100:
                scores["clarity"] = 0.75
            else:
                scores["clarity"] = 0.6
        
        # Conciseness heuristic: information density
        words = output.split()
        unique_words = set(w.lower() for w in words)
        if words:
            density = len(unique_words) / len(words)
            scores["conciseness"] = min(1.0, density * 2)
        else:
            scores["conciseness"] = 0.5
        
        # Correctness heuristic: structure markers
        has_structure = any(marker in output for marker in ['\n', '. ', ':', '- '])
        scores["correctness"] = 0.8 if has_structure else 0.6
        
        # Consistency heuristic: style consistency
        caps_ratio = sum(1 for c in output if c.isupper()) / max(len(output), 1)
        scores["consistency"] = 0.9 if 0.02 < caps_ratio < 0.15 else 0.7
        
        # Custom criteria
        for custom_name in criteria.custom:
            # Default to balanced score for custom criteria
            scores[custom_name] = 0.75
        
        return scores
    
    def _calculate_weighted_score(
        self,
        criteria_scores: Dict[str, float],
        criteria: Criteria,
    ) -> float:
        """Calculate weighted overall score."""
        all_criteria = criteria.get_all_criteria()
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for criterion, weight in all_criteria.items():
            score = criteria_scores.get(criterion, 0.75)  # Default neutral score
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.75
        
        return weighted_sum / total_weight
    
    def _generate_feedback(
        self,
        criteria_scores: Dict[str, float],
        criteria: Criteria,
    ) -> str:
        """Generate human-readable feedback."""
        lines = ["Quality Evaluation Results:"]
        
        # Overall score
        overall = self._calculate_weighted_score(criteria_scores, criteria)
        lines.append(f"\nOverall Score: {overall:.1%}")
        
        # Individual criteria
        lines.append("\nBy Criterion:")
        for criterion, score in sorted(criteria_scores.items()):
            status = "[OK]" if score >= 0.8 else "⚠" if score >= 0.6 else "[FAIL]"
            lines.append(f"  {status} {criterion}: {score:.1%}")
        
        # Strengths
        strengths = [c for c, s in criteria_scores.items() if s >= 0.8]
        if strengths:
            lines.append(f"\nStrengths: {', '.join(strengths)}")
        
        # Areas for improvement
        improvements = [c for c, s in criteria_scores.items() if s < 0.7]
        if improvements:
            lines.append(f"\nAreas for Improvement: {', '.join(improvements)}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get judge statistics."""
        return {
            "total_evaluations": self._evaluation_count,
            "default_threshold": self.default_threshold,
            "has_backend": self._backend is not None,
        }


class JudgeError(Exception):
    """Error during judgment operation."""
    pass
