"""
Critique Aggregator - Production-Ready Implementation

This module provides comprehensive aggregation and analysis of critique reports
from multiple judges (AI models, human evaluators, or automated tests).

It integrates with:
- sovereign_data_models.py: CritiqueReport, SolutionAttempt
- sgd_workflow_orchestrator.py: SGD workflow integration
- openevolve_structures.py: Workflow structures

Features:
- Multi-source critique aggregation
- Weighted scoring with optional judge weights
- Approval calculation with configurable thresholds
- Comprehensive summary generation
- Improvement extraction
- Consensus measurement
- Edge case handling
- Full type hints
- Comprehensive error handling
- Unit tests included

Author: OpenEvolve Frontend Team
Created: 2026-01-22
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import logging
import json
from collections import defaultdict
import statistics
import re

# Configure logging
logger = logging.getLogger(__name__)

# Import data models with fallbacks
try:
    from sovereign_data_models import SolutionAttempt, generate_id
except ImportError:
    def generate_id(prefix: str = "") -> str:
        """Generate a unique ID with optional prefix."""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{unique_id}" if prefix else unique_id

    @dataclass
    class SolutionAttempt:
        """Fallback SolutionAttempt definition."""
        id: str
        sub_problem_id: str
        content: str


# =============================================================================
# DATA MODELS
# =============================================================================

class JudgeType(Enum):
    """Types of judges that can provide critiques."""
    AI_MODEL = "ai_model"
    HUMAN = "human"
    AUTOMATED_TEST = "automated_test"
    LINTING_TOOL = "linting_tool"
    SECURITY_SCANNER = "security_scanner"
    PERFORMANCE_ANALYZER = "performance_analyzer"


class CritiqueSeverity(Enum):
    """Severity levels for critiques."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class JudgeReport:
    """
    Individual critique report from a single judge.

    Attributes:
        judge_name: Name or ID of the judge (e.g., "gpt-4", "human_reviewer_1")
        judge_type: Type of judge (AI, human, automated test, etc.)
        is_approved: Whether the judge approves the solution
        score: Numerical score (0.0 to 1.0)
        feedback: Detailed feedback text
        improvements: List of specific improvements needed
        severity: Severity level of any issues found
        confidence: Judge's confidence in their assessment (0.0 to 1.0)
        metrics: Additional metrics from the judge (performance, security scores, etc.)
        timestamp: When the critique was generated
        metadata: Additional context or metadata
    """
    judge_name: str
    judge_type: JudgeType
    is_approved: bool
    score: float
    feedback: str
    improvements: List[str] = field(default_factory=list)
    severity: CritiqueSeverity = CritiqueSeverity.MEDIUM
    confidence: float = 1.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the judge report after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class CritiqueReport:
    """
    Aggregated critique report from multiple judges.

    Attributes:
        solution_attempt_id: ID of the solution being critiqued
        gauntlet_name: Name of the gauntlet/test suite used
        is_approved: Overall approval status (based on aggregated scores)
        reports_by_judge: List of individual JudgeReport objects
        summary: Comprehensive summary of all critiques
        aggregate_score: Weighted average of all judge scores
        consensus_score: Measure of agreement among judges (0.0 to 1.0)
        improvements_needed: Consolidated list of improvements needed
        approval_threshold: Minimum score required for approval
        created_at: When the report was generated
        metadata: Additional metadata
    """
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool
    reports_by_judge: List[JudgeReport]
    summary: str
    aggregate_score: float = 0.0
    consensus_score: float = 0.0
    improvements_needed: List[str] = field(default_factory=list)
    approval_threshold: float = 0.7
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the critique report to a dictionary."""
        return {
            "solution_attempt_id": self.solution_attempt_id,
            "gauntlet_name": self.gauntlet_name,
            "is_approved": self.is_approved,
            "aggregate_score": self.aggregate_score,
            "consensus_score": self.consensus_score,
            "approval_threshold": self.approval_threshold,
            "summary": self.summary,
            "improvements_needed": self.improvements_needed,
            "reports_by_judge": [
                {
                    "judge_name": r.judge_name,
                    "judge_type": r.judge_type.value,
                    "is_approved": r.is_approved,
                    "score": r.score,
                    "feedback": r.feedback,
                    "improvements": r.improvements,
                    "severity": r.severity.value,
                    "confidence": r.confidence,
                    "metrics": r.metrics,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in self.reports_by_judge
            ],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CritiqueReport":
        """Create a CritiqueReport from a dictionary."""
        reports = [
            JudgeReport(
                judge_name=r["judge_name"],
                judge_type=JudgeType(r["judge_type"]),
                is_approved=r["is_approved"],
                score=r["score"],
                feedback=r["feedback"],
                improvements=r.get("improvements", []),
                severity=CritiqueSeverity(r.get("severity", "medium")),
                confidence=r.get("confidence", 1.0),
                metrics=r.get("metrics", {}),
                timestamp=datetime.fromisoformat(r["timestamp"]) if r.get("timestamp") else datetime.now(),
                metadata=r.get("metadata", {})
            )
            for r in data.get("reports_by_judge", [])
        ]

        return cls(
            solution_attempt_id=data["solution_attempt_id"],
            gauntlet_name=data["gauntlet_name"],
            is_approved=data["is_approved"],
            reports_by_judge=reports,
            summary=data["summary"],
            aggregate_score=data.get("aggregate_score", 0.0),
            consensus_score=data.get("consensus_score", 0.0),
            improvements_needed=data.get("improvements_needed", []),
            approval_threshold=data.get("approval_threshold", 0.7),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {})
        )


@dataclass
class AggregationConfig:
    """
    Configuration for critique aggregation.

    Attributes:
        default_approval_threshold: Default minimum score for approval (0.0 to 1.0)
        default_weights: Default weights for different judge types
        min_judges_required: Minimum number of judges required for aggregation
        enable_outlier_detection: Whether to detect and handle outlier scores
        outlier_std_dev_threshold: Number of standard deviations for outlier detection
        consensus_algorithm: Algorithm to use for consensus calculation
        summary_max_length: Maximum length of generated summary
        extract_improvements: Whether to extract improvements from critiques
    """
    default_approval_threshold: float = 0.7
    default_weights: Dict[JudgeType, float] = field(default_factory=lambda: {
        JudgeType.HUMAN: 1.0,
        JudgeType.AI_MODEL: 0.9,
        JudgeType.AUTOMATED_TEST: 0.8,
        JudgeType.SECURITY_SCANNER: 1.0,
        JudgeType.PERFORMANCE_ANALYZER: 0.7,
        JudgeType.LINTING_TOOL: 0.5
    })
    min_judges_required: int = 1
    enable_outlier_detection: bool = True
    outlier_std_dev_threshold: float = 2.0
    consensus_algorithm: str = "std_dev"  # Options: "std_dev", "mean_deviation", "pairwise_agreement"
    summary_max_length: int = 2000
    extract_improvements: bool = True


# =============================================================================
# MAIN CRITIQUE AGGREGATOR CLASS
# =============================================================================

class CritiqueAggregator:
    """
    Aggregates and analyzes critique reports from multiple judges.

    This class provides methods to:
    - Create comprehensive critique reports
    - Aggregate reports from multiple judges
    - Calculate approval status
    - Generate summaries
    - Extract improvements
    - Calculate consensus

    Usage:
        aggregator = CritiqueAggregator()

        # Create judge reports
        judge_reports = [
            JudgeReport(
                judge_name="gpt-4",
                judge_type=JudgeType.AI_MODEL,
                is_approved=True,
                score=0.85,
                feedback="Good solution with minor improvements needed",
                improvements=["Add error handling", "Optimize loops"]
            ),
            # ... more judges
        ]

        # Create critique report
        critique_report = aggregator.create_critique_report(
            solution_id="solution_123",
            gauntlet_name="red_team_gauntlet",
            critiques=judge_reports
        )
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Initialize the CritiqueAggregator.

        Args:
            config: Optional aggregation configuration
        """
        self.config = config or AggregationConfig()
        logger.info(f"Initialized CritiqueAggregator with config: {self.config}")

    def create_critique_report(
        self,
        solution_id: str,
        gauntlet_name: str,
        critiques: List[Union[JudgeReport, Dict[str, Any]]],
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None
    ) -> CritiqueReport:
        """
        Create a comprehensive CritiqueReport from multiple judge critiques.

        Args:
            solution_id: ID of the solution being critiqued
            gauntlet_name: Name of the gauntlet/test suite
            critiques: List of JudgeReport objects or dicts
            weights: Optional custom weights for judges (overrides config defaults)
            threshold: Optional custom approval threshold (overrides config default)

        Returns:
            CritiqueReport with aggregated analysis

        Raises:
            ValueError: If critiques is empty or contains invalid data
        """
        if not critiques:
            raise ValueError("Cannot create critique report: no critiques provided")

        # Convert dict critiques to JudgeReport objects
        judge_reports = self._normalize_critiques(critiques)

        # Validate minimum judges requirement
        if len(judge_reports) < self.config.min_judges_required:
            logger.warning(
                f"Only {len(judge_reports)} judges provided, "
                f"minimum required: {self.config.min_judges_required}"
            )

        # Aggregate reports by judge
        aggregated_reports = self.aggregate_judge_reports(judge_reports, weights)

        # Calculate approval status
        approval_threshold = threshold or self.config.default_approval_threshold
        is_approved = self.calculate_approval(aggregated_reports, approval_threshold)

        # Calculate aggregate score
        aggregate_score = self._calculate_aggregate_score(aggregated_reports)

        # Calculate consensus
        consensus_score = self.calculate_consensus(aggregated_reports)

        # Generate summary
        summary = self.generate_summary(aggregated_reports)

        # Extract improvements
        improvements = self.extract_improvements(aggregated_reports)

        # Create the critique report
        critique_report = CritiqueReport(
            solution_attempt_id=solution_id,
            gauntlet_name=gauntlet_name,
            is_approved=is_approved,
            reports_by_judge=aggregated_reports,
            summary=summary,
            aggregate_score=aggregate_score,
            consensus_score=consensus_score,
            improvements_needed=improvements,
            approval_threshold=approval_threshold,
            metadata={
                "num_judges": len(aggregated_reports),
                "weights_used": weights or "default",
                "aggregation_timestamp": datetime.now().isoformat()
            }
        )

        logger.info(
            f"Created critique report for solution {solution_id}: "
            f"approved={is_approved}, score={aggregate_score:.2f}, "
            f"consensus={consensus_score:.2f}"
        )

        return critique_report

    def aggregate_judge_reports(
        self,
        reports: List[JudgeReport],
        weights: Optional[Dict[str, float]] = None
    ) -> List[JudgeReport]:
        """
        Aggregate judge reports with optional weighting.

        This method processes raw judge reports and optionally applies weights
        based on judge type or specific judge names.

        Args:
            reports: List of JudgeReport objects
            weights: Optional weights dict. Can be:
                - Dict mapping judge names to weights
                - Dict mapping judge type names to weights

        Returns:
            List of processed JudgeReport objects (weights stored in metadata)

        Raises:
            ValueError: If reports is empty
        """
        if not reports:
            raise ValueError("Cannot aggregate empty reports list")

        # Make a copy to avoid modifying original
        aggregated = list(reports)

        # Apply weights if provided
        if weights:
            for report in aggregated:
                # Try to get weight by judge name first, then by judge type
                weight = weights.get(report.judge_name)
                if weight is None:
                    weight = weights.get(report.judge_type.value)

                # Use config default weight if still None
                if weight is None:
                    weight = self.config.default_weights.get(
                        report.judge_type,
                        1.0
                    )

                # Store weight in metadata for transparency
                report.metadata["applied_weight"] = weight

                # Log weight application
                logger.debug(
                    f"Applied weight {weight} to judge {report.judge_name} "
                    f"(type: {report.judge_type.value})"
                )

        # Handle outlier detection if enabled
        if self.config.enable_outlier_detection and len(aggregated) > 2:
            aggregated = self._handle_outliers(aggregated)

        logger.info(f"Aggregated {len(aggregated)} judge reports")
        return aggregated

    def calculate_approval(
        self,
        reports: List[JudgeReport],
        threshold: float = 0.7
    ) -> bool:
        """
        Calculate approval status based on aggregated reports.

        A solution is approved if:
        1. The weighted average score meets or exceeds the threshold
        2. No critical severity issues are present
        3. At least one judge approves (if multiple judges)

        Args:
            reports: List of JudgeReport objects
            threshold: Minimum score required for approval (0.0 to 1.0)

        Returns:
            True if solution is approved, False otherwise

        Raises:
            ValueError: If reports is empty or threshold is invalid
        """
        if not reports:
            raise ValueError("Cannot calculate approval: no reports provided")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        # Check for critical severity issues
        has_critical = any(
            r.severity == CritiqueSeverity.CRITICAL and not r.is_approved
            for r in reports
        )
        if has_critical:
            logger.warning("Solution rejected due to critical severity issues")
            return False

        # Calculate weighted average score
        aggregate_score = self._calculate_aggregate_score(reports)

        # Check if at least one judge approves
        any_approves = any(r.is_approved for r in reports)

        # Final approval decision
        is_approved = aggregate_score >= threshold and any_approves

        logger.info(
            f"Approval calculation: score={aggregate_score:.2f}, "
            f"threshold={threshold:.2f}, any_approves={any_approves}, "
            f"approved={is_approved}"
        )

        return is_approved

    def generate_summary(
        self,
        reports: List[JudgeReport],
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate a comprehensive summary of all critique reports.

        The summary includes:
        - Overall assessment
        - Key findings from each judge
        - Common themes
        - Critical issues
        - Positive feedback

        Args:
            reports: List of JudgeReport objects
            max_length: Maximum length of summary (uses config default if None)

        Returns:
            Comprehensive summary text

        Raises:
            ValueError: If reports is empty
        """
        if not reports:
            raise ValueError("Cannot generate summary: no reports provided")

        max_length = max_length or self.config.summary_max_length

        # Calculate aggregate metrics
        avg_score = statistics.mean(r.score for r in reports)
        approval_rate = sum(1 for r in reports if r.is_approved) / len(reports)

        # Group reports by severity
        by_severity = defaultdict(list)
        for r in reports:
            by_severity[r.severity].append(r)

        # Start building summary
        summary_parts = []

        # Header
        summary_parts.append(
            f"## Critique Summary\n"
            f"**Overall Score:** {avg_score:.2f}/1.0\n"
            f"**Approval Rate:** {approval_rate:.1%}\n"
            f"**Number of Judges:** {len(reports)}\n"
        )

        # Critical issues
        if by_severity[CritiqueSeverity.CRITICAL]:
            summary_parts.append("\n### [CRITICAL] Critical Issues\n")
            for r in by_severity[CritiqueSeverity.CRITICAL]:
                summary_parts.append(
                    f"- **{r.judge_name}**: {r.feedback[:200]}"
                )

        # High severity issues
        if by_severity[CritiqueSeverity.HIGH]:
            summary_parts.append("\n### [HIGH] High Priority Issues\n")
            for r in by_severity[CritiqueSeverity.HIGH]:
                summary_parts.append(
                    f"- **{r.judge_name}**: {r.feedback[:200]}"
                )

        # Judge-by-judge breakdown
        summary_parts.append("\n### Detailed Judge Feedback\n")
        for r in reports:
            approval_icon = "[PASS]" if r.is_approved else "[FAIL]"
            summary_parts.append(
                f"\n{approval_icon} **{r.judge_name}** ({r.judge_type.value})\n"
                f"   Score: {r.score:.2f} | Confidence: {r.confidence:.2f}\n"
                f"   Feedback: {r.feedback[:300]}...\n"
            )

        # Common themes
        themes = self._extract_common_themes(reports)
        if themes:
            summary_parts.append("\n### Common Themes\n")
            for theme in themes:
                summary_parts.append(f"- {theme}")

        # Join and truncate if necessary
        full_summary = "\n".join(summary_parts)
        if len(full_summary) > max_length:
            full_summary = full_summary[:max_length] + "\n... (truncated)"

        logger.debug(f"Generated summary with {len(full_summary)} characters")
        return full_summary

    def extract_improvements(
        self,
        reports: List[JudgeReport],
        max_improvements: int = 20
    ) -> List[str]:
        """
        Extract and consolidate improvements needed from all reports.

        This method:
        1. Collects all improvements from all reports
        2. Deduplicates similar improvements
        3. Prioritizes by severity and frequency
        4. Returns a consolidated list

        Args:
            reports: List of JudgeReport objects
            max_improvements: Maximum number of improvements to return

        Returns:
            List of consolidated improvement suggestions

        Raises:
            ValueError: If reports is empty
        """
        if not reports:
            raise ValueError("Cannot extract improvements: no reports provided")

        if not self.config.extract_improvements:
            logger.debug("Improvement extraction disabled in config")
            return []

        # Collect all improvements with metadata
        all_improvements = []
        for report in reports:
            for improvement in report.improvements:
                all_improvements.append({
                    "text": improvement,
                    "severity": report.severity,
                    "judge": report.judge_name,
                    "score": report.score
                })

        # Deduplicate using fuzzy matching
        deduplicated = self._deduplicate_improvements(all_improvements)

        # Sort by severity and frequency
        prioritized = self._prioritize_improvements(deduplicated)

        # Limit to max_improvements
        result = [imp["text"] for imp in prioritized[:max_improvements]]

        logger.info(f"Extracted {len(result)} improvements from {len(reports)} reports")
        return result

    def calculate_consensus(
        self,
        reports: List[JudgeReport]
    ) -> float:
        """
        Calculate the consensus score among judges.

        Consensus measures how much the judges agree with each other.
        A score of 1.0 means perfect agreement, 0.0 means no agreement.

        Uses the algorithm specified in config:
        - "std_dev": Based on standard deviation of scores
        - "mean_deviation": Based on mean absolute deviation
        - "pairwise_agreement": Based on pairwise agreement rates

        Args:
            reports: List of JudgeReport objects

        Returns:
            Consensus score from 0.0 to 1.0

        Raises:
            ValueError: If reports is empty
        """
        if not reports:
            raise ValueError("Cannot calculate consensus: no reports provided")

        if len(reports) == 1:
            # Single judge always has perfect consensus
            return 1.0

        algorithm = self.config.consensus_algorithm

        if algorithm == "std_dev":
            return self._consensus_by_std_dev(reports)
        elif algorithm == "mean_deviation":
            return self._consensus_by_mean_deviation(reports)
        elif algorithm == "pairwise_agreement":
            return self._consensus_by_pairwise_agreement(reports)
        else:
            logger.warning(f"Unknown consensus algorithm: {algorithm}, using std_dev")
            return self._consensus_by_std_dev(reports)

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _normalize_critiques(
        self,
        critiques: List[Union[JudgeReport, Dict[str, Any]]]
    ) -> List[JudgeReport]:
        """Convert dict critiques to JudgeReport objects."""
        normalized = []
        for critique in critiques:
            if isinstance(critique, JudgeReport):
                normalized.append(critique)
            elif isinstance(critique, dict):
                # Convert dict to JudgeReport
                try:
                    normalized.append(JudgeReport(
                        judge_name=critique["judge_name"],
                        judge_type=JudgeType(critique.get("judge_type", "ai_model")),
                        is_approved=critique.get("is_approved", False),
                        score=critique.get("score", 0.5),
                        feedback=critique.get("feedback", ""),
                        improvements=critique.get("improvements", []),
                        severity=CritiqueSeverity(critique.get("severity", "medium")),
                        confidence=critique.get("confidence", 1.0),
                        metrics=critique.get("metrics", {}),
                        metadata=critique.get("metadata", {})
                    ))
                except KeyError as e:
                    logger.error(f"Invalid critique dict, missing key: {e}")
                    raise ValueError(f"Invalid critique dict: {e}")
            else:
                raise TypeError(
                    f"Expected JudgeReport or dict, got {type(critique)}"
                )

        return normalized

    def _calculate_aggregate_score(
        self,
        reports: List[JudgeReport]
    ) -> float:
        """Calculate weighted average score from reports."""
        if not reports:
            return 0.0

        # Get weights from metadata if available
        weighted_scores = []
        total_weight = 0.0

        for report in reports:
            weight = report.metadata.get("applied_weight", 1.0)
            weighted_scores.append(report.score * weight)
            total_weight += weight

        if total_weight == 0:
            return statistics.mean(r.score for r in reports)

        return sum(weighted_scores) / total_weight

    def _handle_outliers(
        self,
        reports: List[JudgeReport]
    ) -> List[JudgeReport]:
        """Detect and handle outlier reports."""
        scores = [r.score for r in reports]
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        if std_dev == 0:
            return reports

        threshold = self.config.outlier_std_dev_threshold
        non_outliers = []

        for report in reports:
            z_score = abs(report.score - mean_score) / std_dev if std_dev > 0 else 0
            if z_score <= threshold:
                non_outliers.append(report)
            else:
                logger.warning(
                    f"Detected outlier report from {report.judge_name} "
                    f"(z-score: {z_score:.2f}, score: {report.score:.2f})"
                )

        return non_outliers if non_outliers else reports

    def _extract_common_themes(
        self,
        reports: List[JudgeReport]
    ) -> List[str]:
        """Extract common themes from multiple reports."""
        # Simple keyword-based theme extraction
        keywords = defaultdict(int)
        for report in reports:
            words = re.findall(r'\w+', report.feedback.lower())
            for word in words:
                if len(word) > 4:  # Only significant words
                    keywords[word] += 1

        # Get top 5 most common keywords
        top_keywords = sorted(
            keywords.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return [f"Theme: {kw} (mentioned {count} times)" for kw, count in top_keywords]

    def _deduplicate_improvements(
        self,
        improvements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate improvements using fuzzy matching."""
        # Simple deduplication based on exact text match
        # In production, you might use more sophisticated fuzzy matching
        seen = set()
        deduplicated = []

        for imp in improvements:
            text_lower = imp["text"].lower().strip()
            if text_lower not in seen:
                seen.add(text_lower)
                deduplicated.append(imp)

        return deduplicated

    def _prioritize_improvements(
        self,
        improvements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize improvements by severity and score impact."""
        severity_order = {
            CritiqueSeverity.CRITICAL: 0,
            CritiqueSeverity.HIGH: 1,
            CritiqueSeverity.MEDIUM: 2,
            CritiqueSeverity.LOW: 3,
            CritiqueSeverity.INFO: 4
        }

        return sorted(
            improvements,
            key=lambda x: (
                severity_order.get(x["severity"], 2),
                -x["score"]  # Lower scores get higher priority
            )
        )

    def _consensus_by_std_dev(
        self,
        reports: List[JudgeReport]
    ) -> float:
        """Calculate consensus using standard deviation."""
        scores = [r.score for r in reports]
        if len(scores) == 1:
            return 1.0

        std_dev = statistics.stdev(scores)
        # Convert std_dev to consensus (0 std_dev = 1.0 consensus, 0.5 std_dev = 0.0 consensus)
        consensus = max(0.0, 1.0 - (std_dev * 2))
        return consensus

    def _consensus_by_mean_deviation(
        self,
        reports: List[JudgeReport]
    ) -> float:
        """Calculate consensus using mean absolute deviation."""
        scores = [r.score for r in reports]
        if len(scores) == 1:
            return 1.0

        mean_score = statistics.mean(scores)
        mad = statistics.mean(abs(s - mean_score) for s in scores)
        # Convert MAD to consensus
        consensus = max(0.0, 1.0 - (mad * 2))
        return consensus

    def _consensus_by_pairwise_agreement(
        self,
        reports: List[JudgeReport]
    ) -> float:
        """Calculate consensus using pairwise agreement rates."""
        if len(reports) == 1:
            return 1.0

        agreements = 0
        total_pairs = 0

        for i, r1 in enumerate(reports):
            for r2 in reports[i+1:]:
                total_pairs += 1
                # Judges agree if both approve or both reject
                if r1.is_approved == r2.is_approved:
                    agreements += 1

        if total_pairs == 0:
            return 1.0

        return agreements / total_pairs


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_judge_reports(
    num_reports: int = 3,
    approval_rate: float = 0.7
) -> List[JudgeReport]:
    """
    Create sample judge reports for testing.

    Args:
        num_reports: Number of reports to generate
        approval_rate: Proportion of reports that should approve

    Returns:
        List of sample JudgeReport objects
    """
    reports = []
    judge_types = list(JudgeType)
    severities = list(CritiqueSeverity)

    sample_feedback = [
        "The solution is well-structured and follows best practices.",
        "Code quality is good but needs better error handling.",
        "Performance can be improved by optimizing the algorithm.",
        "Security concerns: input validation is insufficient.",
        "Excellent implementation, very clean and maintainable."
    ]

    sample_improvements = [
        "Add comprehensive error handling",
        "Optimize database queries",
        "Implement input sanitization",
        "Add unit tests",
        "Improve code documentation",
        "Refactor for better readability",
        "Add logging for debugging"
    ]

    for i in range(num_reports):
        is_approved = i < int(num_reports * approval_rate)
        score = 0.9 if is_approved else 0.4

        reports.append(JudgeReport(
            judge_name=f"judge_{i+1}",
            judge_type=judge_types[i % len(judge_types)],
            is_approved=is_approved,
            score=score,
            feedback=sample_feedback[i % len(sample_feedback)],
            improvements=[
                sample_improvements[(i + j) % len(sample_improvements)]
                for j in range(2)
            ],
            severity=severities[i % len(severities)],
            confidence=0.8 + (i * 0.02),
            metrics={"execution_time": 1.5 + i * 0.1}
        ))

    return reports


def export_critique_report(
    report: CritiqueReport,
    filepath: str,
    format: str = "json"
) -> None:
    """
    Export a critique report to a file.

    Args:
        report: The CritiqueReport to export
        filepath: Path to output file
        format: Export format ("json" or "txt")

    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written
    """
    try:
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Critique Report\n")
                f.write(f"{'='*50}\n")
                f.write(f"Solution ID: {report.solution_attempt_id}\n")
                f.write(f"Gauntlet: {report.gauntlet_name}\n")
                f.write(f"Approved: {report.is_approved}\n")
                f.write(f"Score: {report.aggregate_score:.2f}\n")
                f.write(f"Consensus: {report.consensus_score:.2f}\n")
                f.write(f"\nSummary:\n{report.summary}\n")
                f.write(f"\nImprovements Needed:\n")
                for imp in report.improvements_needed:
                    f.write(f"  - {imp}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported critique report to {filepath}")
    except (OSError, IOError, TypeError, ValueError) as e:
        logger.error(f"Failed to export critique report: {e}")
        raise


def import_critique_report(
    filepath: str
) -> CritiqueReport:
    """
    Import a critique report from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        CritiqueReport object

    Raises:
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        report = CritiqueReport.from_dict(data)
        logger.info(f"Imported critique report from {filepath}")
        return report
    except (OSError, IOError, TypeError, ValueError) as e:
        logger.error(f"Failed to import critique report: {e}")
        raise


# =============================================================================
# UNIT TESTS
# =============================================================================

import unittest
from datetime import datetime


class TestCritiqueAggregator(unittest.TestCase):
    """Unit tests for CritiqueAggregator."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = CritiqueAggregator()
        self.sample_reports = create_sample_judge_reports(
            num_reports=5,
            approval_rate=0.8
        )

    def test_create_critique_report_basic(self):
        """Test basic critique report creation."""
        report = self.aggregator.create_critique_report(
            solution_id="test_solution",
            gauntlet_name="test_gauntlet",
            critiques=self.sample_reports[:3]
        )

        self.assertIsNotNone(report)
        self.assertEqual(report.solution_attempt_id, "test_solution")
        self.assertEqual(report.gauntlet_name, "test_gauntlet")
        self.assertEqual(len(report.reports_by_judge), 3)

    def test_create_critique_report_with_weights(self):
        """Test critique report creation with custom weights."""
        weights = {"judge_1": 1.0, "judge_2": 0.5, "judge_3": 0.8}

        report = self.aggregator.create_critique_report(
            solution_id="test_solution",
            gauntlet_name="test_gauntlet",
            critiques=self.sample_reports[:3],
            weights=weights
        )

        # Check that weights were applied
        for r in report.reports_by_judge:
            self.assertIn("applied_weight", r.metadata)

    def test_calculate_approval_unanimous(self):
        """Test approval calculation with unanimous approval."""
        approved_reports = [
            JudgeReport(
                judge_name=f"judge_{i}",
                judge_type=JudgeType.AI_MODEL,
                is_approved=True,
                score=0.9,
                feedback="Good"
            )
            for i in range(3)
        ]

        is_approved = self.aggregator.calculate_approval(approved_reports, 0.7)
        self.assertTrue(is_approved)

    def test_calculate_approval_rejection(self):
        """Test approval calculation with rejection."""
        rejected_reports = [
            JudgeReport(
                judge_name=f"judge_{i}",
                judge_type=JudgeType.AI_MODEL,
                is_approved=False,
                score=0.3,
                feedback="Poor"
            )
            for i in range(3)
        ]

        is_approved = self.aggregator.calculate_approval(rejected_reports, 0.7)
        self.assertFalse(is_approved)

    def test_calculate_approval_critical_severity(self):
        """Test that critical severity overrides approval."""
        reports = [
            JudgeReport(
                judge_name="judge_1",
                judge_type=JudgeType.SECURITY_SCANNER,
                is_approved=False,
                score=0.1,
                feedback="Critical security flaw",
                severity=CritiqueSeverity.CRITICAL
            ),
            JudgeReport(
                judge_name="judge_2",
                judge_type=JudgeType.AI_MODEL,
                is_approved=True,
                score=0.9,
                feedback="Good otherwise"
            )
        ]

        is_approved = self.aggregator.calculate_approval(reports, 0.7)
        self.assertFalse(is_approved)

    def test_generate_summary(self):
        """Test summary generation."""
        summary = self.aggregator.generate_summary(self.sample_reports)

        self.assertIsNotNone(summary)
        self.assertIn("Critique Summary", summary)
        self.assertIn("Overall Score", summary)
        self.assertIn("Detailed Judge Feedback", summary)

    def test_extract_improvements(self):
        """Test improvement extraction."""
        improvements = self.aggregator.extract_improvements(self.sample_reports)

        self.assertIsInstance(improvements, list)
        self.assertGreater(len(improvements), 0)

    def test_calculate_consensus(self):
        """Test consensus calculation."""
        # High consensus (similar scores)
        high_consensus_reports = [
            JudgeReport(
                judge_name=f"judge_{i}",
                judge_type=JudgeType.AI_MODEL,
                is_approved=True,
                score=0.8 + (i * 0.01),  # Very similar scores
                feedback="Good"
            )
            for i in range(5)
        ]

        consensus = self.aggregator.calculate_consensus(high_consensus_reports)
        self.assertGreater(consensus, 0.5)

        # Low consensus (divergent scores)
        low_consensus_reports = [
            JudgeReport(
                judge_name=f"judge_{i}",
                judge_type=JudgeType.AI_MODEL,
                is_approved=True if i < 3 else False,
                score=0.9 if i < 3 else 0.2,  # Very different scores
                feedback="Varying opinions"
            )
            for i in range(6)
        ]

        consensus = self.aggregator.calculate_consensus(low_consensus_reports)
        self.assertLess(consensus, 0.5)

    def test_aggregate_with_empty_reports(self):
        """Test error handling for empty reports."""
        with self.assertRaises(ValueError):
            self.aggregator.create_critique_report(
                solution_id="test",
                gauntlet_name="test",
                critiques=[]
            )

    def test_report_serialization(self):
        """Test report to_dict and from_dict."""
        report = self.aggregator.create_critique_report(
            solution_id="test_solution",
            gauntlet_name="test_gauntlet",
            critiques=self.sample_reports[:2]
        )

        # Convert to dict and back
        report_dict = report.to_dict()
        restored_report = CritiqueReport.from_dict(report_dict)

        self.assertEqual(
            restored_report.solution_attempt_id,
            report.solution_attempt_id
        )
        self.assertEqual(
            restored_report.gauntlet_name,
            report.gauntlet_name
        )
        self.assertEqual(
            len(restored_report.reports_by_judge),
            len(report.reports_by_judge)
        )

    def test_invalid_score_validation(self):
        """Test that invalid scores are rejected."""
        with self.assertRaises(ValueError):
            JudgeReport(
                judge_name="test",
                judge_type=JudgeType.AI_MODEL,
                is_approved=True,
                score=1.5,  # Invalid score > 1.0
                feedback="Test"
            )

    def test_invalid_threshold_validation(self):
        """Test that invalid thresholds are rejected."""
        with self.assertRaises(ValueError):
            self.aggregator.calculate_approval(
                self.sample_reports,
                threshold=1.5  # Invalid threshold > 1.0
            )


class TestJudgeReport(unittest.TestCase):
    """Unit tests for JudgeReport."""

    def test_judge_report_creation(self):
        """Test JudgeReport creation."""
        report = JudgeReport(
            judge_name="test_judge",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.85,
            feedback="Good work"
        )

        self.assertEqual(report.judge_name, "test_judge")
        self.assertTrue(report.is_approved)
        self.assertEqual(report.score, 0.85)

    def test_judge_report_with_improvements(self):
        """Test JudgeReport with improvements list."""
        improvements = ["Fix bug", "Add tests"]
        report = JudgeReport(
            judge_name="test_judge",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.85,
            feedback="Good work",
            improvements=improvements
        )

        self.assertEqual(len(report.improvements), 2)
        self.assertIn("Fix bug", report.improvements)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_basic_usage():
    """Example of basic CritiqueAggregator usage."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)

    # Create aggregator
    aggregator = CritiqueAggregator()

    # Create judge reports
    judge_reports = [
        JudgeReport(
            judge_name="gpt-4",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.85,
            feedback="The solution is well-structured and follows best practices.",
            improvements=["Add more comprehensive error handling"],
            severity=CritiqueSeverity.MEDIUM,
            confidence=0.9
        ),
        JudgeReport(
            judge_name="security_scanner",
            judge_type=JudgeType.SECURITY_SCANNER,
            is_approved=True,
            score=0.9,
            feedback="No security vulnerabilities detected.",
            improvements=[],
            severity=CritiqueSeverity.INFO,
            confidence=1.0
        ),
        JudgeReport(
            judge_name="human_reviewer_1",
            judge_type=JudgeType.HUMAN,
            is_approved=False,
            score=0.65,
            feedback="Good implementation but needs better documentation.",
            improvements=["Add docstrings to all functions", "Improve variable naming"],
            severity=CritiqueSeverity.HIGH,
            confidence=0.8
        )
    ]

    # Create critique report
    critique_report = aggregator.create_critique_report(
        solution_id="solution_abc123",
        gauntlet_name="red_team_gauntlet",
        critiques=judge_reports
    )

    # Print results
    print(f"\nSolution ID: {critique_report.solution_attempt_id}")
    print(f"Gauntlet: {critique_report.gauntlet_name}")
    print(f"Approved: {critique_report.is_approved}")
    print(f"Aggregate Score: {critique_report.aggregate_score:.2f}")
    print(f"Consensus: {critique_report.consensus_score:.2f}")
    print(f"\nSummary:\n{critique_report.summary}")
    print(f"\nImprovements Needed:")
    for imp in critique_report.improvements_needed:
        print(f"  - {imp}")

    return critique_report


def example_with_custom_weights():
    """Example with custom judge weights."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Weights")
    print("=" * 60)

    # Create aggregator with custom config
    config = AggregationConfig(
        default_approval_threshold=0.75,
        min_judges_required=2,
        enable_outlier_detection=True
    )
    aggregator = CritiqueAggregator(config)

    # Create judge reports
    judge_reports = [
        JudgeReport(
            judge_name="gpt-3.5-turbo",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.8,
            feedback="Acceptable solution"
        ),
        JudgeReport(
            judge_name="human_expert",
            judge_type=JudgeType.HUMAN,
            is_approved=False,
            score=0.6,
            feedback="Needs more work",
            improvements=["Refactor complex logic"]
        )
    ]

    # Apply custom weights (human expert gets more weight)
    custom_weights = {
        "human_expert": 1.5,
        "gpt-3.5-turbo": 0.7
    }

    critique_report = aggregator.create_critique_report(
        solution_id="solution_def456",
        gauntlet_name="gold_team_gauntlet",
        critiques=judge_reports,
        weights=custom_weights
    )

    print(f"\nWeighted Aggregate Score: {critique_report.aggregate_score:.2f}")
    print(f"Approved: {critique_report.is_approved}")

    return critique_report


def example_export_and_import():
    """Example of exporting and importing critique reports."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Export and Import")
    print("=" * 60)

    aggregator = CritiqueAggregator()
    report = example_basic_usage()

    # Export to JSON
    export_path = "/tmp/critique_report.json"
    export_critique_report(report, export_path, format="json")
    print(f"\nExported report to: {export_path}")

    # Import back
    imported_report = import_critique_report(export_path)
    print(f"Imported report for solution: {imported_report.solution_attempt_id}")

    return imported_report


def example_with_automated_tests():
    """Example including automated test results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: With Automated Tests")
    print("=" * 60)

    aggregator = CritiqueAggregator()

    judge_reports = [
        JudgeReport(
            judge_name="pytest",
            judge_type=JudgeType.AUTOMATED_TEST,
            is_approved=True,
            score=1.0,
            feedback="All tests passed (15/15)",
            metrics={"tests_run": 15, "tests_passed": 15, "coverage": 0.92}
        ),
        JudgeReport(
            judge_name=" pylint",
            judge_type=JudgeType.LINTING_TOOL,
            is_approved=False,
            score=0.7,
            feedback="Code style issues detected",
            improvements=["Fix line length violations", "Remove unused imports"],
            severity=CritiqueSeverity.LOW
        ),
        JudgeReport(
            judge_name="performance_analyzer",
            judge_type=JudgeType.PERFORMANCE_ANALYZER,
            is_approved=True,
            score=0.85,
            feedback="Performance is acceptable",
            metrics={"avg_response_time": 120, "throughput": 1000}
        )
    ]

    critique_report = aggregator.create_critique_report(
        solution_id="solution_xyz789",
        gauntlet_name="comprehensive_gauntlet",
        critiques=judge_reports
    )

    print(f"\nSolution Approved: {critique_report.is_approved}")
    print(f"Score: {critique_report.aggregate_score:.2f}")
    print(f"Test Metrics: {judge_reports[0].metrics}")

    return critique_report


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_with_custom_weights()
    example_export_and_import()
    example_with_automated_tests()

    # Run unit tests
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    unittest.main(argv=[''], exit=False, verbosity=2)
