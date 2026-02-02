"""
Quality Gate Engine for OpenEvolve Evaluator Team

This module implements the comprehensive quality gate system that acts as the final
quality control checkpoint before solutions are assembled into the final output.

Key Features:
- QualityThreshold: Define and manage quality thresholds
- QualityGateDecision: Pass/fail decisions with rationale
- QualityGateEngine: Main quality gate evaluation engine
- MultiStageValidation: Multi-stage validation workflow
- ConsensusBuilder: Aggregate multiple evaluator opinions
- Appeal and re-evaluation workflow

Architecture:
    Blue Team Solutions → Quality Gate → Solution Integration (if PASSED)
                                           ↓
                                      Appeal (if FAILED/CONDITIONAL)

Author: OpenEvolve
Version: 1.0.0
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict
import statistics
import re

from evaluator_team import (
    EvaluationMetric,
    EvaluationScore,
    EvaluatorAssessment,
    EvaluationCriterion,
    EvaluationConfidence
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class GateDecision(Enum):
    """Quality gate decision types"""
    PASS = "pass"  # Solution meets quality standards
    FAIL = "fail"  # Solution does not meet quality standards
    CONDITIONAL_PASS = "conditional_pass"  # Pass with minor issues
    DEFERRED = "deferred"  # Requires additional review
    APPEAL_PENDING = "appeal_pending"  # Awaiting appeal decision


class QualityLevel(Enum):
    """Quality levels for thresholds"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"
    EXCEPTIONAL = "exceptional"


class ContentType(Enum):
    """Types of content for specialized thresholds"""
    CODE = "code"
    DOCUMENT = "document"
    PROTOCOL = "protocol"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    GENERAL = "general"


class ConsensusMethod(Enum):
    """Methods for building consensus from multiple evaluators"""
    MAJORITY_VOTE = "majority_vote"  # Simple majority
    WEIGHTED_VOTE = "weighted_vote"  # Weighted by expertise
    EXPERTISE_WEIGHTED = "expertise_weighted"  # Expertise-weighted consensus
    BAYESIAN_AGGREGATION = "bayesian_aggregation"  # Bayesian belief aggregation
    MEDIAN = "median"  # Median score
    TRIMMED_MEAN = "trimmed_mean"  # Trimmed mean (remove outliers)


@dataclass
class QualityThreshold:
    """Quality threshold configuration"""
    content_type: ContentType
    quality_level: QualityLevel
    min_overall_score: float  # 0-100 scale
    min_correctness: float = 70.0
    min_completeness: float = 70.0
    min_clarity: float = 65.0
    min_effectiveness: float = 65.0
    min_efficiency: float = 60.0
    min_maintainability: float = 60.0
    min_security: float = 75.0  # For code
    min_compliance: float = 70.0  # For legal/medical
    required_metrics: List[EvaluationMetric] = field(default_factory=list)
    adaptive_thresholds: bool = False  # Adjust thresholds based on complexity
    complexity_modifier: float = 0.0  # Threshold adjustment per complexity point

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['content_type'] = self.content_type.value
        data['quality_level'] = self.quality_level.value
        return data


@dataclass
class QualityGateReport:
    """Report from quality gate evaluation"""
    decision: GateDecision
    overall_score: float
    threshold_used: QualityThreshold
    rationale: str
    improvement_recommendations: List[str]
    critical_issues: List[str]
    minor_issues: List[str]
    scores_by_metric: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['decision'] = self.decision.value
        data['threshold_used'] = self.threshold_used.to_dict()
        return data


@dataclass
class AppealRequest:
    """Request to appeal a quality gate decision"""
    solution_id: str
    original_decision: GateDecision
    appeal_reason: str
    additional_context: str
    requested_revaluation: List[EvaluationMetric]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, approved, rejected


@dataclass
class AppealDecision:
    """Decision on an appeal request"""
    appeal_request: AppealRequest
    new_decision: GateDecision
    decision_rationale: str
    reviewer_comments: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConsensusResult:
    """Result of consensus building"""
    consensus_score: float
    agreement_level: float  # 0-1 scale
    disagreement_details: Dict[str, Any]
    method_used: ConsensusMethod
    participant_count: int
    outlier_evaluators: List[str] = field(default_factory=list)
    confidence: EvaluationConfidence = EvaluationConfidence.MODERATE
    rationale: str = ""


@dataclass
class MultiStageValidationResult:
    """Result from multi-stage validation"""
    pre_evaluation_passed: bool
    comprehensive_evaluation_passed: bool
    post_evaluation_passed: bool
    final_decision: GateDecision
    stage_reports: List[Dict[str, Any]]
    total_time: float
    appeals: List[AppealDecision] = field(default_factory=list)


# =============================================================================
# QUALITY THRESHOLD MANAGER
# =============================================================================

class QualityThresholdManager:
    """Manages quality thresholds for different content types and quality levels"""

    def __init__(self):
        self.thresholds: Dict[Tuple[ContentType, QualityLevel], QualityThreshold] = {}
        self._initialize_default_thresholds()

    def _initialize_default_thresholds(self):
        """Initialize default quality thresholds"""
        # Code thresholds
        self.thresholds[(ContentType.CODE, QualityLevel.MINIMAL)] = QualityThreshold(
            content_type=ContentType.CODE,
            quality_level=QualityLevel.MINIMAL,
            min_overall_score=60.0,
            min_correctness=60.0,
            min_completeness=60.0,
            min_clarity=55.0,
            min_effectiveness=55.0,
            min_efficiency=50.0,
            min_maintainability=50.0,
            min_security=60.0,
        )

        self.thresholds[(ContentType.CODE, QualityLevel.STANDARD)] = QualityThreshold(
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            min_overall_score=75.0,
            min_correctness=75.0,
            min_completeness=75.0,
            min_clarity=70.0,
            min_effectiveness=70.0,
            min_efficiency=65.0,
            min_maintainability=65.0,
            min_security=75.0,
        )

        self.thresholds[(ContentType.CODE, QualityLevel.HIGH)] = QualityThreshold(
            content_type=ContentType.CODE,
            quality_level=QualityLevel.HIGH,
            min_overall_score=85.0,
            min_correctness=85.0,
            min_completeness=85.0,
            min_clarity=80.0,
            min_effectiveness=80.0,
            min_efficiency=75.0,
            min_maintainability=75.0,
            min_security=85.0,
        )

        self.thresholds[(ContentType.CODE, QualityLevel.EXCEPTIONAL)] = QualityThreshold(
            content_type=ContentType.CODE,
            quality_level=QualityLevel.EXCEPTIONAL,
            min_overall_score=95.0,
            min_correctness=95.0,
            min_completeness=95.0,
            min_clarity=90.0,
            min_effectiveness=90.0,
            min_efficiency=85.0,
            min_maintainability=85.0,
            min_security=95.0,
        )

        # Document thresholds
        self.thresholds[(ContentType.DOCUMENT, QualityLevel.MINIMAL)] = QualityThreshold(
            content_type=ContentType.DOCUMENT,
            quality_level=QualityLevel.MINIMAL,
            min_overall_score=60.0,
            min_correctness=60.0,
            min_completeness=60.0,
            min_clarity=65.0,
            min_effectiveness=55.0,
            min_efficiency=50.0,
            min_maintainability=50.0,
        )

        self.thresholds[(ContentType.DOCUMENT, QualityLevel.STANDARD)] = QualityThreshold(
            content_type=ContentType.DOCUMENT,
            quality_level=QualityLevel.STANDARD,
            min_overall_score=75.0,
            min_correctness=75.0,
            min_completeness=75.0,
            min_clarity=80.0,
            min_effectiveness=70.0,
            min_efficiency=60.0,
            min_maintainability=60.0,
        )

        self.thresholds[(ContentType.DOCUMENT, QualityLevel.HIGH)] = QualityThreshold(
            content_type=ContentType.DOCUMENT,
            quality_level=QualityLevel.HIGH,
            min_overall_score=85.0,
            min_correctness=85.0,
            min_completeness=85.0,
            min_clarity=90.0,
            min_effectiveness=80.0,
            min_efficiency=70.0,
            min_maintainability=70.0,
        )

        self.thresholds[(ContentType.DOCUMENT, QualityLevel.EXCEPTIONAL)] = QualityThreshold(
            content_type=ContentType.DOCUMENT,
            quality_level=QualityLevel.EXCEPTIONAL,
            min_overall_score=95.0,
            min_correctness=95.0,
            min_completeness=95.0,
            min_clarity=95.0,
            min_effectiveness=90.0,
            min_efficiency=80.0,
            min_maintainability=80.0,
        )

        # Technical content thresholds
        self.thresholds[(ContentType.TECHNICAL, QualityLevel.STANDARD)] = QualityThreshold(
            content_type=ContentType.TECHNICAL,
            quality_level=QualityLevel.STANDARD,
            min_overall_score=75.0,
            min_correctness=80.0,
            min_completeness=75.0,
            min_clarity=70.0,
            min_effectiveness=75.0,
            min_efficiency=65.0,
            min_maintainability=70.0,
        )

        # Legal content thresholds
        self.thresholds[(ContentType.LEGAL, QualityLevel.STANDARD)] = QualityThreshold(
            content_type=ContentType.LEGAL,
            quality_level=QualityLevel.STANDARD,
            min_overall_score=80.0,
            min_correctness=85.0,
            min_completeness=80.0,
            min_clarity=80.0,
            min_effectiveness=75.0,
            min_compliance=85.0,
        )

        # Medical content thresholds
        self.thresholds[(ContentType.MEDICAL, QualityLevel.STANDARD)] = QualityThreshold(
            content_type=ContentType.MEDICAL,
            quality_level=QualityLevel.STANDARD,
            min_overall_score=85.0,
            min_correctness=90.0,
            min_completeness=85.0,
            min_clarity=80.0,
            min_effectiveness=80.0,
            min_compliance=90.0,
            min_security=85.0,
        )

        # General content thresholds
        self.thresholds[(ContentType.GENERAL, QualityLevel.STANDARD)] = QualityThreshold(
            content_type=ContentType.GENERAL,
            quality_level=QualityLevel.STANDARD,
            min_overall_score=70.0,
            min_correctness=70.0,
            min_completeness=70.0,
            min_clarity=70.0,
            min_effectiveness=70.0,
        )

    def get_threshold(
        self,
        content_type: ContentType,
        quality_level: QualityLevel
    ) -> Optional[QualityThreshold]:
        """Get threshold for content type and quality level"""
        return self.thresholds.get((content_type, quality_level))

    def set_threshold(self, threshold: QualityThreshold):
        """Set a custom threshold"""
        self.thresholds[(threshold.content_type, threshold.quality_level)] = threshold

    def get_all_thresholds(self) -> List[QualityThreshold]:
        """Get all configured thresholds"""
        return list(self.thresholds.values())

    def adjust_for_complexity(
        self,
        threshold: QualityThreshold,
        complexity_score: int  # 1-10 scale
    ) -> QualityThreshold:
        """Adjust threshold based on problem complexity"""
        if not threshold.adaptive_thresholds:
            return threshold

        # Higher complexity = more lenient thresholds
        adjustment = (complexity_score - 5) * threshold.complexity_modifier

        # Create adjusted threshold
        adjusted = QualityThreshold(
            content_type=threshold.content_type,
            quality_level=threshold.quality_level,
            min_overall_score=max(0, threshold.min_overall_score - adjustment),
            min_correctness=max(0, threshold.min_correctness - adjustment),
            min_completeness=max(0, threshold.min_completeness - adjustment),
            min_clarity=max(0, threshold.min_clarity - adjustment),
            min_effectiveness=max(0, threshold.min_effectiveness - adjustment),
            min_efficiency=max(0, threshold.min_efficiency - adjustment),
            min_maintainability=max(0, threshold.min_maintainability - adjustment),
            min_security=max(0, threshold.min_security - adjustment) if hasattr(threshold, 'min_security') else 70.0,
            min_compliance=max(0, threshold.min_compliance - adjustment) if hasattr(threshold, 'min_compliance') else 70.0,
            required_metrics=threshold.required_metrics.copy(),
            adaptive_thresholds=True,
            complexity_modifier=threshold.complexity_modifier
        )

        return adjusted


# =============================================================================
# QUALITY GATE ENGINE
# =============================================================================

class QualityGateEngine:
    """
    Main quality gate evaluation engine.

    Evaluates solutions against quality thresholds and makes pass/fail decisions.
    
    ICR Integration:
    - Stores evaluation patterns for learning
    - Predicts pass/fail probability before evaluation
    - Adapts thresholds based on historical outcomes
    - Learns from refinement results
    """

    def __init__(
        self, 
        threshold_manager: Optional[QualityThresholdManager] = None,
        icr_pattern_store: Optional[Dict[str, Any]] = None,
        enable_icr: bool = True
    ):
        self.threshold_manager = threshold_manager or QualityThresholdManager()
        self.evaluation_history: List[QualityGateReport] = []
        self.performance_metrics: Dict[str, Any] = {
            'total_evaluations': 0,
            'pass_count': 0,
            'fail_count': 0,
            'conditional_pass_count': 0,
            'average_score': 0.0,
            'average_time': 0.0
        }
        
        # ICR Integration: Pattern storage and learning
        self.enable_icr = enable_icr
        self.icr_pattern_store = icr_pattern_store or {
            'content_type_patterns': {},  # content_type -> pattern list
            'quality_level_patterns': {},  # quality_level -> pattern list
            'complexity_patterns': {},  # complexity_range -> pattern list
            'metric_patterns': {},  # metric_name -> {score_range: pass_rate}
            'refinement_history': [],  # Refinement outcomes
        }
        
        # ICR: Adaptive threshold adjustments
        self._adaptive_thresholds: Dict[str, float] = {}
        
        # ICR: Prediction cache
        self._prediction_cache: Dict[str, Dict] = {}
        
        # Z3 Formal Verification Integration
        self._z3_verifier = None
        try:
            from quality_gate_z3_verifier import get_z3_quality_gate_verifier
            self._z3_verifier = get_z3_quality_gate_verifier()
            logger.info("Z3 Quality Gate Verifier integrated")
        except ImportError:
            logger.debug("Z3 Quality Gate Verifier not available")

    def evaluate(
        self,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType = ContentType.GENERAL,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        complexity_score: int = 5,
        store_pattern: bool = True,
        solution_context: Optional[Dict[str, Any]] = None
    ) -> QualityGateReport:
        """
        Evaluate solution against quality gate.

        Args:
            assessments: List of evaluator assessments
            content_type: Type of content being evaluated
            quality_level: Required quality level
            complexity_score: Problem complexity (1-10) for adaptive thresholds
            store_pattern: Whether to store ICR pattern (default True)
            solution_context: Optional context about the solution for ICR learning

        Returns:
            QualityGateReport with decision and rationale
        """
        start_time = time.time()
        logger.info(f"Evaluating quality gate for content type: {content_type.value}, quality level: {quality_level.value}")

        # Get threshold
        threshold = self.threshold_manager.get_threshold(content_type, quality_level)
        if not threshold:
            logger.warning(f"No threshold found for {content_type.value}/{quality_level.value}, using standard")
            threshold = self.threshold_manager.get_threshold(content_type, QualityLevel.STANDARD)

        # Adjust for complexity if adaptive
        if threshold.adaptive_thresholds:
            threshold = self.threshold_manager.adjust_for_complexity(threshold, complexity_score)
            logger.info(f"Adjusted thresholds for complexity: {complexity_score}")

        # ICR: Apply adaptive threshold adjustments based on patterns
        if self.enable_icr:
            threshold = self.adapt_threshold(threshold, content_type, quality_level, complexity_score)

        # Aggregate scores from all assessments
        scores_by_metric = self._aggregate_scores(assessments)

        # Check against threshold
        passed, critical_issues, minor_issues = self._check_threshold(scores_by_metric, threshold)

        # Make decision
        decision = self._make_decision(passed, scores_by_metric, threshold)

        # Generate rationale
        rationale = self._generate_rationale(decision, scores_by_metric, threshold)

        # Generate improvement recommendations
        recommendations = self._generate_recommendations(
            decision, scores_by_metric, threshold, critical_issues, minor_issues
        )

        # Create report
        evaluation_time = time.time() - start_time
        report = QualityGateReport(
            decision=decision,
            overall_score=scores_by_metric.get('overall', 0.0),
            threshold_used=threshold,
            rationale=rationale,
            improvement_recommendations=recommendations,
            critical_issues=critical_issues,
            minor_issues=minor_issues,
            scores_by_metric=scores_by_metric,
            metadata={
                'evaluation_time': evaluation_time,
                'num_assessments': len(assessments),
                'complexity_score': complexity_score,
                'evaluator_ids': [a.evaluator_id for a in assessments]
            }
        )

        # Update performance metrics
        self._update_performance_metrics(report)

        # Store in history
        self.evaluation_history.append(report)
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]

        # ICR: Store pattern for learning
        if self.enable_icr and store_pattern:
            self.store_icr_pattern(assessments, report, solution_context)

        logger.info(f"Quality gate evaluation complete: {decision.value} (score: {report.overall_score:.2f})")
        return report

    def verify_with_z3(
        self,
        verification_type: str,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Run formal verification using Z3.
        
        Args:
            verification_type: Type of verification ("sop_safety", "performance", "security")
            config: Verification configuration
            
        Returns:
            Verification result dict or None if Z3 not available
        """
        if not self._z3_verifier:
            logger.warning("Z3 verifier not available")
            return None
        
        try:
            if verification_type == "sop_safety":
                result = self._z3_verifier.verify_sop_safety(
                    config.get("steps", []),
                    config.get("invariants", [])
                )
            elif verification_type == "performance":
                result = self._z3_verifier.verify_performance_guarantee(
                    config.get("specs", []),
                    config.get("system_model")
                )
            elif verification_type == "security":
                result = self._z3_verifier.verify_security_property(
                    config.get("property", {}),
                    config.get("threat_model")
                )
            else:
                logger.warning(f"Unknown verification type: {verification_type}")
                return None
            
            return result.to_dict() if hasattr(result, 'to_dict') else result
        except Exception as e:
            logger.error(f"Z3 verification failed: {e}")
            return None

    def _aggregate_scores(self, assessments: List[EvaluatorAssessment]) -> Dict[str, float]:
        """Aggregate scores from multiple assessments"""
        if not assessments:
            return {}

        # Collect scores by metric
        metric_scores: Dict[str, List[float]] = defaultdict(list)

        for assessment in assessments:
            for score in assessment.scores:
                metric_name = score.metric.value
                metric_scores[metric_name].append(score.score)

            # Also include composite score
            metric_scores['overall'].append(assessment.composite_score)

        # Calculate average for each metric
        aggregated = {}
        for metric, scores in metric_scores.items():
            if scores:
                aggregated[metric] = statistics.mean(scores)
            else:
                aggregated[metric] = 0.0

        return aggregated

    def _check_threshold(
        self,
        scores: Dict[str, float],
        threshold: QualityThreshold
    ) -> Tuple[bool, List[str], List[str]]:
        """Check if scores meet threshold requirements"""
        passed = True
        critical_issues = []
        minor_issues = []

        # Check overall score
        overall_score = scores.get('overall', 0.0)
        if overall_score < threshold.min_overall_score:
            passed = False
            critical_issues.append(
                f"Overall score ({overall_score:.1f}) below threshold ({threshold.min_overall_score})"
            )

        # Check specific metrics
        metric_checks = [
            ('correctness', threshold.min_correctness),
            ('completeness', threshold.min_completeness),
            ('clarity', threshold.min_clarity),
            ('effectiveness', threshold.min_effectiveness),
            ('efficiency', threshold.min_efficiency),
            ('maintainability', threshold.min_maintainability),
        ]

        # Add security/compliance if relevant
        if hasattr(threshold, 'min_security') and threshold.min_security > 0:
            metric_checks.append(('security', threshold.min_security))
        if hasattr(threshold, 'min_compliance') and threshold.min_compliance > 0:
            metric_checks.append(('compliance', threshold.min_compliance))

        for metric_name, min_score in metric_checks:
            if metric_name in scores:
                score = scores[metric_name]
                if score < min_score:
                    gap = min_score - score
                    if gap > 15:  # Large gap = critical
                        passed = False
                        critical_issues.append(
                            f"{metric_name.capitalize()} score ({score:.1f}) significantly below threshold ({min_score})"
                        )
                    else:  # Small gap = minor issue
                        minor_issues.append(
                            f"{metric_name.capitalize()} score ({score:.1f}) slightly below threshold ({min_score})"
                        )

        return passed, critical_issues, minor_issues

    def _make_decision(
        self,
        passed: bool,
        scores: Dict[str, float],
        threshold: QualityThreshold
    ) -> GateDecision:
        """Make final gate decision"""
        overall_score = scores.get('overall', 0.0)

        if passed:
            # Check if it's a clean pass or conditional
            score_gap = threshold.min_overall_score - overall_score
            if score_gap < -5:  # Well above threshold
                return GateDecision.PASS
            else:  # Just above threshold
                return GateDecision.CONDITIONAL_PASS
        else:
            # Check if it's close or far below
            score_gap = threshold.min_overall_score - overall_score
            if score_gap < 10:  # Close to threshold
                return GateDecision.CONDITIONAL_PASS
            else:  # Far below threshold
                return GateDecision.FAIL

    def _generate_rationale(
        self,
        decision: GateDecision,
        scores: Dict[str, float],
        threshold: QualityThreshold
    ) -> str:
        """Generate rationale for decision"""
        overall = scores.get('overall', 0.0)
        rationale_parts = [
            f"Decision: {decision.value.upper()}",
            f"Overall Score: {overall:.2f} / {threshold.min_overall_score:.2f}",
        ]

        # Add metric breakdown
        key_metrics = ['correctness', 'completeness', 'clarity', 'effectiveness']
        metric_summary = []
        for metric in key_metrics:
            if metric in scores:
                score = scores[metric]
                metric_summary.append(f"{metric.capitalize()}: {score:.1f}")

        if metric_summary:
            rationale_parts.append("Key Metrics: " + ", ".join(metric_summary))

        # Add decision-specific rationale
        if decision == GateDecision.PASS:
            rationale_parts.append("Solution meets all quality standards and is ready for integration.")
        elif decision == GateDecision.CONDITIONAL_PASS:
            rationale_parts.append("Solution meets quality standards but has minor issues that should be addressed.")
        elif decision == GateDecision.FAIL:
            rationale_parts.append("Solution does not meet quality standards and requires significant improvement.")

        return "\n".join(rationale_parts)

    def _generate_recommendations(
        self,
        decision: GateDecision,
        scores: Dict[str, float],
        threshold: QualityThreshold,
        critical_issues: List[str],
        minor_issues: List[str]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Add critical issue recommendations
        for issue in critical_issues:
            if "correctness" in issue.lower():
                recommendations.append("Review and fix correctness issues - verify solution accuracy")
            elif "completeness" in issue.lower():
                recommendations.append("Address missing components - ensure all requirements are met")
            elif "clarity" in issue.lower():
                recommendations.append("Improve clarity - enhance documentation and explanations")
            elif "effectiveness" in issue.lower():
                recommendations.append("Enhance effectiveness - ensure solution achieves objectives")
            elif "security" in issue.lower():
                recommendations.append("Address security vulnerabilities - implement proper security measures")
            elif "compliance" in issue.lower():
                recommendations.append("Ensure compliance - verify all regulatory requirements are met")

        # Add general recommendations based on decision
        if decision == GateDecision.CONDITIONAL_PASS:
            recommendations.append("Minor improvements recommended before final integration")
            recommendations.append("Consider addressing minor issues to improve overall quality")
        elif decision == GateDecision.FAIL:
            recommendations.append("Significant improvements required before re-evaluation")
            recommendations.append("Review all critical issues and make necessary corrections")

        # Add content-type specific recommendations
        if threshold.content_type == ContentType.CODE:
            if scores.get('maintainability', 100) < 70:
                recommendations.append("Improve code maintainability with better structure and documentation")
            if scores.get('efficiency', 100) < 65:
                recommendations.append("Optimize code for better performance and resource usage")
        elif threshold.content_type == ContentType.DOCUMENT:
            if scores.get('clarity', 100) < 70:
                recommendations.append("Enhance document clarity with better organization and language")
            if scores.get('completeness', 100) < 70:
                recommendations.append("Ensure all necessary sections and information are included")

        return recommendations

    def _update_performance_metrics(self, report: QualityGateReport):
        """Update performance tracking metrics"""
        self.performance_metrics['total_evaluations'] += 1

        if report.decision == GateDecision.PASS:
            self.performance_metrics['pass_count'] += 1
        elif report.decision == GateDecision.FAIL:
            self.performance_metrics['fail_count'] += 1
        elif report.decision == GateDecision.CONDITIONAL_PASS:
            self.performance_metrics['conditional_pass_count'] += 1

        # Update average score
        total = self.performance_metrics['total_evaluations']
        current_avg = self.performance_metrics['average_score']
        self.performance_metrics['average_score'] = (
            (current_avg * (total - 1) + report.overall_score) / total
        )

        # Update average time
        eval_time = report.metadata.get('evaluation_time', 0.0)
        current_time_avg = self.performance_metrics['average_time']
        self.performance_metrics['average_time'] = (
            (current_time_avg * (total - 1) + eval_time) / total
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()

    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_evaluations': 0,
            'pass_count': 0,
            'fail_count': 0,
            'conditional_pass_count': 0,
            'average_score': 0.0,
            'average_time': 0.0
        }

    # =========================================================================
    # ICR INTEGRATION METHODS
    # =========================================================================
    
    def predict_pass_probability(
        self,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType = ContentType.GENERAL,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        complexity_score: int = 5
    ) -> Dict[str, Any]:
        """
        Predict pass/fail probability before full evaluation using ICR patterns.
        
        Args:
            assessments: List of evaluator assessments
            content_type: Type of content
            quality_level: Required quality level
            complexity_score: Problem complexity (1-10)
            
        Returns:
            Dictionary with prediction details
        """
        if not self.enable_icr:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'ICR disabled'
            }
        
        logger.info(f"Predicting pass probability for {content_type.value}/{quality_level.value}")
        
        # Aggregate scores for prediction
        scores = self._aggregate_scores(assessments)
        overall_score = scores.get('overall', 0.0)
        
        # Get historical pattern for this content/quality/complexity
        pattern_key = f"{content_type.value}_{quality_level.value}_{complexity_score // 2}"
        historical_patterns = self.icr_pattern_store['content_type_patterns'].get(pattern_key, [])
        
        # Calculate predicted pass probability based on patterns
        if historical_patterns:
            # Use weighted average based on similar historical outcomes
            total_weight = 0.0
            weighted_pass_rate = 0.0
            
            for pattern in historical_patterns:
                # Weight by similarity of scores
                pattern_score = pattern.get('overall_score', 0.0)
                score_diff = abs(overall_score - pattern_score)
                weight = max(0.0, 1.0 - (score_diff / 100.0))  # Higher weight for closer scores
                
                pass_rate = pattern.get('pass_rate', 0.5)
                weighted_pass_rate += pass_rate * weight
                total_weight += weight
            
            if total_weight > 0:
                predicted_pass_prob = weighted_pass_rate / total_weight
            else:
                predicted_pass_prob = 0.5
        else:
            # Fallback: use score-based prediction
            threshold = self.threshold_manager.get_threshold(content_type, quality_level)
            if threshold:
                predicted_pass_prob = max(0.0, min(1.0, (overall_score - threshold.min_overall_score + 20) / 40))
            else:
                predicted_pass_prob = 0.5
        
        # Get metric-specific predictions
        metric_predictions = {}
        for metric, score in scores.items():
            if metric != 'overall':
                metric_patterns = self.icr_pattern_store['metric_patterns'].get(metric, {})
                
                # Find closest score range
                closest_range = None
                closest_rate = 0.5
                for score_range, pass_rate in metric_patterns.items():
                    try:
                        low, high = score_range.split('-')
                        low, high = float(low), float(high)
                        if low <= score <= high:
                            closest_range = score_range
                            closest_rate = pass_rate
                            break
                        elif abs(score - low) < abs(score - (high if closest_range is None else float(closest_range.split('-')[1]))):
                            closest_range = score_range
                            closest_rate = pass_rate
                    except:
                        pass
                
                metric_predictions[metric] = {
                    'score': score,
                    'predicted_pass_rate': closest_rate,
                    'range': closest_range
                }
        
        # Determine confidence based on amount of historical data
        pattern_count = len(historical_patterns)
        if pattern_count >= 20:
            confidence = 0.9
        elif pattern_count >= 10:
            confidence = 0.75
        elif pattern_count >= 5:
            confidence = 0.5
        else:
            confidence = 0.25
        
        # Predict likely decision
        if predicted_pass_prob >= 0.8:
            predicted_decision = 'pass'
        elif predicted_pass_prob >= 0.5:
            predicted_decision = 'conditional_pass'
        elif predicted_pass_prob >= 0.2:
            predicted_decision = 'conditional_pass'
        else:
            predicted_decision = 'fail'
        
        return {
            'prediction': predicted_decision,
            'pass_probability': predicted_pass_prob,
            'confidence': confidence,
            'estimated_score': overall_score,
            'metric_predictions': metric_predictions,
            'pattern_count': pattern_count,
            'recommended_threshold_adj': self._get_threshold_adjustment(content_type, quality_level, complexity_score)
        }
    
    def _get_threshold_adjustment(
        self, 
        content_type: ContentType, 
        quality_level: QualityLevel,
        complexity_score: int
    ) -> float:
        """Get recommended threshold adjustment based on ICR patterns"""
        if not self.enable_icr:
            return 0.0
        
        # Check if we have enough data to recommend adjustment
        pattern_key = f"{content_type.value}_{quality_level.value}_{complexity_score // 2}"
        patterns = self.icr_pattern_store['content_type_patterns'].get(pattern_key, [])
        
        if len(patterns) < 5:
            return 0.0
        
        # Calculate average score gap for passed vs failed evaluations
        passed_scores = [p['overall_score'] for p in patterns if p.get('passed', False)]
        failed_scores = [p['overall_score'] for p in patterns if not p.get('passed', False)]
        
        if not passed_scores or not failed_scores:
            return 0.0
        
        avg_pass = sum(passed_scores) / len(passed_scores)
        avg_fail = sum(failed_scores) / len(failed_scores)
        
        # If pass threshold is too high (avg_pass - avg_fail is small), recommend adjustment
        gap = avg_pass - avg_fail
        if gap < 10:  # Pass/fail scores are close - threshold might be too strict
            return -2.0  # Recommend lowering threshold by 2 points
        elif gap > 30:  # Pass/fail scores are far apart - threshold might be too lenient
            return 2.0  # Recommend raising threshold by 2 points
        
        return 0.0
    
    def store_icr_pattern(
        self,
        assessments: List[EvaluatorAssessment],
        report: QualityGateReport,
        solution_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store evaluation pattern for ICR learning.
        
        Args:
            assessments: Evaluator assessments used
            report: Quality gate report
            solution_context: Optional context about the solution
        """
        if not self.enable_icr:
            return
        
        logger.info(f"Storing ICR pattern for {report.decision.value}")
        
        scores = self._aggregate_scores(assessments)
        threshold = report.threshold_used
        
        # Create pattern record
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': report.overall_score,
            'decision': report.decision.value,
            'passed': report.decision in [GateDecision.PASS, GateDecision.CONDITIONAL_PASS],
            'scores_by_metric': scores,
            'content_type': threshold.content_type.value,
            'quality_level': threshold.quality_level.value,
            'complexity_score': report.metadata.get('complexity_score', 5),
            'evaluation_time': report.metadata.get('evaluation_time', 0.0),
            'context': solution_context or {}
        }
        
        # Store by content_type/quality_level/complexity
        pattern_key = f"{threshold.content_type.value}_{threshold.quality_level.value}_{report.metadata.get('complexity_score', 5) // 2}"
        
        if pattern_key not in self.icr_pattern_store['content_type_patterns']:
            self.icr_pattern_store['content_type_patterns'][pattern_key] = []
        
        # Keep only last 100 patterns per key
        patterns = self.icr_pattern_store['content_type_patterns'][pattern_key]
        patterns.append(pattern)
        if len(patterns) > 100:
            patterns.pop(0)  # Remove oldest
        
        # Store by quality_level
        quality_key = threshold.quality_level.value
        if quality_key not in self.icr_pattern_store['quality_level_patterns']:
            self.icr_pattern_store['quality_level_patterns'][quality_key] = []
        self.icr_pattern_store['quality_level_patterns'][quality_key].append(pattern)
        
        # Store metric-specific patterns
        for metric, score in scores.items():
            if metric != 'overall':
                if metric not in self.icr_pattern_store['metric_patterns']:
                    self.icr_pattern_store['metric_patterns'][metric] = {}
                
                # Create score range bucket
                score_range = f"{int(score // 10) * 10}-{(int(score // 10) + 1) * 10}"
                
                if score_range not in self.icr_pattern_store['metric_patterns'][metric]:
                    self.icr_pattern_store['metric_patterns'][metric][score_range] = {
                        'total': 0,
                        'passed': 0
                    }
                
                # Update pass rate
                stats = self.icr_pattern_store['metric_patterns'][metric][score_range]
                stats['total'] += 1
                if report.decision in [GateDecision.PASS, GateDecision.CONDITIONAL_PASS]:
                    stats['passed'] += 1
        
        # Calculate pass rate for this pattern
        all_patterns = self.icr_pattern_store['content_type_patterns'].get(pattern_key, [])
        passed = sum(1 for p in all_patterns if p.get('passed', False))
        pattern['pass_rate'] = passed / len(all_patterns) if all_patterns else 0.5
        
        logger.info(f"ICR pattern stored: pass_rate={pattern['pass_rate']:.2%}")
    
    def learn_from_refinement(
        self,
        original_report: QualityGateReport,
        refined_report: QualityGateReport,
        refinement_type: str
    ) -> Dict[str, Any]:
        """
        Learn from refinement outcomes to improve future predictions.
        
        Args:
            original_report: Original quality gate report
            refined_report: Quality gate report after refinement
            refinement_type: Type of refinement applied
            
        Returns:
            Learning statistics
        """
        if not self.enable_icr:
            return {'learned': False}
        
        logger.info(f"Learning from refinement: {refinement_type}")
        
        # Record refinement outcome
        refinement_record = {
            'timestamp': datetime.now().isoformat(),
            'refinement_type': refinement_type,
            'original_decision': original_report.decision.value,
            'refined_decision': refined_report.decision.value,
            'original_score': original_report.overall_score,
            'refined_score': refined_report.overall_score,
            'score_improvement': refined_report.overall_score - original_report.overall_score,
            'content_type': original_report.threshold_used.content_type.value,
            'quality_level': original_report.threshold_used.quality_level.value
        }
        
        self.icr_pattern_store['refinement_history'].append(refinement_record)
        
        # Keep only last 200 refinement records
        if len(self.icr_pattern_store['refinement_history']) > 200:
            self.icr_pattern_store['refinement_history'] = self.icr_pattern_store['refinement_history'][-200:]
        
        # Calculate statistics
        ref_type_patterns = [
            r for r in self.icr_pattern_store['refinement_history']
            if r['refinement_type'] == refinement_type
        ]
        
        if ref_type_patterns:
            improvements = [r['score_improvement'] for r in ref_type_patterns]
            avg_improvement = sum(improvements) / len(improvements)
            success_count = sum(
                1 for r in ref_type_patterns 
                if r['refined_decision'] in ['pass', 'conditional_pass']
            )
            success_rate = success_count / len(ref_type_patterns)
        else:
            avg_improvement = 0.0
            success_rate = 0.0
        
        # Update adaptive thresholds based on successful refinements
        content_type = original_report.threshold_used.content_type
        quality_level = original_report.threshold_used.quality_level
        
        if success_rate > 0.7:
            # High success rate - this refinement type works well
            self._adaptive_thresholds[f"{content_type.value}_{quality_level.value}"] = \
                self._adaptive_thresholds.get(f"{content_type.value}_{quality_level.value}", 0) - 1
        elif success_rate < 0.3:
            # Low success rate - this refinement type might not be effective
            self._adaptive_thresholds[f"{content_type.value}_{quality_level.value}"] = \
                self._adaptive_thresholds.get(f"{content_type.value}_{quality_level.value}", 0) + 1
        
        result = {
            'learned': True,
            'refinement_type': refinement_type,
            'avg_score_improvement': avg_improvement,
            'success_rate': success_rate,
            'total_refinements': len(ref_type_patterns),
            'adaptive_threshold_adj': self._adaptive_thresholds.get(
                f"{content_type.value}_{quality_level.value}", 0
            )
        }
        
        logger.info(f"Refinement learning complete: success_rate={success_rate:.2%}")
        return result
    
    def get_icr_statistics(self) -> Dict[str, Any]:
        """Get ICR-related statistics"""
        if not self.enable_icr:
            return {'icr_enabled': False}
        
        total_patterns = sum(
            len(patterns) 
            for patterns in self.icr_pattern_store['content_type_patterns'].values()
        )
        
        # Calculate overall pass rate
        all_patterns = []
        for patterns in self.icr_pattern_store['content_type_patterns'].values():
            all_patterns.extend(patterns)
        
        passed = sum(1 for p in all_patterns if p.get('passed', False))
        overall_pass_rate = passed / len(all_patterns) if all_patterns else 0.0
        
        # Calculate refinement success rates
        refinement_stats = {}
        for record in self.icr_pattern_store['refinement_history']:
            ref_type = record['refinement_type']
            if ref_type not in refinement_stats:
                refinement_stats[ref_type] = {
                    'count': 0,
                    'total_improvement': 0.0,
                    'successes': 0
                }
            refinement_stats[ref_type]['count'] += 1
            refinement_stats[ref_type]['total_improvement'] += record['score_improvement']
            if record['refined_decision'] in ['pass', 'conditional_pass']:
                refinement_stats[ref_type]['successes'] += 1
        
        for ref_type, stats in refinement_stats.items():
            stats['avg_improvement'] = stats['total_improvement'] / stats['count']
            stats['success_rate'] = stats['successes'] / stats['count']
            del stats['total_improvement']
        
        return {
            'icr_enabled': True,
            'total_patterns': total_patterns,
            'overall_pass_rate': overall_pass_rate,
            'patterns_by_content_type': {
                key: len(patterns) 
                for key, patterns in self.icr_pattern_store['content_type_patterns'].items()
            },
            'refinement_statistics': refinement_stats,
            'adaptive_thresholds': self._adaptive_thresholds.copy()
        }
    
    def clear_icr_patterns(self) -> None:
        """Clear all stored ICR patterns"""
        if not self.enable_icr:
            return
        
        logger.info("Clearing all ICR patterns")
        
        self.icr_pattern_store = {
            'content_type_patterns': {},
            'quality_level_patterns': {},
            'complexity_patterns': {},
            'metric_patterns': {},
            'refinement_history': [],
        }
        self._adaptive_thresholds.clear()
        self._prediction_cache.clear()
    
    def adapt_threshold(
        self,
        threshold: QualityThreshold,
        content_type: ContentType,
        quality_level: QualityLevel,
        complexity_score: int
    ) -> QualityThreshold:
        """
        Adapt threshold based on ICR patterns and historical performance.
        
        Args:
            threshold: Original threshold
            content_type: Content type
            quality_level: Quality level
            complexity_score: Problem complexity
            
        Returns:
            Potentially adapted threshold
        """
        if not self.enable_icr:
            return threshold
        
        # Check for adaptive threshold adjustment
        adaptive_key = f"{content_type.value}_{quality_level.value}"
        adaptive_adj = self._adaptive_thresholds.get(adaptive_key, 0)
        
        # Check for complexity-based adjustment
        complexity_key = f"{content_type.value}_{quality_level.value}_{complexity_score // 2}"
        complexity_patterns = self.icr_pattern_store['content_type_patterns'].get(complexity_key, [])
        
        if len(complexity_patterns) >= 5:
            # Have enough data to adjust for complexity
            passed_patterns = [p for p in complexity_patterns if p.get('passed', False)]
            if passed_patterns:
                avg_pass_score = sum(p['overall_score'] for p in passed_patterns) / len(passed_patterns)
                # Adjust threshold to be more realistic based on actual pass scores
                score_gap = avg_pass_score - threshold.min_overall_score
                if score_gap < 5:  # Pass scores are close to threshold
                    # Make threshold slightly more lenient
                    complexity_adj = -2.0
                elif score_gap > 20:  # Pass scores are well above threshold
                    # Could make threshold slightly stricter
                    complexity_adj = 2.0
                else:
                    complexity_adj = 0.0
            else:
                complexity_adj = 0.0
        else:
            complexity_adj = 0.0
        
        # Combine adjustments
        total_adjustment = adaptive_adj + complexity_adj
        
        if total_adjustment == 0:
            return threshold
        
        # Create adapted threshold
        adjusted = QualityThreshold(
            content_type=threshold.content_type,
            quality_level=threshold.quality_level,
            min_overall_score=max(0, threshold.min_overall_score + total_adjustment),
            min_correctness=max(0, threshold.min_correctness + total_adjustment),
            min_completeness=max(0, threshold.min_completeness + total_adjustment),
            min_clarity=max(0, threshold.min_clarity + total_adjustment),
            min_effectiveness=max(0, threshold.min_effectiveness + total_adjustment),
            min_efficiency=max(0, threshold.min_efficiency + total_adjustment),
            min_maintainability=max(0, threshold.min_maintainability + total_adjustment),
            min_security=max(0, getattr(threshold, 'min_security', 70.0) + total_adjustment) if hasattr(threshold, 'min_security') else 70.0,
            min_compliance=max(0, getattr(threshold, 'min_compliance', 70.0) + total_adjustment) if hasattr(threshold, 'min_compliance') else 70.0,
            required_metrics=threshold.required_metrics.copy(),
            adaptive_thresholds=True,
            complexity_modifier=threshold.complexity_modifier
        )
        
        logger.info(f"Adapted threshold: {threshold.min_overall_score:.1f} -> {adjusted.min_overall_score:.1f} (adj={total_adjustment})")
        return adjusted


# =============================================================================
# MULTI-STAGE VALIDATION
# =============================================================================

class MultiStageValidation:
    """
    Multi-stage validation workflow.

    Stages:
    1. Pre-evaluation: Quick sanity checks
    2. Comprehensive evaluation: Full quality gate evaluation
    3. Post-evaluation: Final verification
    4. Appeal: Re-evaluation if requested
    """

    def __init__(self, quality_gate: QualityGateEngine):
        self.quality_gate = quality_gate
        self.validation_history: List[MultiStageValidationResult] = []

    def validate(
        self,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType = ContentType.GENERAL,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        complexity_score: int = 5,
        enable_appeals: bool = True
    ) -> MultiStageValidationResult:
        """
        Run complete multi-stage validation.

        Args:
            assessments: List of evaluator assessments
            content_type: Type of content
            quality_level: Required quality level
            complexity_score: Problem complexity
            enable_appeals: Whether to allow appeals

        Returns:
            MultiStageValidationResult with all stage results
        """
        start_time = time.time()
        logger.info("Starting multi-stage validation")

        stage_reports = []

        # Stage 1: Pre-evaluation
        logger.info("Stage 1: Pre-evaluation checks")
        pre_eval_passed, pre_eval_report = self._pre_evaluation_checks(
            assessments, content_type
        )
        stage_reports.append({
            'stage': 'pre_evaluation',
            'passed': pre_eval_passed,
            'report': pre_eval_report
        })

        if not pre_eval_passed:
            logger.warning("Pre-evaluation failed, aborting validation")
            return MultiStageValidationResult(
                pre_evaluation_passed=False,
                comprehensive_evaluation_passed=False,
                post_evaluation_passed=False,
                final_decision=GateDecision.FAIL,
                stage_reports=stage_reports,
                total_time=time.time() - start_time
            )

        # Stage 2: Comprehensive evaluation
        logger.info("Stage 2: Comprehensive quality gate evaluation")
        gate_report = self.quality_gate.evaluate(
            assessments, content_type, quality_level, complexity_score
        )
        comp_eval_passed = gate_report.decision in [GateDecision.PASS, GateDecision.CONDITIONAL_PASS]
        stage_reports.append({
            'stage': 'comprehensive_evaluation',
            'passed': comp_eval_passed,
            'report': gate_report.to_dict()
        })

        # Stage 3: Post-evaluation verification
        logger.info("Stage 3: Post-evaluation verification")
        post_eval_passed, post_eval_report = self._post_evaluation_verification(
            assessments, gate_report
        )
        stage_reports.append({
            'stage': 'post_evaluation',
            'passed': post_eval_passed,
            'report': post_eval_report
        })

        # Make final decision
        if pre_eval_passed and comp_eval_passed and post_eval_passed:
            final_decision = gate_report.decision
        else:
            final_decision = GateDecision.FAIL

        # Handle appeals
        appeals = []
        if enable_appeals and final_decision in [GateDecision.FAIL, GateDecision.CONDITIONAL_PASS]:
            logger.info("Stage 4: Processing appeals (if any)")
            # Appeals would be handled separately
            pass

        total_time = time.time() - start_time
        result = MultiStageValidationResult(
            pre_evaluation_passed=pre_eval_passed,
            comprehensive_evaluation_passed=comp_eval_passed,
            post_evaluation_passed=post_eval_passed,
            final_decision=final_decision,
            stage_reports=stage_reports,
            total_time=total_time,
            appeals=appeals
        )

        self.validation_history.append(result)
        logger.info(f"Multi-stage validation complete: {final_decision.value} in {total_time:.2f}s")
        return result

    def _pre_evaluation_checks(
        self,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform quick pre-evaluation sanity checks"""
        issues = []

        # Check if we have assessments
        if not assessments:
            issues.append("No evaluator assessments provided")
            return False, {'issues': issues, 'summary': 'No assessments to evaluate'}

        # Check minimum number of evaluators
        if len(assessments) < 1:
            issues.append("Insufficient evaluators (minimum 1 required)")

        # Check if all assessments have scores
        for assessment in assessments:
            if not assessment.scores:
                issues.append(f"Evaluator {assessment.evaluator_id} has no scores")

        # Check for obvious anomalies
        scores = [a.composite_score for a in assessments]
        if any(score < 0 or score > 100 for score in scores):
            issues.append("Scores outside valid range (0-100)")

        # Check for extreme disagreement
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            if score_range > 50:
                issues.append(f"Extreme evaluator disagreement (range: {score_range:.1f})")

        passed = len(issues) == 0
        summary = "Pre-evaluation passed" if passed else f"Pre-evaluation failed: {len(issues)} issues"

        return passed, {
            'issues': issues,
            'summary': summary,
            'num_assessments': len(assessments),
            'score_range': max(scores) - min(scores) if len(scores) > 1 else 0
        }

    def _post_evaluation_verification(
        self,
        assessments: List[EvaluatorAssessment],
        gate_report: QualityGateReport
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify the quality gate decision"""
        checks = []

        # Verify decision consistency with scores
        overall_score = gate_report.overall_score
        threshold = gate_report.threshold_used

        # Check if decision aligns with score
        if gate_report.decision == GateDecision.PASS:
            if overall_score < threshold.min_overall_score:
                checks.append("PASS decision but score below threshold")
        elif gate_report.decision == GateDecision.FAIL:
            if overall_score >= threshold.min_overall_score:
                checks.append("FAIL decision but score meets threshold")

        # Verify all evaluators were considered
        num_evaluators = gate_report.metadata.get('num_assessments', 0)
        if num_evaluators != len(assessments):
            checks.append(f"Evaluator count mismatch: {num_evaluators} vs {len(assessments)}")

        # Verify recommendations are present
        if not gate_report.improvement_recommendations:
            if gate_report.decision != GateDecision.PASS:
                checks.append("No recommendations for non-PASS decision")

        passed = len(checks) == 0
        summary = "Post-evaluation verified" if passed else f"Verification issues: {len(checks)}"

        return passed, {
            'checks': checks,
            'summary': summary,
            'decision_consistent': len(checks) == 0
        }

    def submit_appeal(
        self,
        original_result: MultiStageValidationResult,
        appeal_reason: str,
        additional_context: str,
        requested_revaluation: List[EvaluationMetric]
    ) -> AppealRequest:
        """Submit an appeal for re-evaluation"""
        solution_id = f"solution_{int(time.time())}"

        # Determine original decision
        original_decision = original_result.final_decision

        appeal = AppealRequest(
            solution_id=solution_id,
            original_decision=original_decision,
            appeal_reason=appeal_reason,
            additional_context=additional_context,
            requested_revaluation=requested_revaluation
        )

        logger.info(f"Appeal submitted for {solution_id}: {appeal_reason}")
        return appeal

    def process_appeal(
        self,
        appeal: AppealRequest,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType,
        quality_level: QualityLevel
    ) -> AppealDecision:
        """Process an appeal request"""
        logger.info(f"Processing appeal for {appeal.solution_id}")

        # Re-evaluate with additional context
        # In a real implementation, this would involve new evaluations or manual review
        new_report = self.quality_gate.evaluate(
            assessments, content_type, quality_level
        )

        # Make appeal decision
        # If new evaluation shows significant improvement, grant appeal
        # Otherwise, deny appeal
        if new_report.decision in [GateDecision.PASS, GateDecision.CONDITIONAL_PASS]:
            new_decision = new_report.decision
            rationale = "Appeal approved: Re-evaluation shows solution meets quality standards"
        else:
            new_decision = appeal.original_decision
            rationale = "Appeal denied: Re-evaluation confirms original decision"

        appeal_decision = AppealDecision(
            appeal_request=appeal,
            new_decision=new_decision,
            decision_rationale=rationale,
            reviewer_comments="Processed by quality gate engine"
        )

        logger.info(f"Appeal decision: {new_decision.value}")
        return appeal_decision


# =============================================================================
# CONSENSUS BUILDER
# =============================================================================

class ConsensusBuilder:
    """
    Build consensus from multiple evaluator opinions.

    Handles disagreements and generates consensus decisions.
    """

    def __init__(self):
        self.consensus_history: List[ConsensusResult] = []

    def build_consensus(
        self,
        assessments: List[EvaluatorAssessment],
        method: ConsensusMethod = ConsensusMethod.WEIGHTED_VOTE,
        criteria: Optional[List[EvaluationCriterion]] = None
    ) -> ConsensusResult:
        """
        Build consensus from multiple evaluator assessments.

        Args:
            assessments: List of evaluator assessments
            method: Consensus method to use
            criteria: Optional criteria for weighted methods

        Returns:
            ConsensusResult with consensus score and agreement details
        """
        logger.info(f"Building consensus using method: {method.value}")

        if not assessments:
            return ConsensusResult(
                consensus_score=0.0,
                agreement_level=0.0,
                disagreement_details={},
                method_used=method,
                participant_count=0,
                rationale="No assessments provided"
            )

        # Apply consensus method
        if method == ConsensusMethod.MAJORITY_VOTE:
            result = self._majority_vote_consensus(assessments)
        elif method == ConsensusMethod.WEIGHTED_VOTE:
            result = self._weighted_vote_consensus(assessments, criteria)
        elif method == ConsensusMethod.EXPERTISE_WEIGHTED:
            result = self._expertise_weighted_consensus(assessments)
        elif method == ConsensusMethod.BAYESIAN_AGGREGATION:
            result = self._bayesian_aggregation(assessments)
        elif method == ConsensusMethod.MEDIAN:
            result = self._median_consensus(assessments)
        elif method == ConsensusMethod.TRIMMED_MEAN:
            result = self._trimmed_mean_consensus(assessments)
        else:
            logger.warning(f"Unknown consensus method: {method}, using weighted vote")
            result = self._weighted_vote_consensus(assessments, criteria)

        result.method_used = method
        result.participant_count = len(assessments)
        result.rationale = self._generate_consensus_rationale(result, assessments)

        self.consensus_history.append(result)
        logger.info(f"Consensus built: score={result.consensus_score:.2f}, agreement={result.agreement_level:.2f}")

        return result

    def _majority_vote_consensus(self, assessments: List[EvaluatorAssessment]) -> ConsensusResult:
        """Simple majority vote consensus"""
        scores = [a.composite_score for a in assessments]

        # Calculate consensus as median
        consensus_score = statistics.median(scores)

        # Calculate agreement level (inverse of variance)
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        agreement_level = max(0.0, 1.0 - (variance / 1000.0))  # Normalize

        # Find outliers
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        outliers = [
            a.evaluator_id for a in assessments
            if abs(a.composite_score - mean_score) > 2 * std_dev
        ]

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            disagreement_details={
                'variance': variance,
                'std_dev': std_dev,
                'score_range': max(scores) - min(scores)
            },
            method_used=ConsensusMethod.MAJORITY_VOTE,
            participant_count=len(assessments),
            outlier_evaluators=outliers,
            confidence=self._determine_confidence(agreement_level),
            rationale=f"Majority vote with {len(scores)} participants"
        )

    def _weighted_vote_consensus(
        self,
        assessments: List[EvaluatorAssessment],
        criteria: Optional[List[EvaluationCriterion]]
    ) -> ConsensusResult:
        """Weighted vote consensus based on criterion importance"""
        if not criteria:
            # Use equal weights if no criteria provided
            weights = {a.evaluator_id: 1.0 for a in assessments}
        else:
            # Calculate weights based on expertise match
            weights = {}
            for assessment in assessments:
                weight = 1.0
                for criterion in criteria:
                    # Check if evaluator specializes in this criterion
                    if criterion.metric in [s.metric for s in assessment.scores]:
                        weight *= (1.0 + criterion.weight)
                weights[assessment.evaluator_id] = weight

        # Calculate weighted average
        weighted_sum = sum(a.composite_score * weights[a.evaluator_id] for a in assessments)
        total_weight = sum(weights.values())
        consensus_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate agreement level
        scores = [a.composite_score for a in assessments]
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        agreement_level = max(0.0, 1.0 - (variance / 1000.0))

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            disagreement_details={
                'weights': weights,
                'variance': variance,
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
            },
            method_used=ConsensusMethod.WEIGHTED_VOTE,
            participant_count=len(assessments),
            confidence=self._determine_confidence(agreement_level)
        )

    def _expertise_weighted_consensus(self, assessments: List[EvaluatorAssessment]) -> ConsensusResult:
        """Expertise-weighted consensus based on evaluator expertise level"""
        # In a real implementation, would use actual expertise levels
        # For now, use confidence level as proxy
        confidence_weights = {
            EvaluationConfidence.VERY_LOW: 0.25,
            EvaluationConfidence.LOW: 0.5,
            EvaluationConfidence.MODERATE: 0.75,
            EvaluationConfidence.HIGH: 1.0,
            EvaluationConfidence.VERY_HIGH: 1.25
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for assessment in assessments:
            weight = confidence_weights.get(assessment.confidence_level, 0.75)
            weighted_sum += assessment.composite_score * weight
            total_weight += weight

        consensus_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate agreement level
        scores = [a.composite_score for a in assessments]
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        agreement_level = max(0.0, 1.0 - (variance / 1000.0))

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            disagreement_details={
                'variance': variance,
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                'confidence_distribution': {
                    a.evaluator_id: a.confidence_level.value for a in assessments
                }
            },
            method_used=ConsensusMethod.EXPERTISE_WEIGHTED,
            participant_count=len(assessments),
            confidence=self._determine_confidence(agreement_level)
        )

    def _bayesian_aggregation(self, assessments: List[EvaluatorAssessment]) -> ConsensusResult:
        """Bayesian belief aggregation for consensus"""
        # Simplified Bayesian aggregation
        # In production, would use full Bayesian inference

        # Treat each assessment as a belief with mean and uncertainty
        beliefs = []
        for assessment in assessments:
            mean = assessment.composite_score
            # Use confidence level to determine uncertainty
            confidence_map = {
                EvaluationConfidence.VERY_LOW: 20.0,
                EvaluationConfidence.LOW: 15.0,
                EvaluationConfidence.MODERATE: 10.0,
                EvaluationConfidence.HIGH: 5.0,
                EvaluationConfidence.VERY_HIGH: 2.5
            }
            uncertainty = confidence_map.get(assessment.confidence_level, 10.0)
            beliefs.append({'mean': mean, 'std': uncertainty})

        # Aggregate beliefs (simplified - assuming independent beliefs)
        # In full Bayesian: combine posterior distributions
        combined_mean = sum(b['mean'] / (b['std'] ** 2) for b in beliefs)
        combined_precision = sum(1.0 / (b['std'] ** 2) for b in beliefs)
        consensus_score = combined_mean / combined_precision if combined_precision > 0 else 0.0

        # Calculate agreement level based on overlap of belief distributions
        agreement_level = self._calculate_belief_overlap(beliefs)

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            disagreement_details={
                'belief_count': len(beliefs),
                'combined_precision': combined_precision,
                'individual_beliefs': beliefs
            },
            method_used=ConsensusMethod.BAYESIAN_AGGREGATION,
            participant_count=len(assessments),
            confidence=self._determine_confidence(agreement_level)
        )

    def _median_consensus(self, assessments: List[EvaluatorAssessment]) -> ConsensusResult:
        """Median-based consensus"""
        scores = [a.composite_score for a in assessments]
        consensus_score = statistics.median(scores)

        # Agreement level based on interquartile range
        if len(scores) >= 4:
            q1 = statistics.quantiles(scores, n=4)[0]
            q3 = statistics.quantiles(scores, n=4)[2]
            iqr = q3 - q1
            agreement_level = max(0.0, 1.0 - (iqr / 50.0))
        else:
            variance = statistics.variance(scores) if len(scores) > 1 else 0
            agreement_level = max(0.0, 1.0 - (variance / 1000.0))

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            disagreement_details={
                'median': consensus_score,
                'iqr': iqr if len(scores) >= 4 else None,
                'variance': variance
            },
            method_used=ConsensusMethod.MEDIAN,
            participant_count=len(assessments),
            confidence=self._determine_confidence(agreement_level)
        )

    def _trimmed_mean_consensus(self, assessments: List[EvaluatorAssessment]) -> ConsensusResult:
        """Trimmed mean consensus (removes outliers)"""
        scores = [a.composite_score for a in assessments]
        sorted_scores = sorted(scores)

        # Remove top and bottom 20% (at least 1 each if enough scores)
        trim_count = max(1, int(len(sorted_scores) * 0.2))
        if len(sorted_scores) > 2 * trim_count:
            trimmed = sorted_scores[trim_count:-trim_count]
            outlier_ids = [
                assessments[i].evaluator_id
                for i in range(len(assessments))
                if assessments[i].composite_score in (sorted_scores[:trim_count] + sorted_scores[-trim_count:])
            ]
        else:
            trimmed = sorted_scores
            outlier_ids = []

        consensus_score = statistics.mean(trimmed) if trimmed else 0.0

        # Agreement level based on variance of trimmed set
        variance = statistics.variance(trimmed) if len(trimmed) > 1 else 0
        agreement_level = max(0.0, 1.0 - (variance / 1000.0))

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            disagreement_details={
                'trimmed_count': len(sorted_scores) - len(trimmed),
                'variance': variance,
                'original_range': max(scores) - min(scores)
            },
            method_used=ConsensusMethod.TRIMMED_MEAN,
            participant_count=len(assessments),
            outlier_evaluators=outlier_ids,
            confidence=self._determine_confidence(agreement_level)
        )

    def _calculate_belief_overlap(self, beliefs: List[Dict[str, float]]) -> float:
        """Calculate overlap between belief distributions"""
        if len(beliefs) < 2:
            return 1.0

        # Simplified overlap calculation
        # In production, would integrate overlapping probability distributions
        overlaps = []
        for i in range(len(beliefs)):
            for j in range(i + 1, len(beliefs)):
                b1 = beliefs[i]
                b2 = beliefs[j]
                # Check if means are within 2 std devs of each other
                distance = abs(b1['mean'] - b2['mean'])
                combined_std = (b1['std'] + b2['std']) / 2
                if distance < 2 * combined_std:
                    overlaps.append(1.0 - (distance / (2 * combined_std)))

        return statistics.mean(overlaps) if overlaps else 0.0

    def _determine_confidence(self, agreement_level: float) -> EvaluationConfidence:
        """Determine confidence level based on agreement"""
        if agreement_level >= 0.9:
            return EvaluationConfidence.VERY_HIGH
        elif agreement_level >= 0.75:
            return EvaluationConfidence.HIGH
        elif agreement_level >= 0.5:
            return EvaluationConfidence.MODERATE
        elif agreement_level >= 0.25:
            return EvaluationConfidence.LOW
        else:
            return EvaluationConfidence.VERY_LOW

    def _generate_consensus_rationale(
        self,
        result: ConsensusResult,
        assessments: List[EvaluatorAssessment]
    ) -> str:
        """Generate rationale for consensus result"""
        rationale_parts = [
            f"Consensus Method: {result.method_used.value}",
            f"Consensus Score: {result.consensus_score:.2f}",
            f"Agreement Level: {result.agreement_level:.2%}",
            f"Participants: {result.participant_count}"
        ]

        if result.outlier_evaluators:
            rationale_parts.append(f"Outlier Evaluators: {', '.join(result.outlier_evaluators)}")

        if result.agreement_level >= 0.8:
            rationale_parts.append("Strong consensus among evaluators")
        elif result.agreement_level >= 0.5:
            rationale_parts.append("Moderate consensus with some disagreement")
        else:
            rationale_parts.append("Low consensus, significant disagreement among evaluators")

        return "\n".join(rationale_parts)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_quality_gate(
    threshold_manager: Optional[QualityThresholdManager] = None
) -> QualityGateEngine:
    """Create a quality gate engine"""
    return QualityGateEngine(threshold_manager)


def create_multi_stage_validation(
    quality_gate: Optional[QualityGateEngine] = None
) -> MultiStageValidation:
    """Create a multi-stage validation system"""
    if quality_gate is None:
        quality_gate = create_quality_gate()
    return MultiStageValidation(quality_gate)


def create_consensus_builder() -> ConsensusBuilder:
    """Create a consensus builder"""
    return ConsensusBuilder()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

__all__ = [
    # Data models
    "GateDecision",
    "QualityLevel",
    "ContentType",
    "ConsensusMethod",
    "QualityThreshold",
    "QualityGateReport",
    "AppealRequest",
    "AppealDecision",
    "ConsensusResult",
    "MultiStageValidationResult",
    # Main classes
    "QualityThresholdManager",
    "QualityGateEngine",
    "MultiStageValidation",
    "ConsensusBuilder",
    # Factory functions
    "create_quality_gate",
    "create_multi_stage_validation",
    "create_consensus_builder",
]
