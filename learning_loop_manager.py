"""
Learning Loop Manager - Continuous Learning System
Manages learning from solved problems to improve future performance.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics

from sovereign_data_models import (
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    SolutionAttempt,
    ValidationResult,
    ComplexityScore,
    DomainContext,
    KnowledgeArtifact,
    StrategyPerformanceMetrics,
    TeamPerformanceMetrics,
    SubProblemStatus,
    generate_id
)

# **ACTUAL INTEGRATION**: Alerting and knowledge for Learning Loop Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact as EngineKnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LearningSummary:
    """
    Summary of learning from a solved problem.
    """
    problem_id: str
    learning_session_id: str

    # Performance improvements (non-defaults first)
    estimated_quality_improvement: float = 0.0
    estimated_efficiency_improvement: float = 0.0
    confidence_level: float = 0.5

    # Lessons learned
    lessons_learned: List[str] = field(default_factory=list)

    # Strategy updates
    strategy_preference_updates: Dict[str, float] = field(default_factory=dict)

    # Quality model updates
    quality_threshold_updates: Dict[str, float] = field(default_factory=dict)

    # Team performance updates
    team_performance_updates: Dict[str, float] = field(default_factory=dict)

    # Knowledge artifacts created
    artifacts_created: List[str] = field(default_factory=list)

    # Patterns identified
    patterns_identified: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Lesson:
    """
    Lesson learned from solving a problem.
    """
    lesson_id: str
    source_problem_id: str

    # Content (all required)
    lesson_type: str  # "success", "failure", "insight", "surprise"
    title: str
    description: str
    category: str  # "strategy", "team", "process", "domain"

    # Context (all required)
    domain: str
    problem_type: str
    strategy_used: str

    # Impact (all required)
    impact_level: str  # "high", "medium", "low"
    applicability: str  # "universal", "domain_specific", "situation_specific"
    confidence: float = 0.5  # 0-1

    # Optional fields with defaults
    teams_involved: List[str] = field(default_factory=list)
    actionable: bool = False
    action_description: Optional[str] = None
    expected_benefit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Lesson':
        """Create from dictionary."""
        data = data.copy()
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate lesson data."""
        errors = []

        valid_types = ["success", "failure", "insight", "surprise"]
        if self.lesson_type not in valid_types:
            errors.append(f"Invalid lesson_type: {self.lesson_type}. Must be one of {valid_types}")

        valid_categories = ["strategy", "team", "process", "domain"]
        if self.category not in valid_categories:
            errors.append(f"Invalid category: {self.category}. Must be one of {valid_categories}")

        valid_impacts = ["high", "medium", "low"]
        if self.impact_level not in valid_impacts:
            errors.append(f"Invalid impact_level: {self.impact_level}. Must be one of {valid_impacts}")

        valid_applicabilities = ["universal", "domain_specific", "situation_specific"]
        if self.applicability not in valid_applicabilities:
            errors.append(f"Invalid applicability: {self.applicability}. Must be one of {valid_applicabilities}")

        if not 0.0 <= self.confidence <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        return errors


class LearningLoopManager:
    """
    Manages continuous learning from solved problems.

    Features:
    - Extract lessons learned
    - Update strategy preferences
    - Improve quality thresholds
    - Refine team assignments
    - Close the feedback loop
    """

    def __init__(self, knowledge_store_path: str = "learning_loop_data.json"):
        """
        Initialize the learning loop manager.

        Args:
            knowledge_store_path: Path to persistent storage
        """
        self.knowledge_store_path = knowledge_store_path
        self.learning_history: List[LearningSummary] = []
        self.lessons: List[Lesson] = []
        self.strategy_preferences: Dict[str, float] = {
            "semantic": 0.5,
            "dependency": 0.5,
            "complexity": 0.5,
            "research": 0.5,
            "hybrid": 0.5
        }
        self.quality_thresholds: Dict[str, float] = {
            "completeness": 0.7,
            "consistency": 0.7,
            "feasibility": 0.7,
            "dependency": 0.7,
            "balance": 0.7
        }
        self.team_capability_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.teacher_traces: List[Dict[str, Any]] = []

        # Load existing data
        self._load_from_storage()

        logger.info(f"LearningLoopManager initialized with {len(self.learning_history)} learning sessions")

    def close_learning_loop(
        self,
        problem: ProblemDefinition,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        validations: Dict[str, ValidationResult]
    ) -> LearningSummary:
        """
        Complete learning loop from a solved problem.

        Process:
        1. Extract knowledge artifacts
        2. Update performance metrics
        3. Identify patterns and insights
        4. Update strategy preferences
        5. Improve quality models
        6. Generate learning summary

        Args:
            problem: The original problem
            plan: The decomposition plan used
            solutions: Solutions generated for sub-problems
            validations: Validation results for solutions

        Returns:
            LearningSummary with all learned information
        """
        logger.info(f"Closing learning loop for problem {problem.id}")

        learning_session_id = generate_id("learning_session")
        lessons_learned = []

        # 1. Extract lessons learned
        lessons = self.extract_lessons_learned(plan, solutions, validations)
        self.lessons.extend(lessons)
        lessons_learned.extend([l.title for l in lessons])

        # 2. Update strategy preferences
        strategy_updates = self.update_strategy_preferences(
            lessons,
            self.strategy_preferences.copy()
        )
        self.strategy_preferences.update(strategy_updates)

        # 3. Improve quality models
        self.improve_quality_models(lessons, problem.domain_context.domain)

        # 4. Refine team assignment model
        team_performance = self._calculate_team_performance(solutions, validations)
        self.refine_team_assignment_model(lessons, team_performance)

        # 5. Calculate estimated improvements
        quality_improvement = self._estimate_quality_improvement(lessons)
        efficiency_improvement = self._estimate_efficiency_improvement(lessons)

        # 6. Create learning summary
        summary = LearningSummary(
            problem_id=problem.id,
            learning_session_id=learning_session_id,
            lessons_learned=lessons_learned,
            strategy_preference_updates=strategy_updates,
            team_performance_updates=dict(team_performance),
            estimated_quality_improvement=quality_improvement,
            estimated_efficiency_improvement=efficiency_improvement,
            confidence_level=self._calculate_confidence(lessons, validations)
        )

        self.learning_history.append(summary)

        # Persist to storage
        self._save_to_storage()

        logger.info(f"Learning loop closed: {len(lessons)} lessons extracted, "
                   f"estimated quality improvement: {quality_improvement:.2%}")

        # **ACTUAL INTEGRATION**: Extract knowledge, track performance, and trigger alerts
        self._extract_learning_knowledge("close_learning_loop", summary)
        self._track_learning_performance(
            "close_learning_loop",
            True,
            quality_improvement,
            efficiency_improvement,
            len(lessons)
        )

        # Trigger alert if low improvements
        if quality_improvement < 0.05:
            self._trigger_learning_alerts(
                "close_learning_loop",
                True,
                problem.id,
                len(lessons),
                quality_improvement,
                None,
                {"efficiency_improvement": efficiency_improvement}
            )

        return summary

    def extract_lessons_learned(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        validations: Dict[str, ValidationResult]
    ) -> List[Lesson]:
        """
        Extract specific lessons learned.

        Lesson types:
        - What worked well
        - What didn't work
        - Surprising findings
        - Domain insights
        - Process improvements

        Args:
            plan: The decomposition plan
            solutions: Solutions generated
            validations: Validation results

        Returns:
            List of lessons learned
        """
        lessons = []

        # Analyze what worked well
        successful_solutions = []
        for sub_id, solution in solutions.items():
            if sub_id in validations:
                validation = validations[sub_id]
                if validation.passed:
                    successful_solutions.append((sub_id, solution, validation))

        for sub_id, solution, validation in successful_solutions:
            if validation.score >= 0.9:
                lesson = Lesson(
                    lesson_id=generate_id("lesson"),
                    source_problem_id=plan.problem_id,
                    lesson_type="success",
                    title=f"High-quality solution for {sub_id}",
                    description=f"Solution approach '{solution.approach}' achieved score {validation.score:.2f}",
                    category="strategy",
                    domain=plan.metadata.get("domain", "unknown"),
                    problem_type=plan.metadata.get("problem_type", "unknown"),
                    strategy_used=plan.strategy.value,
                    impact_level="high",
                    applicability="domain_specific",
                    confidence=validation.score,
                    actionable=True,
                    action_description=f"Consider using '{solution.approach}' for similar sub-problems",
                    expected_benefit="High quality outcomes"
                )
                lessons.append(lesson)

        # Analyze what didn't work
        failed_solutions = []
        for sub_id, solution in solutions.items():
            if sub_id in validations:
                validation = validations[sub_id]
                if not validation.passed:
                    failed_solutions.append((sub_id, solution, validation))

        for sub_id, solution, validation in failed_solutions:
            lesson = Lesson(
                lesson_id=generate_id("lesson"),
                source_problem_id=plan.problem_id,
                lesson_type="failure",
                title=f"Failed solution for {sub_id}",
                description=f"Solution approach '{solution.approach}' failed validation: {validation.feedback}",
                category="strategy",
                domain=plan.metadata.get("domain", "unknown"),
                problem_type=plan.metadata.get("problem_type", "unknown"),
                strategy_used=plan.strategy.value,
                impact_level="high",
                applicability="domain_specific",
                confidence=0.8,
                actionable=True,
                action_description=f"Avoid using '{solution.approach}' for similar sub-problems",
                expected_benefit="Prevent future failures"
            )
            lessons.append(lesson)

        # Extract domain insights
        domain_lessons = self._extract_domain_insights(plan, solutions, validations)
        lessons.extend(domain_lessons)

        # Extract process improvements
        process_lessons = self._extract_process_improvements(plan, solutions, validations)
        lessons.extend(process_lessons)

        logger.info(f"Extracted {len(lessons)} lessons from {len(solutions)} solutions")
        return lessons

    def update_strategy_preferences(
        self,
        lessons: List[Lesson],
        current_preferences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update strategy preferences based on lessons.

        If a strategy consistently performs well: increase preference
        If a strategy consistently underperforms: decrease preference

        Args:
            lessons: Lessons learned
            current_preferences: Current strategy preferences

        Returns:
            Updated strategy preferences
        """
        updates = current_preferences.copy()
        strategy_adjustments = defaultdict(float)

        # Group lessons by strategy and lesson type
        for lesson in lessons:
            if lesson.category == "strategy":
                weight = 0.1  # Adjustment weight

                if lesson.lesson_type == "success":
                    if lesson.impact_level == "high":
                        weight = 0.2
                    elif lesson.impact_level == "medium":
                        weight = 0.15

                    strategy_adjustments[lesson.strategy_used] += weight * lesson.confidence

                elif lesson.lesson_type == "failure":
                    if lesson.impact_level == "high":
                        weight = 0.2
                    elif lesson.impact_level == "medium":
                        weight = 0.15

                    strategy_adjustments[lesson.strategy_used] -= weight * lesson.confidence

        # Apply adjustments with bounds checking
        for strategy, adjustment in strategy_adjustments.items():
            if strategy in updates:
                new_value = updates[strategy] + adjustment
                updates[strategy] = max(0.0, min(1.0, new_value))

        logger.info(f"Updated strategy preferences: {strategy_adjustments}")
        return updates

    def improve_quality_models(
        self,
        lessons: List[Lesson],
        domain: str
    ):
        """
        Improve quality assessment models.

        Update:
        - Quality thresholds for domain
        - Weight factors for quality dimensions
        - Common patterns to look for
        - Red flags to watch for

        Args:
            lessons: Lessons learned
            domain: Domain being updated
        """
        domain_key = f"{domain}_quality"

        # Analyze quality-related lessons
        quality_lessons = [l for l in lessons if l.category == "process" and "quality" in l.title.lower()]

        if quality_lessons:
            # Adjust thresholds based on lessons
            for lesson in quality_lessons:
                if lesson.lesson_type == "success":
                    # We can be more stringent if we're succeeding
                    for threshold_name in self.quality_thresholds:
                        if threshold_name in lesson.description.lower():
                            self.quality_thresholds[threshold_name] = min(
                                1.0,
                                self.quality_thresholds[threshold_name] + 0.05
                            )

                elif lesson.lesson_type == "failure":
                    # Loosen thresholds if we're failing too much
                    for threshold_name in self.quality_thresholds:
                        if threshold_name in lesson.description.lower():
                            self.quality_thresholds[threshold_name] = max(
                                0.5,
                                self.quality_thresholds[threshold_name] - 0.05
                            )

        logger.info(f"Updated quality thresholds for {domain}: {self.quality_thresholds}")

    def refine_team_assignment_model(
        self,
        lessons: List[Lesson],
        team_performance: Dict[str, float]
    ):
        """
        Refine team assignment models.

        Update:
        - Team capability scores
        - Domain expertise mappings
        - Collaboration patterns
        - Assignment heuristics

        Args:
            lessons: Lessons learned
            team_performance: Team performance metrics
        """
        # Update team capability scores based on performance
        for team_id, performance_score in team_performance.items():
            # Smooth update to avoid sudden changes
            current_score = self.team_capability_scores["overall"][team_id]
            updated_score = 0.7 * current_score + 0.3 * performance_score
            self.team_capability_scores["overall"][team_id] = updated_score

        # Extract team-specific lessons
        for lesson in lessons:
            if lesson.category == "team":
                for team_id in lesson.teams_involved:
                    if lesson.lesson_type == "success":
                        self.team_capability_scores[team_id][lesson.domain] += 0.1 * lesson.confidence
                    elif lesson.lesson_type == "failure":
                        self.team_capability_scores[team_id][lesson.domain] -= 0.1 * lesson.confidence

        logger.info(f"Updated team capability scores for {len(team_performance)} teams")

    def get_recommended_strategy(
        self,
        domain: str,
        problem_type: str,
        complexity: float
    ) -> Tuple[str, float]:
        """
        Get recommended strategy based on learned preferences.

        Args:
            domain: Problem domain
            problem_type: Type of problem
            complexity: Problem complexity (0-1)

        Returns:
            Tuple of (strategy_name, confidence)
        """
        # Base preferences
        preferences = self.strategy_preferences.copy()

        # Adjust based on domain
        domain_key = f"{domain}_{problem_type}"
        if domain_key in preferences:
            preferences = {k: v * preferences.get(domain_key, 1.0) for k, v in preferences.items()}

        # Select strategy with highest preference
        best_strategy = max(preferences.items(), key=lambda x: x[1])
        confidence = best_strategy[1]

        return best_strategy[0], confidence

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about learning progress.

        Returns:
            Dictionary with learning statistics
        """
        total_lessons = len(self.lessons)

        # Count by type
        lessons_by_type = defaultdict(int)
        for lesson in self.lessons:
            lessons_by_type[lesson.lesson_type] += 1

        # Count by category
        lessons_by_category = defaultdict(int)
        for lesson in self.lessons:
            lessons_by_category[lesson.category] += 1

        # Average confidence
        avg_confidence = statistics.mean([l.confidence for l in self.lessons]) if self.lessons else 0.0

        return {
            "total_learning_sessions": len(self.learning_history),
            "total_lessons": total_lessons,
            "lessons_by_type": dict(lessons_by_type),
            "lessons_by_category": dict(lessons_by_category),
            "average_confidence": avg_confidence,
            "strategy_preferences": self.strategy_preferences,
            "quality_thresholds": self.quality_thresholds,
            "team_count": len(self.team_capability_scores["overall"])
        }

    def _extract_domain_insights(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        validations: Dict[str, ValidationResult]
    ) -> List[Lesson]:
        """Extract domain-specific insights."""
        lessons = []

        # Look for patterns in solutions by domain
        domain = plan.metadata.get("domain", "unknown")

        # High success rate patterns
        if validations:
            success_rate = sum(1 for v in validations.values() if v.passed) / len(validations)

            if success_rate >= 0.9:
                lesson = Lesson(
                    lesson_id=generate_id("lesson"),
                    source_problem_id=plan.problem_id,
                    lesson_type="success",
                    title=f"High success rate in {domain}",
                    description=f"Strategy {plan.strategy.value} achieved {success_rate:.1%} success rate",
                    category="domain",
                    domain=domain,
                    problem_type=plan.metadata.get("problem_type", "unknown"),
                    strategy_used=plan.strategy.value,
                    impact_level="high",
                    applicability="domain_specific",
                    confidence=success_rate,
                    actionable=True,
                    action_description=f"Consider using {plan.strategy.value} strategy for similar {domain} problems"
                )
                lessons.append(lesson)

        return lessons

    def _extract_process_improvements(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        validations: Dict[str, ValidationResult]
    ) -> List[Lesson]:
        """Extract process improvement lessons."""
        lessons = []

        # Check for common failure patterns
        common_failures = defaultdict(int)
        for sub_id, validation in validations.items():
            if not validation.passed:
                # Extract failure type from feedback
                feedback_lower = validation.feedback.lower()
                if "incomplete" in feedback_lower:
                    common_failures["incompleteness"] += 1
                if "inconsistent" in feedback_lower:
                    common_failures["inconsistency"] += 1
                if "unclear" in feedback_lower:
                    common_failures["clarity"] += 1

        for failure_type, count in common_failures.items():
            if count >= 2:  # Repeated failure pattern
                lesson = Lesson(
                    lesson_id=generate_id("lesson"),
                    source_problem_id=plan.problem_id,
                    lesson_type="insight",
                    title=f"Repeated {failure_type} issues detected",
                    description=f"Found {count} instances of {failure_type} in validation feedback",
                    category="process",
                    domain=plan.metadata.get("domain", "unknown"),
                    problem_type=plan.metadata.get("problem_type", "unknown"),
                    strategy_used=plan.strategy.value,
                    impact_level="medium",
                    applicability="universal",
                    confidence=0.8,
                    actionable=True,
                    action_description=f"Add additional checks for {failure_type} during solution generation"
                )
                lessons.append(lesson)

        return lessons

    def _calculate_team_performance(
        self,
        solutions: Dict[str, SolutionAttempt],
        validations: Dict[str, ValidationResult]
    ) -> Dict[str, float]:
        """Calculate team performance metrics."""
        team_performance = defaultdict(list)

        for sub_id, solution in solutions.items():
            if sub_id in validations:
                validation = validations[sub_id]
                team_performance[solution.team_id].append(validation.score)

        # Average scores by team
        return {
            team_id: statistics.mean(scores) if scores else 0.0
            for team_id, scores in team_performance.items()
        }

    def _estimate_quality_improvement(self, lessons: List[Lesson]) -> float:
        """Estimate quality improvement from lessons."""
        if not lessons:
            return 0.0

        # Count high-impact successes and failures
        high_impact_successes = sum(
            1 for l in lessons
            if l.lesson_type == "success" and l.impact_level == "high"
        )
        high_impact_failures = sum(
            1 for l in lessons
            if l.lesson_type == "failure" and l.impact_level == "high"
        )

        # Estimate improvement
        if high_impact_successes > high_impact_failures:
            return min(0.2, (high_impact_successes - high_impact_failures) * 0.05)
        elif high_impact_failures > high_impact_successes:
            return max(-0.1, -(high_impact_failures - high_impact_successes) * 0.03)
        else:
            return 0.0

    def _estimate_efficiency_improvement(self, lessons: List[Lesson]) -> float:
        """Estimate efficiency improvement from lessons."""
        process_improvements = [l for l in lessons if l.category == "process" and l.actionable]

        if not process_improvements:
            return 0.0

        # Each actionable process lesson could improve efficiency by 2-5%
        return min(0.15, len(process_improvements) * 0.03)

    def _calculate_confidence(
        self,
        lessons: List[Lesson],
        validations: Dict[str, ValidationResult]
    ) -> float:
        """Calculate confidence in learning summary."""
        if not lessons:
            return 0.0

        # Base confidence from lessons
        lesson_confidence = statistics.mean([l.confidence for l in lessons])

        # Adjust based on validation coverage
        coverage = len(validations) / max(1, len(lessons))
        adjusted_confidence = lesson_confidence * min(1.0, coverage)

        return adjusted_confidence

    def _load_from_storage(self):
        """Load learning data from persistent storage."""
        try:
            with open(self.knowledge_store_path, 'r') as f:
                data = json.load(f)

                # Load learning history
                if 'learning_history' in data:
                    for item in data['learning_history']:
                        summary = LearningSummary(**item)
                        summary.timestamp = datetime.fromisoformat(summary.timestamp)
                        self.learning_history.append(summary)

                # Load lessons
                if 'lessons' in data:
                    for item in data['lessons']:
                        lesson = Lesson.from_dict(item)
                        self.lessons.append(lesson)

                # Load preferences
                if 'strategy_preferences' in data:
                    self.strategy_preferences = data['strategy_preferences']

                # Load quality thresholds
                if 'quality_thresholds' in data:
                    self.quality_thresholds = data['quality_thresholds']

                # Load team scores
                if 'team_capability_scores' in data:
                    self.team_capability_scores = defaultdict(
                        lambda: defaultdict(float),
                        {
                            k: defaultdict(float, v)
                            for k, v in data['team_capability_scores'].items()
                        }
                    )
                if 'teacher_traces' in data:
                    self.teacher_traces = data['teacher_traces']

            logger.info(f"Loaded {len(self.learning_history)} learning sessions from storage")

        except FileNotFoundError:
            logger.info("No existing learning data found, starting fresh")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error loading learning data: {e}")

    def _save_to_storage(self):
        """Save learning data to persistent storage."""
        try:
            data = {
                'learning_history': [
                    {**asdict(summary), 'timestamp': summary.timestamp.isoformat()}
                    for summary in self.learning_history
                ],
                'lessons': [lesson.to_dict() for lesson in self.lessons],
                'strategy_preferences': self.strategy_preferences,
                'quality_thresholds': self.quality_thresholds,
                'team_capability_scores': dict(self.team_capability_scores),
                'teacher_traces': self.teacher_traces
            }

            with open(self.knowledge_store_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("Learning data saved to storage")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error saving learning data: {e}")

    def register_teacher_trace(self, narrative: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a converged narrative for federated distillation."""
        self.teacher_traces.append({
            "narrative": narrative,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        self._save_to_storage()

    def distill_local_model(
        self,
        student_model_name: str = "local-student",
        max_samples: int = 200
    ) -> Dict[str, Any]:
        """
        Prepare a dataset for federated distillation of a local model.

        Returns:
            Metadata about the prepared distillation dataset.
        """
        dataset = self.teacher_traces[-max_samples:]
        output_path = f"distillation_{student_model_name}.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write distillation dataset: {e}")
            return {"status": "error", "error": str(e)}

        return {
            "status": "prepared",
            "student_model": student_model_name,
            "dataset_path": output_path,
            "samples": len(dataset)
        }

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Learning Loop Manager
    # =========================================================================

    def _trigger_learning_alerts(
        self,
        operation: str,
        success: bool,
        problem_id: Optional[str] = None,
        num_lessons: int = 0,
        quality_improvement: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for learning loop failures or low improvements."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures or very low improvements
            if not success or quality_improvement < 0.05:
                severity = AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Learning Loop Manager Alert: {operation}",
                    description=f"Learning loop operation '{operation}' " +
                                 ("failed" if not success else f"has low improvement: {quality_improvement:.2%}") +
                                 (f" for problem '{problem_id}'" if problem_id else "") +
                                 (f" with {num_lessons} lessons" if num_lessons > 0 else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="learning_loop_manager",
                    component="continuous_learning",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Learning Loop alert: {e}")

    def _extract_learning_knowledge(
        self,
        operation: str,
        learning_summary: LearningSummary
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract learning loop knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = EngineKnowledgeArtifact(
                artifact_id=f"learning_loop_{operation}_{learning_summary.problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="learning_loop",
                source_component="learning_loop_manager",
                title=f"Learning Loop: {learning_summary.problem_id} ({operation})",
                content={
                    "operation": operation,
                    "problem_id": learning_summary.problem_id,
                    "learning_session_id": learning_summary.learning_session_id,
                    "estimated_quality_improvement": learning_summary.estimated_quality_improvement,
                    "estimated_efficiency_improvement": learning_summary.estimated_efficiency_improvement,
                    "confidence_level": learning_summary.confidence_level,
                    "num_lessons": len(learning_summary.lessons_learned),
                    "num_patterns": len(learning_summary.patterns_identified),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "strategy_preference_updates": learning_summary.strategy_preference_updates,
                    "quality_threshold_updates": learning_summary.quality_threshold_updates,
                    "artifacts_created": learning_summary.artifacts_created,
                    "learning_summary_dict": asdict(learning_summary)
                },
                tags=["learning_loop", "continuous_learning", operation]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Learning Loop knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Learning Loop knowledge: {e}")
            return False

    def _track_learning_performance(
        self,
        operation: str,
        success: bool,
        quality_improvement: float = 0.0,
        efficiency_improvement: float = 0.0,
        num_lessons: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track learning loop performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            # Quality based on improvements and lessons learned
            quality = 0.5 if success else 0.0
            if success:
                # Factor in both quality and efficiency improvements
                quality = (quality_improvement + efficiency_improvement) / 2.0
                # Bonus for extracting many lessons
                if num_lessons > 0:
                    quality = min(quality + 0.1, 1.0)
            quality = max(min(quality, 1.0), 0.0)

            performance_data = StrategyPerformanceData(
                strategy_name=f"learning_loop_manager_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "quality_improvement": quality_improvement,
                    "efficiency_improvement": efficiency_improvement,
                    "num_lessons": num_lessons
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Learning Loop performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Learning Loop performance: {e}")
