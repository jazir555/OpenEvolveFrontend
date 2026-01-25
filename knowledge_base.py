"""
Knowledge Base - Central Knowledge Repository
Stores and retrieves knowledge artifacts for continuous learning.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics
from difflib import SequenceMatcher
import re

from sovereign_data_models import (
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    SolutionAttempt,
    ValidationResult,
    DomainContext,
    KnowledgeArtifact,
    ComplexityScore,
    generate_id
)

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeQuery:
    """Query for knowledge base."""

    # Filters
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    artifact_type: Optional[str] = None
    tags: Optional[List[str]] = None

    # Constraints
    min_confidence: float = 0.0
    min_support_count: int = 1
    time_range: Optional[Tuple[datetime, datetime]] = None

    # Sorting
    sort_by: str = "confidence"  # "confidence", "support_count", "recentness"
    max_results: int = 10

    def validate(self) -> List[str]:
        """Validate query parameters."""
        errors = []

        if not 0.0 <= self.min_confidence <= 1.0:
            errors.append(f"min_confidence must be between 0.0 and 1.0, got {self.min_confidence}")

        if self.min_support_count < 1:
            errors.append(f"min_support_count must be at least 1, got {self.min_support_count}")

        valid_sort_fields = ["confidence", "support_count", "recentness"]
        if self.sort_by not in valid_sort_fields:
            errors.append(f"sort_by must be one of {valid_sort_fields}, got {self.sort_by}")

        if self.max_results < 1:
            errors.append(f"max_results must be at least 1, got {self.max_results}")

        return errors


@dataclass
class SimilarProblem:
    """Problem similar to current one."""
    problem_id: str
    title: str
    similarity_score: float  # 0-1

    # Details
    domain: str
    problem_type: str
    strategy_used: str
    quality_achieved: float

    # Relevance
    why_similar: str
    key_differences: List[str] = field(default_factory=list)
    lessons_applicable: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BestPractice:
    """Best practice identified in domain."""
    practice_id: str
    title: str
    description: str
    domain: str
    category: str

    # Evidence
    support_count: int  # How many times observed
    success_rate: float  # When followed, success rate

    # Application
    when_to_apply: str
    how_to_apply: str
    expected_benefit: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AntiPattern:
    """Anti-pattern to avoid."""
    pattern_id: str
    title: str
    description: str
    domain: str

    # Evidence
    support_count: int
    failure_rate: float  # When followed, failure rate

    # Avoidance
    why_avoid: str
    how_to_avoid: str
    alternative: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyRecommendation:
    """Recommended strategy with reasoning."""
    strategy: str
    confidence: float

    # Reasoning
    primary_reason: str
    supporting_evidence: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    # Expected performance
    expected_quality: float = 0.7
    expected_success_rate: float = 0.7

    # Alternatives
    alternative_strategies: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy': self.strategy,
            'confidence': self.confidence,
            'primary_reason': self.primary_reason,
            'supporting_evidence': self.supporting_evidence,
            'caveats': self.caveats,
            'expected_quality': self.expected_quality,
            'expected_success_rate': self.expected_success_rate,
            'alternative_strategies': self.alternative_strategies
        }


@dataclass
class ProblemSolvingExperience:
    """Complete record of solving a problem."""
    experience_id: str

    # Problem
    problem: ProblemDefinition
    domain: str
    problem_type: str

    # Process
    decomposition_plan: DecompositionPlan
    strategy_used: str
    teams_assigned: Dict[str, str] = field(default_factory=dict)

    # Outcomes
    solutions: Dict[str, SolutionAttempt] = field(default_factory=dict)
    validations: Dict[str, ValidationResult] = field(default_factory=dict)

    # Performance
    quality_scores: Dict[str, float] = field(default_factory=dict)
    time_taken: float = 0.0
    success: bool = False

    # Lessons
    lessons_learned: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experience_id': self.experience_id,
            'problem': self.problem.to_dict(),
            'domain': self.domain,
            'problem_type': self.problem_type,
            'decomposition_plan': self.decomposition_plan.to_dict(),
            'strategy_used': self.strategy_used,
            'teams_assigned': self.teams_assigned,
            'solutions': {k: v.to_dict() for k, v in self.solutions.items()},
            'validations': {k: v.to_dict() for k, v in self.validations.items()},
            'quality_scores': self.quality_scores,
            'time_taken': self.time_taken,
            'success': self.success,
            'lessons_learned': self.lessons_learned,
            'artifacts_created': self.artifacts_created,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class KnowledgeReport:
    """Comprehensive knowledge report."""
    report_id: str
    domain: str
    time_period: str

    # Statistics
    total_artifacts: int
    artifact_breakdown: Dict[str, int]  # type -> count
    total_lessons: int

    # Content
    best_practices: List[BestPractice]
    anti_patterns: List[AntiPattern]
    common_patterns: List[str]

    # Performance
    performance_summary: Dict[str, float]
    trends: Dict[str, str]

    # Recommendations
    recommendations: List[str]
    improvement_areas: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeBase:
    """
    Central knowledge repository for continuous learning.

    Stores:
    - Knowledge artifacts
    - Lessons learned
    - Best practices
    - Anti-patterns
    - Domain patterns
    - Performance data
    """

    def __init__(self, storage_path: str = "knowledge_base.json"):
        """
        Initialize with persistent storage.

        Args:
            storage_path: Path to persistent storage
        """
        self.storage_path = storage_path
        self.artifacts: List[KnowledgeArtifact] = []
        self.experiences: List[ProblemSolvingExperience] = []
        self.best_practices: Dict[str, BestPractice] = {}
        self.anti_patterns: Dict[str, AntiPattern] = {}

        # Load existing data
        self._load_from_storage()

        logger.info(f"KnowledgeBase initialized with {len(self.artifacts)} artifacts "
                   f"and {len(self.experiences)} experiences")

    def store_artifact(self, artifact: KnowledgeArtifact):
        """
        Store artifact in knowledge base.

        Args:
            artifact: Artifact to store
        """
        # Check for similar existing artifacts
        similar_artifact = self._find_similar_artifact(artifact)

        if similar_artifact:
            # Update existing artifact
            similar_artifact.support_count += 1
            similar_artifact.confidence = max(
                similar_artifact.confidence,
                artifact.confidence
            )
            similar_artifact.success_rate = (
                similar_artifact.success_rate + artifact.success_rate
            ) / 2

            logger.info(f"Updated existing artifact {similar_artifact.artifact_id} "
                       f"(support count: {similar_artifact.support_count})")
        else:
            # Add new artifact
            self.artifacts.append(artifact)
            logger.info(f"Stored new artifact {artifact.artifact_id}")

        # Update best practices and anti-patterns
        self._update_patterns(artifact)

        # Persist to storage
        self._save_to_storage()

    def retrieve_artifacts(self, query: KnowledgeQuery) -> List[KnowledgeArtifact]:
        """
        Retrieve artifacts matching query.

        Query can filter by:
        - Domain
        - Problem type
        - Artifact type
        - Tags
        - Time range
        - Confidence threshold

        Args:
            query: Knowledge query

        Returns:
            List of matching artifacts
        """
        # Validate query
        errors = query.validate()
        if errors:
            logger.error(f"Invalid query: {errors}")
            return []

        # Filter artifacts
        filtered = self.artifacts

        if query.domain:
            filtered = [a for a in filtered if a.domain == query.domain]

        if query.problem_type:
            filtered = [a for a in filtered if a.problem_type == query.problem_type]

        if query.artifact_type:
            filtered = [a for a in filtered if a.artifact_type == query.artifact_type]

        if query.tags:
            filtered = [
                a for a in filtered
                if any(tag in a.tags for tag in query.tags)
            ]

        if query.min_confidence > 0:
            filtered = [a for a in filtered if a.confidence >= query.min_confidence]

        if query.min_support_count > 1:
            filtered = [a for a in filtered if a.support_count >= query.min_support_count]

        if query.time_range:
            start, end = query.time_range
            filtered = [
                a for a in filtered
                if start <= a.extraction_date <= end
            ]

        # Sort results
        if query.sort_by == "confidence":
            filtered.sort(key=lambda a: a.confidence, reverse=True)
        elif query.sort_by == "support_count":
            filtered.sort(key=lambda a: a.support_count, reverse=True)
        elif query.sort_by == "recentness":
            filtered.sort(key=lambda a: a.extraction_date, reverse=True)

        # Limit results
        return filtered[:query.max_results]

    def find_similar_problems(
        self,
        problem: ProblemDefinition,
        n_results: int = 5
    ) -> List[SimilarProblem]:
        """
        Find similar previously solved problems.

        Uses:
        - Domain similarity
        - Keyword overlap
        - Problem type match
        - Complexity similarity

        Args:
            problem: Current problem
            n_results: Number of results to return

        Returns:
            List of similar problems
        """
        similarities = []

        for experience in self.experiences:
            similarity_score = self._calculate_problem_similarity(
                problem,
                experience.problem
            )

            if similarity_score > 0.3:  # Minimum similarity threshold
                similar = SimilarProblem(
                    problem_id=experience.problem.id,
                    title=experience.problem.title,
                    similarity_score=similarity_score,
                    domain=experience.domain,
                    problem_type=experience.problem_type,
                    strategy_used=experience.strategy_used,
                    quality_achieved=statistics.mean(experience.quality_scores.values()) if experience.quality_scores else 0.0,
                    why_similar=self._explain_similarity(problem, experience.problem),
                    key_differences=self._identify_differences(problem, experience.problem),
                    lessons_applicable=experience.lessons_learned
                )
                similarities.append(similar)

        # Sort by similarity score
        similarities.sort(key=lambda s: s.similarity_score, reverse=True)

        return similarities[:n_results]

    def get_best_practices(
        self,
        domain: str,
        problem_type: Optional[str] = None
    ) -> List[BestPractice]:
        """
        Get best practices for domain/problem type.

        Args:
            domain: Domain of interest
            problem_type: Optional problem type filter

        Returns:
            List of best practices
        """
        practices = [
            bp for bp in self.best_practices.values()
            if bp.domain == domain
        ]

        if problem_type:
            # Filter by problem type in description or metadata
            practices = [
                bp for bp in practices
                if problem_type.lower() in bp.description.lower()
                or problem_type.lower() in bp.title.lower()
            ]

        # Sort by support count and success rate
        practices.sort(key=lambda bp: (bp.support_count, bp.success_rate), reverse=True)

        return practices

    def get_anti_patterns(self, domain: str) -> List[AntiPattern]:
        """
        Get anti-patterns to avoid in domain.

        Args:
            domain: Domain of interest

        Returns:
            List of anti-patterns
        """
        patterns = [
            ap for ap in self.anti_patterns.values()
            if ap.domain == domain
        ]

        # Sort by failure rate and support count
        patterns.sort(key=lambda ap: (ap.failure_rate, ap.support_count), reverse=True)

        return patterns

    def recommend_strategy(
        self,
        problem: ProblemDefinition,
        domain_context: DomainContext
    ) -> StrategyRecommendation:
        """
        Recommend strategy based on knowledge base.

        Uses:
        - Historical performance in domain
        - Similar problems and what worked
        - Team capabilities
        - Current trends

        Args:
            problem: Problem to solve
            domain_context: Domain context

        Returns:
            Strategy recommendation with reasoning
        """
        domain = domain_context.domain

        # Find similar problems
        similar_problems = self.find_similar_problems(problem, n_results=5)

        # Analyze what worked for similar problems
        strategy_performance = defaultdict(list)
        for similar in similar_problems:
            strategy_performance[similar.strategy_used].append(similar.quality_achieved)

        # Calculate average performance by strategy
        avg_performance = {
            strategy: statistics.mean(scores)
            for strategy, scores in strategy_performance.items()
        }

        # Get best practices
        best_practices = self.get_best_practices(domain)

        # Select best strategy
        if avg_performance:
            best_strategy = max(avg_performance.items(), key=lambda x: x[1])
            strategy = best_strategy[0]
            confidence = min(0.95, best_strategy[1])
            expected_quality = best_strategy[1]
        else:
            # Default to hybrid if no data
            strategy = "hybrid"
            confidence = 0.5
            expected_quality = 0.7

        # Calculate expected success rate
        successful_similar = sum(1 for s in similar_problems if s.quality_achieved >= 0.7)
        expected_success_rate = successful_similar / len(similar_problems) if similar_problems else 0.7

        # Generate reasoning
        primary_reason = f"Based on {len(similar_problems)} similar problems in {domain}"

        supporting_evidence = []
        if similar_problems:
            best_similar = similar_problems[0]
            supporting_evidence.append(
                f"Most similar problem used {best_similar.strategy_used} "
                f"and achieved {best_similar.quality_achieved:.2%} quality"
            )

        if best_practices:
            supporting_evidence.append(
                f"Found {len(best_practices)} best practices for {domain}"
            )

        # Get alternative strategies
        alternatives = [
            (s, p)
            for s, p in avg_performance.items()
            if s != strategy
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)

        # Identify caveats
        caveats = []
        if not similar_problems:
            caveats.append("Limited historical data for this problem type")
        if len(similar_problems) < 3:
            caveats.append("Recommendation based on small sample size")

        # Get anti-patterns for domain
        anti_patterns = self.get_anti_patterns(domain)
        if anti_patterns:
            caveats.append(f"Avoid {len(anti_patterns)} known anti-patterns in {domain}")

        recommendation = StrategyRecommendation(
            strategy=strategy,
            confidence=confidence,
            primary_reason=primary_reason,
            supporting_evidence=supporting_evidence,
            caveats=caveats,
            expected_quality=expected_quality,
            expected_success_rate=expected_success_rate,
            alternative_strategies=alternatives[:3]
        )

        logger.info(f"Strategy recommendation: {strategy} (confidence: {confidence:.2f})")

        return recommendation

    def update_from_experience(self, experience: ProblemSolvingExperience):
        """
        Update knowledge base from a new experience.

        Extracts and stores:
        - New artifacts
        - Lessons learned
        - Performance data
        - Pattern observations

        Args:
            experience: Problem solving experience
        """
        # Store experience
        self.experiences.append(experience)

        # Extract and store artifacts
        for artifact_id in experience.artifacts_created:
            # Find artifact in temporary storage or create from lessons
            # This would typically be coordinated with the LearningLoopManager
            pass

        # Update best practices based on success
        if experience.success:
            self._update_best_practices_from_experience(experience)

        # Update anti-patterns based on failures
        if not experience.success:
            self._update_anti_patterns_from_experience(experience)

        # Persist to storage
        self._save_to_storage()

        logger.info(f"Knowledge base updated from experience {experience.experience_id}")

    def generate_knowledge_report(
        self,
        domain: Optional[str] = None,
        time_period: str = "all"
    ) -> KnowledgeReport:
        """
        Generate comprehensive knowledge report.

        Includes:
        - Artifact statistics
        - Performance trends
        - Common patterns
        - Best practices
        - Anti-patterns
        - Recommendations

        Args:
            domain: Optional domain filter
            time_period: Time period to report on

        Returns:
            Comprehensive knowledge report
        """
        # Filter by domain and time
        artifacts = self.artifacts
        experiences = self.experiences

        if domain:
            artifacts = [a for a in artifacts if a.domain == domain]
            experiences = [e for e in experiences if e.domain == domain]

        if time_period != "all":
            # Parse time period (e.g., "7d", "30d", "90d")
            days = int(time_period[:-1])
            cutoff = datetime.now() - timedelta(days=days)
            artifacts = [a for a in artifacts if a.extraction_date >= cutoff]
            experiences = [e for e in experiences if e.timestamp >= cutoff]

        # Calculate statistics
        total_artifacts = len(artifacts)
        artifact_breakdown = defaultdict(int)
        for artifact in artifacts:
            artifact_breakdown[artifact.artifact_type] += 1

        total_lessons = sum(len(e.lessons_learned) for e in experiences)

        # Get best practices and anti-patterns
        if domain:
            best_practices = self.get_best_practices(domain)
            anti_patterns = self.get_anti_patterns(domain)
        else:
            best_practices = list(self.best_practices.values())[:10]
            anti_patterns = list(self.anti_patterns.values())[:10]

        # Extract common patterns
        common_patterns = self._extract_common_patterns(artifacts)

        # Calculate performance summary
        performance_summary = self._calculate_performance_summary(experiences)

        # Identify trends
        trends = self._identify_trends(experiences)

        # Generate recommendations and improvement areas
        recommendations = self._generate_recommendations(
            artifacts,
            experiences,
            performance_summary
        )
        improvement_areas = self._identify_improvement_areas(
            artifacts,
            experiences,
            performance_summary
        )

        report = KnowledgeReport(
            report_id=generate_id("report"),
            domain=domain or "all",
            time_period=time_period,
            total_artifacts=total_artifacts,
            artifact_breakdown=dict(artifact_breakdown),
            total_lessons=total_lessons,
            best_practices=best_practices,
            anti_patterns=anti_patterns,
            common_patterns=common_patterns,
            performance_summary=performance_summary,
            trends=trends,
            recommendations=recommendations,
            improvement_areas=improvement_areas
        )

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with statistics
        """
        # Artifact statistics
        artifact_types = defaultdict(int)
        domains = defaultdict(int)
        total_confidence = 0.0

        for artifact in self.artifacts:
            artifact_types[artifact.artifact_type] += 1
            domains[artifact.domain] += 1
            total_confidence += artifact.confidence

        avg_confidence = total_confidence / len(self.artifacts) if self.artifacts else 0.0

        # Experience statistics
        if self.experiences:
            success_rate = sum(1 for e in self.experiences if e.success) / len(self.experiences)
            quality_values = []
            for e in self.experiences:
                if e.quality_scores:
                    quality_values.append(statistics.mean(e.quality_scores.values()))
                else:
                    quality_values.append(0.0)
            avg_quality = statistics.mean(quality_values) if quality_values else 0.0
        else:
            success_rate = 0.0
            avg_quality = 0.0

        return {
            "total_artifacts": len(self.artifacts),
            "total_experiences": len(self.experiences),
            "artifact_types": dict(artifact_types),
            "domains": dict(domains),
            "average_artifact_confidence": avg_confidence,
            "total_best_practices": len(self.best_practices),
            "total_anti_patterns": len(self.anti_patterns),
            "experience_success_rate": success_rate,
            "average_quality": avg_quality
        }

    def _find_similar_artifact(self, artifact: KnowledgeArtifact) -> Optional[KnowledgeArtifact]:
        """Find similar existing artifact."""
        for existing in self.artifacts:
            if (existing.artifact_type == artifact.artifact_type and
                existing.domain == artifact.domain and
                existing.problem_type == artifact.problem_type and
                existing.title == artifact.title):
                return existing
        return None

    def _update_patterns(self, artifact: KnowledgeArtifact):
        """Update best practices and anti-patterns from artifact."""
        if artifact.artifact_type == "best_practice":
            practice = BestPractice(
                practice_id=artifact.artifact_id,
                title=artifact.title,
                description=artifact.description,
                domain=artifact.domain,
                category="general",
                support_count=artifact.support_count,
                success_rate=artifact.success_rate,
                when_to_apply=artifact.metadata.get("when_to_apply", ""),
                how_to_apply=artifact.metadata.get("how_to_apply", ""),
                expected_benefit=artifact.metadata.get("expected_benefit", "")
            )
            self.best_practices[practice.practice_id] = practice

        elif artifact.artifact_type == "anti_pattern":
            anti = AntiPattern(
                pattern_id=artifact.artifact_id,
                title=artifact.title,
                description=artifact.description,
                domain=artifact.domain,
                support_count=artifact.support_count,
                failure_rate=1.0 - artifact.success_rate,
                why_avoid=artifact.metadata.get("why_avoid", ""),
                how_to_avoid=artifact.metadata.get("how_to_avoid", ""),
                alternative=artifact.metadata.get("alternative", "")
            )
            self.anti_patterns[anti.pattern_id] = anti

    def _update_best_practices_from_experience(self, experience: ProblemSolvingExperience):
        """Update best practices from successful experience."""
        # Create best practice artifacts from successful solutions
        for sub_id, solution in experience.solutions.items():
            if sub_id in experience.validations:
                validation = experience.validations[sub_id]
                if validation.passed and validation.score >= 0.9:
                    practice_id = generate_id("practice")
                    practice = BestPractice(
                        practice_id=practice_id,
                        title=f"Effective approach for {sub_id}",
                        description=f"Solution approach '{solution.approach}' achieved high quality",
                        domain=experience.domain,
                        category="solution_strategy",
                        support_count=1,
                        success_rate=validation.score,
                        when_to_apply=f"When solving similar {experience.problem_type} problems",
                        how_to_apply=f"Use {solution.approach} approach",
                        expected_benefit=f"High quality solutions ({validation.score:.1%} success rate)"
                    )
                    self.best_practices[practice_id] = practice

    def _update_anti_patterns_from_experience(self, experience: ProblemSolvingExperience):
        """Update anti-patterns from failed experience."""
        # Create anti-pattern artifacts from failed solutions
        for sub_id, solution in experience.solutions.items():
            if sub_id in experience.validations:
                validation = experience.validations[sub_id]
                if not validation.passed:
                    anti_id = generate_id("anti")
                    anti = AntiPattern(
                        pattern_id=anti_id,
                        title=f"Ineffective approach for {sub_id}",
                        description=f"Solution approach '{solution.approach}' failed validation",
                        domain=experience.domain,
                        support_count=1,
                        failure_rate=1.0 - validation.score,
                        why_avoid=validation.feedback,
                        how_to_avoid=f"Avoid using {solution.approach} for similar problems",
                        alternative="Consider alternative solution approaches"
                    )
                    self.anti_patterns[anti_id] = anti

    def _calculate_problem_similarity(
        self,
        problem1: ProblemDefinition,
        problem2: ProblemDefinition
    ) -> float:
        """Calculate similarity between two problems (0-1)."""
        similarity = 0.0

        # Domain match (40% weight)
        if problem1.domain_context.domain == problem2.domain_context.domain:
            similarity += 0.4

        # Problem type match (30% weight)
        if problem1.problem_type == problem2.problem_type:
            similarity += 0.3

        # Title similarity (20% weight)
        title_similarity = SequenceMatcher(
            None,
            problem1.title.lower(),
            problem2.title.lower()
        ).ratio()
        similarity += title_similarity * 0.2

        # Complexity similarity (10% weight)
        complexity_diff = abs(
            problem1.complexity_score.overall_complexity -
            problem2.complexity_score.overall_complexity
        ) / 10.0
        similarity += (1.0 - complexity_diff) * 0.1

        return similarity

    def _explain_similarity(
        self,
        problem1: ProblemDefinition,
        problem2: ProblemDefinition
    ) -> str:
        """Explain why two problems are similar."""
        reasons = []

        if problem1.domain_context.domain == problem2.domain_context.domain:
            reasons.append(f"same domain ({problem1.domain_context.domain})")

        if problem1.problem_type == problem2.problem_type:
            reasons.append(f"same type ({problem1.problem_type.value})")

        complexity_diff = abs(
            problem1.complexity_score.overall_complexity -
            problem2.complexity_score.overall_complexity
        )
        if complexity_diff < 2:
            reasons.append("similar complexity")

        return "; ".join(reasons) if reasons else "general similarity"

    def _identify_differences(
        self,
        problem1: ProblemDefinition,
        problem2: ProblemDefinition
    ) -> List[str]:
        """Identify key differences between problems."""
        differences = []

        if problem1.domain_context.domain != problem2.domain_context.domain:
            differences.append(f"different domains")

        if problem1.problem_type != problem2.problem_type:
            differences.append(f"different problem types")

        complexity_diff = abs(
            problem1.complexity_score.overall_complexity -
            problem2.complexity_score.overall_complexity
        )
        if complexity_diff >= 3:
            differences.append(f"significantly different complexity ({complexity_diff:.1f})")

        return differences

    def _extract_common_patterns(self, artifacts: List[KnowledgeArtifact]) -> List[str]:
        """Extract common patterns from artifacts."""
        patterns = []

        # Count artifacts by domain and type
        domain_type_counts = defaultdict(lambda: defaultdict(int))
        for artifact in artifacts:
            domain_type_counts[artifact.domain][artifact.artifact_type] += 1

        # Find common patterns (>= 3 occurrences)
        for domain, type_counts in domain_type_counts.items():
            for artifact_type, count in type_counts.items():
                if count >= 3:
                    patterns.append(
                        f"{artifact_type.replace('_', ' ').title()} in {domain} "
                        f"({count} occurrences)"
                    )

        return patterns

    def _calculate_performance_summary(self, experiences: List[ProblemSolvingExperience]) -> Dict[str, float]:
        """Calculate performance summary from experiences."""
        if not experiences:
            return {}

        # Calculate quality values safely
        quality_values = []
        for e in experiences:
            if e.quality_scores:
                quality_values.append(statistics.mean(e.quality_scores.values()))
            else:
                quality_values.append(0.0)

        # Calculate time values safely
        time_values = [e.time_taken for e in experiences if e.time_taken > 0]

        return {
            "success_rate": sum(1 for e in experiences if e.success) / len(experiences),
            "avg_quality": statistics.mean(quality_values) if quality_values else 0.0,
            "avg_time": statistics.mean(time_values) if time_values else 0.0,
            "total_problems": len(experiences)
        }

    def _identify_trends(self, experiences: List[ProblemSolvingExperience]) -> Dict[str, str]:
        """Identify trends in experiences."""
        trends = {}

        if len(experiences) < 5:
            trends["note"] = "Insufficient data for trend analysis"
            return trends

        # Split into first half and second half
        mid = len(experiences) // 2
        first_half = experiences[:mid]
        second_half = experiences[mid:]

        # Quality trend
        first_quality = statistics.mean([
            statistics.mean(e.quality_scores.values()) if e.quality_scores else 0.0
            for e in first_half
        ])
        second_quality = statistics.mean([
            statistics.mean(e.quality_scores.values()) if e.quality_scores else 0.0
            for e in second_half
        ])

        if second_quality > first_quality + 0.05:
            trends["quality"] = "improving"
        elif second_quality < first_quality - 0.05:
            trends["quality"] = "declining"
        else:
            trends["quality"] = "stable"

        # Success rate trend
        first_success = sum(1 for e in first_half if e.success) / len(first_half)
        second_success = sum(1 for e in second_half if e.success) / len(second_half)

        if second_success > first_success + 0.05:
            trends["success_rate"] = "improving"
        elif second_success < first_success - 0.05:
            trends["success_rate"] = "declining"
        else:
            trends["success_rate"] = "stable"

        return trends

    def _generate_recommendations(
        self,
        artifacts: List[KnowledgeArtifact],
        experiences: List[ProblemSolvingExperience],
        performance: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on knowledge base."""
        recommendations = []

        if not experiences:
            recommendations.append("Collect more problem-solving experiences")
            return recommendations

        success_rate = performance.get("success_rate", 0.0)

        if success_rate < 0.7:
            recommendations.append("Focus on improving solution quality and validation")

        if len(artifacts) < 10:
            recommendations.append("Extract and store more knowledge artifacts from experiences")

        # Check for domains with low performance
        domain_performance = defaultdict(list)
        for exp in experiences:
            if exp.quality_scores:
                avg_quality = statistics.mean(exp.quality_scores.values())
                domain_performance[exp.domain].append(avg_quality)

        for domain, qualities in domain_performance.items():
            if statistics.mean(qualities) < 0.7:
                recommendations.append(
                    f"Review and improve approaches for {domain} domain "
                    f"(current avg quality: {statistics.mean(qualities):.1%})"
                )

        return recommendations

    def _identify_improvement_areas(
        self,
        artifacts: List[KnowledgeArtifact],
        experiences: List[ProblemSolvingExperience],
        performance: Dict[str, float]
    ) -> List[str]:
        """Identify areas needing improvement."""
        areas = []

        # Check artifact diversity
        artifact_types = set(a.artifact_type for a in artifacts)
        if len(artifact_types) < 3:
            areas.append("Increase diversity of knowledge artifact types")

        # Check domain coverage
        domains = set(a.domain for a in artifacts)
        if len(domains) < 3:
            areas.append("Expand knowledge base to cover more domains")

        # Check success rate
        if performance.get("success_rate", 1.0) < 0.8:
            areas.append("Improve overall solution success rate")

        # Check learning rate
        if len(experiences) > 10:
            recent_success = sum(1 for e in experiences[-5:] if e.success) / 5
            overall_success = performance.get("success_rate", 0.0)

            if recent_success < overall_success:
                areas.append("Recent performance below average - investigate recent changes")

        return areas

    def _load_from_storage(self):
        """Load knowledge base from persistent storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

                # Load artifacts
                if 'artifacts' in data:
                    for item in data['artifacts']:
                        artifact = KnowledgeArtifact.from_dict(item)
                        self.artifacts.append(artifact)

                # Load experiences
                if 'experiences' in data:
                    for item in data['experiences']:
                        # Reconstruct ProblemDefinition
                        problem = ProblemDefinition.from_dict(item['problem'])
                        plan = DecompositionPlan.from_dict(item['decomposition_plan'])

                        # Reconstruct solutions and validations
                        solutions = {
                            k: SolutionAttempt.from_dict(v)
                            for k, v in item.get('solutions', {}).items()
                        }
                        validations = {
                            k: ValidationResult.from_dict(v)
                            for k, v in item.get('validations', {}).items()
                        }

                        experience = ProblemSolvingExperience(
                            experience_id=item['experience_id'],
                            problem=problem,
                            domain=item['domain'],
                            problem_type=item['problem_type'],
                            decomposition_plan=plan,
                            strategy_used=item.get('strategy_used', ''),
                            teams_assigned=item.get('teams_assigned', {}),
                            solutions=solutions,
                            validations=validations,
                            quality_scores=item.get('quality_scores', {}),
                            time_taken=item.get('time_taken', 0.0),
                            success=item.get('success', False),
                            lessons_learned=item.get('lessons_learned', []),
                            artifacts_created=item.get('artifacts_created', []),
                            timestamp=datetime.fromisoformat(item['timestamp'])
                        )
                        self.experiences.append(experience)

                # Load best practices
                if 'best_practices' in data:
                    for practice_id, item in data['best_practices'].items():
                        practice = BestPractice(**item)
                        self.best_practices[practice_id] = practice

                # Load anti-patterns
                if 'anti_patterns' in data:
                    for pattern_id, item in data['anti_patterns'].items():
                        anti = AntiPattern(**item)
                        self.anti_patterns[pattern_id] = anti

            logger.info(f"Loaded {len(self.artifacts)} artifacts and {len(self.experiences)} experiences")

        except FileNotFoundError:
            logger.info("No existing knowledge base found, starting fresh")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error loading knowledge base: {e}")

    def _save_to_storage(self):
        """Save knowledge base to persistent storage."""
        try:
            data = {
                'artifacts': [artifact.to_dict() for artifact in self.artifacts],
                'experiences': [exp.to_dict() for exp in self.experiences],
                'best_practices': {
                    pid: practice.to_dict()
                    for pid, practice in self.best_practices.items()
                },
                'anti_patterns': {
                    pid: anti.to_dict()
                    for pid, anti in self.anti_patterns.items()
                }
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("Knowledge base saved to storage")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error saving knowledge base: {e}")
