"""
Knowledge Artifact Extraction System for Sovereign Decomposition

This module implements intelligent extraction of knowledge artifacts from solved problems,
enabling continuous learning and improvement of the decomposition system.

ARTIFACT TYPES:
- Patterns: Reusable decomposition approaches that work
- Anti-patterns: Approaches to avoid
- Best practices: Proven techniques
- Domain insights: Domain-specific knowledge
- Strategy effectiveness: What works where

CAPABILITIES:
- Extract artifacts from completed problem solving
- Store artifacts persistently
- Retrieve relevant artifacts for new problems
- Track artifact confidence and success rates
- Support continuous learning
"""
from __future__ import annotations


import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics

try:
    from sovereign_data_models import (
        DecompositionPlan, SolutionAttempt, ProblemDefinition,
        SubProblem, ValidationResult, ComplexityScore,
        DecompositionStrategy, generate_id
    )
except ImportError:
    # The sovereign_data_models facade may be partially initialised (e.g. when the
    # kernel schema cannot be fully imported); fall back to module-level symbol
    # lookup and provide minimal shim classes for the optional data models.
    import sovereign_data_models as _sdm

    DecompositionPlan = getattr(_sdm, "DecompositionPlan", object)
    SolutionAttempt = getattr(_sdm, "SolutionAttempt", object)
    ProblemDefinition = getattr(_sdm, "ProblemDefinition", object)
    SubProblem = getattr(_sdm, "SubProblem", object)
    ComplexityScore = getattr(_sdm, "ComplexityScore", object)
    generate_id = getattr(_sdm, "generate_id", lambda prefix="id": f"{prefix}_{id(object())}")


    class ValidationResult:
        """Minimal fallback for the optional validation result data model."""

        pass


    class DecompositionStrategy:
        """Minimal fallback for the optional decomposition strategy data model."""

        pass

logger = logging.getLogger(__name__)


class KnowledgeArtifact:
    """
    Knowledge artifact extracted from solved problems.

    Represents learned knowledge including patterns, best practices,
    domain insights, and anti-patterns discovered through problem solving.
    """

    def __init__(
        self,
        artifact_id: str,
        artifact_type: str,  # "pattern", "anti_pattern", "best_practice", "insight"
        title: str,
        description: str,
        domain: str,
        problem_type: str,
        source_problem_id: str,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        anti_pattern: Optional[str] = None,
        best_practice: Optional[str] = None,
        insight: Optional[str] = None,
        source_sub_problem_ids: Optional[List[str]] = None,
        support_count: int = 1,
        success_rate: float = 0.0,
        related_artifacts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.title = title
        self.description = description
        self.domain = domain
        self.problem_type = problem_type
        self.source_problem_id = source_problem_id
        self.extraction_date = datetime.now()
        self.confidence = confidence
        self.tags = tags or []
        self.pattern = pattern
        self.anti_pattern = anti_pattern
        self.best_practice = best_practice
        self.insight = insight
        self.source_sub_problem_ids = source_sub_problem_ids or []
        self.support_count = support_count
        self.success_rate = success_rate
        self.related_artifacts = related_artifacts or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'artifact_id': self.artifact_id,
            'artifact_type': self.artifact_type,
            'title': self.title,
            'description': self.description,
            'domain': self.domain,
            'problem_type': self.problem_type,
            'source_problem_id': self.source_problem_id,
            'extraction_date': self.extraction_date.isoformat(),
            'confidence': self.confidence,
            'tags': self.tags,
            'pattern': self.pattern,
            'anti_pattern': self.anti_pattern,
            'best_practice': self.best_practice,
            'insight': self.insight,
            'source_sub_problem_ids': self.source_sub_problem_ids,
            'support_count': self.support_count,
            'success_rate': self.success_rate,
            'related_artifacts': self.related_artifacts,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeArtifact':
        """Create from dictionary."""
        data = data.copy()
        if 'extraction_date' in data:
            data['extraction_date'] = datetime.fromisoformat(data['extraction_date'])
        # Remove extraction_date from kwargs since we don't store it as a param
        data.pop('extraction_date', None)
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate artifact data."""
        errors = []
        valid_types = ["pattern", "anti_pattern", "best_practice", "insight"]
        if self.artifact_type not in valid_types:
            errors.append(f"Invalid artifact_type: {self.artifact_type}. Must be one of {valid_types}")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.success_rate <= 1.0:
            errors.append(f"Success rate must be between 0.0 and 1.0, got {self.success_rate}")
        return errors


class KnowledgeArtifactExtractor:
    """
    Extracts knowledge artifacts from solved problems.

    Analyzes completed problem solving to extract:
    - Patterns that work
    - Anti-patterns to avoid
    - Best practices discovered
    - Domain insights
    - Strategy effectiveness
    """

    def __init__(self, artifact_store_path: str = "knowledge_artifacts.json"):
        """
        Initialize with persistent artifact storage.

        Args:
            artifact_store_path: Path to JSON file for artifact storage
        """
        self.artifact_store_path = Path(artifact_store_path)
        self.artifacts: Dict[str, KnowledgeArtifact] = {}
        self._load_artifacts()
        logger.info(f"KnowledgeArtifactExtractor initialized with {len(self.artifacts)} artifacts")

    def _load_artifacts(self):
        """Load artifacts from persistent storage."""
        if self.artifact_store_path.exists():
            try:
                with open(self.artifact_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for artifact_data in data.get('artifacts', []):
                        artifact = KnowledgeArtifact.from_dict(artifact_data)
                        self.artifacts[artifact.artifact_id] = artifact
                logger.info(f"Loaded {len(self.artifacts)} artifacts from {self.artifact_store_path}")
            except (OSError, IOError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load artifacts: {e}", exc_info=True)
                self.artifacts = {}

    def _save_artifacts(self):
        """Save artifacts to persistent storage."""
        try:
            self.artifact_store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'artifacts': [artifact.to_dict() for artifact in self.artifacts.values()],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.artifact_store_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {len(self.artifacts)} artifacts to {self.artifact_store_path}")
        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to save artifacts: {e}", exc_info=True)

    def extract_artifacts(
        self,
        decomposition_plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        validation_results: Dict[str, ValidationResult]
    ) -> List[KnowledgeArtifact]:
        """
        Extract knowledge artifacts from completed problem solving.

        Analyzes:
        1. What decomposition strategy was used
        2. What worked well (high quality sub-solutions)
        3. What didn't work (failures, revisions needed)
        4. Patterns in successful solutions
        5. Domain-specific insights
        6. Team performance patterns

        Args:
            decomposition_plan: The decomposition plan used
            solutions: Solutions generated for each sub-problem
            validation_results: Validation results for each solution

        Returns:
            List of extracted knowledge artifacts
        """
        artifacts = []

        try:
            # Extract strategy patterns
            strategy_artifacts = self.extract_strategy_patterns(decomposition_plan, solutions)
            artifacts.extend(strategy_artifacts)

            # Extract domain insights
            domain_artifacts = self.extract_domain_insights(decomposition_plan, solutions)
            artifacts.extend(domain_artifacts)

            # Extract solution patterns (what worked and what didn't)
            solution_artifacts = self._extract_solution_patterns(
                decomposition_plan, solutions, validation_results
            )
            artifacts.extend(solution_artifacts)

            # Extract complexity patterns
            complexity_artifacts = self._extract_complexity_patterns(decomposition_plan, solutions)
            artifacts.extend(complexity_artifacts)

            # Store all artifacts
            for artifact in artifacts:
                self.store_artifact(artifact)

            logger.info(f"Extracted {len(artifacts)} knowledge artifacts from problem {decomposition_plan.problem_id}")
            return artifacts

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract artifacts: {e}", exc_info=True)
            return []

    def extract_strategy_patterns(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt]
    ) -> List[KnowledgeArtifact]:
        """
        Extract patterns about strategy effectiveness.

        Records:
        - Which strategies work for which problem types
        - Average quality scores by strategy
        - Common failure modes

        Args:
            plan: Decomposition plan used
            solutions: Solutions generated

        Returns:
            List of strategy pattern artifacts
        """
        artifacts = []

        try:
            # Calculate strategy effectiveness
            strategy = plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)
            domain = plan.metadata.get('domain', 'general')
            problem_type = plan.metadata.get('problem_type', 'unknown')

            # Calculate quality scores
            quality_scores = []
            for sol in solutions.values():
                if hasattr(sol, 'confidence_score'):
                    quality_scores.append(sol.confidence_score)

            if quality_scores:
                avg_quality = statistics.mean(quality_scores)
                success_rate = sum(1 for s in quality_scores if s >= 0.7) / len(quality_scores)

                # Create pattern artifact if strategy was effective
                if avg_quality >= 0.7 and success_rate >= 0.7:
                    artifact = KnowledgeArtifact(
                        artifact_id=generate_id("strategy_pattern"),
                        artifact_type="pattern",
                        title=f"Effective Strategy: {strategy} for {problem_type}",
                        description=f"The {strategy} decomposition strategy showed strong effectiveness for {problem_type} problems in the {domain} domain",
                        domain=domain,
                        problem_type=problem_type,
                        source_problem_id=plan.problem_id,
                        pattern=strategy,
                        confidence=avg_quality,
                        success_rate=success_rate,
                        support_count=len(solutions),
                        tags=["strategy", problem_type, domain, "effective"],
                        metadata={
                            'avg_quality_score': avg_quality,
                            'num_sub_problems': len(plan.sub_problems),
                            'strategy': strategy
                        }
                    )
                    artifacts.append(artifact)

                # Create anti-pattern if strategy was ineffective
                elif avg_quality < 0.5:
                    artifact = KnowledgeArtifact(
                        artifact_id=generate_id("strategy_anti_pattern"),
                        artifact_type="anti_pattern",
                        title=f"Ineffective Strategy: {strategy} for {problem_type}",
                        description=f"The {strategy} decomposition strategy showed poor effectiveness for {problem_type} problems in the {domain} domain. Consider alternative strategies.",
                        domain=domain,
                        problem_type=problem_type,
                        source_problem_id=plan.problem_id,
                        anti_pattern=strategy,
                        confidence=1.0 - avg_quality,  # Higher confidence when it's clearly bad
                        success_rate=success_rate,
                        support_count=len(solutions),
                        tags=["strategy", problem_type, domain, "avoid"],
                        metadata={
                            'avg_quality_score': avg_quality,
                            'num_sub_problems': len(plan.sub_problems),
                            'strategy': strategy
                        }
                    )
                    artifacts.append(artifact)

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract strategy patterns: {e}", exc_info=True)

        return artifacts

    def extract_domain_insights(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt]
    ) -> List[KnowledgeArtifact]:
        """
        Extract domain-specific insights.

        Records:
        - Common patterns in this domain
        - Typical complexity distribution
        - Frequently used approaches
        - Domain-specific best practices

        Args:
            plan: Decomposition plan used
            solutions: Solutions generated

        Returns:
            List of domain insight artifacts
        """
        artifacts = []

        try:
            domain = plan.metadata.get('domain', 'general')
            problem_type = plan.metadata.get('problem_type', 'unknown')

            # Analyze complexity distribution
            complexity_scores = []
            for sp in plan.sub_problems:
                if hasattr(sp, 'complexity_score') and sp.complexity_score:
                    if isinstance(sp.complexity_score, dict):
                        complexity_scores.append(sp.complexity_score.get('overall_complexity', 5.0))
                    elif hasattr(sp.complexity_score, 'overall_complexity'):
                        complexity_scores.append(sp.complexity_score.overall_complexity)

            if complexity_scores:
                avg_complexity = statistics.mean(complexity_scores)

                # Extract insight about typical complexity
                artifact = KnowledgeArtifact(
                    artifact_id=generate_id("domain_insight"),
                    artifact_type="insight",
                    title=f"Typical Complexity for {problem_type} in {domain}",
                    description=f"Problems of type {problem_type} in domain {domain} typically have an average sub-problem complexity of {avg_complexity:.2f}/10.0",
                    domain=domain,
                    problem_type=problem_type,
                    source_problem_id=plan.problem_id,
                    insight=f"average_complexity_{avg_complexity:.2f}",
                    confidence=0.8,
                    support_count=len(complexity_scores),
                    tags=["domain", "complexity", domain, problem_type],
                    metadata={
                        'avg_complexity': avg_complexity,
                        'num_sub_problems': len(complexity_scores),
                        'complexity_range': [min(complexity_scores), max(complexity_scores)]
                    }
                )
                artifacts.append(artifact)

            # Analyze sub-problem count patterns
            num_sub_problems = len(plan.sub_problems)
            if num_sub_problems > 0:
                artifact = KnowledgeArtifact(
                    artifact_id=generate_id("domain_pattern"),
                    artifact_type="pattern",
                    title=f"Typical Decomposition Granularity for {problem_type}",
                    description=f"Problems of type {problem_type} in domain {domain} were effectively decomposed into {num_sub_problems} sub-problems",
                    domain=domain,
                    problem_type=problem_type,
                    source_problem_id=plan.problem_id,
                    pattern=f"decompose_into_{num_sub_problems}_subproblems",
                    confidence=0.7,
                    support_count=1,
                    tags=["domain", "granularity", domain, problem_type],
                    metadata={
                        'num_sub_problems': num_sub_problems,
                        'strategy': plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)
                    }
                )
                artifacts.append(artifact)

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract domain insights: {e}", exc_info=True)

        return artifacts

    def extract_team_patterns(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        team_assignments: Dict[str, Any]
    ) -> List[KnowledgeArtifact]:
        """
        Extract patterns about team performance.

        Records:
        - Which teams excel at which types of problems
        - Team-specific success rates
        - Collaboration patterns

        Args:
            plan: Decomposition plan used
            solutions: Solutions generated
            team_assignments: Team assignments for sub-problems

        Returns:
            List of team performance artifacts
        """
        artifacts = []

        try:
            # Group solutions by team
            team_performance = defaultdict(list)
            for sub_problem_id, solution in solutions.items():
                team_id = solution.team_id if hasattr(solution, 'team_id') else 'unknown'
                quality = solution.confidence_score if hasattr(solution, 'confidence_score') else 0.5
                team_performance[team_id].append(quality)

            # Extract patterns for each team
            for team_id, qualities in team_performance.items():
                if len(qualities) >= 2:  # Only if we have multiple data points
                    avg_quality = statistics.mean(qualities)
                    success_rate = sum(1 for q in qualities if q >= 0.7) / len(qualities)

                    # Get problem types this team worked on
                    sub_problem_ids = [sp_id for sp_id, sol in solutions.items()
                                      if hasattr(sol, 'team_id') and sol.team_id == team_id]
                    sub_problems = [sp for sp in plan.sub_problems if sp.id in sub_problem_ids]
                    problem_types = set(sp.type.value if hasattr(sp.type, 'value') else str(sp.type)
                                       for sp in sub_problems)

                    for prob_type in problem_types:
                        if avg_quality >= 0.75:
                            artifact = KnowledgeArtifact(
                                artifact_id=generate_id("team_pattern"),
                                artifact_type="pattern",
                                title=f"Team {team_id} Excellence in {prob_type}",
                                description=f"Team {team_id} shows strong performance ({avg_quality:.2f} avg quality) on {prob_type} sub-problems",
                                domain=plan.metadata.get('domain', 'general'),
                                problem_type=prob_type,
                                source_problem_id=plan.problem_id,
                                source_sub_problem_ids=sub_problem_ids,
                                pattern=f"team_{team_id}_for_{prob_type}",
                                confidence=avg_quality,
                                success_rate=success_rate,
                                support_count=len(qualities),
                                tags=["team", team_id, prob_type, "effective"],
                                metadata={
                                    'team_id': team_id,
                                    'avg_quality': avg_quality,
                                    'num_assignments': len(qualities)
                                }
                            )
                            artifacts.append(artifact)

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract team patterns: {e}", exc_info=True)

        return artifacts

    def _extract_solution_patterns(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt],
        validation_results: Dict[str, ValidationResult]
    ) -> List[KnowledgeArtifact]:
        """Extract patterns from successful and failed solutions."""
        artifacts = []

        try:
            for sub_problem_id, solution in solutions.items():
                validation = validation_results.get(sub_problem_id)

                # Check if solution was successful
                is_successful = False
                if validation:
                    is_successful = validation.passed if hasattr(validation, 'passed') else False
                elif hasattr(solution, 'confidence_score'):
                    is_successful = solution.confidence_score >= 0.7

                # Get sub-problem details
                sub_problem = next((sp for sp in plan.sub_problems if sp.id == sub_problem_id), None)
                if not sub_problem:
                    continue

                domain = plan.metadata.get('domain', 'general')
                problem_type = sub_problem.type.value if hasattr(sub_problem.type, 'value') else str(sub_problem.type)

                if is_successful:
                    # Extract best practice from successful solution
                    approach = solution.approach if hasattr(solution, 'approach') else 'unknown'
                    artifact = KnowledgeArtifact(
                        artifact_id=generate_id("solution_pattern"),
                        artifact_type="best_practice",
                        title=f"Effective Approach for {problem_type}",
                        description=f"Approach '{approach}' was effective for {problem_type} sub-problem",
                        domain=domain,
                        problem_type=problem_type,
                        source_problem_id=plan.problem_id,
                        source_sub_problem_ids=[sub_problem_id],
                        best_practice=approach,
                        confidence=solution.confidence_score if hasattr(solution, 'confidence_score') else 0.8,
                        success_rate=1.0,
                        support_count=1,
                        tags=["solution", problem_type, approach, "effective"],
                        metadata={
                            'approach': approach,
                            'sub_problem_title': sub_problem.title
                        }
                    )
                    artifacts.append(artifact)
                else:
                    # Extract anti-pattern from failed solution
                    approach = solution.approach if hasattr(solution, 'approach') else 'unknown'
                    artifact = KnowledgeArtifact(
                        artifact_id=generate_id("solution_anti_pattern"),
                        artifact_type="anti_pattern",
                        title=f"Ineffective Approach for {problem_type}",
                        description=f"Approach '{approach}' was ineffective for {problem_type} sub-problem. Consider alternatives.",
                        domain=domain,
                        problem_type=problem_type,
                        source_problem_id=plan.problem_id,
                        source_sub_problem_ids=[sub_problem_id],
                        anti_pattern=approach,
                        confidence=0.8,
                        success_rate=0.0,
                        support_count=1,
                        tags=["solution", problem_type, approach, "avoid"],
                        metadata={
                            'approach': approach,
                            'sub_problem_title': sub_problem.title
                        }
                    )
                    artifacts.append(artifact)

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract solution patterns: {e}", exc_info=True)

        return artifacts

    def _extract_complexity_patterns(
        self,
        plan: DecompositionPlan,
        solutions: Dict[str, SolutionAttempt]
    ) -> List[KnowledgeArtifact]:
        """Extract patterns related to complexity and effort."""
        artifacts = []

        try:
            # Analyze relationship between complexity and solution quality
            complexity_quality_pairs = []
            for sub_problem in plan.sub_problems:
                solution = solutions.get(sub_problem.id)
                if not solution or not hasattr(solution, 'confidence_score'):
                    continue

                complexity = 5.0  # default
                if hasattr(sub_problem, 'complexity_score') and sub_problem.complexity_score:
                    if isinstance(sub_problem.complexity_score, dict):
                        complexity = sub_problem.complexity_score.get('overall_complexity', 5.0)
                    elif hasattr(sub_problem.complexity_score, 'overall_complexity'):
                        complexity = sub_problem.complexity_score.overall_complexity

                complexity_quality_pairs.append((complexity, solution.confidence_score))

            if complexity_quality_pairs:
                # Check if higher complexity correlates with lower quality
                high_complex_low_quality = sum(1 for c, q in complexity_quality_pairs
                                              if c >= 7.0 and q < 0.6)
                if high_complex_low_quality >= len(complexity_quality_pairs) / 2:
                    artifact = KnowledgeArtifact(
                        artifact_id=generate_id("complexity_pattern"),
                        artifact_type="insight",
                        title="High Complexity Sub-problems Challenge",
                        description=f"Sub-problems with high complexity (>=7.0) showed lower solution quality. Consider further decomposition or specialized approaches.",
                        domain=plan.metadata.get('domain', 'general'),
                        problem_type=plan.metadata.get('problem_type', 'unknown'),
                        source_problem_id=plan.problem_id,
                        insight="high_complexity_requires_specialization",
                        confidence=0.75,
                        support_count=high_complex_low_quality,
                        tags=["complexity", "quality", "insight"],
                        metadata={
                            'num_high_complexity': high_complex_low_quality,
                            'total_sub_problems': len(complexity_quality_pairs)
                        }
                    )
                    artifacts.append(artifact)

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract complexity patterns: {e}", exc_info=True)

        return artifacts

    def store_artifact(self, artifact: KnowledgeArtifact):
        """
        Store artifact in knowledge base.

        Args:
            artifact: Artifact to store
        """
        try:
            # Check if similar artifact already exists
            for existing_id, existing_artifact in self.artifacts.items():
                if (existing_artifact.artifact_type == artifact.artifact_type and
                    existing_artifact.domain == artifact.domain and
                    existing_artifact.problem_type == artifact.problem_type):
                    # Update existing artifact
                    existing_artifact.support_count += artifact.support_count
                    # Update confidence using weighted average
                    total_weight = existing_artifact.support_count
                    existing_artifact.confidence = (
                        (existing_artifact.confidence * (total_weight - artifact.support_count) +
                         artifact.confidence * artifact.support_count) / total_weight
                    )
                    existing_artifact.success_rate = (
                        (existing_artifact.success_rate * (total_weight - artifact.support_count) +
                         artifact.success_rate * artifact.support_count) / total_weight
                    )
                    # Add related artifact link
                    if artifact.artifact_id not in existing_artifact.related_artifacts:
                        existing_artifact.related_artifacts.append(artifact.artifact_id)
                    # Update timestamp
                    existing_artifact.extraction_date = datetime.now()

                    self._save_artifacts()
                    logger.debug(f"Updated existing artifact {existing_id}")
                    return

            # Add new artifact
            self.artifacts[artifact.artifact_id] = artifact
            self._save_artifacts()
            logger.debug(f"Stored new artifact {artifact.artifact_id}")

        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to store artifact: {e}", exc_info=True)

    def retrieve_relevant_artifacts(
        self,
        problem: ProblemDefinition,
        domain: str
    ) -> List[KnowledgeArtifact]:
        """
        Retrieve artifacts relevant to current problem.

        Uses:
        - Domain matching
        - Problem type similarity
        - Strategy recommendations

        Args:
            problem: Current problem definition
            domain: Problem domain

        Returns:
            List of relevant artifacts, sorted by confidence
        """
        try:
            relevant_artifacts = []

            problem_type = problem.problem_type.value if hasattr(problem.problem_type, 'value') else str(problem.problem_type)

            for artifact in self.artifacts.values():
                # Score artifact relevance
                relevance_score = 0.0

                # Domain match (highest weight)
                if artifact.domain == domain:
                    relevance_score += 0.4
                elif domain in artifact.tags:
                    relevance_score += 0.2

                # Problem type match
                if artifact.problem_type == problem_type:
                    relevance_score += 0.3
                elif problem_type in artifact.tags:
                    relevance_score += 0.15

                # Artifact quality
                relevance_score += artifact.confidence * 0.2
                relevance_score += artifact.success_rate * 0.1

                # Only include if minimum relevance threshold met
                if relevance_score >= 0.3:
                    artifact.metadata['relevance_score'] = relevance_score
                    relevant_artifacts.append(artifact)

            # Sort by relevance score
            relevant_artifacts.sort(key=lambda a: a.metadata.get('relevance_score', 0.0), reverse=True)

            logger.info(f"Retrieved {len(relevant_artifacts)} relevant artifacts for {domain}/{problem_type}")
            return relevant_artifacts[:10]  # Return top 10

        except (ValueError, AttributeError, KeyError, TypeError) as e:
            logger.error(f"Failed to retrieve artifacts: {e}", exc_info=True)
            return []

    def get_artifacts_by_type(self, artifact_type: str) -> List[KnowledgeArtifact]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts.values() if a.artifact_type == artifact_type]

    def get_artifacts_by_domain(self, domain: str) -> List[KnowledgeArtifact]:
        """Get all artifacts for a specific domain."""
        return [a for a in self.artifacts.values() if a.domain == domain]

    def get_artifact_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored artifacts."""
        stats = {
            'total_artifacts': len(self.artifacts),
            'by_type': defaultdict(int),
            'by_domain': defaultdict(int),
            'avg_confidence': 0.0,
            'avg_success_rate': 0.0,
            'high_confidence_count': 0
        }

        if not self.artifacts:
            return stats

        confidences = []
        success_rates = []

        for artifact in self.artifacts.values():
            stats['by_type'][artifact.artifact_type] += 1
            stats['by_domain'][artifact.domain] += 1
            confidences.append(artifact.confidence)
            success_rates.append(artifact.success_rate)
            if artifact.confidence >= 0.8:
                stats['high_confidence_count'] += 1

        stats['avg_confidence'] = statistics.mean(confidences) if confidences else 0.0
        stats['avg_success_rate'] = statistics.mean(success_rates) if success_rates else 0.0

        return dict(stats)
