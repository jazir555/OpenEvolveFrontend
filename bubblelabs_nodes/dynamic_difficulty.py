"""
Dynamic Difficulty Adjustment for OpenEvolve Gauntlet System

Automatically adjusts the difficulty of problems based on team performance,
domain-specific baselines, and historical success rates.

Key Features:
- Per-team performance tracking
- Domain-specific difficulty baselines
- Dynamic difficulty selection
- Performance trend analysis
- Adaptive thresholds
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for problems"""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    EXPERT = "expert"


@dataclass
class PerformanceRecord:
    """Record of a team's performance on a problem"""
    problem_id: str
    team_id: str
    domain: str
    difficulty: DifficultyLevel
    success: bool
    score: float
    execution_time: float
    timestamp: datetime = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class TeamPerformance:
    """Aggregated performance metrics for a team"""
    team_id: str
    domain: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    average_score: float = 0.0
    average_execution_time: float = 0.0
    recent_performance: List[PerformanceRecord] = field(default_factory=list)
    performance_history: List[PerformanceRecord] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    @property
    def trend(self) -> str:
        """Determine performance trend"""
        if len(self.recent_performance) < 3:
            return "unknown"

        recent_scores = [r.score for r in self.recent_performance[-5:]]
        if len(recent_scores) < 3:
            return "unknown"

        # Simple trend detection
        first_half = recent_scores[:len(recent_scores)//2]
        second_half = recent_scores[len(recent_scores)//2:]

        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0

        if avg_second > avg_first + 10:
            return "improving"
        elif avg_second < avg_first - 10:
            return "declining"
        else:
            return "stable"


class TeamPerformanceTracker:
    """
    Tracks team performance across problems and domains.
    """

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.records: List[PerformanceRecord] = []
        self.team_performance: Dict[str, Dict[str, TeamPerformance]] = {}

    def record_performance(
        self,
        problem_id: str,
        team_id: str,
        domain: str,
        difficulty: DifficultyLevel,
        success: bool,
        score: float,
        execution_time: float,
        metadata: Dict[str, Any] = None
    ) -> PerformanceRecord:
        """
        Record a team's performance on a problem.

        Args:
            problem_id: Problem identifier
            team_id: Team identifier
            domain: Problem domain
            difficulty: Difficulty level
            success: Whether the team succeeded
            score: Score (0-100)
            execution_time: Time taken
            metadata: Additional metadata

        Returns:
            PerformanceRecord
        """
        record = PerformanceRecord(
            problem_id=problem_id,
            team_id=team_id,
            domain=domain,
            difficulty=difficulty,
            success=success,
            score=score,
            execution_time=execution_time,
            metadata=metadata or {}
        )

        # Add to records
        self.records.append(record)

        # Trim to history size
        if len(self.records) > self.history_size:
            self.records = self.records[-self.history_size:]

        # Update team performance
        if team_id not in self.team_performance:
            self.team_performance[team_id] = {}

        if domain not in self.team_performance[team_id]:
            self.team_performance[team_id][domain] = TeamPerformance(
                team_id=team_id,
                domain=domain
            )

        team_perf = self.team_performance[team_id][domain]
        team_perf.total_attempts += 1
        if success:
            team_perf.successful_attempts += 1
        else:
            team_perf.failed_attempts += 1

        # Update averages
        team_perf.performance_history.append(record)
        if len(team_perf.performance_history) > self.history_size:
            team_perf.performance_history = team_perf.performance_history[-self.history_size:]

        # Update recent performance (last 10)
        team_perf.recent_performance = team_perf.performance_history[-10:]

        # Recalculate averages
        team_perf.average_score = statistics.mean(
            [r.score for r in team_perf.performance_history]
        )
        team_perf.average_execution_time = statistics.mean(
            [r.execution_time for r in team_perf.performance_history]
        )

        logger.info(
            f"Recorded performance: {team_id} on {domain} - "
            f"{'Success' if success else 'Failure'} (score: {score:.0f})"
        )

        return record

    def get_performance(
        self,
        team_id: str,
        domain: str
    ) -> Optional[TeamPerformance]:
        """Get team performance for a specific domain"""
        if team_id not in self.team_performance:
            return None
        return self.team_performance[team_id].get(domain)

    def get_team_average_performance(self, team_id: str) -> Dict[str, Any]:
        """Get average performance across all domains for a team"""
        if team_id not in self.team_performance:
            return {
                'total_attempts': 0,
                'overall_success_rate': 0.0,
                'average_score': 0.0,
                'domains': []
            }

        domains = list(self.team_performance[team_id].values())

        if not domains:
            return {
                'total_attempts': 0,
                'overall_success_rate': 0.0,
                'average_score': 0.0,
                'domains': []
            }

        total_attempts = sum(d.total_attempts for d in domains)
        successful_attempts = sum(d.successful_attempts for d in domains)
        average_score = statistics.mean([d.average_score for d in domains])

        return {
            'total_attempts': total_attempts,
            'overall_success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0.0,
            'average_score': average_score,
            'domains': list(self.team_performance[team_id].keys())
        }

    def detect_trend(self, team_id: str, domain: str) -> str:
        """Detect performance trend for a team in a domain"""
        perf = self.get_performance(team_id, domain)
        if not perf:
            return "unknown"
        return perf.trend


class DomainClassifier:
    """
    Classifies problems into domains for domain-specific difficulty adjustment.
    """

    def __init__(self):
        self.domain_keywords = {
            'web_development': [
                'api', 'web', 'http', 'rest', 'frontend', 'backend',
                'server', 'client', 'javascript', 'html', 'css'
            ],
            'data_processing': [
                'data', 'etl', 'pipeline', 'transform', 'aggregate',
                'database', 'sql', 'query', 'analytics'
            ],
            'machine_learning': [
                'ml', 'model', 'training', 'prediction', 'classification',
                'regression', 'neural', 'deep learning', 'ai'
            ],
            'security': [
                'security', 'auth', 'encrypt', 'vulnerability', 'penetration',
                'firewall', 'intrusion', 'malware'
            ],
            'devops': [
                'deploy', 'ci/cd', 'docker', 'kubernetes', 'infrastructure',
                'monitoring', 'logging', 'scaling'
            ],
            'mobile': [
                'mobile', 'ios', 'android', 'app', 'responsive',
                'touch', 'gesture', 'location'
            ],
        }

        self.default_domain = 'general'

    def classify(self, problem: Dict[str, Any]) -> str:
        """
        Classify a problem into a domain.

        Args:
            problem: Problem definition

        Returns:
            Domain name
        """
        # Check if domain is explicitly specified
        if 'domain' in problem:
            return problem['domain']

        # Classify based on statement and requirements
        text = problem.get('statement', '').lower()
        requirements = problem.get('requirements', [])
        if isinstance(requireabilities, list):
            for req in requirements:
                text += f" {str(req).lower()}"

        # Check against domain keywords
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            domain_scores[domain] = score

        # Return domain with highest score
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)

        return self.default_domain


class DifficultyAdjuster:
    """
    Adjusts difficulty based on team performance and domain baselines.
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_difficulty: DifficultyLevel = DifficultyLevel.VERY_EASY,
        max_difficulty: DifficultyLevel = DifficultyLevel.EXPERT
    ):
        self.sensitivity = sensitivity  # 0-1, how aggressively to adjust
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

        # Domain-specific baselines (success rates)
        self.domain_baselines = {
            'web_development': 0.70,
            'data_processing': 0.60,
            'machine_learning': 0.50,
            'security': 0.55,
            'devops': 0.65,
            'mobile': 0.60,
            'general': 0.65,
        }

    def select_difficulty(
        self,
        problem: Dict[str, Any],
        team_id: str,
        tracker: TeamPerformanceTracker,
        classifier: DomainClassifier
    ) -> DifficultyLevel:
        """
        Select appropriate difficulty for a problem and team.

        Args:
            problem: Problem definition
            team_id: Team identifier
            tracker: Performance tracker
            classifier: Domain classifier

        Returns:
            Selected difficulty level
        """
        # Classify domain
        domain = classifier.classify(problem)

        # Get team performance in domain
        perf = tracker.get_performance(team_id, domain)

        if not perf or perf.total_attempts < 3:
            # Not enough data, use medium difficulty
            logger.info(f"Insufficient data for {team_id} in {domain}, using MEDIUM")
            return DifficultyLevel.MEDIUM

        # Calculate performance metrics
        success_rate = perf.success_rate
        baseline = self.domain_baselines.get(domain, 0.65)

        # Compare to baseline
        performance_ratio = success_rate / baseline if baseline > 0 else 1.0

        # Select difficulty based on performance
        if performance_ratio > 1.3:
            # Performing very well, increase difficulty
            return self._increase_difficulty(perf.average_score)
        elif performance_ratio < 0.7:
            # Struggling, decrease difficulty
            return self._decrease_difficulty(perf.average_score)
        elif perf.trend == "improving":
            # Improving, slightly increase difficulty
            return self._increase_difficulty(perf.average_score, step=1)
        elif perf.trend == "declining":
            # Declining, slightly decrease difficulty
            return self._decrease_difficulty(perf.average_score, step=1)
        else:
            # Stable, maintain current difficulty
            return self._estimate_current_difficulty(perf.average_score)

    def _increase_difficulty(
        self,
        avg_score: float,
        step: int = 2
    ) -> DifficultyLevel:
        """Increase difficulty level"""
        levels = list(DifficultyLevel)
        current_idx = self._score_to_level_index(avg_score)

        new_idx = min(len(levels) - 1, current_idx + step)
        return levels[new_idx]

    def _decrease_difficulty(
        self,
        avg_score: float,
        step: int = 2
    ) -> DifficultyLevel:
        """Decrease difficulty level"""
        levels = list(DifficultyLevel)
        current_idx = self._score_to_level_index(avg_score)

        new_idx = max(0, current_idx - step)
        return levels[new_idx]

    def _estimate_current_difficulty(self, avg_score: float) -> DifficultyLevel:
        """Estimate current difficulty from average score"""
        levels = list(DifficultyLevel)
        idx = self._score_to_level_index(avg_score)
        return levels[idx]

    def _score_to_level_index(self, score: float) -> int:
        """Convert score to difficulty level index"""
        if score >= 90:
            return 5  # EXPERT
        elif score >= 75:
            return 4  # VERY_HARD
        elif score >= 60:
            return 3  # HARD
        elif score >= 45:
            return 2  # MEDIUM
        elif score >= 30:
            return 1  # EASY
        else:
            return 0  # VERY_EASY


class DynamicDifficultySystem:
    """
    Main system for dynamic difficulty adjustment.

    Integrates performance tracking, domain classification, and
    difficulty adjustment.
    """

    def __init__(
        self,
        history_size: int = 100,
        sensitivity: float = 0.5
    ):
        self.tracker = TeamPerformanceTracker(history_size=history_size)
        self.classifier = DomainClassifier()
        self.adjuster = DifficultyAdjuster(sensitivity=sensitivity)

    async def configure_problem(
        self,
        problem: Dict[str, Any],
        team_id: str
    ) -> Dict[str, Any]:
        """
        Configure a problem with appropriate difficulty for a team.

        Args:
            problem: Problem definition
            team_id: Team identifier

        Returns:
            Configured problem with difficulty
        """
        # Select difficulty
        difficulty = self.adjuster.select_difficulty(
            problem=problem,
            team_id=team_id,
            tracker=self.tracker,
            classifier=self.classifier
        )

        # Classify domain
        domain = self.classifier.classify(problem)

        # Get team stats
        team_stats = self.tracker.get_team_average_performance(team_id)

        logger.info(
            f"Configured problem for {team_id}: "
            f"domain={domain}, difficulty={difficulty.value}, "
            f"team_success_rate={team_stats['overall_success_rate']:.1%}"
        )

        # Return configured problem
        return {
            **problem,
            'domain': domain,
            'difficulty': difficulty.value,
            'team_context': {
                'team_success_rate': team_stats['overall_success_rate'],
                'team_avg_score': team_stats['average_score'],
                'team_domains': team_stats['domains'],
            }
        }

    async def record_result(
        self,
        problem_id: str,
        team_id: str,
        domain: str,
        difficulty: str,
        success: bool,
        score: float,
        execution_time: float
    ):
        """Record the result of a team solving a problem"""
        difficulty_level = DifficultyLevel(difficulty)

        self.tracker.record_performance(
            problem_id=problem_id,
            team_id=team_id,
            domain=domain,
            difficulty=difficulty_level,
            success=success,
            score=score,
            execution_time=execution_time
        )

    def get_team_stats(self, team_id: str) -> Dict[str, Any]:
        """Get comprehensive team statistics"""
        return self.tracker.get_team_average_performance(team_id)

    def get_domain_difficulty_baselines(self) -> Dict[str, float]:
        """Get current difficulty baselines by domain"""
        return self.adjuster.domain_baselines.copy()


# Convenience function
def create_difficulty_system(
    history_size: int = 100,
    sensitivity: float = 0.5
) -> DynamicDifficultySystem:
    """Create a dynamic difficulty adjustment system"""
    return DynamicDifficultySystem(
        history_size=history_size,
        sensitivity=sensitivity
    )


# Example usage
async def demo_dynamic_difficulty():
    """Demonstration of dynamic difficulty adjustment"""

    system = create_difficulty_system()

    # Simulate team performance
    team_id = "blue_team_1"

    # Example problem
    problem = {
        'id': 'problem_1',
        'statement': 'Build a REST API for user management',
        'requirements': ['authentication', 'crud', 'validation']
    }

    print("\n" + "=" * 60)
    print("Dynamic Difficulty Adjustment Demo")
    print("=" * 60)

    # Configure problem (first time, no history)
    configured = await system.configure_problem(problem, team_id)
    print(f"\nInitial configuration:")
    print(f"  Domain: {configured['domain']}")
    print(f"  Difficulty: {configured['difficulty']}")
    print(f"  Team context: {configured['team_context']}")

    # Record some successful attempts
    print(f"\nRecording team performance...")
    for i in range(5):
        await system.record_result(
            problem_id=f"problem_{i}",
            team_id=team_id,
            domain=configured['domain'],
            difficulty=configured['difficulty'],
            success=True,
            score=85 + i * 2,  # Improving
            execution_time=300 - i * 10
        )

    # Configure again (should be harder now)
    configured2 = await system.configure_problem(problem, team_id)
    print(f"\nAfter successful attempts:")
    print(f"  Domain: {configured2['domain']}")
    print(f"  Difficulty: {configured2['difficulty']}")

    # Show team stats
    stats = system.get_team_stats(team_id)
    print(f"\nTeam Statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Success rate: {stats['overall_success_rate']:.1%}")
    print(f"  Average score: {stats['average_score']:.1f}")
    print(f"  Domains: {stats['domains']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_dynamic_difficulty())
