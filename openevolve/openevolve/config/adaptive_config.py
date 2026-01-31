"""
Adaptive Configuration

This module provides automatic configuration tuning based on runtime performance
metrics. Enables the system to adapt its parameters during evolution.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import deque
from pydantic import BaseModel

from ..unified.config import UnifiedEvolutionConfig


logger = logging.getLogger(__name__)


class PerformanceMetrics(BaseModel):
    """Performance metrics for adaptation"""
    iteration: int
    fitness: float
    diversity: float
    convergence_rate: float
    improvement_rate: float
    evaluation_time: float
    success_rate: float
    timestamp: datetime


class AdaptiveConfigurator:
    """
    Automatically tune configuration based on performance

    Features:
    - Performance trend analysis
    - Automatic parameter adjustment suggestions
    - Learning from historical data
    - Multi-objective optimization
    """

    def __init__(
        self,
        base_config: UnifiedEvolutionConfig,
        history_size: int = 100
    ):
        """
        Initialize adaptive configurator

        Args:
            base_config: Base configuration to adapt
            history_size: Maximum number of performance records to keep
        """
        self.base_config = base_config
        self.performance_history: deque = deque(maxlen=history_size)
        self.parameter_impact: Dict[str, float] = {}
        self.adjustment_history: List[Dict] = []

        # Adaptation thresholds
        self.slow_convergence_threshold = 0.01  # < 1% improvement
        self.low_diversity_threshold = 0.3  # < 30% diversity
        self.stagnation_iterations = 20  # No improvement for N iterations

    async def adapt_configuration(
        self,
        performance_metrics: Dict[str, float],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Suggest configuration adaptations based on recent performance

        Args:
            performance_metrics: Recent performance (convergence, diversity, etc.)
            iteration: Current iteration number

        Returns:
            Suggested parameter adjustments
        """
        # Store performance
        metrics = PerformanceMetrics(
            iteration=iteration,
            fitness=performance_metrics.get("fitness", 0.0),
            diversity=performance_metrics.get("diversity", 1.0),
            convergence_rate=performance_metrics.get("convergence_rate", 0.0),
            improvement_rate=performance_metrics.get("improvement_rate", 0.0),
            evaluation_time=performance_metrics.get("evaluation_time", 0.0),
            success_rate=performance_metrics.get("success_rate", 1.0),
            timestamp=datetime.utcnow()
        )

        self.performance_history.append(metrics)

        # Need minimum history for analysis
        if len(self.performance_history) < 10:
            logger.debug("Insufficient history for adaptation")
            return {}

        # Analyze and suggest adaptations
        suggestions = await self._analyze_and_suggest(metrics)

        if suggestions:
            logger.info(f"Adaptation suggestions at iteration {iteration}: {suggestions}")

            # Record adjustment
            self.adjustment_history.append({
                "iteration": iteration,
                "suggestions": suggestions,
                "timestamp": datetime.utcnow()
            })

        return suggestions

    async def _analyze_and_suggest(
        self,
        current_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Analyze performance and generate suggestions"""
        suggestions = {}

        # Check convergence speed
        if self._is_slow_convergence():
            suggestions["database.population_size"] = int(
                self.base_config.database.population_size * 1.5
            )
            suggestions["common.concurrency"] = min(
                self.base_config.common.concurrency + 2,
                20  # Max cap
            )

        # Check diversity
        if self._is_low_diversity(current_metrics):
            suggestions["database.exploration_rate"] = min(
                self.base_config.database.exploration_rate + 0.2,
                0.8  # Max cap
            )

        # Check if stuck in local optima
        if self._is_stuck_in_local_optima():
            suggestions["evolution_mode"] = "qd"  # Switch to quality diversity
            if self.base_config.qd is None:
                # Initialize QD config
                from ..unified.config import QDConfig
                suggestions["qd"] = QDConfig().model_dump()

        # Check evaluation efficiency
        if self._is_slow_evaluation():
            suggestions["evaluator.cascade_evaluation"] = True
            suggestions["evaluator.parallel_evaluations"] = min(
                self.base_config.evaluator.parallel_evaluations + 2,
                16  # Max cap
            )

        # Check if high success rate (can be more aggressive)
        if self._is_high_success_rate():
            suggestions["database.exploitation_ratio"] = min(
                self.base_config.database.exploitation_ratio + 0.1,
                0.9  # Max cap
            )

        return suggestions

    def _is_slow_convergence(self) -> bool:
        """Detect if convergence is slower than expected"""
        if len(self.performance_history) < 10:
            return False

        recent = list(self.performance_history)[-10:]
        fitness_values = [m.fitness for m in recent]

        # Calculate improvement
        improvement = (max(fitness_values) - min(fitness_values))
        relative_improvement = improvement / (abs(min(fitness_values)) + 1e-6)

        return relative_improvement < self.slow_convergence_threshold

    def _is_low_diversity(self, metrics: PerformanceMetrics) -> bool:
        """Detect if population diversity is too low"""
        return metrics.diversity < self.low_diversity_threshold

    def _is_stuck_in_local_optima(self) -> bool:
        """Detect if stuck in local optima"""
        if len(self.performance_history) < self.stagnation_iterations:
            return False

        recent = list(self.performance_history)[-self.stagnation_iterations:]
        recent_best = max(m.fitness for m in recent)
        overall_best = max(m.fitness for m in self.performance_history)

        # Within 1% of best but no progress
        return recent_best >= overall_best * 0.99

    def _is_slow_evaluation(self) -> bool:
        """Detect if evaluations are taking too long"""
        if len(self.performance_history) < 10:
            return False

        recent = list(self.performance_history)[-10:]
        avg_time = sum(m.evaluation_time for m in recent) / len(recent)

        # If averaging > 10 seconds per evaluation
        return avg_time > 10.0

    def _is_high_success_rate(self) -> bool:
        """Detect if success rate is consistently high"""
        if len(self.performance_history) < 10:
            return False

        recent = list(self.performance_history)[-10:]
        avg_success = sum(m.success_rate for m in recent) / len(recent)

        return avg_success > 0.9

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.performance_history:
            return {}

        metrics_list = list(self.performance_history)

        return {
            "total_iterations": len(metrics_list),
            "best_fitness": max(m.fitness for m in metrics_list),
            "current_fitness": metrics_list[-1].fitness,
            "average_diversity": sum(m.diversity for m in metrics_list) / len(metrics_list),
            "average_evaluation_time": sum(m.evaluation_time for m in metrics_list) / len(metrics_list),
            "total_adjustments": len(self.adjustment_history),
            "improvement_trend": self._calculate_improvement_trend()
        }

    def _calculate_improvement_trend(self) -> str:
        """Calculate improvement trend direction"""
        if len(self.performance_history) < 20:
            return "insufficient_data"

        recent = list(self.performance_history)[-10:]
        earlier = list(self.performance_history)[-20:-10]

        recent_avg = sum(m.fitness for m in recent) / len(recent)
        earlier_avg = sum(m.fitness for m in earlier) / len(earlier)

        improvement = (recent_avg - earlier_avg) / (abs(earlier_avg) + 1e-6)

        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "declining"
        else:
            return "stable"

    def get_adjustment_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get history of configuration adjustments"""
        if limit:
            return self.adjustment_history[-limit:]
        return self.adjustment_history.copy()

    def reset_history(self) -> None:
        """Clear performance and adjustment history"""
        self.performance_history.clear()
        self.adjustment_history.clear()
        logger.info("Adaptive configurator history reset")


class AutoTuner:
    """
    Advanced automatic tuning with machine learning

    Uses historical data to learn optimal parameter settings
    for different problem domains and performance patterns.
    """

    def __init__(self):
        """Initialize auto-tuner"""
        self.domain_profiles: Dict[str, Dict] = {}
        self.pattern_library: List[Dict] = []
        self.tuning_history: List[Dict] = []

    async def auto_tune(
        self,
        config: UnifiedEvolutionConfig,
        performance_data: Dict,
        domain: str,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Automatically tune configuration based on domain and performance

        Args:
            config: Current configuration
            performance_data: Recent performance metrics
            domain: Problem domain (e.g., "code", "math", "optimization")
            problem_type: Type of problem (e.g., "regression", "classification")

        Returns:
            Recommended parameter adjustments
        """
        recommendations = {}

        # Get domain profile
        profile = self._get_domain_profile(domain, problem_type)

        # Analyze current performance pattern
        pattern = self._identify_pattern(performance_data)

        # Match pattern to library
        matched_pattern = self._match_pattern(pattern)

        if matched_pattern:
            # Apply known good settings
            recommendations = self._apply_pattern_settings(
                matched_pattern,
                config
            )
        else:
            # Use domain profile defaults
            recommendations = self._apply_profile_defaults(
                profile,
                config
            )

        # Record tuning
        self.tuning_history.append({
            "timestamp": datetime.utcnow(),
            "domain": domain,
            "problem_type": problem_type,
            "pattern": pattern,
            "recommendations": recommendations
        })

        return recommendations

    def _get_domain_profile(self, domain: str, problem_type: str) -> Dict:
        """Get or create domain profile"""
        key = f"{domain}:{problem_type}"

        if key not in self.domain_profiles:
            # Create default profile
            self.domain_profiles[key] = self._create_default_profile(domain, problem_type)

        return self.domain_profiles[key]

    def _create_default_profile(self, domain: str, problem_type: str) -> Dict:
        """Create default profile for domain/problem type"""
        # Domain-specific defaults
        domain_defaults = {
            "code": {
                "population_size": 500,
                "exploration_rate": 0.3,
                "mutation_rate": 0.2
            },
            "math": {
                "population_size": 1000,
                "exploration_rate": 0.2,
                "mutation_rate": 0.1
            },
            "optimization": {
                "population_size": 200,
                "exploration_rate": 0.4,
                "mutation_rate": 0.15
            }
        }

        return domain_defaults.get(domain, {
            "population_size": 500,
            "exploration_rate": 0.25,
            "mutation_rate": 0.1
        })

    def _identify_pattern(self, performance_data: Dict) -> Dict:
        """Identify performance pattern"""
        pattern = {
            "convergence_speed": "unknown",
            "diversity_trend": "unknown",
            "efficiency": "unknown"
        }

        # Analyze convergence
        fitness_history = performance_data.get("fitness_history", [])
        if len(fitness_history) >= 10:
            recent_improvement = fitness_history[-1] - fitness_history[-10]
            if recent_improvement > 0.1:
                pattern["convergence_speed"] = "fast"
            elif recent_improvement > 0.01:
                pattern["convergence_speed"] = "moderate"
            else:
                pattern["convergence_speed"] = "slow"

        # Analyze diversity
        diversity_history = performance_data.get("diversity_history", [])
        if diversity_history:
            recent_diversity = diversity_history[-1]
            if recent_diversity > 0.7:
                pattern["diversity_trend"] = "high"
            elif recent_diversity > 0.4:
                pattern["diversity_trend"] = "moderate"
            else:
                pattern["diversity_trend"] = "low"

        # Analyze efficiency
        eval_times = performance_data.get("evaluation_times", [])
        if eval_times:
            avg_time = sum(eval_times) / len(eval_times)
            if avg_time < 1.0:
                pattern["efficiency"] = "high"
            elif avg_time < 10.0:
                pattern["efficiency"] = "moderate"
            else:
                pattern["efficiency"] = "low"

        return pattern

    def _match_pattern(self, pattern: Dict) -> Optional[Dict]:
        """Match pattern to library of known patterns"""
        for known_pattern in self.pattern_library:
            if self._patterns_match(pattern, known_pattern["pattern"]):
                return known_pattern

        return None

    def _patterns_match(self, pattern1: Dict, pattern2: Dict) -> bool:
        """Check if two patterns match"""
        match_threshold = 0.7  # 70% of attributes must match

        matching = 0
        total = 0

        for key in pattern1:
            if key in pattern2:
                total += 1
                if pattern1[key] == pattern2[key]:
                    matching += 1

        return total > 0 and (matching / total) >= match_threshold

    def _apply_pattern_settings(
        self,
        pattern: Dict,
        config: UnifiedEvolutionConfig
    ) -> Dict[str, Any]:
        """Apply settings from matched pattern"""
        return pattern.get("settings", {})

    def _apply_profile_defaults(
        self,
        profile: Dict,
        config: UnifiedEvolutionConfig
    ) -> Dict[str, Any]:
        """Apply domain profile defaults"""
        settings = {}

        for key, value in profile.items():
            if key == "population_size":
                settings["database.population_size"] = value
            elif key == "exploration_rate":
                settings["database.exploration_rate"] = value
            elif key == "mutation_rate":
                settings["openevolve.mutation_rate"] = value

        return settings

    def learn_from_results(
        self,
        pattern: Dict,
        settings: Dict,
        final_performance: float
    ) -> None:
        """
        Learn from completed run to improve future recommendations

        Args:
            pattern: Performance pattern
            settings: Settings that were used
            final_performance: Final achieved performance
        """
        # Add to pattern library
        self.pattern_library.append({
            "pattern": pattern,
            "settings": settings,
            "performance": final_performance,
            "timestamp": datetime.utcnow()
        })

        # Sort by performance and keep best patterns
        self.pattern_library.sort(key=lambda x: x["performance"], reverse=True)
        self.pattern_library = self.pattern_library[:100]  # Keep top 100

        logger.info(f"Learned from run. Pattern library size: {len(self.pattern_library)}")
