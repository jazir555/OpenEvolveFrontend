"""
Configuration Metrics

This module provides tracking and analysis of configuration performance,
including parameter usage, impact scoring, and optimization recommendations.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from ..unified.config import UnifiedEvolutionConfig


logger = logging.getLogger(__name__)


@dataclass
class ConfigPerformanceRecord:
    """Record of configuration performance"""
    config_hash: str
    performance: float
    iteration: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveMetric:
    """
    Result of an adaptive metric computation over an evolution run.

    All scalar fields are real numbers derived deterministically from the
    supplied fitness/population history. ``metric`` is a normalized summary in
    [0, 1] where higher means the population is stagnating (i.e. it needs more
    exploration). Lower means it is still making progress / converging well.
    """
    stagnation_index: float
    improvement_rate: float
    convergence_slope: float
    diversity: float
    stagnation_generations: int
    metric: float


@dataclass
class ParameterUsage:
    """Usage statistics for a parameter"""
    parameter_name: str
    times_used: int
    times_modified: int
    last_value: Any
    value_history: List[Any] = field(default_factory=list)


class ConfigurationMetrics:
    """
    Track and analyze configuration performance

    Features:
    - Parameter usage tracking
    - Performance correlation analysis
    - Configuration history
    - Impact scoring
    - Optimization recommendations
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize configuration metrics

        Args:
            history_size: Maximum number of records to keep
        """
        self.parameter_usage: Dict[str, ParameterUsage] = {}
        self.parameter_impact: Dict[str, float] = {}
        self.config_history: List[ConfigPerformanceRecord] = []
        self.history_size = history_size

        # Analysis results
        self.impact_calculated = False
        self.correlations: Dict[str, float] = {}

    def track_config_usage(
        self,
        config: UnifiedEvolutionConfig,
        modified_params: Optional[List[str]] = None
    ) -> None:
        """
        Track which parameters are non-default

        Args:
            config: Configuration to track
            modified_params: List of parameters that were modified
        """
        config_dict = config.model_dump()

        # Track all parameters
        self._track_parameters_recursive(config_dict, [], modified_params or [])

    def _track_parameters_recursive(
        self,
        config_section: Any,
        path: List[str],
        modified_params: List[str]
    ) -> None:
        """Recursively track parameters"""
        if isinstance(config_section, dict):
            for key, value in config_section.items():
                new_path = path + [key]
                param_name = ".".join(new_path)

                if isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    self._track_parameters_recursive(value, new_path, modified_params)
                else:
                    # Track leaf parameter
                    if param_name not in self.parameter_usage:
                        self.parameter_usage[param_name] = ParameterUsage(
                            parameter_name=param_name,
                            times_used=0,
                            times_modified=0,
                            last_value=value,
                            value_history=[]
                        )

                    self.parameter_usage[param_name].times_used += 1
                    self.parameter_usage[param_name].last_value = value
                    self.parameter_usage[param_name].value_history.append(value)

                    if param_name in modified_params:
                        self.parameter_usage[param_name].times_modified += 1

        elif isinstance(config_section, list):
            # Track list items
            for i, item in enumerate(config_section):
                new_path = path + [str(i)]
                if isinstance(item, (dict, list)):
                    self._track_parameters_recursive(item, new_path, modified_params)

    def track_config_performance(
        self,
        config: UnifiedEvolutionConfig,
        performance: float,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track performance of this configuration

        Args:
            config: Configuration that achieved performance
            performance: Performance score
            iteration: Current iteration
            metadata: Additional metadata
        """
        config_hash = hash_config(config)

        record = ConfigPerformanceRecord(
            config_hash=config_hash,
            performance=performance,
            iteration=iteration,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            config_snapshot=config.model_dump()
        )

        self.config_history.append(record)

        # Trim history if needed
        if len(self.config_history) > self.history_size:
            self.config_history = self.config_history[-self.history_size:]

        # Mark impact for recalculation
        self.impact_calculated = False

    def calculate_parameter_impact(self) -> Dict[str, float]:
        """
        Calculate impact score for each parameter

        Uses correlation analysis to determine which parameters
        most affect performance.

        Returns:
            Dictionary mapping parameter names to impact scores
        """
        if not self.config_history:
            logger.warning("No performance history to analyze")
            return {}

        if self.impact_calculated:
            return self.parameter_impact

        # Group configs by hash to get unique configurations
        unique_configs = defaultdict(list)
        for record in self.config_history:
            unique_configs[record.config_hash].append(record.performance)

        # Calculate average performance per config
        avg_performance = {}
        for config_hash, performances in unique_configs.items():
            avg_performance[config_hash] = sum(performances) / len(performances)

        # For each parameter, compute the absolute Pearson correlation between
        # its recorded value and the achieved performance across all recorded
        # configurations. A higher score means the parameter tracked more of
        # the variation in performance (i.e. it is impactful). Categorical /
        # non-numeric parameter values are skipped (correlation is undefined).
        param_values: Dict[str, List[float]] = defaultdict(list)
        param_perf: Dict[str, List[float]] = defaultdict(list)

        for record in self.config_history:
            flat = {}
            self._flatten_config(record.config_snapshot, [], flat)
            for name, value in flat.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                param_values[name].append(numeric_value)
                param_perf[name].append(record.performance)

        impact: Dict[str, float] = {}
        for name, values in param_values.items():
            if len(values) < 2:
                impact[name] = 0.0
                continue
            corr = _pearson_correlation(values, param_perf[name])
            impact[name] = abs(corr) if corr == corr else 0.0  # guard NaN

        self.parameter_impact = impact
        self.impact_calculated = True

        return self.parameter_impact

    def get_parameter_usage_stats(
        self,
        parameter_name: Optional[str] = None
    ) -> Dict[str, ParameterUsage]:
        """
        Get usage statistics for parameters

        Args:
            parameter_name: Specific parameter (None = all)

        Returns:
            Dictionary of parameter usage statistics
        """
        if parameter_name:
            if parameter_name in self.parameter_usage:
                return {parameter_name: self.parameter_usage[parameter_name]}
            else:
                return {}
        return self.parameter_usage.copy()

    def get_most_used_parameters(
        self,
        limit: int = 10
    ) -> List[tuple]:
        """
        Get most commonly used parameters

        Args:
            limit: Maximum number to return

        Returns:
            List of (parameter_name, usage_count) tuples
        """
        sorted_params = sorted(
            self.parameter_usage.items(),
            key=lambda x: x[1].times_used,
            reverse=True
        )
        return [
            (name, usage.times_used)
            for name, usage in sorted_params[:limit]
        ]

    def get_most_modified_parameters(
        self,
        limit: int = 10
    ) -> List[tuple]:
        """
        Get most commonly modified parameters

        Args:
            limit: Maximum number to return

        Returns:
            List of (parameter_name, modification_count) tuples
        """
        modified = {
            name: usage
            for name, usage in self.parameter_usage.items()
            if usage.times_modified > 0
        }

        sorted_params = sorted(
            modified.items(),
            key=lambda x: x[1].times_modified,
            reverse=True
        )

        return [
            (name, usage.times_modified)
            for name, usage in sorted_params[:limit]
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of performance across all configurations

        Returns:
            Dictionary with performance summary
        """
        if not self.config_history:
            return {}

        performances = [r.performance for r in self.config_history]

        return {
            "total_runs": len(self.config_history),
            "best_performance": max(performances),
            "worst_performance": min(performances),
            "average_performance": sum(performances) / len(performances),
            "performance_std": self._calculate_std(performances),
            "best_config_hash": max(self.config_history, key=lambda r: r.performance).config_hash,
            "first_iteration": self.config_history[0].iteration,
            "last_iteration": self.config_history[-1].iteration,
        }

    def suggest_optimal_parameters(
        self,
        domain: str,
        problem_type: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimal parameters based on historical data

        Args:
            domain: Problem domain (e.g., "code", "math")
            problem_type: Type of problem (e.g., "regression")
            top_n: Number of suggestions to return

        Returns:
            List of suggested configurations with expected performance
        """
        if not self.config_history:
            logger.warning("No performance history for recommendations")
            return []

        # Group by performance
        sorted_records = sorted(
            self.config_history,
            key=lambda r: r.performance,
            reverse=True
        )

        # Get top performing configs
        top_configs = sorted_records[:top_n]

        suggestions = []
        for record in top_configs:
            suggestions.append({
                "config_hash": record.config_hash,
                "expected_performance": record.performance,
                "iteration": record.iteration,
                "timestamp": record.timestamp,
                "metadata": record.metadata
            })

        return suggestions

    def analyze_parameter_trends(
        self,
        parameter_name: str
    ) -> Dict[str, Any]:
        """
        Analyze trends for a specific parameter

        Args:
            parameter_name: Parameter to analyze

        Returns:
            Dictionary with trend analysis
        """
        if parameter_name not in self.parameter_usage:
            return {"error": f"Parameter {parameter_name} not tracked"}

        usage = self.parameter_usage[parameter_name]

        # Calculate statistics
        if usage.value_history:
            # Try to calculate numeric statistics
            try:
                numeric_values = [float(v) for v in usage.value_history if isinstance(v, (int, float))]
                if numeric_values:
                    return {
                        "parameter_name": parameter_name,
                        "times_used": usage.times_used,
                        "times_modified": usage.times_modified,
                        "current_value": usage.last_value,
                        "min_value": min(numeric_values),
                        "max_value": max(numeric_values),
                        "avg_value": sum(numeric_values) / len(numeric_values),
                        "value_range": max(numeric_values) - min(numeric_values),
                        "trend": self._calculate_trend(numeric_values)
                    }
            except (ValueError, TypeError):
                pass

        # Non-numeric parameter
        return {
            "parameter_name": parameter_name,
            "times_used": usage.times_used,
            "times_modified": usage.times_modified,
            "current_value": usage.last_value,
            "unique_values": len(set(str(v) for v in usage.value_history)),
            "value_history": usage.value_history[-10:]  # Last 10 values
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "insufficient_data"

        # Compare first third to last third
        n = len(values)
        first_third = values[:n//3]
        last_third = values[-(n//3):]

        first_avg = sum(first_third) / len(first_third)
        last_avg = sum(last_third) / len(last_third)

        change = (last_avg - first_avg) / (abs(first_avg) + 1e-6)

        if change > 0.1:
            return "increasing"
        elif change < -0.1:
            return "decreasing"
        else:
            return "stable"

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics

        Returns:
            Dictionary with all metrics data
        """
        return {
            "parameter_usage": {
                name: {
                    "times_used": usage.times_used,
                    "times_modified": usage.times_modified,
                    "last_value": str(usage.last_value),
                    "value_count": len(usage.value_history)
                }
                for name, usage in self.parameter_usage.items()
            },
            "parameter_impact": self.parameter_impact,
            "performance_summary": self.get_performance_summary(),
            "config_history_count": len(self.config_history),
            "most_used_parameters": self.get_most_used_parameters(10),
            "most_modified_parameters": self.get_most_modified_parameters(10)
        }

    def reset_metrics(self) -> None:
        """Clear all metrics"""
        self.parameter_usage.clear()
        self.parameter_impact.clear()
        self.config_history.clear()
        self.impact_calculated = False
        logger.info("Configuration metrics reset")


    def _flatten_config(
        self,
        obj: Any,
        path: List[str],
        out: Dict[str, Any]
    ) -> None:
        """Flatten a nested config dict into leaf parameter -> value pairs."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._flatten_config(value, path + [key], out)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._flatten_config(item, path + [str(i)], out)
        else:
            name = ".".join(path)
            if name:
                out[name] = obj


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Deterministic, dependency-free Pearson correlation."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)

    if den_x == 0.0 or den_y == 0.0:
        return 0.0

    return num / (den_x ** 0.5 * den_y ** 0.5)


def _population_diversity(scores: Optional[List[float]]) -> float:
    """
    Coefficient of variation of population scores (std / |mean|).

    Higher means the population is spread out (diverse); 0.0 means the
    population has collapsed to a single value or has no data.
    """
    if not scores or len(scores) < 2:
        return 0.0

    mean = sum(scores) / len(scores)
    if mean == 0.0:
        return 0.0

    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return (variance ** 0.5) / abs(mean)


def compute_adaptive_metrics(
    fitness_history: List[float],
    population_scores: Optional[List[float]] = None,
    iteration: Optional[int] = None,
    window: int = 20,
) -> AdaptiveMetric:
    """
    Compute genuine adaptive metrics from evolution run data.

    Inputs:
        fitness_history: best fitness achieved per generation (higher = better).
        population_scores: fitness of the current generation's individuals.
        iteration: current generation index (unused, kept for call-site symmetry).
        window: number of recent generations to consider for slope/diversity.

    Returns:
        An :class:`AdaptiveMetric`. ``metric`` (and ``stagnation_index``) is a
        normalized value in [0, 1]: 0.0 means the run is still improving
        strongly, 1.0 means it has fully stagnated. Strategy selectors can use
        this directly to decide exploration vs. exploitation.

    Method:
        - ``convergence_slope``: per-generation change in best fitness over the
          recent window, normalized by fitness scale.
        - ``improvement_rate``: the positive part of ``convergence_slope``.
        - ``stagnation_generations``: trailing generations without a new best.
        - ``stagnation_index``: fraction of history spent stagnating, clamped to
          [0, 1]; boosted to >= 0.8 if the normalized slope is negative.
        - ``diversity``: coefficient of variation of ``population_scores``.
    """
    fitness_history = list(fitness_history) if fitness_history else []

    diversity = _population_diversity(population_scores)

    if len(fitness_history) < 2:
        return AdaptiveMetric(
            stagnation_index=0.0,
            improvement_rate=0.0,
            convergence_slope=0.0,
            diversity=diversity,
            stagnation_generations=0,
            metric=0.0,
        )

    w = min(len(fitness_history), max(2, int(window)))
    recent = fitness_history[-w:]
    first, last = recent[0], recent[-1]
    slope = (last - first) / (w - 1)
    scale = max(abs(first), abs(last), 1e-9)
    norm_slope = slope / scale

    best_so_far = fitness_history[0]
    last_improvement_idx = 0
    for i, value in enumerate(fitness_history):
        if value > best_so_far + 1e-9:
            best_so_far = value
            last_improvement_idx = i
    stagnation_generations = (len(fitness_history) - 1) - last_improvement_idx

    stag_frac = stagnation_generations / max(1, len(fitness_history) - 1)
    stagnation_index = min(1.0, stag_frac)
    if norm_slope < 0.0:
        stagnation_index = max(stagnation_index, 0.8)

    return AdaptiveMetric(
        stagnation_index=round(stagnation_index, 6),
        improvement_rate=round(max(0.0, norm_slope), 6),
        convergence_slope=round(norm_slope, 6),
        diversity=round(diversity, 6),
        stagnation_generations=stagnation_generations,
        metric=round(stagnation_index, 6),
    )


def compute_adaptive_metric(
    fitness_history: List[float],
    population_scores: Optional[List[float]] = None,
    iteration: Optional[int] = None,
    window: int = 20,
) -> float:
    """
    Convenience wrapper returning only the scalar adaptive metric in [0, 1].

    0.0 = strongly improving / converging; 1.0 = fully stagnated.
    """
    return compute_adaptive_metrics(
        fitness_history, population_scores, iteration, window
    ).metric


def hash_config(config: UnifiedEvolutionConfig) -> str:
    """
    Generate a hash for configuration

    Args:
        config: Configuration to hash

    Returns:
        SHA256 hash of configuration
    """
    config_dict = config.model_dump()
    config_str = json.dumps(config_dict, sort_keys=True)

    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ConfigComparison:
    """Compare different configurations"""

    @staticmethod
    def compare_configs(
        config1: UnifiedEvolutionConfig,
        config2: UnifiedEvolutionConfig
    ) -> Dict[str, Any]:
        """
        Compare two configurations

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Dictionary with comparison results
        """
        dict1 = config1.model_dump()
        dict2 = config2.model_dump()

        differences = ConfigComparison._find_differences(dict1, dict2, "")

        return {
            "are_identical": len(differences) == 0,
            "num_differences": len(differences),
            "differences": differences
        }

    @staticmethod
    def _find_differences(
        dict1: Dict,
        dict2: Dict,
        path: str
    ) -> List[Dict[str, Any]]:
        """Recursively find differences between dictionaries"""
        differences = []

        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            new_path = f"{path}.{key}" if path else key

            if key not in dict1:
                differences.append({
                    "parameter": new_path,
                    "type": "added",
                    "value_in_config2": dict2[key]
                })
            elif key not in dict2:
                differences.append({
                    "parameter": new_path,
                    "type": "removed",
                    "value_in_config1": dict1[key]
                })
            else:
                val1 = dict1[key]
                val2 = dict2[key]

                if isinstance(val1, dict) and isinstance(val2, dict):
                    differences.extend(
                        ConfigComparison._find_differences(val1, val2, new_path)
                    )
                elif val1 != val2:
                    differences.append({
                        "parameter": new_path,
                        "type": "changed",
                        "value_in_config1": val1,
                        "value_in_config2": val2
                    })

        return differences

    @staticmethod
    def get_default_params(
        config: UnifiedEvolutionConfig
    ) -> List[str]:
        """
        Get list of parameters with default values

        Args:
            config: Configuration to check

        Returns:
            List of parameter names with default values
        """
        # This would require access to the schema defaults
        # For now, return empty list
        return []
