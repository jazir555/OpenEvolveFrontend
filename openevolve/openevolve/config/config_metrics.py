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
            metadata=metadata or {}
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

        # For each parameter, calculate correlation with performance
        # This is simplified - full implementation would parse config hashes
        # For now, return placeholder

        self.parameter_impact = {}
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
