"""
Strategy Performance Tracker for Adaptive Strategy Selection

This module tracks the performance of decomposition strategies over time,
enabling the system to learn from past outcomes and adaptively select
the best strategy for each problem.

Key Features:
- Persistent storage of strategy performance data
- Performance statistics calculation (quality, success rate, time)
- Trend analysis (improving, stable, declining)
- Domain-specific tracking
- Problem-type specific tracking
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class StrategyPerformanceTracker:
    """
    Tracks strategy performance over time for adaptive selection.

    This class records the outcomes of strategy usage and provides
    performance statistics to inform adaptive weight calculation.

    Storage Format:
        {
            "strategies": {
                "semantic": {
                    "overall": {
                        "usage_count": 100,
                        "quality_scores": [0.8, 0.9, 0.85, ...],
                        "success_count": 85,
                        "completion_times": [120, 150, 110, ...],
                        "last_used": "2025-01-03T10:30:00"
                    },
                    "by_problem_type": {
                        "algorithm_design": {
                            "usage_count": 30,
                            "quality_scores": [0.9, 0.85, ...],
                            "success_count": 28,
                            ...
                        }
                    },
                    "by_domain": {
                        "software_engineering": {
                            "usage_count": 50,
                            "quality_scores": [0.85, 0.9, ...],
                            ...
                        }
                    }
                }
            }
        }
    """

    def __init__(self, storage_path: str = "strategy_performance.json"):
        """
        Initialize with persistent storage.

        Args:
            storage_path: Path to JSON file for performance data storage
        """
        self.storage_path = Path(storage_path)
        self.data = self._load_data()
        logger.info(f"StrategyPerformanceTracker initialized with storage: {storage_path}")

    def _load_data(self) -> Dict:
        """Load performance data from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded performance data from {self.storage_path}")
                return data
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to load performance data: {e}. Starting fresh.")
                return self._create_empty_data()
        else:
            logger.info("No existing performance data found. Starting fresh.")
            return self._create_empty_data()

    def _create_empty_data(self) -> Dict:
        """Create empty data structure."""
        return {
            "strategies": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

    def _save_data(self):
        """Save performance data to disk."""
        try:
            # Convert to Path object if needed
            storage_path = Path(self.storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(storage_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Saved performance data to {storage_path}")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to save performance data: {e}")

    def record_strategy_outcome(self,
                               strategy: str,
                               problem_type: str,
                               domain: str,
                               quality_score: float,
                               user_satisfaction: Optional[float] = None,
                               time_to_complete: Optional[float] = None):
        """
        Record outcome of a strategy usage.

        Args:
            strategy: Strategy name (e.g., 'semantic', 'dependency', 'complexity')
            problem_type: Primary problem type (e.g., 'algorithm_design', 'system_architecture')
            domain: Domain context (e.g., 'software_engineering', 'data_science')
            quality_score: Overall quality score (0.0 to 1.0)
            user_satisfaction: Optional user satisfaction score (0.0 to 1.0)
            time_to_complete: Optional completion time in seconds
        """
        timestamp = datetime.now().isoformat()

        # Ensure strategy exists
        if strategy not in self.data["strategies"]:
            self.data["strategies"][strategy] = {
                "overall": {
                    "usage_count": 0,
                    "quality_scores": [],
                    "success_count": 0,
                    "completion_times": [],
                    "last_used": None
                },
                "by_problem_type": {},
                "by_domain": {}
            }

        strategy_data = self.data["strategies"][strategy]

        # Update overall stats
        overall = strategy_data["overall"]
        overall["usage_count"] += 1
        overall["quality_scores"].append(quality_score)
        if quality_score >= 0.7:  # Success threshold
            overall["success_count"] += 1
        if time_to_complete:
            overall["completion_times"].append(time_to_complete)
        overall["last_used"] = timestamp

        # Update by problem type
        if problem_type not in strategy_data["by_problem_type"]:
            strategy_data["by_problem_type"][problem_type] = {
                "usage_count": 0,
                "quality_scores": [],
                "success_count": 0,
                "completion_times": [],
                "last_used": None
            }

        pt_data = strategy_data["by_problem_type"][problem_type]
        pt_data["usage_count"] += 1
        pt_data["quality_scores"].append(quality_score)
        if quality_score >= 0.7:
            pt_data["success_count"] += 1
        if time_to_complete:
            pt_data["completion_times"].append(time_to_complete)
        pt_data["last_used"] = timestamp

        # Update by domain
        if domain not in strategy_data["by_domain"]:
            strategy_data["by_domain"][domain] = {
                "usage_count": 0,
                "quality_scores": [],
                "success_count": 0,
                "completion_times": [],
                "last_used": None
            }

        domain_data = strategy_data["by_domain"][domain]
        domain_data["usage_count"] += 1
        domain_data["quality_scores"].append(quality_score)
        if quality_score >= 0.7:
            domain_data["success_count"] += 1
        if time_to_complete:
            domain_data["completion_times"].append(time_to_complete)
        domain_data["last_used"] = timestamp

        # Save to disk
        self._save_data()

        logger.info(f"Recorded outcome for strategy '{strategy}': "
                   f"quality={quality_score:.2f}, type={problem_type}, domain={domain}")

    def get_strategy_performance(self,
                                strategy: str,
                                problem_type: Optional[str] = None,
                                domain: Optional[str] = None) -> Dict[str, float]:
        """
        Get performance stats for a strategy.

        Args:
            strategy: Strategy name
            problem_type: Optional problem type filter
            domain: Optional domain filter

        Returns:
            Dictionary with performance statistics:
            - avg_quality_score: Average quality (0.0 to 1.0)
            - usage_count: Number of times used
            - success_rate: Proportion of successful outcomes (0.0 to 1.0)
            - avg_time: Average completion time in seconds (None if not tracked)
            - last_used: ISO timestamp of last use (None if never used)
            - trend: Performance trend ('improving', 'stable', 'declining')
            - confidence: Confidence in stats (0.0 to 1.0 based on sample size)
        """
        if strategy not in self.data["strategies"]:
            return {
                "avg_quality_score": 0.5,
                "usage_count": 0,
                "success_rate": 0.0,
                "avg_time": None,
                "last_used": None,
                "trend": "unknown",
                "confidence": 0.0
            }

        # Get relevant data (problem_type and domain specific if provided)
        if problem_type and domain:
            # Combined filter - get intersection
            if (problem_type in self.data["strategies"][strategy]["by_problem_type"] and
                domain in self.data["strategies"][strategy]["by_domain"]):
                # Use overall data for now (could be enhanced for combined filtering)
                data = self.data["strategies"][strategy]["overall"]
            else:
                data = self.data["strategies"][strategy]["overall"]
        elif problem_type:
            data = self.data["strategies"][strategy]["by_problem_type"].get(
                problem_type,
                self.data["strategies"][strategy]["overall"]
            )
        elif domain:
            data = self.data["strategies"][strategy]["by_domain"].get(
                domain,
                self.data["strategies"][strategy]["overall"]
            )
        else:
            data = self.data["strategies"][strategy]["overall"]

        # Calculate statistics
        usage_count = data.get("usage_count", 0)
        quality_scores = data.get("quality_scores", [])
        success_count = data.get("success_count", 0)
        completion_times = data.get("completion_times", [])
        last_used = data.get("last_used")

        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.5
        success_rate = success_count / usage_count if usage_count > 0 else 0.0
        avg_time = statistics.mean(completion_times) if completion_times else None

        # Calculate trend
        trend = self._calculate_trend(quality_scores)

        # Calculate confidence based on sample size
        # 0-3 samples: low confidence (0.0-0.3)
        # 4-10 samples: medium confidence (0.3-0.7)
        # 11+ samples: high confidence (0.7-1.0)
        if usage_count == 0:
            confidence = 0.0
        elif usage_count <= 3:
            confidence = 0.1 + (usage_count / 3) * 0.2
        elif usage_count <= 10:
            confidence = 0.3 + ((usage_count - 3) / 7) * 0.4
        else:
            confidence = 0.7 + min(0.3, (usage_count - 10) / 20)

        return {
            "avg_quality_score": avg_quality,
            "usage_count": usage_count,
            "success_rate": success_rate,
            "avg_time": avg_time,
            "last_used": last_used,
            "trend": trend,
            "confidence": confidence
        }

    def _calculate_trend(self, scores: List[float]) -> str:
        """
        Calculate performance trend from scores.

        Args:
            scores: List of quality scores in chronological order

        Returns:
            'improving', 'stable', or 'declining'
        """
        if len(scores) < 5:
            return "unknown"  # Not enough data

        # Compare recent vs older scores
        split_point = len(scores) // 2
        older_scores = scores[:split_point]
        recent_scores = scores[split_point:]

        older_avg = statistics.mean(older_scores)
        recent_avg = statistics.mean(recent_scores)

        difference = recent_avg - older_avg

        if difference > 0.05:
            return "improving"
        elif difference < -0.05:
            return "declining"
        else:
            return "stable"

    def get_all_strategies(self) -> List[str]:
        """Get list of all tracked strategies."""
        return list(self.data["strategies"].keys())

    def get_strategy_rankings(self,
                             problem_type: Optional[str] = None,
                             domain: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by average quality score.

        Args:
            problem_type: Optional problem type filter
            domain: Optional domain filter

        Returns:
            List of (strategy, avg_quality) tuples sorted by quality
        """
        strategies = self.get_all_strategies()
        rankings = []

        for strategy in strategies:
            perf = self.get_strategy_performance(strategy, problem_type, domain)
            if perf["usage_count"] > 0:
                rankings.append((strategy, perf["avg_quality_score"]))

        # Sort by average quality (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_statistics_summary(self) -> Dict:
        """
        Get overall statistics summary.

        Returns:
            Dictionary with summary statistics across all strategies
        """
        total_usage = 0
        total_successes = 0
        strategy_count = len(self.data["strategies"])

        for strategy_name, strategy_data in self.data["strategies"].items():
            overall = strategy_data["overall"]
            total_usage += overall["usage_count"]
            total_successes += overall["success_count"]

        overall_success_rate = total_successes / total_usage if total_usage > 0 else 0.0

        return {
            "total_strategies": strategy_count,
            "total_decompositions": total_usage,
            "overall_success_rate": overall_success_rate,
            "storage_path": str(self.storage_path),
            "last_updated": datetime.now().isoformat()
        }

    def export_performance_report(self, output_path: str = "performance_report.json"):
        """
        Export detailed performance report to JSON.

        Args:
            output_path: Path for output report
        """
        report = {
            "summary": self.get_statistics_summary(),
            "strategies": {}
        }

        for strategy in self.get_all_strategies():
            overall_perf = self.get_strategy_performance(strategy)

            report["strategies"][strategy] = {
                "overall": overall_perf,
                "by_problem_type": {},
                "by_domain": {}
            }

            # Get problem type breakdown
            if strategy in self.data["strategies"]:
                for pt in self.data["strategies"][strategy]["by_problem_type"]:
                    report["strategies"][strategy]["by_problem_type"][pt] = \
                        self.get_strategy_performance(strategy, problem_type=pt)

                # Get domain breakdown
                for dom in self.data["strategies"][strategy]["by_domain"]:
                    report["strategies"][strategy]["by_domain"][dom] = \
                        self.get_strategy_performance(strategy, domain=dom)

        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported performance report to {output_path}")

    def reset_strategy_data(self, strategy: str):
        """
        Reset all data for a specific strategy.

        Args:
            strategy: Strategy name to reset
        """
        if strategy in self.data["strategies"]:
            self.data["strategies"][strategy] = {
                "overall": {
                    "usage_count": 0,
                    "quality_scores": [],
                    "success_count": 0,
                    "completion_times": [],
                    "last_used": None
                },
                "by_problem_type": {},
                "by_domain": {}
            }
            self._save_data()
            logger.info(f"Reset data for strategy: {strategy}")
        else:
            logger.warning(f"Cannot reset unknown strategy: {strategy}")

    def get_recent_performance(self,
                              strategy: str,
                              days: int = 30) -> Dict[str, float]:
        """
        Get performance for recent time period.

        Args:
            strategy: Strategy name
            days: Number of days to look back

        Returns:
            Dictionary with recent performance stats
        """
        # This would require storing timestamps with each score
        # For now, return overall stats
        return self.get_strategy_performance(strategy)
