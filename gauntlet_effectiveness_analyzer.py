"""
Gauntlet Effectiveness Analyzer - Stage 6 Knowledge Extraction

This module analyzes the effectiveness of quality gauntlets (Red Team, Gold Team).
It provides insights into which rules work best and how to optimize gauntlet configurations.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from collections import defaultdict
import logging

# Import ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_validation_config = None

logger = logging.getLogger(__name__)

from workflow_structures import (
    GauntletEffectivenessArtifact,
    KnowledgeArtifactManager,
)


class GauntletEffectivenessAnalyzer:
    """
    Analyzes gauntlet effectiveness.

    Features:
    - Catch rate tracking
    - False positive analysis
    - Rule effectiveness evaluation
    - Execution time analysis
    - Optimization recommendations
    - A/B testing support

    Attributes:
        artifact_manager: Manager for accessing/storing artifacts
        effectiveness_history: Historical effectiveness data by gauntlet
    """

    def __init__(self, db_path: str = "./knowledge_artifacts.db"):
        """
        Initialize the gauntlet effectiveness analyzer.

        Args:
            db_path: Path to artifact database
        """
        self.artifact_manager = KnowledgeArtifactManager(db_path)
        self.effectiveness_history = defaultdict(list)
        self._load_effectiveness_history()

        # Initialize ROMA-MDAP-MAKER Engine for robust analysis/recomposition
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset for high-reliability checking
                config = get_validation_config(
                    preset="validation",
                    # Can override specific parameters if needed
                    # roma_max_depth_analysis=2  # Example: Override if preset doesn't match needs
                )
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for GauntletEffectivenessAnalyzer")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to initialize ROMA engine: {e}")

    def _load_effectiveness_history(self):
        """Load historical effectiveness data from database."""
        artifacts = self.artifact_manager.list_gauntlet_effectiveness(limit=10000)
        for artifact in artifacts:
            self.effectiveness_history[artifact.gauntlet_id].append(artifact)

    def analyze_gauntlet_run(
        self,
        workflow_id: str,
        gauntlet_id: str,
        gauntlet_type: str,
        total_checks: int,
        issues_caught: int,
        false_positives: int,
        execution_time: float,
        rules_executed: Optional[List[str]] = None,
        problem_type: str = "",
        resource_usage: Optional[Dict[str, float]] = None
    ) -> GauntletEffectivenessArtifact:
        """
        Analyze effectiveness of a gauntlet run.

        Args:
            workflow_id: ID of the workflow
            gauntlet_id: ID of the gauntlet
            gauntlet_type: Type of gauntlet (Red Team, Gold Team, custom)
            total_checks: Total number of checks performed
            issues_caught: Number of issues caught
            false_positives: Number of false positives
            execution_time: Execution time in seconds
            rules_executed: List of rules that were executed
            problem_type: Type of problem being checked
            resource_usage: Dictionary of resource usage metrics

        Returns:
            GauntletEffectivenessArtifact
        """
        # Calculate metrics
        catch_rate = issues_caught / total_checks if total_checks > 0 else 0.0
        false_positive_rate = false_positives / total_checks if total_checks > 0 else 0.0

        # Create artifact
        artifact = GauntletEffectivenessArtifact(
            artifact_id=f"gauntlet_{gauntlet_id}_{workflow_id}_{int(time.time())}",
            source_workflow_id=workflow_id,
            gauntlet_id=gauntlet_id,
            gauntlet_type=gauntlet_type,
            catch_rate=catch_rate,
            false_positive_rate=false_positive_rate,
            execution_time=execution_time,
            resource_usage=resource_usage or {},
            confidence=0.8,
        )

        # Add problem type effectiveness
        if problem_type:
            artifact.problem_type_effectiveness = {problem_type: catch_rate}

        # Analyze rule effectiveness if rules provided
        if rules_executed:
            # For now, assign equal effectiveness to all rules
            # In a real implementation, this would track which rules caught issues
            artifact.rule_effectiveness = {
                rule: catch_rate / len(rules_executed)
                for rule in rules_executed
            }
            artifact.rules_recommended = [
                {"rule": rule, "enabled": True, "priority": "medium"}
                for rule in rules_executed
            ]

        # Generate recommendations
        artifact.recommended_improvements = artifact.recommend_optimization()

        # Add to history
        self.effectiveness_history[gauntlet_id].append(artifact)

        return artifact

    def get_gauntlet_summary(self, gauntlet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a gauntlet's effectiveness.

        Args:
            gauntlet_id: ID of the gauntlet

        Returns:
            Dictionary with gauntlet effectiveness summary
        """
        if gauntlet_id not in self.effectiveness_history or not self.effectiveness_history[gauntlet_id]:
            return None

        history = self.effectiveness_history[gauntlet_id]

        # Calculate aggregate metrics
        avg_catch_rate = np.mean([h.catch_rate for h in history])
        avg_false_positive_rate = np.mean([h.false_positive_rate for h in history])
        avg_execution_time = np.mean([h.execution_time for h in history])
        total_runs = len(history)

        # Calculate overall effectiveness score
        latest = history[-1]
        effectiveness_score = latest.get_effectiveness_score()

        # Get problem type effectiveness
        problem_type_effectiveness = defaultdict(list)
        for h in history:
            for problem_type, score in h.problem_type_effectiveness.items():
                problem_type_effectiveness[problem_type].append(score)

        avg_problem_type_effectiveness = {
            problem_type: np.mean(scores)
            for problem_type, scores in problem_type_effectiveness.items()
        }

        # Get rule effectiveness (from latest run)
        rule_effectiveness = latest.rule_effectiveness if latest.rule_effectiveness else {}

        # Get recommendations
        recommended_improvements = latest.recommended_improvements if latest.recommended_improvements else []

        return {
            "gauntlet_id": gauntlet_id,
            "gauntlet_type": latest.gauntlet_type,
            "total_runs": total_runs,
            "avg_catch_rate": avg_catch_rate,
            "avg_false_positive_rate": avg_false_positive_rate,
            "avg_execution_time": avg_execution_time,
            "effectiveness_score": effectiveness_score,
            "problem_type_effectiveness": avg_problem_type_effectiveness,
            "rule_effectiveness": rule_effectiveness,
            "recommended_improvements": recommended_improvements,
            "latest_artifact_id": latest.artifact_id,
        }

    def compare_gauntlets(self, gauntlet_ids: List[str]) -> Dict[str, Any]:
        """
        Compare effectiveness of multiple gauntlets.

        Args:
            gauntlet_ids: List of gauntlet IDs to compare

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "gauntlets": {},
            "metrics": ["catch_rate", "false_positive_rate", "effectiveness_score", "execution_time"],
            "rankings": {},
        }

        # Collect metrics for each gauntlet
        for gauntlet_id in gauntlet_ids:
            summary = self.get_gauntlet_summary(gauntlet_id)
            if summary:
                comparison["gauntlets"][gauntlet_id] = {
                    "catch_rate": summary["avg_catch_rate"],
                    "false_positive_rate": summary["avg_false_positive_rate"],
                    "effectiveness_score": summary["effectiveness_score"],
                    "execution_time": summary["avg_execution_time"],
                    "total_runs": summary["total_runs"],
                }

        # Rank gauntlets by each metric
        for metric in comparison["metrics"]:
            if metric == "false_positive_rate" or metric == "execution_time":
                # Lower is better for these
                ranked = sorted(
                    [(gauntlet_id, data[metric]) for gauntlet_id, data in comparison["gauntlets"].items()],
                    key=lambda x: x[1]
                )
            else:
                # Higher is better
                ranked = sorted(
                    [(gauntlet_id, data[metric]) for gauntlet_id, data in comparison["gauntlets"].items()],
                    key=lambda x: x[1],
                    reverse=True
                )
            comparison["rankings"][metric] = [{"gauntlet_id": g, "value": v} for g, v in ranked]

        return comparison

    def recommend_optimal_configuration(self, gauntlet_id: str) -> Dict[str, Any]:
        """
        Recommend optimal configuration for a gauntlet.

        Args:
            gauntlet_id: ID of the gauntlet

        Returns:
            Dictionary with configuration recommendations
        """
        summary = self.get_gauntlet_summary(gauntlet_id)

        if not summary:
            return {}

        recommendations = {
            "gauntlet_id": gauntlet_id,
            "current_performance": {
                "catch_rate": summary["avg_catch_rate"],
                "false_positive_rate": summary["avg_false_positive_rate"],
                "execution_time": summary["avg_execution_time"],
            },
            "recommended_changes": [],
            "rules_to_enable": [],
            "rules_to_disable": [],
            "rules_to_tune": [],
        }

        # Analyze rule effectiveness
        rule_effectiveness = summary["rule_effectiveness"]
        if rule_effectiveness:
            # Identify low-effectiveness rules
            low_effectiveness = [rule for rule, score in rule_effectiveness.items() if score < 0.3]
            recommendations["rules_to_disable"] = low_effectiveness

            # Identify high-effectiveness rules
            high_effectiveness = [rule for rule, score in rule_effectiveness.items() if score > 0.7]
            recommendations["rules_to_enable"] = high_effectiveness

            # Rules that need tuning
            medium_effectiveness = [rule for rule, score in rule_effectiveness.items() if 0.3 <= score <= 0.7]
            recommendations["rules_to_tune"] = medium_effectiveness

        # Performance-based recommendations
        if summary["avg_catch_rate"] < 0.5:
            recommendations["recommended_changes"].append("Add more rules or enable high-effectiveness rules")

        if summary["avg_false_positive_rate"] > 0.3:
            recommendations["recommended_changes"].append("Reduce false positives by tuning rule strictness")

        if summary["avg_execution_time"] > 10.0:
            recommendations["recommended_changes"].append("Optimize execution time by disabling low-effectiveness rules")

        return recommendations

    def recommend_optimization(self, gauntlet_id: str) -> Dict[str, Any]:
        """
        Recommend optimizations for a gauntlet.

        This is an alias for recommend_optimal_configuration() for MASTER_TASKLIST compatibility.

        Args:
            gauntlet_id: ID of the gauntlet

        Returns:
            Dictionary with optimization recommendations
        """
        return self.recommend_optimal_configuration(gauntlet_id)

    def analyze_rule_effectiveness(self, gauntlet_id: str, min_runs: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Analyze effectiveness of individual rules in a gauntlet.

        Args:
            gauntlet_id: ID of the gauntlet
            min_runs: Minimum number of runs required for analysis

        Returns:
            Dictionary mapping rule IDs to effectiveness metrics
        """
        summary = self.get_gauntlet_summary(gauntlet_id)

        if not summary:
            return {}

        rule_stats = {}

        # Aggregate rule effectiveness from historical runs
        artifacts = self.artifact_manager.list_gauntlet_effectiveness(limit=1000)
        gauntlet_artifacts = [a for a in artifacts if a.gauntlet_id == gauntlet_id]

        if len(gauntlet_artifacts) < min_runs:
            return {"error": f"Need at least {min_runs} runs for analysis, got {len(gauntlet_artifacts)}"}

        # Calculate rule effectiveness metrics
        for artifact in gauntlet_artifacts:
            if artifact.rules_recommended:
                for rule_id, rule_data in artifact.rules_recommended.items():
                    if rule_id not in rule_stats:
                        rule_stats[rule_id] = {
                            "times_fired": 0,
                            "times_caught_issue": 0,
                            "times_false_positive": 0,
                        }

                    rule_stats[rule_id]["times_fired"] += 1

                    # Check if rule caught an issue
                    if isinstance(rule_data, dict):
                        if rule_data.get("caught_issue", False):
                            rule_stats[rule_id]["times_caught_issue"] += 1
                        if rule_data.get("false_positive", False):
                            rule_stats[rule_id]["times_false_positive"] += 1

        # Calculate effectiveness scores
        rule_effectiveness = {}
        for rule_id, stats in rule_stats.items():
            if stats["times_fired"] > 0:
                catch_rate = stats["times_caught_issue"] / stats["times_fired"]
                false_positive_rate = stats["times_false_positive"] / stats["times_fired"]

                # Overall effectiveness (catch rate - false positive penalty)
                effectiveness = catch_rate - (false_positive_rate * 0.5)

                rule_effectiveness[rule_id] = {
                    "effectiveness_score": max(0.0, min(1.0, effectiveness)),
                    "catch_rate": catch_rate,
                    "false_positive_rate": false_positive_rate,
                    "times_fired": stats["times_fired"],
                    "times_caught_issue": stats["times_caught_issue"],
                    "times_false_positive": stats["times_false_positive"],
                }

        return rule_effectiveness

    def identify_redundant_rules(self, gauntlet_id: str, correlation_threshold: float = 0.9) -> List[str]:
        """
        Identify redundant rules that fire together.

        Args:
            gauntlet_id: ID of the gauntlet
            correlation_threshold: Correlation threshold for redundancy

        Returns:
            List of redundant rule IDs
        """
        # This would analyze rule execution history to find correlated rules
        # For now, return empty list
        return []

    def ab_test_gauntlets(
        self,
        gauntlet_a_id: str,
        gauntlet_b_id: str,
        metric: str = "effectiveness_score"
    ) -> Dict[str, Any]:
        """
        Perform A/B testing between two gauntlets.

        Args:
            gauntlet_a_id: ID of gauntlet A
            gauntlet_b_id: ID of gauntlet B
            metric: Metric to compare

        Returns:
            Dictionary with A/B test results
        """
        summary_a = self.get_gauntlet_summary(gauntlet_a_id)
        summary_b = self.get_gauntlet_summary(gauntlet_b_id)

        if not summary_a or not summary_b:
            return {"error": "One or both gauntlets not found"}

        value_a = summary_a[metric]
        value_b = summary_b[metric]

        # Calculate improvement
        if metric == "false_positive_rate" or metric == "execution_time":
            # Lower is better
            improvement = (value_a - value_b) / value_a * 100 if value_a > 0 else 0
            winner = "B" if value_b < value_a else "A"
        else:
            # Higher is better
            improvement = (value_b - value_a) / value_a * 100 if value_a > 0 else 0
            winner = "B" if value_b > value_a else "A"

        return {
            "gauntlet_a": gauntlet_a_id,
            "gauntlet_b": gauntlet_b_id,
            "metric": metric,
            "value_a": value_a,
            "value_b": value_b,
            "improvement_percent": improvement,
            "winner": winner,
            "recommendation": f"Use gauntlet {winner}" if abs(improvement) > 10 else "Both gauntlets perform similarly",
        }

    def get_top_performing_gauntlets(
        self,
        metric: str = "effectiveness_score",
        n_top: int = 5,
        gauntlet_type: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top performing gauntlets by a given metric.

        Args:
            metric: Metric to rank by
            n_top: Number of top gauntlets to return
            gauntlet_type: Optional filter by gauntlet type

        Returns:
            List of (gauntlet_id, score) tuples
        """
        scores = []

        for gauntlet_id in self.effectiveness_history.keys():
            summary = self.get_gauntlet_summary(gauntlet_id)
            if summary:
                if gauntlet_type and summary["gauntlet_type"] != gauntlet_type:
                    continue

                score = summary.get(metric, 0.0)
                scores.append((gauntlet_id, score))

        # Sort and return top
        reverse_sort = metric not in ["false_positive_rate", "execution_time"]
        scores.sort(key=lambda x: x[1], reverse=reverse_sort)
        return scores[:n_top]

    def generate_gauntlet_report(self, gauntlet_id: str) -> str:
        """
        Generate a human-readable report for a gauntlet.

        Args:
            gauntlet_id: ID of the gauntlet

        Returns:
            Formatted report string
        """
        summary = self.get_gauntlet_summary(gauntlet_id)

        if not summary:
            return f"No effectiveness data available for gauntlet {gauntlet_id}"

        report_lines = [
            f"# Gauntlet Effectiveness Report: {gauntlet_id}",
            "",
            "## Overview",
            f"- Type: {summary['gauntlet_type']}",
            f"- Total Runs: {summary['total_runs']}",
            f"- Average Catch Rate: {summary['avg_catch_rate']:.1%}",
            f"- Average False Positive Rate: {summary['avg_false_positive_rate']:.1%}",
            f"- Average Execution Time: {summary['avg_execution_time']:.2f}s",
            f"- Overall Effectiveness Score: {summary['effectiveness_score']:.2f}/1.00",
            "",
            "## Problem Type Effectiveness",
        ]

        for problem_type, score in summary["problem_type_effectiveness"].items():
            report_lines.append(f"- {problem_type}: {score:.1%}")

        report_lines.append("")
        report_lines.append("## Recommendations")

        if summary["recommended_improvements"]:
            for rec in summary["recommended_improvements"]:
                report_lines.append(f"- {rec}")
        else:
            report_lines.append("- Gauntlet is well-configured")

        # Get optimal configuration recommendations
        config_recs = self.recommend_optimal_configuration(gauntlet_id)
        if config_recs.get("rules_to_disable"):
            report_lines.append("")
            report_lines.append("## Rules to Disable")
            for rule in config_recs["rules_to_disable"]:
                report_lines.append(f"- {rule}")

        if config_recs.get("rules_to_tune"):
            report_lines.append("")
            report_lines.append("## Rules to Tune")
            for rule in config_recs["rules_to_tune"]:
                report_lines.append(f"- {rule}")

        return "\n".join(report_lines)

    def track_effectiveness_over_time(self, gauntlet_id: str) -> Dict[str, Any]:
        """
        Track gauntlet effectiveness over time.

        Args:
            gauntlet_id: ID of the gauntlet

        Returns:
            Dictionary with time-series data
        """
        if gauntlet_id not in self.effectiveness_history:
            return {}

        history = self.effectiveness_history[gauntlet_id]

        time_series = {
            "timestamps": [],
            "catch_rates": [],
            "false_positive_rates": [],
            "execution_times": [],
        }

        for artifact in history:
            time_series["timestamps"].append(artifact.created_at)
            time_series["catch_rates"].append(artifact.catch_rate)
            time_series["false_positive_rates"].append(artifact.false_positive_rate)
            time_series["execution_times"].append(artifact.execution_time)

        # Calculate trends
        if len(time_series["catch_rates"]) > 1:
            catch_rate_trend = np.polyfit(range(len(time_series["catch_rates"])), time_series["catch_rates"], 1)[0]
            time_series["catch_rate_trend"] = "improving" if catch_rate_trend > 0.01 else "stable" if catch_rate_trend > -0.01 else "declining"
        else:
            time_series["catch_rate_trend"] = "insufficient_data"

        return time_series


# ========== Convenience Functions ==========

def analyze_gauntlet(
    gauntlet_id: str,
    workflow_id: str,
    issues_caught: int,
    total_issues: int,
    false_positives: int,
    execution_time: float,
    db_path: str = "./knowledge_artifacts.db"
) -> Dict[str, Any]:
    """
    Convenience function to analyze gauntlet effectiveness.

    Args:
        gauntlet_id: ID of the gauntlet
        workflow_id: ID of the workflow
        issues_caught: Number of issues caught
        total_issues: Total number of issues
        false_positives: Number of false positives
        execution_time: Execution time in seconds
        db_path: Path to artifact database

    Returns:
        Dictionary with analysis results
    """
    analyzer = GauntletEffectivenessAnalyzer(db_path)

    artifact = analyzer.analyze_gauntlet_run(
        workflow_id=workflow_id,
        gauntlet_id=gauntlet_id,
        gauntlet_type="custom",
        total_checks=total_issues,
        issues_caught=issues_caught,
        false_positives=false_positives,
        execution_time=execution_time,
    )

    # Store artifact
    analyzer.artifact_manager.create_gauntlet_effectiveness(artifact)

    return analyzer.get_gauntlet_summary(gauntlet_id)
