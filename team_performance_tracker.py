"""
Team Performance Tracker - Stage 6 Knowledge Extraction

This module tracks and analyzes team performance across workflow executions.
It provides insights into optimal team compositions and training recommendations.
"""


import time
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from collections import defaultdict

from workflow_structures import (
    TeamPerformanceArtifact,
    KnowledgeArtifactManager,
)


class TeamPerformanceTracker:
    """
    Tracks and analyzes team performance.

    Features:
    - Team composition analysis
    - Performance by domain/complexity
    - Team velocity tracking
    - Optimal team recommendations
    - Skill gap identification
    - Training recommendations

    Attributes:
        artifact_manager: Manager for accessing/storing artifacts
        performance_history: Historical performance data by team
    """

    def __init__(self, db_path: str = "./knowledge_artifacts.db"):
        """
        Initialize the team performance tracker.

        Args:
            db_path: Path to artifact database
        """
        self.artifact_manager = KnowledgeArtifactManager(db_path)
        self.performance_history = defaultdict(list)
        self._load_performance_history()

    def _load_performance_history(self):
        """Load historical performance data from database."""
        artifacts = self.artifact_manager.list_team_performance(limit=10000)
        for artifact in artifacts:
            self.performance_history[artifact.team_id].append(artifact)

    def track_team_performance(
        self,
        workflow_id: str,
        team_id: str,
        team_composition: Dict[str, Any],
        problems_solved: int,
        total_problems: int,
        quality_metrics: Dict[str, float],
        execution_time: float,
        domain: str = "",
        complexity: int = 5
    ) -> TeamPerformanceArtifact:
        """
        Track performance of a team for a workflow.

        Args:
            workflow_id: ID of the workflow
            team_id: ID of the team
            team_composition: Composition of the team
            problems_solved: Number of problems solved
            total_problems: Total number of problems
            quality_metrics: Quality metrics dictionary
            execution_time: Execution time in seconds
            domain: Problem domain
            complexity: Problem complexity

        Returns:
            TeamPerformanceArtifact
        """
        # Calculate velocity (problems per hour)
        velocity = problems_solved / (execution_time / 3600) if execution_time > 0 else 0.0

        # Create artifact
        artifact = TeamPerformanceArtifact(
            artifact_id=f"team_perf_{team_id}_{workflow_id}_{int(time.time())}",
            source_workflow_id=workflow_id,
            team_id=team_id,
            team_composition=team_composition,
            velocity=velocity,
            quality_metrics=quality_metrics,
            confidence=0.8,
        )

        # Add to performance history
        self.performance_history[team_id].append(artifact)

        # Analyze performance
        self._analyze_team_performance(artifact, domain, complexity)

        return artifact

    def _analyze_team_performance(self, artifact: TeamPerformanceArtifact, domain: str, complexity: int):
        """
        Analyze team performance and update insights.

        Args:
            artifact: The performance artifact to analyze
            domain: Problem domain
            complexity: Problem complexity
        """
        team_id = artifact.team_id
        history = self.performance_history[team_id]

        # Calculate historical trends
        if len(history) > 1:
            # Velocity trend
            recent_velocity = np.mean([h.velocity for h in history[-5:]])
            overall_velocity = np.mean([h.velocity for h in history])
            velocity_trend = "increasing" if recent_velocity > overall_velocity * 1.1 else "stable" if recent_velocity > overall_velocity * 0.9 else "decreasing"

            # Success rate trend
            recent_success = np.mean([h.quality_metrics.get("success_rate", 0) for h in history[-5:]])
            overall_success = np.mean([h.quality_metrics.get("success_rate", 0) for h in history])
            success_trend = "improving" if recent_success > overall_success * 1.1 else "stable" if recent_success > overall_success * 0.9 else "declining"

            artifact.historical_trends = [
                {"metric": "velocity", "trend": velocity_trend, "recent": recent_velocity, "overall": overall_velocity},
                {"metric": "success_rate", "trend": success_trend, "recent": recent_success, "overall": overall_success},
            ]

        # Identify optimal domains
        domain_performance = defaultdict(list)
        for hist in history:
            # Extract domain from metadata or context
            hist_domain = hist.metadata.get("domain", "") if hist.metadata else ""
            if hist_domain:
                domain_performance[hist_domain].append(hist.get_overall_performance_score())

        optimal_domains = []
        for dom, scores in domain_performance.items():
            avg_score = np.mean(scores)
            if avg_score > 0.7:  # High performance threshold
                optimal_domains.append(dom)

        artifact.optimal_domains = optimal_domains

        # Identify skill gaps
        skill_gaps = []
        overall_success = artifact.quality_metrics.get("success_rate", 0)

        if overall_success < 0.5:
            skill_gaps.append("problem_solving")

        if artifact.velocity < 1.0:
            skill_gaps.append("efficiency")

        # Check domain-specific performance
        if domain and domain not in optimal_domains:
            skill_gaps.append(f"{domain}_expertise")

        # Check complexity handling
        complexity_performance = defaultdict(list)
        for hist in history:
            hist_complexity = hist.metadata.get("complexity", 5) if hist.metadata else 5
            complexity_performance[hist_complexity].append(hist.get_overall_performance_score())

        for comp, scores in complexity_performance.items():
            if np.mean(scores) < 0.5 and comp >= complexity:
                skill_gaps.append(f"complexity_{comp}")

        artifact.skill_gaps = skill_gaps

        # Generate training recommendations
        training_recommendations = []
        if "problem_solving" in skill_gaps:
            training_recommendations.append("Practice more algorithmic problems")
        if "efficiency" in skill_gaps:
            training_recommendations.append("Focus on optimizing code and reducing time complexity")
        if f"{domain}_expertise" in skill_gaps:
            training_recommendations.append(f"Study more {domain} problems and patterns")
        if any(f"complexity_{c}" in gap for c, gap in enumerate(skill_gaps)):
            training_recommendations.append("Work on progressively more complex problems")

        artifact.training_recommendations = training_recommendations

    def get_team_summary(self, team_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a team's performance.

        Args:
            team_id: ID of the team

        Returns:
            Dictionary with team performance summary
        """
        if team_id not in self.performance_history or not self.performance_history[team_id]:
            return None

        history = self.performance_history[team_id]

        # Calculate aggregate metrics
        avg_velocity = np.mean([h.velocity for h in history])
        avg_success_rate = np.mean([h.quality_metrics.get("success_rate", 0) for h in history])
        total_workflows = len(history)
        total_problems_solved = sum(h.quality_metrics.get("problems_solved", 0) for h in history if isinstance(h.quality_metrics.get("problems_solved"), (int, float)))

        # Get latest insights
        latest = history[-1]
        optimal_domains = latest.optimal_domains
        skill_gaps = latest.skill_gaps
        training_recs = latest.training_recommendations

        return {
            "team_id": team_id,
            "total_workflows": total_workflows,
            "total_problems_solved": total_problems_solved,
            "avg_velocity": avg_velocity,
            "avg_success_rate": avg_success_rate,
            "optimal_domains": optimal_domains,
            "skill_gaps": skill_gaps,
            "training_recommendations": training_recs,
            "latest_artifact_id": latest.artifact_id,
        }

    def recommend_team_for_problem(
        self,
        problem_domain: str,
        complexity: int,
        n_recommendations: int = 3
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Recommend the best teams for a given problem.

        Args:
            problem_domain: Domain of the problem
            complexity: Complexity level (1-10)
            n_recommendations: Number of recommendations to return

        Returns:
            List of (team_id, suitability_score, summary) tuples
        """
        recommendations = []

        for team_id in self.performance_history.keys():
            if not self.performance_history[team_id]:
                continue

            # Get latest artifact for this team
            latest = self.performance_history[team_id][-1]

            # Calculate suitability score
            score = latest.recommend_team_for_problem(problem_domain, complexity)

            # Get summary
            summary = self.get_team_summary(team_id)

            recommendations.append((team_id, score, summary))

        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def compare_teams(self, team_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance of multiple teams.

        Args:
            team_ids: List of team IDs to compare

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "teams": {},
            "metrics": ["velocity", "success_rate", "overall_performance"],
            "rankings": {},
        }

        # Collect metrics for each team
        for team_id in team_ids:
            summary = self.get_team_summary(team_id)
            if summary:
                comparison["teams"][team_id] = {
                    "velocity": summary["avg_velocity"],
                    "success_rate": summary["avg_success_rate"],
                    "overall_performance": summary.get("avg_success_rate", 0) * 0.6 + summary["avg_velocity"] * 0.4,
                    "total_workflows": summary["total_workflows"],
                    "optimal_domains": summary["optimal_domains"],
                }

        # Rank teams by each metric
        for metric in comparison["metrics"]:
            ranked = sorted(
                [(team_id, data[metric]) for team_id, data in comparison["teams"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            comparison["rankings"][metric] = [{"team_id": t, "value": v} for t, v in ranked]

        return comparison

    def identify_collaboration_patterns(self) -> Dict[str, List[str]]:
        """
        Identify patterns of collaboration between teams.

        Returns:
            Dictionary mapping team pairs to their collaboration frequency
        """
        # This would analyze workflows where multiple teams worked together
        # For now, return a placeholder
        return {}

    def get_top_performers(self, metric: str = "overall_performance", n_top: int = 5) -> List[Tuple[str, float]]:
        """
        Get top performing teams by a given metric.

        Args:
            metric: Metric to rank by ('velocity', 'success_rate', 'overall_performance')
            n_top: Number of top teams to return

        Returns:
            List of (team_id, score) tuples
        """
        scores = []

        for team_id in self.performance_history.keys():
            summary = self.get_team_summary(team_id)
            if summary:
                if metric == "velocity":
                    score = summary["avg_velocity"]
                elif metric == "success_rate":
                    score = summary["avg_success_rate"]
                else:  # overall_performance
                    score = summary["avg_success_rate"] * 0.6 + summary["avg_velocity"] * 0.4

                scores.append((team_id, score))

        # Sort and return top
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_top]

    def identify_skill_gaps(self, team_id: str) -> List[str]:
        """
        Identify skill gaps for a specific team.

        Args:
            team_id: ID of the team

        Returns:
            List of identified skill gaps
        """
        summary = self.get_team_summary(team_id)
        if not summary:
            return []

        return summary.get("skill_gaps", [])

    def recommend_training(self, team_id: str) -> List[str]:
        """
        Recommend training for a specific team based on performance gaps.

        Args:
            team_id: ID of the team

        Returns:
            List of training recommendations
        """
        summary = self.get_team_summary(team_id)
        if not summary:
            return []

        return summary.get("training_recommendations", [])

    def generate_team_report(self, team_id: str) -> str:
        """
        Generate a human-readable report for a team.

        Args:
            team_id: ID of the team

        Returns:
            Formatted report string
        """
        summary = self.get_team_summary(team_id)

        if not summary:
            return f"No performance data available for team {team_id}"

        report_lines = [
            f"# Team Performance Report: {team_id}",
            "",
            "## Overview",
            f"- Total Workflows: {summary['total_workflows']}",
            f"- Problems Solved: {summary['total_problems_solved']}",
            f"- Average Velocity: {summary['avg_velocity']:.2f} problems/hour",
            f"- Average Success Rate: {summary['avg_success_rate']:.1%}",
            "",
            "## Strengths",
        ]

        if summary['optimal_domains']:
            report_lines.append(f"- Excels in: {', '.join(summary['optimal_domains'])}")
        else:
            report_lines.append("- No specific domain strengths identified yet")

        report_lines.append("")
        report_lines.append("## Areas for Improvement")

        if summary['skill_gaps']:
            report_lines.append("- Skill gaps:")
            for gap in summary['skill_gaps']:
                report_lines.append(f"  - {gap}")
        else:
            report_lines.append("- No significant skill gaps identified")

        report_lines.append("")
        report_lines.append("## Training Recommendations")

        if summary['training_recommendations']:
            for rec in summary['training_recommendations']:
                report_lines.append(f"- {rec}")
        else:
            report_lines.append("- Keep up the good work!")

        return "\n".join(report_lines)


# ========== Convenience Functions ==========

def track_team(
    team_id: str,
    workflow_id: str,
    problems_solved: int,
    total_problems: int,
    execution_time: float,
    db_path: str = "./knowledge_artifacts.db"
) -> Dict[str, Any]:
    """
    Convenience function to track team performance.

    Args:
        team_id: ID of the team
        workflow_id: ID of the workflow
        problems_solved: Number of problems solved
        total_problems: Total number of problems
        execution_time: Execution time in seconds
        db_path: Path to artifact database

    Returns:
        Dictionary with tracking results
    """
    tracker = TeamPerformanceTracker(db_path)

    artifact = tracker.track_team_performance(
        workflow_id=workflow_id,
        team_id=team_id,
        team_composition={},
        problems_solved=problems_solved,
        total_problems=total_problems,
        quality_metrics={"success_rate": problems_solved / total_problems if total_problems > 0 else 0},
        execution_time=execution_time,
    )

    # Store artifact
    tracker.artifact_manager.create_team_performance(artifact)

    return tracker.get_team_summary(team_id)
