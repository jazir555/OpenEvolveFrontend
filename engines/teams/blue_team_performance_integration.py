"""
Integration module for adding performance tracking to existing Blue Team.

This module provides wrapper functions and classes to integrate performance tracking
with the existing BlueTeam implementation in blue_team.py without requiring
modifications to the original file.

Usage:
    from blue_team import BlueTeam
    from blue_team_performance_integration import PerformanceTrackingBlueTeam

    # Wrap existing BlueTeam with performance tracking
    blue_team = BlueTeam()
    tracked_team = PerformanceTrackingBlueTeam(blue_team)

    # Use as normal - performance is automatically tracked
    assessment = tracked_team.assess_and_fix(content, issues)
"""
from __future__ import annotations



import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from blue_team_performance_tracker import (
    BlueTeamPerformanceTracker,
    SpecializationType,
    track_blue_team_performance,
)
from blue_team import (
    BlueTeam,
    BlueTeamAssessment,
    BlueTeamMember,
    FixType,
    FixPriority,
    IssueFinding,
)

logger = logging.getLogger(__name__)


class FixTypeMapper:
    """Maps BlueTeam FixType to PerformanceTracker SpecializationType."""

    FIX_TYPE_TO_SPECIALIZATION = {
        FixType.SECURITY_PATCH: SpecializationType.SECURITY,
        FixType.PERFORMANCE_OPTIMIZATION: SpecializationType.PERFORMANCE,
        FixType.LOGIC_CORRECTION: SpecializationType.LOGIC,
        FixType.DOCUMENTATION_ADDITION: SpecializationType.DOCUMENTATION,
        FixType.CODE_REFACTORING: SpecializationType.REFACTORING,
        FixType.MAINTAINABILITY_IMPROVEMENT: SpecializationType.REFACTORING,
        FixType.CLARITY_IMPROVEMENT: SpecializationType.DOCUMENTATION,
        FixType.ERROR_HANDLING: SpecializationType.LOGIC,
        FixType.INPUT_VALIDATION: SpecializationType.SECURITY,
        FixType.COMPLIANCE_FIX: SpecializationType.SECURITY,
        FixType.STRUCTURE_REORGANIZATION: SpecializationType.ARCHITECTURE,
    }

    @classmethod
    def to_specialization(cls, fix_type: FixType) -> SpecializationType:
        """Convert FixType to SpecializationType."""
        return cls.FIX_TYPE_TO_SPECIALIZATION.get(
            fix_type,
            SpecializationType.LOGIC  # Default fallback
        )

    @classmethod
    def to_specializations(cls, fix_types: List[FixType]) -> List[SpecializationType]:
        """Convert list of FixTypes to SpecializationTypes."""
        return [cls.to_specialization(ft) for ft in fix_types]


class PerformanceTrackingBlueTeam:
    """
    Wrapper around BlueTeam that adds performance tracking.

    This wrapper intercepts BlueTeam operations and automatically tracks
    performance metrics without modifying the original BlueTeam class.
    """

    def __init__(
        self,
        blue_team: BlueTeam,
        storage_path: Optional[str] = None,
        auto_track: bool = True
    ):
        """
        Initialize performance tracking wrapper.

        Args:
            blue_team: The BlueTeam instance to wrap
            storage_path: Optional path for performance data storage
            auto_track: Whether to automatically track all operations
        """
        self.blue_team = blue_team
        self.tracker = BlueTeamPerformanceTracker(storage_path=storage_path)
        self.auto_track = auto_track

        # Register all existing team members
        self._register_team_members()

        # Map team member names to IDs
        self.member_name_to_id = {
            member.name: member.name
            for member in self.blue_team.team_members
        }

        logger.info(f"Performance tracking enabled for Blue Team with {len(self.blue_team.team_members)} members")

    def _register_team_members(self):
        """Register all BlueTeam members with the performance tracker."""
        for member in self.blue_team.team_members:
            # Map member's specializations to performance tracker specializations
            specializations = FixTypeMapper.to_specializations(member.specializations)

            # Register with tracker (will create TeamMemberPerformance if needed)
            self.tracker.register_team_member(member.name)

            logger.debug(f"Registered team member: {member.name} with specializations: {[s.value for s in specializations]}")

    def _get_task_specializations(self, issues: List[IssueFinding]) -> List[SpecializationType]:
        """
        Determine required specializations based on issues.

        Args:
            issues: List of issues to analyze

        Returns:
            List of required specializations
        """
        # Analyze issues to determine required specializations
        # This is a simplified implementation - could be more sophisticated
        specializations = set()

        for issue in issues:
            # Map issue severity/category to specialization
            if hasattr(issue, 'category'):
                category = str(issue.category).lower()
                if 'security' in category:
                    specializations.add(SpecializationType.SECURITY)
                elif 'performance' in category:
                    specializations.add(SpecializationType.PERFORMANCE)
                elif 'logic' in category:
                    specializations.add(SpecializationType.LOGIC)

        # Default to LOGIC if no specific specialization identified
        if not specializations:
            specializations.add(SpecializationType.LOGIC)

        return list(specializations)

    def _assess_difficulty(self, issues: List[IssueFinding], content: str) -> float:
        """
        Assess task difficulty based on issues and content.

        Args:
            issues: List of issues
            content: Content being analyzed

        Returns:
            Difficulty score (0-1)
        """
        # Simple heuristic: more issues + longer content = higher difficulty
        issue_count = len(issues)
        content_length = len(content)

        # Normalize
        difficulty = min(1.0, (issue_count / 20.0) + (content_length / 100000.0))
        return max(0.1, difficulty)  # Ensure minimum difficulty of 0.1

    def _select_best_team_member(
        self,
        specializations: List[SpecializationType],
        difficulty: float
    ) -> str:
        """
        Select the best team member for a task.

        Args:
            specializations: Required specializations
            difficulty: Task difficulty

        Returns:
            Team member name
        """
        optimal_member = self.tracker.get_optimal_team_member(
            required_specializations=specializations,
            difficulty_level=difficulty
        )

        if optimal_member:
            return optimal_member

        # Fallback: select member with matching specializations
        for member in self.blue_team.team_members:
            member_specs = FixTypeMapper.to_specializations(member.specializations)
            if any(spec in member_specs for spec in specializations):
                return member.name

        # Final fallback: first member
        return self.blue_team.team_members[0].name if self.blue_team.team_members else "default"

    def assess_and_fix(
        self,
        content: str,
        issues: List[IssueFinding],
        content_type: str = "general",
        fix_limit: int = 10
    ) -> BlueTeamAssessment:
        """
        Assess content and apply fixes with performance tracking.

        This wraps the original BlueTeam.assess_and_fix method and adds
        automatic performance tracking.

        Args:
            content: Content to assess and fix
            issues: List of issues to address
            content_type: Type of content
            fix_limit: Maximum number of fixes to apply

        Returns:
            BlueTeamAssessment with fixes applied
        """
        if not self.auto_track:
            # Pass through to original BlueTeam without tracking
            return self.blue_team.assess_and_fix(content, issues, content_type, fix_limit)

        # Determine task characteristics
        specializations = self._get_task_specializations(issues)
        difficulty = self._assess_difficulty(issues, content)

        # Select best team member
        team_member_id = self._select_best_team_member(specializations, difficulty)

        # Create task ID
        task_id = f"fix_task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Track performance
        with track_blue_team_performance(
            tracker=self.tracker,
            task_id=task_id,
            team_member_id=team_member_id,
            specializations=specializations,
            difficulty_level=difficulty
        ) as record:
            # Execute the actual fix operation
            assessment = self.blue_team.assess_and_fix(content, issues, content_type, fix_limit)

            # Update quality score based on assessment results
            quality_score = assessment.overall_improvement_score

            # The context manager will automatically complete the task
            # We need to update it with the actual quality score
            self.tracker.complete_task(task_id, True, quality_score)

        return assessment

    def apply_fixes(
        self,
        content: str,
        fix_suggestions: list,
        content_type: str = "general"
    ) -> BlueTeamAssessment:
        """
        Apply specific fix suggestions with performance tracking.

        Args:
            content: Content to fix
            fix_suggestions: List of fix suggestions to apply
            content_type: Type of content

        Returns:
            BlueTeamAssessment with fixes applied
        """
        if not self.auto_track:
            return self.blue_team.apply_fixes(content, fix_suggestions, content_type)

        # Determine task characteristics
        specializations = []
        for suggestion in fix_suggestions:
            spec = FixTypeMapper.to_specialization(suggestion.fix_type)
            if spec not in specializations:
                specializations.append(spec)

        difficulty = min(1.0, len(fix_suggestions) / 20.0)  # More fixes = higher difficulty

        # Select best team member
        team_member_id = self._select_best_team_member(specializations, difficulty)

        # Create task ID
        task_id = f"apply_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Track performance
        with track_blue_team_performance(
            tracker=self.tracker,
            task_id=task_id,
            team_member_id=team_member_id,
            specializations=specializations,
            difficulty_level=difficulty
        ):
            # Execute the actual fix application
            assessment = self.blue_team.apply_fixes(content, fix_suggestions, content_type)

            # Update with actual quality score
            quality_score = assessment.overall_improvement_score
            self.tracker.complete_task(task_id, True, quality_score)

        return assessment

    def get_performance_report(
        self,
        time_window_days: int = 7,
        format: str = 'dict'
    ) -> Dict[str, Any]:
        """
        Generate performance report for the Blue Team.

        Args:
            time_window_days: Number of days to include in report
            format: Report format ('dict', 'json')

        Returns:
            Performance report dictionary
        """
        report = self.tracker.generate_report(time_window_days=time_window_days)

        if format == 'json':
            import json
            return json.dumps(report, indent=2)

        return report

    def get_team_member_performance(self, member_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance data for a specific team member.

        Args:
            member_name: Name of the team member

        Returns:
            Performance data dictionary, or None if not found
        """
        member_performance = self.tracker.get_team_member_performance(member_name)

        if not member_performance:
            return None

        return {
            'member_name': member_name,
            'specialization_effectiveness': member_performance.get_specialization_effectiveness(),
            'performance_trend': member_performance.get_performance_trend(),
            'reliability_score': member_performance.calculate_reliability_score(),
            'strengths': [s.value for s in member_performance.get_strengths_and_weaknesses()[0]],
            'weaknesses': [w.value for w in member_performance.get_strengths_and_weaknesses()[1]],
        }

    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for performance alerts across the team.

        Returns:
            List of alert dictionaries
        """
        alerts = self.tracker.check_performance_alerts()

        return [
            {
                'level': alert.level.value,
                'type': alert.metric_type.value,
                'message': alert.message,
                'team_member': alert.team_member_id,
                'recommendations': alert.recommendations
            }
            for alert in alerts
        ]

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get performance optimization recommendations.

        Returns:
            List of recommendation dictionaries
        """
        return self.tracker.get_workload_recommendations()

    def export_performance_report(
        self,
        output_path: str,
        format: str = 'html',
        time_window_days: int = 7
    ):
        """
        Export performance report to file.

        Args:
            output_path: Path to output file
            format: Export format ('html', 'json', 'csv')
            time_window_days: Number of days to include
        """
        self.tracker.generate_report(
            time_window_days=time_window_days,
            format=format,
            output_path=output_path
        )

        logger.info(f"Performance report exported to {output_path}")

    def get_all_team_members_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance data for all team members.

        Returns:
            Dictionary mapping member names to performance data
        """
        all_performance = {}

        for member in self.blue_team.team_members:
            performance = self.get_team_member_performance(member.name)
            if performance:
                all_performance[member.name] = performance

        return all_performance

    # Delegate all other attributes and methods to the wrapped BlueTeam
    def __getattr__(self, name):
        """Delegate undefined attributes to the wrapped BlueTeam."""
        return getattr(self.blue_team, name)


def enable_performance_tracking(
    blue_team: BlueTeam,
    storage_path: Optional[str] = None
) -> PerformanceTrackingBlueTeam:
    """
    Enable performance tracking for an existing BlueTeam instance.

    This is a convenience function for wrapping a BlueTeam with performance tracking.

    Args:
        blue_team: The BlueTeam instance to wrap
        storage_path: Optional path for performance data storage

    Returns:
        PerformanceTrackingBlueTeam wrapper instance

    Example:
        >>> from blue_team import BlueTeam
        >>> from blue_team_performance_integration import enable_performance_tracking
        >>>
        >>> blue_team = BlueTeam()
        >>> tracked_team = enable_performance_tracking(blue_team)
        >>>
        >>> # Use tracked_team as you would blue_team
        >>> assessment = tracked_team.assess_and_fix(content, issues)
    """
    return PerformanceTrackingBlueTeam(blue_team, storage_path=storage_path)


# Convenience functions for common operations

def quick_team_summary(blue_team: BlueTeam, days: int = 7) -> str:
    """
    Get a quick performance summary for a Blue Team.

    Args:
        blue_team: BlueTeam instance (with or without performance tracking)
        days: Number of days to analyze

    Returns:
        Summary string
    """
    # Check if already wrapped
    if isinstance(blue_team, PerformanceTrackingBlueTeam):
        tracker = blue_team.tracker
    else:
        # Create temporary tracker
        tracker = BlueTeamPerformanceTracker()

    from blue_team_performance_tracker import quick_performance_report
    return quick_performance_report(tracker, days=days)


def get_best_member_for_task(
    blue_team: BlueTeam,
    issues: List[IssueFinding],
    content: str
) -> str:
    """
    Get the best team member for a specific task.

    Args:
        blue_team: BlueTeam instance
        issues: Issues to address
        content: Content to analyze

    Returns:
        Name of the best team member
    """
    if isinstance(blue_team, PerformanceTrackingBlueTeam):
        # Determine characteristics
        specializations = blue_team._get_task_specializations(issues)
        difficulty = blue_team._assess_difficulty(issues, content)

        return blue_team._select_best_team_member(specializations, difficulty)
    else:
        # Fallback: select based on specializations
        return blue_team.team_members[0].name
