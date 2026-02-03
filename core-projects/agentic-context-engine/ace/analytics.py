"""Playbook/Skillbook Analytics for ACE.

This module provides comprehensive analytics for skillbook performance tracking,
including statistics generation, usage tracking, effectiveness scoring, and
analytics export functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .skillbook import Skill, Skillbook


@dataclass
class SkillbookStats:
    """
    Statistics about a skillbook.

    Attributes:
        total_skills: Total number of skills (active only)
        high_performing: Skills with helpful > 5 and harmful < 2
        problematic: Skills where harmful >= helpful (and harmful > 0)
        unused: Skills where helpful + harmful = 0
        by_section: Count of skills per section
        average_helpful: Mean helpful score across all skills
        average_harmful: Mean harmful score across all skills
        total_helpful: Sum of all helpful votes
        total_harmful: Sum of all harmful votes
        total_neutral: Sum of all neutral votes
    """

    total_skills: int
    high_performing: int
    problematic: int
    unused: int
    by_section: Dict[str, int]
    average_helpful: float
    average_harmful: float
    total_helpful: int
    total_harmful: int
    total_neutral: int


def get_skillbook_stats(skillbook: Skillbook) -> SkillbookStats:
    """
    Generate comprehensive statistics about a skillbook.

    Analyzes:
    - Total skill count
    - High-performing skills (helpful > 5, harmful < 2)
    - Problematic skills (harmful >= helpful, harmful > 0)
    - Unused skills (helpful + harmful = 0)
    - Skills per section
    - Average helpful/harmful scores

    Args:
        skillbook: The Skillbook instance to analyze

    Returns:
        SkillbookStats dataclass containing all analytics

    Example:
        >>> from ace import Skillbook
        >>> from ace.analytics import get_skillbook_stats
        >>> skillbook = Skillbook()
        >>> skillbook.add_skill("general", "Be clear", metadata={"helpful": 10, "harmful": 0})
        >>> stats = get_skillbook_stats(skillbook)
        >>> print(f"Total skills: {stats.total_skills}")
    """
    skills = skillbook.skills(include_invalid=False)

    if not skills:
        return SkillbookStats(
            total_skills=0,
            high_performing=0,
            problematic=0,
            unused=0,
            by_section={},
            average_helpful=0.0,
            average_harmful=0.0,
            total_helpful=0,
            total_harmful=0,
            total_neutral=0,
        )

    # Calculate totals
    total_helpful = sum(skill.helpful for skill in skills)
    total_harmful = sum(skill.harmful for skill in skills)
    total_neutral = sum(skill.neutral for skill in skills)
    total_skills = len(skills)

    # Calculate averages
    average_helpful = total_helpful / total_skills if total_skills > 0 else 0.0
    average_harmful = total_harmful / total_skills if total_skills > 0 else 0.0

    # Categorize skills
    high_performing = 0
    problematic = 0
    unused = 0
    by_section: Dict[str, int] = {}

    for skill in skills:
        # Count by section
        by_section[skill.section] = by_section.get(skill.section, 0) + 1

        # Check if high performing (helpful > 5 AND harmful < 2)
        if skill.helpful > 5 and skill.harmful < 2:
            high_performing += 1

        # Check if problematic (harmful >= helpful AND harmful > 0)
        if skill.harmful > 0 and skill.harmful >= skill.helpful:
            problematic += 1

        # Check if unused (no votes at all)
        if skill.helpful + skill.harmful == 0:
            unused += 1

    return SkillbookStats(
        total_skills=total_skills,
        high_performing=high_performing,
        problematic=problematic,
        unused=unused,
        by_section=by_section,
        average_helpful=average_helpful,
        average_harmful=average_harmful,
        total_helpful=total_helpful,
        total_harmful=total_harmful,
        total_neutral=total_neutral,
    )


class SkillUsageTracker:
    """
    Track which skills are used by the Agent.

    This analyzes agent outputs to see which skills are cited
    and how effective they are.

    Attributes:
        citations: Dict mapping skill_id to citation counts
        correct_usage: Dict mapping skill_id to correct usage counts
        incorrect_usage: Dict mapping skill_id to incorrect usage counts

    Example:
        >>> tracker = SkillUsageTracker()
        >>> tracker.track_citation("general-00001", was_correct=True)
        >>> stats = tracker.get_usage_stats()
        >>> print(stats["general-00001"])
        {'citations': 1, 'correct': 1, 'incorrect': 0}
    """

    def __init__(self) -> None:
        """Initialize an empty usage tracker."""
        self._citations: Dict[str, int] = {}
        self._correct_usage: Dict[str, int] = {}
        self._incorrect_usage: Dict[str, int] = {}

    def track_citation(self, skill_id: str, was_correct: bool) -> None:
        """
        Record a citation event for a skill.

        Args:
            skill_id: The ID of the skill being cited
            was_correct: Whether the citation led to a correct result

        Example:
            >>> tracker = SkillUsageTracker()
            >>> tracker.track_citation("math-00001", was_correct=True)
            >>> tracker.track_citation("math-00001", was_correct=False)
        """
        # Increment citation count
        self._citations[skill_id] = self._citations.get(skill_id, 0) + 1

        # Track correctness
        if was_correct:
            self._correct_usage[skill_id] = self._correct_usage.get(skill_id, 0) + 1
        else:
            self._incorrect_usage[skill_id] = self._incorrect_usage.get(skill_id, 0) + 1

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get usage statistics for all tracked skills.

        Returns:
            Dict mapping skill_id to stats dict with:
            - 'citations': total number of citations
            - 'correct': number of correct usages
            - 'incorrect': number of incorrect usages

        Example:
            >>> tracker = SkillUsageTracker()
            >>> tracker.track_citation("skill-001", was_correct=True)
            >>> stats = tracker.get_usage_stats()
            >>> stats["skill-001"]
            {'citations': 1, 'correct': 1, 'incorrect': 0}
        """
        all_skill_ids = set(self._citations.keys())

        result = {}
        for skill_id in all_skill_ids:
            result[skill_id] = {
                "citations": self._citations.get(skill_id, 0),
                "correct": self._correct_usage.get(skill_id, 0),
                "incorrect": self._incorrect_usage.get(skill_id, 0),
            }

        return result

    def get_most_used_skills(self, limit: int = 10) -> List[tuple[str, int]]:
        """
        Get top N most cited skills.

        Args:
            limit: Maximum number of skills to return (default: 10)

        Returns:
            List of (skill_id, citation_count) tuples sorted by count descending

        Example:
            >>> tracker = SkillUsageTracker()
            >>> tracker.track_citation("skill-001", was_correct=True)
            >>> tracker.track_citation("skill-002", was_correct=True)
            >>> tracker.track_citation("skill-001", was_correct=True)
            >>> tracker.get_most_used_skills(2)
            [('skill-001', 2), ('skill-002', 1)]
        """
        sorted_skills = sorted(
            self._citations.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_skills[:limit]

    def get_effectiveness_by_skill(self) -> Dict[str, float]:
        """
        Get correctness rate per skill.

        Returns:
            Dict mapping skill_id to effectiveness score (0.0 to 1.0)
            Score = correct / (correct + incorrect)

        Example:
            >>> tracker = SkillUsageTracker()
            >>> tracker.track_citation("skill-001", was_correct=True)
            >>> tracker.track_citation("skill-001", was_correct=False)
            >>> tracker.track_citation("skill-001", was_correct=True)
            >>> tracker.get_effectiveness_by_skill()
            {'skill-001': 0.6666666666666666}
        """
        effectiveness = {}

        for skill_id in self._citations.keys():
            correct = self._correct_usage.get(skill_id, 0)
            incorrect = self._incorrect_usage.get(skill_id, 0)
            total = correct + incorrect

            if total > 0:
                effectiveness[skill_id] = correct / total
            else:
                effectiveness[skill_id] = 0.0

        return effectiveness


def calculate_effectiveness_score(skill: Skill) -> float:
    """
    Calculate effectiveness score for a skill.

    Score = (helpful - harmful) / (helpful + harmful + 1)

    Returns float between -1.0 (always harmful) and 1.0 (always helpful).

    Args:
        skill: The Skill instance to score

    Returns:
        Effectiveness score from -1.0 to 1.0

    Example:
        >>> from ace import Skill
        >>> from ace.analytics import calculate_effectiveness_score
        >>> skill = Skill(id="test", section="test", content="Test",
        ...               helpful=10, harmful=2)
        >>> calculate_effectiveness_score(skill)
        0.6363636363636364
    """
    numerator = skill.helpful - skill.harmful
    denominator = skill.helpful + skill.harmful + 1

    if denominator == 0:
        return 0.0

    return numerator / denominator


def export_analytics(
    skillbook: Skillbook, usage_tracker: Optional[SkillUsageTracker] = None
) -> Dict[str, Any]:
    """
    Export analytics to JSON-serializable dictionary.

    Includes:
    - SkillbookStats
    - Per-skill effectiveness scores
    - Usage statistics (if tracker provided)
    - Top skills by various metrics

    Args:
        skillbook: The Skillbook to analyze
        usage_tracker: Optional SkillUsageTracker for usage analytics

    Returns:
        Dict ready for json.dump()

    Example:
        >>> from ace import Skillbook
        >>> from ace.analytics import export_analytics
        >>> skillbook = Skillbook()
        >>> skillbook.add_skill("general", "Be clear")
        >>> analytics = export_analytics(skillbook)
        >>> import json
        >>> print(json.dumps(analytics, indent=2))
    """
    stats = get_skillbook_stats(skillbook)

    # Calculate per-skill effectiveness
    skills = skillbook.skills(include_invalid=False)
    skill_effectiveness = {
        skill.id: {
            "section": skill.section,
            "content": skill.content,
            "helpful": skill.helpful,
            "harmful": skill.harmful,
            "neutral": skill.neutral,
            "effectiveness_score": calculate_effectiveness_score(skill),
        }
        for skill in skills
    }

    # Sort skills by effectiveness
    top_performing = sorted(
        skill_effectiveness.items(), key=lambda x: x[1]["effectiveness_score"], reverse=True
    )[:10]

    worst_performing = sorted(
        skill_effectiveness.items(), key=lambda x: x[1]["effectiveness_score"]
    )[:10]

    result = {
        "summary": {
            "total_skills": stats.total_skills,
            "high_performing": stats.high_performing,
            "problematic": stats.problematic,
            "unused": stats.unused,
            "average_helpful": stats.average_helpful,
            "average_harmful": stats.average_harmful,
            "total_votes": {
                "helpful": stats.total_helpful,
                "harmful": stats.total_harmful,
                "neutral": stats.total_neutral,
            },
        },
        "by_section": stats.by_section,
        "top_performing_skills": [
            {"id": sid, **data} for sid, data in top_performing
        ],
        "worst_performing_skills": [
            {"id": sid, **data} for sid, data in worst_performing
        ],
        "all_skills": skill_effectiveness,
    }

    # Add usage statistics if tracker provided
    if usage_tracker is not None:
        usage_stats = usage_tracker.get_usage_stats()
        effectiveness = usage_tracker.get_effectiveness_by_skill()

        result["usage"] = {
            "most_cited": [
                {"skill_id": sid, "citations": count}
                for sid, count in usage_tracker.get_most_used_skills(10)
            ],
            "by_skill": usage_stats,
            "effectiveness_by_skill": effectiveness,
        }

    return result
