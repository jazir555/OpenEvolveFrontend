"""Tests for Skillbook Analytics functionality."""

import unittest

from ace import Skill, Skillbook
from ace.analytics import (
    SkillUsageTracker,
    calculate_effectiveness_score,
    export_analytics,
    get_skillbook_stats,
    SkillbookStats,
)


class TestSkillbookStats(unittest.TestCase):
    """Test SkillbookStats dataclass and generation."""

    def test_skillbook_stats_empty(self):
        """Test statistics generation for empty skillbook."""
        skillbook = Skillbook()
        stats = get_skillbook_stats(skillbook)

        self.assertEqual(stats.total_skills, 0)
        self.assertEqual(stats.high_performing, 0)
        self.assertEqual(stats.problematic, 0)
        self.assertEqual(stats.unused, 0)
        self.assertEqual(stats.by_section, {})
        self.assertEqual(stats.average_helpful, 0.0)
        self.assertEqual(stats.average_harmful, 0.0)
        self.assertEqual(stats.total_helpful, 0)
        self.assertEqual(stats.total_harmful, 0)
        self.assertEqual(stats.total_neutral, 0)

    def test_skillbook_stats_with_data(self):
        """Test statistics generation with sample skills."""
        skillbook = Skillbook()

        # Add various skills with different performance levels
        skillbook.add_skill(
            "general", "Be clear", metadata={"helpful": 10, "harmful": 0, "neutral": 2}
        )
        skillbook.add_skill(
            "math", "Show work", metadata={"helpful": 5, "harmful": 1, "neutral": 0}
        )
        skillbook.add_skill(
            "coding", "Use comments", metadata={"helpful": 0, "harmful": 0, "neutral": 0}
        )

        stats = get_skillbook_stats(skillbook)

        self.assertEqual(stats.total_skills, 3)
        self.assertEqual(stats.total_helpful, 15)
        self.assertEqual(stats.total_harmful, 1)
        self.assertEqual(stats.total_neutral, 2)
        self.assertAlmostEqual(stats.average_helpful, 5.0)
        self.assertAlmostEqual(stats.average_harmful, 0.333333, places=5)

    def test_high_performing_detection(self):
        """Test detection of high-performing skills (helpful > 5, harmful < 2)."""
        skillbook = Skillbook()

        # High performing: helpful > 5 AND harmful < 2
        skillbook.add_skill(
            "general", "Excellent skill", metadata={"helpful": 10, "harmful": 0}
        )
        skillbook.add_skill(
            "math", "Good skill", metadata={"helpful": 6, "harmful": 1}
        )

        # Not high performing: harmful >= 2
        skillbook.add_skill(
            "coding", "OK skill", metadata={"helpful": 10, "harmful": 2}
        )

        # Not high performing: helpful <= 5
        skillbook.add_skill(
            "writing", "Mediocre skill", metadata={"helpful": 5, "harmful": 0}
        )

        stats = get_skillbook_stats(skillbook)

        self.assertEqual(stats.high_performing, 2)

    def test_problematic_detection(self):
        """Test detection of problematic skills (harmful >= helpful, harmful > 0)."""
        skillbook = Skillbook()

        # Problematic: harmful >= helpful AND harmful > 0
        skillbook.add_skill(
            "bad", "Bad skill", metadata={"helpful": 2, "harmful": 3}
        )
        skillbook.add_skill(
            "terrible", "Terrible skill", metadata={"helpful": 5, "harmful": 5}
        )

        # Not problematic: harmful = 0
        skillbook.add_skill(
            "good", "Good skill", metadata={"helpful": 10, "harmful": 0}
        )

        # Not problematic: helpful > harmful
        skillbook.add_skill(
            "ok", "OK skill", metadata={"helpful": 10, "harmful": 2}
        )

        stats = get_skillbook_stats(skillbook)

        self.assertEqual(stats.problematic, 2)

    def test_unused_detection(self):
        """Test detection of unused skills (helpful + harmful = 0)."""
        skillbook = Skillbook()

        # Unused: no votes
        skillbook.add_skill("new", "New skill", metadata={"helpful": 0, "harmful": 0})

        # Used: has votes
        skillbook.add_skill(
            "used", "Used skill", metadata={"helpful": 5, "harmful": 1}
        )

        # Only neutral votes (still unused)
        skillbook.add_skill(
            "neutral", "Neutral skill", metadata={"helpful": 0, "harmful": 0, "neutral": 5}
        )

        stats = get_skillbook_stats(skillbook)

        # Both "new" and "neutral" have helpful + harmful = 0
        self.assertEqual(stats.unused, 2)

    def test_per_section_counts(self):
        """Test counting skills per section."""
        skillbook = Skillbook()

        skillbook.add_skill("general", "Skill 1")
        skillbook.add_skill("general", "Skill 2")
        skillbook.add_skill("math", "Skill 3")
        skillbook.add_skill("math", "Skill 4")
        skillbook.add_skill("math", "Skill 5")
        skillbook.add_skill("coding", "Skill 6")

        stats = get_skillbook_stats(skillbook)

        self.assertEqual(stats.by_section["general"], 2)
        self.assertEqual(stats.by_section["math"], 3)
        self.assertEqual(stats.by_section["coding"], 1)

    def test_invalid_skills_excluded(self):
        """Test that soft-deleted skills are excluded from stats."""
        skillbook = Skillbook()

        skillbook.add_skill("general", "Active skill", metadata={"helpful": 10})
        skill_id = skillbook.add_skill("math", "Deleted skill").id
        skillbook.remove_skill(skill_id, soft=True)

        stats = get_skillbook_stats(skillbook)

        self.assertEqual(stats.total_skills, 1)
        self.assertEqual(stats.total_helpful, 10)


class TestEffectivenessScore(unittest.TestCase):
    """Test effectiveness score calculation."""

    def test_effectiveness_score(self):
        """Test basic effectiveness score calculation."""
        skill = Skill(
            id="test", section="test", content="Test", helpful=10, harmful=2
        )
        score = calculate_effectiveness_score(skill)

        # (10 - 2) / (10 + 2 + 1) = 8 / 13
        self.assertAlmostEqual(score, 8 / 13)

    def test_perfect_effectiveness(self):
        """Test effectiveness score for perfect skill (no harmful)."""
        skill = Skill(
            id="test", section="test", content="Test", helpful=10, harmful=0
        )
        score = calculate_effectiveness_score(skill)

        # (10 - 0) / (10 + 0 + 1) = 10 / 11
        self.assertAlmostEqual(score, 10 / 11)

    def test_terrible_effectiveness(self):
        """Test effectiveness score for terrible skill (all harmful)."""
        skill = Skill(
            id="test", section="test", content="Test", helpful=0, harmful=10
        )
        score = calculate_effectiveness_score(skill)

        # (0 - 10) / (0 + 10 + 1) = -10 / 11
        self.assertAlmostEqual(score, -10 / 11)

    def test_balanced_effectiveness(self):
        """Test effectiveness score for balanced skill (equal helpful/harmful)."""
        skill = Skill(
            id="test", section="test", content="Test", helpful=5, harmful=5
        )
        score = calculate_effectiveness_score(skill)

        # (5 - 5) / (5 + 5 + 1) = 0 / 11
        self.assertAlmostEqual(score, 0.0)

    def test_no_votes_effectiveness(self):
        """Test effectiveness score for skill with no votes."""
        skill = Skill(
            id="test", section="test", content="Test", helpful=0, harmful=0
        )
        score = calculate_effectiveness_score(skill)

        # (0 - 0) / (0 + 0 + 1) = 0 / 1
        self.assertAlmostEqual(score, 0.0)


class TestSkillUsageTracker(unittest.TestCase):
    """Test SkillUsageTracker functionality."""

    def test_usage_tracker_basic(self):
        """Test basic citation tracking."""
        tracker = SkillUsageTracker()

        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-001", was_correct=False)
        tracker.track_citation("skill-002", was_correct=True)

        stats = tracker.get_usage_stats()

        self.assertEqual(stats["skill-001"]["citations"], 2)
        self.assertEqual(stats["skill-001"]["correct"], 1)
        self.assertEqual(stats["skill-001"]["incorrect"], 1)

        self.assertEqual(stats["skill-002"]["citations"], 1)
        self.assertEqual(stats["skill-002"]["correct"], 1)
        self.assertEqual(stats["skill-002"]["incorrect"], 0)

    def test_usage_tracker_most_used(self):
        """Test getting most used skills."""
        tracker = SkillUsageTracker()

        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-002", was_correct=True)
        tracker.track_citation("skill-003", was_correct=True)

        most_used = tracker.get_most_used_skills(limit=2)

        self.assertEqual(len(most_used), 2)
        self.assertEqual(most_used[0], ("skill-001", 2))
        self.assertEqual(most_used[1][0], "skill-002")  # Either skill-002 or skill-003

    def test_usage_tracker_effectiveness(self):
        """Test effectiveness calculation per skill."""
        tracker = SkillUsageTracker()

        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-001", was_correct=False)

        effectiveness = tracker.get_effectiveness_by_skill()

        # 2 correct out of 3 total = 0.666...
        self.assertAlmostEqual(effectiveness["skill-001"], 2 / 3)

    def test_usage_tracker_empty(self):
        """Test tracker with no data."""
        tracker = SkillUsageTracker()

        stats = tracker.get_usage_stats()
        most_used = tracker.get_most_used_skills()
        effectiveness = tracker.get_effectiveness_by_skill()

        self.assertEqual(stats, {})
        self.assertEqual(most_used, [])
        self.assertEqual(effectiveness, {})

    def test_usage_tracker_all_correct(self):
        """Test effectiveness when all usages are correct."""
        tracker = SkillUsageTracker()

        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-001", was_correct=True)
        tracker.track_citation("skill-001", was_correct=True)

        effectiveness = tracker.get_effectiveness_by_skill()

        self.assertEqual(effectiveness["skill-001"], 1.0)

    def test_usage_tracker_all_incorrect(self):
        """Test effectiveness when all usages are incorrect."""
        tracker = SkillUsageTracker()

        tracker.track_citation("skill-001", was_correct=False)
        tracker.track_citation("skill-001", was_correct=False)

        effectiveness = tracker.get_effectiveness_by_skill()

        self.assertEqual(effectiveness["skill-001"], 0.0)


class TestExportAnalytics(unittest.TestCase):
    """Test analytics export functionality."""

    def test_export_analytics_basic(self):
        """Test basic analytics export."""
        skillbook = Skillbook()

        skillbook.add_skill(
            "general", "Be clear", metadata={"helpful": 10, "harmful": 0, "neutral": 2}
        )
        skillbook.add_skill(
            "math", "Show work", metadata={"helpful": 5, "harmful": 1}
        )

        analytics = export_analytics(skillbook)

        # Check summary section
        self.assertIn("summary", analytics)
        self.assertEqual(analytics["summary"]["total_skills"], 2)
        self.assertEqual(analytics["summary"]["total_votes"]["helpful"], 15)
        self.assertEqual(analytics["summary"]["total_votes"]["harmful"], 1)

        # Check by_section
        self.assertIn("by_section", analytics)
        self.assertEqual(analytics["by_section"]["general"], 1)
        self.assertEqual(analytics["by_section"]["math"], 1)

        # Check all_skills
        self.assertIn("all_skills", analytics)
        self.assertEqual(len(analytics["all_skills"]), 2)

    def test_export_analytics_with_usage_tracker(self):
        """Test analytics export with usage tracker."""
        skillbook = Skillbook()

        skill_id = skillbook.add_skill(
            "general", "Be clear", metadata={"helpful": 10}
        ).id

        tracker = SkillUsageTracker()
        tracker.track_citation(skill_id, was_correct=True)
        tracker.track_citation(skill_id, was_correct=True)

        analytics = export_analytics(skillbook, usage_tracker=tracker)

        # Check usage section
        self.assertIn("usage", analytics)
        self.assertIn("most_cited", analytics["usage"])
        self.assertIn("by_skill", analytics["usage"])
        self.assertIn("effectiveness_by_skill", analytics["usage"])

        # Check citation data
        self.assertEqual(analytics["usage"]["most_cited"][0]["skill_id"], skill_id)
        self.assertEqual(analytics["usage"]["most_cited"][0]["citations"], 2)

    def test_export_analytics_top_performing(self):
        """Test top performing skills ranking."""
        skillbook = Skillbook()

        # Add skills with different effectiveness
        skillbook.add_skill(
            "general", "Excellent", metadata={"helpful": 10, "harmful": 0}
        )
        skillbook.add_skill(
            "math", "Good", metadata={"helpful": 5, "harmful": 1}
        )
        skillbook.add_skill(
            "coding", "Bad", metadata={"helpful": 1, "harmful": 10}
        )

        analytics = export_analytics(skillbook)

        # Check top performing
        self.assertIn("top_performing_skills", analytics)
        self.assertEqual(len(analytics["top_performing_skills"]), 3)

        # First should have highest effectiveness
        top_skill = analytics["top_performing_skills"][0]
        self.assertEqual(top_skill["section"], "general")
        self.assertGreater(top_skill["effectiveness_score"], 0.9)

    def test_export_analytics_worst_performing(self):
        """Test worst performing skills ranking."""
        skillbook = Skillbook()

        skillbook.add_skill(
            "general", "Excellent", metadata={"helpful": 10, "harmful": 0}
        )
        skillbook.add_skill(
            "coding", "Terrible", metadata={"helpful": 0, "harmful": 10}
        )

        analytics = export_analytics(skillbook)

        # Check worst performing
        self.assertIn("worst_performing_skills", analytics)

        # Last should have lowest effectiveness
        worst_skill = analytics["worst_performing_skills"][0]
        self.assertEqual(worst_skill["section"], "coding")
        self.assertLess(worst_skill["effectiveness_score"], -0.8)

    def test_export_analytics_json_serializable(self):
        """Test that export output is JSON serializable."""
        import json

        skillbook = Skillbook()

        skillbook.add_skill("general", "Be clear", metadata={"helpful": 10})

        tracker = SkillUsageTracker()
        tracker.track_citation(
            list(skillbook.skills())[0].id, was_correct=True
        )

        analytics = export_analytics(skillbook, usage_tracker=tracker)

        # Should not raise an exception
        json_str = json.dumps(analytics)
        self.assertIsInstance(json_str, str)

        # Should be able to load it back
        loaded = json.loads(json_str)
        self.assertEqual(loaded["summary"]["total_skills"], 1)


if __name__ == "__main__":
    unittest.main()
