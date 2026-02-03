"""Demonstration of ACE Analytics functionality.

This script shows how to use the analytics module to track and analyze
skillbook performance.
"""

from ace import Skillbook
from ace.analytics import (
    SkillUsageTracker,
    calculate_effectiveness_score,
    export_analytics,
    get_skillbook_stats,
    SkillbookStats,
)
import json


def main():
    """Run analytics demonstration."""

    print("=" * 60)
    print("ACE Analytics Demonstration")
    print("=" * 60)

    # 1. Create a skillbook with sample data
    print("\n1. Creating skillbook with sample skills...")
    skillbook = Skillbook()

    # Add high-performing skills
    skillbook.add_skill(
        "general", "Be clear and concise", metadata={"helpful": 10, "harmful": 0}
    )
    skillbook.add_skill(
        "math", "Show your work step-by-step", metadata={"helpful": 8, "harmful": 1}
    )
    skillbook.add_skill(
        "coding", "Use descriptive variable names", metadata={"helpful": 7, "harmful": 0}
    )

    # Add problematic skill
    skillbook.add_skill(
        "writing", "Use complex vocabulary", metadata={"helpful": 2, "harmful": 5}
    )

    # Add unused skills
    skillbook.add_skill("general", "New skill without feedback")
    skillbook.add_skill("math", "Another new skill")

    print(f"   Created skillbook with {len(skillbook.skills())} skills")

    # 2. Generate statistics
    print("\n2. Generating skillbook statistics...")
    stats: SkillbookStats = get_skillbook_stats(skillbook)

    print(f"   Total Skills: {stats.total_skills}")
    print(f"   High Performing: {stats.high_performing}")
    print(f"   Problematic: {stats.problematic}")
    print(f"   Unused: {stats.unused}")
    print(f"   Average Helpful Score: {stats.average_helpful:.2f}")
    print(f"   Average Harmful Score: {stats.average_harmful:.2f}")
    print(f"   Skills by Section:")
    for section, count in stats.by_section.items():
        print(f"     - {section}: {count}")

    # 3. Calculate effectiveness scores
    print("\n3. Calculating effectiveness scores...")
    for skill in skillbook.skills():
        score = calculate_effectiveness_score(skill)
        print(f"   [{skill.id}] {skill.content[:50]}...")
        print(f"       Helpful: {skill.helpful}, Harmful: {skill.harmful}")
        print(f"       Effectiveness: {score:.3f} (range: -1.0 to 1.0)")

    # 4. Track usage
    print("\n4. Demonstrating usage tracking...")
    tracker = SkillUsageTracker()

    # Simulate agent citations
    general_skills = [s for s in skillbook.skills() if s.section == "general"]
    math_skills = [s for s in skillbook.skills() if s.section == "math"]

    general_skill_id = general_skills[0].id
    math_skill_id = math_skills[0].id

    # Track some citations
    tracker.track_citation(general_skill_id, was_correct=True)
    tracker.track_citation(general_skill_id, was_correct=True)
    tracker.track_citation(general_skill_id, was_correct=False)
    tracker.track_citation(math_skill_id, was_correct=True)
    tracker.track_citation(math_skill_id, was_correct=True)

    print("   Most cited skills:")
    for skill_id, count in tracker.get_most_used_skills(limit=3):
        skill = skillbook.get_skill(skill_id)
        print(f"     - [{skill.id}] {skill.content[:40]}... ({count} citations)")

    print("\n   Usage effectiveness:")
    effectiveness = tracker.get_effectiveness_by_skill()
    for skill_id, score in effectiveness.items():
        skill = skillbook.get_skill(skill_id)
        print(f"     - [{skill.id}] {score:.1%} correct")

    # 5. Export analytics
    print("\n5. Exporting complete analytics...")
    analytics_data = export_analytics(skillbook, usage_tracker=tracker)

    # Save to file
    output_file = "analytics_export.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analytics_data, f, indent=2)

    print(f"   Analytics exported to {output_file}")

    # Show summary
    print("\n   Export Summary:")
    summary = analytics_data["summary"]
    print(f"     - Total Skills: {summary['total_skills']}")
    print(f"     - High Performing: {summary['high_performing']}")
    print(f"     - Problematic: {summary['problematic']}")
    print(f"     - Total Votes: {summary['total_votes']}")

    print("\n   Top Performing Skills:")
    for i, skill_data in enumerate(analytics_data["top_performing_skills"][:3], 1):
        print(f"     {i}. [{skill_data['id']}] - Score: {skill_data['effectiveness_score']:.3f}")

    print("\n" + "=" * 60)
    print("Analytics demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
