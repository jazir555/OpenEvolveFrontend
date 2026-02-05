"""
Team Assignment Engine - Demonstration Script

This script demonstrates the key features of the Team Assignment Engine.
Run this to verify the installation and see the system in action.
"""

import sys
import tempfile
import os
from datetime import datetime

# Import required modules
try:
    from team_assignment_engine import (
        TeamAssignmentEngine,
        TeamCapabilityAssessor,
        TeamPerformanceTracker,
        TeamCapability
    )
    from sovereign_data_models import (
        SubProblem, SubProblemTeamAssignment, ProblemDefinition,
        ComplexityScore, SuccessCriterion, DomainContext,
        SubProblemType, ProblemType, DecompositionPlan, DecompositionStrategy
    )
    from openevolve_structures import Team, ModelConfig
    from team_manager import TeamManager
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("\nPlease ensure all required modules are available:")
    print("  - team_assignment_engine.py")
    print("  - sovereign_data_models.py")
    print("  - openevolve_structures.py")
    print("  - team_manager.py")
    sys.exit(1)


def create_sample_teams():
    """Create sample teams for demonstration."""
    teams = []

    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Blue team - Security specialized
    blue_security = Team(
        name="Blue-Security",
        role="Blue",
        members=[ModelConfig(model_id="gpt-4o", api_key=api_key, temperature=0.5)],
        description="Security specialized blue team",
        domain_specialization=["security", "authentication", "cryptography"],
        problem_type_specialization=["implementation", "validation"],
        sub_role="Solver",
        performance_metrics={"accuracy": 0.90, "security_score": 0.95}
    )
    teams.append(blue_security)

    # Blue team - General purpose
    blue_general = Team(
        name="Blue-General",
        role="Blue",
        members=[ModelConfig(model_id="gpt-4o", api_key=api_key, temperature=0.7)],
        description="General purpose blue team",
        domain_specialization=["general", "implementation"],
        problem_type_specialization=["implementation", "design"],
        sub_role="Solver",
        performance_metrics={"accuracy": 0.85, "speed": 0.75}
    )
    teams.append(blue_general)

    # Red team
    red_team = Team(
        name="Red-Critique",
        role="Red",
        members=[ModelConfig(model_id="claude-3-opus", api_key=api_key, temperature=0.8)],
        description="Critique specialized red team",
        domain_specialization=["security", "testing"],
        problem_type_specialization=["validation", "analysis"],
        performance_metrics={"catch_rate": 0.85}
    )
    teams.append(red_team)

    # Gold team
    gold_team = Team(
        name="Gold-Verification",
        role="Gold",
        members=[ModelConfig(model_id="gpt-4o", api_key=api_key, temperature=0.3)],
        description="Verification specialized gold team",
        domain_specialization=["formal_verification", "testing"],
        problem_type_specialization=["validation"],
        performance_metrics={"verification_accuracy": 0.92}
    )
    teams.append(gold_team)

    return teams


def create_sample_sub_problem():
    """Create a sample sub-problem for demonstration."""
    complexity = ComplexityScore(
        explanation="Moderately complex security implementation",
        cognitive_complexity=6.0,
        computational_complexity=5.0,
        domain_complexity=7.0,
        integration_complexity=5.0,
        overall_complexity=6.0
    )

    sub_problem = SubProblem(
        id="sub_001",
        parent_id="prob_001",
        title="Implement Authentication System",
        description="Implement a secure JWT-based authentication system with role-based access control",
        type=SubProblemType.IMPLEMENTATION,
        complexity_score=complexity,
        required_expertise=["security", "authentication", "JWT", "backend"],
        ai_suggested_evolution_mode="standard"
    )

    return sub_problem


def demo_capability_assessment(team_manager, sub_problem):
    """Demonstrate team capability assessment."""
    print("\n" + "="*60)
    print("DEMO 1: Team Capability Assessment")
    print("="*60)

    assessor = TeamCapabilityAssessor(team_manager)
    teams = team_manager.get_all_teams()

    print(f"\nSub-problem: {sub_problem.title}")
    print(f"Required expertise: {', '.join(sub_problem.required_expertise)}\n")

    for team in teams:
        capability = assessor.assess_team_capability(team, sub_problem)
        overall = capability.calculate_overall_capability()

        print(f"Team: {team.name}")
        print(f"  Role: {team.role}")
        print(f"  Overall Capability: {overall:.2%}")
        print(f"  Capability Score: {capability.capability_score:.2%}")
        print(f"  Success Rate: {capability.success_rate:.2%}")
        print(f"  Workload Score: {capability.workload_score:.2%}")
        print(f"  Specialization Fit: {capability.specialization_fit:.2%}")
        print(f"  Confidence: {capability.confidence_score:.2%}")
        if capability.expertise_areas:
            print(f"  Matched Expertise: {', '.join(capability.expertise_areas)}")
        print()


def demo_team_assignment(team_manager, sub_problem):
    """Demonstrate team assignment."""
    print("\n" + "="*60)
    print("DEMO 2: Team Assignment to Sub-Problem")
    print("="*60)

    engine = TeamAssignmentEngine(team_manager)
    teams = team_manager.get_all_teams()

    assignment = engine.assign_teams_to_subproblem(sub_problem, teams)

    print(f"\nSub-problem: {sub_problem.title}")
    print(f"\nTeam Assignments:")
    print(f"  Solver (Blue):   {assignment.solver}")
    print(f"  Patcher (Blue):  {assignment.patcher}")
    print(f"  Red Team:        {assignment.red_team}")
    print(f"  Gold Team:       {assignment.gold_team}")

    if assignment.metadata:
        print(f"\nMetadata:")
        print(f"  Candidates Evaluated: {assignment.metadata.get('num_candidates', 0)}")
        print(f"  Solver Confidence: {assignment.metadata.get('solver_confidence', 0):.2%}")


def demo_performance_tracking():
    """Demonstrate performance tracking."""
    print("\n" + "="*60)
    print("DEMO 3: Performance Tracking")
    print("="*60)

    # Create temporary tracker
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)

    try:
        tracker = TeamPerformanceTracker(storage_path=temp_path)

        # Record some assignments and outcomes
        assignment = SubProblemTeamAssignment(
            solver="Blue-Security",
            patcher="Blue-Security",
            red_team="Red-Critique",
            gold_team="Gold-Verification"
        )

        print("\nRecording assignments and outcomes...")
        for i in range(5):
            tracker.record_assignment(
                "Blue-Security",
                f"sub_{i:03d}",
                "solver",
                assignment
            )

            success = i < 4  # 4 out of 5 successful
            tracker.record_outcome(
                "Blue-Security",
                f"sub_{i:03d}",
                success=success,
                quality_score=0.8 + (i * 0.02),
                time_taken=100.0 + (i * 10.0)
            )

        print("[OK] Recorded 5 assignments and outcomes")

        # Get stats
        stats = tracker.get_team_performance_stats("Blue-Security")

        print(f"\nTeam Performance Statistics:")
        print(f"  Total Assignments: {stats['total_assignments']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Average Quality: {stats['average_quality_score']:.2f}")
        print(f"  Average Time: {stats['average_time_taken']:.1f}s")
        print(f"  Recent Trend: {stats['recent_performance_trend']:.2%}")

        # Get rankings
        rankings = tracker.get_team_ranking()
        print(f"\nTeam Rankings:")
        for i, (team_id, score) in enumerate(rankings[:5], 1):
            print(f"  {i}. {team_id}: {score:.2f}")

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in demo_team_assignment.py: {e}", exc_info=True)
            raise


def demo_plan_assignment(team_manager):
    """Demonstrate team assignment to a full plan."""
    print("\n" + "="*60)
    print("DEMO 4: Team Assignment to Decomposition Plan")
    print("="*60)

    engine = TeamAssignmentEngine(team_manager)
    teams = team_manager.get_all_teams()

    # Create sample plan with multiple sub-problems
    complexity = ComplexityScore(
        explanation="Test complexity",
        cognitive_complexity=5.0,
        computational_complexity=5.0,
        domain_complexity=5.0,
        integration_complexity=5.0,
        overall_complexity=5.0
    )

    sub_problems = [
        SubProblem(
            id=f"sub_{i:03d}",
            parent_id="prob_001",
            title=f"Sub-problem {i}",
            description=f"Description {i}",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=complexity,
            required_expertise=["security"] if i % 2 == 0 else ["general"]
        )
        for i in range(5)
    ]

    plan = DecompositionPlan(
        id="plan_001",
        problem_id="prob_001",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=sub_problems
    )

    print(f"\nOriginal plan: {len(plan.sub_problems)} sub-problems")

    # Assign teams
    updated_plan = engine.assign_teams_to_plan(plan, teams)

    print(f"\nTeam Assignments:")
    for sp in updated_plan.sub_problems:
        assignment = sp.ai_suggested_team_assignment
        print(f"\n  {sp.title}:")
        print(f"    Solver: {assignment.solver}")
        print(f"    Red Team: {assignment.red_team}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("Team Assignment Engine - Demonstration")
    print("="*60)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create temporary team manager
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)

    try:
        team_manager = TeamManager(teams_file=temp_path)

        # Create and add sample teams
        teams = create_sample_teams()
        for team in teams:
            team_manager.create_team(team)

        print(f"[OK] Created {len(teams)} sample teams")

        # Create sample sub-problem
        sub_problem = create_sample_sub_problem()
        print(f"[OK] Created sample sub-problem: {sub_problem.title}")

        # Run demonstrations
        demo_capability_assessment(team_manager, sub_problem)
        demo_team_assignment(team_manager, sub_problem)
        demo_performance_tracking()
        demo_plan_assignment(team_manager)

        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        print("\nThe Team Assignment Engine is ready to use.")
        print("\nNext steps:")
        print("  1. Review the documentation in TEAM_ASSIGNMENT_COMPLETE.md")
        print("  2. Run tests: pytest test_team_assignment.py -v")
        print("  3. Integrate into your workflow")
        print()

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"\n[FAIL] Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in demo_team_assignment.py: {e}", exc_info=True)
            raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
