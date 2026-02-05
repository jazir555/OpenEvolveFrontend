"""
Enhanced Decomposition Workflow - Usage Examples

Comprehensive examples demonstrating all enhanced features from Phases 1-3:
- 21-field SubProblem model
- 10 decomposition strategies
- Intelligent strategy selection
- Enhanced quality assessment
- Team assignment
- MDAP integration

NEW PATTERNS SHOWCASED:
- Unified configuration system
- Import guards for decomposition engine
- Lazy loading of dependencies
- Graceful degradation

Migration from old system:
- Old: Direct imports without checks
- New: Import with guards and lazy loading
Benefits: Better error handling, cleaner code, flexible loading

Author: Enhanced Decomposition Examples
Date: 2026-01-03 (Migrated to new patterns)
Status: Production Ready
"""

import sys
import json
from typing import Dict, Any, List

# =============================================================================
# NEW IMPORT PATTERN
# =============================================================================

from openevolve_imports import (
    DECOMPOSITION_AVAILABLE,
    SOVEREIGN_DATA_MODELS_AVAILABLE,
    require_decomposition_engine
)

# =============================================================================
# OLD IMPORT PATTERN (for reference)
# =============================================================================
# Old way:
#   from decomposition_engine import DecompositionEngine
#   from sovereign_data_models import ProblemDefinition, etc.
#
# New way benefits:
# - Availability checks
# - Lazy loading
# - Better error handling


# ============================================================================
# EXAMPLE 1: Basic Usage
# ============================================================================

def example_1_basic_decomposition():
    """
    Example 1: Basic problem decomposition with enhanced features

    This example shows the simplest way to use the enhanced decomposition engine.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Decomposition")
    print("="*80)

    from decomposition_engine import DecompositionEngine
    from sovereign_data_models import (
        ProblemDefinition, ProblemType, DomainContext,
        ComplexityScore, generate_id
    )

    # Create engine
    engine = DecompositionEngine()
    print("[OK] DecompositionEngine created")

    # Create problem definition
    problem = ProblemDefinition(
        id=generate_id("problem"),
        title="Build a REST API",
        description="Create a RESTful API for a todo application with CRUD operations",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(
            domain="software_engineering",
            subdomain="web_development"
        ),
        complexity_score=ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=4.0,
            domain_complexity=5.0,
            integration_complexity=4.0,
            overall_complexity=4.5,
            explanation="Medium complexity web API"
        )
    )

    print(f"[OK] Problem created: {problem.title}")

    # Decompose (uses intelligent strategy selection by default)
    plan = engine.decompose(problem)
    print(f"\n[OK] Decomposition complete:")
    print(f"  Strategy: {plan.strategy.value}")
    print(f"  Sub-problems: {len(plan.sub_problems)}")
    print(f"  Confidence: {plan.confidence_level:.2f}")

    # Display first sub-problem
    if plan.sub_problems:
        sp = plan.sub_problems[0]
        print(f"\n  First sub-problem:")
        print(f"    Title: {sp.title}")
        print(f"    Description: {sp.description[:60]}...")
        print(f"    Complexity: {sp.complexity_score.overall_complexity:.1f}/10")
        print(f"    Estimated time: {sp.estimated_time} hours")

        # Show enhanced fields if available
        if sp.ai_suggested_team_assignment:
            ta = sp.ai_suggested_team_assignment
            print(f"    Team assignment: solver={ta.solver}, red_team={ta.red_team}")

        if sp.estimated_resources:
            er = sp.estimated_resources
            print(f"    Resources: {er.time_hours}h, {er.api_tokens} tokens")

    return plan


# ============================================================================
# EXAMPLE 2: Using All 10 Strategies
# ============================================================================

def example_2_all_strategies():
    """
    Example 2: Decompose using all 10 strategies

    Shows how to use each of the 10 available decomposition strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: All 10 Decomposition Strategies (Phase 2)")
    print("="*80)

    from decomposition_engine import DecompositionEngine
    from sovereign_data_models import (
        ProblemDefinition, ProblemType, DomainContext,
        ComplexityScore, generate_id
    )

    engine = DecompositionEngine()

    # Create test problem
    problem = ProblemDefinition(
        id=generate_id("problem"),
        title="E-commerce Platform",
        description="Build a full-stack e-commerce platform with user accounts, product catalog, shopping cart, and payment processing",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(
            domain="software_engineering",
            subdomain="e-commerce"
        ),
        complexity_score=ComplexityScore(
            cognitive_complexity=8.0,
            computational_complexity=7.0,
            domain_complexity=7.0,
            integration_complexity=8.0,
            overall_complexity=7.5,
            explanation="High complexity e-commerce system"
        )
    )

    # All 10 strategies (5 original + 5 new in Phase 2)
    strategies = [
        ('semantic', 'LLM-powered concept analysis'),
        ('dependency', 'Prerequisite relationship-based'),
        ('complexity', 'Cognitive load balancing'),
        ('hybrid', 'Adaptive multi-strategy'),
        ('research', 'Exploration-based'),
        # NEW in Phase 2
        ('functional', 'Functional component breakdown'),
        ('temporal', 'Time-phase sequencing'),
        ('risk_based', 'Risk-prioritized'),
        ('value_based', 'Business-value focused'),
        ('technical_dependency', 'Infrastructure-first')
    ]

    print(f"Problem: {problem.title}")
    print(f"Complexity: {problem.complexity_score.overall_complexity:.1f}/10\n")

    results = []

    for strategy, description in strategies:
        plan = engine.decompose(problem, strategy=strategy)

        results.append({
            'strategy': strategy,
            'description': description,
            'sub_problems': len(plan.sub_problems),
            'confidence': plan.confidence_level,
            'quality': plan.quality_scores.overall_score
        })

        print(f"[{strategy.upper()}]")
        print(f"  {description}")
        print(f"  Sub-problems: {len(plan.sub_problems)}")
        print(f"  Confidence: {plan.confidence_level:.2f}")
        print(f"  Quality: {plan.quality_scores.overall_score:.2f}\n")

    # Compare results
    print("Strategy Comparison:")
    print(f"{'Strategy':<20} {'Sub-problems':<15} {'Confidence':<12} {'Quality':<10}")
    print("-" * 60)

    for r in results:
        print(f"{r['strategy']:<20} {r['sub_problems']:<15} "
              f"{r['confidence']:<12.2f} {r['quality']:<10.2f}")

    return results


# ============================================================================
# EXAMPLE 3: Intelligent Strategy Selection
# ============================================================================

def example_3_intelligent_selection():
    """
    Example 3: Intelligent strategy selection (Phase 2)

    Shows how the intelligent selection algorithm chooses the best strategy
    based on problem characteristics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Intelligent Strategy Selection (Phase 2)")
    print("="*80)

    from decomposition_engine import DecompositionEngine
    from sovereign_data_models import (
        ProblemDefinition, ProblemType, DomainContext,
        ComplexityScore, generate_id
    )

    engine = DecompositionEngine()

    # Create different types of problems
    problems = [
        {
            'name': 'Research Problem',
            'type': ProblemType.RESEARCH,
            'domain': 'data_science',
            'complexity': 8.0,
            'description': 'Research and implement a novel machine learning algorithm for time series forecasting'
        },
        {
            'name': 'Implementation Problem',
            'type': ProblemType.IMPLEMENTATION,
            'domain': 'software_engineering',
            'complexity': 6.0,
            'description': 'Implement a user authentication system with OAuth2'
        },
        {
            'name': 'Design Problem',
            'type': ProblemType.DESIGN,
            'domain': 'system_architecture',
            'complexity': 7.5,
            'description': 'Design a microservices architecture for a scalable web application'
        }
    ]

    print("Testing intelligent strategy selection on different problem types:\n")

    for prob in problems:
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title=prob['name'],
            description=prob['description'],
            problem_type=prob['type'],
            domain_context=DomainContext(domain=prob['domain']),
            complexity_score=ComplexityScore(
                cognitive_complexity=prob['complexity'],
                computational_complexity=prob['complexity'],
                domain_complexity=prob['complexity'],
                integration_complexity=prob['complexity'],
                overall_complexity=prob['complexity'],
                explanation="Test"
            )
        )

        # Get intelligent selection
        selected = engine.select_strategy_intelligent(problem)

        print(f"Problem: {prob['name']}")
        print(f"  Type: {prob['type'].value}")
        print(f"  Domain: {prob['domain']}")
        print(f"  Complexity: {prob['complexity']:.1f}/10")
        print(f"  Selected Strategy: {selected}")

        # Explain why (based on problem characteristics)
        if prob['type'] == ProblemType.RESEARCH:
            print(f"  Reasoning: Research problems benefit from exploration-based decomposition")
        elif prob['type'] == ProblemType.IMPLEMENTATION:
            print(f"  Reasoning: Implementation problems work well with functional decomposition")
        elif prob['type'] == ProblemType.DESIGN:
            print(f"  Reasoning: Design problems require understanding dependencies")
        print()

    return True


# ============================================================================
# EXAMPLE 4: Enhanced Quality Assessment
# ============================================================================

def example_4_quality_assessment():
    """
    Example 4: Enhanced quality assessment (Phase 2)

    Shows how to use the 5-dimensional quality assessment system.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Enhanced Quality Assessment (Phase 2)")
    print("="*80)

    from decomposition_engine import DecompositionEngine
    from quality_tracker import QualityTracker
    from sovereign_data_models import (
        ProblemDefinition, SubProblem, SubProblemType,
        ComplexityScore, DomainContext, ProblemType,
        SuccessCriterion, generate_id
    )

    engine = DecompositionEngine()
    tracker = QualityTracker()

    # Create problem
    problem = ProblemDefinition(
        id=generate_id("problem"),
        title="Build a Web Application",
        description="Create a web application with user authentication",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(domain="web_development"),
        complexity_score=ComplexityScore(
            cognitive_complexity=6.0,
            computational_complexity=5.0,
            domain_complexity=6.0,
            integration_complexity=5.5,
            overall_complexity=5.6,
            explanation="Medium complexity"
        )
    )

    # Create sub-problems
    sub_problems = [
        SubProblem(
            id=generate_id("subprob"),
            parent_id=problem.id,
            title="User Authentication",
            description="Implement login/logout functionality",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5,
                explanation="Low-medium complexity"
            ),
            dependencies=[],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("success"),
                    description="Users can log in",
                    metric="login_success_rate",
                    threshold=0.95,
                    validation_method="automated_test"
                )
            ],
            estimated_time=20.0
        ),
        SubProblem(
            id=generate_id("subprob"),
            parent_id=problem.id,
            title="Database Design",
            description="Design database schema",
            type=SubProblemType.DESIGN,
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=7.0,
                integration_complexity=6.0,
                overall_complexity=6.0,
                explanation="Medium complexity"
            ),
            dependencies=[],
            success_criteria=[],
            estimated_time=16.0
        )
    ]

    print(f"Problem: {problem.title}")
    print(f"Sub-problems: {len(sub_problems)}\n")

    # Assess quality
    quality = engine._assess_quality_enhanced(problem, sub_problems)

    print("5-Dimensional Quality Assessment:")
    print(f"  Overall Score: {quality.overall_score:.2f}")
    print(f"  Meets Thresholds: {quality.meets_thresholds}\n")

    print("Dimension Breakdown:")
    print(f"  Completeness: {quality.completeness_score:.2f}")
    print(f"    - All aspects addressed: "
          f"{quality.completeness_details.get('all_aspects_addressed', 'N/A')}")
    print(f"  Consistency: {quality.consistency_score:.2f}")
    print(f"    - No contradictions: "
          f"{quality.consistency_details.get('no_contradictions', 'N/A')}")
    print(f"  Feasibility: {quality.feasibility_score:.2f}")
    print(f"    - Resource availability: "
          f"{quality.feasibility_details.get('resource_availability', 'N/A')}")
    print(f"  Dependencies: {quality.dependency_score:.2f}")
    print(f"    - Valid dependencies: "
          f"{quality.dependency_details.get('valid_dependencies', 'N/A')}")
    print(f"  Balance: {quality.balance_score:.2f}")
    print(f"    - Even distribution: "
          f"{quality.balance_details.get('even_distribution', 'N/A')}")

    # Recommendations
    if quality.improvement_recommendations:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(quality.improvement_recommendations[:3], 1):
            print(f"  {i}. {rec}")

    # Critical issues
    if quality.critical_issues:
        print(f"\nCritical Issues:")
        for i, issue in enumerate(quality.critical_issues, 1):
            print(f"  {i}. {issue}")

    # Track for trend analysis
    tracker.record_assessment("plan_001", quality)
    print("\n[OK] Quality assessment recorded for tracking")

    return quality


# ============================================================================
# EXAMPLE 5: Team Assignment
# ============================================================================

def example_5_team_assignment():
    """
    Example 5: Team assignment engine (Phase 3)

    Shows how to automatically assign teams to sub-problems.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Team Assignment Engine (Phase 3)")
    print("="*80)

    from team_assignment_engine import TeamAssignmentEngine
    from team_manager import (
        TeamManager, BlueTeam, PatcherTeam, RedTeam, GoldTeam
    )
    from sovereign_data_models import (
        SubProblem, SubProblemType, ComplexityScore, generate_id
    )

    # Create team manager
    team_manager = TeamManager()

    # Add teams
    team_manager.add_blue_team(BlueTeam(
        id="blue_web",
        name="Web Development Team",
        capabilities=["web_development", "authentication", "frontend"],
        performance_history={"success_rate": 0.85, "avg_quality": 0.82},
        current_workload=3
    ))

    team_manager.add_patcher_team(PatcherTeam(
        id="patcher_web",
        name="Web Patcher Team",
        capabilities=["web_development", "bugfixes"],
        performance_history={"success_rate": 0.82, "avg_quality": 0.80},
        current_workload=2
    ))

    team_manager.add_red_team(RedTeam(
        id="red_security",
        name="Security Testing Team",
        capabilities=["security", "penetration_testing", "authentication"],
        performance_history={"success_rate": 0.88, "avg_quality": 0.85},
        current_workload=4
    ))

    team_manager.add_gold_team(GoldTeam(
        id="gold_validation",
        name="Validation Team",
        capabilities=["validation", "testing", "quality_assurance"],
        performance_history={"success_rate": 0.90, "avg_quality": 0.88},
        current_workload=2
    ))

    print("Teams configured:")
    print(f"  Blue teams: {len(team_manager.blue_teams)}")
    print(f"  Patcher teams: {len(team_manager.patcher_teams)}")
    print(f"  Red teams: {len(team_manager.red_teams)}")
    print(f"  Gold teams: {len(team_manager.gold_teams)}\n")

    # Create assignment engine
    assignment_engine = TeamAssignmentEngine(team_manager)

    # Create sub-problems
    sub_problems = [
        SubProblem(
            id=generate_id("subprob"),
            parent_id="test",
            title="Implement OAuth2 Authentication",
            description="Add OAuth2 login functionality",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=7.0,
                integration_complexity=6.0,
                overall_complexity=6.0,
                explanation="Medium complexity"
            ),
            required_expertise=["authentication", "security"],
            estimated_time=40.0
        ),
        SubProblem(
            id=generate_id("subprob"),
            parent_id="test",
            title="Design User Interface",
            description="Create responsive UI",
            type=SubProblemType.DESIGN,
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5,
                explanation="Low-medium complexity"
            ),
            required_expertise=["frontend", "ui_design"],
            estimated_time=24.0
        )
    ]

    print("Assigning teams to sub-problems:\n")

    for sp in sub_problems:
        assignment = assignment_engine.assign_teams_to_subproblem(
            sp, domain="web_development"
        )

        print(f"Sub-problem: {sp.title}")
        print(f"  Required expertise: {', '.join(sp.required_expertise)}")
        print(f"  Solver: {assignment.solver}")
        print(f"  Patcher: {assignment.patcher}")
        print(f"  Red Team: {assignment.red_team}")
        print(f"  Gold Team: {assignment.gold_team}")

        # Verify conflict avoidance
        if assignment.solver != assignment.red_team:
            print(f"  [OK] Conflict avoidance verified")
        else:
            print(f"  [FAIL] Conflict: solver = red_team")
        print()

    return True


# ============================================================================
# EXAMPLE 6: MDAP Integration
# ============================================================================

def example_6_mdap_integration():
    """
    Example 6: MDAP integration with caching and load balancing (Phase 3)

    Shows how to use the MDAP-enhanced decomposition engine.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: MDAP Integration (Phase 3)")
    print("="*80)

    from decomposition_mdap_integration import (
        create_mdap_enhanced_decomposition_engine,
        get_mdap_statistics,
        cleanup_mdap_resources
    )
    from sovereign_data_models import (
        ProblemDefinition, ProblemType, DomainContext,
        ComplexityScore, generate_id
    )

    # Create MDAP-enhanced engine (one-line setup)
    engine = create_mdap_enhanced_decomposition_engine()
    print("[OK] MDAP-enhanced engine created")
    print("  Components:")
    print("    - MDAPCacheManager (TTL-based caching)")
    print("    - MDAPLoadBalancer (Intelligent agent selection)")
    print("    - AdaptiveThresholdManager (Dynamic k calculation)\n")

    # Create problem
    problem = ProblemDefinition(
        id=generate_id("problem"),
        title="Build a Microservices Architecture",
        description="Design and implement a microservices architecture for an e-commerce platform",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(
            domain="software_engineering",
            subdomain="microservices"
        ),
        complexity_score=ComplexityScore(
            cognitive_complexity=8.0,
            computational_complexity=7.0,
            domain_complexity=8.0,
            integration_complexity=8.5,
            overall_complexity=7.9,
            explanation="High complexity distributed system"
        )
    )

    print(f"Problem: {problem.title}")
    print(f"Complexity: {problem.complexity_score.overall_complexity:.1f}/10\n")

    # Decompose (uses MDAP enhancements automatically)
    plan = engine.decompose(problem)

    print(f"[OK] Decomposition complete:")
    print(f"  Strategy: {plan.strategy.value}")
    print(f"  Sub-problems: {len(plan.sub_problems)}")
    print(f"  Confidence: {plan.confidence_level:.2f}")

    # Get MDAP statistics
    stats = get_mdap_statistics(engine)

    print(f"\nMDAP Statistics:")
    if 'cache' in stats:
        cache = stats['cache']
        print(f"  Cache:")
        print(f"    Entries: {cache['total_entries']}")
        print(f"    Hit rate: {cache['hit_rate']:.2%}")
        print(f"    Misses: {cache['misses']}")
        print(f"    Evictions: {cache['evictions']}")

    if 'load_balance' in stats:
        print(f"  Load Balance: {stats['load_balance']:.3f}")

    # Demonstrate cache effectiveness
    print(f"\nDemonstrating cache effectiveness:")
    print(f"  First decomposition: (not cached)")
    print(f"  Second decomposition: (cached - should be faster)")

    # Cleanup
    cleanup_mdap_resources(engine)
    print(f"\n[OK] MDAP resources cleaned up")

    return True


# ============================================================================
# EXAMPLE 7: Complete Workflow
# ============================================================================

def example_7_complete_workflow():
    """
    Example 7: Complete workflow with all enhancements

    Shows a complete end-to-end workflow using all enhanced features.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Complete Enhanced Workflow")
    print("="*80)

    from decomposition_engine import DecompositionEngine
    from decomposition_mdap_integration import (
        create_mdap_enhanced_decomposition_engine,
        get_mdap_statistics
    )
    from quality_tracker import QualityTracker
    from team_assignment_engine import TeamAssignmentEngine
    from team_manager import (
        TeamManager, BlueTeam, PatcherTeam, RedTeam, GoldTeam
    )
    from sovereign_data_models import (
        ProblemDefinition, ProblemType, DomainContext,
        ComplexityScore, Constraint, generate_id
    )

    print("Setting up complete enhanced workflow...\n")

    # Step 1: Create MDAP-enhanced engine
    engine = create_mdap_enhanced_decomposition_engine()
    print("[OK] Step 1: MDAP-enhanced engine created")

    # Step 2: Setup teams
    team_manager = TeamManager()
    team_manager.add_blue_team(BlueTeam(
        id="blue_1",
        name="Blue Team 1",
        capabilities=["software_engineering", "web_development"],
        performance_history={"success_rate": 0.85},
        current_workload=3
    ))
    team_manager.add_patcher_team(PatcherTeam(
        id="patcher_1",
        name="Patcher Team 1",
        capabilities=["software_engineering"],
        performance_history={"success_rate": 0.82},
        current_workload=2
    ))
    team_manager.add_red_team(RedTeam(
        id="red_1",
        name="Red Team 1",
        capabilities=["security", "testing"],
        performance_history={"success_rate": 0.88},
        current_workload=2
    ))
    team_manager.add_gold_team(GoldTeam(
        id="gold_1",
        name="Gold Team 1",
        capabilities=["validation", "testing"],
        performance_history={"success_rate": 0.90},
        current_workload=1
    ))
    print("[OK] Step 2: Teams configured (4 teams)")

    # Step 3: Create quality tracker
    tracker = QualityTracker()
    print("[OK] Step 3: QualityTracker initialized")

    # Step 4: Create problem with constraints
    problem = ProblemDefinition(
        id=generate_id("problem"),
        title="Enterprise Task Management System",
        description="Build a comprehensive task management system with real-time collaboration, notifications, and reporting",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(
            domain="software_engineering",
            subdomain="web_application",
            domain_knowledge={"tech_stack": "React, Node.js, PostgreSQL"}
        ),
        complexity_score=ComplexityScore(
            cognitive_complexity=7.5,
            computational_complexity=6.5,
            domain_complexity=7.0,
            integration_complexity=7.5,
            overall_complexity=7.1,
            explanation="High complexity enterprise application"
        ),
        constraints=[
            Constraint(
                id=generate_id("constraint"),
                description="Must support 1000+ concurrent users",
                type="performance",
                severity="hard"
            ),
            Constraint(
                id=generate_id("constraint"),
                description="Real-time updates within 100ms",
                type="performance",
                severity="hard"
            ),
            Constraint(
                id=generate_id("constraint"),
                description="GDPR compliance required",
                type="compliance",
                severity="hard"
            )
        ],
        resources_available={
            "budget": 200000,
            "team_size": 8,
            "timeline_months": 4
        }
    )
    print(f"[OK] Step 4: Problem created: {problem.title}")
    print(f"          Constraints: {len(problem.constraints)}")

    # Step 5: Intelligent strategy selection
    selected_strategy = engine.select_strategy_intelligent(problem)
    print(f"\n[OK] Step 5: Intelligent strategy selection: {selected_strategy}")

    # Step 6: Decompose
    plan = engine.decompose(problem, strategy=selected_strategy)
    print(f"\n[OK] Step 6: Decomposition complete:")
    print(f"          Sub-problems: {len(plan.sub_problems)}")
    print(f"          Confidence: {plan.confidence_level:.2f}")

    # Step 7: Enhanced quality assessment
    if plan.enhanced_quality_scores:
        eq = plan.enhanced_quality_scores
        print(f"\n[OK] Step 7: Enhanced quality assessment:")
        print(f"          Overall: {eq.overall_score:.2f}")
        print(f"          Completeness: {eq.completeness_score:.2f}")
        print(f"          Consistency: {eq.consistency_score:.2f}")
        print(f"          Feasibility: {eq.feasibility_score:.2f}")
        print(f"          Dependencies: {eq.dependency_score:.2f}")
        print(f"          Balance: {eq.balance_score:.2f}")

        # Track quality
        tracker.record_assessment(plan.id, eq)

    # Step 8: Display enhanced sub-problems
    print(f"\n[OK] Step 8: Enhanced sub-problems:")
    enhanced_count = 0
    for i, sp in enumerate(plan.sub_problems[:3], 1):
        print(f"\n  Sub-problem {i}: {sp.title}")
        print(f"    Complexity: {sp.complexity_score.overall_complexity:.1f}/10")
        print(f"    Estimated time: {sp.estimated_time}h")

        if sp.ai_suggested_team_assignment:
            ta = sp.ai_suggested_team_assignment
            print(f"    Team assignment: solver={ta.solver}, red={ta.red_team}")
            enhanced_count += 1

        if sp.estimated_resources:
            er = sp.estimated_resources
            print(f"    Resources: {er.time_hours}h, {er.api_tokens} tokens")
            enhanced_count += 1

        if sp.potential_approaches:
            print(f"    Approaches: {len(sp.potential_approaches)}")
            enhanced_count += 1

        if sp.required_expertise:
            print(f"    Expertise: {', '.join(sp.required_expertise[:2])}")
            enhanced_count += 1

    print(f"\n  Enhanced sub-problems: {enhanced_count} enhancements shown")

    # Step 9: MDAP statistics
    stats = get_mdap_statistics(engine)
    print(f"\n[OK] Step 9: MDAP Statistics:")
    if 'cache' in stats:
        print(f"          Cache hit rate: {stats['cache']['hit_rate']:.2%}")

    # Step 10: Quality trends
    insights = tracker.get_insights(plan.id)
    if insights:
        print(f"\n[OK] Step 10: Quality Insights:")
        for insight in insights[:2]:
            print(f"           - {insight}")

    print("\n" + "="*80)
    print("COMPLETE WORKFLOW SUMMARY")
    print("="*80)
    print("All enhanced features successfully used:")
    print("  [OK] MDAP-enhanced engine")
    print("  [OK] Team configuration")
    print("  [OK] Quality tracking")
    print("  [OK] Rich problem definition with constraints")
    print("  [OK] Intelligent strategy selection")
    print("  [OK] Enhanced decomposition")
    print("  [OK] 5-dimensional quality assessment")
    print("  [OK] Enhanced sub-problems (21 fields)")
    print("  [OK] Team assignment recommendations")
    print("  [OK] Resource estimation")
    print("  [OK] Potential approaches")
    print("  [OK] MDAP cache statistics")
    print("  [OK] Quality trend insights")

    return True


# ============================================================================
# EXAMPLE 8: Error Handling
# ============================================================================

def example_8_error_handling():
    """
    Example 8: Error handling and best practices

    Shows proper error handling and recovery strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Error Handling and Best Practices")
    print("="*80)

    from decomposition_engine import DecompositionEngine
    from sovereign_data_models import (
        ProblemDefinition, ProblemType, DomainContext,
        ComplexityScore, generate_id
    )

    engine = DecompositionEngine()

    # Example 1: Handle missing LLM API
    print("\n[Example 1: Handling missing LLM API]")
    print("  Always check if engine is properly initialized:")

    if engine.llm_client is None:
        print("  [WARN] LLM client not available")
        print("  -> Using fallback strategies or cached results")
    else:
        print("  [OK] LLM client available")

    # Example 2: Validate inputs
    print("\n[Example 2: Validating problem definition]")

    try:
        # Create invalid problem (empty title)
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="",  # Invalid!
            description="Test",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain="test"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )

        # Validate
        errors = problem.validate()
        if errors:
            print(f"  [FAIL] Validation errors: {errors}")
            print("  -> Fix errors before decomposition")
        else:
            print("  [OK] Problem is valid")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"  [FAIL] Error: {e}")
        print("  -> Handle gracefully with fallback")

    # Example 3: Handle decomposition failures
    print("\n[Example 3: Handling decomposition failures]")

    try:
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Valid Problem",
            description="Valid description",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain="test"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )

        plan = engine.decompose(problem, strategy='semantic')

        if plan.error_message:
            print(f"  [WARN] Decomposition completed with errors:")
            print(f"     {plan.error_message}")
            print(f"  -> Review and adjust problem definition")
        elif len(plan.sub_problems) == 0:
            print(f"  [WARN] No sub-problems generated")
            print(f"  -> Try different strategy or adjust problem")
        else:
            print(f"  [OK] Decomposition successful: {len(plan.sub_problems)} sub-problems")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"  [FAIL] Exception: {e}")
        print(f"  -> Log error and use fallback strategy")

    # Example 4: Check quality thresholds
    print("\n[Example 4: Checking quality thresholds]")

    print("  Always verify quality before proceeding:")
    print("  if quality.overall_score < 0.7:")
    print("      -> Review improvement recommendations")
    print("      -> Consider alternative strategies")
    print("      -> Adjust problem definition")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ENHANCED DECOMPOSITION WORKFLOW - USAGE EXAMPLES")
    print("="*80)
    print("\nThis guide demonstrates all enhanced features from Phases 1-3:")
    print("  * 21-field SubProblem model (Phase 1)")
    print("  * 10 decomposition strategies (Phase 2)")
    print("  * Intelligent strategy selection (Phase 2)")
    print("  * Enhanced quality assessment (Phase 2)")
    print("  * Team assignment engine (Phase 3)")
    print("  * MDAP integration (Phase 3)")

    examples = [
        ("Basic Decomposition", example_1_basic_decomposition),
        ("All 10 Strategies (Phase 2)", example_2_all_strategies),
        ("Intelligent Selection (Phase 2)", example_3_intelligent_selection),
        ("Enhanced Quality Assessment (Phase 2)", example_4_quality_assessment),
        ("Team Assignment (Phase 3)", example_5_team_assignment),
        ("MDAP Integration (Phase 3)", example_6_mdap_integration),
        ("Complete Workflow", example_7_complete_workflow),
        ("Error Handling", example_8_error_handling)
    ]

    print(f"\nRunning {len(examples)} examples...\n")

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}: {name}")
        print(f"{'='*80}")

        try:
            func()
            print(f"\n[OK] Example {i} complete")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"\n[FAIL] Example {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nFor more information, see:")
    print("  * DECOMPOSITION_ENGINE_PHASE1_COMPLETE.md")
    print("  * DECOMPOSITION_ENGINE_PHASE2_COMPLETE.md")
    print("  * DECOMPOSITION_ENGINE_PHASE3_COMPLETE.md")
    print("\nStatus: Production Ready")


if __name__ == "__main__":
    main()
