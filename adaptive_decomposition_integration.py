"""
Example integration of adaptive strategy selection with DecompositionEngine.

This module demonstrates how to add adaptive learning to the decompose method
with a feedback loop for recording outcomes.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from decomposition_engine_adaptive_enhancement import (
    select_decomposition_strategy_v3,
    record_decomposition_outcome,
    setup_adaptive_selection
)

# Import ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE as ROMA_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_AVAILABLE = False
    get_validation_config = None

logger = logging.getLogger(__name__)


def decompose_with_adaptive_selection(
    engine,
    problem,
    strategy: Optional[str] = None,
    assign_teams: bool = False,
    teams: Optional[list] = None,
    use_adaptive_selection: bool = True,
    record_outcome_for_learning: bool = True,
    use_roma_decomposition: bool = False,
    roma_config: Optional[Any] = None
):
    """
    Enhanced decompose method with adaptive strategy selection and feedback loop.

    This is a wrapper function that adds adaptive capabilities to the existing
    DecompositionEngine.decompose() method without modifying the original code.

    Args:
        engine: DecompositionEngine instance (will be enhanced if needed)
        problem: ProblemDefinition to decompose
        strategy: Optional strategy name (auto-selected if not provided)
        assign_teams: Whether to assign teams
        teams: Optional list of teams
        use_adaptive_selection: Whether to use adaptive strategy selection
        record_outcome_for_learning: Whether to record outcome for learning
        use_roma_decomposition: Whether to use ROMA hierarchical decomposition
        roma_config: Optional configuration for ROMA

    Returns:
        DecompositionPlan with enhanced metadata including:
            - strategy_selection: Details of how strategy was chosen
            - adaptive_adjustments: Performance-based weight adjustments (if adaptive)
            - learning_metadata: Feedback loop information
    """
    start_time = datetime.now()

    # Phase 0: ROMA Decomposition (High Reliability)
    if use_roma_decomposition and ROMA_AVAILABLE:
        logger.info("Using ROMA hierarchical decomposition for high reliability")
        try:
            # Use SSOT for standardized high-reliability config
            config = roma_config or get_validation_config(roma_max_depth_analysis=2)
            roma_engine = ROMAMDAPMakerAssociativeEngine(config)
            
            # ROMA works on text problems primarily
            problem_text = problem.description if hasattr(problem, 'description') else str(problem)
            # Use solve_with_roma_mdap_maker which is available on the associative engine
            roma_result = roma_engine.solve_with_roma_mdap_maker(problem_text)
            
            # Map ROMA result back to DecompositionPlan
            # (In a real implementation, we'd convert roma_hierarchy to plan.sub_problems)
            logger.info("ROMA decomposition successful")
            
            # For now, we'll continue with standard decomposition but add ROMA as 'advisor'
            # or use it to force a specific strategy.
            if not strategy:
                strategy = "roma_informed"
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"ROMA decomposition failed, falling back: {e}")

    # Setup adaptive selection if not already done
    if not hasattr(engine, 'performance_tracker'):
        engine = setup_adaptive_selection(engine, use_adaptive_selection=use_adaptive_selection)

    # Select strategy (adaptively if enabled)
    if not strategy:
        if use_adaptive_selection and hasattr(engine, 'select_strategy_adaptive'):
            strategy, selection_metadata = engine.select_strategy_adaptive(problem)
            logger.info(f"Adaptive selection chose: {strategy}")
        else:
            # Fall back to engine's default selection
            strategy = engine.select_strategy(problem)
            selection_metadata = {
                'version': 'v2_intelligent',
                'selected_strategy': strategy,
                'selection_reason': 'Used engine default selection'
            }
    else:
        selection_metadata = {
            'version': 'manual',
            'selected_strategy': strategy,
            'selection_reason': 'Strategy manually specified'
        }

    # Perform decomposition
    plan = engine.decompose(
        problem=problem,
        strategy=strategy,
        assign_teams=assign_teams,
        teams=teams
    )

    # Calculate completion time
    completion_time = (datetime.now() - start_time).total_seconds()

    # Assess quality (enhanced if available)
    quality_score = _assess_decomposition_quality(plan)

    # Record outcome for learning
    if record_outcome_for_learning and use_adaptive_selection:
        try:
            if hasattr(engine, 'record_outcome'):
                engine.record_outcome(
                    strategy=strategy,
                    problem=problem,
                    quality_score=quality_score,
                    time_to_complete=completion_time
                )
                logger.info(f"Recorded outcome for learning: quality={quality_score:.2f}")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"Failed to record outcome for learning: {e}", exc_info=True)

    # Enhance plan metadata
    if not plan.metadata:
        plan.metadata = {}

    plan.metadata['adaptive_selection'] = {
        'enabled': use_adaptive_selection,
        'selection_metadata': selection_metadata,
        'quality_score': quality_score,
        'completion_time_seconds': completion_time,
        'timestamp': datetime.now().isoformat()
    }

    if record_outcome_for_learning:
        plan.metadata['adaptive_selection']['recorded_for_learning'] = True

    return plan


def _assess_decomposition_quality(plan) -> float:
    """
    Assess the quality of a decomposition plan.

    Args:
        plan: DecompositionPlan to assess

    Returns:
        Quality score from 0.0 to 1.0
    """
    scores = []

    # 1. Number of sub-problems (optimal: 3-7)
    num_sub_problems = len(plan.sub_problems)
    if 3 <= num_sub_problems <= 7:
        scores.append(1.0)
    elif 2 <= num_sub_problems <= 8:
        scores.append(0.8)
    elif num_sub_problems >= 1:
        scores.append(0.5)
    else:
        scores.append(0.0)

    # 2. Quality scores from plan
    if plan.quality_scores:
        overall_quality = getattr(plan.quality_scores, 'overall_quality', None)
        if overall_quality is not None:
            scores.append(overall_quality)

    # 3. Confidence level
    if plan.confidence_level:
        scores.append(plan.confidence_level)

    # 4. Sub-problem completeness
    complete_sub_problems = sum(
        1 for sp in plan.sub_problems
        if sp.title and sp.description and sp.type
    )
    if num_sub_problems > 0:
        completeness = complete_sub_problems / num_sub_problems
        scores.append(completeness)

    # 5. Complexity balance
    if plan.sub_problems:
        complexities = [
            sp.complexity_score.overall_complexity
            for sp in plan.sub_problems
            if sp.complexity_score
        ]
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            # Optimal average complexity is 4-6
            if 4 <= avg_complexity <= 6:
                scores.append(1.0)
            elif 3 <= avg_complexity <= 7:
                scores.append(0.8)
            else:
                scores.append(0.6)

    # Calculate overall quality
    if scores:
        overall_quality = sum(scores) / len(scores)
    else:
        overall_quality = 0.5  # Neutral if no metrics available

    return max(0.0, min(1.0, overall_quality))


# Example usage demonstration
def example_adaptive_decomposition():
    """
    Demonstrate how to use adaptive decomposition with feedback loop.
    """
    from decomposition_engine import DecompositionEngine
    from problem_analyzer import ProblemDefinition, DomainContext, ProblemType, ComplexityScore

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create engine with adaptive selection
    engine = DecompositionEngine()

    # Example problem
    problem = ProblemDefinition(
        id="test-problem-1",
        title="Build a web-based e-commerce platform",
        description="Create a full-featured e-commerce platform with user authentication, product catalog, shopping cart, and payment processing.",
        domain_context=DomainContext(domain="software_engineering", subdomain="web_development"),
        problem_type=ProblemType.SYSTEM_DESIGN,
        complexity_score=ComplexityScore(
            overall_complexity=7.0,
            cognitive_complexity=7.0,
            computational_complexity=6.0,
            domain_complexity=7.0,
            integration_complexity=8.0
        )
    )

    # Decompose with adaptive selection
    print("Decomposing with adaptive selection...")
    plan = decompose_with_adaptive_selection(
        engine=engine,
        problem=problem,
        use_adaptive_selection=True,
        record_outcome_for_learning=True
    )

    print(f"\nDecomposition complete!")
    print(f"Strategy: {plan.strategy}")
    print(f"Sub-problems: {len(plan.sub_problems)}")
    print(f"Quality: {plan.metadata['adaptive_selection']['quality_score']:.2f}")

    if 'selection_metadata' in plan.metadata['adaptive_selection']:
        sel_meta = plan.metadata['adaptive_selection']['selection_metadata']
        print(f"\nSelection Details:")
        print(f"  Version: {sel_meta['version']}")
        print(f"  Reason: {sel_meta['selection_reason']}")

        if 'base_weights' in sel_meta:
            print(f"\n  Base Weights:")
            for strat, weight in sorted(sel_meta['base_weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"    {strat}: {weight:.3f}")

        if 'final_weights' in sel_meta:
            print(f"\n  Final Weights:")
            for strat, weight in sorted(sel_meta['final_weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"    {strat}: {weight:.3f}")

    # Check learning progress
    if hasattr(engine, 'get_learning_progress'):
        progress = engine.get_learning_progress()
        print(f"\nLearning Progress:")
        print(f"  Stage: {progress['learning_stage']}")
        print(f"  Description: {progress['stage_description']}")
        print(f"  Confidence: {progress['average_confidence']:.2f}")
        print(f"  Total Decompositions: {progress['total_decompositions']}")

    # Export performance report
    if hasattr(engine, 'export_performance_report'):
        engine.export_performance_report("adaptive_performance_report.json")
        print("\nPerformance report exported to adaptive_performance_report.json")


def simulate_learning_iterations(num_iterations: int = 10):
    """
    Simulate multiple decompositions to demonstrate learning.

    This shows how the system adapts over time based on feedback.
    """
    from decomposition_engine import DecompositionEngine
    from problem_analyzer import ProblemDefinition, DomainContext, ProblemType, ComplexityScore
    import random

    logging.basicConfig(level=logging.WARNING)  # Reduce log noise

    print(f"Simulating {num_iterations} decompositions with learning...\n")

    # Create engine
    engine = DecompositionEngine()
    setup_adaptive_selection(engine, use_adaptive_selection=True)

    # Define problem templates
    problems = [
        {
            'title': 'Build machine learning pipeline',
            'domain': 'data_science',
            'type': ProblemType.ALGORITHM_DESIGN,
            'complexity': 8.0
        },
        {
            'title': 'Design microservices architecture',
            'domain': 'software_engineering',
            'type': ProblemType.SYSTEM_ARCHITECTURE,
            'complexity': 7.0
        },
        {
            'title': 'Optimize database queries',
            'domain': 'database_management',
            'type': ProblemType.OPTIMIZATION,
            'complexity': 6.0
        }
    ]

    for i in range(num_iterations):
        # Select random problem
        problem_template = random.choice(problems)

        problem = ProblemDefinition(
            id=f"sim-problem-{i}",
            title=problem_template['title'],
            description=f"Simulated problem {i} for learning demonstration",
            domain_context=DomainContext(domain=problem_template['domain']),
            problem_type=problem_template['type'],
            complexity_score=ComplexityScore(overall_complexity=problem_template['complexity'])
        )

        # Decompose
        plan = decompose_with_adaptive_selection(
            engine=engine,
            problem=problem,
            use_adaptive_selection=True,
            record_outcome_for_learning=True
        )

        # Simulate varying quality (realistic learning scenario)
        quality = random.uniform(0.6, 0.95)

        print(f"Iteration {i+1}: Strategy={plan.strategy}, Quality={quality:.2f}, "
              f"Sub-problems={len(plan.sub_problems)}")

    # Show final learning state
    progress = engine.get_learning_progress()
    print(f"\nFinal Learning State:")
    print(f"  Stage: {progress['learning_stage']}")
    print(f"  Total Decompositions: {progress['total_decompositions']}")
    print(f"  Average Confidence: {progress['average_confidence']:.2f}")

    # Show strategy rankings
    summary = engine.get_performance_summary()
    print(f"\nStrategy Performance:")
    for strategy, data in summary['strategies'].items():
        if data['usage_count'] > 0:
            print(f"  {strategy}:")
            print(f"    Usage: {data['usage_count']}")
            print(f"    Quality: {data['avg_quality']:.2f}")
            print(f"    Success Rate: {data['success_rate']:.0%}")
            print(f"    Trend: {data['trend']}")


if __name__ == "__main__":
    # Run example
    print("=" * 80)
    print("Adaptive Strategy Selection Example")
    print("=" * 80)

    # Single decomposition example
    example_adaptive_decomposition()

    print("\n" + "=" * 80)
    print("Learning Simulation Example")
    print("=" * 80)

    # Simulate learning iterations
    simulate_learning_iterations(num_iterations=15)
