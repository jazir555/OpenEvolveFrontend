#!/usr/bin/env python3
"""
Complete Example: All Adaptive MDAP/MAKER Adapter Features

This script demonstrates the complete capabilities of the enhanced adapter,
including all advanced features and integrations.

Run with:
    python example_complete_features.py
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set environment variables
os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"] = "5000"
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_1_basic_complexity_analysis():
    """Example 1: Basic Complexity Analysis."""
    print_section("EXAMPLE 1: Basic Complexity Analysis")

    from src import get_adapter, CanonicalSubProblem, TaskStatus

    adapter = get_adapter()

    # Analyze complexity
    subproblem = CanonicalSubProblem(
        id="example_1",
        description="Implement distributed caching system with cache invalidation",
        domain="distributed_systems",
        depth=2
    )

    response = adapter.analyze_complexity(subproblem)

    print(f"[OK] Complexity Analysis Complete")

    # Handle graceful degradation when core projects unavailable
    if response.complexity_score:
        print(f"  Overall Score: {response.complexity_score.overall_score:.3f}")
        print(f"  Text Length: {response.complexity_score.text_length_score:.3f}")
        print(f"  Dependencies: {response.complexity_score.dependency_score:.3f}")
        print(f"  Depth: {response.complexity_score.depth_score:.3f}")
    else:
        print(f"  Status: {response.status.value}")
        print(f"  Note: Core projects not available - using graceful degradation")

    if response.execution_time_ms:
        print(f"  Execution Time: {response.execution_time_ms:.0f}ms")


def example_2_advanced_decomposition():
    """Example 2: Advanced Decomposition."""
    print_section("EXAMPLE 2: Advanced Problem Decomposition")

    from src import get_advanced_openevolve_integration

    advanced = get_advanced_openevolve_integration()

    # Decompose problem
    decomposition = advanced.decompose_problem(
        workflow_id="example_2",
        problem_statement="Design and implement microservices architecture with service mesh",
        workflow_type="sovereign",
        max_depth=3
    )

    print(f"[OK] Advanced Decomposition Complete")
    print(f"  Sub-Problems: {len(decomposition.sub_problems)}")
    print(f"  Strategy: {decomposition.decomposition_strategy}")
    print(f"  Parallelization: {decomposition.recommended_parallelization}")
    print(f"\n  Sub-Problem Breakdown:")

    for i, sub in enumerate(decomposition.sub_problems[:3], 1):
        print(f"    {i}. {sub['description'][:60]}...")
        print(f"       Complexity: {sub['complexity']:.3f}")

    # Team selection
    team_selection = advanced.select_teams_for_stage(
        workflow_id="example_2",
        stage="solving",
        workflow_type="sovereign",
        complexity_score=0.7
    )

    print(f"\n  Team Selection:")
    print(f"    Recommended Teams: {list(team_selection.recommended_teams.keys())}")
    print(f"    Estimated Cost: ${team_selection.estimated_cost:.2f}")

    # Resource optimization
    optimization = advanced.optimize_resources(
        workflow_id="example_2",
        stage="solving",
        complexity_score=0.7,
        estimated_duration_ms=60000
    )

    print(f"\n  Resource Optimization:")
    print(f"    CPU: {optimization.cpu_allocation} cores")
    print(f"    Memory: {optimization.memory_allocation_mb}MB")
    print(f"    Timeout: {optimization.timeout_ms}ms")
    print(f"    Cost Savings: {optimization.estimated_cost_savings:.1%}")


def example_3_gauntlet_pipeline():
    """Example 3: Multi-Gauntlet Pipeline."""
    print_section("EXAMPLE 3: Multi-Gauntlet Pipeline Verification")

    from src import get_advanced_gauntlet_integration, GauntletType

    gauntlet = get_advanced_gauntlet_integration()

    # Create pipeline
    pipeline = gauntlet.create_gauntlet_pipeline(
        complexity_score=0.75,
        base_gauntlet_type=GauntletType.ADVERSARIAL,
        include_cross_validation=True
    )

    print(f"[OK] Gauntlet Pipeline Created")
    print(f"  Total Gauntlets: {len(pipeline.gauntlets)}")
    print(f"  Execution Mode: {pipeline.execution_mode}")
    print(f"  Aggregation: {pipeline.aggregation_strategy}")
    print(f"\n  Gauntlet Types:")

    for i, g in enumerate(pipeline.gauntlets, 1):
        print(f"    {i}. {g.gauntlet_type.value} ({g.severity.value})")

    # Execute pipeline
    result = gauntlet.execute_pipeline(
        pipeline=pipeline,
        solution="example_solution",
        context={"test": True}
    )

    print(f"\n[OK] Pipeline Execution Complete")
    print(f"  Total Gauntlets: {result.total_gauntlets}")
    print(f"  Passed: {result.passed_gauntlets}")
    print(f"  Failed: {result.failed_gauntlets}")
    print(f"  Overall Pass: {result.overall_pass}")
    print(f"  Aggregate Score: {result.aggregate_score:.3f}")
    print(f"  Execution Time: {result.execution_time_ms:.0f}ms")


def example_4_icr_learning():
    """Example 4: ICR Pattern Learning."""
    print_section("EXAMPLE 4: ICR Pattern Learning")

    from src import get_advanced_icr_integration, ICRPatternType

    icr = get_advanced_icr_integration()

    # Store patterns
    print("Storing patterns for learning...")

    for i in range(5):
        pattern_id = icr.store_pattern_advanced(
            pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
            passed=(i % 2 == 0),  # Alternate pass/fail
            context={"iteration": i, "domain": "test"},
            metrics={"complexity": 0.5 + i * 0.1}
        )
        print(f"  [OK] Pattern {i+1} stored: {pattern_id}")

    # Get insights
    insights = icr.get_pattern_insights()

    print(f"\n[OK] ICR Insights Generated")
    print(f"  Available: {insights.get('available', False)}")
    print(f"  Pattern Types Tracked: {len(insights.get('pattern_types', {}))}")

    if insights.get('available'):
        print(f"\n  Pattern Statistics:")
        for ptype, stats in insights.get('pattern_types', {}).items():
            print(f"    {ptype}:")
            print(f"      Count: {stats.get('count', 0)}")
            print(f"      Pass Rate: {stats.get('pass_rate', 0):.1%}")
            print(f"      Confidence: {stats.get('confidence', 0):.1%}")

    # Adaptive threshold
    threshold = icr.get_adaptive_threshold(ICRPatternType.WORKFLOW_EXECUTION)
    print(f"\n  Adaptive Threshold: {threshold:.3f}")


def example_5_performance_optimization():
    """Example 5: Performance Optimization."""
    print_section("EXAMPLE 5: Performance Optimization (Async & Cached)")

    from src import get_async_adapter, get_performance_monitor
    from src import CanonicalSubProblem

    async_adapter = get_async_adapter()
    monitor = get_performance_monitor()

    # Create multiple sub-problems
    subproblems = [
        CanonicalSubProblem(
            id=f"async_{i}",
            description=f"Async analysis problem {i}",
            domain="test",
            depth=1
        )
        for i in range(5)
    ]

    # Batch analyze concurrently
    print(f"Running {len(subproblems)} concurrent analyses...")

    async def run_async_analysis():
        """Run async analysis with proper timing."""
        import time
        start = time.time()

        results = await async_adapter.batch_analyze_complexity(
            subproblems,
            max_concurrency=3
        )

        duration = (time.time() - start) * 1000
        return results, duration

    # Run async operations
    try:
        import time
        start = time.time()

        results, duration = asyncio.run(run_async_analysis())

        print(f"\n[OK] Concurrent Analysis Complete")
        print(f"  Total Operations: {len(results)}")
        print(f"  Total Time: {duration:.0f}ms")
        print(f"  Average Time per Operation: {duration / len(results):.0f}ms")

        # Performance stats
        print(f"\n[OK] Performance Metrics:")
        cache_stats = async_adapter.get_cache_stats()
        print(f"  Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
        print(f"  Cache Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 1000)}")
    except RuntimeError as e:
        # Handle event loop issues gracefully
        print(f"\n[WARN] Async execution not available: {e}")
        print("[INFO] Running synchronous fallback...")

        # Fallback to synchronous analysis
        start = time.time()
        results = []
        for sp in subproblems:
            result = async_adapter.adapter.analyze_complexity(sp)
            results.append(result)

        duration = (time.time() - start) * 1000
        print(f"\n[OK] Synchronous Analysis Complete (fallback)")
        print(f"  Total Operations: {len(results)}")
        print(f"  Total Time: {duration:.0f}ms")
        print(f"  Average Time per Operation: {duration / len(results):.0f}ms")


def example_6_ui_dashboard():
    """Example 6: UI Dashboard Generation."""
    print_section("EXAMPLE 6: UI Dashboard Generation")

    from src import get_bubblelab_ui_integration, get_advanced_bubblelab_ui

    # Use base integration for analysis
    ui = get_bubblelab_ui_integration()
    advanced_ui = get_advanced_bubblelab_ui()

    # Analyze for UI
    result = ui.analyze_complexity_for_ui(
        problem_description="Build real-time analytics dashboard",
        domain="analytics",
        depth=2
    )

    print(f"[OK] UI Analysis Complete")
    print(f"  Problem ID: {result.problem_id}")
    print(f"  Complexity: {result.overall_complexity:.3f}")

    # Get radar chart data from advanced UI
    chart = advanced_ui.create_complexity_radar_chart(result.problem_id)

    if chart:
        print(f"\n[OK] Radar Chart Generated")
        print(f"  Type: {chart.chart_type.value}")
        print(f"  Labels: {chart.data.get('labels', [])}")

    # Get health dashboard
    dashboard = advanced_ui.create_adapter_health_dashboard()

    print(f"\n[OK] Health Dashboard Generated")
    mdap_health = dashboard.get('health', {}).get('mdap_adapter', {})
    print(f"  Overall Status: {mdap_health.get('status', 'unknown')}")
    print(f"  Active Alerts: {len(dashboard.get('alerts', []))}")

    # Export report from base integration
    report_json = ui.export_ui_data(format="json")

    print(f"\n[OK] Report Exported")
    print(f"  Size: {len(report_json)} characters")


def example_7_cross_system_workflow():
    """Example 7: Cross-System Workflow."""
    print_section("EXAMPLE 7: Cross-System Workflow Execution")

    from src import get_unified_system_monitor

    monitor = get_unified_system_monitor()

    # Check system health
    health = monitor.get_overall_health()

    print(f"[OK] System Health Check")
    print(f"  Overall: {health['overall_status']}")
    print(f"  Available Systems: {health['available_systems']}/{health['total_systems']}")

    # Execute cross-system workflow
    results = monitor.execute_workflow(
        workflow_type="formal_verification",
        parameters={
            "query": "Explain Byzantine fault tolerance",
            "constraints": ["x > 0", "y > x", "z > y"],
            "statement": "Theorem: Natural numbers are well-ordered"
        }
    )

    print(f"\n[OK] Cross-System Workflow Complete")
    print(f"  Steps Completed: {len(results['steps'])}")

    for step in results['steps']:
        status_icon = "[OK]" if step.get('success') else "[FAIL]"
        print(f"    {status_icon} {step['step']}: {step.get('system', 'unknown')}")


def example_8_full_unified_workflow():
    """Example 8: Complete End-to-End Workflow."""
    print_section("EXAMPLE 8: Complete End-to-End Workflow")

    from unified_entry import UnifiedAdapterInterface

    interface = UnifiedAdapterInterface()

    # Step 1: Basic analysis
    print("Step 1: Basic Complexity Analysis")
    basic = interface.analyze(
        problem="Design fault-tolerant distributed database",
        domain="distributed_systems"
    )
    print(f"  Complexity: {basic['complexity']:.3f}")

    # Step 2: Advanced analysis
    print("\nStep 2: Advanced Analysis with Decomposition")
    advanced = interface.analyze_advanced(
        problem="Design fault-tolerant distributed database",
        workflow_type="sovereign"
    )
    print(f"  Sub-Problems: {advanced['decomposition']['sub_problems']}")
    print(f"  Recommended Teams: {len(advanced['team_selection']['teams'])}")

    # Step 3: Verification
    print("\nStep 3: Multi-Gauntlet Verification")
    verify = interface.verify(
        solution="distributed_database_solution",
        complexity=basic['complexity']
    )
    print(f"  Overall Pass: {verify['overall_pass']}")
    print(f"  Aggregate Score: {verify['aggregate_score']:.3f}")

    # Step 4: Get status
    print("\nStep 4: System Status")
    status = interface.get_status()
    print(f"  MDAP: {status['adapter']['mdap']['status']}")
    print(f"  MAKER: {status['adapter']['maker']['status']}")
    print(f"  Advanced Components: {len(status['advanced_components'])}")
    print(f"  Systems: {status['systems']['available_systems']}/{status['systems']['total_systems']}")

    print("\n[OK] Complete End-to-End Workflow Successful!")


def main():
    """Run all examples."""
    print_section("ADAPTIVE MDAP/MAKER ADAPTER - COMPLETE FEATURE DEMONSTRATION")
    print(f"Start Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Run all examples
    try:
        example_1_basic_complexity_analysis()
        example_2_advanced_decomposition()
        example_3_gauntlet_pipeline()
        example_4_icr_learning()
        example_5_performance_optimization()
        example_6_ui_dashboard()
        example_7_cross_system_workflow()
        example_8_full_unified_workflow()
    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print(f"End Time: {datetime.now(timezone.utc).isoformat()}")
    print("\n[OK] All features demonstrated successfully!")
    print("[OK] Integration is complete and operational!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
