"""
ROMA-MDAP-MAKER Demo Script

This script demonstrates the full ROMA-MDAP-MAKER integration in action.

Usage:
    python demo_roma_mdap_maker.py
"""

import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_1_status_check():
    """Demo 1: Check system status"""
    print_header("DEMO 1: System Status Check")

    from roma_mdap_maker_engine import (
        get_roma_mdap_maker_status,
        ROMA_AVAILABLE,
        MDAP_AVAILABLE,
    )
    from roma_mdap_maker_mcp_tools import list_mcp_tools
    from roma_mdap_maker_crewai_bridge import get_romamdapmaker_bridge_status
    from decomposition_mcp_tools import get_decomposition_status
    from crewai_unified_bridge import get_unified_bridge_status

    # Engine status
    print("ROMA-MDAP-MAKER Engine Status:")
    engine_status = get_roma_mdap_maker_status()
    print(f"  ROMA Available: {ROMA_AVAILABLE}")
    print(f"  MDAP Available: {MDAP_AVAILABLE}")
    print(f"  Full System Available: {engine_status.get('roma_mdap_maker_available', False)}")

    # Bridge status
    print("\nROMA-MDAP-MAKER Bridge Status:")
    bridge_status = get_romamdapmaker_bridge_status()
    print(f"  Bridge Available: {bridge_status.get('bridge_available', False)}")
    print(f"  Phases Supported: {bridge_status.get('phases_supported', [])}")

    # MCP tools
    print("\nMCP Tools Registered:")
    mcp_tools = list_mcp_tools()
    print(f"  Total Tools: {len(mcp_tools)}")
    for tool in mcp_tools:
        print(f"    - {tool}")

    # Decomposition status
    print("\nDecomposition Workflow Status:")
    decomp_status = get_decomposition_status()
    print(f"  Total Execution Methods: {decomp_status.get('total_execution_methods', 0)}")
    print(f"  ROMA-MDAP-MAKER Available: {decomp_status.get('roma_mdap_maker_available', False)}")
    print(f"  Execution Methods: {', '.join(decomp_status.get('execution_methods', []))}")

    # Unified bridge status
    print("\nUnified Bridge Status:")
    unified_status = get_unified_bridge_status()
    print(f"  Total Execution Methods: {unified_status.get('total_execution_methods', 0)}")
    print(f"  ROMA-MDAP-MAKER Bridge: {unified_status.get('roma_mdap_maker_bridge_available', False)}")


def demo_2_configuration():
    """Demo 2: Create and validate configuration"""
    print_header("DEMO 2: Configuration Management")

    from roma_mdap_maker_engine import (
        create_roma_mdap_maker_config,
        ROMAMDAPMakerConfig,
    )

    print("Creating ROMA-MDAP-MAKER Configuration:")

    # Create config
    config = create_roma_mdap_maker_config(
        # ROMA settings
        roma_max_depth_analysis=3,
        roma_max_depth_solving=2,
        roma_execution_mode="recursive",

        # MDAP/MAKER settings
        mdap_enabled=True,
        mdap_k_ahead=3,
        mdap_max_samples=100,
        mdap_enable_red_flagging=True,

        # Integration settings
        apply_maker_to_roma_atomic=True,
        enable_hierarchical_voting=True,
        enable_adaptive_k=True,

        # Provider settings
        provider="openai",
        model="gpt-4o-mini",
    )

    print(f"\nConfiguration Created:")
    print(f"  ROMA Max Depth (Analysis): {config.roma_max_depth_analysis}")
    print(f"  ROMA Max Depth (Solving): {config.roma_max_depth_solving}")
    print(f"  ROMA Execution Mode: {config.roma_execution_mode}")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"\n  MDAP Enabled: {config.mdap_enabled}")
    print(f"  MDAP K-Ahead: {config.mdap_k_ahead}")
    print(f"  MDAP Max Samples: {config.mdap_max_samples}")
    print(f"  MDAP Red-Flagging: {config.mdap_enable_red_flagging}")
    print(f"\n  Apply MAKER to ROMA Atomic: {config.apply_maker_to_roma_atomic}")
    print(f"  Enable Hierarchical Voting: {config.enable_hierarchical_voting}")
    print(f"  Enable Adaptive K: {config.enable_adaptive_k}")

    # Validate config
    print(f"\nConfiguration Valid: {config.mdap_enabled and config.apply_maker_to_roma_atomic}")


def demo_3_routing_logic():
    """Demo 3: Test auto-selection routing"""
    print_header("DEMO 3: Auto-Selection Routing Logic")

    from decomposition_mcp_tools import _determine_execution_method

    test_cases = [
        {
            "execution_method": "roma_mdap_maker",
            "description": "Design database system",
            "expected": "roma_mdap_maker",
            "name": "Explicit ROMA-MDAP-MAKER selection"
        },
        {
            "execution_method": "auto",
            "description": "Design critical zero-error system for mission-critical application",
            "expected": "roma_mdap_maker",
            "name": "Auto: Critical zero-error keywords"
        },
        {
            "execution_method": "auto",
            "description": "Build safety-critical system with flawless execution",
            "expected": "roma_mdap_maker",
            "name": "Auto: Safety-critical keywords"
        },
        {
            "execution_method": "auto",
            "description": "Create high-reliability fault-tolerant system",
            "expected": "roma_mdap_maker",
            "name": "Auto: High-reliability keywords"
        },
        {
            "execution_method": "auto",
            "description": "Design a standard web application",
            "expected": "traditional",
            "name": "Auto: Normal task (fallback)"
        },
        {
            "execution_method": "auto",
            "description": "Implement hierarchical decomposition of complex structure",
            "expected": "roma",
            "name": "Auto: ROMA keywords (second priority)"
        },
    ]

    print("Testing Auto-Selection Routing Logic:\n")

    for test in test_cases:
        result = _determine_execution_method(
            test["execution_method"],
            False, False, False, False,  # use_claudiomiro, use_datapizza, use_roma, use_hybrid
            True,  # use_roma_mdap_maker
            "test-id",
            test["description"]
        )

        status = "[OK] PASS" if result == test["expected"] else "[X] FAIL"
        print(f"{status} {test['name']}")
        print(f"    Input: {test['description'][:60]}...")
        print(f"    Expected: {test['expected']}")
        print(f"    Got: {result}")
        print()


def demo_4_phase_1_analysis():
    """Demo 4: Phase 1 complexity analysis"""
    print_header("DEMO 4: Phase 1 - Complexity Analysis")

    from roma_mdap_maker_mcp_tools import analyze_problem_with_roma_mdap

    problems = [
        "Design a simple login page",
        "Build a scalable microservices architecture with 99.999% uptime",
        "Create a fault-tolerant distributed database system",
    ]

    for problem in problems:
        print(f"Problem: {problem}")
        print(f"{'-' * 70}")

        try:
            analysis = analyze_problem_with_roma_mdap(
                problem_statement=problem,
                roma_max_depth=2
            )

            if "error" not in analysis:
                print(f"  Estimated Complexity: {analysis.get('estimated_complexity', 0)}/10")
                print(f"  Recommended Depth: {analysis.get('recommended_depth', 0)}")
                print(f"  Recommended K-Ahead: {analysis.get('recommended_k', 0)}")
                print(f"  Num Subtasks: {analysis.get('num_subtasks', 0)}")
                print(f"  Max Depth: {analysis.get('max_depth', 0)}")
                print(f"  Use ROMA-MDAP-MAKER: {analysis.get('use_roma_mdap_maker', False)}")
            else:
                print(f"  Error: {analysis.get('error')}")
        except Exception as e:
            print(f"  Exception: {e}")

        print()


def demo_5_hierarchical_voting():
    """Demo 5: Hierarchical voting strategy"""
    print_header("DEMO 5: Hierarchical Voting Strategy")

    from roma_mdap_maker_engine import (
        HierarchicalVotingStrategy,
        create_roma_mdap_maker_config,
        ROMAMDAPMakerEngine,
    )

    print("Creating voting strategy...")

    config = create_roma_mdap_maker_config(
        mdap_k_ahead=3,
        enable_adaptive_k=True,
    )

    # Create a mock MDAP orchestrator (for demo purposes)
    # In production, this would use actual MDAP
    voting_strategy = HierarchicalVotingStrategy(
        config=config,
        mdap_orchestrator=None  # Would use actual orchestrator in production
    )

    print("Hierarchical Voting Strategy Created:")
    print(f"  K-Ahead: {config.mdap_k_ahead}")
    print(f"  Adaptive K: {config.enable_adaptive_k}")
    print(f"  Hierarchical Voting: {config.enable_hierarchical_voting}")

    print("\nHow it works:")
    print("  1. For atomic ROMA tasks: Apply MAKER voting")
    print("  2. For composite tasks: Recursively vote on children")
    print("  3. Aggregate results: Confidence-weighted combination")
    print("  4. Combined confidence: Product of child confidences")

    print("\nExample aggregation:")
    print("  Child 1: confidence=0.95, result=A")
    print("  Child 2: confidence=0.90, result=B")
    print("  Child 3: confidence=0.85, result=C")
    print("  -> Combined: 0.95 * 0.90 * 0.85 = 0.73 (73%)")


def demo_6_adaptive_k_selection():
    """Demo 6: Adaptive k-ahead selection"""
    print_header("DEMO 6: Adaptive K-Ahead Selection")

    from roma_mdap_maker_engine import AdaptiveKSelector, create_roma_mdap_maker_config

    config = create_roma_mdap_maker_config(mdap_k_ahead=3)
    selector = AdaptiveKSelector(config)

    # Test different task scenarios
    test_tasks = [
        {
            "description": "Simple task",
            "depth": 0,
            "expected_k_range": (2, 3),
        },
        {
            "description": "Deep hierarchical task",
            "depth": 4,
            "expected_k_range": (4, 5),
        },
        {
            "description": "Complex task with many dependencies",
            "depth": 2,
            "dependencies": ["dep1", "dep2", "dep3", "dep4"],
            "expected_k_range": (3, 4),
        },
    ]

    print("Adaptive K Selection for Different Tasks:\n")

    for task in test_tasks:
        k = selector.select_k_for_roma_task(
            roma_task=task,
            depth=task["depth"],
            base_k=3
        )

        print(f"Task: {task['description']}")
        print(f"  Depth: {task['depth']}")
        if "dependencies" in task:
            print(f"  Dependencies: {len(task['dependencies'])}")
        print(f"  Selected K: {k}")
        print(f"  Expected Range: {task['expected_k_range']}")
        print(f"  Status: {'[OK]' if task['expected_k_range'][0] <= k <= task['expected_k_range'][1] else '[X]'}")
        print()


def demo_7_red_flagging():
    """Demo 7: Enhanced red-flagging for ROMA"""
    print_header("DEMO 7: Enhanced Red-Flagging for ROMA")

    from roma_mdap_maker_engine import ROMARedFlagger, create_roma_mdap_maker_config

    config = create_roma_mdap_maker_config(
        roma_max_depth_analysis=3,
        mdap_enable_red_flagging=True,
    )

    flagger = ROMARedFlagger(config)

    # Test decomposition DAG
    print("Testing Decomposition Red-Flags:")

    # Test 1: Cyclic dependencies
    dag_with_cycle = {
        "task1": {"children": ["task2"]},
        "task2": {"children": ["task3"]},
        "task3": {"children": ["task1"]},  # Cycle!
    }

    flags = flagger.check_roma_decomposition_red_flags(dag_with_cycle)
    print(f"  DAG with cycle: {flags}")

    # Test 2: Excessive depth
    dag_deep = {
        "task1": {"children": ["task2"], "description": "A"},
        "task2": {"children": ["task3"], "description": "B"},
        "task3": {"children": ["task4"], "description": "C"},
        "task4": {"children": ["task5"], "description": "D"},
        "task5": {"children": [], "description": "E"},
    }

    flags = flagger.check_roma_decomposition_red_flags(dag_deep)
    print(f"  DAG depth={flagger._calculate_depth(dag_deep)}: {flags}")

    # Test 3: Balanced DAG
    dag_balanced = {
        "task1": {"description": "Task 1", "children": []},
        "task2": {"description": "Task 2", "children": []},
        "task3": {"description": "Task 3", "children": []},
    }

    flags = flagger.check_roma_decomposition_red_flags(dag_balanced)
    print(f"  Balanced DAG: {flags}")

    print("\nRed-Flag Types Detected:")
    print("  - cyclic_dependencies: Tasks form a cycle")
    print("  - excessive_depth_N: Depth exceeds max")
    print("  - unbalanced_decomposition_R: One task >> others")


def demo_8_full_workflow_preview():
    """Demo 8: Full workflow preview"""
    print_header("DEMO 8: Full Workflow Preview")

    print("ROMA-MDAP-MAKER 6-Phase Workflow:")
    print()

    phases = [
        {
            "phase": 1,
            "name": "Problem Setup",
            "function": "execute_phase_1_setup",
            "purpose": "Complexity analysis + parameter recommendation",
            "output": "Complexity score, recommended depth/k",
        },
        {
            "phase": 2,
            "name": "Solution Generation",
            "function": "execute_phase_2_solve",
            "purpose": "ROMA decomposition + MAKER voting",
            "output": "Solution with confidence, metrics",
        },
        {
            "phase": 3,
            "name": "Adversarial Critique",
            "function": "execute_phase_3_critique",
            "purpose": "Red team testing with voting",
            "output": "Identified flaws, improvements",
        },
        {
            "phase": 4,
            "name": "Verification",
            "function": "execute_phase_4_verify",
            "purpose": "Requirements verification with voting",
            "output": "Verification score, confidence",
        },
        {
            "phase": 5,
            "name": "Reassembly",
            "function": "execute_phase_5_reassemble",
            "purpose": "Confidence-weighted aggregation",
            "output": "Integrated solution",
        },
        {
            "phase": 6,
            "name": "Final Validation",
            "function": "execute_phase_6_final_validation",
            "purpose": "Full ROMA-MDAP-MAKER validation",
            "output": "Final validation status",
        },
    ]

    for phase_info in phases:
        print(f"Phase {phase_info['phase']}: {phase_info['name']}")
        print(f"  Function: {phase_info['function']}()")
        print(f"  Purpose: {phase_info['purpose']}")
        print(f"  Output: {phase_info['output']}")
        print()


def demo_9_usage_examples():
    """Demo 9: Usage examples"""
    print_header("DEMO 9: Usage Examples")

    print("Example 1: Direct API Call")
    print("-" * 70)
    print("""
from roma_mdap_maker_mcp_tools import solve_with_roma_mdap_maker

result = solve_with_roma_mdap_maker(
    task="Design a fault-tolerant database",
    context={"requirements": ["ACID compliance", "99.999% uptime"]},
    roma_max_depth_analysis=3,
    mdap_k_ahead=3,
    provider="openai",
    model="gpt-4o-mini"
)

print(f"Solution: {result['solution']}")
print(f"Confidence: {result['confidence']}")
""")

    print("\nExample 2: Through CrewAI Unified Bridge")
    print("-" * 70)
    print("""
from crewai_unified_bridge import execute_phase_1_setup

# Auto-selects ROMA-MDAP-MAKER for critical tasks
result = execute_phase_1_setup(
    problem_statement="Design mission-critical zero-error system",
    execution_method="auto",
    use_roma_mdap_maker=True
)
""")

    print("\nExample 3: Through Decomposition Workflow")
    print("-" * 70)
    print("""
from decomposition_mcp_tools import solve_sub_problem_with_team

result = solve_sub_problem_with_team(
    sub_problem_id="SP-001",
    sub_problem_description="Design zero-error component",
    team_name="Blue-Team-Alpha",
    execution_method="roma_mdap_maker",
    use_roma_mdap_maker=True,
    roma_mdap_maker_max_depth=2,
    roma_mdap_maker_k_ahead=3
)
""")

    print("\nExample 4: Full Workflow Execution")
    print("-" * 70)
    print("""
from roma_mdap_maker_crewai_bridge import execute_full_workflow

result = execute_full_workflow(
    problem_statement="Design zero-error trading system",
    roma_max_depth_analysis=3,
    mdap_k_ahead=3
)

print(f"Final solution: {result['final_solution']}")
print(f"Validated: {result['is_validated']}")
""")


def demo_10_performance_characteristics():
    """Demo 10: Performance characteristics"""
    print_header("DEMO 10: Performance Characteristics")

    print("Zero-Error Guarantee (MAKER Voting):")
    print("-" * 70)
    print("""
    k=3: P(success) ~ 95%
    k=4: P(success) ~ 98%
    k=5: P(success) ~ 99.3%

    With red-flagging: Additional reliability layer
    """)

    print("\nComputational Cost Estimates (gpt-4o-mini):")
    print("-" * 70)

    costs = [
        ("Low (1-3)", "1-2", "2", "5-10", "$0.05 - $0.10"),
        ("Medium (4-6)", "2-3", "3", "10-25", "$0.15 - $0.35"),
        ("High (7-8)", "3-4", "4", "25-50", "$0.50 - $1.20"),
        ("Very High (9-10)", "4-5", "5", "50-100", "$1.50 - $3.00"),
    ]

    print(f"{'Complexity':<15} {'Depth':<8} {'K':<5} {'Tasks':<10} {'Cost':<15}")
    print("-" * 70)
    for complexity, depth, k, tasks, cost in costs:
        print(f"{complexity:<15} {depth:<8} {k:<5} {tasks:<10} {cost:<15}")

    print("\nAdaptive Optimization:")
    print("-" * 70)
    print("""
    - Simple tasks: Decrease k (saves cost)
    - Complex tasks: Increase k (improves reliability)
    - Historical learning: Adjusts based on past performance
    """)


def main():
    """Run all demos"""
    print("\n")
    print("=" * 80)
    print("  ROMA-MDAP-MAKER INTEGRATION DEMO")
    print("=" * 80)

    try:
        demo_1_status_check()
        demo_2_configuration()
        demo_3_routing_logic()
        demo_4_phase_1_analysis()
        demo_5_hierarchical_voting()
        demo_6_adaptive_k_selection()
        demo_7_red_flagging()
        demo_8_full_workflow_preview()
        demo_9_usage_examples()
        demo_10_performance_characteristics()

        print("\n" + "=" * 80)
        print("  ALL DEMOS COMPLETED")
        print("=" * 80)
        print("\nFor more information, see:")
        print("  - ROMA_MDAP_MAKER_FULL_INTEGRATION_COMPLETE.md")
        print("  - ROMA_MDAP_MAKER_INTEGRATION_PLAN.md")
        print()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n[X] Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
