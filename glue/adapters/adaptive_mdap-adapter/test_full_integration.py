#!/usr/bin/env python3
"""
End-to-End Integration Test for Adaptive MDAP/MAKER Adapter

This script demonstrates and validates the complete integration between:
- OpenEvolve workflows
- BubbleLab UI
- Gauntlet system
- ICR pattern learning

Usage:
    python test_full_integration.py
"""

import os
import sys
import logging
import time
from datetime import datetime, timezone

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set required environment variables
os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"] = "5000"
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def test_openevolve_integration():
    """Test OpenEvolve workflow integration."""
    print_section("TEST 1: OpenEvolve Workflow Integration")

    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Analyze workflow complexity
        print("Analyzing workflow complexity...")
        analysis = manager.analyze_workflow(
            workflow_id="test_workflow_001",
            problem_statement="Implement secure OAuth2 authentication system with role-based access control",
            workflow_type="evolution",
            context={
                "domain": "security",
                "depth": 3
            }
        )

        print(f"SUCCESS: Complexity Analysis Complete")
        print(f"  Workflow ID: {analysis.workflow_id}")
        print(f"  Workflow Type: {analysis.workflow_type}")
        print(f"  Overall Complexity: {analysis.overall_complexity:.3f}")
        print(f"  Recommended Strategy: {analysis.recommended_strategy}")
        print(f"  Recommended Resources: {analysis.recommended_resources}")
        print(f"  Estimated Duration: {analysis.estimated_duration_ms:.0f}ms")
        print(f"  Timestamp: {analysis.timestamp}")

        # Make workflow decision
        print("\nMaking workflow decision...")
        decision = manager.make_decision(
            workflow_id="test_workflow_001",
            stage="planning",
            decision_point="Select execution approach",
            options=[
                {"action": "mdap_parallel", "description": "Use MDAP parallel execution"},
                {"action": "maker_sequential", "description": "Use MAKER sequential voting"},
                {"action": "hybrid", "description": "Use hybrid MDAP+MAKER approach"}
            ]
        )

        print(f"SUCCESS: Workflow Decision Complete")
        print(f"  Stage: {decision.stage}")
        print(f"  Decision Point: {decision.decision_point}")
        print(f"  Votes Collected: {decision.votes_collected}")
        print(f"  Consensus Reached: {decision.consensus_reached}")
        print(f"  Consensus Score: {decision.consensus_score:.3f}")
        print(f"  Recommended Action: {decision.recommended_action}")
        print(f"  Red Flags: {len(decision.red_flags)}")

        # Select adaptive gauntlet
        print("\nSelecting adaptive gauntlet...")
        gauntlet = manager.select_gauntlet(
            workflow_id="test_workflow_001",
            complexity_score=analysis.overall_complexity,
            base_gauntlet_type="adversarial"
        )

        print(f"SUCCESS: Gauntlet Selection Complete")
        print(f"  Gauntlet Type: {gauntlet.get('gauntlet_type')}")
        print(f"  Adapted: {gauntlet.get('adapted')}")
        print(f"  Complexity Score: {gauntlet.get('complexity_score'):.3f}")
        print(f"  Adaptation Reason: {gauntlet.get('adaptation_reason')}")

        return True

    except Exception as e:
        print(f"FAILED: OpenEvolve integration test failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bubblelab_ui_integration():
    """Test BubbleLab UI integration."""
    print_section("TEST 2: BubbleLab UI Integration")

    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Analyze for UI
        print("Analyzing complexity for UI...")
        ui_result = manager.analyze_for_ui(
            problem_description="Design and implement scalable microservices architecture",
            domain="architecture",
            depth=2,
            dependencies=["kubernetes", "docker", "service_mesh"]
        )

        print(f"SUCCESS: UI Analysis Complete")
        print(f"  Problem ID: {ui_result.problem_id}")
        print(f"  Overall Complexity: {ui_result.overall_complexity:.3f}")
        print(f"  Text Length Score: {ui_result.text_length_score:.3f}")
        print(f"  Dependency Score: {ui_result.dependency_score:.3f}")
        print(f"  Depth Score: {ui_result.depth_score:.3f}")
        print(f"  Recommended Strategy: {ui_result.recommended_strategy}")
        print(f"  Execution Time: {ui_result.execution_time_ms:.2f}ms")

        # Get UI data
        print("\nGetting UI data...")
        ui_data = manager.get_ui_data()

        print(f"SUCCESS: UI Data Retrieved")
        print(f"  Analyses: {len(ui_data.get('analyses', {}))}")
        print(f"  Votings: {len(ui_data.get('votings', {}))}")
        print(f"  Active Workflows: {ui_data.get('workflow_monitor', {}).get('active_workflows', 0)}")

        # Get health status
        print("\nGetting health status...")
        health = manager.get_health_status()

        print(f"SUCCESS: Health Status Retrieved")
        print(f"  Overall Status: {health.overall_status.value}")
        print(f"  MDAP Adapter: {health.mdap_adapter_status}")
        print(f"  MAKER Adapter: {health.maker_adapter_status}")
        print(f"  OpenEvolve Integration: {health.openevolve_integration_status}")
        print(f"  BubbleLab UI: {health.bubblelab_ui_status}")
        print(f"  ICR Integration: {health.icr_integration_status}")
        print(f"  Gauntlet Integration: {health.gauntlet_integration_status}")

        return True

    except Exception as e:
        print(f"FAILED: BubbleLab UI integration test failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow_execution():
    """Test complete workflow execution."""
    print_section("TEST 3: Full Workflow Execution")

    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        print("Executing full workflow...")
        results = manager.execute_full_workflow(
            workflow_id="test_full_workflow_001",
            problem_statement="Build real-time data processing pipeline with fault tolerance",
            workflow_type="evolution",
            context={
                "domain": "data_engineering",
                "base_gauntlet_type": "statistical"
            }
        )

        print(f"SUCCESS: Full Workflow Execution Complete")
        print(f"  Workflow ID: {results.get('workflow_id')}")
        print(f"  Workflow Type: {results.get('workflow_type')}")
        print(f"  Overall Status: {results.get('overall_status')}")
        print(f"  Execution Time: {results.get('execution_time_ms', 0):.2f}ms")
        print(f"  Steps Completed: {len(results.get('steps', []))}")

        for i, step in enumerate(results.get('steps', []), 1):
            print(f"\n  Step {i}: {step.get('step')}")
            print(f"    Status: {step.get('status')}")
            if step.get('step') == 'complexity_analysis':
                print(f"    Complexity: {step.get('complexity', 0):.3f}")
                print(f"    Strategy: {step.get('strategy')}")
            elif step.get('step') == 'gauntlet_selection':
                print(f"    Gauntlet Type: {step.get('gauntlet_type')}")
                print(f"    Adapted: {step.get('adapted')}")
            elif step.get('step') == 'initial_decision':
                print(f"    Action: {step.get('action')}")
                print(f"    Consensus: {step.get('consensus_reached')}")

        return results.get('overall_status') == 'completed'

    except Exception as e:
        print(f"FAILED: Full workflow execution test failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_icr_pattern_learning():
    """Test ICR pattern learning integration."""
    print_section("TEST 4: ICR Pattern Learning")

    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Check if ICR is available
        health = manager.get_health_status()

        if health.icr_integration_status == "disabled":
            print("INFO: ICR integration disabled, skipping test")
            return True

        # Get ICR insights
        print("Getting ICR insights...")
        ui_data = manager.get_ui_data()
        icr_insights = ui_data.get('icr_insights', {})

        print(f"SUCCESS: ICR Insights Retrieved")
        print(f"  Available: {icr_insights.get('available', False)}")

        if icr_insights.get('available'):
            patterns = icr_insights.get('patterns', {})
            print(f"  Pattern Types: {len(patterns)}")

            for pattern_type, stats in patterns.items():
                print(f"\n  {pattern_type}:")
                print(f"    Count: {stats.get('count', 0)}")
                print(f"    Pass Rate: {stats.get('pass_rate', 0):.3f}")
                print(f"    Confidence: {stats.get('confidence', 0):.3f}")

        return True

    except Exception as e:
        print(f"FAILED: ICR pattern learning test failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print_section("ADAPTIVE MDAP/MAKER ADAPTER - FULL INTEGRATION TEST")
    print(f"Start Time: {datetime.now(timezone.utc).isoformat()}")

    results = {}

    # Run tests
    results['openevolve'] = test_openevolve_integration()
    results['bubblelab_ui'] = test_bubblelab_ui_integration()
    results['full_workflow'] = test_full_workflow_execution()
    results['icr'] = test_icr_pattern_learning()

    # Print summary
    print_section("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "[OK]" if passed else "[FAIL]"
        print(f"  {symbol} {test_name}: {status}")

    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}")

    if passed == total:
        print("\nSUCCESS: All integration tests passed!")
        return 0
    else:
        print(f"\nFAILED: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
