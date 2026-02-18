#!/usr/bin/env python3
"""
Example: Cross-System Workflow Execution

This example demonstrates how to execute workflows across multiple
integrated systems including CrewAI, MCP Tools, Knowledge Engine, LeanAide, and Z3.

Usage:
    cd examples
    python example_cross_system_workflow.py
"""

import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")

from src import get_unified_system_monitor


def main():
    """Demonstrate cross-system workflow execution."""
    print("=" * 70)
    print("  EXAMPLE: Cross-System Workflow Execution")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get unified system monitor
    monitor = get_unified_system_monitor()

    # Phase 1: Check system health
    print("Phase 1: System Health Check")
    print("-" * 70)

    health = monitor.get_overall_health()

    print(f"\nOverall Status: {health['overall_status']}")
    print(f"Available Systems: {health['available_systems']}/{health['total_systems']}")

    print("\nIndividual System Status:")
    for system_name, system_health in health['systems'].items():
        status = system_health['status']
        available = system_health['available']
        status_icon = "[UP]" if available else "[DOWN]"
        print(f"  {status_icon} {system_name}: {status}")

        if not available:
            print(f"      Reason: {system_health.get('reason', 'Unknown')}")

    # Phase 2: Execute formal verification workflow
    print("\n" + "=" * 70)
    print("Phase 2: Formal Verification Workflow")
    print("=" * 70)

    print("\nWorkflow: Verify theorem using Z3 Prover with knowledge from RAGBits")

    verification_results = monitor.execute_workflow(
        workflow_type="formal_verification",
        parameters={
            "statement": "For all natural numbers n, n + 0 = n",
            "constraints": [
                "n >= 0",
                "n is integer"
            ],
            "query": "What is the additive identity property?"
        }
    )

    print(f"\nWorkflow Complete: {verification_results['success']}")
    print(f"Total Steps: {len(verification_results['steps'])}")

    print("\nExecution Steps:")
    for i, step in enumerate(verification_results['steps'], 1):
        success = step.get('success', False)
        status = "[OK]" if success else "[FAIL]"
        system = step.get('system', 'unknown')
        action = step.get('action', 'unknown')

        print(f"\n  Step {i}: {status}")
        print(f"    System: {system}")
        print(f"    Action: {action}")

        if step.get('result'):
            print(f"    Result: {str(step['result'])[:100]}...")

        if step.get('error'):
            print(f"    Error: {step['error']}")

    # Phase 3: Execute agent collaboration workflow
    print("\n" + "=" * 70)
    print("Phase 3: Agent Collaboration Workflow")
    print("=" * 70)

    print("\nWorkflow: Coordinate agents using CrewAI")

    collaboration_results = monitor.execute_workflow(
        workflow_type="agent_collaboration",
        parameters={
            "task": "Design microservices architecture for e-commerce platform",
            "agents": [
                {"name": "architect", "role": "system_architect"},
                {"name": "security_expert", "role": "security_analyst"},
                {"name": "database_specialist", "role": "database_designer"}
            ],
            "collaboration_mode": "hierarchical"
        }
    )

    print(f"\nWorkflow Complete: {collaboration_results['success']}")
    print(f"Total Steps: {len(collaboration_results['steps'])}")

    print("\nExecution Steps:")
    for i, step in enumerate(collaboration_results['steps'], 1):
        success = step.get('success', False)
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} Step {i}: {step.get('system', 'unknown')} - {step.get('action', 'unknown')}")

    # Phase 4: Execute knowledge retrieval workflow
    print("\n" + "=" * 70)
    print("Phase 4: Knowledge Retrieval Workflow")
    print("=" * 70)

    print("\nWorkflow: Query knowledge engine with MCP tools")

    knowledge_results = monitor.execute_workflow(
        workflow_type="knowledge_retrieval",
        parameters={
            "query": "Explain Byzantine fault tolerance in distributed systems",
            "max_results": 5,
            "include_context": True
        }
    )

    print(f"\nWorkflow Complete: {knowledge_results['success']}")
    print(f"Results Found: {knowledge_results.get('result_count', 0)}")
    print(f"Total Steps: {len(knowledge_results['steps'])}")

    # Phase 5: Execute multi-system workflow
    print("\n" + "=" * 70)
    print("Phase 5: Multi-System Workflow")
    print("=" * 70)

    print("\nWorkflow: Complex problem requiring multiple systems")

    multi_results = monitor.execute_workflow(
        workflow_type="multi_system",
        parameters={
            "problem": "Prove system correctness and deploy with agent coordination",
            "steps": [
                {
                    "system": "knowledge_engine",
                    "action": "retrieve_relevant_theorems",
                    "params": {"topic": "distributed_consensus"}
                },
                {
                    "system": "z3_prover",
                    "action": "verify_theorem",
                    "params": {"theorem": "Consensus ensures safety"}
                },
                {
                    "system": "crewai",
                    "action": "coordinate_deployment",
                    "params": {"agents": ["deployer", "monitor", "scaler"]}
                }
            ]
        }
    )

    print(f"\nWorkflow Complete: {multi_results['success']}")
    print(f"Total Steps: {len(multi_results['steps'])}")

    print("\nDetailed Steps:")
    for i, step in enumerate(multi_results['steps'], 1):
        print(f"\n  {i}. System: {step.get('system', 'unknown')}")
        print(f"     Action: {step.get('action', 'unknown')}")
        print(f"     Success: {step.get('success', False)}")

        if step.get('duration_ms'):
            print(f"     Duration: {step['duration_ms']:.0f}ms")

    # Phase 6: System statistics
    print("\n" + "=" * 70)
    print("Phase 6: System Statistics")
    print("=" * 70)

    stats = monitor.get_system_statistics()

    print(f"\nTotal Workflows Executed: {stats.get('total_workflows', 0)}")
    print(f"Successful Workflows: {stats.get('successful_workflows', 0)}")
    print(f"Failed Workflows: {stats.get('failed_workflows', 0)}")
    print(f"Success Rate: {stats.get('success_rate', 0):.1%}")

    print("\nSystem Usage:")
    for system, usage in stats.get('system_usage', {}).items():
        print(f"  {system}: {usage.get('calls', 0)} calls, {usage.get('success_rate', 0):.1%} success")

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
