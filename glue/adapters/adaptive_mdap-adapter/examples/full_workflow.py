"""
Example: Full MDAP/MAKER Workflow with BubbleLab Integration

Demonstrates a complete workflow using:
1. Adaptive MDAP for complexity analysis
2. Resource allocation based on complexity
3. MAKER voting for solution validation
4. BubbleLab API client for remote execution
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    TaskStatus,
    AdaptiveMDAPAdapterConfig
)

from maker_adapter import (
    get_maker_adapter,
    CanonicalMakerStep
)

from bubblelab_api_client import (
    get_bubblelab_client,
    BubbleLabAPIClientConfig
)


def analyze_with_mdap(problem: dict) -> dict:
    """
    Step 1: Analyze problem complexity using Adaptive MDAP.

    Returns:
        Problem analysis with complexity score
    """
    print("Step 1: Analyzing complexity with Adaptive MDAP")
    print("-" * 60)

    adapter = get_adapter()

    subproblem = CanonicalSubProblem(
        id=problem["id"],
        description=problem["description"],
        domain=problem.get("domain", "general"),
        depth=problem.get("depth", 1),
        dependencies=problem.get("dependencies", []),
        metadata=problem.get("metadata", {})
    )

    response = adapter.analyze_complexity(
        subproblem=subproblem,
        correlation_id=f"workflow-{problem['id']}"
    )

    if response.status == TaskStatus.COMPLETED:
        print(f"✓ Complexity Analysis Complete")
        print(f"  Overall Score: {response.complexity_score.overall_score:.2f}")
        print(f"  Text Length: {response.complexity_score.text_length_score:.2f}")
        print(f"  Dependencies: {response.complexity_score.dependency_score:.2f}")
        print(f"  Depth: {response.complexity_score.depth_score:.2f}")

        return {
            "complexity": response.complexity_score,
            "execution_time_ms": response.execution_time_ms
        }
    else:
        print(f"✗ Complexity Analysis Failed: {response.error}")
        return {"error": str(response.error)}


def allocate_resources(complexity_score: float) -> dict:
    """
    Step 2: Allocate resources based on complexity.

    Returns:
        Resource allocation strategy
    """
    print("\nStep 2: Allocating resources based on complexity")
    print("-" * 60)

    adapter = get_adapter()

    # Create complexity score object
    from adaptive_mdap_adapter import CanonicalComplexityScore
    complexity = CanonicalComplexityScore(overall_score=complexity_score)

    response = adapter.allocate_resources(
        complexity_score=complexity,
        correlation_id="workflow-allocation"
    )

    if response.status == TaskStatus.COMPLETED:
        strategy = response.strategy
        print(f"✓ Resource Allocation Complete")
        print(f"  Strategy: {strategy.strategy}")
        print(f"  Number of Agents: {strategy.n_agents}")
        print(f"  K-Ahead: {strategy.k_ahead}")
        print(f"  Max Retries: {strategy.max_retries}")
        print(f"  Timeout: {strategy.timeout_ms}ms")

        return {
            "strategy": strategy.strategy,
            "n_agents": strategy.n_agents,
            "k_ahead": strategy.k_ahead,
            "max_retries": strategy.max_retries,
            "timeout_ms": strategy.timeout_ms
        }
    else:
        print(f"✗ Resource Allocation Failed: {response.error}")
        return {"error": str(response.error)}


def validate_with_maker(problem: dict, strategy: dict) -> dict:
    """
    Step 3: Validate solution using MAKER voting.

    Returns:
        MAKER validation result
    """
    print("\nStep 3: Validating with MAKER voting")
    print("-" * 60)

    maker_adapter = get_maker_adapter()

    # Create MAKER step
    step = CanonicalMakerStep(
        step_id=f"maker-{problem['id']}",
        prompt_template=f"Analyze this solution: {{state}}\nProblem: {problem['description']}",
        task_type=problem.get("domain", "general"),
        priority=1 if strategy.get("n_agents", 1) <= 3 else 2,
        system_prompt="You are a validation expert",
        expected_schema={"type": "object"},
        metadata={"strategy": strategy.get("strategy", "UNKNOWN")}
    )

    # Create mock team (in production, use actual team configuration)
    class MockTeam:
        def __init__(self):
            self.name = "validation-team"

    current_state = {"problem": problem, "strategy": strategy}
    history = []
    team = MockTeam()

    response = maker_adapter.execute_maker_step(
        step=step,
        current_state=current_state,
        history=history,
        team=team,
        correlation_id=f"workflow-maker-{problem['id']}"
    )

    print(f"✓ MAKER Validation Complete")
    print(f"  Steps Completed: {response.steps_completed}")
    print(f"  Votes Cast: {response.votes_cast}")
    print(f"  Red Flags: {response.red_flags_detected}")
    print(f"  Terminated: {response.terminated_reason}")

    return {
        "steps_completed": response.steps_completed,
        "votes_cast": response.votes_cast,
        "red_flags": response.red_flags_detected,
        "terminated_reason": response.terminated_reason,
        "success": response.success
    }


def execute_on_bubblelab(problem: dict, complexity: float, strategy: dict) -> dict:
    """
    Step 4: Execute on BubbleLab API (if available).

    Returns:
        BubbleLab execution result
    """
    print("\nStep 4: Executing on BubbleLab API")
    print("-" * 60)

    try:
        client = get_bubblelab_client()

        # Check API health
        health = client.health_check()
        print(f"  API Health: {health['status']}")

        if health['status'] != "healthy":
            print("  ⚠ BubbleLab API not healthy, skipping remote execution")
            return {"skipped": "API not healthy"}

        # Determine MAKER config based on strategy
        num_agents = strategy.get("n_agents", 1)

        # Execute MDAP/MAKER solve
        result = client.solve_with_mdap_maker(
            problem_statement=problem["description"],
            use_mdap=complexity > 0.5,
            use_associative=True,
            num_mdap_agents=num_agents
        )

        print(f"✓ BubbleLab Execution Complete")
        print(f"  Success: {result.get('success')}")

        return result

    except Exception as e:
        print(f"⚠ BubbleLab execution failed: {e}")
        return {"error": str(e), "skipped": True}


def full_workflow_example():
    """
    Run a complete workflow example demonstrating all components.
    """
    print("=" * 60)
    print("FULL MDAP/MAKER/BUBBLELAB WORKFLOW EXAMPLE")
    print("=" * 60)
    print()

    # Define a complex problem
    problem = {
        "id": "workflow-example-001",
        "description": "Design and implement a secure, scalable microservices architecture for a distributed e-commerce platform with real-time inventory management, payment processing, and order fulfillment",
        "domain": "distributed_systems",
        "depth": 5,
        "dependencies": [
            "authentication-service",
            "inventory-service",
            "payment-service",
            "fulfillment-service",
            "message-bus"
        ],
        "metadata": {
            "scale": "enterprise",
            "security": "high",
            "availability": "99.9"
        }
    }

    print(f"Problem: {problem['description']}")
    print(f"Domain: {problem['domain']}")
    print(f"Depth: {problem['depth']}")
    print(f"Dependencies: {len(problem['dependencies'])}")
    print()

    # Step 1: Analyze complexity
    analysis = analyze_with_mdap(problem)

    if "error" in analysis:
        print("\n✗ Workflow failed at complexity analysis")
        return

    complexity_score = analysis["complexity"].overall_score
    print(f"\n→ Complexity Score: {complexity_score:.2f}")

    # Step 2: Allocate resources
    allocation = allocate_resources(complexity_score)

    if "error" in allocation:
        print("\n✗ Workflow failed at resource allocation")
        return

    # Step 3: Validate with MAKER
    validation = validate_with_maker(problem, allocation)

    if not validation.get("success"):
        print(f"\n⚠ MAKER validation had issues: {validation['terminated_reason']}")

    # Step 4: Execute on BubbleLab (if available)
    bubblelab_result = execute_on_bubblelab(problem, complexity_score, allocation)

    # Summary
    print("\n" + "=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"Complexity Score: {complexity_score:.2f}")
    print(f"Strategy: {allocation.get('strategy', 'UNKNOWN')}")
    print(f"Agents: {allocation.get('n_agents', 0)}")
    print(f"MAKER Votes: {validation.get('votes_cast', 0)}")
    print(f"Red Flags: {validation.get('red_flags', 0)}")
    print(f"BubbleLab: {'✓ Executed' if bubblelab_result.get('success') else '⚠ Skipped/Failed'}")

    # Get health status
    adapter = get_adapter()
    health = adapter.health_check()

    print(f"\nAdapter Health: {health['status']}")
    print(f"Circuit Breaker: {health['circuit_breaker_state']}")
    print(f"Total Requests: {health['metrics']['requests_total']}")
    print(f"Success Rate: {health['metrics']['requests_success'] / health['metrics']['requests_total'] * 100:.1f}%")


if __name__ == "__main__":
    full_workflow_example()
