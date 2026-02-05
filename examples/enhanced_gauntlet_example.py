"""
Enhanced Gauntlet Example: LoongFlow Integration

This example demonstrates how to use the enhanced gauntlet system with
LoongFlow integration for 3-round validation of solutions.

The 3 rounds are:
1. LoongFlow AI Evaluation (quick quality screen)
2. Red Team Attack (adversarial testing)
3. Gold Team Verification (consensus approval)
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SolutionAttempt:
    """Sample solution attempt for demonstration."""

    def __init__(self, solution_id: str, content: str):
        self.id = solution_id
        self.content = content
        self.solution_content = content
        self.status = "generated"
        self.timestamp = datetime.now().timestamp()


async def example_1_basic_loongflow_evaluation():
    """
    Example 1: Basic LoongFlow evaluation

    Demonstrates using the LoongFlow adapter directly for quick evaluation.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic LoongFlow Evaluation")
    print("="*70 + "\n")

    from evaluators.loongflow_adapter import create_loongflow_evaluator

    # Configure LLM
    llm_config = {
        'model': 'claude-3-5-sonnet-20241022',
        'api_key': 'your-api-key',  # Replace with actual key
        'url': 'http://localhost:8001',
        'temperature': 0.3,
        'max_tokens': 4096
    }

    # Create adapter (with fallback enabled for this example)
    adapter = create_loongflow_evaluator(
        llm_config=llm_config,
        timeout=60,
        enable_loongflow=False  # Using fallback for demo
    )

    # Create a solution
    solution = SolutionAttempt(
        solution_id="sol_001",
        content="""
        # Algorithm Implementation

        ## Problem
        Need to implement efficient sorting algorithm.

        ## Approach
        Using quicksort with median-of-three pivot selection for average
        case O(n log n) performance.

        ## Implementation
        ```python
        def quicksort(arr):
            if len(arr) <= 1:
                return arr

            pivot = median_of_three(arr)
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]

            return quicksort(left) + middle + quicksort(right)

        def median_of_three(arr):
            """Select median of first, middle, last elements."""
            first, middle, last = arr[0], arr[len(arr)//2], arr[-1]
            return sorted([first, middle, last])[1]
        ```

        ## Analysis
        - Time Complexity: O(n log n) average, O(n²) worst case
        - Space Complexity: O(log n) for recursion stack
        - Stability: Not stable (but can be made stable)

        This approach works well because median-of-three pivot selection
        reduces the likelihood of worst-case performance on sorted or
        nearly-sorted input.
        """
    )

    # Create round rule
    class RoundRule:
        def __init__(self):
            self.rule_id = "quality_check"
            self.min_score = 0.7

    round_rule = RoundRule()

    # Set context
    context = {
        'problem': 'Implement efficient sorting algorithm',
        'criteria': ['correctness', 'efficiency', 'clarity', 'completeness'],
        'trace_id': 'example_1'
    }

    # Evaluate
    print("Evaluating solution with LoongFlow adapter...")
    result = await adapter.evaluate_round(
        solution=solution,
        round_rule=round_rule,
        context=context
    )

    # Display results
    print("\n" + "-"*70)
    print("EVALUATION RESULTS")
    print("-"*70)
    print(f"Rule ID:        {result.rule_id}")
    print(f"Passed:         {result.passed}")
    print(f"Score:          {result.score:.3f}")
    print(f"Feedback:       {result.feedback}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"\nDetails:")
    for key, value in result.details.items():
        print(f"  {key}: {value}")


async def example_2_batch_evaluation():
    """
    Example 2: Batch evaluation of multiple solutions

    Demonstrates evaluating multiple solutions in parallel.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Evaluation")
    print("="*70 + "\n")

    from evaluators.loongflow_adapter import create_loongflow_evaluator

    llm_config = {
        'model': 'claude-3-5-sonnet-20241022',
        'api_key': 'your-api-key',
        'url': 'http://localhost:8001'
    }

    adapter = create_loongflow_evaluator(
        llm_config=llm_config,
        timeout=30,
        enable_loongflow=False  # Fallback for demo
    )

    # Create multiple solutions
    solutions = [
        SolutionAttempt(f"sol_{i:03d}", f"Solution {i}\n" + "Content " * (i * 20))
        for i in range(1, 6)
    ]

    class RoundRule:
        def __init__(self):
            self.rule_id = "batch_check"
            self.min_score = 0.6

    context = {
        'problem': 'Generate solution approaches',
        'criteria': ['creativity', 'feasibility']
    }

    # Batch evaluate
    print(f"Evaluating {len(solutions)} solutions in parallel...")
    results = await adapter.batch_evaluate(
        solutions=solutions,
        round_rule=RoundRule(),
        context=context
    )

    # Display results
    print("\n" + "-"*70)
    print("BATCH EVALUATION RESULTS")
    print("-"*70)
    for i, result in enumerate(results, 1):
        print(f"\nSolution {i}:")
        print(f"  Passed:  {result.passed}")
        print(f"  Score:   {result.score:.3f}")
        print(f"  Time:    {result.execution_time:.2f}s")


async def example_3_enhanced_gauntlet():
    """
    Example 3: Complete 3-round enhanced gauntlet

    Demonstrates the full enhanced gauntlet system with LoongFlow integration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Enhanced 3-Round Gauntlet")
    print("="*70 + "\n")

    from enhanced_gauntlet_manager import create_enhanced_gauntlet_system

    # Configure system
    llm_config = {
        'model': 'claude-3-5-sonnet-20241022',
        'api_key': 'your-api-key',
        'url': 'http://localhost:8001',
        'temperature': 0.3
    }

    system = create_enhanced_gauntlet_system(
        llm_config=llm_config,
        enable_loongflow=False  # Using fallback for demo
    )

    # Create gauntlet for engineering problem
    gauntlet = system.create_enhanced_gauntlet(
        problem_type="engineering",
        strictness="standard"
    )

    print(f"Created gauntlet: {gauntlet.name}")
    print(f"Description: {gauntlet.description}")
    print(f"\nRounds ({len(gauntlet.rounds)}):")
    for i, round_rule in enumerate(gauntlet.rounds, 1):
        print(f"  Round {i}: {round_rule.rule_id}")
        print(f"    Type: {round_rule.rule_type}")
        print(f"    Min Score: {round_rule.min_overall_confidence}")
        print(f"    Description: {round_rule.description}")

    # Create solution
    solution = SolutionAttempt(
        solution_id="eng_solution_001",
        content="""
        # Bridge Design Solution

        ## Requirements
        - Span: 100 meters
        - Load capacity: 50 tons
        - Material: Steel
        - Budget constraint: $2M

        ## Design Approach: Truss Bridge

        Selected a Pratt truss design for optimal strength-to-weight ratio.

        ## Structural Analysis

        ### Load Calculations
        - Dead load: 200 tons (bridge self-weight)
        - Live load: 50 tons (vehicles)
        - Safety factor: 1.5
        - Total design load: 375 tons

        ### Material Selection
        - Main chords: W14x90 steel I-beams
        - Vertical members: W10x45 steel I-beams
        - Diagonal members: W12x50 steel I-beams
        - Steel grade: A992 (Fy = 50 ksi)

        ## Implementation
        ```python
        def calculate_member_force(load, angle):
            \"\"\"
            Calculate axial force in truss member.

            Args:
                load: Point load in kips
                angle: Member angle from horizontal (degrees)

            Returns:
                Tension/compression force in kips
            \"\"\"
            import math

            force = load / (2 * math.sin(math.radians(angle)))
            return force

        def design_truss_span(span_length, num_panels=10):
            \"\"\"
            Design truss dimensions for given span.

            Returns panel length and member forces.
            \"\"\"
            panel_length = span_length / num_panels
            # Force calculations...
            return panel_length, member_forces
        ```

        ## Cost Analysis
        - Steel cost: $800/ton × 200 tons = $160,000
        - Fabrication: $400,000
        - Labor: $600,000
        - Foundation: $300,000
        - Contingency (15%): $217,500
        - **Total: $1,677,500** (within $2M budget)

        ## Validation
        [OK] Meets span requirement
        [OK] Exceeds load capacity with safety factor
        [OK] Within budget constraints
        [OK] Uses standard materials
        [OK] Proven design methodology
        """
    )

    # Set evaluation context
    context = {
        'problem': 'Design a 100m span bridge with 50-ton capacity',
        'criteria': ['safety', 'efficiency', 'cost_effectiveness', 'constructability'],
        'trace_id': 'gauntlet_example_3',
        'workspace_dir': '/tmp/gauntlet_eval'
    }

    # Execute gauntlet
    print("\n" + "-"*70)
    print("EXECUTING GAUNTLET")
    print("-"*70 + "\n")

    execution = await system.execute_gauntlet(
        gauntlet=gauntlet,
        solution=solution,
        context=context
    )

    # Display results
    print("\n" + "-"*70)
    print("GAUNTLET EXECUTION RESULTS")
    print("-"*70)
    print(f"Gauntlet ID:    {execution.gauntlet_id}")
    print(f"Solution ID:    {execution.solution_id}")
    print(f"Overall Passed: {execution.overall_passed}")
    print(f"Final Score:    {execution.final_score:.3f}")
    print(f"Execution Time: {execution.execution_time:.2f}s")
    print(f"\nRounds Passed:  {len(execution.rounds_passed)}/{len(execution.rounds_results)}")
    print(f"Rounds Failed:  {len(execution.rounds_failed)}/{len(execution.rounds_results)}")

    print("\nRound-by-Round Results:")
    for i, round_result in enumerate(execution.rounds_results, 1):
        status_symbol = "[OK]" if round_result.status.value == "passed" else "[FAIL]"
        print(f"\n  Round {i}: {round_result.rule_id} {status_symbol}")
        print(f"    Status:   {round_result.status.value}")
        print(f"    Score:    {round_result.score:.3f}")
        print(f"    Feedback: {round_result.feedback[:100]}...")
        print(f"    Time:     {round_result.execution_time:.2f}s")


async def example_4_strictness_levels():
    """
    Example 4: Comparing different strictness levels

    Demonstrates how strictness affects gauntlet thresholds.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Strictness Level Comparison")
    print("="*70 + "\n")

    from enhanced_gauntlet_manager import create_enhanced_gauntlet_system

    llm_config = {
        'model': 'claude-3-5-sonnet-20241022',
        'api_key': 'your-api-key',
        'url': 'http://localhost:8001'
    }

    system = create_enhanced_gauntlet_system(
        llm_config=llm_config,
        enable_loongflow=False
    )

    # Create gauntlets with different strictness
    strictness_levels = ["lenient", "standard", "strict"]

    print("Comparing strictness levels:\n")

    for strictness in strictness_levels:
        gauntlet = system.create_enhanced_gauntlet(
            problem_type="security",
            strictness=strictness
        )

        print(f"{strictness.capitalize()} Gauntlet:")
        print(f"  Round 1 (LoongFlow):   {gauntlet.rounds[0].min_overall_confidence:.2f}")
        print(f"  Round 2 (Red Team):    {gauntlet.rounds[1].min_overall_confidence:.2f}")
        print(f"  Round 3 (Gold Team):   {gauntlet.rounds[2].min_overall_confidence:.2f}")
        print()


async def example_5_domain_specific_gauntlets():
    """
    Example 5: Domain-specific gauntlets

    Demonstrates creating gauntlets for different problem domains.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Domain-Specific Gauntlets")
    print("="*70 + "\n")

    from enhanced_gauntlet_manager import create_enhanced_gauntlet_system

    system = create_enhanced_gauntlet_system(
        llm_config={'model': 'claude-3-5-sonnet-20241022', 'api_key': 'test'},
        enable_loongflow=False
    )

    domains = [
        ("trading", "Financial trading algorithm"),
        ("engineering", "Engineering design problem"),
        ("security", "Security system"),
        ("scientific", "Scientific research"),
        ("finance", "Financial analysis")
    ]

    print("Domain-specific attack modes:\n")

    for domain, description in domains:
        gauntlet = system.create_enhanced_gauntlet(
            problem_type=domain,
            strictness="standard"
        )

        attack_modes = gauntlet.attack_modes

        print(f"{domain.upper()} ({description}):")
        print(f"  Attack Modes: {', '.join(attack_modes[:3])}{'...' if len(attack_modes) > 3 else ''}")
        print()


async def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ENHANCED GAUNTLET SYSTEM - LOONGFLOW INTEGRATION EXAMPLES".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    try:
        # Example 1: Basic evaluation
        await example_1_basic_loongflow_evaluation()

        # Example 2: Batch evaluation
        await example_2_batch_evaluation()

        # Example 3: Complete gauntlet
        await example_3_enhanced_gauntlet()

        # Example 4: Strictness comparison
        await example_4_strictness_levels()

        # Example 5: Domain-specific gauntlets
        await example_5_domain_specific_gauntlets()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print(f"\n[FAIL] Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
