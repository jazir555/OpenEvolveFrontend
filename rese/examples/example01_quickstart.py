"""
Example 1: Quick Start - Your First RESE Pipeline

This example demonstrates the basic usage of RESE to solve a simple optimization problem.

Problem: Optimize delivery routes for 10 locations
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from rese_pipeline import run_rese

def main():
    print("=" * 60)
    print("Example 1: Quick Start - Your First RESE Pipeline")
    print("=" * 60)
    print()

    # Define problem
    problem_description = """
    Optimize delivery routes for 10 locations.
    Minimize total travel distance while visiting all locations.
    """

    constraints = [
        {
            "id": "visit_all",
            "type": "hard",
            "description": "All locations must be visited",
            "formalization": "∀ location ∈ locations: visited(location) = 1"
        },
        {
            "id": "minimize_distance",
            "type": "soft",
            "description": "Minimize total travel distance",
            "formalization": "minimize Σ distance(route[i], route[i+1])"
        },
        {
            "id": "start_depot",
            "type": "hard",
            "description": "Route must start and end at depot",
            "formalization": "route[0] = depot ∧ route[n] = depot"
        }
    ]

    variables = {
        "num_locations": 10,
        "depot": 0,
        "coordinates": [
            (0, 0),    # Depot
            (10, 5),   # Location 1
            (15, 12),  # Location 2
            (8, 18),   # Location 3
            (20, 10),  # Location 4
            (12, 8),   # Location 5
            (5, 15),   # Location 6
            (18, 5),   # Location 7
            (10, 20),  # Location 8
            (22, 15),  # Location 9
            (15, 3)    # Location 10
        ]
    }

    print("Problem Description:")
    print(problem_description)
    print()

    print("Constraints:")
    for constraint in constraints:
        print(f"  - [{constraint['type'].upper()}] {constraint['description']}")
    print()

    print("Variables:")
    for key, value in variables.items():
        print(f"  - {key}: {value}")
    print()

    # Run RESE pipeline
    print("Running RESE Pipeline...")
    print("-" * 60)

    result = run_rese(
        problem_description=problem_description,
        constraints=constraints,
        variables=variables
    )

    # Display results
    print("-" * 60)
    print()
    print("Results:")
    print(f"  Status: {result.status.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Elapsed Time: {result.elapsed_seconds:.2f}s")
    print()

    print("Phase Results:")
    for phase_name, phase_result in result.phase_results.items():
        print(f"  {phase_name}:")
        print(f"    Status: {phase_result.status.value}")
        print(f"    Time: {phase_result.elapsed_seconds:.2f}s")
        print(f"    Metrics: {phase_result.metrics}")
    print()

    print("ACI History:")
    print(f"  {result.aci_history}")
    if len(result.aci_history) > 1:
        reduction = result.aci_history[0] - result.aci_history[-1]
        print(f"  ACI Reduction: {reduction:.2f} ({reduction/aci_history[0]*100:.1f}%)")
    print()

    if result.final_solution:
        print("Solution Found:")
        print(f"  {result.final_solution}")
    print()

    print("=" * 60)
    print("Example 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
