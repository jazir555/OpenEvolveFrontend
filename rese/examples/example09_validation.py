"""
Example 9: Solution Validation (Δ₃)

This example demonstrates how to validate solutions using ACI reduction
and statistical testing.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from phase4.types import Problem
from phase4.aci_reduction_validator (
    Delta3Validator,
    Problem,
    RESESolution
)
from gamma1.core.aci_calculator import ACICalculator

def main():
    print("=" * 60)
    print("Example 9: Solution Validation (Δ₃)")
    print("=" * 60)
    print()

    # Create a problem
    print("Creating Problem:")
    print("-" * 60)

    problem = Problem(
        id="tsp_10",
        description="10-city Traveling Salesman Problem",
        constraints=[
            {
                "id": "visit_all",
                "type": "hard",
                "description": "Visit all cities",
                "variables": ["cities", "route"]
            },
            {
                "id": "minimize_distance",
                "type": "soft",
                "description": "Minimize distance",
                "variables": ["route", "distances"]
            }
        ],
        variables={
            "num_cities": 10,
            "coordinates": [(i, i*2) for i in range(10)]
        }
    )

    print(f"Problem ID: {problem.id}")
    print(f"Description: {problem.description}")
    print(f"Constraints: {len(problem.constraints)}")
    print(f"Variables: {list(problem.variables.keys())}")
    print()

    # Create a solution
    print("Creating Solution:")
    print("-" * 60)

    # Simulate RESE pipeline results
    aci_history = [0.85, 0.70, 0.55, 0.40]  # Decreasing ACI

    solution = RESESolution(
        problem_id=problem.id,
        solution={
            "route": [0, 2, 4, 6, 8, 9, 7, 5, 3, 1, 0],
            "total_distance": 42.5,
            "method": "rese_pipeline"
        },
        aci_history=aci_history,
        stage_results={
            "phase1": {"status": "completed", "time": 2.3},
            "phase2": {"status": "completed", "time": 5.1},
            "phase3": {"status": "completed", "time": 12.7},
            "phase4": {"status": "completed", "time": 3.2}
        }
    )

    print(f"Solution Route: {solution.solution['route']}")
    print(f"Total Distance: {solution.solution['total_distance']}")
    print(f"ACI History: {solution.aci_history}")
    print()

    # Validate solution
    print("=" * 60)
    print("Validating Solution:")
    print("-" * 60)

    validator = Delta3Validator(
        validation_threshold=0.7,
        min_aci_reduction=0.2,
        holdout_ratio=0.2
    )

    validation_result = validator.validate(problem, solution)

    print()
    print("Validation Results:")
    print("-" * 60)
    print(f"Valid: {validation_result.is_valid}")
    print(f"Validation Score: {validation_result.validation_score:.2f}")
    print(f"Confidence: {validation_result.confidence:.2f}")
    print()

    # ACI Reduction Analysis
    print("ACI Reduction Analysis:")
    print("-" * 60)
    initial_aci = solution.aci_history[0]
    final_aci = solution.aci_history[-1]
    aci_reduction = initial_aci - final_aci
    aci_reduction_pct = (aci_reduction / initial_aci) * 100

    print(f"Initial ACI: {initial_aci:.3f}")
    print(f"Final ACI: {final_aci:.3f}")
    print(f"ACI Reduction: {aci_reduction:.3f} ({aci_reduction_pct:.1f}%)")
    print()

    if aci_reduction >= 0.2:
        print("✓ Meets minimum ACI reduction requirement (20%)")
    else:
        print("✗ Below minimum ACI reduction requirement")
    print()

    # Statistical Significance
    if validation_result.statistical_significance:
        print("Statistical Significance:")
        print(f"  P-value: {validation_result.statistical_significance:.4f}")

        if validation_result.statistical_significance < 0.05:
            print("  ✓ Statistically significant (p < 0.05)")
        else:
            print("  ✗ Not statistically significant (p >= 0.05)")
    print()

    # Phase-wise results
    print("Phase-wise Results:")
    print("-" * 60)

    for stage, result in solution.stage_results.items():
        print(f"{stage}:")
        print(f"  Status: {result['status']}")
        print(f"  Time: {result['time']:.1f}s")
    print()

    # Recommendations
    print("=" * 60)
    print("Recommendations:")
    print("-" * 60)

    if validation_result.is_valid:
        print("✓ Solution is VALID and recommended for deployment")
        print()
        print("  The solution demonstrates:")
        print("  - Significant ACI reduction (>20%)")
        print("  - High validation score (>0.7)")
        print("  - Statistical significance (p < 0.05)")
    else:
        print("✗ Solution is NOT VALID")
        print()
        print("  Possible issues:")
        if validation_result.validation_score < 0.7:
            print("  - Validation score below threshold")
        if aci_reduction < 0.2:
            print("  - Insufficient ACI reduction")
        if validation_result.statistical_significance >= 0.05:
            print("  - Not statistically significant")

    if validation_result.errors:
        print()
        print("  Errors:")
        for error in validation_result.errors:
            print(f"    - {error}")

    print()

    # Comparison with different solutions
    print("=" * 60)
    print("Comparing Multiple Solutions:")
    print("-" * 60)

    solutions_to_compare = [
        ("Solution A", [0.85, 0.70, 0.55, 0.40]),
        ("Solution B", [0.85, 0.75, 0.65, 0.55]),
        ("Solution C", [0.85, 0.80, 0.75, 0.70])
    ]

    print("Solution Comparison:")
    print(f"{'Solution':<15} {'Init ACI':<10} {'Final ACI':<10} {'Reduction':<12} {'Score':<10}")
    print("-" * 60)

    for name, aci_hist in solutions_to_compare:
        sol = RESESolution(
            problem_id=problem.id,
            solution={"route": []},
            aci_history=aci_hist,
            stage_results={}
        )

        result = validator.validate(problem, sol)
        init = aci_hist[0]
        final = aci_hist[-1]
        reduction = ((init - final) / init) * 100

        print(f"{name:<15} {init:<10.3f} {final:<10.3f} {reduction:<11.1f}% {result.validation_score:<10.2f}")

    print()
    print("Solution A has best ACI reduction and highest validation score")

    print()
    print("=" * 60)
    print("Example 9 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
