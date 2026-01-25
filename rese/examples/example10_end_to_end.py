"""
Example 10: End-to-End Pipeline with Custom Problem

This example demonstrates a complete end-to-end RESE pipeline
for solving a custom optimization problem.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from rese_pipeline import RESEPipeline, ProblemInput
from config import RESEConfig, Phase1Config, Phase3Config

def main():
    print("=" * 70)
    print("Example 10: End-to-End Pipeline with Custom Problem")
    print("=" * 70)
    print()

    # Define custom problem: Resource Allocation
    print("Problem Definition: Resource Allocation")
    print("=" * 70)
    print()

    problem_description = """
    Allocate limited resources to multiple projects to maximize value
    while respecting constraints.

    - 5 projects competing for resources
    - 3 types of resources (budget, personnel, equipment)
    - Each project has resource requirements and value
    - Total resources limited
    """

    print(problem_description)

    # Define constraints
    constraints = [
        {
            "id": "budget_constraint",
            "type": "hard",
            "description": "Total budget allocation must not exceed $1M",
            "formalization": "Σ budget[i] ≤ 1000000",
            "variables": ["budget_allocations"]
        },
        {
            "id": "personnel_constraint",
            "type": "hard",
            "description": "Total personnel must not exceed 50 people",
            "formalization": "Σ personnel[i] ≤ 50",
            "variables": ["personnel_allocations"]
        },
        {
            "id": "equipment_constraint",
            "type": "hard",
            "description": "Total equipment must not exceed 20 units",
            "formalization": "Σ equipment[i] ≤ 20",
            "variables": ["equipment_allocations"]
        },
        {
            "id": "min_allocation",
            "type": "soft",
            "description": "Each project should receive minimum resources",
            "formalization": "budget[i] ≥ 50000",
            "variables": ["budget_allocations"]
        },
        {
            "id": "maximize_value",
            "type": "soft",
            "description": "Maximize total project value",
            "formalization": "maximize Σ value[i]",
            "variables": ["project_values"]
        }
    ]

    print("Constraints:")
    print("-" * 70)
    for i, constraint in enumerate(constraints, 1):
        print(f"{i}. [{constraint['type'].upper()}] {constraint['description']}")
        print(f"   {constraint['formalization']}")
    print()

    # Define variables
    variables = {
        "num_projects": 5,
        "project_requirements": [
            {"budget": 300000, "personnel": 15, "equipment": 5, "value": 500},
            {"budget": 200000, "personnel": 10, "equipment": 4, "value": 350},
            {"budget": 250000, "personnel": 12, "equipment": 6, "value": 400},
            {"budget": 150000, "personnel": 8, "equipment": 3, "value": 250},
            {"budget": 400000, "personnel": 20, "equipment": 8, "value": 600}
        ],
        "total_budget": 1000000,
        "total_personnel": 50,
        "total_equipment": 20
    }

    print("Variables:")
    print("-" * 70)
    print(f"Number of Projects: {variables['num_projects']}")
    print(f"Total Budget: ${variables['total_budget']:,}")
    print(f"Total Personnel: {variables['total_personnel']}")
    print(f"Total Equipment: {variables['total_equipment']}")
    print()

    print("Project Requirements:")
    for i, req in enumerate(variables['project_requirements'], 1):
        print(f"  Project {i}: Budget=${req['budget']:,}, "
              f"Personnel={req['personnel']}, "
              f"Equipment={req['equipment']}, "
              f"Value=${req['value']}")
    print()

    # Create problem input
    problem = ProblemInput(
        id="resource_allocation_5",
        description="Resource allocation for 5 projects",
        constraints=constraints,
        variables=variables,
        objective="Maximize total value while respecting resource constraints"
    )

    # Configure pipeline
    print("=" * 70)
    print("Pipeline Configuration")
    print("=" * 70)
    print()

    config = RESEConfig(
        environment="development",
        phase1=Phase1Config(
            sce_max_constraints=100,
            phi15_assumption_threshold=0.6,
            phi2_bias_threshold=0.5
        ),
        phase3=Phase3Config(
            gamma2_iterations=500,
            convergence_patience=50
        )
    )

    print("Configuration Settings:")
    print(f"  Environment: {config.environment}")
    print(f"  Phase I max constraints: {config.phase1.sce_max_constraints}")
    print(f"  Phase III iterations: {config.phase3.gamma2_iterations}")
    print()

    # Run pipeline
    print("=" * 70)
    print("Running RESE Pipeline")
    print("=" * 70)
    print()

    pipeline = RESEPipeline(config)

    # Add progress callback
    def progress_callback(result):
        print(f"[Progress] Status: {result.status.value}, "
              f"Phases: {len(result.phase_results)}/4")

    pipeline.add_progress_callback(progress_callback)

    print("Starting pipeline execution...")
    print()

    result = pipeline.run(problem)

    print()
    print("=" * 70)
    print("Pipeline Results")
    print("=" * 70)
    print()

    # Overall results
    print("Overall Status:")
    print(f"  Pipeline Status: {result.status.value}")
    print(f"  Total Time: {result.elapsed_seconds:.2f}s")
    print(f"  Final Confidence: {result.confidence:.2f}")
    print()

    # Phase results
    print("Phase-wise Results:")
    print("-" * 70)

    for phase_name, phase_result in result.phase_results.items():
        print(f"\n{phase_name.upper()}:")
        print(f"  Status: {phase_result.status.value}")
        print(f"  Time: {phase_result.elapsed_seconds:.2f}s")

        if phase_result.metrics:
            print(f"  Metrics:")
            for key, value in phase_result.metrics.items():
                print(f"    {key}: {value}")

        if phase_result.errors:
            print(f"  Errors:")
            for error in phase_result.errors:
                print(f"    - {error}")

        if phase_result.warnings:
            print(f"  Warnings:")
            for warning in phase_result.warnings:
                print(f"    - {warning}")

    print()

    # ACI analysis
    print("=" * 70)
    print("ACI Analysis")
    print("=" * 70)
    print()

    if result.aci_history:
        print(f"ACI Progression:")
        for i, aci in enumerate(result.aci_history):
            phase_names = ["Initial", "After Phase I", "After Phase II",
                          "After Phase III", "After Phase IV"]
            print(f"  {phase_names[i]:<20} ACI = {aci:.3f}")

        if len(result.aci_history) > 1:
            initial = result.aci_history[0]
            final = result.aci_history[-1]
            reduction = initial - final
            reduction_pct = (reduction / initial) * 100

            print()
            print(f"ACI Reduction: {reduction:.3f} ({reduction_pct:.1f}%)")

            if reduction > 0.3:
                print("✓ Excellent ACI reduction (>30%)")
            elif reduction > 0.2:
                print("✓ Good ACI reduction (>20%)")
            elif reduction > 0.1:
                print("⚠ Moderate ACI reduction (>10%)")
            else:
                print("✗ Low ACI reduction (<10%)")

    print()

    # Final solution
    print("=" * 70)
    print("Final Solution")
    print("=" * 70)
    print()

    if result.final_solution:
        print("Recommended Resource Allocation:")
        print()

        # Parse solution (example format)
        if 'allocations' in result.final_solution:
            allocations = result.final_solution['allocations']

            print(f"{'Project':<12} {'Budget':<15} {'Personnel':<12} {'Equipment':<12} {'Value':<10}")
            print("-" * 70)

            total_value = 0
            for i, alloc in enumerate(allocations, 1):
                budget = alloc.get('budget', 0)
                personnel = alloc.get('personnel', 0)
                equipment = alloc.get('equipment', 0)
                value = alloc.get('value', 0)
                total_value += value

                print(f"Project {i:<6} ${budget:>12,.0f}  {personnel:>10}      {equipment:>10}      ${value:>8,.0f}")

            print("-" * 70)
            print(f"{'TOTAL':<12} ${sum(a['budget'] for a in allocations):>12,.0f}  "
                  f"{sum(a['personnel'] for a in allocations):>10}      "
                  f"{sum(a['equipment'] for a in allocations):>10}      "
                  f"${total_value:>8,.0f}")
        else:
            print(f"Solution: {result.final_solution}")

    print()

    # Validation
    if result.validation_score:
        print("Validation:")
        print(f"  Validation Score: {result.validation_score:.2f}")

        if result.validation_score >= 0.8:
            print("  ✓ High confidence solution")
        elif result.validation_score >= 0.7:
            print("  ✓ Acceptable solution")
        else:
            print("  ⚠ Low confidence - review recommended")

    print()

    # Recommendations
    print("=" * 70)
    print("Recommendations")
    print("=" * 70)
    print()

    if result.confidence > 0.8:
        print("1. ✓ Solution has high confidence - recommended for implementation")
    elif result.confidence > 0.6:
        print("1. ⚠ Solution has moderate confidence - consider sensitivity analysis")
    else:
        print("1. ✗ Solution has low confidence - additional refinement needed")

    if result.aci_history and (result.aci_history[0] - result.aci_history[-1]) > 0.3:
        print("2. ✓ Strong ACI reduction indicates effective problem transformation")

    if len(result.errors) == 0:
        print("3. ✓ Pipeline executed without errors")
    else:
        print(f"3. ⚠ Pipeline had {len(result.errors)} errors - review logs")

    print()
    print("=" * 70)
    print("Example 10 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
