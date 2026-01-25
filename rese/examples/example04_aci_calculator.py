"""
Example 4: ACI Calculator (Γ₁)

This example demonstrates how to calculate the Algorithmic Complexity Index (ACI)
for constraint satisfaction problems.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from gamma1.core.aci_calculator import ACICalculator
from gamma1.core.csp_models import CSPInstance, Variable, Constraint

def main():
    print("=" * 60)
    print("Example 4: ACI Calculator (Γ₁)")
    print("=" * 60)
    print()

    # Create a simple CSP instance
    print("Creating CSP Instance:")
    print("-" * 60)

    # Variables: x, y, z with domain {0, 1, 2}
    variables = {
        'x': Variable(name='x', domain=[0, 1, 2]),
        'y': Variable(name='y', domain=[0, 1, 2]),
        'z': Variable(name='z', domain=[0, 1, 2])
    }

    # Constraints: x < y, y < z
    constraints = [
        Constraint(
            id='c1',
            variables=['x', 'y'],
            relation='less_than'
        ),
        Constraint(
            id='c2',
            variables=['y', 'z'],
            relation='less_than'
        )
    ]

    csp = CSPInstance(
        variables=variables,
        domains={v.name: v.domain for v in variables.values()},
        constraints=constraints
    )

    print(f"Variables: {list(variables.keys())}")
    print(f"Domains: {[v.domain for v in variables.values()]}")
    print(f"Constraints: {[c.relation for c in constraints]}")
    print()

    # Calculate ACI
    print("Calculating ACI...")
    print("-" * 60)

    aci_calc = ACICalculator(
        alpha=0.35,  # Weight for (1-H)
        beta=0.35,   # Weight for C
        gamma=0.30,  # Weight for S
        use_cache=True
    )

    result = aci_calc.calculate(csp)

    print()
    print("ACI Result:")
    print("=" * 60)
    print(f"ACI = {result.ACI:.3f}")
    print(f"Confidence = {result.confidence:.2f}")
    print(f"Computation Time = {result.computation_time:.4f}s")
    print(f"From Cache = {result.cached}")
    print()

    print("Component Breakdown:")
    print("-" * 60)
    print(f"H (Disorder Entropy) = {result.components.get('disorder_entropy', 0):.3f}")
    print(f"  Higher = more disordered = harder to solve")
    print()
    print(f"C (Causal Coherence) = {result.components.get('causal_coherence', 0):.3f}")
    print(f"  Higher = more coherent = easier to solve")
    print()
    print(f"S (Solvability Index) = {result.components.get('solvability_index', 0):.3f}")
    print(f"  Higher = more solvable = easier to solve")
    print()

    print("Formula:")
    print("-" * 60)
    alpha, beta, gamma = 0.35, 0.35, 0.30
    H = result.components.get('disorder_entropy', 0)
    C = result.components.get('causal_coherence', 0)
    S = result.components.get('solvability_index', 0)

    print(f"ACI = {alpha}·(1-H) + {beta}·C + {gamma}·S")
    print(f"ACI = {alpha}·(1-{H:.3f}) + {beta}·{C:.3f} + {gamma}·{S:.3f}")
    print(f"ACI = {alpha*(1-H):.3f} + {beta*C:.3f} + {gamma*S:.3f}")
    print(f"ACI = {result.ACI:.3f}")
    print()

    print("Interpretation:")
    print("-" * 60)
    if result.ACI > 0.7:
        difficulty = "Easy"
        color = "🟢"
    elif result.ACI > 0.4:
        difficulty = "Medium"
        color = "🟡"
    else:
        difficulty = "Hard"
        color = "🔴"

    print(f"{color} Difficulty Level: {difficulty}")
    print(f"  ACI > 0.7: Easy (highly solvable)")
    print(f"  ACI 0.4-0.7: Medium (moderately solvable)")
    print(f"  ACI < 0.4: Hard (difficult to solve)")
    print()

    print("Recommendation:")
    print("-" * 60)
    recommendation = result.interpretation.get('recommendation', 'Unknown')
    print(f"{recommendation}")
    print()

    # Compare multiple CSP instances
    print("=" * 60)
    print("Comparing Multiple CSP Instances:")
    print("-" * 60)

    csp_easy = CSPInstance(variables, domains, [])  # No constraints
    csp_medium = CSPInstance(variables, domains, constraints[:1])  # 1 constraint
    csp_hard = CSPInstance(variables, domains, constraints)  # 2 constraints

    aci_easy = aci_calc.calculate(csp_easy)
    aci_medium = aci_calc.calculate(csp_medium)
    aci_hard = aci_calc.calculate(csp_hard)

    print(f"Easy (no constraints):   ACI = {aci_easy.ACI:.3f}")
    print(f"Medium (1 constraint):   ACI = {aci_medium.ACI:.3f}")
    print(f"Hard (2 constraints):    ACI = {aci_hard.ACI:.3f}")
    print()
    print("As expected: More constraints → Lower ACI → Harder to solve")

    print()
    print("=" * 60)
    print("Example 4 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
