"""
Example: Research Problem Decomposition

This example demonstrates semantic decomposition of a research problem.

Problem: Investigate quantum computing applications for drug discovery
"""

from decomposition_engine import SemanticDecomposition
from sovereign_data_models import (
    ProblemDefinition,
    ProblemType,
    DomainContext,
    ComplexityScore
)


def main():
    problem = ProblemDefinition(
        id="research-001",
        title="Quantum Computing for Drug Discovery",
        description="""Investigate and develop quantum computing algorithms that
        can accelerate drug discovery processes:

        Research Goals:
        - Identify quantum algorithms for molecular simulation
        - Compare quantum vs classical approaches
        - Develop proof-of-concept implementations
        - Analyze potential speedups
        - Assess near-term feasibility (NISQ devices)

        Focus Areas:
        - Molecular docking simulations
        - Protein folding predictions
        - Drug-target interaction modeling
        - Optimization of molecular structures

        Constraints:
        - Focus on NISQ-era quantum computers (50-100 qubits)
        - Consider noise resilience requirements
        - Compare against classical ML approaches
        """,
        problem_type=ProblemType.RESEARCH,
        domain_context=DomainContext(
            domain="Quantum Computing",
            subdomain="Molecular Simulation"
        ),
        complexity_score=ComplexityScore(
            overall_complexity=9,
            cognitive_complexity=10,
            computational_complexity=8,
            domain_complexity=10,
            integration_complexity=7
        )
    )

    print("=" * 80)
    print(f"RESEARCH PROBLEM: {problem.title}")
    print(f"Domain: {problem.domain_context.domain}")
    print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    print("=" * 80)

    # Use semantic decomposition (best for research)
    print("\nDecomposing with SEMANTIC strategy...")
    strategy = SemanticDecomposition()
    sub_problems = strategy.decompose(problem)

    print(f"\n[OK] Generated {len(sub_problems)} research sub-problems\n")

    # Display by research phase
    print("RESEARCH PHASES:")
    print("=" * 80)

    phases = {
        'literature': [],
        'algorithm': [],
        'implementation': [],
        'experimentation': [],
        'analysis': []
    }

    for sp in sub_problems:
        desc_lower = sp.description.lower()

        if 'literature' in desc_lower or 'survey' in desc_lower or 'review' in desc_lower:
            phases['literature'].append(sp)
        elif 'algorithm' in desc_lower or 'design' in desc_lower:
            phases['algorithm'].append(sp)
        elif 'implement' in desc_lower or 'develop' in desc_lower or 'code' in desc_lower:
            phases['implementation'].append(sp)
        elif 'experiment' in desc_lower or 'test' in desc_lower or 'validate' in desc_lower:
            phases['experimentation'].append(sp)
        else:
            phases['analysis'].append(sp)

    for phase_name, phase_sps in phases.items():
        if not phase_sps:
            continue

        print(f"\n### {phase_name.upper()} PHASE ###")
        for i, sp in enumerate(phase_sps, 1):
            print(f"\n{i}. {sp.title}")
            print(f"   Priority: {sp.priority}/10")
            print(f"   Focus: {sp.description[:150]}...")

            if sp.potential_approaches:
                print(f"   Approaches:")
                for approach in sp.potential_approaches:
                    if isinstance(approach, dict):
                        print(f"     - {approach.get('name', 'Unknown')}: {approach.get('description', 'N/A')[:80]}")

    print("\n" + "=" * 80)
    print("RESEARCH ROADMAP")
    print("=" * 80)

    # Calculate estimated timeline
    total_effort = sum(sp.estimated_effort for sp in sub_problems)
    print(f"Total Estimated Effort: {total_effort} hours")
    print(f"Estimated Duration: {total_effort / 40:.1f} weeks (assuming 40h/week)")


if __name__ == "__main__":
    main()
