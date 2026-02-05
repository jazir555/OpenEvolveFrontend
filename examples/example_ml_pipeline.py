"""
Example: ML Pipeline Decomposition

This example demonstrates how to decompose a complex machine learning
pipeline using the OpenEvolve decomposition engine.

Problem: Build a complete ML pipeline for fraud detection
"""

from decomposition_engine import DecompositionEngine, SemanticDecomposition
from problem_analyzer import ProblemAnalyzer
from sovereign_data_models import (
    ProblemDefinition,
    ProblemType,
    DomainContext,
    ComplexityScore,
    SubProblemType
)
import json


def main():
    """Main execution function"""

    # ============================================================================
    # STEP 1: Define the Problem
    # ============================================================================

    problem = ProblemDefinition(
        id="ml-pipeline-001",
        title="Real-Time Fraud Detection ML Pipeline",
        description="""Build a production-grade machine learning pipeline for detecting
        fraudulent financial transactions in real-time.

        Business Requirements:
        - Process 1000 transactions/second
        - Latency < 100ms per transaction
        - Detect fraud with 99%+ accuracy
        - False positive rate < 1%
        - Handle 10M+ transactions daily
        - Real-time model updates

        Technical Requirements:
        - Stream processing architecture
        - Feature engineering pipeline
        - Model training and validation
        - Model serving infrastructure
        - Monitoring and alerting
        - A/B testing framework
        - Data quality checks
        - Explainability and reporting

        Data Sources:
        - Transaction logs (JSON, 1K/sec)
        - User profiles (PostgreSQL)
        - Historical fraud cases (S3)
        - External risk APIs (REST)

        Constraints:
        - Must comply with PCI-DSS
        - Must explain fraud decisions
        - Must handle model drift
        - Must support multiple fraud types
        """,
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(
            domain="Machine Learning",
            subdomain="Fraud Detection"
        ),
        complexity_score=ComplexityScore(
            overall_complexity=9,
            cognitive_complexity=8,
            computational_complexity=10,
            domain_complexity=9,
            integration_complexity=9
        )
    )

    print("=" * 80)
    print(f"PROBLEM: {problem.title}")
    print(f"Domain: {problem.domain_context.domain} / {problem.domain_context.subdomain}")
    print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    print("=" * 80)

    # ============================================================================
    # STEP 2: Analyze the Problem
    # ============================================================================

    print("\n[STEP 1] Analyzing problem...")
    analyzer = ProblemAnalyzer()
    analysis = analyzer.analyze_problem(problem)

    print(f"\nDomain: {analysis['domain']}")
    print(f"Complexity Breakdown:")
    for aspect, score in analysis['complexity'].items():
        print(f"  - {aspect.capitalize()}: {score}/10")

    print(f"\nEstimated Sub-Problems: {analysis['estimated_sub_problems']}")

    print(f"\nRequired Expertise:")
    for exp in analysis['required_expertise']:
        print(f"  - {exp}")

    print(f"\nKey Challenges:")
    for challenge in analysis['key_challenges']:
        print(f"  - {challenge}")

    # ============================================================================
    # STEP 3: Decompose the Problem
    # ============================================================================

    print("\n[STEP 2] Decomposing problem (Semantic Strategy)...")
    engine = DecompositionEngine()
    result = engine.decompose(problem, strategy="semantic")

    print(f"\n[OK] Generated {len(result.sub_problems)} sub-problems\n")

    # ============================================================================
    # STEP 4: Display Decomposition Results
    # ============================================================================

    # Group by type
    by_type = {
        'research': [],
        'analysis': [],
        'implementation': [],
        'validation': [],
        'integration': []
    }

    for sp in result.sub_problems:
        by_type[sp.type.value].append(sp)

    print("=" * 80)
    print("DECOMPOSITION BY TYPE")
    print("=" * 80)

    for sp_type, sp_list in by_type.items():
        if not sp_list:
            continue

        print(f"\n### {sp_type.upper()} ({len(sp_list)}) ###\n")

        for i, sp in enumerate(sp_list, 1):
            print(f"{i}. {sp.title}")
            print(f"   Priority: {sp.priority}/10 | Effort: {sp.estimated_effort}h | Complexity: {sp.complexity_score.overall_complexity}/10")

            if sp.dependencies:
                deps = ", ".join(sp.dependencies)
                print(f"   Dependencies: {deps}")
            else:
                print(f"   Dependencies: None (can start immediately)")

            # Show acceptance criteria
            if sp.acceptance_criteria:
                print(f"   Acceptance Criteria:")
                for criterion in sp.acceptance_criteria[:2]:  # Show first 2
                    print(f"     - {criterion}")
                if len(sp.acceptance_criteria) > 2:
                    print(f"     - ... and {len(sp.acceptance_criteria) - 2} more")

            # Show required expertise
            if sp.required_expertise:
                expertise = ", ".join(sp.required_expertise[:3])
                print(f"   Expertise: {expertise}")

            print()

    # ============================================================================
    # STEP 5: Show Execution Plan
    # ============================================================================

    print("=" * 80)
    print("SUGGESTED EXECUTION PLAN")
    print("=" * 80)

    # Calculate execution order (topological sort)
    executed = set()
    order = []

    while len(executed) < len(result.sub_problems):
        ready = [
            sp for sp in result.sub_problems
            if sp.id not in executed and
            all(d in executed for d in sp.dependencies)
        ]

        if not ready:
            print("Warning: Circular dependency detected!")
            break

        # Sort by priority
        ready.sort(key=lambda x: x.priority, reverse=True)
        order.extend(ready)
        for sp in ready:
            executed.add(sp.id)

    # Print phases
    phase = 1
    phase_tasks = []
    current_effort = 0
    max_phase_effort = 40  # Hours per phase

    for sp in order:
        if current_effort + sp.estimated_effort > max_phase_effort and phase_tasks:
            print(f"\nPhase {phase}: {len(phase_tasks)} tasks (~{current_effort}h)")
            for task in phase_tasks:
                print(f"  - {task.title} ({task.estimated_effort}h, priority {task.priority}/10)")
            phase += 1
            phase_tasks = []
            current_effort = 0

        phase_tasks.append(sp)
        current_effort += sp.estimated_effort

    # Print final phase
    if phase_tasks:
        print(f"\nPhase {phase}: {len(phase_tasks)} tasks (~{current_effort}h)")
        for task in phase_tasks:
            print(f"  - {task.title} ({task.estimated_effort}h, priority {task.priority}/10)")

    # ============================================================================
    # STEP 6: Quality Metrics
    # ============================================================================

    print("\n" + "=" * 80)
    print("QUALITY METRICS")
    print("=" * 80)

    # Effort distribution
    efforts = [sp.estimated_effort for sp in result.sub_problems]
    total_effort = sum(efforts)
    avg_effort = sum(efforts) / len(efforts)

    print(f"\nTotal Estimated Effort: {total_effort} hours ({total_effort/8:.1f} days)")
    print(f"Average per Sub-Problem: {avg_effort:.1f} hours")
    print(f"Min/Max Effort: {min(efforts)}h / {max(efforts)}h")

    # Complexity distribution
    complexities = [sp.complexity_score.overall_complexity for sp in result.sub_problems]
    avg_complexity = sum(complexities) / len(complexities)

    print(f"\nAverage Complexity: {avg_complexity:.1f}/10")
    print(f"Complexity Range: {min(complexities)} - {max(complexities)}")

    # Parallelization potential
    no_deps = sum(1 for sp in result.sub_problems if not sp.dependencies)
    parallelizable_pct = (no_deps / len(result.sub_problems)) * 100

    print(f"\nParallelizable Tasks: {no_deps}/{len(result.sub_problems)} ({parallelizable_pct:.1f}%)")

    # ============================================================================
    # STEP 7: Risk Assessment
    # ============================================================================

    print("\n" + "=" * 80)
    print("RISK ASSESSMENT")
    print("=" * 80)

    high_complexity = [sp for sp in result.sub_problems if sp.complexity_score.overall_complexity >= 8]
    high_effort = [sp for sp in result.sub_problems if sp.estimated_effort >= 24]
    many_deps = [sp for sp in result.sub_problems if len(sp.dependencies) >= 3]

    print(f"\nHigh Complexity Tasks (≥8/10): {len(high_complexity)}")
    for sp in high_complexity:
        print(f"  [WARN]  {sp.title} (complexity: {sp.complexity_score.overall_complexity}/10)")

    print(f"\nHigh Effort Tasks (≥24h): {len(high_effort)}")
    for sp in high_effort:
        print(f"  [WARN]  {sp.title} (effort: {sp.estimated_effort}h)")

    print(f"\nMany Dependencies (≥3): {len(many_deps)}")
    for sp in many_deps:
        print(f"  [WARN]  {sp.title} (dependencies: {len(sp.dependencies)})")

    # ============================================================================
    # STEP 8: Export Results
    # ============================================================================

    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)

    # Export to JSON
    export_data = {
        'problem': {
            'id': problem.id,
            'title': problem.title,
            'description': problem.description,
            'domain': problem.domain_context.domain,
            'complexity': problem.complexity_score.overall_complexity
        },
        'sub_problems': [
            {
                'id': sp.id,
                'title': sp.title,
                'type': sp.type.value,
                'priority': sp.priority,
                'effort_hours': sp.estimated_effort,
                'complexity': sp.complexity_score.overall_complexity,
                'dependencies': sp.dependencies,
                'acceptance_criteria': sp.acceptance_criteria,
                'required_expertise': sp.required_expertise,
                'associated_risks': sp.associated_risks
            }
            for sp in result.sub_problems
        ],
        'execution_plan': [
            {
                'phase': i + 1,
                'tasks': [
                    {
                        'id': sp.id,
                        'title': sp.title
                    }
                    for sp in order if sp in phase_tasks
                ]
            }
        ]
    }

    output_file = "ml_pipeline_decomposition.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\n[OK] Exported to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DECOMPOSITION SUMMARY")
    print("=" * 80)
    print(f"[OK] Problem: {problem.title}")
    print(f"[OK] Generated {len(result.sub_problems)} sub-problems")
    print(f"[OK] Total effort: {total_effort} hours ({total_effort/8:.1f} days)")
    print(f"[OK] Estimated phases: {phase}")
    print(f"[OK] Parallelizable: {parallelizable_pct:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
