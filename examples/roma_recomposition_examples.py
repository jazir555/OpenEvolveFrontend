"""
ROMA Recomposition Examples

Demonstrates ROMA-based solution recomposition with various scenarios.
"""

from typing import Any
from problem_recomposition import (
    SolutionAssembler,
    ConflictDetector,
    ConflictResolver,
)
from sovereign_data_models import (
    ProblemDefinition,
    SubProblem,
    DecompositionPlan,
    SolutionAttempt,
    generate_id,
)
from roma_recomposition_config import (
    ROMARecompositionConfig,
    ROMARecompositionPresets,
    get_recommended_recomposition_config,
)


def example_1_basic_roma_recomposition():
    """Example 1: Basic ROMA recomposition"""
    print("=" * 70)
    print("Example 1: Basic ROMA Recomposition")
    print("=" * 70)

    # Create assembler with ROMA enabled
    assembler = SolutionAssembler(
        enable_roma=True,
        roma_max_depth=2,
        roma_model="gpt-4o",
    )

    # Create mock sub-solutions
    sub_solutions = {
        "sub_1": SolutionAttempt(
            solution_id="sol_1",
            sub_problem_id="sub_1",
            solution_content="""## User Authentication Module

Implement secure user authentication using JWT tokens with the following features:
- Login form with email/password validation
- JWT token generation and validation
- Session management with refresh tokens
- Password reset via email""",
            confidence_score=0.9,
        ),
        "sub_2": SolutionAttempt(
            solution_id="sol_2",
            sub_problem_id="sub_2",
            solution_content="""## User Profile Management

Create user profile management with:
- Profile creation and editing
- Avatar upload functionality
- User preferences storage
- Profile visibility settings""",
            confidence_score=0.85,
        ),
        "sub_3": SolutionAttempt(
            solution_id="sol_3",
            sub_problem_id="sub_3",
            solution_content="""## Authorization System

Implement role-based access control with:
- Admin, user, and moderator roles
- Permission checking middleware
- Role assignment interface
- Access control on sensitive endpoints""",
            confidence_score=0.88,
        ),
    }

    # Create mock decomposition plan
    plan = DecompositionPlan(
        id=generate_id("plan"),
        problem_id=generate_id("problem"),
        problem_statement="Build a complete user management system",
        sub_problems=[
            SubProblem(id="sub_1", description="Authentication", dependencies=[]),
            SubProblem(id="sub_2", description="Profile Management", dependencies=[]),
            SubProblem(id="sub_3", description="Authorization", dependencies=[]),
        ],
    )

    # Assemble with ROMA
    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        assembly_strategy="roma",
    )

    print(f"\n[OK] ROMA Recomposition Complete")
    print(f"  Strategy: {result.assembly_strategy}")
    print(f"  Length: {len(result.assembled_content)} chars")
    print(f"  Quality: {result.quality_metrics.overall_score:.2f}")
    print(f"  Conflicts: {result.metadata.get('num_conflicts', 0)}")

    print(f"\n{'=' * 60}")
    print("Assembled Solution (first 500 chars):")
    print('=' * 60)
    print(result.assembled_content[:500] + "...")


def example_2_recomposition_with_presets():
    """Example 2: Using ROMA recomposition presets"""
    print("\n" + "=" * 70)
    print("Example 2: ROMA Recomposition with Presets")
    print("=" * 70)

    # Create sub-solutions with a conflict
    sub_solutions = {
        "sub_1": SolutionAttempt(
            solution_id="sol_1",
            sub_problem_id="sub_1",
            solution_content="Use PostgreSQL for the primary database with strong consistency.",
            confidence_score=0.9,
        ),
        "sub_2": SolutionAttempt(
            solution_id="sol_2",
            sub_problem_id="sub_2",
            solution_content="Use MongoDB for flexible schema design and horizontal scaling.",
            confidence_score=0.85,
        ),
    }

    plan = DecompositionPlan(
        id=generate_id("plan"),
        problem_id=generate_id("problem"),
        problem_statement="Design database architecture",
        sub_problems=[
            SubProblem(id="sub_1", description="Database choice", dependencies=[]),
            SubProblem(id="sub_2", description="Database choice", dependencies=[]),
        ],
    )

    # Try different presets
    presets = [
        ("Fast", ROMARecompositionPresets.fast()),
        ("Balanced", ROMARecompositionPresets.balanced()),
        ("Thorough", ROMARecompositionPresets.thorough()),
        ("High Conflict", ROMARecompositionPresets.high_conflict()),
    ]

    for preset_name, preset in presets:
        print(f"\n--- {preset_name} Preset ---")

        assembler = SolutionAssembler(
            enable_roma=True,
            roma_max_depth=preset.max_depth,
            roma_model=preset.model,
        )

        kwargs = preset.to_kwargs()
        result = assembler.assemble_solution(
            decomposition_plan=plan,
            sub_solutions=sub_solutions,
            **kwargs
        )

        print(f"  Length: {len(result.assembled_content)} chars")
        print(f"  Quality: {result.quality_metrics.overall_score:.2f}")
        print(f"  Coherence: {result.quality_metrics.coherence_score:.2f}")
        print(f"  Integration: {result.quality_metrics.integration_quality:.2f}")


def example_3_recommended_config():
    """Example 3: Auto-selecting recommended ROMA recomposition config"""
    print("\n" + "=" * 70)
    print("Example 3: Recommended ROMA Recomposition Configuration")
    print("=" * 70)

    # Scenario: Complex solution with many conflicts
    print("\nScenario: 8 sub-solutions, 5 conflicts, high complexity")
    config = get_recommended_recomposition_config(
        num_sub_solutions=8,
        num_conflicts=5,
        complexity="high",
        content_type="code"
    )

    print(f"\nRecommended Configuration:")
    print(f"  Model: {config.model or 'default'}")
    print(f"  Max Depth: {config.max_depth}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Conflict Resolution: {config.enable_conflict_resolution}")
    print(f"  Fallback Strategy: {config.conflict_resolution_fallback}")

    # Validate and show kwargs
    errors = config.validate()
    if errors:
        print(f"\n⚠ Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n[OK] Configuration is valid!")

        kwargs = config.to_kwargs()
        print(f"\nGenerated Kwargs ({len(kwargs)} parameters):")
        for key, value in sorted(kwargs.items()):
            print(f"  {key}: {value}")


def example_4_custom_context():
    """Example 4: ROMA recomposition with custom domain context"""
    print("\n" + "=" * 70)
    print("Example 4: ROMA Recomposition with Custom Context")
    print("=" * 70)

    assembler = SolutionAssembler(
        enable_roma=True,
        roma_max_depth=2,
    )

    # Create sub-solutions
    sub_solutions = {
        "sub_1": SolutionAttempt(
            solution_id="sol_1",
            sub_problem_id="sub_1",
            solution_content="Implement REST API endpoints for user management.",
            confidence_score=0.9,
        ),
        "sub_2": SolutionAttempt(
            solution_id="sol_2",
            sub_problem_id="sub_2",
            solution_content="Create GraphQL API for flexible data queries.",
            confidence_score=0.85,
        ),
    }

    plan = DecompositionPlan(
        id=generate_id("plan"),
        problem_id=generate_id("problem"),
        problem_statement="Design API architecture",
        sub_problems=[
            SubProblem(id="sub_1", description="REST API", dependencies=[]),
            SubProblem(id="sub_2", description="GraphQL API", dependencies=[]),
        ],
    )

    # Custom domain context
    custom_context = """
Domain: E-commerce Platform Architecture
Key Constraints:
- Must support both REST and GraphQL
- REST for simple CRUD operations
- GraphQL for complex queries and dashboard
- Ensure consistent authentication across both APIs
- Performance target: <100ms for 95th percentile
"""

    # Assemble with custom context
    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        assembly_strategy="roma",
        roma_context=custom_context,
        roma_extra_context="Focus on API consistency and performance.",
    )

    print(f"\n[OK] Recomposition with Custom Context")
    print(f"  Length: {len(result.assembled_content)} chars")
    print(f"  Quality: {result.quality_metrics.overall_score:.2f}")

    print(f"\n{'=' * 60}")
    print("Assembled Solution (first 400 chars):")
    print('=' * 60)
    print(result.assembled_content[:400] + "...")


def example_5_comparison():
    """Example 5: Compare ROMA vs traditional recomposition"""
    print("\n" + "=" * 70)
    print("Example 5: ROMA vs Traditional Recomposition Comparison")
    print("=" * 70)

    # Create sub-solutions
    sub_solutions = {
        "sub_1": SolutionAttempt(
            solution_id="sol_1",
            sub_problem_id="sub_1",
            solution_content="Implement microservices architecture with service mesh.",
            confidence_score=0.9,
        ),
        "sub_2": SolutionAttempt(
            solution_id="sol_2",
            sub_problem_id="sub_2",
            solution_content="Use API gateway for request routing and load balancing.",
            confidence_score=0.88,
        ),
        "sub_3": SolutionAttempt(
            solution_id="sol_3",
            sub_problem_id="sub_3",
            solution_content="Deploy services using Kubernetes with auto-scaling.",
            confidence_score=0.85,
        ),
    }

    plan = DecompositionPlan(
        id=generate_id("plan"),
        problem_id=generate_id("problem"),
        problem_statement="Design scalable system architecture",
        sub_problems=[
            SubProblem(id="sub_1", description="Architecture", dependencies=[]),
            SubProblem(id="sub_2", description="Gateway", dependencies=[]),
            SubProblem(id="sub_3", description="Deployment", dependencies=[]),
        ],
    )

    # Compare strategies
    strategies = [
        ("Hierarchical", "hierarchical"),
        ("Linear", "linear"),
        ("Parallel", "parallel"),
        ("ROMA", "roma"),
    ]

    print(f"\n{'Strategy':<15} {'Length':<10} {'Quality':<10} {'Coherence':<10} {'Integration':<12}")
    print("-" * 70)

    for strategy_name, strategy_key in strategies:
        assembler = SolutionAssembler(enable_roma=True, roma_max_depth=2)

        result = assembler.assemble_solution(
            decomposition_plan=plan,
            sub_solutions=sub_solutions,
            assembly_strategy=strategy_key,
        )

        metrics = result.quality_metrics
        print(f"{strategy_name:<15} {len(result.assembled_content):<10} "
              f"{metrics.overall_score:<10.2f} {metrics.coherence_score:<10.2f} "
              f"{metrics.integration_quality:<12.2f}")


def example_6_conflict_rich_scenario():
    """Example 6: ROMA recomposition with conflict resolution"""
    print("\n" + "=" * 70)
    print("Example 6: ROMA Recomposition with Conflict Resolution")
    print("=" * 70)

    # Create sub-solutions with contradictions
    sub_solutions = {
        "sub_1": SolutionAttempt(
            solution_id="sol_1",
            sub_problem_id="sub_1",
            solution_content="The system MUST use synchronous communication for all services.",
            confidence_score=0.9,
        ),
        "sub_2": SolutionAttempt(
            solution_id="sol_2",
            sub_problem_id="sub_2",
            solution_content="The system should use asynchronous messaging for better scalability.",
            confidence_score=0.85,
        ),
    }

    plan = DecompositionPlan(
        id=generate_id("plan"),
        problem_id=generate_id("problem"),
        problem_statement="Design communication architecture",
        sub_problems=[
            SubProblem(id="sub_1", description="Communication style", dependencies=[]),
            SubProblem(id="sub_2", description="Communication style", dependencies=[]),
        ],
    )

    # Use high-conflict preset
    config = ROMARecompositionPresets.high_conflict()
    assembler = SolutionAssembler(
        enable_roma=True,
        roma_max_depth=config.max_depth,
        roma_model=config.model,
    )

    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        **config.to_kwargs()
    )

    print(f"\n[OK] Conflict-Aware Recomposition")
    print(f"  Conflicts Detected: {result.metadata.get('num_conflicts', 0)}")
    print(f"  Conflicts Resolved: {result.metadata.get('num_resolved', 0)}")
    print(f"  Overall Quality: {result.quality_metrics.overall_score:.2f}")
    print(f"  Consistency: {result.quality_metrics.consistency_score:.2f}")
    print(f"  Coherence: {result.quality_metrics.coherence_score:.2f}")

    print(f"\n{'=' * 60}")
    print("Assembled Solution (first 500 chars):")
    print('=' * 60)
    print(result.assembled_content[:500] + "...")


def example_7_deterministic_vs_creative():
    """Example 7: Compare deterministic vs creative ROMA recomposition"""
    print("\n" + "=" * 70)
    print("Example 7: Deterministic vs Creative ROMA Recomposition")
    print("=" * 70)

    # Create code sub-solutions (technical precision matters!)
    sub_solutions = {
        "sub_1": SolutionAttempt(
            solution_id="sol_1",
            sub_problem_id="sub_1",
            solution_content="""```python
def authenticate_user(username: str, password: str) -> bool:
    '''Authenticate user with credentials'''
    if not username or not password:
        return False
    # Hash password and compare with stored hash
    return verify_password(username, password)
```""",
            confidence_score=0.95,
        ),
        "sub_2": SolutionAttempt(
            solution_id="sol_2",
            sub_problem_id="sub_2",
            solution_content="""```python
class UserProfile:
    '''User profile data model'''
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.preferences = {}

    def update_preference(self, key: str, value: Any) -> None:
        '''Update user preference'''
        self.preferences[key] = value
```""",
            confidence_score=0.90,
        ),
    }

    plan = DecompositionPlan(
        id=generate_id("plan"),
        problem_id=generate_id("problem"),
        problem_statement="Build user authentication and profile system",
        sub_problems=[
            SubProblem(id="sub_1", description="Authentication", dependencies=[]),
            SubProblem(id="sub_2", description="User Profile", dependencies=[]),
        ],
    )

    print("\n" + "-" * 70)
    print("DETERMINISTIC MODE (Default)")
    print("-" * 70)

    # Deterministic: ROMA decides structure, sub-solutions unchanged
    config_deterministic = ROMARecompositionConfig(
        deterministic=True,
        enable_roma=True,
    )

    assembler = SolutionAssembler(enable_roma=True)
    result_deterministic = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        **config_deterministic.to_kwargs()
    )

    print(f"\n[OK] Deterministic Assembly Complete")
    print(f"  Mode: Sub-solutions inserted VERBATIM")
    print(f"  Length: {len(result_deterministic.assembled_content)} chars")
    print(f"  Original code preserved: YES")
    print(f"  Reproducible: YES")

    # Check if original code is preserved
    original_preserved = (
        "def authenticate_user" in result_deterministic.assembled_content and
        "class UserProfile:" in result_deterministic.assembled_content
    )
    print(f"  Code integrity verified: {original_preserved}")

    print("\n" + "-" * 70)
    print("CREATIVE MODE")
    print("-" * 70)

    # Creative: ROMA may rewrite and integrate
    config_creative = ROMARecompositionConfig(
        deterministic=False,
        enable_roma=True,
    )

    result_creative = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        **config_creative.to_kwargs()
    )

    print(f"\n[OK] Creative Assembly Complete")
    print(f"  Mode: ROMA may rewrite sub-solutions")
    print(f"  Length: {len(result_creative.assembled_content)} chars")
    print(f"  Original code preserved: MAYBE")
    print(f"  Reproducible: NO (LLM non-determinism)")

    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)

    print("\nDETERMINISTIC output (first 400 chars):")
    print('=' * 60)
    print(result_deterministic.assembled_content[:400] + "...")

    print("\n\nCREATIVE output (first 400 chars):")
    print('=' * 60)
    print(result_creative.assembled_content[:400] + "...")

    print("\n\nKEY DIFFERENCES:")
    print("  Deterministic: Original code blocks preserved exactly")
    print("  Creative: Code may be refactored or rewritten by LLM")
    print("\nRECOMMENDATION:")
    print("  - Use DETERMINISTIC for: Code, technical specs, APIs")
    print("  - Use CREATIVE for: Documentation, prose, summaries")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROMA RECOMPOSITION EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_1_basic_roma_recomposition()
    example_2_recomposition_with_presets()
    example_3_recommended_config()
    example_4_custom_context()
    example_5_comparison()
    example_6_conflict_rich_scenario()
    example_7_deterministic_vs_creative()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
