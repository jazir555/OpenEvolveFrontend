"""
Basic ROMA Decomposition Example

Demonstrates basic usage of ROMA integration with problem_decomposition.py
"""

from problem_decomposition import (
    ProblemDecomposer,
    DecompositionStrategy,
    get_roma_integration_status,
    get_recommended_strategy,
)
from roma_config_helper import ROMAConfig, ROMAConfigPresets


def example_1_basic_roma_decomposition():
    """Example 1: Basic ROMA decomposition with default settings"""
    print("=" * 60)
    print("Example 1: Basic ROMA Decomposition")
    print("=" * 60)

    # Create decomposer
    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    # Simple problem to decompose
    content = """
    Design a user authentication system with the following requirements:
    1. Support email/password login
    2. Support OAuth2 integration (Google, GitHub)
    3. Implement password reset functionality
    4. Include session management
    5. Add two-factor authentication
    """

    # Decompose using ROMA
    result = decomposer.decompose_content(
        content=content,
        strategy=DecompositionStrategy.ROMA,
        max_components=10,
        use_problem_analyzer=False,
    )

    # Display results
    print(f"\nStrategy Used: {result.decomposition_strategy.value}")
    print(f"Components Created: {len(result.components)}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Decomposition Time: {result.metadata.get('decomposition_time', 0):.2f}s")

    print("\nComponents:")
    for i, component in enumerate(result.components, 1):
        print(f"\n{i}. {component.title}")
        print(f"   Type: {component.component_type.value}")
        print(f"   Complexity: {component.complexity_score:.2f}")
        print(f"   Size: {len(component.content)} chars")
        if component.dependencies:
            print(f"   Dependencies: {', '.join(component.dependencies)}")

    # Check if ROMA was used
    if decomposer.last_roma_error:
        print(f"\nNote: ROMA had an error, fallback used: {decomposer.last_roma_error}")
    else:
        print("\n✓ ROMA decomposition successful!")


def example_2_roma_with_config():
    """Example 2: ROMA decomposition with custom configuration"""
    print("\n" + "=" * 60)
    print("Example 2: ROMA with Custom Configuration")
    print("=" * 60)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Implement a microservices architecture for an e-commerce platform with:
    - Product catalog service
    - Order management service
    - Payment processing service
    - User profile service
    - Inventory management service
    """

    # Use ROMA with custom parameters
    result = decomposer.decompose_content(
        content=content,
        strategy=DecompositionStrategy.ROMA,
        max_components=15,
        min_component_size=30,
        roma_fractal=True,
        roma_max_depth=3,
        roma_max_nodes=50,
        roma_allow_small_components=False,
    )

    print(f"\nComponents: {len(result.components)}")
    print(f"Quality: {result.quality_score:.2f}")

    # Display ROMA-specific metadata
    if result.components:
        component = result.components[0]
        print(f"\nExample Component Metadata:")
        if metadata := component.metadata:
            if metadata.get('roma_task_type'):
                print(f"  Task Type: {metadata.get('roma_task_type')}")
            if metadata.get('roma_depth') is not None:
                print(f"  ROMA Depth: {metadata.get('roma_depth')}")
            if metadata.get('roma_is_atomic') is not None:
                print(f"  Is Atomic: {metadata.get('roma_is_atomic')}")
            if metadata.get('roma_node_kind'):
                print(f"  Node Kind: {metadata.get('roma_node_kind')}")


def example_3_using_presets():
    """Example 3: Using ROMA configuration presets"""
    print("\n" + "=" * 60)
    print("Example 3: Using ROMA Configuration Presets")
    print("=" * 60)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Analyze the architecture of a distributed database system focusing on:
    - Data partitioning strategies
    - Replication mechanisms
    - Consistency models
    - Fault tolerance
    - Scalability approaches
    """

    # Try different presets
    presets = [
        ("Fast", ROMAConfigPresets.fast()),
        ("Balanced", ROMAConfigPresets.balanced()),
        ("Thorough", ROMAConfigPresets.thorough()),
    ]

    for preset_name, preset in presets:
        print(f"\n--- {preset_name} Preset ---")

        kwargs = preset.to_kwargs()
        result = decomposer.decompose_content(
            content=content,
            max_components=8,
            **kwargs,
        )

        print(f"Components: {len(result.components)}")
        print(f"Quality: {result.quality_score:.2f}")
        print(f"Max Depth: {kwargs.get('roma_max_depth')}")
        print(f"Max Nodes: {kwargs.get('roma_max_nodes', 'default')}")


def example_4_strategy_comparison():
    """Example 4: Compare ROMA with other decomposition strategies"""
    print("\n" + "=" * 60)
    print("Example 4: Strategy Comparison")
    print("=" * 60)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Design a RESTful API for a task management system with features for
    creating, updating, deleting, and querying tasks with user authentication
    and project organization.
    """

    strategies = [
        DecompositionStrategy.ROMA,
        DecompositionStrategy.SEMANTIC,
        DecompositionStrategy.HIERARCHICAL,
        DecompositionStrategy.FUNCTIONAL,
    ]

    results = {}
    for strategy in strategies:
        result = decomposer.decompose_content(
            content=content,
            strategy=strategy,
            max_components=8,
            use_problem_analyzer=False,
        )
        results[strategy.value] = result

    # Display comparison
    print(f"\n{'Strategy':<15} {'Components':<12} {'Quality':<10} {'Time':<10}")
    print("-" * 50)
    for strategy_name, result in results.items():
        print(f"{strategy_name:<15} {len(result.components):<12} "
              f"{result.quality_score:<10.2f} "
              f"{result.metadata.get('decomposition_time', 0):<10.2f}")

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].quality_score)
    print(f"\nBest Strategy: {best_strategy[0]} (Quality: {best_strategy[1].quality_score:.2f})")


def example_5_reassembly():
    """Example 5: Decompose and reassemble content"""
    print("\n" + "=" * 60)
    print("Example 5: Decomposition and Reassembly")
    print("=" * 60)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Implement a caching layer for a web application with strategies for
    cache invalidation, expiration policies, and distributed caching.
    """

    # Decompose
    result = decomposer.decompose_content(
        content=content,
        strategy=DecompositionStrategy.ROMA,
        max_components=5,
    )

    print(f"Original content length: {len(result.original_content)} chars")
    print(f"Decomposed into {len(result.components)} components")

    # Get reassembly instructions
    instructions = result.reassembly_instructions
    print(f"\nAssembly Order: {instructions['assembly_order']}")

    print("\nMerge Strategies:")
    for comp_id, strategy in instructions['merge_strategies'].items():
        print(f"  {comp_id}: {strategy}")

    print(f"\nValidation Checks:")
    for check in instructions['validation_checks']:
        print(f"  - {check}")

    # Reassemble (simplified example)
    reassembly_result = decomposer.reassemble_components(
        components=result.components,
        reassembly_instructions=instructions,
    )

    print(f"\nReassembled length: {len(reassembly_result.reassembled_content)} chars")
    print(f"Reassembly quality: {reassembly_result.quality_score:.2f}")


def example_6_checking_status():
    """Example 6: Check ROMA integration status"""
    print("\n" + "=" * 60)
    print("Example 6: ROMA Integration Status")
    print("=" * 60)

    # Check ROMA status
    status = get_roma_integration_status()

    print("\nROMA Integration Status:")
    print(f"  ROMA DSPy Available: {status['roma_dspy_available']}")
    print(f"  ROMA MCP Available: {status['roma_mcp_available']}")
    print(f"  ROMA Available: {status['roma_available']}")
    print(f"  Recommendation: {status['recommendation']}")

    if 'roma_dspy_version' in status:
        print(f"  ROMA DSPy Version: {status['roma_dspy_version']}")

    # Get recommended strategy
    test_content = "Implement a complex distributed system with multiple services"
    recommended = get_recommended_strategy(test_content, prefer_roma=True)

    print(f"\nFor content: '{test_content[:50]}...'")
    print(f"Recommended Strategy: {recommended.value}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ROMA DECOMPOSITION EXAMPLES")
    print("=" * 60)

    # Run all examples
    example_1_basic_roma_decomposition()
    example_2_roma_with_config()
    example_3_using_presets()
    example_4_strategy_comparison()
    example_5_reassembly()
    example_6_checking_status()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
