"""
Advanced ROMA Decomposition Examples

Demonstrates advanced features including:
- Domain-aware decomposition with ProblemAnalyzer
- Custom ROMA contexts
- Error handling and fallbacks
- Performance optimization
- Integration with other OpenEvolve components
"""

import time
from typing import Dict, Any
from problem_decomposition import (
    ProblemDecomposer,
    DecompositionStrategy,
    DecompositionResult,
    get_recommended_strategy,
)
from roma_config_helper import (
    ROMAConfig,
    ROMAConfigPresets,
    validate_roma_config,
    merge_roma_configs,
)


def example_domain_aware_decomposition():
    """Example: Domain-aware ROMA decomposition with custom context"""
    print("=" * 70)
    print("Example: Domain-Aware ROMA Decomposition")
    print("=" * 70)

    # Create decomposer without ProblemAnalyzer (to avoid config issues)
    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    # Complex domain-specific problem
    content = """
    Design a blockchain-based supply chain tracking system with the following features:
    1. Product provenance tracking from manufacturer to consumer
    2. Smart contracts for automated compliance checks
    3. RFID/IoT sensor integration for real-time tracking
    4. Privacy-preserving data sharing between stakeholders
    5. Cryptographic verification of product authenticity
    """

    # Provide domain context manually
    domain_context = """
    Domain: Blockchain and Supply Chain Management
    Key Concepts:
    - Distributed ledger technology
    - Smart contracts (Solidity, Vyper)
    - Consensus mechanisms
    - Cryptographic hashing
    - RFID/IoT integration
    - Supply chain provenance
    - Privacy-preserving computation (zero-knowledge proofs)

    Constraints:
    - Must ensure data immutability
    - Support multiple stakeholders with different access levels
    - Comply with supply chain regulations (GS1 standards)
    - Handle high-volume real-time data from IoT sensors
    """

    # Decompose with custom domain context
    result = decomposer.decompose_content(
        content=content,
        strategy=DecompositionStrategy.ROMA,
        max_components=12,
        min_component_size=50,
        roma_context=domain_context,
        roma_max_depth=4,
        roma_fractal=True,
    )

    print(f"\nDomain-Aware Decomposition Results:")
    print(f"  Components: {len(result.components)}")
    print(f"  Quality Score: {result.quality_score:.2f}")

    # Analyze component complexity distribution
    complexity_dist = result.metadata.get('complexity_distribution', {})
    print(f"\nComplexity Distribution:")
    print(f"  Min: {complexity_dist.get('min', 0):.2f}")
    print(f"  Max: {complexity_dist.get('max', 0):.2f}")
    print(f"  Avg: {complexity_dist.get('avg', 0):.2f}")
    print(f"  High Complexity Count: {complexity_dist.get('high_complexity_count', 0)}")

    # Show highest complexity components
    sorted_components = sorted(
        result.components,
        key=lambda c: c.complexity_score,
        reverse=True
    )[:3]

    print(f"\nTop 3 Most Complex Components:")
    for i, comp in enumerate(sorted_components, 1):
        print(f"  {i}. {comp.title}")
        print(f"     Complexity: {comp.complexity_score:.2f}")
        print(f"     Dependencies: {len(comp.dependencies)}")


def example_error_handling():
    """Example: Robust error handling and fallbacks"""
    print("\n" + "=" * 70)
    print("Example: Error Handling and Fallbacks")
    print("=" * 70)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Implement a machine learning pipeline for image classification with:
    - Data preprocessing and augmentation
    - Model training and validation
    - Hyperparameter tuning
    - Model deployment and monitoring
    """

    # Try ROMA decomposition
    result = decomposer.decompose_content(
        content=content,
        strategy=DecompositionStrategy.ROMA,
        max_components=8,
        roma_max_depth=3,
        use_problem_analyzer=False,
    )

    # Check if ROMA succeeded or used fallback
    if decomposer.last_roma_error:
        print(f"\n⚠ ROMA encountered an error:")
        print(f"  {decomposer.last_roma_error}")
        print(f"\n✓ Fallback decomposition used:")
        print(f"  Strategy: {result.decomposition_strategy.value}")
        print(f"  Components: {len(result.components)}")
    else:
        print(f"\n✓ ROMA decomposition successful!")
        print(f"  Components: {len(result.components)}")
        print(f"  Quality: {result.quality_score:.2f}")

    # Verify result quality regardless of method used
    if result.quality_score < 0.5:
        print(f"\n⚠ Low quality score detected. Consider:")
        print(f"  - Increasing max_components")
        print(f"  - Adjusting min_component_size")
        print(f"  - Using a different decomposition strategy")
    else:
        print(f"\n✓ Good quality decomposition achieved")


def example_performance_optimization():
    """Example: Performance optimization techniques"""
    print("\n" + "=" * 70)
    print("Example: Performance Optimization")
    print("=" * 70)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Design a multi-tenant SaaS application architecture with:
    - Tenant isolation and data segregation
    - Scalable resource allocation
    - Customizable tenant configurations
    - API rate limiting per tenant
    - Centralized administration dashboard
    """

    # Test different performance configurations
    configs = [
        ("Fast (Low Depth)", ROMAConfigPresets.fast()),
        ("Balanced", ROMAConfigPresets.balanced()),
        ("Thorough (High Depth)", ROMAConfigPresets.thorough()),
    ]

    print(f"\n{'Configuration':<25} {'Time':<10} {'Components':<12} {'Quality':<10}")
    print("-" * 60)

    for config_name, config in configs:
        start = time.time()

        result = decomposer.decompose_content(
            content=content,
            max_components=10,
            use_problem_analyzer=False,
            **config.to_kwargs(),
        )

        elapsed = time.time() - start

        print(f"{config_name:<25} {elapsed:<10.2f} {len(result.components):<12} "
              f"{result.quality_score:<10.2f}")

    # Performance recommendations
    print(f"\nPerformance Tips:")
    print(f"  1. Use lower max_depth (2-3) for faster decomposition")
    print(f"  2. Set max_nodes to limit total processing")
    print(f"  3. Disable ProblemAnalyzer if domain context not needed")
    print(f"  4. Use roma_allow_small_components=True to avoid filtering overhead")


def example_custom_config_validation():
    """Example: Creating and validating custom ROMA configurations"""
    print("\n" + "=" * 70)
    print("Example: Custom Configuration Validation")
    print("=" * 70)

    # Create custom config
    config = ROMAConfig(
        model="gpt-4o",
        atomizer_model="claude-3-5-sonnet-20241022",
        planner_model="gpt-4o",
        max_depth=4,
        max_nodes=80,
        use_fractal=True,
        include_non_leaf=False,
    )

    # Validate configuration
    errors = validate_roma_config(config)

    if errors:
        print(f"\n⚠ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n✓ Configuration is valid!")
        print(f"\nConfiguration Details:")
        print(f"  Model: {config.model}")
        print(f"  Atomizer Model: {config.atomizer_model}")
        print(f"  Planner Model: {config.planner_model}")
        print(f"  Max Depth: {config.max_depth}")
        print(f"  Max Nodes: {config.max_nodes}")
        print(f"  Use Fractal: {config.use_fractal}")

        # Show kwargs that would be generated
        kwargs = config.to_kwargs()
        print(f"\nGenerated Kwargs ({len(kwargs)} parameters):")
        for key, value in sorted(kwargs.items()):
            print(f"  {key}: {value}")


def example_config_merging():
    """Example: Merging multiple ROMA configurations"""
    print("\n" + "=" * 70)
    print("Example: Configuration Merging")
    print("=" * 70)

    # Create base config
    base_config = ROMAConfig(
        model="gpt-4o",
        max_depth=3,
        use_fractal=True,
    )

    # Create override config
    override_config = ROMAConfig(
        max_depth=5,  # Override depth
        max_nodes=100,  # Add nodes
        enable_problem_analyzer=True,  # Enable analyzer
    )

    # Merge configs
    merged = merge_roma_configs(base_config, override_config)

    print(f"\nBase Config:")
    print(f"  Model: {base_config.model}")
    print(f"  Max Depth: {base_config.max_depth}")
    print(f"  Max Nodes: {base_config.max_nodes}")
    print(f"  Enable Analyzer: {base_config.enable_problem_analyzer}")

    print(f"\nOverride Config:")
    print(f"  Model: {override_config.model or 'N/A'}")
    print(f"  Max Depth: {override_config.max_depth}")
    print(f"  Max Nodes: {override_config.max_nodes}")
    print(f"  Enable Analyzer: {override_config.enable_problem_analyzer}")

    print(f"\nMerged Config:")
    print(f"  Model: {merged.model}")
    print(f"  Max Depth: {merged.max_depth}")
    print(f"  Max Nodes: {merged.max_nodes}")
    print(f"  Enable Analyzer: {merged.enable_problem_analyzer}")


def example_component_analysis():
    """Example: Detailed component analysis after decomposition"""
    print("\n" + "=" * 70)
    print("Example: Detailed Component Analysis")
    print("=" * 70)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    content = """
    Develop a real-time chat application with:
    - WebSocket-based messaging
    - User presence and online status
    - Message history and search
    - File sharing capabilities
    - End-to-end encryption
    """

    result = decomposer.decompose_content(
        content=content,
        strategy=DecompositionStrategy.ROMA,
        max_components=8,
        use_problem_analyzer=False,
    )

    print(f"\nDecomposition Analysis:")
    print(f"  Total Components: {len(result.components)}")

    # Analyze by component type
    type_counts = {}
    for comp in result.components:
        comp_type = comp.component_type.value
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

    print(f"\nComponent Type Distribution:")
    for comp_type, count in sorted(type_counts.items()):
        print(f"  {comp_type}: {count}")

    # Analyze dependencies
    total_deps = sum(len(comp.dependencies) for comp in result.components)
    avg_deps = total_deps / len(result.components) if result.components else 0

    print(f"\nDependency Analysis:")
    print(f"  Total Dependencies: {total_deps}")
    print(f"  Avg Dependencies per Component: {avg_deps:.2f}")

    # Find most connected component
    most_connected = max(
        result.components,
        key=lambda c: len(c.dependencies),
        default=None
    )

    if most_connected:
        print(f"\nMost Connected Component:")
        print(f"  {most_connected.title}")
        print(f"  Dependencies: {len(most_connected.dependencies)}")
        print(f"  Dependents: {[c.id for c in result.components if most_connected.id in c.dependencies]}")

    # Analyze evolution priorities
    print(f"\nEvolution Priorities (top 3):")
    prioritized = sorted(
        result.components,
        key=lambda c: c.evolution_priority,
        reverse=True
    )[:3]

    for i, comp in enumerate(prioritized, 1):
        print(f"  {i}. {comp.title} (Priority: {comp.evolution_priority:.2f})")


def example_decomposition_history():
    """Example: Using decomposition history for tracking"""
    print("\n" + "=" * 70)
    print("Example: Decomposition History")
    print("=" * 70)

    decomposer = ProblemDecomposer(auto_create_analyzer=False)

    contents = [
        "Implement a user registration system",
        "Design a database schema for e-commerce",
        "Create a RESTful API for task management",
    ]

    # Decompose multiple contents
    for i, content in enumerate(contents, 1):
        print(f"\n--- Decomposition {i} ---")
        result = decomposer.decompose_content(
            content=content,
            strategy=DecompositionStrategy.ROMA,
            max_components=5,
            use_problem_analyzer=False,
        )
        print(f"Content: {content[:50]}...")
        print(f"Components: {len(result.components)}")

    # Access history
    history = decomposer.get_decomposition_history()
    print(f"\n--- Decomposition History ---")
    print(f"Total decompositions: {len(history)}")

    for i, result in enumerate(history, 1):
        print(f"\n{i}. Strategy: {result.decomposition_strategy.value}")
        print(f"   Components: {len(result.components)}")
        print(f"   Quality: {result.quality_score:.2f}")
        print(f"   Time: {result.metadata.get('decomposition_time', 0):.2f}s")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ADVANCED ROMA DECOMPOSITION EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_domain_aware_decomposition()
    example_error_handling()
    example_performance_optimization()
    example_custom_config_validation()
    example_config_merging()
    example_component_analysis()
    example_decomposition_history()

    print("\n" + "=" * 70)
    print("All advanced examples completed!")
    print("=" * 70)
