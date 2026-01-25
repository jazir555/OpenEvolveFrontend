"""
BubbleLabs Nodes - Usage Examples

This file demonstrates how to use the 8 OpenEvolve nodes
integrated with BubbleLabs.
"""

from bubblelabs_nodes import (
    get_node,
    NodeRegistry,
    NodeExecutionError
)


# Example 1: Basic Node Usage
def example_basic_usage():
    """Basic node creation and execution"""
    print("=" * 60)
    print("Example 1: Basic Node Usage")
    print("=" * 60)

    # Create a decomposition node
    node = get_node('decomposition', {
        'method': 'roma',
        'max_depth': 2,
        'parallel': True
    })

    print(f"Created node: {node.get_display_name()}")
    print(f"Category: {node.get_category()}")
    print(f"Version: {node.get_version()}")
    print(f"Description: {node.get_description()}")
    print()


# Example 2: Node Registry
def example_node_registry():
    """Explore available nodes"""
    print("=" * 60)
    print("Example 2: Node Registry")
    print("=" * 60)

    # List all available nodes
    nodes = NodeRegistry.list_nodes()
    print(f"Available nodes ({len(nodes)}):")
    for node_type in sorted(nodes.keys()):
        node_info = NodeRegistry.get_node_info(node_type)
        print(f"  - {node_info['display_name']}")
        print(f"    Category: {node_info['category']}")
        print(f"    Icon: {node_info['icon']}")
    print()


# Example 3: Get Node Parameter Schema
def example_parameter_schema():
    """Get parameter schema for a node"""
    print("=" * 60)
    print("Example 3: Parameter Schema")
    print("=" * 60)

    # Get schema for solution node
    solution_node = get_node('solution')
    schema = solution_node.get_parameter_schema()

    print(f"Parameter schema for {solution_node.get_display_name()}:")
    print(f"  Type: {schema['type']}")
    print(f"  Properties:")
    for prop_name, prop_def in schema['properties'].items():
        default = prop_def.get('default', 'N/A')
        description = prop_def.get('description', 'No description')
        print(f"    - {prop_name}: {description}")
        print(f"      Default: {default}")
    print()


# Example 4: Input Validation
def example_input_validation():
    """Validate inputs before execution"""
    print("=" * 60)
    print("Example 4: Input Validation")
    print("=" * 60)

    # Create output node
    node = get_node('output')

    # Test invalid inputs
    invalid_inputs = {}  # Missing required 'solution' field
    errors = node.validate_inputs(invalid_inputs)

    print("Validation errors for empty inputs:")
    for error in errors:
        print(f"  ❌ {error}")

    # Test valid inputs
    valid_inputs = {
        'solution': {'description': 'Test solution'},
        'output_format': 'markdown'
    }
    errors = node.validate_inputs(valid_inputs)

    print(f"\nValidation errors for valid inputs: {len(errors)}")
    if len(errors) == 0:
        print("  ✅ Inputs are valid!")
    print()


# Example 5: Simple Workflow (Decomposition → Solution)
def example_simple_workflow():
    """Execute a simple 2-node workflow"""
    print("=" * 60)
    print("Example 5: Simple Workflow")
    print("=" * 60)

    # Mock workflow state (in real usage, use WorkflowState)
    class MockContext:
        def __init__(self):
            self.progress = 0
            self.artifacts = {}
            self.errors = []

        def update_progress(self, progress, message):
            self.progress = progress
            print(f"  [{progress:3d}%] {message}")

        def add_artifact(self, name, artifact):
            self.artifacts[name] = artifact

        def add_error(self, error):
            self.errors.append(error)

        def generate_execution_id(self):
            return "mock_exec_001"

    # Create context
    context = MockContext()

    # Step 1: Decompose problem
    print("Step 1: Decompose problem")
    decomp_node = get_node('decomposition', {'method': 'roma', 'max_depth': 2})

    try:
        decomp_result = decomp_node.execute_safe({
            'problem_statement': 'Design a sustainable house'
        }, context)

        print(f"  ✅ Decomposition complete!")
        print(f"  Sub-problems: {decomp_result.get('total_sub_problems', 'N/A')}")
        print()

    except NodeExecutionError as e:
        print(f"  ❌ Error: {e.message}")
        print(f"  (This is expected if decomposition engine is not installed)")
        print()


# Example 6: Error Handling
def example_error_handling():
    """Demonstrate error handling"""
    print("=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    class MockContext:
        def update_progress(self, progress, message):
            pass
        def add_artifact(self, name, artifact):
            pass
        def add_error(self, error):
            pass
        def generate_execution_id(self):
            return "mock_exec_002"

    # Try to execute with invalid inputs
    node = get_node('verification')
    context = MockContext()

    try:
        result = node.execute_safe({}, context)
    except NodeExecutionError as e:
        print(f"✅ Error caught successfully!")
        print(f"  Node: {e.node_name}")
        print(f"  Message: {e.message}")
        print(f"  Details: {e.details}")
    print()


# Example 7: Explore All Node Metadata
def example_all_node_metadata():
    """Show metadata for all nodes"""
    print("=" * 60)
    print("Example 7: All Node Metadata")
    print("=" * 60)

    nodes_info = NodeRegistry.list_all_info()

    print(f"Total nodes: {len(nodes_info)}\n")

    for node_type, info in sorted(nodes_info.items()):
        print(f"📦 {info['display_name']}")
        print(f"   Type: {node_type}")
        print(f"   Category: {info['category']}")
        print(f"   Icon: {info['icon']}")
        print(f"   Version: {info['version']}")
        print(f"   Description: {info['description'][:80]}...")
        print(f"   Parameters: {len(info['parameter_schema']['properties'])}")
        print()


# Example 8: Node with Custom Configuration
def example_custom_config():
    """Create nodes with custom configurations"""
    print("=" * 60)
    print("Example 8: Custom Configuration")
    print("=" * 60)

    # Create nodes with different configurations
    configs = [
        ('solution', {
            'strategy': 'hybrid',
            'model': 'gpt-4o',
            'iterations': 200,
            'quality_threshold': 0.9
        }),
        ('gauntlet', {
            'gauntlet_type': 'full',
            'rounds': 5,
            'difficulty': 'adaptive',
            'pass_threshold': 80
        }),
        ('verification', {
            'verification_methods': ['lean4', 'automated', 'statistical'],
            'strictness': 'strict',
            'timeout': 600
        })
    ]

    for node_type, config in configs:
        node = get_node(node_type, config)
        print(f"Node: {node.get_display_name()}")
        print(f"  Config: {config}")
        print()


# Main function to run all examples
def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "BubbleLabs Nodes - Usage Examples" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run examples
    example_basic_usage()
    example_node_registry()
    example_parameter_schema()
    example_input_validation()
    example_simple_workflow()
    example_error_handling()
    example_all_node_metadata()
    example_custom_config()

    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install required dependencies (decomposition_engine, etc.)")
    print("  2. Run actual workflows with WorkflowState")
    print("  3. Integrate with BubbleLabs UI")
    print("  4. Create custom workflows for your use cases")
    print()


if __name__ == '__main__':
    main()
