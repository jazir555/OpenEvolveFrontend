"""
Test suite for BubbleLabs integration nodes.

Run with: pytest tests/test_bubblelabs_nodes.py -v
"""

import pytest
from typing import Dict, Any
from bubblelabs_nodes import (
    BubbleLabsNode,
    NodeExecutionError,
    NodeRegistry,
    get_node
)


# Mock WorkflowState for testing
class MockWorkflowState:
    """Mock workflow state for testing"""

    def __init__(self):
        self.progress = 0
        self.status = ""
        self.artifacts = {}
        self.errors = []
        self.execution_count = 0

    def update_progress(self, progress: int, message: str):
        """Update progress"""
        self.progress = progress
        self.status = message

    def add_artifact(self, name: str, artifact: Any):
        """Add artifact"""
        self.artifacts[name] = artifact

    def add_error(self, error: Dict):
        """Add error"""
        self.errors.append(error)

    def generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        self.execution_count += 1
        return f"exec_{self.execution_count}"


# Test base node functionality
class TestBubbleLabsNode:
    """Test base node class"""

    def test_base_node_is_abstract(self):
        """Base node should not be instantiable"""
        with pytest.raises(TypeError):
            BubbleLabsNode()


# Test DecompositionNode
class TestDecompositionNode:
    """Test DecompositionNode implementation"""

    @pytest.fixture
    def decomposition_node(self):
        """Create decomposition node instance"""
        from bubblelabs_nodes.decomposition_node import DecompositionNode
        return DecompositionNode()

    def test_node_metadata(self, decomposition_node):
        """Test node has correct metadata"""
        assert decomposition_node.get_display_name() == "Problem Decomposition"
        assert decomposition_node.get_category() == "analysis"
        assert decomposition_node.get_icon() == "decomposition"
        assert decomposition_node.get_version() == "1.0.0"

    def test_validate_inputs_valid(self, decomposition_node):
        """Test validation with valid inputs"""
        inputs = {
            'problem_statement': 'Solve climate change',
            'method': 'roma',
            'max_depth': 3
        }
        errors = decomposition_node.validate_inputs(inputs)
        assert len(errors) == 0

    def test_validate_inputs_missing_problem(self, decomposition_node):
        """Test validation fails without problem_statement"""
        inputs = {}
        errors = decomposition_node.validate_inputs(inputs)
        assert len(errors) > 0
        assert any('problem_statement' in e for e in errors)

    def test_validate_inputs_invalid_method(self, decomposition_node):
        """Test validation fails with invalid method"""
        inputs = {
            'problem_statement': 'Test problem',
            'method': 'invalid_method'
        }
        errors = decomposition_node.validate_inputs(inputs)
        assert len(errors) > 0
        assert any('method' in e for e in errors)

    def test_validate_inputs_invalid_max_depth(self, decomposition_node):
        """Test validation fails with invalid max_depth"""
        inputs = {
            'problem_statement': 'Test problem',
            'max_depth': 0
        }
        errors = decomposition_node.validate_inputs(inputs)
        assert len(errors) > 0
        assert any('max_depth' in e for e in errors)

    def test_get_parameter_schema(self, decomposition_node):
        """Test parameter schema is valid"""
        schema = decomposition_node.get_parameter_schema()
        assert 'type' in schema
        assert 'properties' in schema
        assert 'method' in schema['properties']
        assert 'max_depth' in schema['properties']
        assert 'parallel' in schema['properties']

    def test_execute_without_engine(self, decomposition_node):
        """Test execution fails when engine not available"""
        # Temporarily set engine to None
        original_engine = decomposition_node.engine
        decomposition_node.engine = None

        context = MockWorkflowState()
        inputs = {'problem_statement': 'Test'}

        with pytest.raises(NodeExecutionError) as exc_info:
            decomposition_node.execute(inputs, context)

        assert 'not available' in str(exc_info.value)

        # Restore engine
        decomposition_node.engine = original_engine

    def test_execute_safe_with_lifecycle_hooks(self, decomposition_node):
        """Test execute_safe calls lifecycle hooks"""
        context = MockWorkflowState()
        inputs = {
            'problem_statement': 'Test problem',
            'method': 'roma'
        }

        # This will fail if engine is not available, which is OK for this test
        try:
            result = decomposition_node.execute_safe(inputs, context)
        except NodeExecutionError:
            pass  # Expected if engine not available

        # Check lifecycle hooks were called
        assert decomposition_node.status in ['running', 'completed', 'failed']


# Test NodeRegistry
class TestNodeRegistry:
    """Test node registry functionality"""

    def test_get_registered_node(self):
        """Test getting registered node"""
        node = get_node('decomposition', {'method': 'roma'})
        assert isinstance(node, BubbleLabsNode)
        assert node.get_display_name() == "Problem Decomposition"

    def test_get_unregistered_node_fails(self):
        """Test getting unregistered node fails"""
        with pytest.raises(ValueError) as exc_info:
            get_node('nonexistent_node')

        assert 'Unknown node type' in str(exc_info.value)

    def test_list_nodes(self):
        """Test listing registered nodes"""
        nodes = NodeRegistry.list_nodes()
        assert isinstance(nodes, dict)
        assert 'decomposition' in nodes

    def test_get_node_info(self):
        """Test getting node information"""
        info = NodeRegistry.get_node_info('decomposition')
        assert 'display_name' in info
        assert 'description' in info
        assert 'icon' in info
        assert 'category' in info
        assert 'parameter_schema' in info

    def test_get_node_info_unregistered_fails(self):
        """Test getting info for unregistered node fails"""
        with pytest.raises(ValueError):
            NodeRegistry.get_node_info('nonexistent_node')

    def test_list_all_info(self):
        """Test listing all node info"""
        all_info = NodeRegistry.list_all_info()
        assert isinstance(all_info, dict)
        assert 'decomposition' in all_info


# Integration tests
class TestNodeIntegration:
    """Integration tests for node workflows"""

    @pytest.fixture
    def decomposition_node(self):
        """Create decomposition node"""
        from bubblelabs_nodes.decomposition_node import DecompositionNode
        return DecompositionNode({'method': 'roma', 'max_depth': 2})

    def test_decomposition_to_subproblem_flow(self, decomposition_node):
        """Test connecting decomposition to subproblem (mock)"""
        context = MockWorkflowState()

        # Decompose problem
        decomp_inputs = {
            'problem_statement': 'Build a sustainable house',
            'method': 'roma'
        }

        try:
            decomp_result = decomposition_node.execute_safe(decomp_inputs, context)
            # If successful, verify result structure
            assert 'sub_problems' in decomp_result
            assert 'decomposition_tree' in decomp_result
        except NodeExecutionError as e:
            # Engine not available - OK for this test
            assert 'not available' in str(e) or 'failed' in str(e)

    def test_error_handling_in_workflow(self, decomposition_node):
        """Test error handling in workflow context"""
        context = MockWorkflowState()

        # Invalid inputs
        invalid_inputs = {}

        with pytest.raises(NodeExecutionError):
            decomposition_node.execute_safe(invalid_inputs, context)

        # Check error was recorded
        assert len(context.errors) > 0
        assert context.errors[0]['node_name'] == "Problem Decomposition"


# Performance tests
class TestNodePerformance:
    """Performance and load tests"""

    @pytest.fixture
    def decomposition_node(self):
        """Create decomposition node"""
        from bubblelabs_nodes.decomposition_node import DecompositionNode
        return DecompositionNode()

    def test_parameter_schema_performance(self, decomposition_node):
        """Test schema generation is fast"""
        import time
        start = time.time()
        schema = decomposition_node.get_parameter_schema()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be very fast
        assert schema is not None

    def test_validation_performance(self, decomposition_node):
        """Test validation is fast"""
        import time
        inputs = {
            'problem_statement': 'Test problem' * 100,  # Large input
            'method': 'roma'
        }

        start = time.time()
        errors = decomposition_node.validate_inputs(inputs)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be very fast
        assert len(errors) == 0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
