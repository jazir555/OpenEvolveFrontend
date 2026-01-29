<<<<<<< HEAD
"""
Test script to verify DecompositionNode integration with existing DecompositionEngine
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_decomposition_node():
    """Test DecompositionNode with existing DecompositionEngine"""
    print("=" * 80)
    print("Testing DecompositionNode Integration")
    print("=" * 80)

    try:
        # Import the node
        from bubblelabs_nodes.decomposition_node import DecompositionNode
        print("[OK] Successfully imported DecompositionNode")

        # Create node instance
        node = DecompositionNode()
        print("[OK] Successfully created DecompositionNode instance")

        # Check if engine is available
        if hasattr(node, 'engine') and node.engine is not None:
            print(f"[OK] DecompositionEngine is available")
            print(f"   Available strategies: {getattr(node, 'available_strategies', 'unknown')}")
        else:
            print("[WARN] DecompositionEngine not available (will use fallback)")
            return False

        # Create mock inputs
        test_inputs = {
            'problem_statement': 'Design a scalable microservices architecture for an e-commerce platform that can handle 10,000 concurrent users',
            'method': 'semantic',  # Use semantic as it's confirmed to exist
            'requirements': {
                'scalability': 'high',
                'availability': '99.9%'
            },
            'constraints': {
                'budget': 'moderate',
                'timeline': '6 months'
            }
        }
        print("\n[INFO] Test Inputs:")
        print(f"   Problem: {test_inputs['problem_statement'][:60]}...")
        print(f"   Method: {test_inputs['method']}")

        # Validate inputs
        errors = node.validate_inputs(test_inputs)
        if errors:
            print(f"[FAIL] Validation failed: {errors}")
            return False
        print("[OK] Input validation passed")

        # Create mock context
        class MockContext:
            def __init__(self):
                self.progress = 0
                self.artifacts = {}

            def update_progress(self, percent, message):
                self.progress = percent
                print(f"   [{percent}%] {message}")

            def add_artifact(self, name, data):
                self.artifacts[name] = data

            def generate_execution_id(self):
                return 'test_exec_001'

        context = MockContext()
        print("\n[EXEC] Executing DecompositionNode...")

        # Execute the node
        result = node.execute(test_inputs, context)
        print("[OK] Execution completed successfully")

        # Verify output structure
        print("\n[RESULTS]")
        required_keys = [
            'sub_problems', 'decomposition_tree', 'complexity_metrics',
            'estimated_time', 'method_used', 'total_sub_problems',
            'confidence', 'plan_id', 'problem_id'
        ]

        for key in required_keys:
            if key in result:
                print(f"[OK] {key}: {result[key] if key != 'sub_problems' else f'{len(result[key])} items'}")
            else:
                print(f"[FAIL] Missing key: {key}")
                return False

        # Display sample sub-problem
        if result['sub_problems']:
            print("\n[INFO] Sample Sub-Problem:")
            sp = result['sub_problems'][0]
            print(f"   Title: {sp.get('title', 'N/A')}")
            print(f"   Description: {sp.get('description', 'N/A')[:60]}...")
            print(f"   Priority: {sp.get('priority', 'N/A')}")
            print(f"   Complexity: {sp.get('complexity', 'N/A')}")

        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED - DecompositionNode integration is working!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_decomposition_node()
    sys.exit(0 if success else 1)
=======
"""
Test script to verify DecompositionNode integration with existing DecompositionEngine
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_decomposition_node():
    """Test DecompositionNode with existing DecompositionEngine"""
    print("=" * 80)
    print("Testing DecompositionNode Integration")
    print("=" * 80)

    try:
        # Import the node
        from bubblelabs_nodes.decomposition_node import DecompositionNode
        print("[OK] Successfully imported DecompositionNode")

        # Create node instance
        node = DecompositionNode()
        print("[OK] Successfully created DecompositionNode instance")

        # Check if engine is available
        if hasattr(node, 'engine') and node.engine is not None:
            print(f"[OK] DecompositionEngine is available")
            print(f"   Available strategies: {getattr(node, 'available_strategies', 'unknown')}")
        else:
            print("[WARN] DecompositionEngine not available (will use fallback)")
            return False

        # Create mock inputs
        test_inputs = {
            'problem_statement': 'Design a scalable microservices architecture for an e-commerce platform that can handle 10,000 concurrent users',
            'method': 'semantic',  # Use semantic as it's confirmed to exist
            'requirements': {
                'scalability': 'high',
                'availability': '99.9%'
            },
            'constraints': {
                'budget': 'moderate',
                'timeline': '6 months'
            }
        }
        print("\n[INFO] Test Inputs:")
        print(f"   Problem: {test_inputs['problem_statement'][:60]}...")
        print(f"   Method: {test_inputs['method']}")

        # Validate inputs
        errors = node.validate_inputs(test_inputs)
        if errors:
            print(f"[FAIL] Validation failed: {errors}")
            return False
        print("[OK] Input validation passed")

        # Create mock context
        class MockContext:
            def __init__(self):
                self.progress = 0
                self.artifacts = {}

            def update_progress(self, percent, message):
                self.progress = percent
                print(f"   [{percent}%] {message}")

            def add_artifact(self, name, data):
                self.artifacts[name] = data

            def generate_execution_id(self):
                return 'test_exec_001'

        context = MockContext()
        print("\n[EXEC] Executing DecompositionNode...")

        # Execute the node
        result = node.execute(test_inputs, context)
        print("[OK] Execution completed successfully")

        # Verify output structure
        print("\n[RESULTS]")
        required_keys = [
            'sub_problems', 'decomposition_tree', 'complexity_metrics',
            'estimated_time', 'method_used', 'total_sub_problems',
            'confidence', 'plan_id', 'problem_id'
        ]

        for key in required_keys:
            if key in result:
                print(f"[OK] {key}: {result[key] if key != 'sub_problems' else f'{len(result[key])} items'}")
            else:
                print(f"[FAIL] Missing key: {key}")
                return False

        # Display sample sub-problem
        if result['sub_problems']:
            print("\n[INFO] Sample Sub-Problem:")
            sp = result['sub_problems'][0]
            print(f"   Title: {sp.get('title', 'N/A')}")
            print(f"   Description: {sp.get('description', 'N/A')[:60]}...")
            print(f"   Priority: {sp.get('priority', 'N/A')}")
            print(f"   Complexity: {sp.get('complexity', 'N/A')}")

        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED - DecompositionNode integration is working!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_decomposition_node()
    sys.exit(0 if success else 1)
>>>>>>> 1cb9c5e35 (update)
