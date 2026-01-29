"""
Simple test to verify PyGraphistry integration components exist.
"""

def test_basic_integration():
    print("Testing basic integration components...")
    
    # Test 1: Check if knowledge_graph_visualizer has pygraphistry support
    try:
        from knowledge_graph_visualizer import KnowledgeGraphVisualizer
        print("✓ KnowledgeGraphVisualizer import successful")
        
        # Create instance with pygraphistry parameter
        viz = KnowledgeGraphVisualizer(use_pygraphistry=True)
        print("✓ KnowledgeGraphVisualizer with use_pygraphistry parameter works")
        
        # Check if pygraphistry bridge attribute exists
        if hasattr(viz, 'pygraphistry_bridge'):
            print("✓ pygraphistry_bridge attribute exists")
        else:
            print("✗ pygraphistry_bridge attribute missing")
            
        # Check if new methods exist
        if hasattr(viz, 'analyze_patterns_with_pygraphistry'):
            print("✓ analyze_patterns_with_pygraphistry method exists")
        else:
            print("✗ analyze_patterns_with_pygraphistry method missing")
            
        if hasattr(viz, 'connect_pygraphistry'):
            print("✓ connect_pygraphistry method exists")
        else:
            print("✗ connect_pygraphistry method missing")
            
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"✗ KnowledgeGraphVisualizer test failed: {e}")
    
    # Test 2: Check if visualization function exists
    try:
        from openevolve_visualization import get_pygraphistry_viz
        print("✓ get_pygraphistry_viz function import successful")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"✗ get_pygraphistry_viz import failed: {e}")
    
    # Test 3: Check if integration factory works
    try:
        from integrations import IntegrationFactory
        print("✓ IntegrationFactory import successful")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"✗ IntegrationFactory import failed: {e}")
    
    # Test 4: Check if API endpoint exists
    try:
        with open("openevolve_api.py", 'r') as f:
            content = f.read()
            if "/api/openevolve/visualize/pygraphistry" in content:
                print("✓ PyGraphistry API endpoint exists")
            else:
                print("✗ PyGraphistry API endpoint missing")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"✗ API endpoint check failed: {e}")
    
    # Test 5: Check if pygraphistry files exist
    import os
    pygraphistry_files = [
        "integrations/pygraphistry/adapter.py",
        "integrations/pygraphistry/bridge.py",
        "integrations/pygraphistry/config.yaml"
    ]
    
    all_exist = True
    for file in pygraphistry_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    if all_exist:
        print("✓ All PyGraphistry integration files present")
    
    print("\nBasic integration test complete!")

if __name__ == "__main__":
    test_basic_integration()