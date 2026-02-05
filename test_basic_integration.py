"""
Simple test to verify PyGraphistry integration components exist.
"""

def test_basic_integration():
    print("Testing basic integration components...")
    
    # Test 1: Check if knowledge_graph_visualizer has pygraphistry support
    try:
        from knowledge_graph_visualizer import KnowledgeGraphVisualizer
        print("[OK] KnowledgeGraphVisualizer import successful")
        
        # Create instance with pygraphistry parameter
        viz = KnowledgeGraphVisualizer(use_pygraphistry=True)
        print("[OK] KnowledgeGraphVisualizer with use_pygraphistry parameter works")
        
        # Check if pygraphistry bridge attribute exists
        if hasattr(viz, 'pygraphistry_bridge'):
            print("[OK] pygraphistry_bridge attribute exists")
        else:
            print("[FAIL] pygraphistry_bridge attribute missing")
            
        # Check if new methods exist
        if hasattr(viz, 'analyze_patterns_with_pygraphistry'):
            print("[OK] analyze_patterns_with_pygraphistry method exists")
        else:
            print("[FAIL] analyze_patterns_with_pygraphistry method missing")
            
        if hasattr(viz, 'connect_pygraphistry'):
            print("[OK] connect_pygraphistry method exists")
        else:
            print("[FAIL] connect_pygraphistry method missing")
            
    except (ImportError, ValueError, AttributeError, TypeError) as e:
        print(f"[FAIL] KnowledgeGraphVisualizer test failed: {e}")
    
    # Test 2: Check if visualization function exists
    try:
        from openevolve_visualization import get_pygraphistry_viz
        print("[OK] get_pygraphistry_viz function import successful")
    except (ImportError, AttributeError) as e:
        print(f"[FAIL] get_pygraphistry_viz import failed: {e}")
    
    # Test 3: Check if integration factory works
    try:
        from integrations import IntegrationFactory
        print("[OK] IntegrationFactory import successful")
    except (ImportError, AttributeError) as e:
        print(f"[FAIL] IntegrationFactory import failed: {e}")
    
    # Test 4: Check if API endpoint exists
    try:
        with open("openevolve_api.py", 'r') as f:
            content = f.read()
            if "/api/openevolve/visualize/pygraphistry" in content:
                print("[OK] PyGraphistry API endpoint exists")
            else:
                print("[FAIL] PyGraphistry API endpoint missing")
    except (OSError, IOError) as e:
        print(f"[FAIL] API endpoint check failed: {e}")
    
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
            print(f"[OK] {file} exists")
        else:
            print(f"[FAIL] {file} missing")
            all_exist = False
    
    if all_exist:
        print("[OK] All PyGraphistry integration files present")
    
    print("\nBasic integration test complete!")

if __name__ == "__main__":
    test_basic_integration()