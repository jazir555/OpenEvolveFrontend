#!/usr/bin/env python3
"""
Quick test to verify PyGraphistry-BubbleLab integration components exist.
"""

def test_integration_components():
    """Test that all integration components exist and are accessible."""
    print("Testing PyGraphistry-BubbleLab Integration Components...")
    
    # Test 1: Check if API endpoint exists
    print("\n1. Checking API endpoint availability...")
    try:
        from api_server import app
        routes = [route.path for route in app.routes]
        pygraphistry_routes = [route for route in routes if 'pygraphistry' in route]
        if pygraphistry_routes:
            print(f"   + Found PyGraphistry API endpoint: {pygraphistry_routes[0]}")
        else:
            print("   - PyGraphistry API endpoint not found")
    except Exception as e:
        print(f"   - Error checking API endpoint: {e}")
    
    # Test 2: Check if visualization module exists
    print("\n2. Checking visualization module...")
    try:
        from openevolve_visualization import get_pygraphistry_viz
        print("   + get_pygraphistry_viz function exists")
    except ImportError as e:
        print(f"   - get_pygraphistry_viz import failed: {e}")
    except Exception as e:
        print(f"   - Error with visualization module: {e}")
    
    # Test 3: Check if knowledge graph visualizer exists
    print("\n3. Checking KnowledgeGraphVisualizer...")
    try:
        from knowledge_graph_visualizer import KnowledgeGraphVisualizer
        print("   + KnowledgeGraphVisualizer exists")
    except ImportError as e:
        print(f"   - KnowledgeGraphVisualizer import failed: {e}")
    except Exception as e:
        print(f"   - Error with KnowledgeGraphVisualizer: {e}")
    
    # Test 4: Check if BubbleLabs integration exists
    print("\n4. Checking BubbleLabs integration...")
    try:
        from bubblelabs_integration import bubblelabs_integration
        if hasattr(bubblelabs_integration, 'get_knowledge_graph_visualization'):
            print("   + BubbleLabs integration with visualization method exists")
        else:
            print("   - BubbleLabs integration missing visualization method")
    except ImportError as e:
        print(f"   - BubbleLabs integration import failed: {e}")
    except Exception as e:
        print(f"   - Error with BubbleLabs integration: {e}")
    
    # Test 5: Check if PyGraphistry bridge exists
    print("\n5. Checking PyGraphistry bridge...")
    try:
        from integrations.pygraphistry.bridge import PygraphistryBridge
        print("   + PyGraphistryBridge exists")
    except ImportError as e:
        print(f"   - PyGraphistryBridge import failed: {e}")
    except Exception as e:
        print(f"   - Error with PyGraphistryBridge: {e}")

    # Test 6: Check if PyGraphistry adapter exists
    print("\n6. Checking PyGraphistry adapter...")
    try:
        from integrations.pygraphistry.adapter import PygraphistryAdapter
        print("   + PyGraphistryAdapter exists")
    except ImportError as e:
        print(f"   - PyGraphistryAdapter import failed: {e}")
    except Exception as e:
        print(f"   - Error with PyGraphistryAdapter: {e}")
    
    print("\n" + "="*60)
    print("Integration Component Test Complete")
    print("="*60)
    print("\nIf all components are marked with '+', the integration is properly set up.")
    print("The actual visualization functionality depends on PyGraphistry installation.")

if __name__ == "__main__":
    test_integration_components()