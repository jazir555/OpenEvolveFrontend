#!/usr/bin/env python3
"""
Test script to verify PyGraphistry integration with BubbleLab.
"""

import asyncio
import tempfile
import os
from typing import Dict, Any, List

async def test_pygraphistry_bubblelab_integration():
    """Test the complete PyGraphistry integration with BubbleLab."""
    print("Testing PyGraphistry integration with BubbleLab...")
    
    # Test 1: Check if knowledge graph visualizer can be imported with PyGraphistry
    print("\n1. Testing KnowledgeGraphVisualizer with PyGraphistry...")
    try:
        from knowledge_graph_visualizer import KnowledgeGraphVisualizer
        viz = KnowledgeGraphVisualizer(use_pygraphistry=True)
        print("   ✓ KnowledgeGraphVisualizer with PyGraphistry initialized successfully")
        
        # Test building a simple graph
        stats = viz.build_graph(max_nodes=10)
        print(f"   Graph stats: {stats}")
        
        # Test visualization
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            temp_path = tmp.name
            
        result = viz.visualize_interactive(
            output_path=temp_path,
            apply_clustering=True,
            clustering_method="dbscan",
            embedding_method="umap"
        )
        
        if os.path.exists(temp_path):
            print(f"   ✓ Visualization created at: {temp_path}")
            os.unlink(temp_path)  # Clean up
        else:
            print("   ⚠ Visualization file not created (expected if PyGraphistry not configured)")
            
    except ImportError as e:
        print(f"   X KnowledgeGraphVisualizer import failed: {e}")
    except Exception as e:
        print(f"   X Error in KnowledgeGraphVisualizer test: {e}")

    # Test 2: Check if get_pygraphistry_viz function exists and works
    print("\n2. Testing get_pygraphistry_viz function...")
    try:
        from openevolve_visualization import get_pygraphistry_viz
        
        # Create sample nodes and edges
        sample_nodes = [
            {"id": "node1", "label": "Node 1", "type": "test", "value": 10},
            {"id": "node2", "label": "Node 2", "type": "test", "value": 20},
            {"id": "node3", "label": "Node 3", "type": "test", "value": 15}
        ]
        
        sample_edges = [
            {"source": "node1", "target": "node2", "relationship": "connected"},
            {"source": "node2", "target": "node3", "relationship": "connected"}
        ]
        
        result = await get_pygraphistry_viz(sample_nodes, sample_edges)
        if result:
            print(f"   ✓ get_pygraphistry_viz returned: {result}")
            if os.path.exists(result):
                print("   ✓ Visualization file exists")
                os.unlink(result)  # Clean up
        else:
            print("   ⚠ get_pygraphistry_viz returned None (expected if PyGraphistry not configured)")
            
    except ImportError as e:
        print(f"   X get_pygraphistry_viz import failed: {e}")
    except Exception as e:
        print(f"   X Error in get_pygraphistry_viz test: {e}")

    # Test 3: Check if BubbleLabs integration has visualization method
    print("\n3. Testing BubbleLabs integration visualization method...")
    try:
        from bubblelabs_integration import bubblelabs_integration

        result = bubblelabs_integration.get_knowledge_graph_visualization(
            use_pygraphistry=True,
            max_nodes=10
        )

        if result:
            print(f"   + BubbleLabs integration visualization returned: {result}")
            if os.path.exists(result):
                print("   + Visualization file exists")
                os.unlink(result)  # Clean up
        else:
            print("   ~ BubbleLabs integration visualization returned None (expected if PyGraphistry not configured)")

    except ImportError as e:
        print(f"   X BubbleLabs integration import failed: {e}")
    except Exception as e:
        print(f"   X Error in BubbleLabs integration test: {e}")

    # Test 4: Check if API endpoint would work (without actually calling it)
    print("\n4. Testing API endpoint availability...")
    try:
        from api_server import app
        # Check if the route exists by looking at the routes
        routes = [route.path for route in app.routes]
        pygraphistry_routes = [route for route in routes if 'pygraphistry' in route]
        if pygraphistry_routes:
            print(f"   + PyGraphistry API endpoint available: {pygraphistry_routes}")
        else:
            print("   X PyGraphistry API endpoint not found")
    except Exception as e:
        print(f"   X Error checking API endpoint: {e}")
    
    print("\n" + "="*60)
    print("PyGraphistry-BubbleLab Integration Test Complete")
    print("="*60)
    print("\nSummary:")
    print("- PyGraphistry should be available for knowledge graph visualization")
    print("- API endpoint /api/openevolve/visualize/pygraphistry should be accessible")
    print("- BubbleLab can request visualizations through the API")
    print("- Clustering and embedding capabilities should be available if dependencies are installed")


if __name__ == "__main__":
    asyncio.run(test_pygraphistry_bubblelab_integration())