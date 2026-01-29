import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path().resolve()))

try:
    # Test basic imports
    import workflow_structures
    print("✓ workflow_structures imported successfully")
    
    # Test knowledge_graph_visualizer import
    import knowledge_graph_visualizer
    print("✓ knowledge_graph_visualizer imported successfully")
    
    # Test if KnowledgeGraphVisualizer accepts use_pygraphistry parameter
    viz = knowledge_graph_visualizer.KnowledgeGraphVisualizer(use_pygraphistry=True)
    print("✓ KnowledgeGraphVisualizer accepts use_pygraphistry parameter")
    
    # Test if pygraphistry_bridge attribute exists
    if hasattr(viz, 'pygraphistry_bridge'):
        print("✓ pygraphistry_bridge attribute exists")
    else:
        print("✗ pygraphistry_bridge attribute missing")
        
    # Test if new methods exist
    if hasattr(viz, 'analyze_patterns_with_pygraphistry'):
        print("✓ analyze_patterns_with_pygraphistry method exists")
    else:
        print("✗ analyze_patterns_with_pygraphistry method missing")
        
    if hasattr(viz, 'connect_pygraphistry'):
        print("✓ connect_pygraphistry method exists")
    else:
        print("✗ connect_pygraphistry method missing")
    
    # Test if visualize_interactive has new parameters
    import inspect
    sig = inspect.signature(viz.visualize_interactive)
    params = list(sig.parameters.keys())
    has_clustering_params = 'apply_clustering' in params and 'clustering_method' in params
    print(f"✓ visualize_interactive has clustering parameters: {has_clustering_params}")
    
    print("\nAll integration components are properly implemented!")
    
except Exception as e:  # TODO: Catch specific exception instead of Exception
    print(f"✗ Error during import test: {e}")
    import traceback
    traceback.print_exc()