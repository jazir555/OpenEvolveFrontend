
import sys
import os

# Add current directory to sys.path so we can import knowledge_engine
sys.path.append(os.getcwd())

try:
    from knowledge_engine.integrations.ragbits_integration import RagbitsIntegration
    print("Import successful")
    
    # Instantiate to trigger _initialize_components
    # We expect it to fail importing ragbits and call _initialize_mock_components
    ri = RagbitsIntegration()
    print("Instantiation successful")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
