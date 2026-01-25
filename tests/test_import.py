
import sys
from unittest.mock import MagicMock, patch

# Mock EVERYTHING
sys.modules['llm_utils'] = MagicMock()
sys.modules['ace_steer_integration'] = MagicMock()
sys.modules['hephaestus'] = MagicMock()
sys.modules['bubblelabs_integration'] = MagicMock()
sys.modules['bubblelabs_analytics'] = MagicMock()
sys.modules['bubblelabs_hephaestus_bridge'] = MagicMock()
sys.modules['hephaestus_integration'] = MagicMock()
sys.modules['mdap_engine'] = MagicMock()
sys.modules['maker_engine'] = MagicMock()
# Real module for workflow_structures
sys.path.append('c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend')
# sys.modules['workflow_structures'] = MagicMock() # Use real one 

print("Attempting import...")
try:
    sys.path.append('c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend')
    import openevolve_workflow_manager_integrated
    print("Import SUCCESS")
except Exception as e:
    with open("import_error.log", "w") as f:
        f.write(f"Import FAILED: {e}\n")
        import traceback
        traceback.print_exc(file=f)
    print("Import FAILED")
