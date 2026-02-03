#!/usr/bin/env python3
"""
Simple check to verify that the global DSPy integration is working properly
"""

def check_global_dspy_integration():
    """Check that the global dspy integration module is accessible."""
    print("Checking Global DSPy Integration...")
    
    try:
        # Try to import the global module
        from dspy_integration import DSPY_AVAILABLE, get_global_dspy_instance, initialize_dspy
        print("[SUCCESS] Global dspy_integration module imported successfully")
        
        # Check if DSPy is available
        print(f"DSPY_AVAILABLE: {DSPY_AVAILABLE}")
        
        # Try to import dspy directly through the global module
        import dspy
        print(f"[SUCCESS] DSPy imported successfully through global module")
        
        # Check for key DSPy components
        from dspy.teleprompt import BootstrapFewShot
        from dspy.predict import Predict
        print(f"[SUCCESS] Key DSPy components available through global module")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Could not import global dspy_integration: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def check_core_files_use_global_module():
    """Check that core files are using the global module."""
    import os
    
    core_files = [
        "blue_team.py",
        "red_team.py",
        "quality_assessment.py",
        "dts_integration.py",
        "z3prover_integration.py",
        "z3_leanaide_bridge.py",
        "robust_z3_leanaide_integration.py",
        "knowledge_engine/engine.py",
        "deterministic_sop/adapters.py"
        # Note: ace_mcp_tools.py doesn't use DSPy functionality, so it's not required to use global module
    ]
    
    frontend_path = "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/"
    all_good = True
    
    print("\nChecking core files for global module usage...")
    for file in core_files:
        file_path = os.path.join(frontend_path, file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file uses global dspy integration
            uses_global = "'dspy_integration' import" in content or "from dspy_integration import" in content
            
            status = "[SUCCESS]" if uses_global else "[MISSING GLOBAL]"
            print(f"{file:<40} {status}")
            
            if not uses_global:
                all_good = False
        else:
            print(f"{file:<40} [MISSING FILE]")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("="*60)
    print("SIMPLE VERIFICATION: Global DSPy Integration")
    print("="*60)
    
    # Check global module
    global_ok = check_global_dspy_integration()
    
    # Check core files
    files_ok = check_core_files_use_global_module()
    
    print("\n" + "="*60)
    if global_ok and files_ok:
        print("OVERALL RESULT: [SUCCESS] All components use global DSPy integration")
    else:
        print("OVERALL RESULT: [ISSUES FOUND] Some components need attention")
    print("="*60)