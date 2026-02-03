#!/usr/bin/env python3
"""
Final verification script to ensure all core OpenEvolve components use the global DSPy integration module.
"""

import sys
import os

# Add the frontend directory to the path
frontend_path = "C:/Users/mmeadow/Documents/OpenEvolve/Frontend"
sys.path.insert(0, frontend_path)

def verify_global_dspy_usage():
    """Verify that core components use the global dspy integration."""
    print("Verifying Global DSPy Integration Usage")
    print("="*50)
    
    core_files = [
        "blue_team.py",
        "dts_integration.py", 
        "quality_assessment.py",
        "red_team.py",
        "solution_pattern_miner.py",
        "workflow_knowledge_extractor.py",
        "z3prover_integration.py",
        "z3_leanaide_bridge.py",
        "robust_z3_leanaide_integration.py",
        "knowledge_engine/engine.py",
        "deterministic_sop/adapters.py"
    ]
    
    results = []
    
    for file in core_files:
        file_path = os.path.join(frontend_path, file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file uses global dspy integration
            uses_global = "'dspy_integration' import" in content or "from dspy_integration import" in content
            has_fallback = "except ImportError:" in content and "import dspy" in content.split("except ImportError:")[-1]
            
            status = "[GLOBAL]" if uses_global else "[LOCAL]"
            fallback_status = "[FALLBACK_OK]" if has_fallback else "[NO_FALLBACK]"

            results.append((file, status, fallback_status, uses_global))
            print(f"{file:<40} {status} {fallback_status}")
        else:
            results.append((file, "❌ MISSING", "❌ MISSING", False))
            print(f"{file:<40} ❌ MISSING")
    
    print("\n" + "="*50)
    global_count = sum(1 for _, status, _, _ in results if "GLOBAL" in status)
    total_count = len(results)

    print(f"SUMMARY: {global_count}/{total_count} files use global DSPy integration")

    if global_count == total_count:
        print("[SUCCESS] ALL CORE FILES USE GLOBAL DSPY INTEGRATION!")
        return True
    else:
        print("[WARNING] Some files still need to be updated")
        return False

if __name__ == "__main__":
    success = verify_global_dspy_usage()
    if success:
        print("\nVerification completed successfully!")
    else:
        print("\nSome files need updating to use global DSPy integration.")