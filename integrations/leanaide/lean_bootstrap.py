"""
Lean 4 Integration Bootstrap

This module ensures Python paths are correctly set up for Lean imports.
Import this module BEFORE importing any Lean-related modules to ensure
LEAN_AVAILABLE becomes True in all wired files.

Usage:
    import lean_bootstrap  # Must be first!
    from leanaide_client import LeanAideClient
    
Or run as script:
    python lean_bootstrap.py --verify
"""

import sys
import os
from pathlib import Path

def _setup_lean_paths():
    """Set up Python paths for Lean imports."""
    # Get project root
    project_root = Path(__file__).parent.resolve()
    
    paths_to_add = [
        str(project_root),
        str(project_root / "openevolve"),
        str(project_root / "glue" / "lib"),
        str(project_root / "glue" / "lib" / "lean4_bridge"),
        str(project_root / "knowledge_engine"),
    ]
    
    # Add to beginning of path (higher priority)
    for path in reversed(paths_to_add):
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Set environment variables
    if 'LEAN_EXECUTABLE' not in os.environ:
        # Try to find lean
        lean_paths = [
            Path.home() / ".elan" / "bin" / "lean.exe",
            Path.home() / ".elan" / "bin" / "lean",
            Path("/usr/local/bin/lean"),
            Path("/usr/bin/lean"),
        ]
        for lean_path in lean_paths:
            if lean_path.exists():
                os.environ['LEAN_EXECUTABLE'] = str(lean_path)
                break
    
    if 'MATHLIB_PATH' not in os.environ:
        mathlib_paths = [
            project_root / "lean_workspace" / "mathlib_project",
            project_root / "mathlib_project",
        ]
        for mathlib_path in mathlib_paths:
            if mathlib_path.exists():
                os.environ['MATHLIB_PATH'] = str(mathlib_path)
                break

def verify_lean_imports():
    """Verify that all Lean imports work correctly."""
    results = {
        "success": True,
        "imports": {},
        "errors": []
    }
    
    # Test basic imports
    test_cases = [
        ("leanaide_client", "LeanAideClient"),
        ("leanaide_config", "LeanAideConfig"),
        ("lean4_integration", "Lean4VerificationEngine"),
        ("leanaide_integration", "LeanAIDEIntegration"),
        ("config", "LeanAideConfig"),
    ]
    
    for module_name, class_name in test_cases:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            results["imports"][module_name] = {
                "imported": True,
                "class_available": cls is not None
            }
        except Exception as e:
            results["imports"][module_name] = {
                "imported": False,
                "error": str(e)
            }
            results["errors"].append(f"{module_name}: {e}")
            results["success"] = False
    
    # Check LEAN_AVAILABLE flags
    lean_available_checks = [
        "leanaide_client",
        "leanaide_integration",
        "lean4_integration",
    ]
    
    for module_name in lean_available_checks:
        try:
            module = __import__(module_name)
            lean_available = getattr(module, 'LEAN_AVAILABLE', 'NOT_SET')
            results["imports"][module_name]["LEAN_AVAILABLE"] = lean_available
        except Exception as e:
            results["imports"][module_name]["LEAN_AVAILABLE"] = f"Error: {e}"
    
    return results

# Set up paths on module import
_setup_lean_paths()

if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Lean 4 integration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    results = verify_lean_imports()
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("="*60)
        print("LEAN 4 INTEGRATION VERIFICATION")
        print("="*60)
        overall = "SUCCESS" if results['success'] else "FAILED"
        print(f"\nOverall: {overall}")
        
        print("\n--- Module Imports ---")
        for module, status in results["imports"].items():
            imported = status.get("imported", False)
            lean_avail = status.get("LEAN_AVAILABLE", "N/A")
            symbol = "[OK]" if imported else "[FAIL]"
            print(f"  {symbol} {module}: imported={imported}, LEAN_AVAILABLE={lean_avail}")
        
        if results["errors"]:
            print("\n--- Errors ---")
            for error in results["errors"]:
                print(f"  [ERROR] {error}")
        
        print("\n" + "="*60)
    
    sys.exit(0 if results["success"] else 1)
