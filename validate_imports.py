#!/usr/bin/env python3
"""
Import Validator for OpenEvolve Frontend
Run this script to validate all imports before starting services
"""

import sys
import importlib
from typing import List, Tuple

def check_import(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return False, f"Error: {e}"


def main():
    """Validate all critical imports"""
    print("=" * 60)
    print("OPENEVOLVE IMPORT VALIDATOR")
    print("=" * 60)
    print()

    critical_imports = [
        ("openevolve_structures", "Core data structures"),
        ("team_manager", "Team management"),
        ("gauntlet_manager", "Gauntlet management"),
        ("decomposition_engine", "Problem decomposition"),
        ("ace_mcp_tools", "ACE MCP tools"),
        ("openevolve_mcp_tools", "OpenEvolve MCP tools"),
        ("steer_mcp_tools", "Steer verification (optional)"),
    ]

    optional_imports = [
        ("steer.core", "Steer core (optional)"),
        ("roma_dspy", "ROMA decomposition (optional)"),
        ("datapizza.agents", "DataPizza (optional)"),
        ("leanaide_client", "LeanAide (optional)"),
    ]

    all_ok = True

    print("CRITICAL IMPORTS:")
    print("-" * 60)
    for module, description in critical_imports:
        ok, msg = check_import(module)
        status = "OK" if ok else "FAIL"
        print(f"{status} {module:30s} - {description}")
        if not ok:
            print(f"  Error: {msg}")
            all_ok = False

    print()
    print("OPTIONAL IMPORTS:")
    print("-" * 60)
    for module, description in optional_imports:
        ok, msg = check_import(module)
        status = "OK" if ok else "○"
        print(f"{status} {module:30s} - {description}")
        if not ok:
            print(f"  Note: {msg[:80]}")

    print()
    print("=" * 60)
    if all_ok:
        print("OK All critical imports successful!")
        print("  Ready to start OpenEvolve services.")
        return 0
    else:
        print("FAIL Some critical imports failed!")
        print("  Please fix the errors above before starting services.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
