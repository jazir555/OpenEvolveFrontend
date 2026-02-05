#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick import test for RESE Health APIs

This script verifies that all health API modules can be imported successfully.

Usage:
    python test_health_api_imports.py

Author: RESE Team
Created: 2026-02-04
"""

import sys
import io
from pathlib import Path

# Set stdout to UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add paths
frontend_dir = Path(__file__).parent
sys.path.insert(0, str(frontend_dir / "glue" / "adapters" / "rese-phase1" / "src"))
sys.path.insert(0, str(frontend_dir / "glue" / "adapters" / "rese-phase2" / "src"))
sys.path.insert(0, str(frontend_dir / "glue" / "adapters" / "rese-phase3" / "src"))
sys.path.insert(0, str(frontend_dir / "glue" / "adapters" / "rese-phase4" / "src"))
sys.path.insert(0, str(frontend_dir / "glue" / "adapters" / "rese-integration" / "health"))


def test_import(module_name, module_path):
    """Test importing a module."""
    try:
        # Import module
        if module_path:
            sys.path.insert(0, str(Path(module_path).parent))
            module = __import__(Path(module_path).stem)
        else:
            module = __import__(module_name)

        # Check if app exists
        if hasattr(module, 'app'):
            return True, "PASS: App object found"
        else:
            return False, "FAIL: App object not found"

    except ImportError as e:
        return False, f"FAIL: Import error: {e}"
    except Exception as e:
        return False, f"FAIL: Error: {e}"


def main():
    """Run all import tests."""
    print("=" * 80)
    print("RESE Health API Import Tests")
    print("=" * 80)
    print()

    tests = [
        ("Phase I Health API", "health_api", frontend_dir / "glue" / "adapters" / "rese-phase1" / "src" / "health_api.py"),
        ("Phase II Health API", "health_api", frontend_dir / "glue" / "adapters" / "rese-phase2" / "src" / "health_api.py"),
        ("Phase III Health API", "health_api", frontend_dir / "glue" / "adapters" / "rese-phase3" / "src" / "health_api.py"),
        ("Phase IV Health API", "health_api", frontend_dir / "glue" / "adapters" / "rese-phase4" / "src" / "health_api.py"),
        ("Aggregate Health API", "aggregate_health", frontend_dir / "glue" / "adapters" / "rese-integration" / "health" / "aggregate_health.py"),
    ]

    passed = 0
    failed = 0

    for test_name, module_name, module_path in tests:
        print(f"Testing {test_name}...")
        print(f"  Path: {module_path}")

        success, message = test_import(module_name, str(module_path))
        print(f"  {message}")

        if success:
            passed += 1
            print(f"  {test_name}: PASS")
        else:
            failed += 1
            print(f"  {test_name}: FAIL")

        print()

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print("All import tests passed!")
        return 0
    else:
        print("Some import tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
