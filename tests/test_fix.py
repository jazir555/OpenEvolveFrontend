#!/usr/bin/env python3
"""
Test script to verify the integrated_workflow.py fixes
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_integrated_workflow_imports():
    """Test that integrated_workflow imports correctly"""
    from integrated_workflow import run_fully_integrated_adversarial_evolution
    from integrated_workflow import analyze_with_model

    # Test that the functions are callable
    assert callable(run_fully_integrated_adversarial_evolution)
    assert callable(analyze_with_model)

if __name__ == "__main__":
    try:
        test_integrated_workflow_imports()
        print("[OK] SUCCESS: integrated_workflow imports successfully!")
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        sys.exit(1)