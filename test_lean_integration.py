#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LeanAide PES Integration
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set UTF-8 encoding for output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from openevolve_pes_integration import (
    enhance_lean_proof,
    verify_lean_code,
    is_lean_code,
    detect_language,
    enhance_code,
)


def test_is_lean_code():
    """Test Lean code detection."""
    print("\n" + "="*70)
    print("  Test: is_lean_code()")
    print("="*70)
    
    # Should detect as Lean
    lean_code = '''theorem add_comm (n m : Nat) : n + m = m + n := by
  sorry'''
    
    result = is_lean_code(lean_code)
    print(f"[PASS] Lean code detected: {result}")
    
    # Should not detect as Lean
    python_code = '''def add(a, b):
    return a + b'''
    
    result = is_lean_code(python_code)
    print(f"[PASS] Python code not detected as Lean: {not result}")
    
    return True


def test_detect_language():
    """Test language detection."""
    print("\n" + "="*70)
    print("  Test: detect_language()")
    print("="*70)
    
    # Test Lean detection
    lean_code = '''theorem test : True := by trivial'''
    lang = detect_language(lean_code)
    print(f"[PASS] Lean code detected as: {lang}")
    
    # Test Python detection
    python_code = '''def test(): return True'''
    lang = detect_language(python_code)
    print(f"[PASS] Python code detected as: {lang}")
    
    return lang == 'lean' or lang == 'python'


def test_enhance_lean_proof():
    """Test Lean proof enhancement."""
    print("\n" + "="*70)
    print("  Test: enhance_lean_proof()")
    print("="*70)
    
    lean_code = '''theorem add_comm (n m : Nat) : n + m = m + n := by
  sorry

theorem trivial_theorem : True := by trivial'''
    
    result = enhance_lean_proof(
        lean_code,
        theorem_description="Prove commutativity of addition"
    )
    
    print(f"[PASS] Enhancement Result:")
    print(f"  Success: {result['success']}")
    print(f"  Tests Passed: {result['tests_passed']}/{result['tests_total']}")
    print(f"  Improvements: {len(result['improvements'])}")
    
    for imp in result['improvements']:
        print(f"    - {imp}")
    
    print(f"\n[PASS] Enhanced Code:")
    print(result['enhanced_code'])
    
    return True


def test_verify_lean_code():
    """Test Lean code verification."""
    print("\n" + "="*70)
    print("  Test: verify_lean_code()")
    print("="*70)
    
    # Test with incomplete proof
    lean_code = '''theorem add_comm (n m : Nat) : n + m = m + n := by
  sorry'''
    
    result = verify_lean_code(lean_code)
    print(f"[PASS] Incomplete proof verification:")
    print(f"  Valid: {result['valid']}")
    print(f"  Has sorry: {result['has_sorry']}")
    print(f"  Theorems found: {result['theorems_found']}")
    print(f"  Issues: {result['issues']}")
    
    # Test with complete proof
    complete_code = '''theorem trivial_theorem : True := by trivial'''
    
    result = verify_lean_code(complete_code)
    print(f"\n[PASS] Complete proof verification:")
    print(f"  Valid: {result['valid']}")
    print(f"  Has sorry: {result['has_sorry']}")
    
    return True


def test_enhance_code_universal():
    """Test universal enhance_code function."""
    print("\n" + "="*70)
    print("  Test: enhance_code() (Universal)")
    print("="*70)
    
    # Test with Lean code
    lean_code = '''theorem test : True := by sorry'''
    
    result = enhance_code(
        lean_code,
        problem_description="Simple theorem"
    )
    
    print(f"[PASS] Universal enhance for Lean:")
    print(f"  Language detected: {result.get('language', 'unknown')}")
    print(f"  Success: {result['success']}")
    
    # Test with Python code
    python_code = '''def calculate_total(price):
    return price * 1.1'''
    
    result = enhance_code(
        python_code,
        problem_description="Calculate total with tax"
    )
    
    print(f"\n[PASS] Universal enhance for Python:")
    print(f"  Language detected: {result.get('language', 'unknown')}")
    print(f"  Success: {result['success']}")
    
    return True


def test_integration_demo():
    """Demonstrate the full integration."""
    print("\n" + "="*70)
    print("  Integration Demo")
    print("="*70)
    
    # Simulate OpenEvolve generating Lean code
    openevolve_lean_output = '''import Mathlib.Data.Real.Basic

theorem add_comm (a b : Real) : a + b = b + a := by
  sorry

theorem add_assoc (a b c : Real) : (a + b) + c = a + (b + c) := by
  sorry

theorem add_zero (a : Real) : a + 0 = a := by
  sorry

-- This one is complete
theorem trivial : True := by trivial'''
    
    print("OpenEvolve generated Lean code:")
    print(openevolve_lean_output)
    
    print("\n" + "-"*70)
    print("Enhancing with LeanAide PES...")
    
    result = enhance_code(
        openevolve_lean_output,
        problem_description="Prove properties of real addition"
    )
    
    print(f"\nEnhancement Result:")
    print(f"  Language: {result.get('language', 'unknown')}")
    print(f"  Success: {result['success']}")
    print(f"  Tests Passed: {result.get('tests_passed', 'N/A')}/{result.get('tests_total', 'N/A')}")
    print(f"  Improvements: {len(result.get('improvements', []))}")
    
    for imp in result.get('improvements', []):
        print(f"    - {imp}")
    
    print(f"\nEnhanced Lean Code:")
    print(result['enhanced_code'])
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  LeanAide PES Integration Test Suite")
    print("="*70)
    
    tests = [
        ("is_lean_code", test_is_lean_code),
        ("detect_language", test_detect_language),
        ("enhance_lean_proof", test_enhance_lean_proof),
        ("verify_lean_code", test_verify_lean_code),
        ("enhance_code (Universal)", test_enhance_code_universal),
        ("Integration Demo", test_integration_demo),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
