#!/usr/bin/env python3
"""Verify LeanAide integration works correctly."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from leanaide_pes_handler import LeanCodeAnalyzer, LeanPESHandler, complete_lean_proof, ProofStrategySelector

# Test 1: Theorem detection
print("=" * 60)
print("TEST 1: Theorem Detection")
print("=" * 60)
code = 'theorem add_comm (n m : Nat) : n + m = m + n := by sorry'
analysis = LeanCodeAnalyzer.analyze_structure(code)
print(f"Input: {code}")
print(f"Theorems found: {len(analysis['theorems'])}")
print(f"Has sorry: {analysis['has_sorry']}")
assert len(analysis['theorems']) == 1, "Should find 1 theorem"
assert analysis['has_sorry'] == True, "Should detect sorry"
print("PASS")

# Test 2: Generate proof
print("\n" + "=" * 60)
print("TEST 2: Proof Generation")
print("=" * 60)
selector = ProofStrategySelector()
thm = {
    'name': 'add_comm',
    'goal': 'n + m = m + n',
    'hypotheses': [{'name': 'n', 'type': 'Nat'}, {'name': 'm', 'type': 'Nat'}],
    'uses_sorry': True
}
proof = selector.generate_proof(thm)
print(f"Theorem: {thm['name']}")
print(f"Goal: {thm['goal']}")
print(f"Generated proof:\n  {proof}")
assert 'intro' in proof or 'simp' in proof or 'rfl' in proof, "Should generate actual tactics"
print("PASS")

# Test 3: PES Handler
print("\n" + "=" * 60)
print("TEST 3: PES Handler Plan")
print("=" * 60)
handler = LeanPESHandler()
plan = handler.plan(code)
print(f"Theorems needing proof: {plan['theorems_needing_proof']}")
print(f"Strategies: {plan['recommended_strategies']}")
assert plan['theorems_needing_proof'] > 0, "Should have theorems needing proof"
print("PASS")

# Test 4: Complete proof
print("\n" + "=" * 60)
print("TEST 4: Complete Proof")
print("=" * 60)
code_with_proof = complete_lean_proof(code)
print(f"Original: {code}")
print(f"Completed:\n{code_with_proof}")
assert 'sorry' not in code_with_proof, "Should replace sorry"
print("PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - LeanAide Integration Complete!")
print("=" * 60)
