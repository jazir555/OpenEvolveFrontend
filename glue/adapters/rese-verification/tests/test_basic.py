"""
Basic functionality test for tiered verification system
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification_result import (
    VerificationTier,
    VerificationStatus,
    ProblemClass,
    ProblemDomain,
    Z3VerificationResult,
    LeanAideVerificationResult,
    Lean4VerificationResult,
    UnifiedVerificationResult,
)

print("=" * 70)
print("BASIC FUNCTIONALITY TEST")
print("=" * 70)

# Test 1: Create Z3 result
print("\n1. Testing Z3 Verification Result...")
z3_result = Z3VerificationResult(
    status=VerificationStatus.VERIFIED,
    z3_result="sat",
    model={"x": "1"},
    execution_time_ms=100.0,
    constraints_checked=5,
    correlation_id="test-123",
)
assert z3_result.is_successful()
assert not z3_result.should_escalate()
print("   ✓ Z3 result created successfully")

# Test 2: Create LeanAide result
print("\n2. Testing LeanAide Verification Result...")
leanaide_result = LeanAideVerificationResult(
    status=VerificationStatus.VERIFIED,
    proof_status="proved",
    proof_script="theorem test : True := by trivial",
    tactics_used=["trivial"],
    execution_time_ms=5000.0,
    constraints_checked=50,
    correlation_id="test-123",
)
assert leanaide_result.is_successful()
assert not leanaide_result.should_escalate()
print("   ✓ LeanAide result created successfully")

# Test 3: Create Lean 4 result
print("\n3. Testing Lean 4 Verification Result...")
lean4_result = Lean4VerificationResult(
    status=VerificationStatus.VERIFIED,
    verification_status="verified",
    lean4_code="theorem test : True := by trivial",
    theorem_name="test_theorem",
    execution_time_ms=10000.0,
    constraints_checked=500,
    correlation_id="test-123",
)
assert lean4_result.is_successful()
assert not lean4_result.should_escalate()
print("   ✓ Lean 4 result created successfully")

# Test 4: Unified result
print("\n4. Testing Unified Verification Result...")
unified = UnifiedVerificationResult(correlation_id="test-123")
unified.add_tier_result(z3_result, "Initial verification")
assert unified.is_successful()
assert unified.successful_tier == VerificationTier.TIER1_Z3
assert unified.confidence == 0.7
print("   ✓ Unified result works correctly")

# Test 5: Result serialization
print("\n5. Testing Result Serialization...")
data = z3_result.to_dict()
z3_result2 = Z3VerificationResult.from_dict(data)
assert z3_result.status == z3_result2.status
assert z3_result.z3_result == z3_result2.z3_result
print("   ✓ Serialization works correctly")

# Test 6: Summary generation
print("\n6. Testing Summary Generation...")
summary = unified.get_summary()
assert "✓" in summary
assert "Tier1_Z3" in summary
print(f"   Summary: {summary}")
print("   ✓ Summary generation works correctly")

print("\n" + "=" * 70)
print("ALL BASIC TESTS PASSED!")
print("=" * 70)
