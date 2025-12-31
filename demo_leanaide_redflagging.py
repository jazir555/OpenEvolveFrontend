"""
Quick Demonstration of LeanAide Red-Flagging System

This script demonstrates the key features of the red-flagging system.
"""

from leanaide_redflagging import (
    parse_lean_code,
    quick_red_flag_check,
    comprehensive_validation,
    score_proof_quality,
    create_lean_red_flagger,
    create_lean_validator,
    LeanRedFlagRules,
    LeanProof,
)

print("=" * 80)
print("LeanAide Red-Flagging System - Quick Demo")
print("=" * 80)

# Example proofs
SIMPLE_PROOF = """
theorem add_zero (n : Nat) : n + 0 = n := by
  rw [Nat.add_zero]
"""

PROOF_WITH_SORRY = """
theorem mul_one (n : Nat) : n * 1 = n := by
  sorry
"""

ELEGANT_PROOF = """
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction n
  case zero =>
    rw [Nat.add_zero, Nat.zero_add]
  case succ n ih =>
    rw [Nat.add_succ, ih, Nat.succ_add]
"""

# Demo 1: Quick red-flag check
print("\n1. Quick Red-Flag Check")
print("-" * 80)
print("Checking proof with sorry...")

is_flagged, reasons = quick_red_flag_check(PROOF_WITH_SORRY)
print(f"Flagged: {is_flagged}")
if reasons:
    print("Reasons:")
    for reason in reasons[:5]:
        print(f"  - {reason}")

# Demo 2: Comprehensive validation
print("\n2. Comprehensive Validation")
print("-" * 80)
print("Validating elegant proof...")

result = comprehensive_validation(ELEGANT_PROOF)
print(f"Valid: {result.valid}")
print(f"Errors: {len(result.errors)}")
print(f"Warnings: {len(result.warnings)}")

if result.quality_score:
    qs = result.quality_score
    print(f"\nQuality Score: {qs.overall_score:.2f}")
    print(f"  Elegance:  {qs.elegance:.2f}")
    print(f"  Clarity:   {qs.clarity:.2f}")
    print(f"  Efficiency: {qs.efficiency:.2f}")
    print(f"  Correctness: {qs.correctness:.2f}")

    if qs.flags:
        print(f"\nFlags: {', '.join(qs.flags[:3])}")

    if qs.suggestions:
        print(f"\nSuggestions:")
        for suggestion in qs.suggestions[:2]:
            print(f"  - {suggestion}")

# Demo 3: Custom rules
print("\n3. Custom Rules")
print("-" * 80)
print("Using strict rules for student submissions...")

strict_rules = LeanRedFlagRules(
    require_no_sorries=True,
    min_elegance_score=0.5,
    max_simplification_ratio=0.5
)

flagger = create_lean_red_flagger(rules=strict_rules)
proof = parse_lean_code(PROOF_WITH_SORRY)

is_flagged, reasons = flagger.is_flagged(proof)
print(f"Flagged: {is_flagged}")
print(f"Reasons: {len(reasons)}")

# Demo 4: Quality comparison
print("\n4. Quality Comparison")
print("-" * 80)

score1 = score_proof_quality(SIMPLE_PROOF)
score2 = score_proof_quality(ELEGANT_PROOF)

print(f"Simple proof: {score1.overall_score:.2f}")
print(f"Elegant proof: {score2.overall_score:.2f}")

if score2.elegance > score1.elegance:
    print("✓ Elegant proof has better elegance")

if score2.clarity > score1.clarity:
    print("✓ Elegant proof has better clarity")

print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)
print("\nFor full usage guide, see: LEANAIDE_REDFLAGGING_GUIDE.md")
print("For quick reference, see: LEANAIDE_REDFLAGGING_QUICKREF.md")
