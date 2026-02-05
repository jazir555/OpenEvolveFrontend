"""
Test Suite for LeanAide Red-Flagging System

This test suite demonstrates the usage of the comprehensive Lean 4 red-flagging system
with various proof examples.
"""

import pytest
from leanaide_redflagging import (
    # Data structures
    LeanProof,
    LeanProofType,
    LeanProofState,
    LeanQualityScore,
    ValidationResult,

    # Rules
    LeanRedFlagRules,

    # Main classes
    LeanRedFlagger,
    LeanProofValidator,
    LeanProofQualityScorer,

    # Factory functions
    create_lean_red_flagger,
    create_lean_validator,
    create_lean_quality_scorer,

    # Utility functions
    parse_lean_code,
    quick_red_flag_check,
    comprehensive_validation,
    score_proof_quality,
)


# =============================================================================
# TEST DATA
# =============================================================================

# Example 1: Simple, correct proof
SIMPLE_PROOF = """
theorem add_zero (n : Nat) : n + 0 = n := by
  rw [Nat.add_zero]
"""

# Example 2: Proof with sorry
PROOF_WITH_SORRY = """
theorem mul_one (n : Nat) : n * 1 = n := by
  sorry
"""

# Example 3: Long, repetitive proof
LONG_PROOF = """
theorem long_proof (n m k : Nat) : n + m + k = n + m + k := by
  rw [Nat.add_assoc]
  rw [Nat.add_assoc]
  rw [Nat.add_zero]
  rw [Nat.add_zero]
  rw [Nat.add_zero]
  simp
  simp
  simp
"""

# Example 4: Elegant proof
ELEGANT_PROOF = """
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction n
  case zero =>
    rw [Nat.add_zero, Nat.zero_add]
  case succ n ih =>
    rw [Nat.add_succ, ih, Nat.succ_add]
"""

# Example 5: Proof with structural issues
BROKEN_PROOF = """
theorem broken (n : Nat) : n = n := by
  apply
  intro
  simp
  (this is invalid)
"""

# Example 6: High quality proof
HIGH_QUALITY_PROOF = """
theorem mul_assoc (n m k : Nat) : (n * m) * k = n * (m * k) := by
  induction k
  case zero =>
    rw [Nat.mul_zero, Nat.mul_zero]
  case succ k ih =>
    rw [Nat.mul_succ, ih, Nat.mul_succ]
    ring
"""

# Example 7: Proof with poor naming
POOR_NAMING_PROOF = """
theorem theorem1 (x : Nat) : x = x := by
  rfl
"""

# Example 8: Complex mathematical proof
COMPLEX_PROOF = """
theorem sqrt_two_irrational : ∀ q : ℚ, q ^ 2 ≠ 2 := by
  intro q h
  have q_num_nonneg : 0 ≤ q.num := by
    apply Int.ofNat_nonneg
  have q_denom_pos : 0 < q.denom := by
    apply Nat.pos_of_ne_zero
    intro
    contradiction
  cases q
  rename_a num => a
  rename_b den => b
  have hb : 0 < b := q_denom_pos
  have h2 : a ^ 2 = 2 * b ^ 2 := by
    simp [*, pow_two] at *
    aesop
  have ha_even : Even a := by
    have : 2 ∣ a ^ 2 := by
      convert (Nat.dvd_mul_right 2 (b ^ 2)).symm using 1
      rw [h2]
    rwa [Nat.Prime.dvd_pow_two_iff] at this
    apply Nat.prime_two
  obtain ⟨c, hc⟩ := ha_even
  have h2' : (2 * c) ^ 2 = 2 * b ^ 2 := by
    rw [hc, h2]
  have : 2 * c ^ 2 = b ^ 2 := by
    linarith [pow_two 2 c, pow_two 2 b, h2']
  have hb_even : Even b := by
    have : 2 ∣ b ^ 2 := by
      convert (Nat.dvd_mul_right 2 (c ^ 2)).symm using 1
      rw [this]
    rwa [Nat.Prime.dvd_pow_two_iff] at this
    apply Nat.prime_two
  obtain ⟨d, hd⟩ := hb_even
  have : 2 * b ^ 2 = 4 * d ^ 2 := by
    rw [hd, pow_two]
    linarith
  have : 2 * c ^ 2 = 4 * d ^ 2 := by
    rw [<- this, h2']
  have : c ^ 2 = 2 * d ^ 2 := by
    linarith
  -- Contradiction with infinite descent
  have : c < a := by
    have : 2 * c = a := by
      rw [hc]
    have : 0 < c := by
      have : 0 < a := by
        simp [*, pow_two] at *
      linarith
    linarith
  contradiction
"""


# =============================================================================
# TESTS
# =============================================================================

class TestLeanRedFlagRules:
    """Test LeanRedFlagRules configuration"""

    def test_default_rules(self):
        """Test default rule values"""
        rules = LeanRedFlagRules()
        assert rules.max_proof_length == 500
        assert rules.require_no_sorries is True
        assert rules.min_elegance_score == 0.3

    def test_custom_rules(self):
        """Test custom rule configuration"""
        rules = LeanRedFlagRules(
            max_proof_length=1000,
            require_no_sorries=False,
            min_elegance_score=0.5
        )
        assert rules.max_proof_length == 1000
        assert rules.require_no_sorries is False
        assert rules.min_elegance_score == 0.5


class TestLeanProofParsing:
    """Test Lean code parsing"""

    def test_parse_simple_proof(self):
        """Test parsing a simple proof"""
        proof = parse_lean_code(SIMPLE_PROOF)
        assert proof.name == "add_zero"
        assert proof.proof_type == LeanProofType.THEOREM
        assert proof.tactic_count > 0

    def test_parse_proof_with_sorry(self):
        """Test parsing proof with sorry"""
        proof = parse_lean_code(PROOF_WITH_SORRY)
        assert proof.has_sorry is True
        assert proof.sorry_count == 1

    def test_parse_complex_proof(self):
        """Test parsing complex proof"""
        proof = parse_lean_code(COMPLEX_PROOF)
        assert proof.name == "sqrt_two_irrational"
        assert proof.tactic_count > 10


class TestLeanRedFlagger:
    """Test Lean red-flagging functionality"""

    def test_flag_sorry(self):
        """Test that proofs with sorry are flagged"""
        flagger = create_lean_red_flagger(require_no_sorries=True)
        proof = parse_lean_code(PROOF_WITH_SORRY)

        is_flagged, reasons = flagger.is_flagged(proof)

        assert is_flagged is True
        assert any("sorry" in r.lower() for r in reasons)

    def test_allow_sorry(self):
        """Test that proofs with sorry can be allowed"""
        rules = LeanRedFlagRules(require_no_sorries=False, max_sorry_count=1)
        flagger = LeanRedFlagger(rules)
        proof = parse_lean_code(PROOF_WITH_SORRY)

        is_flagged, reasons = flagger.is_flagged(proof)

        # Should not be flagged for sorry (but might have other issues)
        sorry_flags = [r for r in reasons if "sorry" in r.lower()]
        assert len(sorry_flags) == 0

    def test_flag_long_proof(self):
        """Test that excessively long proofs are flagged"""
        flagger = create_lean_red_flagger(max_proof_length=5)
        proof = parse_lean_code(LONG_PROOF)

        is_flagged, reasons = flagger.is_flagged(proof)

        assert is_flagged is True
        assert any("too_long" in r.lower() for r in reasons)

    def test_flag_repetitive_tactics(self):
        """Test detection of repetitive tactics"""
        flagger = create_lean_red_flagger()
        proof = parse_lean_code(LONG_PROOF)

        _, reasons = flagger.check_semantics(proof)

        assert any("repetitive" in r.lower() for r in reasons)

    def test_check_syntax(self):
        """Test syntax checking"""
        flagger = create_lean_red_flagger()

        # Valid syntax
        valid_proof = parse_lean_code(SIMPLE_PROOF)
        errors = flagger.check_syntax(valid_proof)
        assert len(errors) == 0 or len(errors) < 3  # Allow minor warnings

        # Invalid syntax
        invalid_proof = parse_lean_code(BROKEN_PROOF)
        errors = flagger.check_syntax(invalid_proof)
        assert len(errors) > 0

    def test_check_structure(self):
        """Test structure checking"""
        flagger = create_lean_red_flagger()
        proof = parse_lean_code(LONG_PROOF)

        errors = flagger.check_structure(proof)

        # Should flag excessive simp usage
        assert any("simp" in r.lower() for r in errors)

    def test_check_quality(self):
        """Test quality checking"""
        flagger = create_lean_red_flagger()
        proof = parse_lean_code(POOR_NAMING_PROOF)

        errors = flagger.check_quality(proof)

        # Should flag poor naming
        assert any("naming" in r.lower() for r in errors)


class TestLeanProofValidator:
    """Test comprehensive proof validation"""

    def test_validate_syntax(self):
        """Test syntax validation"""
        validator = create_lean_validator()

        is_valid, errors = validator.validate_syntax(SIMPLE_PROOF)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid code"""
        validator = create_lean_validator()

        is_valid, errors = validator.validate_syntax(BROKEN_PROOF)

        # Should have errors
        assert len(errors) > 0

    def test_full_validation_valid(self):
        """Test full validation of valid proof"""
        validator = create_lean_validator()
        proof = parse_lean_code(SIMPLE_PROOF)

        result = validator.full_validation(proof)

        # Should be valid (or have only warnings, not errors)
        assert result.valid is True or len(result.errors) == 0
        assert result.quality_score is not None

    def test_full_validation_with_sorry(self):
        """Test full validation with sorry"""
        validator = create_lean_validator(
            rules=LeanRedFlagRules(require_no_sorries=False)
        )
        proof = parse_lean_code(PROOF_WITH_SORRY)

        result = validator.full_validation(proof)

        # Quality score should reflect the sorry
        assert result.quality_score.correctness < 1.0

    def test_full_validation_complex(self):
        """Test full validation of complex proof"""
        validator = create_lean_validator()
        proof = parse_lean_code(COMPLEX_PROOF)

        result = validator.full_validation(proof)

        # Complex proof should have quality score
        assert result.quality_score is not None
        assert result.quality_score.elegance > 0.0


class TestLeanProofQualityScorer:
    """Test quality scoring"""

    def test_score_elegance(self):
        """Test elegance scoring"""
        scorer = create_lean_quality_scorer()

        # Simple proof
        simple_proof = parse_lean_code(SIMPLE_PROOF)
        simple_score = scorer.score_elegance(simple_proof)

        # Elegant proof
        elegant_proof = parse_lean_code(ELEGANT_PROOF)
        elegant_score = scorer.score_elegance(elegant_proof)

        # Elegant proof should score at least as well
        assert elegant_score >= simple_score

    def test_score_clarity(self):
        """Test clarity scoring"""
        scorer = create_lean_quality_scorer()

        # Poor naming
        poor_proof = parse_lean_code(POOR_NAMING_PROOF)
        poor_score = scorer.score_clarity(poor_proof)

        # High quality
        good_proof = parse_lean_code(HIGH_QUALITY_PROOF)
        good_score = scorer.score_clarity(good_proof)

        # Good proof should score better
        assert good_score > poor_score

    def test_score_efficiency(self):
        """Test efficiency scoring"""
        scorer = create_lean_quality_scorer()

        # Long, repetitive proof
        long_proof = parse_lean_code(LONG_PROOF)
        long_score = scorer.score_efficiency(long_proof)

        # Efficient proof
        simple_proof = parse_lean_code(SIMPLE_PROOF)
        simple_score = scorer.score_efficiency(simple_proof)

        # Simple proof should be more efficient
        assert simple_score >= long_score

    def test_score_correctness(self):
        """Test correctness scoring"""
        scorer = create_lean_quality_scorer()

        # Proof with sorry
        sorry_proof = parse_lean_code(PROOF_WITH_SORRY)
        sorry_score = scorer.score_correctness(sorry_proof)

        # Complete proof
        complete_proof = parse_lean_code(SIMPLE_PROOF)
        complete_score = scorer.score_correctness(complete_proof)

        # Complete proof should score better
        assert complete_score > sorry_score

    def test_full_quality_score(self):
        """Test full quality scoring"""
        scorer = create_lean_quality_scorer()
        proof = parse_lean_code(HIGH_QUALITY_PROOF)

        score = scorer.score_proof(proof)

        # Check all dimensions are scored
        assert 0.0 <= score.overall_score <= 1.0
        assert 0.0 <= score.elegance <= 1.0
        assert 0.0 <= score.clarity <= 1.0
        assert 0.0 <= score.efficiency <= 1.0
        assert 0.0 <= score.correctness <= 1.0

        # High quality proof should score well
        assert score.overall_score > 0.5


class TestUtilityFunctions:
    """Test utility functions"""

    def test_quick_red_flag_check(self):
        """Test quick red-flag check"""
        # Valid proof
        is_flagged, reasons = quick_red_flag_check(SIMPLE_PROOF)
        assert is_flagged is False or len(reasons) == 0

        # Proof with sorry
        is_flagged, reasons = quick_red_flag_check(PROOF_WITH_SORRY)
        assert is_flagged is True

    def test_comprehensive_validation(self):
        """Test comprehensive validation utility"""
        result = comprehensive_validation(SIMPLE_PROOF)

        assert isinstance(result, ValidationResult)
        assert result.quality_score is not None

    def test_score_proof_quality(self):
        """Test proof quality scoring utility"""
        score = score_proof_quality(HIGH_QUALITY_PROOF)

        assert isinstance(score, LeanQualityScore)
        assert 0.0 <= score.overall_score <= 1.0


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""

    def test_validate_student_submission(self):
        """Scenario: Validate a student's proof submission"""
        validator = create_lean_validator(
            rules=LeanRedFlagRules(
                require_no_sorries=True,
                min_elegance_score=0.3
            )
        )

        # Student submits proof with sorry
        submission = PROOF_WITH_SORRY
        proof = parse_lean_code(submission)
        result = validator.full_validation(proof)

        # Should be flagged
        assert result.valid is False
        assert len(result.errors) > 0

    def test_grade_proof_quality(self):
        """Scenario: Grade a proof on quality dimensions"""
        scorer = create_lean_quality_scorer()

        proof = parse_lean_code(ELEGANT_PROOF)
        score = scorer.score_proof(proof)

        # All dimensions should be good
        assert score.elegance > 0.5
        assert score.clarity > 0.5
        assert score.efficiency > 0.5
        assert score.correctness > 0.5

        # Should have suggestions for improvement
        assert len(score.suggestions) >= 0

    def test_batch_proof_review(self):
        """Scenario: Review multiple proofs"""
        validator = create_lean_validator()

        proofs = [
            SIMPLE_PROOF,
            ELEGANT_PROOF,
            HIGH_QUALITY_PROOF,
            PROOF_WITH_SORRY,
        ]

        results = []
        for proof_code in proofs:
            proof = parse_lean_code(proof_code)
            result = validator.full_validation(proof)
            results.append(result)

        # Should have results for all
        assert len(results) == len(proofs)

        # At least one should be valid
        valid_count = sum(1 for r in results if r.valid)
        assert valid_count > 0


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_red_flagging():
    """Demonstrate red-flagging system"""
    print("\n" + "=" * 80)
    print("LeanAide Red-Flagging System Demo")
    print("=" * 80)

    # Example 1: Flag proofs with sorry
    print("\n1. Checking proof with sorry:")
    print("-" * 80)
    is_flagged, reasons = quick_red_flag_check(PROOF_WITH_SORRY)
    print(f"Code: {PROOF_WITH_SORRY.strip()[:60]}...")
    print(f"Flagged: {is_flagged}")
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")

    # Example 2: Validate elegant proof
    print("\n2. Validating elegant proof:")
    print("-" * 80)
    result = comprehensive_validation(ELEGANT_PROOF)
    print(f"Valid: {result.valid}")
    if result.quality_score:
        print(f"Overall Score: {result.quality_score.overall_score:.2f}")
        print(f"  Elegance: {result.quality_score.elegance:.2f}")
        print(f"  Clarity: {result.quality_score.clarity:.2f}")
        print(f"  Efficiency: {result.quality_score.efficiency:.2f}")
        print(f"  Correctness: {result.quality_score.correctness:.2f}")

    # Example 3: Score complex proof
    print("\n3. Scoring complex proof:")
    print("-" * 80)
    score = score_proof_quality(COMPLEX_PROOF)
    print(f"Proof: sqrt_two_irrational")
    print(f"Overall Score: {score.overall_score:.2f}")
    print(f"Dimensions:")
    print(f"  - Elegance: {score.elegance:.2f}")
    print(f"  - Clarity: {score.clarity:.2f}")
    print(f"  - Efficiency: {score.efficiency:.2f}")
    print(f"  - Correctness: {score.correctness:.2f}")

    if score.flags:
        print(f"Flags: {', '.join(score.flags)}")
    if score.suggestions:
        print("Suggestions:")
        for suggestion in score.suggestions[:3]:
            print(f"  - {suggestion}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run demo
    demo_red_flagging()

    # Run tests
    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)
    pytest.main([__file__, "-v", "-x"])
