"""
Comprehensive tests for Lean 4 / LeanAide integration in workflow_structures.py

This test file validates:
1. New dataclasses (LeanProof, LeanTheorem, LeanVerificationResult, MathematicalComponent)
2. New enums (MathematicalDomain, VerificationMethod, LeanProofStatus)
3. Extended structures (VerificationReport, SubProblem, GauntletDefinition)
4. JSON serialization/deserialization
5. Validation logic
6. Backward compatibility
"""

import json
import time
from workflow_structures import (
    # New enums
    MathematicalDomain,
    VerificationMethod,
    LeanProofStatus,
    # New dataclasses
    LeanProof,
    LeanTheorem,
    LeanVerificationResult,
    MathematicalComponent,
    # Extended dataclasses
    VerificationReport,
    SubProblem,
    GauntletDefinition,
    GauntletRoundRule,
)


def test_mathematical_domain_enum():
    """Test MathematicalDomain enum."""
    print("\n=== Testing MathematicalDomain Enum ===")

    # Test all enum values
    domains = [
        MathematicalDomain.ALGEBRA,
        MathematicalDomain.ANALYSIS,
        MathematicalDomain.TOPOLOGY,
        MathematicalDomain.NUMBER_THEORY,
        MathematicalDomain.COMBINATORICS,
        MathematicalDomain.GEOMETRY,
        MathematicalDomain.LOGIC,
        MathematicalDomain.SET_THEORY,
        MathematicalDomain.CATEGORY_THEORY,
        MathematicalDomain.LINEAR_ALGEBRA,
        MathematicalDomain.CALCULUS,
        MathematicalDomain.PROBABILITY,
        MathematicalDomain.GENERAL,
    ]

    for domain in domains:
        assert isinstance(domain.value, str)
        print(f"[OK] {domain.name}: {domain.value}")

    print(f"[OK] Total domains: {len(domains)}")
    assert len(domains) == 13


def test_verification_method_enum():
    """Test VerificationMethod enum."""
    print("\n=== Testing VerificationMethod Enum ===")

    methods = [
        VerificationMethod.MANUAL,
        VerificationMethod.AUTOMATED_TESTING,
        VerificationMethod.PEER_REVIEW,
        VerificationMethod.LEAN4,
        VerificationMethod.HYBRID,
        VerificationMethod.STATISTICAL,
        VerificationMethod.CROSS_VALIDATION,
    ]

    for method in methods:
        assert isinstance(method.value, str)
        print(f"[OK] {method.name}: {method.value}")

    print(f"[OK] Total methods: {len(methods)}")
    assert len(methods) == 7


def test_lean_proof_status_enum():
    """Test LeanProofStatus enum."""
    print("\n=== Testing LeanProofStatus Enum ===")

    statuses = [
        LeanProofStatus.PENDING,
        LeanProofStatus.IN_PROGRESS,
        LeanProofStatus.VERIFIED,
        LeanProofStatus.FAILED,
        LeanProofStatus.PARTIAL,
        LeanProofStatus.TIMEOUT,
        LeanProofStatus.ERROR,
    ]

    for status in statuses:
        assert isinstance(status.value, str)
        print(f"[OK] {status.name}: {status.value}")

    print(f"[OK] Total statuses: {len(statuses)}")
    assert len(statuses) == 7


def test_lean_proof_dataclass():
    """Test LeanProof dataclass."""
    print("\n=== Testing LeanProof Dataclass ===")

    proof = LeanProof(
        proof_id="proof_001",
        theorem_name="infinitely_many_primes",
        lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p} := by sorry",
        natural_language_statement="There are infinitely many prime numbers",
        proof_status=LeanProofStatus.PENDING,
        domain=MathematicalDomain.NUMBER_THEORY,
        complexity_score=5,
        proof_steps=["Assume finite primes", "Construct new prime", "Contradiction"],
        dependencies=["Nat.Prime", "Infinite"],
        verification_time=1.5,
        elaborated_type="Prop",
        proof_obligations=["Show construction is valid"],
        tactics_used=["by", "sorry"],
    )

    # Test to_dict
    proof_dict = proof.to_dict()
    assert proof_dict["proof_id"] == "proof_001"
    assert proof_dict["domain"] == "number_theory"
    assert proof_dict["proof_status"] == "pending"
    print("[OK] to_dict() works correctly")

    # Test from_dict
    proof_reconstructed = LeanProof.from_dict(proof_dict)
    assert proof_reconstructed.proof_id == proof.proof_id
    assert proof_reconstructed.domain == proof.domain
    assert proof_reconstructed.proof_status == proof.proof_status
    print("[OK] from_dict() works correctly")

    # Test validation
    errors = proof.validate()
    assert len(errors) == 0
    print("[OK] Validation passes for valid proof")

    # Test validation failures
    invalid_proof = LeanProof(
        proof_id="",
        theorem_name="",
        lean_code="invalid code",
        natural_language_statement="",
    )
    errors = invalid_proof.validate()
    assert len(errors) > 0
    print(f"[OK] Validation catches {len(errors)} errors")


def test_lean_theorem_dataclass():
    """Test LeanTheorem dataclass."""
    print("\n=== Testing LeanTheorem Dataclass ===")

    theorem = LeanTheorem(
        theorem_id="thm_001",
        name="Infinitely Many Primes",
        statement="There are infinitely many prime numbers",
        lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
        domain=MathematicalDomain.NUMBER_THEORY,
        keywords=["prime", "infinite", "number theory"],
        difficulty=6,
        is_verified=False,
        related_theorems=["euclid's_theorem"],
        references=["Euclid's Elements"],
    )

    # Test to_dict
    theorem_dict = theorem.to_dict()
    assert theorem_dict["theorem_id"] == "thm_001"
    assert theorem_dict["domain"] == "number_theory"
    print("[OK] to_dict() works correctly")

    # Test from_dict
    theorem_reconstructed = LeanTheorem.from_dict(theorem_dict)
    assert theorem_reconstructed.theorem_id == theorem.theorem_id
    assert theorem_reconstructed.domain == theorem.domain
    print("[OK] from_dict() works correctly")

    # Test validation
    errors = theorem.validate()
    assert len(errors) == 0
    print("[OK] Validation passes for valid theorem")


def test_lean_verification_result_dataclass():
    """Test LeanVerificationResult dataclass."""
    print("\n=== Testing LeanVerificationResult Dataclass ===")

    verification = LeanVerificationResult(
        verification_id="ver_001",
        success=True,
        theorem_id="thm_001",
        proof_id="proof_001",
        verification_method=VerificationMethod.LEAN4,
        status=LeanProofStatus.VERIFIED,
        confidence_score=0.95,
        verification_time=2.5,
        proof_steps=["Step 1", "Step 2"],
        remaining_obligations=[],
        errors=[],
        warnings=["Minor warning"],
        server_used=True,
        fallback_used=False,
    )

    # Test to_dict
    verification_dict = verification.to_dict()
    assert verification_dict["verification_id"] == "ver_001"
    assert verification_dict["verification_method"] == "lean4"
    assert verification_dict["status"] == "verified"
    print("[OK] to_dict() works correctly")

    # Test from_dict
    verification_reconstructed = LeanVerificationResult.from_dict(verification_dict)
    assert verification_reconstructed.verification_id == verification.verification_id
    assert verification_reconstructed.verification_method == verification.verification_method
    print("[OK] from_dict() works correctly")

    # Test validation
    errors = verification.validate()
    assert len(errors) == 0
    print("[OK] Validation passes for valid verification")


def test_mathematical_component_dataclass():
    """Test MathematicalComponent dataclass."""
    print("\n=== Testing MathematicalComponent Dataclass ===")

    component = MathematicalComponent(
        component_id="comp_001",
        type="theorem",
        name="Infinitely Many Primes",
        statement="There are infinitely many prime numbers",
        domain=MathematicalDomain.NUMBER_THEORY,
        complexity=5,
        dependencies=["Nat.Prime"],
        formalized=True,
        lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
        verification_status=LeanProofStatus.VERIFIED,
    )

    # Test to_dict
    component_dict = component.to_dict()
    assert component_dict["component_id"] == "comp_001"
    assert component_dict["domain"] == "number_theory"
    assert component_dict["verification_status"] == "verified"
    print("[OK] to_dict() works correctly")

    # Test from_dict
    component_reconstructed = MathematicalComponent.from_dict(component_dict)
    assert component_reconstructed.component_id == component.component_id
    assert component_reconstructed.domain == component.domain
    print("[OK] from_dict() works correctly")


def test_verification_report_extension():
    """Test VerificationReport with Lean 4 fields."""
    print("\n=== Testing VerificationReport Extension ===")

    # Create a Lean verification result
    lean_result = LeanVerificationResult(
        verification_id="ver_001",
        success=True,
        theorem_id="thm_001",
        verification_method=VerificationMethod.LEAN4,
        status=LeanProofStatus.VERIFIED,
        confidence_score=0.95,
    )

    # Create VerificationReport with Lean 4 fields
    report = VerificationReport(
        solution_attempt_id="attempt_001",
        gauntlet_name="verification_gauntlet",
        is_approved=True,
        reports_by_judge=[{"judge_1": {"score": 0.9}}],
        average_score=0.9,
        summary="Solution verified with Lean 4",
        lean_verification=lean_result,
        verification_method=VerificationMethod.LEAN4,
        mathematical_verified=True,
        formal_proof_available=True,
        mathematical_confidence=0.95,
        mathematical_components_verified=["thm_001"],
    )

    # Test that Lean fields are present
    assert report.lean_verification is not None
    assert report.verification_method == VerificationMethod.LEAN4
    assert report.mathematical_verified is True
    assert report.formal_proof_available is True
    assert report.mathematical_confidence == 0.95
    assert len(report.mathematical_components_verified) == 1
    print("[OK] VerificationReport Lean 4 fields work correctly")

    # Test backward compatibility - old code without Lean fields should still work
    old_report = VerificationReport(
        solution_attempt_id="attempt_002",
        gauntlet_name="old_gauntlet",
        is_approved=False,
        reports_by_judge=[],
    )
    assert old_report.lean_verification is None
    assert old_report.verification_method == VerificationMethod.PEER_REVIEW
    assert old_report.mathematical_verified is False
    print("[OK] Backward compatibility maintained")


def test_subproblem_extension():
    """Test SubProblem with mathematical components."""
    print("\n=== Testing SubProblem Extension ===")

    # Create mathematical components
    math_component = MathematicalComponent(
        component_id="comp_001",
        type="theorem",
        name="Infinitely Many Primes",
        statement="There are infinitely many primes",
        domain=MathematicalDomain.NUMBER_THEORY,
        complexity=5,
    )

    # Create a Lean theorem
    lean_theorem = LeanTheorem(
        theorem_id="thm_001",
        name="Infinitely Many Primes",
        statement="There are infinitely many primes",
        lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
        domain=MathematicalDomain.NUMBER_THEORY,
    )

    # Create SubProblem with mathematical fields
    subproblem = SubProblem(
        id="sub_001",
        description="Prove there are infinitely many primes",
        mathematical_components=[math_component],
        requires_formal_verification=True,
        mathematical_domain=MathematicalDomain.NUMBER_THEORY,
        formal_verification_enabled=True,
        mathematical_properties=["infinite", "prime", "existence"],
        lean_theorems=[lean_theorem],
    )

    # Test that mathematical fields are present
    assert len(subproblem.mathematical_components) == 1
    assert subproblem.requires_formal_verification is True
    assert subproblem.mathematical_domain == MathematicalDomain.NUMBER_THEORY
    assert subproblem.formal_verification_enabled is True
    assert len(subproblem.mathematical_properties) == 3  # infinite, prime, existence
    assert len(subproblem.lean_theorems) == 1
    print("[OK] SubProblem mathematical fields work correctly")

    # Test backward compatibility
    old_subproblem = SubProblem(
        id="sub_002",
        description="Simple problem",
    )
    assert len(old_subproblem.mathematical_components) == 0
    assert old_subproblem.requires_formal_verification is False
    assert old_subproblem.formal_verification_enabled is False
    print("[OK] Backward compatibility maintained")


def test_gauntlet_definition_extension():
    """Test GauntletDefinition with formal verification."""
    print("\n=== Testing GauntletDefinition Extension ===")

    # Create a GauntletDefinition with formal verification enabled
    gauntlet = GauntletDefinition(
        name="mathematical_verification_gauntlet",
        team_name="gold_team",
        rounds=[],
        formal_verification_enabled=True,
        verification_methods=[VerificationMethod.LEAN4, VerificationMethod.PEER_REVIEW],
        proof_generation_enabled=True,
        automatic_formalization=True,
        formal_verification_threshold=0.95,
        lean_verification_config={
            "timeout": 300,
            "max_complexity": 8,
        },
    )

    # Test that formal verification fields are present
    assert gauntlet.formal_verification_enabled is True
    assert VerificationMethod.LEAN4 in gauntlet.verification_methods
    assert gauntlet.proof_generation_enabled is True
    assert gauntlet.automatic_formalization is True
    assert gauntlet.formal_verification_threshold == 0.95
    assert "timeout" in gauntlet.lean_verification_config
    print("[OK] GauntletDefinition formal verification fields work correctly")

    # Test backward compatibility
    old_gauntlet = GauntletDefinition(
        name="old_gauntlet",
        team_name="red_team",
        rounds=[],
    )
    assert old_gauntlet.formal_verification_enabled is False
    assert VerificationMethod.PEER_REVIEW in old_gauntlet.verification_methods
    assert old_gauntlet.proof_generation_enabled is False
    print("[OK] Backward compatibility maintained")


def test_json_serialization_compatibility():
    """Test JSON serialization and ensure compatibility with existing code."""
    print("\n=== Testing JSON Serialization Compatibility ===")

    # Test LeanProof serialization
    proof = LeanProof(
        proof_id="proof_001",
        theorem_name="test_theorem",
        lean_code="theorem test : True := by trivial",
        natural_language_statement="Test theorem",
    )
    proof_json = json.dumps(proof.to_dict())
    proof_loaded = LeanProof.from_dict(json.loads(proof_json))
    assert proof_loaded.proof_id == proof.proof_id
    print("[OK] LeanProof JSON serialization works")

    # Test LeanTheorem serialization
    theorem = LeanTheorem(
        theorem_id="thm_001",
        name="Test",
        statement="Test theorem",
        lean_code="theorem test : True := by trivial",
    )
    theorem_json = json.dumps(theorem.to_dict())
    theorem_loaded = LeanTheorem.from_dict(json.loads(theorem_json))
    assert theorem_loaded.theorem_id == theorem.theorem_id
    print("[OK] LeanTheorem JSON serialization works")

    # Test LeanVerificationResult serialization
    verification = LeanVerificationResult(
        verification_id="ver_001",
        success=True,
        theorem_id="thm_001",
    )
    verification_json = json.dumps(verification.to_dict())
    verification_loaded = LeanVerificationResult.from_dict(json.loads(verification_json))
    assert verification_loaded.verification_id == verification.verification_id
    print("[OK] LeanVerificationResult JSON serialization works")


def test_default_values_and_backward_compatibility():
    """Test that all new fields have sensible defaults for backward compatibility."""
    print("\n=== Testing Default Values and Backward Compatibility ===")

    # Test VerificationReport defaults
    report = VerificationReport(
        solution_attempt_id="test",
        gauntlet_name="test",
        is_approved=True,
        reports_by_judge=[],
    )
    assert report.lean_verification is None
    assert report.verification_method == VerificationMethod.PEER_REVIEW
    assert report.mathematical_verified is False
    assert report.formal_proof_available is False
    assert report.mathematical_confidence == 0.0
    assert len(report.mathematical_components_verified) == 0
    print("[OK] VerificationReport defaults are safe")

    # Test SubProblem defaults
    subproblem = SubProblem(
        id="test",
        description="test",
    )
    assert len(subproblem.mathematical_components) == 0
    assert subproblem.requires_formal_verification is False
    assert subproblem.mathematical_domain is None
    assert subproblem.formal_verification_enabled is False
    assert len(subproblem.mathematical_properties) == 0
    assert len(subproblem.lean_theorems) == 0
    print("[OK] SubProblem defaults are safe")

    # Test GauntletDefinition defaults
    gauntlet = GauntletDefinition(
        name="test",
        team_name="test",
        rounds=[],
    )
    assert gauntlet.formal_verification_enabled is False
    assert VerificationMethod.PEER_REVIEW in gauntlet.verification_methods
    assert gauntlet.proof_generation_enabled is False
    assert gauntlet.automatic_formalization is False
    assert gauntlet.formal_verification_threshold == 0.9
    assert len(gauntlet.lean_verification_config) == 0
    print("[OK] GauntletDefinition defaults are safe")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Lean 4 / LeanAide Integration Tests")
    print("="*60)

    test_mathematical_domain_enum()
    test_verification_method_enum()
    test_lean_proof_status_enum()
    test_lean_proof_dataclass()
    test_lean_theorem_dataclass()
    test_lean_verification_result_dataclass()
    test_mathematical_component_dataclass()
    test_verification_report_extension()
    test_subproblem_extension()
    test_gauntlet_definition_extension()
    test_json_serialization_compatibility()
    test_default_values_and_backward_compatibility()

    print("\n" + "="*60)
    print("All tests passed! [OK]")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
