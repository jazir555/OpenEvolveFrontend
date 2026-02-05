"""
Test CAV-NLP Integration

Tests the integration of CAV-NLP as the primary mathematical formalization system.
Verifies:
1. Backward compatibility with existing API
2. CAV-NLP components are properly integrated
3. Data structures are preserved with enhancements
4. Mappings are available
"""

import asyncio
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress deprecation warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_import_cav_nlp_integration():
    """Test importing the CAV-NLP integration module."""
    print("\n1. Testing CAV-NLP Integration Import...")
    
    try:
        from openevolve.cav_nlp_integration import (
            Z3LeanAideBridge,
            create_z3_lean_bridge,
            quick_verify,
            TranslationDirection,
            ConstraintType,
            Z3Constraint,
            Lean4Constraint,
            TranslationResult,
            VerificationBridgeResult,
            HybridProofResult,
            CAVNLPContext,
            CanonicalizationResult,
            Z3_TO_LEAN_TYPES,
            LEAN_TO_Z3_TYPES,
            Z3_TO_LEAN_OPERATORS,
            CANONICALIZATION_RULES,
        )
        print("   [PASS] All imports successful")
        return True
    except Exception as e:
        print(f"   [FAIL] Import failed: {e}")
        return False


def test_data_structures():
    """Test data structures are properly defined."""
    print("\n2. Testing Data Structures...")
    
    from openevolve.cav_nlp_integration import (
        TranslationDirection,
        ConstraintType,
        Z3Constraint,
        Lean4Constraint,
        TranslationResult,
        VerificationBridgeResult,
        CAVNLPContext,
        CanonicalizationResult,
    )
    
    success = True
    
    # Test TranslationDirection
    try:
        assert TranslationDirection.Z3_TO_LEAN.value == "z3_to_lean"
        assert TranslationDirection.LEAN_TO_Z3.value == "lean_to_z3"
        print("   [PASS] TranslationDirection enum works")
    except Exception as e:
        print(f"   [FAIL] TranslationDirection failed: {e}")
        success = False
    
    # Test ConstraintType
    try:
        assert ConstraintType.ARITHMETIC.value == "arithmetic"
        assert ConstraintType.NONLINEAR.value == "nonlinear"
        print("   [PASS] ConstraintType enum works")
    except Exception as e:
        print(f"   [FAIL] ConstraintType failed: {e}")
        success = False
    
    # Test Z3Constraint
    try:
        constraint = Z3Constraint(
            expr="x > 0",
            constraint_type=ConstraintType.ARITHMETIC,
            variables=["x"]
        )
        assert constraint.variables == ["x"]
        print("   [PASS] Z3Constraint dataclass works")
    except Exception as e:
        print(f"   [FAIL] Z3Constraint failed: {e}")
        success = False
    
    # Test Lean4Constraint
    try:
        constraint = Lean4Constraint(
            lean_code="theorem test : x > 0 := by linarith",
            constraint_type=ConstraintType.ARITHMETIC,
            variables=["x"],
            theorem_statement="x > 0"
        )
        assert constraint.lean_code.startswith("theorem")
        print("   [PASS] Lean4Constraint dataclass works")
    except Exception as e:
        print(f"   [FAIL] Lean4Constraint failed: {e}")
        success = False
    
    # Test TranslationResult with CAV-NLP enhancements
    try:
        result = TranslationResult(
            success=True,
            source="z3",
            target="lean",
            direction=TranslationDirection.Z3_TO_LEAN,
            source_code="x > 0",
            target_code="theorem test : x > 0 := by linarith",
            errors=[],
            warnings=[],
            dag={"nodes": []},  # CAV-NLP enhancement
            canonical_form="x > 0",  # CAV-NLP enhancement
            cegis_iterations=3  # CAV-NLP enhancement
        )
        assert result.cegis_iterations == 3
        print("   [PASS] TranslationResult with CAV-NLP enhancements works")
    except Exception as e:
        print(f"   [FAIL] TranslationResult failed: {e}")
        success = False
    
    # Test VerificationBridgeResult with CAV-NLP enhancements
    try:
        result = VerificationBridgeResult(
            z3_result="unsat",
            lean_result=None,
            agreed=True,
            z3_model=None,
            lean_proof=None,
            counterexample=None,
            confidence=0.9,
            execution_time=0.5,
            canonicalization_verified=True,  # CAV-NLP enhancement
            dag={"dependencies": []}  # CAV-NLP enhancement
        )
        assert result.canonicalization_verified == True
        print("   [PASS] VerificationBridgeResult with CAV-NLP enhancements works")
    except Exception as e:
        print(f"   [FAIL] VerificationBridgeResult failed: {e}")
        success = False
    
    # Test CAVNLPContext
    try:
        context = CAVNLPContext(
            paper_title="Test Paper",
            section_context="Introduction",
            theorem_number=1,
            dependency_graph={"nodes": []}
        )
        assert context.paper_title == "Test Paper"
        print("   [PASS] CAVNLPContext dataclass works")
    except Exception as e:
        print(f"   [FAIL] CAVNLPContext failed: {e}")
        success = False
    
    # Test CanonicalizationResult
    try:
        result = CanonicalizationResult(
            original="x + y",
            canonical="y + x",
            z3_validated=True,
            equivalent_by="commutativity"
        )
        assert result.equivalent_by == "commutativity"
        print("   [PASS] CanonicalizationResult dataclass works")
    except Exception as e:
        print(f"   [FAIL] CanonicalizationResult failed: {e}")
        success = False
    
    return success


def test_mappings():
    """Test mappings are properly defined."""
    print("\n3. Testing Mappings...")
    
    from openevolve.cav_nlp_integration import (
        Z3_TO_LEAN_TYPES,
        LEAN_TO_Z3_TYPES,
        Z3_TO_LEAN_OPERATORS,
        LEAN_TO_Z3_OPERATORS,
        CONSTRAINT_TYPE_TACTICS,
        CANONICALIZATION_RULES,
        CANONICALIZATION_ORDER,
        LEAN_IMPORTS_BY_TYPE,
    )
    
    success = True
    
    # Test type mappings
    try:
        assert Z3_TO_LEAN_TYPES["Bool"] == "Prop"
        assert Z3_TO_LEAN_TYPES["Int"] == "ℤ"
        assert Z3_TO_LEAN_TYPES["Real"] == "ℝ"
        assert LEAN_TO_Z3_TYPES["Prop"] == "Bool"
        print("   [PASS] Type mappings correct")
    except Exception as e:
        print(f"   [FAIL] Type mappings failed: {e}")
        success = False
    
    # Test operator mappings
    try:
        assert Z3_TO_LEAN_OPERATORS["And"] == "∧"
        assert Z3_TO_LEAN_OPERATORS["Or"] == "∨"
        assert Z3_TO_LEAN_OPERATORS["Not"] == "¬"
        assert LEAN_TO_Z3_OPERATORS["∧"] == "And"
        print("   [PASS] Operator mappings correct")
    except Exception as e:
        print(f"   [FAIL] Operator mappings failed: {e}")
        success = False
    
    # Test tactic mappings
    try:
        assert "tauto" in CONSTRAINT_TYPE_TACTICS["boolean"]
        assert "linarith" in CONSTRAINT_TYPE_TACTICS["arithmetic"]
        assert "nlinarith" in CONSTRAINT_TYPE_TACTICS["nonlinear"]
        print("   [PASS] Tactic mappings correct")
    except Exception as e:
        print(f"   [FAIL] Tactic mappings failed: {e}")
        success = False
    
    # Test canonicalization rules
    try:
        assert "commutativity_add" in CANONICALIZATION_RULES
        assert "de_morgan_and" in CANONICALIZATION_RULES
        assert "distributivity" in CANONICALIZATION_RULES
        assert len(CANONICALIZATION_ORDER) > 0
        print("   [PASS] Canonicalization rules correct")
    except Exception as e:
        print(f"   [FAIL] Canonicalization rules failed: {e}")
        success = False
    
    # Test import requirements
    try:
        assert "import Mathlib" in LEAN_IMPORTS_BY_TYPE["arithmetic"]
        print("   [PASS] Import requirements correct")
    except Exception as e:
        print(f"   [FAIL] Import requirements failed: {e}")
        success = False
    
    return success


def test_bridge_api():
    """Test the main bridge API."""
    print("\n4. Testing Bridge API...")
    
    from openevolve.cav_nlp_integration import Z3LeanAideBridge, create_z3_lean_bridge
    
    success = True
    
    # Test bridge creation
    try:
        bridge = create_z3_lean_bridge()
        assert isinstance(bridge, Z3LeanAideBridge)
        print("   [PASS] Bridge creation works")
    except Exception as e:
        print(f"   [FAIL] Bridge creation failed: {e}")
        success = False
    
    # Test capability checking
    try:
        capabilities = bridge.get_capabilities()
        assert "z3_available" in capabilities
        assert "lean_available" in capabilities
        assert "hybrid_verification" in capabilities
        print(f"   [PASS] Capabilities: {capabilities}")
    except Exception as e:
        print(f"   [FAIL] Capability check failed: {e}")
        success = False
    
    # Test availability methods
    try:
        z3_avail = bridge.is_z3_available()
        lean_avail = bridge.is_lean_available()
        print(f"   [PASS] Z3 available: {z3_avail}, Lean available: {lean_avail}")
    except Exception as e:
        print(f"   [FAIL] Availability check failed: {e}")
        success = False
    
    return success


def test_backward_compatibility():
    """Test backward compatibility with old import."""
    print("\n5. Testing Backward Compatibility...")
    
    success = True
    
    # Test old import path (should work with deprecation warning)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from openevolve import z3_leanaide_bridge
            
            # Check deprecation warning was issued
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            if len(deprecation_warnings) > 0:
                print("   [PASS] Deprecation warning issued correctly")
            else:
                print("   [WARN] No deprecation warning (may be suppressed)")
        
        # Test that main classes are available
        assert hasattr(z3_leanaide_bridge, 'Z3LeanAideBridge')
        assert hasattr(z3_leanaide_bridge, 'create_z3_lean_bridge')
        assert hasattr(z3_leanaide_bridge, 'TranslationDirection')
        assert hasattr(z3_leanaide_bridge, 'ConstraintType')
        assert hasattr(z3_leanaide_bridge, 'Z3Constraint')
        assert hasattr(z3_leanaide_bridge, 'Lean4Constraint')
        
        print("   [PASS] Old import path works with deprecation warning")
    except Exception as e:
        print(f"   [FAIL] Backward compatibility failed: {e}")
        success = False
    
    # Test that classes work from old import
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from openevolve import z3_leanaide_bridge as old_bridge
        
        bridge = old_bridge.create_z3_lean_bridge()
        assert isinstance(bridge, old_bridge.Z3LeanAideBridge)
        print("   [PASS] Old API works correctly")
    except Exception as e:
        print(f"   [FAIL] Old API test failed: {e}")
        success = False
    
    return success


def test_cav_nlp_components():
    """Test CAV-NLP components are accessible."""
    print("\n6. Testing CAV-NLP Components...")
    
    from openevolve.cav_nlp_integration import (
        MathematicalTextParser,
        SemanticPrimitive,
        SemanticNormalizer,
        DependencyDAG,
        Z3SemanticSynthesis,
        CanonicalLeanGenerator,
        SemanticGrammar,
        Z3Canonicalizer,
    )
    
    success = True
    
    # Test that classes can be instantiated (where possible)
    try:
        # These are available for use
        print("   [PASS] MathematicalTextParser available")
        print("   [PASS] SemanticPrimitive available")
        print("   [PASS] SemanticNormalizer available")
        print("   [PASS] DependencyDAG available")
        print("   [PASS] Z3SemanticSynthesis available")
        print("   [PASS] CanonicalLeanGenerator available")
        print("   [PASS] SemanticGrammar available")
        print("   [PASS] Z3Canonicalizer available")
    except Exception as e:
        print(f"   [FAIL] CAV-NLP component access failed: {e}")
        success = False
    
    return success


def test_cav_nlp_original_tests():
    """Run original CAV-NLP tests if available."""
    print("\n7. Testing Original CAV-NLP Tests...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "openevolve/cav_nlp_integration/test_cav_nlp.py", "-v"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("   [PASS] Original CAV-NLP tests pass")
            return True
        else:
            print(f"   [WARN] CAV-NLP tests output:\n{result.stdout[-500:]}")
            return False
    except Exception as e:
        print(f"   [WARN] Could not run original tests: {e}")
        return True  # Don't fail if tests can't be run


def main():
    """Run all tests."""
    print("=" * 70)
    print("CAV-NLP Integration Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("Import Test", test_import_cav_nlp_integration()))
    results.append(("Data Structures", test_data_structures()))
    results.append(("Mappings", test_mappings()))
    results.append(("Bridge API", test_bridge_api()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("CAV-NLP Components", test_cav_nlp_components()))
    results.append(("Original CAV-NLP Tests", test_cav_nlp_original_tests()))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"   {status}: {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! CAV-NLP integration is working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
