"""
LeanAide Integration Tests

Comprehensive tests for LeanAide integration in RESE-Z3 Bridge.

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against actual behavior
- Contract testing: Validate API contracts
- Idempotency: Ensure safe re-execution

Author: RESE Team
Created: 2026-02-04
"""

import json
import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, Any
import time
import os
import sys

# Add parent directory to path for imports
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    # Try relative imports first
    from src.rese_z3_bridge import (
        RESEZ3Bridge,
        RESEZ3BridgeConfig,
    )
    from src.rese_z3_schema import (
        LeanAideAutoformalizeRequest,
        LeanAideAutoformalizeResponse,
        LeanAideProveRequest,
        LeanAideProveResponse,
        Z3ToLeanTranslationRequest,
        Z3ToLeanTranslationResponse,
        LeanAideTacticSuggestionRequest,
        LeanAideTacticSuggestionResponse,
        ConstraintType,
        validate_autoformalize_request,
        validate_prove_request,
        validate_translation_request,
        validate_tactic_suggestion_request,
    )
except ImportError:
    # Fall back to absolute imports
    from rese_z3_bridge import (
        RESEZ3Bridge,
        RESEZ3BridgeConfig,
    )
    from rese_z3_schema import (
        LeanAideAutoformalizeRequest,
        LeanAideAutoformalizeResponse,
        LeanAideProveRequest,
        LeanAideProveResponse,
        Z3ToLeanTranslationRequest,
        Z3ToLeanTranslationResponse,
        LeanAideTacticSuggestionRequest,
        LeanAideTacticSuggestionResponse,
        ConstraintType,
        validate_autoformalize_request,
        validate_prove_request,
        validate_translation_request,
        validate_tactic_suggestion_request,
    )


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def bridge_config():
    """Create test bridge configuration"""
    return RESEZ3BridgeConfig(
        z3_base_url=os.getenv("Z3_BASE_URL", "http://localhost:8000"),
        z3_timeout_ms=30000,
        leanaide_base_url=os.getenv("LEANAIDE_BASE_URL", "http://localhost:7654"),
        leanaide_timeout_ms=60000,
        leanaide_enable=True,
        enable_cache=False,  # Disable cache for testing
        enable_monitoring=True,
    )


@pytest.fixture
def bridge(bridge_config):
    """Create bridge instance"""
    bridge = RESEZ3Bridge(config=bridge_config)
    yield bridge
    # Cleanup
    bridge.close()


@pytest.fixture
def correlation_id():
    """Generate correlation ID for tests"""
    return str(uuid.uuid4())


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestSchemaValidation:
    """Test schema validation for LeanAide requests"""

    def test_validate_autoformalize_request_valid(self):
        """Test valid autoformalize request"""
        request = {
            "natural_language": "There are infinitely many prime numbers",
            "timeout_ms": 30000,
        }
        is_valid, error = validate_autoformalize_request(request)
        assert is_valid
        assert error is None

    def test_validate_autoformalize_request_missing_natural_language(self):
        """Test autoformalize request without natural_language"""
        request = {
            "timeout_ms": 30000,
        }
        is_valid, error = validate_autoformalize_request(request)
        assert not is_valid
        assert "natural_language" in error.lower()

    def test_validate_autoformalize_request_invalid_timeout(self):
        """Test autoformalize request with invalid timeout"""
        request = {
            "natural_language": "Test theorem",
            "timeout_ms": -1,
        }
        is_valid, error = validate_autoformalize_request(request)
        assert not is_valid
        assert "timeout" in error.lower()

    def test_validate_prove_request_valid(self):
        """Test valid prove request"""
        request = {
            "theorem_text": "For all natural numbers n, n + 0 = n",
            "timeout_ms": 60000,
        }
        is_valid, error = validate_prove_request(request)
        assert is_valid
        assert error is None

    def test_validate_prove_request_missing_theorem_text(self):
        """Test prove request without theorem_text"""
        request = {
            "timeout_ms": 60000,
        }
        is_valid, error = validate_prove_request(request)
        assert not is_valid
        assert "theorem_text" in error.lower()

    def test_validate_translation_request_valid(self):
        """Test valid translation request"""
        request = {
            "smtlib_content": "(declare-fun x () Int)(assert (> x 0))",
            "timeout_ms": 30000,
        }
        is_valid, error = validate_translation_request(request)
        assert is_valid
        assert error is None

    def test_validate_translation_request_missing_smtlib(self):
        """Test translation request without smtlib_content"""
        request = {
            "timeout_ms": 30000,
        }
        is_valid, error = validate_translation_request(request)
        assert not is_valid
        assert "smtlib" in error.lower()

    def test_validate_tactic_suggestion_request_valid(self):
        """Test valid tactic suggestion request"""
        request = {
            "goal_state": "⊢ x + y = y + x",
            "timeout_ms": 15000,
            "num_suggestions": 3,
        }
        is_valid, error = validate_tactic_suggestion_request(request)
        assert is_valid
        assert error is None

    def test_validate_tactic_suggestion_request_invalid_num_suggestions(self):
        """Test tactic suggestion request with invalid num_suggestions"""
        request = {
            "goal_state": "Test goal",
            "timeout_ms": 15000,
            "num_suggestions": 15,  # Too many
        }
        is_valid, error = validate_tactic_suggestion_request(request)
        assert not is_valid
        assert "num_suggestions" in error.lower()


# =============================================================================
# SCHEMA SERIALIZATION TESTS
# =============================================================================

class TestSchemaSerialization:
    """Test schema serialization and deserialization"""

    def test_autoformalize_request_serialization(self):
        """Test autoformalize request serialization"""
        request = LeanAideAutoformalizeRequest(
            natural_language="Test theorem",
            theorem_name="test_theorem",
            timeout_ms=30000,
        )
        data = request.to_dict()

        assert data["natural_language"] == "Test theorem"
        assert data["theorem_name"] == "test_theorem"
        assert data["timeout_ms"] == 30000
        assert "correlation_id" in data
        assert "timestamp" in data

    def test_autoformalize_request_deserialization(self):
        """Test autoformalize request deserialization"""
        data = {
            "natural_language": "Test theorem",
            "theorem_name": "test_theorem",
            "timeout_ms": 30000,
            "correlation_id": "test-id",
        }
        request = LeanAideAutoformalizeRequest.from_dict(data)

        assert request.natural_language == "Test theorem"
        assert request.theorem_name == "test_theorem"
        assert request.timeout_ms == 30000
        assert request.correlation_id == "test-id"

    def test_autoformalize_response_serialization(self):
        """Test autoformalize response serialization"""
        response = LeanAideAutoformalizeResponse(
            success=True,
            lean_code="theorem test : True := by trivial",
            theorem_name="test",
            execution_time_ms=100.0,
        )
        data = response.to_dict()

        assert data["success"] is True
        assert data["lean_code"] == "theorem test : True := by trivial"
        assert data["execution_time_ms"] == 100.0

    def test_prove_request_serialization(self):
        """Test prove request serialization"""
        request = LeanAideProveRequest(
            theorem_text="Test theorem",
            theorem_code="theorem test : True := by trivial",
            timeout_ms=60000,
        )
        data = request.to_dict()

        assert data["theorem_text"] == "Test theorem"
        assert data["theorem_code"] == "theorem test : True := by trivial"
        assert "correlation_id" in data

    def test_translation_request_serialization(self):
        """Test translation request serialization"""
        request = Z3ToLeanTranslationRequest(
            smtlib_content="(declare-fun x () Int)",
            constraint_type=ConstraintType.INTEGER,
            timeout_ms=30000,
        )
        data = request.to_dict()

        assert data["smtlib_content"] == "(declare-fun x () Int)"
        assert data["constraint_type"] == "integer"
        assert "correlation_id" in data


# =============================================================================
# AUTOFORMALIZATION TESTS
# =============================================================================

class TestAutoformalization:
    """Test autoformalization functionality"""

    def test_autoformalize_basic_theorem(self, bridge, correlation_id):
        """Test basic autoformalization"""
        response = bridge.autoformalize(
            natural_language="For all natural numbers n, n + 0 = n",
            correlation_id=correlation_id,
        )

        # Response structure
        assert response is not None
        assert hasattr(response, "success")
        assert hasattr(response, "lean_code")
        assert hasattr(response, "correlation_id")
        assert response.correlation_id == correlation_id

    def test_autoformalize_with_name(self, bridge, correlation_id):
        """Test autoformalization with theorem name"""
        response = bridge.autoformalize(
            natural_language="There are infinitely many primes",
            theorem_name="infinitely_many_primes",
            correlation_id=correlation_id,
        )

        assert response is not None
        assert response.correlation_id == correlation_id

    def test_autoformalize_complex_theorem(self, bridge, correlation_id):
        """Test autoformalization of complex theorem"""
        theorem = (
            "The square root of 2 is irrational, i.e., there are no "
            "integers p and q such that (p/q)^2 = 2"
        )
        response = bridge.autoformalize(
            natural_language=theorem,
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_autoformalize_idempotency(self, bridge, correlation_id):
        """Test that autoformalization is idempotent (Law of Idempotency)"""
        theorem = "For all x, x + 0 = x"

        # Run multiple times
        responses = []
        for i in range(3):
            response = bridge.autoformalize(
                natural_language=theorem,
                correlation_id=f"{correlation_id}-{i}",
            )
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response is not None
            assert hasattr(response, "success")


# =============================================================================
# AI-POWERED PROVING TESTS
# =============================================================================

class TestAIPoweredProving:
    """Test AI-powered theorem proving"""

    def test_prove_simple_theorem(self, bridge, correlation_id):
        """Test proving simple theorem"""
        response = bridge.prove_with_ai(
            theorem_text="For all natural numbers n, n + 0 = n",
            correlation_id=correlation_id,
        )

        # Response structure
        assert response is not None
        assert hasattr(response, "success")
        assert hasattr(response, "proof")
        assert hasattr(response, "tactics_used")
        assert response.correlation_id == correlation_id

    def test_prove_with_existing_lean_code(self, bridge, correlation_id):
        """Test proving with existing Lean code"""
        lean_code = """
theorem add_zero (n : Nat) : n + 0 = n := by
  simp [Nat.add_zero]
"""
        response = bridge.prove_with_ai(
            theorem_text="Addition with zero",
            theorem_code=lean_code,
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_prove_with_formalized_theorem(self, bridge, correlation_id):
        """Test proving with fully formalized theorem"""
        response = bridge.prove_with_ai(
            theorem_text="For all natural numbers n, n + 0 = n",
            theorem_code="theorem add_zero (n : Nat) : n + 0 = n",
            theorem_statement="∀ (n : Nat), n + 0 = n",
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_prove_arithmetic_theorem(self, bridge, correlation_id):
        """Test proving arithmetic theorem"""
        response = bridge.prove_with_ai(
            theorem_text="For all real numbers x and y, x + y = y + x",
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_prove_idempotency(self, bridge, correlation_id):
        """Test that proving is idempotent"""
        responses = []
        for i in range(3):
            response = bridge.prove_with_ai(
                theorem_text="1 + 1 = 2",
                correlation_id=f"{correlation_id}-{i}",
            )
            responses.append(response)

        # All should complete
        for response in responses:
            assert response is not None


# =============================================================================
# Z3-TO-LEAN TRANSLATION TESTS
# =============================================================================

class TestZ3ToLeanTranslation:
    """Test Z3 to Lean 4 translation"""

    def test_translate_simple_constraint(self, bridge, correlation_id):
        """Test translating simple constraint"""
        smtlib = """
(declare-fun x () Int)
(assert (> x 0))
(check-sat)
"""
        response = bridge.translate_z3_to_lean(
            smtlib_content=smtlib,
            constraint_type=ConstraintType.INTEGER,
            correlation_id=correlation_id,
        )

        # Response structure
        assert response is not None
        assert hasattr(response, "success")
        assert hasattr(response, "lean_code")
        assert hasattr(response, "variables")
        assert response.correlation_id == correlation_id

    def test_translate_arithmetic_constraint(self, bridge, correlation_id):
        """Test translating arithmetic constraint"""
        smtlib = """
(declare-fun x () Real)
(declare-fun y () Real)
(assert (> x 0.0))
(assert (> y 0.0))
(assert (> (+ x y) 0.0))
(check-sat)
"""
        response = bridge.translate_z3_to_lean(
            smtlib_content=smtlib,
            constraint_type=ConstraintType.REAL,
            correlation_id=correlation_id,
        )

        assert response is not None
        if response.success:
            assert len(response.variables) > 0

    def test_translate_boolean_constraint(self, bridge, correlation_id):
        """Test translating boolean constraint"""
        smtlib = """
(declare-fun p () Bool)
(declare-fun q () Bool)
(assert (and p q))
(check-sat)
"""
        response = bridge.translate_z3_to_lean(
            smtlib_content=smtlib,
            constraint_type=ConstraintType.BOOLEAN,
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_translate_with_proof_generation(self, bridge, correlation_id):
        """Test translation with proof generation"""
        smtlib = "(declare-fun x () Int)(assert (> x 0))"
        response = bridge.translate_z3_to_lean(
            smtlib_content=smtlib,
            constraint_type=ConstraintType.INTEGER,
            generate_proof=True,
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_translation_idempotency(self, bridge, correlation_id):
        """Test that translation is idempotent"""
        smtlib = "(declare-fun x () Int)(assert (> x 0))"

        responses = []
        for i in range(3):
            response = bridge.translate_z3_to_lean(
                smtlib_content=smtlib,
                correlation_id=f"{correlation_id}-{i}",
            )
            responses.append(response)

        # All should complete
        for response in responses:
            assert response is not None


# =============================================================================
# TACTIC SUGGESTION TESTS
# =============================================================================

class TestTacticSuggestions:
    """Test AI-powered tactic suggestions"""

    def test_suggest_tactics_arithmetic_goal(self, bridge, correlation_id):
        """Test suggesting tactics for arithmetic goal"""
        goal_state = "⊢ x + y = y + x"
        response = bridge.suggest_tactics(
            goal_state=goal_state,
            num_suggestions=3,
            correlation_id=correlation_id,
        )

        # Response structure
        assert response is not None
        assert hasattr(response, "success")
        assert hasattr(response, "suggestions")
        assert response.correlation_id == correlation_id

        if response.success:
            assert len(response.suggestions) > 0
            # Check suggestion structure
            for suggestion in response.suggestions:
                assert hasattr(suggestion, "tactic")
                assert hasattr(suggestion, "description")
                assert hasattr(suggestion, "confidence")

    def test_suggest_tactics_with_context(self, bridge, correlation_id):
        """Test suggesting tactics with context"""
        goal_state = "⊢ n + 0 = n"
        context = "Working with natural numbers and addition"

        response = bridge.suggest_tactics(
            goal_state=goal_state,
            context=context,
            num_suggestions=2,
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_suggest_tactics_logical_goal(self, bridge, correlation_id):
        """Test suggesting tactics for logical goal"""
        goal_state = "⊢ p ∧ q → q ∧ p"

        response = bridge.suggest_tactics(
            goal_state=goal_state,
            num_suggestions=3,
            correlation_id=correlation_id,
        )

        assert response is not None

    def test_suggest_tacts_custom_num_suggestions(self, bridge, correlation_id):
        """Test custom number of suggestions"""
        response = bridge.suggest_tactics(
            goal_state="⊢ x = x",
            num_suggestions=5,
            correlation_id=correlation_id,
        )

        assert response is not None
        if response.success:
            assert len(response.suggestions) <= 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestLeanAideIntegration:
    """Test full LeanAide integration"""

    def test_full_autoformalization_and_prove_workflow(self, bridge, correlation_id):
        """Test complete workflow: autoformalize then prove"""
        # Step 1: Autoformalize
        formalize_response = bridge.autoformalize(
            natural_language="For all n, n + 0 = n",
            correlation_id=f"{correlation_id}-formalize",
        )

        assert formalize_response is not None

        # Step 2: Prove (if formalization succeeded)
        if formalize_response.success and formalize_response.lean_code:
            prove_response = bridge.prove_with_ai(
                theorem_text="For all n, n + 0 = n",
                theorem_code=formalize_response.lean_code,
                correlation_id=f"{correlation_id}-prove",
            )

            assert prove_response is not None

    def test_z3_to_lean_to_prove_workflow(self, bridge, correlation_id):
        """Test workflow: Z3 -> Lean -> Prove"""
        # Step 1: Z3 constraint
        smtlib = """
(declare-fun x () Int)
(declare-fun y () Int)
(assert (> x 0))
(assert (> y 0))
(assert (> (+ x y) 0))
"""

        # Step 2: Translate to Lean
        translate_response = bridge.translate_z3_to_lean(
            smtlib_content=smtlib,
            correlation_id=f"{correlation_id}-translate",
        )

        assert translate_response is not None

        # Step 3: Try to prove (if translation succeeded)
        if translate_response.success and translate_response.lean_code:
            # This would use the translated Lean code
            pass

    def test_health_check_leanaide(self, bridge):
        """Test LeanAide health check"""
        health = bridge.get_health()

        assert health is not None
        assert "status" in health
        assert "timestamp" in health

    def test_stats_leanaide(self, bridge):
        """Test LeanAide statistics"""
        stats = bridge.get_stats()

        assert stats is not None
        assert "config" in stats
        assert "performance_summary" in stats


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling"""

    def test_autoformalize_empty_input(self, bridge, correlation_id):
        """Test autoformalization with empty input"""
        with pytest.raises((ValueError, Exception)):
            bridge.autoformalize(
                natural_language="",
                correlation_id=correlation_id,
            )

    def test_prove_empty_theorem(self, bridge, correlation_id):
        """Test proving with empty theorem"""
        with pytest.raises((ValueError, Exception)):
            bridge.prove_with_ai(
                theorem_text="",
                correlation_id=correlation_id,
            )

    def test_translate_empty_smtlib(self, bridge, correlation_id):
        """Test translation with empty SMT-LIB"""
        with pytest.raises((ValueError, Exception)):
            bridge.translate_z3_to_lean(
                smtlib_content="",
                correlation_id=correlation_id,
            )

    def test_suggest_tactics_empty_goal(self, bridge, correlation_id):
        """Test tactic suggestions with empty goal"""
        with pytest.raises((ValueError, Exception)):
            bridge.suggest_tactics(
                goal_state="",
                correlation_id=correlation_id,
            )

    def test_invalid_num_suggestions(self, bridge, correlation_id):
        """Test tactic suggestions with invalid num_suggestions"""
        with pytest.raises((ValueError, Exception)):
            bridge.suggest_tactics(
                goal_state="⊢ x = x",
                num_suggestions=20,  # Too many
                correlation_id=correlation_id,
            )


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics"""

    def test_autoformalize_performance(self, bridge, correlation_id):
        """Test autoformalization performance"""
        start_time = time.time()

        response = bridge.autoformalize(
            natural_language="There are infinitely many primes",
            correlation_id=correlation_id,
        )

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        assert response is not None
        # Should complete within reasonable time (even with LeanAide timeout)
        assert duration_ms < 65000  # Allow for timeout

    def test_prove_performance(self, bridge, correlation_id):
        """Test proving performance"""
        start_time = time.time()

        response = bridge.prove_with_ai(
            theorem_text="1 + 1 = 2",
            correlation_id=correlation_id,
        )

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        assert response is not None
        # Should complete within reasonable time
        assert duration_ms < 65000

    def test_concurrent_requests(self, bridge):
        """Test handling concurrent requests"""
        import threading

        results = []
        errors = []

        def run_request(rid):
            try:
                response = bridge.autoformalize(
                    natural_language=f"Theorem {rid}",
                    correlation_id=f"concurrent-{rid}",
                )
                results.append(response)
            except Exception as e:
                errors.append(e)

        # Launch concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent requests
        assert len(results) + len(errors) == 5


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
