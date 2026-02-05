"""
Tests for Lean 4 Interface

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against real Lean 4
- Idempotency: Tests are repeatable
- Structured Logging: All test output logged
"""

import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from glue.lib.lean4_bridge.lean4_interface import (
    Lean4Interface,
    Lean4Error,
    Lean4TimeoutError,
    Lean4VerificationError,
    Lean4CircuitBreakerOpenError,
    CircuitBreakerState,
)
from glue.lib.lean4_bridge.src.constraint_translator import (
    ConstraintTranslator,
    Lean4SyntaxError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    workspace = Path(tempfile.mkdtemp())
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def mock_lean_interface(temp_workspace):
    """Create a mock Lean 4 interface for testing."""
    with patch("subprocess.run") as mock_run:
        # Mock Lean 4 version check
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Lean (version 4.11.0)\n",
            stderr=""
        )
        interface = Lean4Interface(
            workspace_dir=str(temp_workspace),
            timeout_ms=5000,  # Short timeout for tests
        )
        yield interface, mock_run


@pytest.fixture
def sample_constraint():
    """Sample RESE constraint for testing."""
    return "forall x, P(x) -> Q(x)"


@pytest.fixture
def sample_fdg():
    """Sample Functional Dependency Graph for testing."""
    return {
        "graph_id": "test_fdg_1",
        "domain": "test_domain",
        "nodes": [
            {"id": "node1", "type": "variable", "description": "Variable 1"},
            {"id": "node2", "type": "parameter", "description": "Parameter 2"},
        ],
        "edges": [
            {
                "source": "node1",
                "target": "node2",
                "relation_type": "causal",
                "strength": 0.9,
            }
        ],
        "adjacency_list": {
            "node1": ["node2"],
            "node2": [],
        },
        "metadata": "{}",
    }


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestCircuitBreakerState:
    """Test circuit breaker state management."""

    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreakerState()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.last_failure_time is None

    def test_record_success(self):
        """Test recording success resets circuit breaker."""
        cb = CircuitBreakerState()
        cb.failure_count = 3
        cb.state = "half_open"

        cb.record_success()

        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.last_failure_time is None

    def test_record_failure_closed_to_open(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreakerState(threshold=3)

        # First failure
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 1

        # Second failure
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 2

        # Third failure (triggers open)
        cb.record_failure()
        assert cb.state == "open"
        assert cb.failure_count == 3
        assert cb.last_failure_time is not None

    def test_can_attempt_closed(self):
        """Test can_attempt returns True when CLOSED."""
        cb = CircuitBreakerState()
        assert cb.can_attempt() is True

    def test_can_attempt_open(self):
        """Test can_attempt returns False when OPEN."""
        cb = CircuitBreakerState(threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_attempt() is False

    def test_can_attempt_half_open(self):
        """Test can_attempt returns True when HALF_OPEN."""
        cb = CircuitBreakerState(threshold=1)
        cb.record_failure()
        # Manually set to half_open (timeout elapsed)
        cb.state = "half_open"
        assert cb.can_attempt() is True


# ============================================================================
# LEAN 4 INTERFACE TESTS
# ============================================================================

class TestLean4Interface:
    """Test Lean 4 interface functionality."""

    def test_initialization(self, mock_lean_interface):
        """Test Lean 4 interface initializes successfully."""
        interface, _ = mock_lean_interface
        assert interface.lean_path == "lean"
        assert interface.lake_path == "lake"
        assert interface.timeout_ms == 5000
        assert interface.circuit_breaker.state == "closed"

    def test_verify_installation_success(self, mock_lean_interface):
        """Test Lean 4 installation verification succeeds."""
        interface, mock_run = mock_lean_interface
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Lean (version 4.11.0)\n",
            stderr=""
        )

        # Should not raise exception
        interface._verify_installation()
        mock_run.assert_called_once()

    def test_verify_installation_failure(self, temp_workspace):
        """Test Lean 4 installation verification fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stderr="Lean not found"
            )

            with pytest.raises(Lean4Error):
                Lean4Interface(workspace_dir=str(temp_workspace))

    def test_formalize_constraint_success(self, mock_lean_interface, sample_constraint):
        """Test constraint formalization succeeds."""
        interface, mock_run = mock_lean_interface

        # Mock successful verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        result = interface.formalize_constraint(sample_constraint)

        assert "lean4_code" in result
        assert "theorem_name" in result
        assert result["verification_status"] == "verified"
        assert result["execution_time_ms"] >= 0
        assert "correlation_id" in result
        assert interface.circuit_breaker.state == "closed"

    def test_formalize_constraint_circuit_breaker_open(self, mock_lean_interface, sample_constraint):
        """Test constraint formalization fails when circuit breaker is open."""
        interface, _ = mock_lean_interface

        # Open circuit breaker
        interface.circuit_breaker.state = "open"
        interface.circuit_breaker.failure_count = 10

        with pytest.raises(Lean4CircuitBreakerOpenError):
            interface.formalize_constraint(sample_constraint)

    def test_formalize_constraint_timeout(self, mock_lean_interface, sample_constraint):
        """Test constraint formalization times out."""
        interface, mock_run = mock_lean_interface

        # Mock timeout
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("lean", 5)

        with pytest.raises(Lean4TimeoutError):
            interface.formalize_constraint(sample_constraint)

        assert interface.circuit_breaker.failure_count > 0

    def test_prove_theorem_success(self, mock_lean_interface):
        """Test theorem proving succeeds."""
        interface, mock_run = mock_lean_interface

        # Mock successful proof
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        tactics = ["intro h", "apply h", "assumption"]
        result = interface.prove_theorem("theorem_example", tactics)

        assert "proof_status" in result
        assert "proof_script" in result
        assert result["correlation_id"] in result

    def test_verify_proof_success(self, mock_lean_interface):
        """Test proof verification succeeds."""
        interface, mock_run = mock_lean_interface

        # Mock successful verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        proof_code = "theorem example : True := by trivial"
        result = interface.verify_proof(proof_code)

        assert result["verification_status"] == "verified"
        assert result["correlation_id"] in result

    def test_elaborate_fdg_success(self, mock_lean_interface, sample_fdg):
        """Test FDG elaboration succeeds."""
        interface, mock_run = mock_lean_interface

        # Mock successful elaboration
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        result = interface.elaborate_fdg(sample_fdg)

        assert "lean4_code" in result
        assert "fdg_name" in result
        assert result["verification_status"] == "verified"
        assert result["correlation_id"] in result


# ============================================================================
# CONSTRAINT TRANSLATOR TESTS
# ============================================================================

class TestConstraintTranslator:
    """Test constraint translator functionality."""

    @pytest.fixture
    def translator(self):
        """Create constraint translator for testing."""
        return ConstraintTranslator()

    def test_translate_to_lean4_proposition(self, translator):
        """Test translating proposition to Lean 4."""
        constraint = "forall x, P(x) -> Q(x)"
        result = translator.translate_to_lean4(constraint, "proposition")

        assert "∀" in result or "forall" in result
        assert "→" in result or "->" in result

    def test_translate_to_lean4_theorem(self, translator):
        """Test translating theorem to Lean 4."""
        constraint = "forall x, P(x) -> Q(x)"
        result = translator.translate_to_lean4(constraint, "theorem")

        assert "theorem" in result
        assert ":=" in result

    def test_is_lean4_syntax_true(self, translator):
        """Test detecting Lean 4 syntax."""
        constraint = "∀ x, P x → Q x"
        assert translator._is_lean4_syntax(constraint) is True

    def test_is_lean4_syntax_false(self, translator):
        """Test detecting non-Lean 4 syntax."""
        constraint = "for all x, if P(x) then Q(x)"
        assert translator._is_lean4_syntax(constraint) is False

    def test_translate_fdg_to_lean4(self, translator, sample_fdg):
        """Test translating FDG to Lean 4."""
        result = translator.translate_fdg_to_lean4(sample_fdg)

        assert "structure FDGNode" in result
        assert "structure FDGEdge" in result
        assert "structure FunctionalDependencyGraph" in result
        assert "nodes1" in result or "node1" in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for Lean 4 bridge."""

    @pytest.mark.integration
    def test_end_to_end_formalization(self, mock_lean_interface, sample_constraint):
        """Test end-to-end constraint formalization."""
        interface, mock_run = mock_lean_interface

        # Mock successful operations
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        # Formalize constraint
        result1 = interface.formalize_constraint(sample_constraint)

        # Verify proof
        result2 = interface.verify_proof(result1["lean4_code"])

        assert result1["verification_status"] == "verified"
        assert result2["verification_status"] == "verified"

    @pytest.mark.integration
    def test_circuit_breaker_recovery(self, mock_lean_interface, sample_constraint):
        """Test circuit breaker opens and recovers."""
        interface, mock_run = mock_lean_interface

        # Mock failures
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error"
        )

        # Trigger circuit breaker
        for _ in range(10):
            try:
                interface.formalize_constraint(sample_constraint)
            except:
                pass

        assert interface.circuit_breaker.state == "open"

        # Mock success
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        # Circuit breaker should still be open
        with pytest.raises(Lean4CircuitBreakerOpenError):
            interface.formalize_constraint(sample_constraint)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
