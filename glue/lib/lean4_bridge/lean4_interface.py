"""
Lean 4 Interface for RESE Formal Verification

Main interface class for communicating with Lean 4 formal verification system.
Provides methods to formalize constraints, prove theorems, verify proofs, and
elaborate Functional Dependency Graphs (FDGs).

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Runtime Truth: Verify Lean 4 works before using
- Circuit Breaker: Stop hammering if Lean 4 is down
- Structured Logging: JSON with correlation_id
- Law of UTC: All timestamps in UTC

Usage:
    >>> interface = Lean4Interface()
    >>> result = interface.formalize_constraint("forall x, P(x) -> Q(x)")
"""

import os
import json
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import structlog

from .src.constraint_translator import ConstraintTranslator, Lean4SyntaxError


# ============================================================================
# EXCEPTIONS
# ============================================================================

class Lean4Error(Exception):
    """Base exception for Lean 4 interface errors."""
    pass


class Lean4TimeoutError(Lean4Error):
    """Lean 4 execution timeout (Law of Configuration Explicitness)."""
    pass


class Lean4VerificationError(Lean4Error):
    """Lean 4 proof verification failed."""
    pass


class Lean4CircuitBreakerOpenError(Lean4Error):
    """Circuit breaker is open (too many failures)."""
    pass


# ============================================================================
# CIRCUIT BREAKER (Following CLAUDE.md)
# ============================================================================

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for Lean 4 failures (Law of Zero Trust)."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open

    # Configuration (from env vars, Law of Configuration Explicitness)
    threshold: int = field(default_factory=lambda: int(os.getenv("LEAN4_CIRCUIT_BREAKER_THRESHOLD", "5")))
    timeout_ms: int = field(default_factory=lambda: int(os.getenv("LEAN4_CIRCUIT_BREAKER_TIMEOUT_MS", "60000")))
    half_open_attempts: int = field(default_factory=lambda: int(os.getenv("LEAN4_CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS", "3")))

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.threshold:
            self.state = "open"

    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() * 1000
                if elapsed >= self.timeout_ms:
                    self.state = "half_open"
                    return False  # Need explicit half-open attempt
            return False

        if self.state == "half_open":
            return True

        return False


# ============================================================================
# LEAN 4 INTERFACE
# ============================================================================

class Lean4Interface:
    """
    Python interface to Lean 4 formal verification system.

    Responsibilities:
    1. Formalize RESE constraints in Lean 4
    2. Prove theorems using Lean 4 tactics
    3. Verify proof correctness
    4. Elaborate Functional Dependency Graphs (FDGs)

    Attributes:
        lean_path: Path to Lean 4 executable
        lake_path: Path to Lake build system
        workspace_dir: Lean 4 workspace directory
        timeout_ms: Timeout for Lean 4 operations (from env var)
        logger: Structured logger (JSON output)
    """

    def __init__(
        self,
        lean_path: Optional[str] = None,
        lake_path: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ):
        """
        Initialize Lean 4 interface.

        Args:
            lean_path: Path to Lean 4 executable (default: from env or 'lean')
            lake_path: Path to Lake executable (default: from env or 'lake')
            workspace_dir: Lean 4 workspace directory (default: /workspace/lean4)
            timeout_ms: Timeout in milliseconds (default: from env or 30000)

        Raises:
            Lean4Error: If Lean 4 is not available
        """
        # Configuration from environment (Law of Configuration Explicitness)
        self.lean_path = lean_path or os.getenv("LEAN4_PATH", "lean")
        self.lake_path = lake_path or os.getenv("LAKE4_PATH", "lake")
        self.workspace_dir = Path(workspace_dir or os.getenv("LEAN4_WORKSPACE_DIR", "/workspace/lean4"))
        self.timeout_ms = int(timeout_ms or os.getenv("LEAN4_TIMEOUT_MS", "30000"))

        # Circuit breaker for failure handling
        self.circuit_breaker = CircuitBreakerState()

        # Structured logger (JSON output, Law of Structured Logging)
        self.logger = structlog.get_logger()
        self.logger = self.logger.bind(
            component="lean4_interface",
            source_service="lean4_bridge",
            target_service="lean4_formal_verification",
        )

        # Constraint translator
        self.translator = ConstraintTranslator(logger=self.logger)

        # Verify Lean 4 installation (Law of Runtime Truth)
        self._verify_installation()

        self.logger.info(
            "Lean4Interface initialized",
            lean_path=self.lean_path,
            lake_path=self.lake_path,
            workspace_dir=str(self.workspace_dir),
            timeout_ms=self.timeout_ms,
        )

    def _verify_installation(self) -> None:
        """
        Verify Lean 4 installation (Law of Runtime Truth).

        Raises:
            Lean4Error: If Lean 4 is not available
        """
        correlation_id = str(uuid.uuid4())

        try:
            # Check Lean 4 version
            result = subprocess.run(
                [self.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                raise Lean4Error(f"Lean 4 not available: {result.stderr}")

            version = result.stdout.strip()
            self.logger.info(
                "Lean 4 verified",
                version=version,
                correlation_id=correlation_id,
            )

        except FileNotFoundError:
            raise Lean4Error(
                f"Lean 4 executable not found at {self.lean_path}. "
                "Please install Lean 4 or set LEAN4_PATH environment variable."
            )
        except subprocess.TimeoutExpired:
            raise Lean4TimeoutError("Lean 4 version check timed out")

    # ========================================================================
    # CONSTRAINT FORMALIZATION
    # ========================================================================

    def formalize_constraint(
        self,
        constraint: str,
        constraint_type: str = "proposition",
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Formalize a RESE constraint in Lean 4.

        Args:
            constraint: Natural language or formal constraint
            constraint_type: Type of constraint (proposition, theorem, axiom)
            correlation_id: Correlation ID for distributed tracing

        Returns:
            Dict with:
                - lean4_code: Lean 4 formalization
                - theorem_name: Generated theorem name
                - verification_status: Status of verification
                - correlation_id: For distributed tracing

        Raises:
            Lean4CircuitBreakerOpenError: If circuit breaker is open
            Lean4TimeoutError: If formalization times out
            Lean4VerificationError: If formalization fails
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        self.logger.info(
            "Formalizing constraint",
            constraint=constraint,
            constraint_type=constraint_type,
            correlation_id=correlation_id,
        )

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            self.logger.error(
                "Circuit breaker open, refusing request",
                failure_count=self.circuit_breaker.failure_count,
                correlation_id=correlation_id,
            )
            raise Lean4CircuitBreakerOpenError(
                f"Circuit breaker open after {self.circuit_breaker.failure_count} failures"
            )

        try:
            # Translate constraint to Lean 4 syntax
            lean4_code = self.translator.translate_to_lean4(
                constraint,
                constraint_type=constraint_type,
            )

            # Create Lean 4 file
            theorem_name = self._generate_theorem_name(constraint)
            lean4_file = self._create_lean_file(theorem_name, lean4_code)

            # Verify syntax with Lean 4
            verification_result = self._verify_lean_file(lean4_file)

            execution_time_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            result = {
                "lean4_code": lean4_code,
                "theorem_name": theorem_name,
                "verification_status": verification_result["status"],
                "errors": verification_result.get("errors", []),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Record success
            self.circuit_breaker.record_success()

            self.logger.info(
                "Constraint formalized successfully",
                theorem_name=theorem_name,
                verification_status=verification_result["status"],
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )

            return result

        except Lean4SyntaxError as e:
            # Record failure
            self.circuit_breaker.record_failure()

            self.logger.error(
                "Constraint translation failed",
                error=str(e),
                correlation_id=correlation_id,
            )
            raise Lean4VerificationError(f"Translation failed: {e}")

        except subprocess.TimeoutExpired:
            # Record failure
            self.circuit_breaker.record_failure()

            execution_time_ms = self.timeout_ms
            self.logger.error(
                "Constraint formalization timed out",
                timeout_ms=self.timeout_ms,
                correlation_id=correlation_id,
            )
            raise Lean4TimeoutError(
                f"Formalization timed out after {self.timeout_ms}ms"
            )

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()

            execution_time_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            self.logger.error(
                "Constraint formalization failed",
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )
            raise Lean4Error(f"Formalization failed: {e}")

    # ========================================================================
    # THEOREM PROVING
    # ========================================================================

    def prove_theorem(
        self,
        theorem_name: str,
        tactics: List[str],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prove a theorem using Lean 4 tactics.

        Args:
            theorem_name: Name of the theorem to prove
            tactics: List of Lean 4 tactics to apply
            correlation_id: Correlation ID for distributed tracing

        Returns:
            Dict with:
                - proof_status: Status of proof (proved, failed, partial)
                - proof_script: Complete proof script
                - goals_remaining: Goals not yet solved (if any)
                - correlation_id: For distributed tracing

        Raises:
            Lean4TimeoutError: If proof times out
            Lean4VerificationError: If proof fails
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        self.logger.info(
            "Proving theorem",
            theorem_name=theorem_name,
            tactic_count=len(tactics),
            correlation_id=correlation_id,
        )

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            raise Lean4CircuitBreakerOpenError(
                f"Circuit breaker open after {self.circuit_breaker.failure_count} failures"
            )

        try:
            # Create proof script
            proof_script = self._create_proof_script(theorem_name, tactics)

            # Create Lean 4 file
            lean4_file = self._create_lean_file(theorem_name, proof_script)

            # Run Lean 4 to verify proof
            verification_result = self._verify_lean_file(lean4_file)

            execution_time_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            result = {
                "proof_status": verification_result["status"],
                "proof_script": proof_script,
                "goals_remaining": verification_result.get("goals_remaining", []),
                "errors": verification_result.get("errors", []),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Record success or partial success
            if verification_result["status"] in ["proved", "partial"]:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()

            self.logger.info(
                "Theorem proof completed",
                theorem_name=theorem_name,
                proof_status=verification_result["status"],
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )

            return result

        except subprocess.TimeoutExpired:
            self.circuit_breaker.record_failure()
            raise Lean4TimeoutError(f"Proof timed out after {self.timeout_ms}ms")

        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Lean4Error(f"Proof failed: {e}")

    # ========================================================================
    # PROOF VERIFICATION
    # ========================================================================

    def verify_proof(
        self,
        proof_code: str,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify a Lean 4 proof.

        Args:
            proof_code: Lean 4 proof code to verify
            correlation_id: Correlation ID for distributed tracing

        Returns:
            Dict with:
                - verification_status: Status (verified, failed, errors)
                - errors: List of errors (if any)
                - correlation_id: For distributed tracing
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        self.logger.info(
            "Verifying proof",
            proof_length=len(proof_code),
            correlation_id=correlation_id,
        )

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            raise Lean4CircuitBreakerOpenError(
                f"Circuit breaker open after {self.circuit_breaker.failure_count} failures"
            )

        try:
            # Create Lean 4 file
            theorem_name = f"verify_{uuid.uuid4().hex[:8]}"
            lean4_file = self._create_lean_file(theorem_name, proof_code)

            # Verify with Lean 4
            verification_result = self._verify_lean_file(lean4_file)

            execution_time_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            result = {
                "verification_status": verification_result["status"],
                "errors": verification_result.get("errors", []),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Record success or failure
            if verification_result["status"] == "verified":
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()

            self.logger.info(
                "Proof verification completed",
                verification_status=verification_result["status"],
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )

            return result

        except subprocess.TimeoutExpired:
            self.circuit_breaker.record_failure()
            raise Lean4TimeoutError(f"Verification timed out after {self.timeout_ms}ms")

        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Lean4Error(f"Verification failed: {e}")

    # ========================================================================
    # FDG ELABORATION
    # ========================================================================

    def elaborate_fdg(
        self,
        fdg: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Elaborate a Functional Dependency Graph (FDG) in Lean 4.

        Args:
            fdg: Functional dependency graph (from RESE Phase II)
            correlation_id: Correlation ID for distributed tracing

        Returns:
            Dict with:
                - lean4_code: Lean 4 formalization of FDG
                - fdg_theorems: Theorems about the FDG
                - verification_status: Status of verification
                - correlation_id: For distributed tracing
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        self.logger.info(
            "Elaborating FDG",
            node_count=len(fdg.get("nodes", [])),
            edge_count=len(fdg.get("edges", [])),
            correlation_id=correlation_id,
        )

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            raise Lean4CircuitBreakerOpenError(
                f"Circuit breaker open after {self.circuit_breaker.failure_count} failures"
            )

        try:
            # Translate FDG to Lean 4
            lean4_code = self.translator.translate_fdg_to_lean4(fdg)

            # Create Lean 4 file
            fdg_name = f"fdg_{uuid.uuid4().hex[:8]}"
            lean4_file = self._create_lean_file(fdg_name, lean4_code)

            # Verify with Lean 4
            verification_result = self._verify_lean_file(lean4_file)

            execution_time_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            result = {
                "lean4_code": lean4_code,
                "fdg_name": fdg_name,
                "fdg_theorems": verification_result.get("theorems", []),
                "verification_status": verification_result["status"],
                "errors": verification_result.get("errors", []),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Record success or failure
            if verification_result["status"] == "verified":
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()

            self.logger.info(
                "FDG elaboration completed",
                fdg_name=fdg_name,
                verification_status=verification_result["status"],
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Lean4Error(f"FDG elaboration failed: {e}")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _generate_theorem_name(self, constraint: str) -> str:
        """Generate a unique theorem name from constraint."""
        # Create safe name from constraint
        safe_name = constraint.lower()[:50]
        safe_name = "".join(c if c.isalnum() else "_" for c in safe_name)
        unique_id = uuid.uuid4().hex[:8]
        return f"theorem_{safe_name}_{unique_id}"

    def _create_lean_file(self, name: str, code: str) -> Path:
        """Create a Lean 4 file in the workspace."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        lean_file = self.workspace_dir / f"{name}.lean"

        with open(lean_file, "w") as f:
            f.write(code)

        return lean_file

    def _create_proof_script(self, theorem_name: str, tactics: List[str]) -> str:
        """Create a complete proof script from tactics."""
        tactic_string = "\n  ".join(tactics)
        return f"""
theorem {theorem_name} : True := by
  {tactic_string}
"""

    def _verify_lean_file(self, lean_file: Path) -> Dict[str, Any]:
        """
        Verify a Lean 4 file.

        Returns:
            Dict with verification results

        Raises:
            subprocess.TimeoutExpired: If verification times out
        """
        try:
            result = subprocess.run(
                [self.lean_path, str(lean_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_ms / 1000,  # Convert ms to seconds
                cwd=self.workspace_dir,
            )

            if result.returncode == 0:
                return {
                    "status": "verified",
                    "errors": [],
                }
            else:
                # Parse errors from stderr
                errors = self._parse_lean_errors(result.stderr)
                return {
                    "status": "failed",
                    "errors": errors,
                }

        except subprocess.TimeoutExpired as e:
            raise Lean4TimeoutError(f"Lean 4 verification timed out: {e}")

    def _parse_lean_errors(self, stderr: str) -> List[Dict[str, Any]]:
        """Parse Lean 4 error messages from stderr."""
        errors = []
        for line in stderr.split("\n"):
            if "error:" in line.lower():
                errors.append({
                    "message": line.strip(),
                    "severity": "error",
                })
        return errors


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Lean4Interface",
    "Lean4Error",
    "Lean4TimeoutError",
    "Lean4VerificationError",
    "Lean4CircuitBreakerOpenError",
]
