"""
Lean 4 API Module for Mathematical Verification

Provides a REAL implementation of the Lean 4 verification API.
This module interfaces with the actual Lean 4 compiler.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import real implementation
try:
    from lean4_integration_enhanced import (
        Lean4VerificationEngine,
        Lean4AutoformalizationEngine,
        Lean4ServerConfig,
        LLMProvider,
        VerificationResult as RealVerificationResult,
        AutoformalizationResult as RealAutoformalizationResult
    )
    REAL_IMPLEMENTATION_AVAILABLE = True
except ImportError:
    REAL_IMPLEMENTATION_AVAILABLE = False


class VerificationStatus(Enum):
    """Verification status codes"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VerificationResult:
    """Result of a mathematical verification request"""
    request_id: str
    is_verified: bool
    status: str
    proof_status: Optional[str] = None
    lean4_output: str = ""
    error_message: str = ""
    tactics_used: List[str] = field(default_factory=list)
    proof_state: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "request_id": self.request_id,
            "is_verified": self.is_verified,
            "status": self.status,
            "proof_status": self.proof_status,
            "lean4_output": self.lean4_output,
            "error_message": self.error_message,
            "tactics_used": self.tactics_used,
            "proof_state": self.proof_state,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "details": self.details
        }


class MathematicalVerificationAPI:
    """
    API for mathematical verification using REAL Lean 4.
    
    This class provides a production-ready interface to the Lean 4 compiler
    with mathlib4 support.
    """

    def __init__(self, base_url: str = "http://localhost:7654", timeout: float = 30.0):
        """
        Initialize the Lean 4 verification API.

        Args:
            base_url: Base URL (kept for compatibility, not used for local Lean)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self._pending_requests: Dict[str, VerificationResult] = {}
        
        # Initialize real implementation
        if REAL_IMPLEMENTATION_AVAILABLE:
            config = Lean4ServerConfig(
                timeout_seconds=timeout
            )
            self._verification_engine = Lean4VerificationEngine(config)
            self._autoformalization_engine = Lean4AutoformalizationEngine(config)
            self._real_available = True
        else:
            self._verification_engine = None
            self._autoformalization_engine = None
            self._real_available = False

    async def submit_verification_request(
        self,
        code: str,
        theorem_name: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Submit a verification request to Lean 4.

        Args:
            code: Lean 4 code to verify
            theorem_name: Optional name of the theorem
            context: Optional context for the theorem

        Returns:
            Request ID for tracking
        """
        import uuid
        request_id = f"lean_req_{uuid.uuid4().hex[:8]}"

        if self._real_available:
            # Use real implementation
            try:
                result = await self._verification_engine.verify(code)
                
                # Convert to API format
                api_result = VerificationResult(
                    request_id=request_id,
                    is_verified=result.success,
                    status=VerificationStatus.COMPLETED.value,
                    proof_status="verified" if result.success else "failed",
                    lean4_output=result.output,
                    error_message="; ".join(result.errors) if result.errors else "",
                    tactics_used=result.warnings if hasattr(result, 'warnings') else [],
                    timestamp=result.timestamp if hasattr(result, 'timestamp') else datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                api_result = VerificationResult(
                    request_id=request_id,
                    is_verified=False,
                    status=VerificationStatus.FAILED.value,
                    error_message=str(e)
                )
        else:
            # Fallback - no real implementation
            api_result = VerificationResult(
                request_id=request_id,
                is_verified=False,
                status=VerificationStatus.FAILED.value,
                error_message="Lean 4 integration not available"
            )

        self._pending_requests[request_id] = api_result
        return request_id

    async def get_parallel_verification_results(
        self,
        request_ids: List[str]
    ) -> Dict[str, VerificationResult]:
        """
        Get results for multiple verification requests in parallel.

        Args:
            request_ids: List of request IDs to check

        Returns:
            Dictionary mapping request IDs to their results
        """
        return {rid: self._pending_requests.get(rid) for rid in request_ids if rid in self._pending_requests}

    async def get_verification_result(self, request_id: str) -> Optional[VerificationResult]:
        """
        Get the result of a single verification request.

        Args:
            request_id: Request ID to check

        Returns:
            Verification result or None if not found
        """
        return self._pending_requests.get(request_id)

    async def cancel_verification(self, request_id: str) -> bool:
        """
        Cancel a pending verification request.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if successfully cancelled
        """
        if request_id in self._pending_requests:
            result = self._pending_requests[request_id]
            result.status = VerificationStatus.CANCELLED.value
            return True
        return False

    async def autoformalize(
        self,
        natural_language: str,
        domain: str = "general",
        statement_type: str = "theorem"
    ) -> Dict[str, Any]:
        """
        Convert natural language to Lean 4 code.

        Args:
            natural_language: Natural language description
            domain: Mathematical domain hint
            statement_type: theorem, definition, or lemma

        Returns:
            Dictionary with formalization result
        """
        if not self._real_available:
            return {
                "success": False,
                "error": "Autoformalization not available - Lean 4 integration not initialized"
            }

        try:
            result = await self._autoformalization_engine.autoformalize(
                natural_language=natural_language,
                domain=domain,
                statement_type=statement_type
            )

            return {
                "success": result.success,
                "natural_language": result.natural_language,
                "lean_code": result.lean_code,
                "domain": result.domain,
                "confidence": result.confidence,
                "iterations": result.iterations,
                "errors": result.errors_encountered,
                "timestamp": result.timestamp
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the Lean 4 API service.

        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self._real_available else "degraded",
            "real_implementation": self._real_available,
            "base_url": self.base_url,
            "pending_requests": len(self._pending_requests),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Export the API class
__all__ = ['MathematicalVerificationAPI', 'VerificationResult', 'VerificationStatus']
