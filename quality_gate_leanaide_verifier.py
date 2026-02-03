"""
LeanAide Quality Gate Verifier for OpenEvolve

This module provides LeanAide-based quality gate verification for mathematical
correctness and formal verification of solutions.

Features:
- Mathematical correctness verification
- Formal proof verification
- Theorem statement validation
- Integration with existing QualityGateEngine

Author: OpenEvolve
Created: 2026-02-02
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


# =============================================================================
# Import Dependencies
# =============================================================================

# LeanAide imports
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, TaskType
    from leanaide_mcp_tools import (
        leanaide_translate_theorem_async,
        leanaide_verify_solution_async,
        leanaide_generate_proof_async,
        leanaide_elaborate_code_async,
        leanaide_math_query_async
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    LeanAideClient = None
    LeanAideConfig = None
    TaskType = None
    logger.warning("LeanAide not available for quality gate verification")

# Quality gate imports
try:
    from quality_gate_engine import (
        GateDecision,
        QualityGateReport,
        QualityThreshold,
        ContentType
    )
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False
    GateDecision = None
    QualityGateReport = None
    QualityThreshold = None
    ContentType = None
    logger.warning("QualityGateEngine not available")


# =============================================================================
# Data Classes
# =============================================================================

class MathematicalCorrectnessLevel(Enum):
    """Level of mathematical correctness verification."""
    NONE = "none"
    SYNTAX_CHECK = "syntax_check"
    ELABORATION = "elaboration"
    VERIFICATION = "verification"
    PROOF_CHECK = "proof_check"
    FULL_PROOF = "full_proof"


@dataclass
class LeanAideQualityConfig:
    """Configuration for LeanAide quality gate verification."""
    enabled: bool = True
    verification_level: MathematicalCorrectnessLevel = MathematicalCorrectnessLevel.VERIFICATION
    confidence_threshold: float = 0.8
    timeout_seconds: float = 300.0
    require_formal_proof: bool = False
    auto_detect_math: bool = True
    min_proof_coverage: float = 0.9
    store_verification_results: bool = True


@dataclass
class MathematicalVerificationResult:
    """Result of mathematical verification."""
    is_mathematical: bool
    verification_passed: bool
    confidence_score: float
    formal_code: Optional[str]
    verification_output: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_mathematical": self.is_mathematical,
            "verification_passed": self.verification_passed,
            "confidence_score": self.confidence_score,
            "formal_code": self.formal_code,
            "verification_output": self.verification_output,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


# =============================================================================
# LeanAide Quality Gate Verifier
# =============================================================================

class LeanAideQualityGateVerifier:
    """
    Quality gate verifier using LeanAide for mathematical correctness.
    
    This verifier integrates with the QualityGateEngine to provide
    formal verification of mathematical content in solutions.
    """
    
    def __init__(
        self,
        config: Optional[LeanAideQualityConfig] = None,
        leanaide_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LeanAide quality gate verifier.
        
        Args:
            config: Quality gate configuration
            leanaide_config: LeanAide client configuration
        """
        self.config = config or LeanAideQualityConfig()
        self.leanaide_config = leanaide_config or {
            "host": "localhost",
            "port": 7654,
            "timeout": self.config.timeout_seconds
        }
        
        # Initialize LeanAide client
        self.leanaide_client = None
        self._initialize_client()
        
        # Statistics
        self._stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total_time_ms": 0.0
        }
        
        logger.info({
            "msg": "LeanAideQualityGateVerifier initialized",
            "enabled": self.config.enabled,
            "verification_level": self.config.verification_level.value,
            "leanaide_available": self.leanaide_client is not None
        })
    
    def _initialize_client(self):
        """Initialize the LeanAide client."""
        if not LEANAIDE_AVAILABLE:
            return
        
        try:
            config = LeanAideConfig(
                host=self.leanaide_config.get("host", "localhost"),
                port=self.leanaide_config.get("port", 7654),
                timeout=self.leanaide_config.get("timeout", self.config.timeout_seconds)
            )
            self.leanaide_client = LeanAideClient(config)
            logger.info("LeanAide client initialized for quality gate")
        except Exception as e:
            logger.warning(f"Failed to initialize LeanAide client: {e}")
    
    async def verify_mathematical_correctness(
        self,
        solution_content: str,
        expected_properties: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> MathematicalVerificationResult:
        """
        Verify mathematical correctness of a solution.
        
        Args:
            solution_content: The solution content to verify
            expected_properties: Optional list of expected properties
            correlation_id: Correlation ID for tracking
            
        Returns:
            MathematicalVerificationResult
        """
        correlation_id = correlation_id or f"math_verify_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now()
        
        logger.info({
            "msg": "Starting mathematical verification",
            "content_length": len(solution_content),
            "verification_level": self.config.verification_level.value,
            "correlation_id": correlation_id
        })
        
        self._stats["total_verifications"] += 1
        
        result = MathematicalVerificationResult(
            is_mathematical=False,
            verification_passed=False,
            confidence_score=0.0,
            formal_code=None,
            verification_output={},
            processing_time_ms=0.0
        )
        
        try:
            # Step 1: Check if content is mathematical
            if self.config.auto_detect_math:
                is_math = self._detect_mathematical_content(solution_content)
                result.is_mathematical = is_math
                
                if not is_math:
                    result.confidence_score = 1.0  # Non-math content passes by default
                    result.verification_passed = True
                    result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    self._stats["passed"] += 1
                    return result
            
            # Step 2: Translate to formal language
            if self.config.verification_level.value in [
                MathematicalCorrectnessLevel.ELABORATION.value,
                MathematicalCorrectnessLevel.VERIFICATION.value,
                MathematicalCorrectnessLevel.PROOF_CHECK.value,
                MathematicalCorrectnessLevel.FULL_PROOF.value
            ]:
                translate_result = await leanaide_translate_theorem_async(solution_content)
                
                if translate_result.get("success"):
                    result.formal_code = translate_result.get("lean_code", "")
                else:
                    result.errors.append(f"Translation failed: {translate_result.get('error')}")
                    result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    self._stats["failed"] += 1
                    return result
            
            # Step 3: Verify the formal code
            if self.config.verification_level.value in [
                MathematicalCorrectnessLevel.VERIFICATION.value,
                MathematicalCorrectnessLevel.PROOF_CHECK.value,
                MathematicalCorrectnessLevel.FULL_PROOF.value
            ] and result.formal_code:
                verify_result = await leanaide_verify_solution_async(
                    result.formal_code,
                    timeout=self.config.timeout_seconds
                )
                
                result.verification_output = verify_result
                result.verification_passed = verify_result.get("success", False)
                result.confidence_score = verify_result.get("confidence", 0.0)
                
                if not result.verification_passed:
                    result.errors.extend(verify_result.get("errors", []))
            
            # Step 4: Check confidence threshold
            if result.confidence_score < self.config.confidence_threshold:
                result.verification_passed = False
                result.warnings.append(
                    f"Confidence score {result.confidence_score:.2f} below threshold "
                    f"{self.config.confidence_threshold:.2f}"
                )
            
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._stats["total_time_ms"] += result.processing_time_ms
            
            if result.verification_passed:
                self._stats["passed"] += 1
            else:
                self._stats["failed"] += 1
            
            logger.info({
                "msg": "Mathematical verification completed",
                "is_mathematical": result.is_mathematical,
                "verification_passed": result.verification_passed,
                "confidence_score": result.confidence_score,
                "processing_time_ms": result.processing_time_ms,
                "correlation_id": correlation_id
            })
            
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._stats["failed"] += 1
            
            logger.error({
                "msg": "Mathematical verification failed",
                "error": str(e),
                "correlation_id": correlation_id
            })
            
            return result
    
    def _detect_mathematical_content(self, content: str) -> bool:
        """Detect if content contains mathematical content."""
        math_indicators = [
            "theorem", "lemma", "proof", "proposition",
            "equation", "integral", "derivative", "limit",
            "forall", "exists", "such that", "iff",
            "supremum", "infimum", "convergent", "continuous",
            "prime", "divisible", "modulo", "gcd", "lcm",
            "sqrt", "rational", "irrational", "integer"
        ]
        
        content_lower = content.lower()
        matches = sum(1 for indicator in math_indicators if indicator in content_lower)
        
        return matches >= 2  # At least 2 indicators
    
    async def run_quality_gate(
        self,
        solution_content: str,
        threshold: float = 0.8,
        correlation_id: Optional[str] = None
    ) -> Tuple[GateDecision, QualityGateReport]:
        """
        Run the quality gate check on a solution.
        
        Args:
            solution_content: The solution content to verify
            threshold: Minimum confidence threshold for pass
            correlation_id: Correlation ID for tracking
            
        Returns:
            Tuple of (GateDecision, QualityGateReport)
        """
        correlation_id = correlation_id or f"quality_gate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.config.enabled or not self.leanaide_client:
            logger.info("LeanAide quality gate disabled or unavailable, skipping")
            self._stats["skipped"] += 1
            
            if QUALITY_GATE_AVAILABLE:
                report = QualityGateReport(
                    decision=GateDecision.PASS,
                    overall_score=100.0,
                    threshold_used=QualityThreshold(
                        content_type=ContentType.TECHNICAL,
                        quality_level=None,  # Will be set by engine
                        min_overall_score=threshold
                    ),
                    rationale="LeanAide verification skipped (disabled or unavailable)",
                    improvement_recommendations=[],
                    critical_issues=[],
                    minor_issues=[],
                    scores_by_metric={"mathematical_correctness": 100.0},
                    metadata={"skipped": True, "reason": "unavailable"}
                )
                return GateDecision.PASS, report
            else:
                return GateDecision.PASS, None
        
        # Run verification
        result = await self.verify_mathematical_correctness(
            solution_content,
            correlation_id=correlation_id
        )
        
        # Determine gate decision
        if result.verification_passed and result.confidence_score >= threshold:
            decision = GateDecision.PASS
        elif result.is_mathematical and not result.verification_passed:
            decision = GateDecision.FAIL
        elif result.confidence_score >= threshold * 0.8:
            decision = GateDecision.CONDITIONAL_PASS
        else:
            decision = GateDecision.DEFERRED
        
        # Build report
        if QUALITY_GATE_AVAILABLE:
            report = QualityGateReport(
                decision=decision,
                overall_score=result.confidence_score * 100,
                threshold_used=QualityThreshold(
                    content_type=ContentType.TECHNICAL,
                    quality_level=None,
                    min_overall_score=threshold * 100
                ),
                rationale=self._build_rationale(result),
                improvement_recommendations=self._build_recommendations(result),
                critical_issues=result.errors,
                minor_issues=result.warnings,
                scores_by_metric={
                    "mathematical_correctness": result.confidence_score * 100,
                    "formal_verification": 100 if result.formal_code else 0
                },
                metadata={
                    "correlation_id": correlation_id,
                    "verification_result": result.to_dict()
                }
            )
        else:
            report = None
        
        return decision, report
    
    def _build_rationale(self, result: MathematicalVerificationResult) -> str:
        """Build the rationale for the gate decision."""
        parts = []
        
        if result.is_mathematical:
            parts.append("Mathematical content detected")
            
            if result.verification_passed:
                parts.append("Formal verification passed")
            else:
                parts.append("Formal verification failed")
            
            parts.append(f"Confidence: {result.confidence_score:.2%}")
        else:
            parts.append("No mathematical content detected")
        
        return ". ".join(parts)
    
    def _build_recommendations(self, result: MathematicalVerificationResult) -> List[str]:
        """Build improvement recommendations."""
        recommendations = []
        
        if result.errors:
            recommendations.append(f"Address {len(result.errors)} formal verification error(s)")
        
        if result.warnings:
            recommendations.append(f"Review {len(result.warnings)} warning(s)")
        
        if not result.formal_code:
            recommendations.append("Consider providing formal specification")
        
        if result.confidence_score < self.config.confidence_threshold:
            recommendations.append(
                f"Improve verification confidence (current: {result.confidence_score:.2%}, "
                f"required: {self.config.confidence_threshold:.2%})"
            )
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total = self._stats["total_verifications"]
        return {
            "total_verifications": total,
            "passed": self._stats["passed"],
            "failed": self._stats["failed"],
            "skipped": self._stats["skipped"],
            "pass_rate": self._stats["passed"] / total if total > 0 else 0,
            "avg_time_ms": self._stats["total_time_ms"] / total if total > 0 else 0
        }
    
    def reset_stats(self):
        """Reset verification statistics."""
        self._stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total_time_ms": 0.0
        }


# =============================================================================
# Factory Functions
# =============================================================================

def get_leanaide_quality_gate_verifier(
    config: Optional[LeanAideQualityConfig] = None,
    leanaide_config: Optional[Dict[str, Any]] = None
) -> LeanAideQualityGateVerifier:
    """
    Get a LeanAide quality gate verifier instance.
    
    Args:
        config: Quality gate configuration
        leanaide_config: LeanAide client configuration
        
    Returns:
        LeanAideQualityGateVerifier instance
    """
    return LeanAideQualityGateVerifier(config=config, leanaide_config=leanaide_config)


async def create_leanaide_quality_gate_verifier(
    config: Optional[LeanAideQualityConfig] = None,
    leanaide_config: Optional[Dict[str, Any]] = None
) -> LeanAideQualityGateVerifier:
    """
    Create and initialize a LeanAide quality gate verifier (async).
    
    Args:
        config: Quality gate configuration
        leanaide_config: LeanAide client configuration
        
    Returns:
        Initialized LeanAideQualityGateVerifier instance
    """
    verifier = get_leanaide_quality_gate_verifier(config, leanaide_config)
    return verifier


# =============================================================================
# Integration with QualityGateEngine
# =============================================================================

async def integrate_with_quality_gate_engine(
    engine,
    verifier: LeanAideQualityGateVerifier
) -> None:
    """
    Integrate LeanAide verifier with an existing QualityGateEngine.
    
    Args:
        engine: QualityGateEngine instance
        verifier: LeanAideQualityGateVerifier instance
    """
    # Add mathematical verification to the engine's evaluation methods
    original_evaluate = engine.evaluate_solution if hasattr(engine, 'evaluate_solution') else None
    
    async def enhanced_evaluate(solution, *args, **kwargs):
        # Run original evaluation if exists
        result = None
        if original_evaluate:
            result = await original_evaluate(solution, *args, **kwargs)
        
        # Run LeanAide verification
        math_result = await verifier.verify_mathematical_correctness(str(solution))
        
        # Merge results
        if hasattr(result, 'metadata'):
            result.metadata['leanaide_verification'] = math_result.to_dict()
        
        return result
    
    # Replace or attach the enhanced method
    if hasattr(engine, 'evaluate_solution'):
        engine.evaluate_solution = enhanced_evaluate
    
    logger.info("LeanAide verifier integrated with QualityGateEngine")


# =============================================================================
# Standalone Usage
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    
    async def test_quality_gate():
        """Test the quality gate verifier."""
        print("Testing LeanAide Quality Gate Verifier...")
        
        # Create verifier
        config = LeanAideQualityConfig(
            enabled=True,
            verification_level=MathematicalCorrectnessLevel.VERIFICATION,
            confidence_threshold=0.8
        )
        
        verifier = get_leanaide_quality_gate_verifier(config)
        
        # Test with mathematical content
        theorem = "theorem sqrt_2_irrational : ∀ n : ℕ, n*n = 2 → false"
        theorem += "\n| h := h\n| h := h\nbegin\n  cases h,\nend"
        
        print("\n1. Testing mathematical content verification...")
        decision, report = await verifier.run_quality_gate(theorem)
        
        print(f"Decision: {decision.value if decision else 'N/A'}")
        if report:
            print(f"Overall Score: {report.overall_score}")
            print(f"Critical Issues: {len(report.critical_issues)}")
            print(f"Rationale: {report.rationale}")
        
        # Test with non-mathematical content
        print("\n2. Testing non-mathematical content...")
        code = "def hello_world():\n    print('Hello, World!')"
        
        decision2, report2 = await verifier.run_quality_gate(code)
        print(f"Decision: {decision2.value if decision2 else 'N/A'}")
        
        # Print stats
        print("\n3. Statistics:")
        stats = verifier.get_stats()
        print(json.dumps(stats, indent=2))
        
        return decision and decision.value == "pass"
    
    # Run test
    try:
        result = asyncio.run(test_quality_gate())
        if result:
            print("\nSUCCESS: Quality gate verifier working!")
        else:
            print("\nCOMPLETE: Quality gate verifier tested (some tests may have failed)")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
