"""
LeanAide Workflow Integration Module

This module provides integration between LeanAide formal verification system
and the OpenEvolve workflow stages, enabling formal mathematical verification
for mathematical problems within the decomposition workflow.

Key Features:
- Seamless integration with Stage 3C (Gold Team Gauntlet) for sub-problem verification
- Integration with Stage 5 (Final Verification) for final solution verification
- Automatic mathematical problem detection
- LeanAide formal verification with confidence scoring
- Graceful fallback for non-mathematical problems
- Comprehensive error handling and logging

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json
import uuid

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult, TaskType
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    LeanAideClient = None
    LeanAideConfig = None
    LeanAideResult = None

# Configure logging
logger = logging.getLogger(__name__)


class VerificationMethod(Enum):
    """Available verification methods."""
    LEANAIDE_FORMAL = "leanaide_formal"
    STANDARD_GAUNTLET = "standard_gauntlet"
    HYBRID = "hybrid"


@dataclass
class LeanAideVerificationResult:
    """Result from LeanAide formal verification."""
    success: bool
    is_mathematical: bool
    confidence_score: float
    verification_method: str
    lean_code: Optional[str] = None
    formal_proof: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "is_mathematical": self.is_mathematical,
            "confidence_score": self.confidence_score,
            "verification_method": self.verification_method,
            "lean_code": self.lean_code,
            "formal_proof": self.formal_proof,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "execution_time": self.execution_time
        }


@dataclass
class LeanAideWorkflowConfig:
    """Configuration for LeanAide workflow integration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 7654
    timeout: float = 300.0
    max_retries: int = 2
    auto_detect_math: bool = True
    fallback_to_standard: bool = True
    confidence_threshold: float = 0.7
    require_formal_proof: bool = False
    store_proofs: bool = True
    use_subprocess: bool = True
    lean_binary: str = "lean"
    lake_binary: str = "lake"
    project_dir: Optional[str] = None
    lean_timeout: float = 120.0

    # Mathematical detection patterns
    math_keywords: List[str] = field(default_factory=lambda: [
        "prove", "theorem", "lemma", "proof", "mathematical",
        "equation", "inequality", "function", "integral", "derivative",
        "limit", "series", "convergence", "divergence", "continuity",
        "differentiability", "optimization", "algorithm", "complexity",
        "graph theory", "number theory", "combinatorics", "probability",
        "statistics", "linear algebra", "calculus", "geometry", "topology"
    ])

    math_patterns: List[str] = field(default_factory=lambda: [
        r'\b(prove|show|demonstrate)\s+(that|the)?\s*\w+',
        r'\b(theorem|lemma|corollary|proposition)\b',
        r'\b(equals?|≈|≤|≥|<|>|≠|∈|∉|⊂|⊃)\b',
        r'\b(for all|∀|exists?|∃)\b',
        r'\b(function|mapping|transformation)\b',
        r'\b(optimization|minimize|maximize)\b',
        r'\b(complexity|O\(n\)|O\(log n\)|O\(n²\))\b'
    ])


class MathematicalProblemDetector:
    """Detects whether a problem or solution is mathematical in nature."""

    def __init__(self, config: LeanAideWorkflowConfig):
        self.config = config
        self.keyword_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in config.math_keywords) + r')\b',
            re.IGNORECASE
        )
        self.patterns = [re.compile(p, re.IGNORECASE) for p in config.math_patterns]

    def is_mathematical_problem(
        self,
        problem_statement: str,
        solution_content: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Detect if the problem is mathematical.

        Returns:
            Tuple of (is_mathematical, confidence_score)
        """
        text = problem_statement.lower()
        if solution_content:
            text += " " + solution_content.lower()

        # Check for mathematical keywords
        keyword_matches = len(self.keyword_pattern.findall(text))
        keyword_score = min(keyword_matches / 3.0, 1.0)  # Cap at 1.0

        # Check for mathematical patterns
        pattern_matches = sum(1 for p in self.patterns if p.search(text))
        pattern_score = min(pattern_matches / 2.0, 1.0)  # Cap at 1.0

        # Check for mathematical symbols and notation
        symbol_score = 0.0
        if any(c in text for c in ['∑', '∫', '∂', '√', '∞', '±', '×', '÷']):
            symbol_score = 0.3

        # Check for code-like mathematical expressions
        math_expr_score = 0.0
        if re.search(r'\$[^$]+\$', text):  # LaTeX math
            math_expr_score = 0.4
        if re.search(r'\\[a-zA-Z]+\{', text):  # LaTeX commands
            math_expr_score = max(math_expr_score, 0.3)

        # Combine scores
        overall_confidence = (
            keyword_score * 0.4 +
            pattern_score * 0.3 +
            symbol_score * 0.1 +
            math_expr_score * 0.2
        )

        is_math = overall_confidence >= 0.3
        return is_math, overall_confidence


class LeanAideWorkflowIntegrator:
    """
    Main integration class for LeanAide verification in OpenEvolve workflows.
    """

    def __init__(self, config: Optional[LeanAideWorkflowConfig] = None):
        """
        Initialize the LeanAide workflow integrator.

        Args:
            config: Configuration for LeanAide integration
        """
        self.config = config or LeanAideWorkflowConfig()
        self.detector = MathematicalProblemDetector(self.config)
        self.client: Optional[LeanAideClient] = None
        self._lean_subprocess_available = False
        self._lean_subprocess_command: Optional[List[str]] = None

        if not LEANAIDE_AVAILABLE and not self.config.use_subprocess:
            logger.warning("LeanAide client not available. Formal verification will be disabled.")
            self.config.enabled = False
        elif not LEANAIDE_AVAILABLE:
            logger.warning("LeanAide client not available. Falling back to Lean subprocess if configured.")

    async def initialize(self) -> bool:
        """
        Initialize LeanAide client connection.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.config.enabled:
            return False

        ready = False

        if LEANAIDE_AVAILABLE:
            try:
                client_config = LeanAideConfig(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
                self.client = LeanAideClient(config=client_config)

                # Health check
                is_healthy = await self.client.health_check()
                if not is_healthy:
                    logger.warning(f"LeanAide server at {self.config.host}:{self.config.port} is not responding")
                    self.client = None
                else:
                    logger.info(f"LeanAide client initialized successfully at {self.config.host}:{self.config.port}")
                    ready = True

            except Exception as e:
                logger.error(f"Failed to initialize LeanAide client: {e}")
                self.client = None

        if self.config.use_subprocess:
            self._lean_subprocess_command = self._detect_lean_subprocess_command()
            if self._lean_subprocess_command:
                self._lean_subprocess_available = True
                ready = True
                logger.info("Lean subprocess available via: %s", " ".join(self._lean_subprocess_command))
            else:
                logger.warning("Lean subprocess not available (missing lake/lean binary)")

        return ready

    def _detect_lean_subprocess_command(self) -> Optional[List[str]]:
        """Detect a usable Lean subprocess command."""
        if not self.config.use_subprocess:
            return None

        lake_path = shutil.which(self.config.lake_binary) if self.config.lake_binary else None
        if lake_path:
            return [lake_path, "env", "lean"]

        lean_path = shutil.which(self.config.lean_binary) if self.config.lean_binary else None
        if lean_path:
            return [lean_path]

        return None

    @staticmethod
    def _looks_like_lean(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(token in lowered for token in ("theorem", "lemma", "def ", "import ", "example", "axiom", "structure"))

    async def _run_lean_subprocess(self, lean_code: str) -> Tuple[bool, str, str]:
        """Run Lean/Lake subprocess to elaborate Lean code."""
        if not self._lean_subprocess_available or not self._lean_subprocess_command:
            raise RuntimeError("Lean subprocess not available")

        timeout = self.config.lean_timeout or self.config.timeout
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False, encoding="utf-8") as tmp_file:
                tmp_file.write(lean_code)
                tmp_path = tmp_file.name

            proc = await asyncio.create_subprocess_exec(
                *self._lean_subprocess_command,
                tmp_path,
                cwd=self.config.project_dir or None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                return False, stdout.decode("utf-8", errors="replace"), "Lean subprocess timed out"

            return (
                proc.returncode == 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace")
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def verify_sub_problem_solution(
        self,
        sub_problem_id: str,
        problem_statement: str,
        solution_content: str,
        verification_requirements: Optional[Dict[str, Any]] = None
    ) -> LeanAideVerificationResult:
        """
        Verify a sub-problem solution using LeanAide formal verification.
        Intended for use in Stage 3C (Gold Team Gauntlet).

        Args:
            sub_problem_id: ID of the sub-problem
            problem_statement: Original problem statement
            solution_content: Solution to verify
            verification_requirements: Optional verification requirements

        Returns:
            LeanAideVerificationResult with verification outcome
        """
        start_time = time.time()

        # Detect if this is a mathematical problem
        is_math, math_confidence = self.detector.is_mathematical_problem(
            problem_statement,
            solution_content
        )

        if not is_math:
            logger.info(f"Sub-problem {sub_problem_id} is not mathematical (confidence: {math_confidence:.2f})")
            return LeanAideVerificationResult(
                success=True,  # Not a failure, just not applicable
                is_mathematical=False,
                confidence_score=math_confidence,
                verification_method="none",
                metadata={"reason": "Non-mathematical problem"}
            )

        # Attempt formal verification
        try:
            result = await self._perform_formal_verification(
                problem_statement,
                solution_content,
                verification_requirements
            )

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            return result

        except Exception as e:
            logger.error(f"LeanAide verification failed for {sub_problem_id}: {e}")

            # Fallback to standard verification if enabled
            if self.config.fallback_to_standard:
                return LeanAideVerificationResult(
                    success=False,
                    is_mathematical=True,
                    confidence_score=0.0,
                    verification_method="standard_fallback",
                    errors=[f"LeanAide verification failed: {str(e)}"],
                    execution_time=time.time() - start_time
                )

            return LeanAideVerificationResult(
                success=False,
                is_mathematical=True,
                confidence_score=0.0,
                verification_method="leanaide_failed",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    async def verify_final_solution(
        self,
        problem_statement: str,
        final_solution: str,
        sub_problems: List[Dict[str, Any]],
        verification_requirements: Optional[Dict[str, Any]] = None
    ) -> LeanAideVerificationResult:
        """
        Verify the final integrated solution using LeanAide formal verification.
        Intended for use in Stage 5 (Final Verification).

        Args:
            problem_statement: Original problem statement
            final_solution: Final integrated solution
            sub_problems: List of sub-problems with their solutions
            verification_requirements: Optional verification requirements

        Returns:
            LeanAideVerificationResult with verification outcome
        """
        start_time = time.time()

        # Check if overall problem is mathematical
        is_math, math_confidence = self.detector.is_mathematical_problem(
            problem_statement,
            final_solution
        )

        if not is_math:
            logger.info("Final solution is not mathematical (confidence: {:.2f})".format(math_confidence))
            return LeanAideVerificationResult(
                success=True,
                is_mathematical=False,
                confidence_score=math_confidence,
                verification_method="none",
                metadata={"reason": "Non-mathematical problem"}
            )

        # Check if any sub-problems are mathematical
        math_sub_problems = []
        for sp in sub_problems:
            sp_is_math, sp_confidence = self.detector.is_mathematical_problem(
                sp.get("description", ""),
                sp.get("solution", "")
            )
            if sp_is_math:
                math_sub_problems.append({
                    "id": sp.get("id"),
                    "confidence": sp_confidence
                })

        # Attempt formal verification of the complete solution
        try:
            result = await self._perform_formal_verification(
                problem_statement,
                final_solution,
                verification_requirements
            )

            # Add metadata about mathematical sub-problems
            result.metadata["mathematical_sub_problems"] = math_sub_problems
            result.metadata["total_sub_problems"] = len(sub_problems)

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            return result

        except Exception as e:
            logger.error(f"LeanAide final verification failed: {e}")

            if self.config.fallback_to_standard:
                return LeanAideVerificationResult(
                    success=False,
                    is_mathematical=True,
                    confidence_score=0.0,
                    verification_method="standard_fallback",
                    errors=[f"LeanAide verification failed: {str(e)}"],
                    metadata={
                        "mathematical_sub_problems": math_sub_problems,
                        "total_sub_problems": len(sub_problems)
                    },
                    execution_time=time.time() - start_time
                )

            return LeanAideVerificationResult(
                success=False,
                is_mathematical=True,
                confidence_score=0.0,
                verification_method="leanaide_failed",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    async def _perform_formal_verification(
        self,
        problem_statement: str,
        solution_content: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> LeanAideVerificationResult:
        """
        Perform the actual formal verification using LeanAide.

        Args:
            problem_statement: Problem to verify
            solution_content: Proposed solution
            requirements: Optional verification requirements

        Returns:
            LeanAideVerificationResult with detailed verification outcome
        """
        translation_result = None
        lean_code = ""

        if self.client:
            # Step 1: Translate the problem/solution to Lean
            translation_result = await self.client.translate_thm_detailed(
                theorem_text=problem_statement + "\n\n" + solution_content,
                theorem_name=f"theorem_{uuid.uuid4()}"
            )

            if not translation_result.success:
                return LeanAideVerificationResult(
                    success=False,
                    is_mathematical=True,
                    confidence_score=0.0,
                    verification_method="leanaide_formal",
                    errors=["Translation failed: " + (translation_result.error or "Unknown error")]
                )

            lean_code = translation_result.data.get("result", "") if translation_result.data else ""

        elif self._lean_subprocess_available:
            # Use existing Lean code if provided
            if self._looks_like_lean(solution_content):
                lean_code = solution_content
            elif self._looks_like_lean(problem_statement):
                lean_code = problem_statement
            else:
                return LeanAideVerificationResult(
                    success=False,
                    is_mathematical=True,
                    confidence_score=0.0,
                    verification_method="leanaide_formal",
                    errors=["No Lean code available for subprocess verification"]
                )
        else:
            raise RuntimeError("LeanAide client not initialized and Lean subprocess unavailable")

        # Step 2: Generate a proof (if required)
        formal_proof = None
        if self.config.require_formal_proof and self.client and translation_result:
            theorem_code = translation_result.data.get("type", "") if translation_result.data else ""

            proof_result = await self.client.prove_for_formalization(
                theorem_text=problem_statement,
                theorem_code=lean_code,
                theorem_statement=theorem_code
            )

            if proof_result.success:
                formal_proof = proof_result.data.get("result", "") if proof_result.data else None

        # Step 3: Elaborate the Lean code to check for errors
        errors = []
        warnings = []
        unsolved_goals: List[str] = []
        elaboration_backend = "leanaide_api"
        subprocess_stdout = ""
        subprocess_stderr = ""

        if self.config.use_subprocess and self._lean_subprocess_available:
            elaboration_backend = "lean_subprocess"
            success, stdout, stderr = await self._run_lean_subprocess(lean_code)
            subprocess_stdout = stdout
            subprocess_stderr = stderr
            logs = "\n".join([line for line in (stdout, stderr) if line])

            if not success:
                errors.append("Lean subprocess elaboration failed")

            if "error" in logs.lower():
                errors.append("Elaboration contained errors")

            confidence_score = 0.9 if success else 0.0
        else:
            if not self.client:
                raise RuntimeError("LeanAide client not initialized")

            elaboration_result = await self.client.elaborate(document_code=lean_code)
            success = elaboration_result.success

            if elaboration_result.success:
                elaboration_data = elaboration_result.data or {}
                unsolved_goals = elaboration_data.get("unsolved_goals", [])
                logs = elaboration_data.get("logs", "")

                if unsolved_goals:
                    warnings.append(f"{len(unsolved_goals)} unsolved goals remain")

                if "error" in logs.lower():
                    success = False
                    errors.append("Elaboration contained errors")

                confidence_score = 0.9 if not unsolved_goals else max(0.5, 0.9 - len(unsolved_goals) * 0.1)

            else:
                errors.append("Elaboration failed: " + (elaboration_result.error or "Unknown error"))
                confidence_score = 0.0

        return LeanAideVerificationResult(
            success=success and confidence_score >= self.config.confidence_threshold,
            is_mathematical=True,
            confidence_score=confidence_score,
            verification_method="leanaide_formal",
            lean_code=lean_code if self.config.store_proofs else None,
            formal_proof=formal_proof if self.config.store_proofs else None,
            errors=errors,
            warnings=warnings,
            metadata={
                "translation_success": translation_result.success if translation_result else False,
                "elaboration_success": success,
                "unsolved_goals": unsolved_goals,
                "elaboration_backend": elaboration_backend,
                "subprocess_stdout": subprocess_stdout[:1000] if subprocess_stdout else "",
                "subprocess_stderr": subprocess_stderr[:1000] if subprocess_stderr else ""
            }
        )

    async def batch_verify_sub_problems(
        self,
        sub_problems: List[Dict[str, Any]]
    ) -> Dict[str, LeanAideVerificationResult]:
        """
        Verify multiple sub-problems in parallel.

        Args:
            sub_problems: List of sub-problems with their solutions

        Returns:
            Dictionary mapping sub-problem IDs to verification results
        """
        if not self.client and not self._lean_subprocess_available:
            logger.warning("LeanAide client not initialized and Lean subprocess unavailable")
            return {}

        verification_tasks = []
        for sp in sub_problems:
            task = self.verify_sub_problem_solution(
                sub_problem_id=sp.get("id", ""),
                problem_statement=sp.get("description", ""),
                solution_content=sp.get("solution", ""),
                verification_requirements=sp.get("verification_requirements")
            )
            verification_tasks.append(task)

        results = await asyncio.gather(*verification_tasks, return_exceptions=True)

        result_dict = {}
        for sp, result in zip(sub_problems, results):
            sp_id = sp.get("id", "")
            if isinstance(result, Exception):
                logger.error(f"Exception verifying {sp_id}: {result}")
                result_dict[sp_id] = LeanAideVerificationResult(
                    success=False,
                    is_mathematical=True,
                    confidence_score=0.0,
                    verification_method="error",
                    errors=[str(result)]
                )
            else:
                result_dict[sp_id] = result

        return result_dict

    async def close(self):
        """Close the LeanAide client connection."""
        if self.client:
            await self.client.close()
            self.client = None


# =============================================================================
# Convenience Functions for Workflow Integration
# =============================================================================

async def verify_with_leanaide(
    problem_statement: str,
    solution_content: str,
    config: Optional[LeanAideWorkflowConfig] = None
) -> LeanAideVerificationResult:
    """
    Convenience function to verify a solution with LeanAide.

    Args:
        problem_statement: The problem statement
        solution_content: The solution to verify
        config: Optional configuration

    Returns:
        LeanAideVerificationResult
    """
    integrator = LeanAideWorkflowIntegrator(config)
    try:
        initialized = await integrator.initialize()
        if not initialized:
            return LeanAideVerificationResult(
                success=False,
                is_mathematical=False,
                confidence_score=0.0,
                verification_method="unavailable",
                errors=["LeanAide server not available"]
            )

        return await integrator.verify_sub_problem_solution(
            sub_problem_id="adhoc",
            problem_statement=problem_statement,
            solution_content=solution_content
        )
    finally:
        await integrator.close()


def is_leanaide_configured() -> bool:
    """Check if LeanAide is available and configured."""
    if LEANAIDE_AVAILABLE:
        return True
    return bool(shutil.which("lean") or shutil.which("lake"))


def create_standard_leanaide_config(
    host: str = "localhost",
    port: int = 7654,
    enable_auto_detect: bool = True,
    confidence_threshold: float = 0.7
) -> LeanAideWorkflowConfig:
    """
    Create a standard LeanAide workflow configuration.

    Args:
        host: LeanAide server host
        port: LeanAide server port
        enable_auto_detect: Enable automatic mathematical problem detection
        confidence_threshold: Minimum confidence threshold for verification success

    Returns:
        LeanAideWorkflowConfig instance
    """
    return LeanAideWorkflowConfig(
        enabled=True,
        host=host,
        port=port,
        auto_detect_math=enable_auto_detect,
        fallback_to_standard=True,
        confidence_threshold=confidence_threshold,
        require_formal_proof=False,
        store_proofs=True
    )


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example demonstrating LeanAide workflow integration."""

    # Create configuration
    config = create_standard_leanaide_config(
        host="localhost",
        port=7654,
        confidence_threshold=0.7
    )

    # Create integrator
    integrator = LeanAideWorkflowIntegrator(config)

    try:
        # Initialize connection
        if not await integrator.initialize():
            print("Failed to connect to LeanAide server")
            return

        # Example 1: Verify a mathematical sub-problem solution
        print("\n=== Example 1: Verify Sub-Problem Solution ===")
        result = await integrator.verify_sub_problem_solution(
            sub_problem_id="sp_001",
            problem_statement="Prove that the square root of 2 is irrational",
            solution_content="Assume for contradiction that √2 is rational and can be expressed as a/b..."
        )
        print(f"Success: {result.success}")
        print(f"Is Mathematical: {result.is_mathematical}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Method: {result.verification_method}")

        # Example 2: Verify a final solution
        print("\n=== Example 2: Verify Final Solution ===")
        result = await integrator.verify_final_solution(
            problem_statement="Develop an efficient algorithm for finding prime numbers",
            final_solution="We implement the Sieve of Eratosthenes...",
            sub_problems=[
                {
                    "id": "sp_001",
                    "description": "Design the sieve algorithm",
                    "solution": "Use a boolean array to mark composites..."
                },
                {
                    "id": "sp_002",
                    "description": "Optimize memory usage",
                    "solution": "Use bit manipulation for compact storage..."
                }
            ]
        )
        print(f"Success: {result.success}")
        print(f"Is Mathematical: {result.is_mathematical}")
        print(f"Confidence: {result.confidence_score:.2f}")

        # Example 3: Batch verify multiple sub-problems
        print("\n=== Example 3: Batch Verify Sub-Problems ===")
        sub_problems = [
            {
                "id": "sp_001",
                "description": "Prove the intermediate value theorem",
                "solution": "Let f be continuous on [a,b]..."
            },
            {
                "id": "sp_002",
                "description": "Design a user interface",
                "solution": "Create a responsive web interface using React..."
            }
        ]
        results = await integrator.batch_verify_sub_problems(sub_problems)
        for sp_id, result in results.items():
            print(f"{sp_id}: success={result.success}, is_math={result.is_mathematical}, confidence={result.confidence_score:.2f}")

    finally:
        await integrator.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
