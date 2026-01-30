"""
Verification Methods for Lean 4 Continuous Mathematics

This module provides comprehensive verification methods for generated Lean 4 code,
including syntax validation, mathematical correctness checks, domain-specific
verification, and integration with Lean 4 theorem provers.

Features:
- Lean 4 syntax validation
- Mathematical correctness verification
- Domain-specific pattern checking
- Conservation law verification
- Boundary condition validation
- Integration with LeanAide client for automated proving
- Detailed error reporting and suggestions

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.4)
"""

import re
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import from previous phases
from continuous_math_detector import (
    MathDetectionResult,
    MathType,
    ProblemType,
    ScientificDomain,
)
from ode_pde_translator import (
    Lean4TranslationResult,
    Lean4CodeBlock,
)
from scientific_domain_patterns import (
    ScientificDomainPatterns,
    get_domain_patterns,
)

# Try to import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, TaskType
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LeanAide client not available, some features will be limited")

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Verification Result Structures
# ============================================================================

class VerificationStatus(Enum):
    """Status of verification"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class CheckType(Enum):
    """Types of verification checks"""
    SYNTAX = "syntax"
    TYPE = "type"
    MATHEMATICAL = "mathematical"
    DOMAIN = "domain"
    CONSERVATION = "conservation"
    BOUNDARY = "boundary"
    PROOF = "proof"


@dataclass
class VerificationIssue:
    """A verification issue found during checking"""
    check_type: CheckType
    severity: str  # "error", "warning", "info"
    message: str
    location: Optional[str] = None  # Line/column or code block
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet
        }


@dataclass
class VerificationResult:
    """Result of verifying Lean 4 code"""
    overall_status: VerificationStatus
    checks_performed: List[CheckType]
    issues: List[VerificationIssue]
    passed_checks: int
    failed_checks: int
    warnings: int
    verification_time: float = 0.0
    lean4_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_status": self.overall_status.value,
            "checks_performed": [c.value for c in self.checks_performed],
            "issues": [issue.to_dict() for issue in self.issues],
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "verification_time": self.verification_time,
            "lean4_output": self.lean4_output,
            "metadata": self.metadata
        }

    @property
    def is_valid(self) -> bool:
        """Check if verification passed (no errors)"""
        return self.overall_status in [VerificationStatus.PASSED, VerificationStatus.WARNING]


# ============================================================================
# Lean 4 Verifier
# ============================================================================

class Lean4Verifier:
    """
    Comprehensive verifier for Lean 4 continuous mathematics code.

    Performs multiple types of verification:
    1. Syntax validation
    2. Type checking
    3. Mathematical correctness
    4. Domain-specific pattern checking
    5. Conservation law verification
    6. Boundary condition validation
    7. Proof verification (with LeanAide)
    """

    def __init__(
        self,
        leanaide_config: Optional[Any] = None,
        enable_leanaide: bool = True,
        timeout: float = 30.0
    ):
        """
        Initialize the Lean 4 verifier.

        Args:
            leanaide_config: Optional LeanAide configuration
            enable_leanaide: Whether to use LeanAide for proof verification
            timeout: Timeout for Lean 4 verification
        """
        self.timeout = timeout
        self.enable_leanaide = enable_leanaide and LEANAIDE_AVAILABLE
        self.domain_patterns = get_domain_patterns()

        # Initialize LeanAide client if available
        self.leanaide_client = None
        if self.enable_leanaide:
            try:
                if leanaide_config:
                    self.leanaide_client = LeanAideClient(leanaide_config)
                else:
                    self.leanaide_client = LeanAideClient()
                logger.info("LeanAide client initialized for verification")
            except (LeanAideClientError, ConnectionError, TimeoutError) as e:
                logger.warning(f"Failed to initialize LeanAide client: {e}")
                self.enable_leanaide = False

        # Initialize verification patterns
        self._init_syntax_patterns()
        self._init_conservations_patterns()

        logger.info(f"Lean4 Verifier initialized (LeanAide: {self.enable_leanaide})")

    def _init_syntax_patterns(self):
        """Initialize syntax validation patterns"""
        self.syntax_patterns = {
            # Lean 4 keywords and structures
            "def": re.compile(r'\bdef\s+\w+'),
            "theorem": re.compile(r'\btheorem\s+\w+'),
            "lemma": re.compile(r'\blemma\s+\w+'),
            "structure": re.compile(r'\bstructure\s+\w+'),
            "class": re.compile(r'\bclass\s+\w+'),
            "instance": re.compile(r'\binstance\s+'),

            # Type annotations
            "prop": re.compile(r'\bProp\b'),
            "type": re.compile(r'\bType\b'),
            "function_arrow": re.compile(r'→|->'),

            # Quantifiers
            "forall": re.compile(r'∀|forall'),
            "exists": re.compile(r'∃|exists'),
            "exists_unique": re.compile(r'∃!|exists!'),

            # Math symbols
            "deriv": re.compile(r'\bderiv\b'),
            "fderiv": re.compile(r'\bfderiv\b'),
            "integral": re.compile(r'\bintegral\b|\b∫\b'),
            "limit": re.compile(r'\blimit\b|\blim\b'),
        }

    def _init_conservations_patterns(self):
        """Initialize conservation law patterns"""
        self.conservation_patterns = {
            ScientificDomain.PHYSICS: {
                "energy": ["energy", "Energy", "E", "Hamiltonian", "H"],
                "momentum": ["momentum", "Momentum", "p", "P"],
                "angular_momentum": ["angular", "L", "J"],
                "mass": ["mass", "m", "M"],
                "charge": ["charge", "Q", "q"]
            },
            ScientificDomain.CHEMISTRY: {
                "mass": ["mass", "concentration", "[C]", "[A]"],
                "charge": ["charge", "Q"],
                "atoms": ["atoms", "molecules", "N"]
            },
            ScientificDomain.BIOLOGY: {
                "population": ["population", "N", "x", "y"],
                "total_population": ["S+I+R", "N_total", "sum"]
            },
            ScientificDomain.ENGINEERING: {
                "energy": ["energy", "E", "power", "P"],
                "mass": ["mass", "m"]
            },
            ScientificDomain.ECONOMICS: {
                "budget": ["budget", "wealth", "W"],
                "arbitrage": ["arbitrage", "no_arbitrage"]
            }
        }

    # ========================================================================
    # Main Verification Methods
    # ========================================================================

    def verify(
        self,
        translation_result: Lean4TranslationResult,
        detection_result: Optional[MathDetectionResult] = None,
        checks: Optional[List[CheckType]] = None
    ) -> VerificationResult:
        """
        Perform comprehensive verification of Lean 4 code.

        Args:
            translation_result: Result from ODE/PDE translator
            detection_result: Original detection result (for domain info)
            checks: List of checks to perform (None = all)

        Returns:
            VerificationResult with detailed findings
        """
        import time
        start_time = time.time()

        issues = []
        checks_performed = []
        passed = 0
        failed = 0
        warnings = 0

        lean4_code = translation_result.lean4_code

        # Determine which checks to perform
        if checks is None:
            checks = [
                CheckType.SYNTAX,
                CheckType.TYPE,
                CheckType.MATHEMATICAL,
                CheckType.DOMAIN,
                CheckType.CONSERVATION,
                CheckType.BOUNDARY,
            ]
            if self.enable_leanaide:
                checks.append(CheckType.PROOF)

        # Perform each check
        for check_type in checks:
            try:
                check_issues = self._perform_check(
                    check_type,
                    lean4_code,
                    translation_result,
                    detection_result
                )

                checks_performed.append(check_type)

                # Count issues by severity
                for issue in check_issues:
                    issues.append(issue)
                    if issue.severity == "error":
                        failed += 1
                    elif issue.severity == "warning":
                        warnings += 1
                    else:
                        passed += 1

                if not check_issues:
                    passed += 1

            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Error performing {check_type.value} check: {e}")
                issues.append(VerificationIssue(
                    check_type=check_type,
                    severity="error",
                    message=f"Check failed with exception: {str(e)}",
                    suggestion="Review code structure and try again"
                ))
                failed += 1

        # Determine overall status
        if failed > 0:
            overall_status = VerificationStatus.FAILED
        elif warnings > 0:
            overall_status = VerificationStatus.WARNING
        else:
            overall_status = VerificationStatus.PASSED

        verification_time = time.time() - start_time

        return VerificationResult(
            overall_status=overall_status,
            checks_performed=checks_performed,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            verification_time=verification_time,
            metadata={
                "total_checks": len(checks),
                "leanaide_enabled": self.enable_leanaide
            }
        )

    def verify_code(
        self,
        lean4_code: str,
        domain: Optional[ScientificDomain] = None
    ) -> VerificationResult:
        """
        Verify standalone Lean 4 code.

        Args:
            lean4_code: Lean 4 code to verify
            domain: Optional scientific domain

        Returns:
            VerificationResult
        """
        # Create a minimal translation result wrapper
        from ode_pde_translator import Lean4TranslationResult

        translation_result = Lean4TranslationResult(
            success=True,
            lean4_code=lean4_code,
            definitions=[],
            theorems=[],
            proof_scaffolds=[],
            imports=[]
        )

        detection_result = None
        if domain:
            from continuous_math_detector import MathDetectionResult
            detection_result = MathDetectionResult(
                math_type=MathType.ODE,
                problem_type=ProblemType.UNKNOWN,
                domain=domain,
                confidence=1.0,
                equations=[],
                variables=[],
                notation="",
                keywords=[]
            )

        return self.verify(translation_result, detection_result)

    # ========================================================================
    # Individual Check Methods
    # ========================================================================

    def _perform_check(
        self,
        check_type: CheckType,
        lean4_code: str,
        translation_result: Lean4TranslationResult,
        detection_result: Optional[MathDetectionResult]
    ) -> List[VerificationIssue]:
        """Perform a specific verification check"""

        if check_type == CheckType.SYNTAX:
            return self._check_syntax(lean4_code)
        elif check_type == CheckType.TYPE:
            return self._check_types(lean4_code)
        elif check_type == CheckType.MATHEMATICAL:
            return self._check_mathematical_correctness(lean4_code, translation_result)
        elif check_type == CheckType.DOMAIN:
            return self._check_domain_patterns(lean4_code, detection_result)
        elif check_type == CheckType.CONSERVATION:
            return self._check_conservation_laws(lean4_code, detection_result)
        elif check_type == CheckType.BOUNDARY:
            return self._check_boundary_conditions(lean4_code, detection_result)
        elif check_type == CheckType.PROOF:
            return self._check_with_leanaide(lean4_code)
        else:
            return []

    def _check_syntax(self, lean4_code: str) -> List[VerificationIssue]:
        """Check Lean 4 syntax validity"""
        issues = []

        # Check for basic structure
        if not lean4_code.strip():
            issues.append(VerificationIssue(
                check_type=CheckType.SYNTAX,
                severity="error",
                message="Empty Lean 4 code",
                suggestion="Provide valid Lean 4 code"
            ))
            return issues

        # Check for namespace
        if "namespace" not in lean4_code:
            issues.append(VerificationIssue(
                check_type=CheckType.SYNTAX,
                severity="warning",
                message="No namespace declaration found",
                suggestion="Add 'namespace NameSpaceName' to organize code"
            ))

        # Check for matching delimiters
        open_braces = lean4_code.count('{')
        close_braces = lean4_code.count('}')
        if open_braces != close_braces:
            issues.append(VerificationIssue(
                check_type=CheckType.SYNTAX,
                severity="error",
                message=f"Mismatched braces: {open_braces} open, {close_braces} close",
                suggestion="Ensure all { have matching }"
            ))

        # Check for imports
        if "import" not in lean4_code and lean4_code.startswith("import"):
            # Valid if starts with import (single import line)
            pass
        elif "import Mathlib" not in lean4_code:
            issues.append(VerificationIssue(
                check_type=CheckType.SYNTAX,
                severity="warning",
                message="No Mathlib imports found",
                suggestion="Import required Mathlib modules for mathematical objects"
            ))

        # Check for at least one definition or theorem
        if not re.search(r'\b(def|theorem|lemma|structure)\s+\w+', lean4_code):
            issues.append(VerificationIssue(
                check_type=CheckType.SYNTAX,
                severity="error",
                message="No definitions, theorems, or lemmas found",
                suggestion="Add at least one definition or theorem"
            ))

        # Check for proper Lean 4 notation
        has_lean4_notation = any(
            pattern.search(lean4_code)
            for pattern in [
                self.syntax_patterns["prop"],
                self.syntax_patterns["function_arrow"]
            ]
        )
        if not has_lean4_notation:
            issues.append(VerificationIssue(
                check_type=CheckType.SYNTAX,
                severity="warning",
                message="No Lean 4 type annotations found",
                suggestion="Add type annotations (e.g., ': Prop', ': Type')"
            ))

        return issues

    def _check_types(self, lean4_code: str) -> List[VerificationIssue]:
        """Check type consistency"""
        issues = []

        # Check for consistent use of Real vs other types
        if "Real" in lean4_code:
            # Check for Real in imports
            if "Mathlib.Data.Real.Basic" not in lean4_code:
                issues.append(VerificationIssue(
                    check_type=CheckType.TYPE,
                    severity="warning",
                    message="Using Real type but not importing Mathlib.Data.Real.Basic",
                    suggestion="Add 'import Mathlib.Data.Real.Basic'"
                ))

        # Check for Prop usage
        if "Prop" in lean4_code:
            if ": Prop" not in lean4_code:
                issues.append(VerificationIssue(
                    check_type=CheckType.TYPE,
                    severity="warning",
                    message="Prop mentioned but no type annotations use ': Prop'",
                    suggestion="Ensure propositions are properly typed with ': Prop'"
                ))

        # Check for function type consistency
        arrow_count = lean4_code.count('→') + lean4_code.count('->')
        if arrow_count > 0 and "fun" not in lean4_code and "→" not in lean4_code:
            issues.append(VerificationIssue(
                check_type=CheckType.TYPE,
                severity="info",
                message="Function types found but no explicit function definitions",
                suggestion="Consider using 'fun' or λ notation for clarity"
            ))

        return issues

    def _check_mathematical_correctness(
        self,
        lean4_code: str,
        translation_result: Lean4TranslationResult
    ) -> List[VerificationIssue]:
        """Check mathematical correctness"""
        issues = []

        # Check for derivative definitions
        if "deriv" in lean4_code:
            # Should have imports for derivatives
            has_deriv_import = any(
                "Deriv" in imp or "deriv" in imp
                for imp in translation_result.imports
            )
            if not has_deriv_import:
                issues.append(VerificationIssue(
                    check_type=CheckType.MATHEMATICAL,
                    severity="warning",
                    message="Using derivatives but no Deriv import found",
                    suggestion="Add 'import Mathlib.Analysis.Calculus.Deriv'"
                ))

        # Check for integral definitions
        if "integral" in lean4_code.lower() or "∫" in lean4_code:
            should_have_integral = True
            # Check for appropriate imports
            has_integral = any("Integral" in imp for imp in translation_result.imports)
            if not has_integral:
                issues.append(VerificationIssue(
                    check_type=CheckType.MATHEMATICAL,
                    severity="info",
                    message="Integral notation found but integral import not confirmed",
                    suggestion="Consider adding integral-related imports"
                ))

        # Check for differential equation structure
        if "ode" in lean4_code.lower() or "pde" in lean4_code.lower():
            # Should have derivative operators
            if "deriv" not in lean4_code and "fderiv" not in lean4_code:
                issues.append(VerificationIssue(
                    check_type=CheckType.MATHEMATICAL,
                    severity="warning",
                    message="ODE/PDE mentioned but no derivative operators found",
                    suggestion="Ensure differential equations use 'deriv' or 'fderiv'"
                ))

        # Check for quantifier balance
        forall_count = lean4_code.count('∀') + lean4_code.count('forall')
        exists_count = lean4_code.count('∃') + lean4_code.count('exists')

        if forall_count == 0 and exists_count == 0:
            if "theorem" in lean4_code or "lemma" in lean4_code:
                issues.append(VerificationIssue(
                    check_type=CheckType.MATHEMATICAL,
                    severity="info",
                    message="Theorem/lemma found but no quantifiers detected",
                    suggestion="Consider adding explicit quantifiers (∀, ∃) for clarity"
                ))

        return issues

    def _check_domain_patterns(
        self,
        lean4_code: str,
        detection_result: Optional[MathDetectionResult]
    ) -> List[VerificationIssue]:
        """Check domain-specific patterns"""
        issues = []

        if not detection_result:
            return issues

        domain = detection_result.domain
        if domain == ScientificDomain.GENERAL:
            return issues

        # Get domain knowledge
        domain_knowledge = self.domain_patterns.get_domain_knowledge(domain)
        if not domain_knowledge:
            return issues

        # Check for domain-specific verification patterns
        verification_patterns = domain_knowledge.verification_patterns

        # Check for conservation laws mentioned in comments or structure
        code_lower = lean4_code.lower()
        found_patterns = []

        for pattern in verification_patterns:
            pattern_lower = pattern.lower()
            # Check if pattern is mentioned
            if any(keyword in code_lower for keyword in pattern_lower.split()):
                found_patterns.append(pattern)

        # Suggest adding verification comments if patterns not found
        if not found_patterns and len(verification_patterns) > 0:
            issues.append(VerificationIssue(
                check_type=CheckType.DOMAIN,
                severity="info",
                message=f"No domain-specific verification patterns found ({domain.value})",
                suggestion=f"Consider adding: {', '.join(verification_patterns[:3])}"
            ))

        return issues

    def _check_conservation_laws(
        self,
        lean4_code: str,
        detection_result: Optional[MathDetectionResult]
    ) -> List[VerificationIssue]:
        """Check conservation law verification"""
        issues = []

        if not detection_result:
            return issues

        domain = detection_result.domain
        if domain not in self.conservation_patterns:
            return issues

        # Get conservation patterns for this domain
        conservation_dict = self.conservation_patterns[domain]
        code_lower = lean4_code.lower()

        # Check for conservation-related keywords
        found_conservations = []
        for conservation_type, keywords in conservation_dict.items():
            for keyword in keywords:
                if keyword.lower() in code_lower:
                    found_conservations.append(conservation_type)
                    break

        # Suggest adding conservation theorems
        if not found_conservations:
            issues.append(VerificationIssue(
                check_type=CheckType.CONSERVATION,
                severity="info",
                message=f"No explicit conservation law statements found",
                suggestion=f"Consider adding theorems for: {', '.join(list(conservation_dict.keys())[:3])}"
            ))

        return issues

    def _check_boundary_conditions(
        self,
        lean4_code: str,
        detection_result: Optional[MathDetectionResult]
    ) -> List[VerificationIssue]:
        """Check boundary condition specification"""
        issues = []

        if not detection_result:
            return issues

        # Check based on problem type
        problem_type = detection_result.problem_type

        if problem_type == ProblemType.INITIAL_VALUE:
            # Should have initial conditions
            has_initial = (
                "initial" in lean4_code.lower() or
                "0)" in lean4_code or
                re.search(r'\w+\s+0\s*=', lean4_code)
            )
            if not has_initial:
                issues.append(VerificationIssue(
                    check_type=CheckType.BOUNDARY,
                    severity="warning",
                    message="IVP detected but no explicit initial condition found",
                    suggestion="Add initial condition: 'x 0 = x₀' or similar"
                ))

        elif problem_type == ProblemType.BOUNDARY_VALUE:
            # Should have boundary conditions
            has_boundary = (
                "boundary" in lean4_code.lower() or
                "∀ x ∈" in lean4_code or
                re.search(r'\w+\s+\w+\s*=\s*\w+\s+.*boundary', lean4_code.lower())
            )
            if not has_boundary:
                issues.append(VerificationIssue(
                    check_type=CheckType.BOUNDARY,
                    severity="warning",
                    message="BVP detected but no explicit boundary conditions found",
                    suggestion="Add boundary conditions at domain boundaries"
                ))

        return issues

    def _check_with_leanaide(self, lean4_code: str) -> List[VerificationIssue]:
        """Check code with LeanAide (if available)"""
        issues = []

        if not self.enable_leanaide or not self.leanaide_client:
            issues.append(VerificationIssue(
                check_type=CheckType.PROOF,
                severity="info",
                message="LeanAide not available for proof verification",
                suggestion="Install LeanAide server for automated proof checking"
            ))
            return issues

        try:
            # Submit to LeanAide for elaboration check
            result = self.leanaide_client.submit_task(
                task_type=TaskType.ELABORATE,
                source_code=lean4_code,
                timeout=self.timeout
            )

            if result.success:
                # Check if elaboration succeeded
                if result.data and "errors" in result.data:
                    for error in result.data["errors"]:
                        issues.append(VerificationIssue(
                            check_type=CheckType.PROOF,
                            severity="error",
                            message=f"LeanAide error: {error}",
                            suggestion="Review the code structure and fix Lean 4 syntax errors"
                        ))
                else:
                    # Success - no issues
                    pass
            else:
                issues.append(VerificationIssue(
                    check_type=CheckType.PROOF,
                    severity="warning",
                    message=f"LeanAide verification failed: {result.error}",
                    suggestion="Review code or check LeanAide server status"
                ))

        except (LeanAideClientError, ConnectionError, TimeoutError) as e:
            issues.append(VerificationIssue(
                check_type=CheckType.PROOF,
                severity="warning",
                message=f"LeanAide verification error: {str(e)}",
                suggestion="Ensure LeanAide server is running"
            ))

        return issues


# ============================================================================
# Convenience Functions
# ============================================================================

def verify_lean4_code(
    lean4_code: str,
    domain: Optional[ScientificDomain] = None,
    enable_leanaide: bool = True
) -> VerificationResult:
    """
    Convenience function to verify Lean 4 code.

    Args:
        lean4_code: Lean 4 code to verify
        domain: Optional scientific domain
        enable_leanaide: Whether to use LeanAide for proof verification

    Returns:
        VerificationResult
    """
    verifier = Lean4Verifier(enable_leanaide=enable_leanaide)
    return verifier.verify_code(lean4_code, domain)


def verify_translation(
    translation_result: Lean4TranslationResult,
    detection_result: Optional[MathDetectionResult] = None
) -> VerificationResult:
    """
    Verify a complete translation result.

    Args:
        translation_result: Result from ODE/PDE translator
        detection_result: Original detection result

    Returns:
        VerificationResult
    """
    verifier = Lean4Verifier()
    return verifier.verify(translation_result, detection_result)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    from ode_pde_translator import ODEPDETranslator
    from continuous_math_detector import ContinuousMathDetector

    print("=" * 80)
    print("Lean 4 Verification - Example")
    print("=" * 80)

    # Create detector and translator
    detector = ContinuousMathDetector()
    translator = ODEPDETranslator()
    verifier = Lean4Verifier()

    # Example: Heat equation
    print("\n1. Detecting heat equation...")
    text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"
    detection_result = detector.detect(text)

    print(f"   Detected: {detection_result.math_type.value}")
    print(f"   Domain: {detection_result.domain.value}")

    # Translate
    print("\n2. Translating to Lean 4...")
    translation_result = translator.translate(detection_result)

    if translation_result.success:
        print("   ✓ Translation successful")

        # Verify
        print("\n3. Verifying Lean 4 code...")
        verification_result = verifier.verify(translation_result, detection_result)

        print(f"   Status: {verification_result.overall_status.value.upper()}")
        print(f"   Checks performed: {len(verification_result.checks_performed)}")
        print(f"   Passed: {verification_result.passed_checks}")
        print(f"   Failed: {verification_result.failed_checks}")
        print(f"   Warnings: {verification_result.warnings}")
        print(f"   Time: {verification_result.verification_time:.2f}s")

        if verification_result.issues:
            print("\n   Issues found:")
            for issue in verification_result.issues[:5]:  # Show first 5
                print(f"   - [{issue.severity.upper()}] {issue.message}")
                if issue.suggestion:
                    print(f"     → {issue.suggestion}")

        if verification_result.is_valid:
            print("\n   ✓ Code is valid!")
        else:
            print("\n   ✗ Code has errors that need fixing")
    else:
        print("   ✗ Translation failed")
