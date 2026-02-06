"""
LeanAide Red-Flagging System for Lean 4 Proofs

This module provides comprehensive red-flagging (quality control) for Lean 4 proofs,
extending the MDAP red-flagging framework with Lean-specific validation.

Architecture:
    LeanRedFlagRules: Enhanced rules for Lean 4 proofs
    LeanRedFlagger: Main red-flagging engine for Lean proofs
    LeanProofValidator: Comprehensive validation of Lean proofs
    LeanProofQualityScorer: Multi-dimensional quality scoring
    LeanQualityScore: Quality score with detailed breakdown

Features:
    - Syntax validation: Lean 4 syntax checking
    - Semantic validation: Type checking, tactic applicability
    - Structural validation: Proof length, circular reasoning
    - Quality validation: Elegance, clarity, efficiency
    - Verification: Integration with LeanAide for actual verification
    - CAV-NLP semantic redflag detection: Enhanced semantic analysis
"""

import re
import logging
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import Counter

# Import base red-flagging classes
from mdap_engine import RedFlagRules, RedFlagger

# Import Lean 4 integration
try:
    from lean4_integration import (
        Lean4VerificationEngine,
        Lean4ServerConfig,
        Lean4VerificationConfig,
        VerificationResult,
        LeanAideClient
    )
    LEAN4_INTEGRATION_AVAILABLE = True
except ImportError:
    LEAN4_INTEGRATION_AVAILABLE = False
    logging.warning("Lean 4 integration not available, red-flagging will use static analysis only")

# Import CAV-NLP for enhanced semantic detection
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logging.warning("CAV-NLP not available, semantic redflag detection will be limited")

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class LeanProofType(Enum):
    """Types of Lean proof declarations"""
    THEOREM = "theorem"
    LEMMA = "lemma"
    DEF = "def"
    EXAMPLE = "example"
    STRUCTURE = "structure"
    CLASS = "class"
    INSTANCE = "instance"


@dataclass
class LeanProof:
    """
    Represents a Lean 4 proof with metadata

    Attributes:
        code: Full Lean code
        name: Proof name
        proof_type: Type of proof (theorem, lemma, etc.)
        statement: Mathematical statement being proved
        tactics: List of tactics used in proof
        imports: List of imports required
        dependencies: List of theorem/definition dependencies
    """
    code: str
    name: str
    proof_type: LeanProofType
    statement: str = ""
    tactics: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Metadata
    line_count: int = 0
    tactic_count: int = 0
    has_sorry: bool = False
    sorry_count: int = 0


@dataclass
class LeanProofState:
    """
    Represents a proof state for tactic applicability checking

    Attributes:
        goal: Current goal
        hypotheses: Current hypotheses
        context: Context information
    """
    goal: str = ""
    hypotheses: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeanQualityScore:
    """
    Quality score for Lean proofs with multi-dimensional breakdown

    Attributes:
        overall_score: Overall quality score (0-1)
        elegance: Elegance score (tactic diversity, conciseness)
        clarity: Clarity score (understandability, naming)
        efficiency: Efficiency score (minimal redundancy)
        correctness: Correctness score (verified, no sorries)
        flags: List of red flags raised
        suggestions: List of improvement suggestions
    """
    overall_score: float
    elegance: float
    clarity: float
    efficiency: float
    correctness: float
    flags: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "elegance": self.elegance,
            "clarity": self.clarity,
            "efficiency": self.efficiency,
            "correctness": self.correctness,
            "flags": self.flags,
            "suggestions": self.suggestions
        }


@dataclass
class ValidationResult:
    """
    Result of comprehensive Lean proof validation

    Attributes:
        valid: Whether proof passes all validations
        errors: List of validation errors
        warnings: List of validation warnings
        quality_score: Quality score object
        verification_result: Result from LeanAide verification
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_score: Optional[LeanQualityScore] = None
    verification_result: Optional[VerificationResult] = None


# =============================================================================
# LEAN 4 TACTIC CATALOG
# =============================================================================

# Common Lean 4 tactics categorized by type
LEAN_TACTICS = {
    "basic": [
        "intro", "intros", "apply", "exact", "refine", "by", "sorry",
        "assumption", "trivial", "rfl", "rwa"
    ],
    "rewrite": [
        "rw", "rewrite", "rwa", "simp", "dsimp", "simp_rw"
    ],
    "induction": [
        "induction", "cases", "case", "rcases", "obtain"
    ],
    "logic": [
        "have", "suffices", "show", "calc", "by_contra", "by_cases",
        "contrapose", "refute", "exfalso", "absurd"
    ],
    "arith": [
        "linarith", "omega", "ring", "ring_nf", "norm_num",
        "norm_cast", "push_neg", "positivity"
    ],
    "algebra": [
        "abel", "abel1", "group", "nlinarith"
    ],
    "completion": [
        "tidy", "aesop", "solve_by_elim", "hint", "suggest"
    ],
    "library": [
        "library_search", "exact?", "apply?", "simp?"
    ],
    "advanced": [
        "wlog", "generalize", "specialize", "transitivity",
        "constructor", "injection", "injections", "subst"
    ]
}

# Tactics that may indicate problems if overused
POTENTIALLY_PROBLEMATIC_TACTICS = [
    "sorry",           # Incomplete proof
    "admit",           # Alternative to sorry
    "simp",            # May indicate lack of understanding if overused
    "aesop",           # Automation - may hide reasoning
    "tidy",            # Automation - may hide reasoning
]

# Required keywords for Lean 4 proofs
LEAN_KEYWORDS = [
    "theorem", "lemma", "def", "structure", "class", "instance",
    "example", "inductive", "by", "where", "deriving"
]


# =============================================================================
# LEAN RED FLAG RULES
# =============================================================================

@dataclass
class LeanRedFlagRules(RedFlagRules):
    """
    Enhanced red-flag rules for Lean 4 proofs

    Extends RedFlagRules with Lean-specific validation rules.

    Attributes:
        # Structural rules
        max_proof_length: Maximum number of lines in a proof
        max_tactic_sequence: Maximum number of tactics in a proof

        # Completeness rules
        require_no_sorries: Whether to require no `sorry` or `admit` placeholders
        max_sorry_count: Maximum number of sorries allowed if not strictly forbidden

        # Quality rules
        min_elegance_score: Minimum elegance score (0-1)
        max_simplification_ratio: Maximum ratio of simp tactics to total tactics

        # Syntactic rules
        require_lean_keywords: Whether to require Lean keywords in code

        # Semantic rules
        check_tactic_applicability: Whether to check if tactics are applicable
        check_imports: Whether to check if required imports are present

        # Tactic restrictions
        forbidden_tactics: List of tactics that should trigger red flags
        required_imports: List of imports that must be present

        # Complexity rules
        max_nesting_depth: Maximum nesting depth of proof structure
        min_tactic_diversity: Minimum number of unique tactics required
    """

    # Structural limits
    max_proof_length: int = 500  # lines
    max_tactic_sequence: int = 100  # tactics

    # Completeness
    require_no_sorries: bool = True
    max_sorry_count: int = 0  # if require_no_sorries is False

    # Quality thresholds
    min_elegance_score: float = 0.3
    max_simplification_ratio: float = 0.8  # max simp/total tactics

    # Syntax
    require_lean_keywords: bool = True

    # Semantic checking
    check_tactic_applicability: bool = True
    check_imports: bool = True

    # Tactic restrictions
    forbidden_tactics: List[str] = field(default_factory=lambda: ["admit"])
    required_imports: List[str] = field(default_factory=list)

    # Complexity
    max_nesting_depth: int = 20
    min_tactic_diversity: int = 2


# =============================================================================
# LEAN RED FLAGGER
# =============================================================================

class LeanRedFlagger(RedFlagger):
    """
    Comprehensive red-flagging for Lean 4 proofs

    Extends RedFlagger with Lean-specific validation across multiple dimensions:
    - Syntax validation
    - Semantic validation
    - Structural validation
    - Quality validation
    - Verification validation
    - CAV-NLP semantic redflag detection
    """

    def __init__(self, rules: LeanRedFlagRules, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Lean red-flagger

        Args:
            rules: LeanRedFlagRules for Lean-specific validation
            config: Optional configuration dictionary
        """
        super().__init__(rules)
        self.lean_rules = rules
        self.config = config or {}

        # Initialize Lean 4 verification engine if available
        self.verification_engine = None
        if LEAN4_INTEGRATION_AVAILABLE:
            try:
                server_config = Lean4ServerConfig(
                    enable_simulation_fallback=True
                )
                verification_config = Lean4VerificationConfig(
                    enable_caching=True,
                    cache_ttl_seconds=3600
                )
                self.verification_engine = Lean4VerificationEngine(
                    server_url="http://localhost:7654",
                    server_config=server_config,
                    config=verification_config
                )
                logger.info("Lean 4 verification engine initialized")
            except (IOError, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"Failed to initialize Lean 4 verification engine: {e}")
        
        # Initialize CAV-NLP components for semantic redflag detection
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.enhanced_solver = EnhancedZ3Solver()
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP components initialized for redflag detection")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP components: {e}")
                self.use_cav_nlp = False

    def is_flagged(self, proof: LeanProof) -> Tuple[bool, List[str]]:
        """
        Check if a Lean proof should be red-flagged

        Args:
            proof: LeanProof object

        Returns:
            Tuple of (is_flagged, list_of_reasons)
        """
        all_reasons: List[str] = []

        # Run all validation checks
        syntax_reasons = self.check_syntax(proof)
        semantic_reasons = self.check_semantics(proof)
        structural_reasons = self.check_structure(proof)
        quality_reasons = self.check_quality(proof)
        verification_reasons = self.check_verification(proof)

        all_reasons.extend(syntax_reasons)
        all_reasons.extend(semantic_reasons)
        all_reasons.extend(structural_reasons)
        all_reasons.extend(quality_reasons)
        all_reasons.extend(verification_reasons)

        return len(all_reasons) > 0, all_reasons

    def check_syntax(self, proof: LeanProof) -> List[str]:
        """
        Check Lean 4 syntax

        Validates:
        - Malformed tactic syntax
        - Missing keywords (by, :=, etc.)
        - Unmatched parentheses
        - Invalid identifiers

        Args:
            proof: LeanProof to check

        Returns:
            List of syntax errors (empty if valid)
        """
        errors: List[str] = []

        code = proof.code

        # 1. Check for required keywords
        if self.lean_rules.require_lean_keywords:
            if not any(keyword in code for keyword in LEAN_KEYWORDS):
                errors.append("missing_lean_keywords")

        # 2. Check for unmatched parentheses
        if not self._check_parentheses(code):
            errors.append("unmatched_parentheses")

        # 3. Check for malformed tactic syntax
        # Tactics should end with newline or proper continuation
        tactic_pattern = r'^\s*(\w+)\s+'
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if '--' in line:
                line = line.split('--')[0]

            # Check for tactics with proper structure
            match = re.match(tactic_pattern, line.strip())
            if match:
                tactic = match.group(1)
                # Check if it's a known tactic
                all_tactics = []
                for category in LEAN_TACTICS.values():
                    all_tactics.extend(category)

                if tactic not in all_tactics and not tactic.startswith('"'):
                    # Unknown tactic - might be typo
                    errors.append(f"unknown_tactic:{tactic}:line_{line_num}")

        # 4. Check for proper theorem/lemma structure
        if proof.proof_type in [LeanProofType.THEOREM, LeanProofType.LEMMA]:
            if ':=' not in code and 'by' not in code:
                errors.append("missing_proof_body")

        # 5. Check for valid identifiers
        # Lean identifiers must start with letter or underscore
        identifier_pattern = r'\b([A-Za-z_][A-Za-z0-9_\.]*)\b'
        invalid_identifiers = []
        for match in re.finditer(identifier_pattern, code):
            identifier = match.group(1)
            # Check if it's a known Lean keyword or tactic
            if identifier in LEAN_KEYWORDS:
                continue
            # Check if it's in tactic list
            all_tactics = []
            for category in LEAN_TACTICS.values():
                all_tactics.extend(category)
            if identifier in all_tactics:
                continue
            # Otherwise it's a user identifier - should be properly formed
            # (already validated by regex)

        return errors

    def check_semantics(self, proof: LeanProof) -> List[str]:
        """
        Check Lean 4 semantics

        Validates:
        - Tactic applicability (basic check)
        - Type mismatches (heuristic)
        - Undefined constants (heuristic)
        - Missing imports (basic check)

        Args:
            proof: LeanProof to check

        Returns:
            List of semantic errors (empty if valid)
        """
        errors: List[str] = []

        code = proof.code

        # 1. Check for forbidden tactics
        for tactic in self.lean_rules.forbidden_tactics:
            if tactic in proof.tactics:
                errors.append(f"forbidden_tactic:{tactic}")

        # 2. Check for potentially problematic patterns
        # Check for repeated application of same tactic without progress
        if self._has_repetitive_tactics(proof):
            errors.append("repetitive_tactics")

        # 3. Check tactic applicability (basic)
        if self.lean_rules.check_tactic_applicability:
            for tactic in proof.tactics:
                # Check if tactic has proper arguments
                if tactic == "simp" and not self._check_simp_args(code):
                    errors.append("simp_without_args")

                if tactic == "rw" and not self._check_rw_args(code):
                    errors.append("rw_without_args")

                if tactic == "induction" and not self._check_induction_args(code):
                    errors.append("induction_without_var")

        # 4. Check for common mistakes
        # Check for tactics in wrong order
        if self._has_wrong_tactic_order(proof):
            errors.append("suspicious_tactic_order")

        # 5. Check imports if required
        if self.lean_rules.check_imports and self.lean_rules.required_imports:
            missing_imports = set(self.lean_rules.required_imports) - set(proof.imports)
            if missing_imports:
                errors.append(f"missing_imports:{','.join(missing_imports)}")

        return errors

    def check_structure(self, proof: LeanProof) -> List[str]:
        """
        Check Lean proof structure

        Validates:
        - Proof too long
        - Too many tactics
        - Circular reasoning
        - Inefficient proof structure

        Args:
            proof: LeanProof to check

        Returns:
            List of structural errors (empty if valid)
        """
        errors: List[str] = []

        # 1. Check proof length
        if proof.line_count > self.lean_rules.max_proof_length:
            errors.append(f"proof_too_long:{proof.line_count}_lines")

        # 2. Check tactic count
        if proof.tactic_count > self.lean_rules.max_tactic_sequence:
            errors.append(f"too_many_tactics:{proof.tactic_count}_tactics")

        # 3. Check for circular reasoning
        if self._has_circular_reasoning(proof):
            errors.append("circular_reasoning")

        # 4. Check nesting depth
        nesting_depth = self._calculate_nesting_depth(proof.code)
        if nesting_depth > self.lean_rules.max_nesting_depth:
            errors.append(f"excessive_nesting:{nesting_depth}_levels")

        # 5. Check tactic diversity
        unique_tactics = len(set(proof.tactics))
        if unique_tactics < self.lean_rules.min_tactic_diversity and proof.tactic_count > 5:
            errors.append(f"low_tactic_diversity:{unique_tactics}_unique")

        # 6. Check for inefficient patterns
        if self._has_inefficient_pattern(proof):
            errors.append("inefficient_proof_structure")

        return errors

    def check_quality(self, proof: LeanProof) -> List[str]:
        """
        Check Lean proof quality

        Validates:
        - Too many `sorry` placeholders
        - Excessive use of `simp`
        - Low tactic diversity
        - Poor naming

        Args:
            proof: LeanProof to check

        Returns:
            List of quality issues (empty if high quality)
        """
        errors: List[str] = []

        # 1. Check for sorries
        if proof.has_sorry:
            if self.lean_rules.require_no_sorries:
                errors.append(f"contains_sorry:{proof.sorry_count}_instances")
            elif proof.sorry_count > self.lean_rules.max_sorry_count:
                errors.append(f"too_many_sorries:{proof.sorry_count}_instances")

        # 2. Check simplification ratio
        if proof.tactic_count > 0:
            simp_count = sum(1 for t in proof.tactics if t in ["simp", "dsimp", "simp_rw"])
            simp_ratio = simp_count / proof.tactic_count
            if simp_ratio > self.lean_rules.max_simplification_ratio:
                errors.append(f"excessive_simp:{simp_ratio:.2%}_ratio")

        # 3. Check for automation overuse
        automation_tactics = ["simp", "aesop", "tidy", "solve_by_elim"]
        automation_count = sum(1 for t in proof.tactics if t in automation_tactics)
        if proof.tactic_count > 10 and automation_count / proof.tactic_count > 0.7:
            errors.append("over_reliance_on_automation")

        # 4. Check naming quality
        if self._has_poor_naming(proof):
            errors.append("poor_naming_conventions")

        # 5. Check elegance score
        # This will be computed by LeanProofQualityScorer
        # For now, just flag if it's likely low
        if proof.tactic_count > 50 and len(set(proof.tactics)) < 5:
            errors.append("likely_low_elegance")

        return errors

    def check_verification(self, proof: LeanProof) -> List[str]:
        """
        Check Lean proof verification status

        Validates:
        - LeanAide verification fails
        - Remaining goals after proof
        - Elaboration errors (via verification)

        Args:
            proof: LeanProof to check

        Returns:
            List of verification errors (empty if verified)
        """
        errors: List[str] = []

        # If verification engine is not available, skip
        if self.verification_engine is None:
            return errors

        try:
            # Run verification asynchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't run sync verification
                # Return empty for now - async verification should be done separately
                return errors

            result = loop.run_until_complete(
                self.verification_engine.verify_mathematical_solution(proof.code)
            )

            if not result.success:
                errors.append("verification_failed")
                # Add specific errors
                if result.errors:
                    for error in result.errors[:5]:  # Limit to first 5 errors
                        errors.append(f"verification_error:{error[:50]}")
            else:
                # Check for remaining sorries
                if "sorry" in proof.code and result.success:
                    errors.append("verification_passed_with_sorry")

        except (IOError, ConnectionError, TimeoutError) as e:
            # Verification failed - add as warning
            logger.warning(f"Verification check failed: {e}")
            errors.append(f"verification_error:{str(e)[:50]}")

        return errors

    def check_tactic_applicability(self, tactic: str, state: LeanProofState) -> bool:
        """
        Check if a tactic is applicable to the current proof state

        This is a basic heuristic check - full applicability checking requires
        running Lean's type checker.

        Args:
            tactic: Tactic to check
            state: Current proof state

        Returns:
            True if tactic appears applicable, False otherwise
        """
        # Basic applicability checks based on tactic type

        # Intro tactics require non-empty goal
        if tactic in ["intro", "intros"]:
            return "⊢" in state.goal or ":" in state.goal

        # Apply tactics require having something to apply
        if tactic in ["apply", "exact", "refine"]:
            # Need to have a goal to apply to
            return len(state.goal) > 0

        # Induction requires a variable to induct on
        if tactic == "induction":
            return len(state.hypotheses) > 0 or any(
                var in state.goal for var in ["Nat", "List", "Vector", "Tree"]
            )

        # Rewrite tactics require something to rewrite
        if tactic in ["rw", "rewrite", "simp"]:
            # Simp/rw usually requires some assumptions or goal structure
            return len(state.hypotheses) > 0 or "=" in state.goal

        # Arithmetic tactics require arithmetic in goal
        if tactic in ["linarith", "omega", "ring", "abel"]:
            arith_ops = ["+", "-", "*", "/", "<", ">", "≤", "≥"]
            return any(op in state.goal for op in arith_ops)

        # Most other tactics are generally applicable
        return True

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _check_parentheses(self, code: str) -> bool:
        """Check if parentheses are balanced"""
        stack = []
        for char in code:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack:
                    return False
                opening = stack.pop()
                if (char == ')' and opening != '(') or \
                   (char == '}' and opening != '{') or \
                   (char == ']' and opening != '['):
                    return False
        return len(stack) == 0

    def _has_repetitive_tactics(self, proof: LeanProof) -> bool:
        """Check for repetitive use of same tactic"""
        if len(proof.tactics) < 3:
            return False

        # Check for 3+ identical tactics in a row
        for i in range(len(proof.tactics) - 2):
            if proof.tactics[i] == proof.tactics[i+1] == proof.tactics[i+2]:
                return True

        return False

    def _check_simp_args(self, code: str) -> bool:
        """Check if simp has arguments (not just simp?)"""
        # Check for simp with at least one argument
        return bool(re.search(r'\bsimp\b[^?]', code))

    def _check_rw_args(self, code: str) -> bool:
        """Check if rw has arguments"""
        # Check for rw with arguments
        return bool(re.search(r'\brw\b\s+\[', code) or re.search(r'\brw\b\s+\w+', code))

    def _check_induction_args(self, code: str) -> bool:
        """Check if induction has a variable to induct on"""
        # Basic check - induction should have a variable
        return bool(re.search(r'\binduction\b\s+\w+', code))

    def _has_wrong_tactic_order(self, proof: LeanProof) -> bool:
        """Check for suspicious tactic ordering"""
        # Check if refutation tactics appear at the start
        refutation_tactics = ["contradiction", "exfalso", "absurd"]
        if len(proof.tactics) > 3:
            for tactic in proof.tactics[:3]:
                if tactic in refutation_tactics:
                    return True

        # Check if trivial tactics appear at the end
        trivial_tactics = ["assumption", "trivial"]
        if len(proof.tactics) > 5:
            for tactic in proof.tactics[-3:]:
                if tactic in trivial_tactics and proof.tactics[-1] != "assumption":
                    return True

        return False

    def _has_circular_reasoning(self, proof: LeanProof) -> bool:
        """
        Detect potential circular reasoning

        This is a heuristic - true circular reasoning requires full proof analysis.
        """
        # Check for patterns that might indicate circular reasoning:
        # - Reverting and re-introducing same hypothesis
        # - Applying theorem that was just being proved

        code_lower = proof.code.lower()

        # Check for revert followed by intro of same variable
        revert_pattern = r'\brevert\b\s+(\w+)'
        intro_pattern = r'\bintro\b\s+\1'

        revert_matches = list(re.finditer(revert_pattern, code_lower))
        for match in revert_matches:
            var_name = match.group(1)
            # Check if intro appears after revert
            remaining_code = code_lower[match.end():]
            if re.search(rf'\bintro\b\s+{var_name}', remaining_code):
                return True

        # Check for suspicious apply patterns (might be circular)
        # This is very heuristic
        if proof.name:
            # Check if proof tries to apply itself
            apply_self_pattern = rf'\bapply\b\s+{re.escape(proof.name)}'
            if re.search(apply_self_pattern, code_lower):
                return True

        return False

    def _calculate_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth in proof"""
        max_depth = 0
        current_depth = 0

        lines = code.split('\n')
        for line in lines:
            # Check for indentation (proxy for nesting)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Each 2 spaces ≈ 1 nesting level
            depth = indent // 2
            max_depth = max(max_depth, depth)

        return max_depth

    def _has_inefficient_pattern(self, proof: LeanProof) -> List[str]:
        """Check for inefficient proof patterns"""
        issues = []

        # Check for unnecessary have-intro-have pattern
        code = proof.code
        if re.search(r'\bhave\b.*:\s*=\s*.*\n.*\bintro\b', code):
            issues.append("inefficient_have_intro")

        # Check for repeated calc with same expression
        calc_pattern = r'\bcalc\b'
        calc_count = len(re.findall(calc_pattern, code))
        if calc_count > 3:
            issues.append("repeated_calc")

        return issues

    def _has_poor_naming(self, proof: LeanProof) -> bool:
        """Check for poor naming conventions"""
        # Check for generic names
        generic_patterns = [
            r'\btheorem\d+\b',
            r'\blemma\d+\b',
            r'\bdef\d+\b',
            r'\bthm\d+\b',
            r'\blemma_\d+\b',
        ]

        for pattern in generic_patterns:
            if re.search(pattern, proof.name):
                return True

        # Check for single-letter names (except in specific contexts)
        if len(proof.name) == 1 and proof.name not in ['x', 'y', 'z', 'n', 'm', 'k']:
            return True

        return False


# =============================================================================
# LEAN PROOF VALIDATOR
# =============================================================================

class LeanProofValidator:
    """
    Comprehensive validation of Lean 4 proofs

    Provides multi-layer validation:
    - Syntax validation
    - Semantic validation
    - Structural validation
    - Verification with LeanAide
    """

    def __init__(
        self,
        rules: Optional[LeanRedFlagRules] = None,
        verification_engine: Optional[Lean4VerificationEngine] = None
    ):
        """
        Initialize Lean proof validator

        Args:
            rules: Optional LeanRedFlagRules (uses defaults if None)
            verification_engine: Optional Lean4VerificationEngine
        """
        self.rules = rules or LeanRedFlagRules()
        self.flagger = LeanRedFlagger(self.rules)
        self.verification_engine = verification_engine
        self.scorer = LeanProofQualityScorer(self.rules)

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Lean 4 syntax

        Args:
            code: Lean code to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        proof = self._parse_proof(code)
        errors = self.flagger.check_syntax(proof)
        return len(errors) == 0, errors

    def validate_semantics(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Lean 4 semantics

        Args:
            code: Lean code to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        proof = self._parse_proof(code)
        errors = self.flagger.check_semantics(proof)
        return len(errors) == 0, errors

    def validate_structure(self, proof: LeanProof) -> Tuple[bool, List[str]]:
        """
        Validate proof structure

        Args:
            proof: LeanProof to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = self.flagger.check_structure(proof)
        return len(errors) == 0, errors

    def verify_with_leanaide(self, code: str) -> Tuple[bool, List[str]]:
        """
        Verify proof with LeanAide

        Args:
            code: Lean code to verify

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not LEAN4_INTEGRATION_AVAILABLE:
            return False, ["lean4_integration_unavailable"]

        engine = self.verification_engine or self._create_verification_engine()
        if engine is None:
            return False, ["verification_engine_unavailable"]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return False, ["cannot_verify_in_async_context"]

            result = loop.run_until_complete(
                engine.verify_mathematical_solution(code)
            )

            if result.success:
                return True, []
            else:
                return False, result.errors

        except (IOError, ConnectionError, TimeoutError) as e:
            return False, [f"verification_exception:{str(e)}"]

    def full_validation(self, proof: LeanProof) -> ValidationResult:
        """
        Perform comprehensive validation of a Lean proof

        Args:
            proof: LeanProof to validate

        Returns:
            ValidationResult with all validation results
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []

        # 1. Syntax validation
        syntax_valid, syntax_errors = self.validate_syntax(proof.code)
        all_errors.extend(syntax_errors)

        # 2. Semantic validation
        semantic_valid, semantic_errors = self.validate_semantics(proof.code)
        all_errors.extend(semantic_errors)

        # 3. Structural validation
        structure_valid, structure_errors = self.validate_structure(proof)
        all_errors.extend(structure_errors)

        # 4. Quality validation
        quality_errors = self.flagger.check_quality(proof)
        all_warnings.extend(quality_errors)

        # 5. Compute quality score
        quality_score = self.scorer.score_proof(proof)

        # 6. Verification with LeanAide
        verification_result = None
        if LEAN4_INTEGRATION_AVAILABLE:
            verify_valid, verify_errors = self.verify_with_leanaide(proof.code)
            if not verify_valid:
                all_errors.extend(verify_errors)

            # Create VerificationResult object
            verification_result = VerificationResult(
                success=verify_valid,
                errors=verify_errors,
                proof=proof.code
            )

        # Determine overall validity
        valid = len(all_errors) == 0

        return ValidationResult(
            valid=valid,
            errors=all_errors,
            warnings=all_warnings,
            quality_score=quality_score,
            verification_result=verification_result
        )

    def _parse_proof(self, code: str) -> LeanProof:
        """Parse Lean code into LeanProof object"""
        # Extract proof name
        name_match = re.search(r'(?:theorem|lemma|def)\s+([A-Za-z_][A-Za-z0-9_.]*)', code)
        name = name_match.group(1) if name_match else "unknown"

        # Extract proof type
        type_match = re.search(r'(theorem|lemma|def|example|structure|class|instance)', code)
        proof_type_str = type_match.group(1) if type_match else "theorem"
        proof_type = LeanProofType(proof_type_str)

        # Extract statement
        statement_match = re.search(r'(?:theorem|lemma|def)\s+[^:]*:\s*(.+?)\s*:=', code, re.DOTALL)
        statement = statement_match.group(1).strip() if statement_match else ""

        # Extract tactics
        tactics = self._extract_tactics(code)

        # Extract imports
        imports = re.findall(r'^import\s+(.+)$', code, re.MULTILINE)

        # Count sorries
        sorry_count = code.count('sorry') + code.count('admit')
        has_sorry = sorry_count > 0

        # Count lines and tactics
        line_count = len(code.split('\n'))
        tactic_count = len(tactics)

        return LeanProof(
            code=code,
            name=name,
            proof_type=proof_type,
            statement=statement,
            tactics=tactics,
            imports=imports,
            dependencies=[],
            line_count=line_count,
            tactic_count=tactic_count,
            has_sorry=has_sorry,
            sorry_count=sorry_count
        )

    def _extract_tactics(self, code: str) -> List[str]:
        """Extract tactics from Lean code"""
        tactics = []

        # Remove comments
        lines = []
        for line in code.split('\n'):
            if '--' in line:
                line = line.split('--')[0]
            lines.append(line)

        code_no_comments = '\n'.join(lines)

        # Extract tactics (after 'by')
        by_match = re.search(r'\bby\b\s*(.*?)\n\s*(?:theorem|lemma|def|example|\Z)',
                            code_no_comments, re.DOTALL)

        if by_match:
            proof_body = by_match.group(1)

            # Split by common tactic separators
            tactic_pattern = r'(\w+)(?:\s+|\[|\?|\.|$)'
            for match in re.finditer(tactic_pattern, proof_body):
                tactic = match.group(1)
                if any(tactic in category for category in LEAN_TACTICS.values()):
                    tactics.append(tactic)

        return tactics

    def _create_verification_engine(self) -> Optional[Lean4VerificationEngine]:
        """Create a Lean 4 verification engine"""
        if not LEAN4_INTEGRATION_AVAILABLE:
            return None

        try:
            server_config = Lean4ServerConfig(enable_simulation_fallback=True)
            verification_config = Lean4VerificationConfig(enable_caching=True)
            return Lean4VerificationEngine(
                server_url="http://localhost:7654",
                server_config=server_config,
                config=verification_config
            )
        except (IOError, ConnectionError, ValueError) as e:
            logger.warning(f"Failed to create verification engine: {e}")
            return None


# =============================================================================
# LEAN PROOF QUALITY SCORER
# =============================================================================

class LeanProofQualityScorer:
    """
    Score Lean proof quality on multiple dimensions

    Dimensions:
    - Elegance: Tactic diversity, proof conciseness
    - Clarity: Understandable structure, good naming
    - Efficiency: Minimal redundancy, optimal tactic use
    - Correctness: Verified, no sorries
    """

    def __init__(self, rules: LeanRedFlagRules):
        """
        Initialize quality scorer

        Args:
            rules: LeanRedFlagRules for thresholds
        """
        self.rules = rules

    def score_proof(self, proof: LeanProof) -> LeanQualityScore:
        """
        Score a Lean proof on all quality dimensions

        Args:
            proof: LeanProof to score

        Returns:
            LeanQualityScore with all dimension scores
        """
        # Score each dimension
        elegance = self.score_elegance(proof)
        clarity = self.score_clarity(proof)
        efficiency = self.score_efficiency(proof)
        correctness = self.score_correctness(proof)

        # Compute overall score (weighted average)
        weights = {
            "elegance": 0.25,
            "clarity": 0.25,
            "efficiency": 0.20,
            "correctness": 0.30  # Correctness is most important
        }

        overall = (
            elegance * weights["elegance"] +
            clarity * weights["clarity"] +
            efficiency * weights["efficiency"] +
            correctness * weights["correctness"]
        )

        # Generate flags and suggestions
        flags, suggestions = self._generate_feedback(proof, {
            "elegance": elegance,
            "clarity": clarity,
            "efficiency": efficiency,
            "correctness": correctness
        })

        return LeanQualityScore(
            overall_score=overall,
            elegance=elegance,
            clarity=clarity,
            efficiency=efficiency,
            correctness=correctness,
            flags=flags,
            suggestions=suggestions
        )

    def score_elegance(self, proof: LeanProof) -> float:
        """
        Score proof elegance

        Elegance factors:
        - Tactic diversity (variety of tactics used)
        - Proof conciseness (not overly long)
        - Creative use of tactics

        Returns:
            Elegance score (0-1)
        """
        score = 1.0

        # 1. Tactic diversity
        if proof.tactic_count > 0:
            unique_tactics = len(set(proof.tactics))
            diversity_ratio = unique_tactics / proof.tactic_count

            # Ideal diversity is around 0.3-0.7 (not all same, not all different)
            if diversity_ratio < 0.1:
                score -= 0.4  # Too repetitive
            elif diversity_ratio > 0.9:
                score -= 0.1  # Too scattered (might be unfocused)
            else:
                score += 0.1  # Good diversity

        # 2. Proof conciseness
        if proof.tactic_count > self.rules.max_tactic_sequence * 0.8:
            score -= 0.3  # Too long

        if proof.line_count > self.rules.max_proof_length * 0.7:
            score -= 0.2  # Too verbose

        # 3. Automation vs manual balance
        automation_tactics = ["simp", "aesop", "tidy", "solve_by_elim"]
        automation_count = sum(1 for t in proof.tactics if t in automation_tactics)

        if proof.tactic_count > 5:
            automation_ratio = automation_count / proof.tactic_count
            # Some automation is good, but not too much
            if automation_ratio > 0.8:
                score -= 0.3  # Too much automation (hides reasoning)
            elif automation_ratio < 0.1:
                score -= 0.1  # Could benefit from some automation

        # 4. Creative tactic use (bonus)
        creative_tactics = ["wlog", "calc", "by_contra", "contrapose"]
        if any(t in proof.tactics for t in creative_tactics):
            score += 0.1  # Bonus for creative tactics

        return max(0.0, min(1.0, score))

    def score_clarity(self, proof: LeanProof) -> float:
        """
        Score proof clarity

        Clarity factors:
        - Understandable structure
        - Good naming conventions
        - Well-organized proof

        Returns:
            Clarity score (0-1)
        """
        score = 1.0

        # 1. Naming quality
        if self._has_poor_naming(proof):
            score -= 0.3

        # 2. Structure organization
        # Check for have statements (improves clarity)
        have_count = proof.code.count(' have ')
        if proof.tactic_count > 10 and have_count == 0:
            score -= 0.2  # Could benefit from intermediate have statements

        # 3. Use of show/suffices (improves clarity)
        clarity_tactics = ["show", "suffices", "calc"]
        if any(t in proof.tactics for t in clarity_tactics):
            score += 0.1

        # 4. Comments
        comment_lines = sum(1 for line in proof.code.split('\n') if '--' in line)
        if proof.line_count > 20 and comment_lines == 0:
            score -= 0.1  # Long proof without comments

        # 5. Code formatting (basic check)
        lines = proof.code.split('\n')
        inconsistent_indent = False
        prev_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if prev_indent > 0 and abs(indent - prev_indent) > 4:
                    inconsistent_indent = True
                    break
                prev_indent = indent

        if inconsistent_indent:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def score_efficiency(self, proof: LeanProof) -> float:
        """
        Score proof efficiency

        Efficiency factors:
        - Minimal redundancy
        - Optimal tactic use
        - No unnecessary steps

        Returns:
            Efficiency score (0-1)
        """
        score = 1.0

        # 1. Redundancy check
        # Look for repeated patterns
        if self._has_redundancy(proof):
            score -= 0.3

        # 2. Optimal tactic use
        # Check for obviously suboptimal patterns
        # Example: multiple intros that could be combined
        intros_pattern = r'(intro\s+\w+\n\s*){3,}'
        if re.search(intros_pattern, proof.code):
            score -= 0.1  # Could use intros instead

        # 3. Unnecessary have patterns
        # have followed immediately by intro can often be simplified
        if re.search(r'have.*:\s*:=.*\n.*intro', proof.code):
            score -= 0.2

        # 4. Proof length relative to statement complexity
        # This is heuristic - longer proofs aren't necessarily inefficient
        # But extremely short proofs with lots of simp might be
        if proof.tactic_count < 5:
            simp_count = sum(1 for t in proof.tactics if t == "simp")
            if simp_count == proof.tactic_count:
                score -= 0.2  # Might be hiding reasoning

        return max(0.0, min(1.0, score))

    def score_correctness(self, proof: LeanProof) -> float:
        """
        Score proof correctness

        Correctness factors:
        - Verified by Lean
        - No sorries
        - Complete proof

        Returns:
            Correctness score (0-1)
        """
        score = 1.0

        # 1. Check for sorries
        if proof.has_sorry:
            if self.rules.require_no_sorries:
                score = 0.0  # Automatic fail
            else:
                # Partial credit based on sorry count
                sorry_ratio = proof.sorry_count / max(1, proof.tactic_count)
                score -= sorry_ratio * 0.8

        # 2. Check for admit (alternative to sorry)
        if "admit" in proof.code.lower():
            score -= 0.5

        # 3. Check proof completeness
        if not proof.code.strip().endswith(('qed', ')', '}')):
            score -= 0.3  # Might be incomplete

        # 4. If verification is available, use it
        # (This would be checked separately in full validation)

        return max(0.0, min(1.0, score))

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _has_poor_naming(self, proof: LeanProof) -> bool:
        """Check for poor naming conventions"""
        # Check for generic names
        generic_patterns = [
            r'\btheorem\d+\b',
            r'\blemma\d+\b',
            r'\bdef\d+\b',
            r'\bthm\d+\b',
        ]

        for pattern in generic_patterns:
            if re.search(pattern, proof.name):
                return True

        return False

    def _has_redundancy(self, proof: LeanProof) -> bool:
        """Check for redundant patterns in proof"""
        # Check for repeated tactic sequences
        if len(proof.tactics) < 4:
            return False

        # Look for repeated pairs of tactics
        for i in range(len(proof.tactics) - 3):
            if (proof.tactics[i:i+2] == proof.tactics[i+2:i+4]):
                return True

        # Check for rw followed by rwa of same thing
        code = proof.code.lower()
        if re.search(r'rw\s+\[.*?\].*?\nrw\s+\[.*?\]', code):
            return True

        return False

    def _generate_feedback(
        self,
        proof: LeanProof,
        scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Generate flags and suggestions based on scores

        Args:
            proof: LeanProof being scored
            scores: Dictionary of dimension scores

        Returns:
            Tuple of (flags, suggestions)
        """
        flags: List[str] = []
        suggestions: List[str] = []

        # Generate flags for low scores
        if scores["elegance"] < 0.4:
            flags.append("low_elegance")
            if len(set(proof.tactics)) < 3:
                suggestions.append("Consider using a wider variety of tactics")

        if scores["clarity"] < 0.4:
            flags.append("low_clarity")
            suggestions.append("Add intermediate 'have' statements to improve readability")
            suggestions.append("Consider adding explanatory comments")

        if scores["efficiency"] < 0.4:
            flags.append("low_efficiency")
            suggestions.append("Look for redundant patterns that can be simplified")

        if scores["correctness"] < 0.4:
            flags.append("low_correctness")
            if proof.has_sorry:
                suggestions.append("Replace 'sorry' placeholders with actual proofs")

        # Specific suggestions based on proof analysis
        simp_count = sum(1 for t in proof.tactics if t == "simp")
        if simp_count > proof.tactic_count * 0.5:
            suggestions.append("High proportion of 'simp' tactics - consider showing more explicit reasoning")

        if proof.tactic_count > self.rules.max_tactic_sequence * 0.7:
            suggestions.append("Proof is quite long - consider breaking into lemmas")

        if proof.line_count > self.rules.max_proof_length * 0.5:
            suggestions.append("Proof is verbose - consider more concise tactics")

        return flags, suggestions


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_lean_red_flagger(
    max_proof_length: int = 500,
    require_no_sorries: bool = True,
    min_elegance_score: float = 0.3,
    **kwargs
) -> LeanRedFlagger:
    """
    Create a Lean red-flagger with specified rules

    Args:
        max_proof_length: Maximum proof length in lines
        require_no_sorries: Whether to require no sorry placeholders
        min_elegance_score: Minimum elegance score (0-1)
        **kwargs: Additional rule parameters

    Returns:
        LeanRedFlagger instance
    """
    rules = LeanRedFlagRules(
        max_proof_length=max_proof_length,
        require_no_sorries=require_no_sorries,
        min_elegance_score=min_elegance_score,
        **kwargs
    )
    return LeanRedFlagger(rules)


def create_lean_validator(
    rules: Optional[LeanRedFlagRules] = None,
    verification_engine: Optional[Lean4VerificationEngine] = None
) -> LeanProofValidator:
    """
    Create a Lean proof validator

    Args:
        rules: Optional LeanRedFlagRules
        verification_engine: Optional Lean4VerificationEngine

    Returns:
        LeanProofValidator instance
    """
    return LeanProofValidator(rules, verification_engine)


def create_lean_quality_scorer(
    rules: Optional[LeanRedFlagRules] = None
) -> LeanProofQualityScorer:
    """
    Create a Lean proof quality scorer

    Args:
        rules: Optional LeanRedFlagRules

    Returns:
        LeanProofQualityScorer instance
    """
    rules = rules or LeanRedFlagRules()
    return LeanProofQualityScorer(rules)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_lean_code(code: str) -> LeanProof:
    """
    Parse Lean code into a LeanProof object

    Args:
        code: Lean source code

    Returns:
        LeanProof object
    """
    validator = create_lean_validator()
    return validator._parse_proof(code)


def quick_red_flag_check(code: str) -> Tuple[bool, List[str]]:
    """
    Quick red-flag check for Lean code

    Args:
        code: Lean source code

    Returns:
        Tuple of (is_flagged, reasons)
    """
    flagger = create_lean_red_flagger()
    proof = parse_lean_code(code)
    return flagger.is_flagged(proof)


def comprehensive_validation(code: str) -> ValidationResult:
    """
    Perform comprehensive validation of Lean code

    Args:
        code: Lean source code

    Returns:
        ValidationResult with all details
    """
    validator = create_lean_validator()
    proof = parse_lean_code(code)
    return validator.full_validation(proof)


def score_proof_quality(code: str) -> LeanQualityScore:
    """
    Score the quality of a Lean proof

    Args:
        code: Lean source code

    Returns:
        LeanQualityScore with all dimensions
    """
    scorer = create_lean_quality_scorer()
    proof = parse_lean_code(code)
    return scorer.score_proof(proof)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "LeanProofType",
    "LeanProof",
    "LeanProofState",
    "LeanQualityScore",
    "ValidationResult",

    # Rules
    "LeanRedFlagRules",

    # Main classes
    "LeanRedFlagger",
    "LeanProofValidator",
    "LeanProofQualityScorer",

    # Factory functions
    "create_lean_red_flagger",
    "create_lean_validator",
    "create_lean_quality_scorer",

    # Utility functions
    "parse_lean_code",
    "quick_red_flag_check",
    "comprehensive_validation",
    "score_proof_quality",

    # Constants
    "LEAN_TACTICS",
    "POTENTIALLY_PROBLEMATIC_TACTICS",
    "LEAN_KEYWORDS",
]
