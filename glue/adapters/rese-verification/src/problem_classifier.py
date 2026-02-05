"""
Problem Classifier for Tiered Verification

Analyzes verification problems and classifies them by:
1. Problem class (constraint sat, theorem proving, optimization, etc.)
2. Complexity (constraint count, quantifier depth, etc.)
3. Domain (algebra, analysis, topology, physics, etc.)

This classification drives adaptive solver selection in the tiered verification system.

Following CLAUDE.md principles:
- Law of Runtime Truth: Classify based on actual problem structure
- Law of Configuration Explicitness: Thresholds via env vars
- Structured Logging: JSON with correlation_id

Author: RESE Team
Created: 2026-02-04
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from .verification_result import ProblemClass, ProblemDomain
except ImportError:
    from verification_result import ProblemClass, ProblemDomain


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ClassifierConfig:
    """Problem classifier configuration"""
    # Complexity thresholds
    max_tier1_constraints: int = 100  # Max constraints for Tier 1 (Z3)
    max_tier2_constraints: int = 1000  # Max constraints for Tier 2 (LeanAide)

    # Quantifier depth thresholds
    max_tier1_quantifier_depth: int = 2  # Max quantifier depth for Tier 1
    max_tier2_quantifier_depth: int = 5  # Max quantifier depth for Tier 2

    # Timeout thresholds (milliseconds)
    tier1_timeout_ms: int = 1000  # 1 second
    tier2_timeout_ms: int = 60000  # 1 minute

    @classmethod
    def from_env(cls) -> 'ClassifierConfig':
        """Load configuration from environment variables"""
        return cls(
            max_tier1_constraints=int(os.getenv("TIER1_MAX_CONSTRAINTS", "100")),
            max_tier2_constraints=int(os.getenv("TIER2_MAX_CONSTRAINTS", "1000")),
            max_tier1_quantifier_depth=int(os.getenv("TIER1_MAX_QUANTIFIER_DEPTH", "2")),
            max_tier2_quantifier_depth=int(os.getenv("TIER2_MAX_QUANTIFIER_DEPTH", "5")),
            tier1_timeout_ms=int(os.getenv("TIER1_TIMEOUT_MS", "1000")),
            tier2_timeout_ms=int(os.getenv("TIER2_TIMEOUT_MS", "60000")),
        )


# =============================================================================
# PROBLEM CLASSIFIER
# =============================================================================

class ProblemClassifier:
    """
    Classifies verification problems for adaptive solver selection.

    Analyzes problem structure to determine:
    1. Problem class (type of problem)
    2. Problem domain (mathematical domain)
    3. Complexity metrics (difficulty assessment)
    4. Recommended tier (initial tier to try)
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize problem classifier

        Args:
            config: Classifier configuration (defaults to environment variables)
        """
        self.config = config or ClassifierConfig.from_env()
        self.logger = logging.getLogger("rese.verification.classifier")

        # Domain-specific patterns
        self._init_domain_patterns()

    def _init_domain_patterns(self):
        """Initialize domain-specific pattern recognizers"""
        self.domain_patterns = {
            ProblemDomain.ARITHMETIC: [
                r'\b\d+\s*[\+\-\*\/]\s*\d+',  # Arithmetic operations
                r'\b(sums?|products?|quotients?)\b',  # Arithmetic keywords
                r'\b(linarith|nlinarith)\b',  # Lean tactics
                r'\b\d+\s*\+\s*\d+\s*=\s*\d+',  # Simple equations like "2 + 2 = 4"
                r'\bprove\s+that\b',  # "Prove that" for arithmetic theorems
            ],
            ProblemDomain.ALGEBRA: [
                r'\b(polynomial|inequality)\b',  # Removed "equation" as it's too generic
                r'\b(algebraic|factor(iz|s)ation)\b',
                r'\b(ring|field|group)\b',
                r'\b[a-z]\^\d+',  # Polynomials like x^2 (with exponent)
                r'\b[a-z]\s*[\+\-\*]\s*[a-z]',  # Variable operations like x + y
                r'\bvariables?\b',  # Variable keywords
            ],
            ProblemDomain.ANALYSIS: [
                r'\b(limit|continuity|derivative|integral)\b',
                r'\b(converges?|diverges?)\b',
                r'\b(real\s+analysis|measure)\b',
            ],
            ProblemDomain.TOPOLOGY: [
                r'\b(topological|compact|connected)\b',
                r'\b(continuum|manifold)\b',
                r'\b(homeomorph|isomorph)\b',
            ],
            ProblemDomain.LOGIC: [
                r'\bforall\b.*\bexists\b',  # Nested quantifiers
                r'\bexists\b.*\bforall\b',  # Nested quantifiers
                r'\b(forall|exists|quantifier)\b',  # Quantifiers
                r'\bP\(|Q\(|R\(',  # Predicate notation P(x), Q(x), etc.
                r'\b(proposition|predicate)\b',
                r'\b(tautology|contradiction)\b',
            ],
            ProblemDomain.PHYSICS: [
                r'\b(energy|force|momentum)\b',
                r'\b(velocity|acceleration)\b',
                r'\b(equation\s+of\s+motion)\b',
            ],
            ProblemDomain.GEOMETRY: [
                r'\b(triangle|circle|polygon)\b',
                r'\b(angle|perpendicular|parallel)\b',
                r'\b(euclidean|non\-euclidean)\b',
            ],
        }

        # Problem class patterns
        self.class_patterns = {
            ProblemClass.CONSTRAINT_SAT: [
                r'\b(satisfiability|satisfy|constraint)\b',
                r'\bfind\s+(a|an|all)\s+\w+',
                r'\bexists?\s+.*?\s+such\s+that\b',
                r'\bfind\s+\w+\s+such\s+that\b',  # "Find x such that"
            ],
            ProblemClass.THEOREM_PROVING: [
                r'\b(prove|theorem|lemma|corollary)\b',
                r'\b(show\s+that)\b',
                r'\b(demonstrate)\b',
            ],
            ProblemClass.OPTIMIZATION: [
                r'\b(minimize|maximize|optimi[sz]e)\b',
                r'\b(objective\s+function)\b',
                r'\b(optimal|minimum|maximum)\b',
            ],
            ProblemClass.CONTRADICTION_DETECTION: [
                r'\b(contradiction|inconsistent|conflict)\b',
                r'\b(unsatisfiable|unsat)\b',
            ],
            ProblemClass.MODEL_VALIDATION: [
                r'\b(validate\s+model|model\s+checking)\b',
                r'\b(invariant|property)\b',
            ],
        }

    def classify(
        self,
        problem: str,
        constraints: Optional[List[Any]] = None,
        variables: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ProblemClass, ProblemDomain, Dict[str, Any]]:
        """
        Classify a verification problem.

        Args:
            problem: Problem statement (natural language or formal)
            constraints: List of constraints (if available)
            variables: List of variables (if available)
            metadata: Additional metadata

        Returns:
            Tuple of (problem_class, problem_domain, complexity_metrics)
        """
        # Lowercase for pattern matching
        problem_lower = problem.lower()

        # Classify problem class
        problem_class = self._classify_class(problem_lower, constraints)

        # Classify domain
        problem_domain = self._classify_domain(problem_lower)

        # Compute complexity metrics
        complexity = self._compute_complexity(
            problem,
            constraints,
            variables,
            metadata
        )

        self.logger.info({
            "level": "info",
            "component": "ProblemClassifier",
            "message": "Problem classified",
            "problem_class": problem_class.value,
            "problem_domain": problem_domain.value,
            "complexity": complexity,
        })

        return problem_class, problem_domain, complexity

    def _classify_class(self, problem: str, constraints: Optional[List[Any]]) -> ProblemClass:
        """Classify problem class"""
        scores = {}

        for problem_class, patterns in self.class_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, problem, re.IGNORECASE))
                score += matches
            scores[problem_class] = score

        # Get class with highest score
        best_class = max(scores, key=scores.get)

        # If no patterns matched, check constraints
        if scores[best_class] == 0:
            if constraints and len(constraints) > 0:
                return ProblemClass.CONSTRAINT_SAT
            else:
                return ProblemClass.THEOREM_PROVING

        return best_class

    def _classify_domain(self, problem: str) -> ProblemDomain:
        """Classify problem domain"""
        scores = {}

        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, problem, re.IGNORECASE))
                score += matches
            scores[domain] = score

        # Get domain with highest score
        # In case of ties, prefer more specific domains in this order:
        # LOGIC > ARITHMETIC > ALGEBRA > others > GENERAL
        priority_order = [
            ProblemDomain.LOGIC,
            ProblemDomain.ARITHMETIC,
            ProblemDomain.ALGEBRA,
            ProblemDomain.ANALYSIS,
            ProblemDomain.TOPOLOGY,
            ProblemDomain.PHYSICS,
            ProblemDomain.GEOMETRY,
            ProblemDomain.GENERAL,
        ]

        # Filter domains with max score
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return ProblemDomain.GENERAL

        # Get domains with max score
        best_domains = [d for d, s in scores.items() if s == max_score]

        # Return highest priority domain
        for domain in priority_order:
            if domain in best_domains:
                return domain

        return ProblemDomain.GENERAL

    def _compute_complexity(
        self,
        problem: str,
        constraints: Optional[List[Any]],
        variables: Optional[List[Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute complexity metrics.

        Returns:
            Dict with complexity metrics:
            - constraint_count: Number of constraints
            - variable_count: Number of variables
            - quantifier_depth: Maximum quantifier nesting depth
            - has_quantifiers: Whether problem has quantifiers
            - has_nonlinear: Whether problem has nonlinear operations
            - has_arrays: Whether problem uses arrays
            - estimated_tier: Recommended starting tier (1, 2, or 3)
        """
        constraint_count = len(constraints) if constraints else 0
        variable_count = len(variables) if variables else 0

        # Analyze problem for complexity indicators
        quantifier_depth = self._count_quantifier_depth(problem)
        has_quantifiers = quantifier_depth > 0
        has_nonlinear = self._has_nonlinear(problem)
        has_arrays = self._has_arrays(problem)

        # Estimate recommended tier
        estimated_tier = self._estimate_tier(
            constraint_count,
            quantifier_depth,
            has_quantifiers,
            has_nonlinear,
            has_arrays
        )

        return {
            "constraint_count": constraint_count,
            "variable_count": variable_count,
            "quantifier_depth": quantifier_depth,
            "has_quantifiers": has_quantifiers,
            "has_nonlinear": has_nonlinear,
            "has_arrays": has_arrays,
            "estimated_tier": estimated_tier,
        }

    def _count_quantifier_depth(self, problem: str) -> int:
        """Count maximum quantifier nesting depth"""
        quantifiers = ['forall', 'exists', '∀', '∃', 'for all', 'there exists']
        max_depth = 0
        current_depth = 0

        # Simple depth counting (not perfect, but good enough)
        tokens = problem.split()
        for token in tokens:
            if any(q in token.lower() for q in quantifiers):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif current_depth > 0:
                # Naive: assume each clause closes one level
                if token in ['.', ':', ',']:
                    current_depth = max(0, current_depth - 1)

        return max_depth

    def _has_nonlinear(self, problem: str) -> bool:
        """Check if problem has nonlinear operations"""
        nonlinear_patterns = [
            r'\*\*',  # Exponentiation
            r'\^',    # Power
            r'\bsin\b', r'\bcos\b', r'\btan\b',  # Trigonometric
            r'\blog\b', r'\bexp\b',  # Logarithmic/exponential
            r'\bsqrt\b',  # Square root
            r'\/\s*[a-zA-Z]',  # Division by variable
        ]

        return any(re.search(pattern, problem) for pattern in nonlinear_patterns)

    def _has_arrays(self, problem: str) -> bool:
        """Check if problem uses arrays"""
        array_patterns = [
            r'\barray\b',
            r'\[\]',
            r'\bselect\b',
            r'\bstore\b',
        ]

        return any(re.search(pattern, problem, re.IGNORECASE) for pattern in array_patterns)

    def _estimate_tier(
        self,
        constraint_count: int,
        quantifier_depth: int,
        has_quantifiers: bool,
        has_nonlinear: bool,
        has_arrays: bool
    ) -> int:
        """
        Estimate recommended starting tier.

        Returns:
            1, 2, or 3 (tier number)
        """
        # Tier 3 (Lean 4) for:
        # - Very high constraint count
        # - Deep quantifier nesting
        # - Complex nonlinear operations
        if (
            constraint_count > self.config.max_tier2_constraints or
            quantifier_depth > self.config.max_tier2_quantifier_depth or
            (has_nonlinear and quantifier_depth > 2)
        ):
            return 3

        # Tier 2 (LeanAide) for:
        # - Medium constraint count
        # - Some quantifiers
        # - Nonlinear operations
        if (
            constraint_count > self.config.max_tier1_constraints or
            quantifier_depth > self.config.max_tier1_quantifier_depth or
            has_quantifiers or
            has_nonlinear or
            has_arrays
        ):
            return 2

        # Tier 1 (Z3) for:
        # - Simple constraint problems
        # - No quantifiers
        # - Linear operations
        return 1

    def should_escalate(
        self,
        current_tier: int,
        constraint_count: int,
        execution_time_ms: float,
        status: str,
        quantifier_depth: int = 0
    ) -> Tuple[bool, str]:
        """
        Determine if should escalate to next tier.

        Args:
            current_tier: Current tier (1, 2, or 3)
            constraint_count: Number of constraints
            execution_time_ms: Execution time in milliseconds
            status: Current status (sat, unsat, unknown, timeout, error)
            quantifier_depth: Quantifier nesting depth

        Returns:
            Tuple of (should_escalate, reason)
        """
        # Never escalate from Tier 3 (it's the final tier)
        if current_tier >= 3:
            return False, "Already at final tier"

        # Escalate on timeout
        timeout = (
            (current_tier == 1 and execution_time_ms > self.config.tier1_timeout_ms) or
            (current_tier == 2 and execution_time_ms > self.config.tier2_timeout_ms)
        )
        if timeout:
            return True, f"Tier {current_tier} timeout ({execution_time_ms:.0f}ms)"

        # Escalate on unknown status
        if status in ["unknown", "error"]:
            return True, f"Tier {current_tier} returned {status}"

        # Escalate if too complex for current tier
        if current_tier == 1:
            if constraint_count > self.config.max_tier1_constraints:
                return True, f"Too many constraints for Tier 1 ({constraint_count})"
            if quantifier_depth > self.config.max_tier1_quantifier_depth:
                return True, f"Quantifier depth too high for Tier 1 ({quantifier_depth})"

        elif current_tier == 2:
            if constraint_count > self.config.max_tier2_constraints:
                return True, f"Too many constraints for Tier 2 ({constraint_count})"
            if quantifier_depth > self.config.max_tier2_quantifier_depth:
                return True, f"Quantifier depth too high for Tier 2 ({quantifier_depth})"

        # Don't escalate
        return False, "Problem suitable for current tier"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_problem(
    problem: str,
    constraints: Optional[List[Any]] = None,
    config: Optional[ClassifierConfig] = None
) -> Tuple[ProblemClass, ProblemDomain, Dict[str, Any]]:
    """
    Convenience function to classify a problem.

    Args:
        problem: Problem statement
        constraints: Optional list of constraints
        config: Optional classifier configuration

    Returns:
        Tuple of (problem_class, problem_domain, complexity_metrics)
    """
    classifier = ProblemClassifier(config)
    return classifier.classify(problem, constraints)


def should_escalate(
    current_tier: int,
    constraint_count: int,
    execution_time_ms: float,
    status: str,
    config: Optional[ClassifierConfig] = None
) -> Tuple[bool, str]:
    """
    Convenience function to check if should escalate.

    Args:
        current_tier: Current tier number
        constraint_count: Number of constraints
        execution_time_ms: Execution time
        status: Current status
        config: Optional classifier configuration

    Returns:
        Tuple of (should_escalate, reason)
    """
    classifier = ProblemClassifier(config)
    return classifier.should_escalate(
        current_tier,
        constraint_count,
        execution_time_ms,
        status
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ClassifierConfig",
    "ProblemClassifier",
    "classify_problem",
    "should_escalate",
]
