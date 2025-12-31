"""
LeanAide Evolutionary Proof Strategies Library

A comprehensive library of proof strategies for Lean 4 that supports evolutionary
exploration and automated proof search. This library provides:

- Tactic library with categorized Lean 4 tactics
- Proof templates for common patterns
- Strategy selection based on theorem characteristics
- Strategy mutation and combination for evolution
- Success rate tracking and learning

Author: LeanAide Evolutionary System
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Union, Callable, Any, Set
)
from enum import Enum
import json
import re
import random
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ProofDifficulty(Enum):
    """Difficulty levels for proofs"""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    RESEARCH = "research"


class TacticCategory(Enum):
    """Categories of Lean 4 tactics"""
    SIMPLIFICATION = "simplification"
    REWRITE = "rewrite"
    INDUCTIVE = "inductive"
    LOGICAL = "logical"
    ALGEBRAIC = "algebraic"
    ARITHMETIC = "arithmetic"
    AUTOMATED = "automated"
    STRUCTURAL = "structural"
    CONSTRUCTIVE = "constructive"
    CLASSICAL = "classical"
    ADVANCED = "advanced"


class StrategyCategory(Enum):
    """High-level strategy categories"""
    INDUCTION = "induction"
    ALGEBRAIC = "algebraic"
    LOGICAL = "logical"
    COMPUTATIONAL = "computational"
    AUTOMATED_SEARCH = "automated_search"
    CASE_ANALYSIS = "case_analysis"
    CONTRADICTORY = "contradictory"
    CONSTRUCTIVE = "constructive"
    HYBRID = "hybrid"


@dataclass
class TacticMetadata:
    """Metadata for a Lean tactic"""
    name: str
    category: TacticCategory
    description: str
    success_rate: float = 0.5
    usage_count: int = 0
    avg_time: float = 0.0
    difficulty_range: Tuple[ProofDifficulty, ProofDifficulty] = (ProofDifficulty.EASY, ProofDifficulty.HARD)
    requires_context: List[str] = field(default_factory=list)
    produces_goals: bool = False
    is_safe: bool = False  # Safe tactics won't backtrack
    examples: List[str] = field(default_factory=list)

    def record_usage(self, success: bool, time_taken: float = 0.0) -> None:
        """Record usage and update success rate"""
        self.usage_count += 1
        if success:
            # Update moving average of success rate
            alpha = 0.1  # Learning rate
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 1.0
        else:
            alpha = 0.1
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 0.0

        # Update average time
        if self.usage_count == 1:
            self.avg_time = time_taken
        else:
            self.avg_time = 0.9 * self.avg_time + 0.1 * time_taken


@dataclass
class ProofContext:
    """Context information for a theorem to be proven"""
    theorem_statement: str
    goal_type: str  # e.g., "∀ n : Nat, n + 0 = n"
    hypotheses: List[str] = field(default_factory=list)
    local_context: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    difficulty: ProofDifficulty = ProofDifficulty.MEDIUM
    domain: str = "general"  # algebra, analysis, combinatorics, etc.
    requires_classical: bool = False
    is_constructive: bool = False


@dataclass
class LeanProof:
    """A complete Lean proof"""
    tactic_sequence: List[str]
    intermediate_goals: List[List[str]] = field(default_factory=list)
    proof_script: str = ""
    success: bool = False
    error_message: str = ""
    time_taken: float = 0.0
    strategy_used: Optional['LeanProofStrategy'] = None


@dataclass
class StrategyStatistics:
    """Statistics for a proof strategy"""
    total_attempts: int = 0
    successful_proofs: int = 0
    failed_proofs: int = 0
    avg_time: float = 0.0
    avg_tactics_used: float = 0.0
    success_by_difficulty: Dict[ProofDifficulty, int] = field(default_factory=dict)
    last_used: Optional[str] = None  # Timestamp

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_proofs / self.total_attempts

    def record_attempt(self, success: bool, time_taken: float, tactics_used: int, difficulty: ProofDifficulty) -> None:
        """Record a proof attempt"""
        self.total_attempts += 1
        if success:
            self.successful_proofs += 1
        else:
            self.failed_proofs += 1

        # Update averages
        if self.total_attempts == 1:
            self.avg_time = time_taken
            self.avg_tactics_used = tactics_used
        else:
            self.avg_time = 0.9 * self.avg_time + 0.1 * time_taken
            self.avg_tactics_used = 0.9 * self.avg_tactics_used + 0.1 * tactics_used

        # Track by difficulty
        if difficulty not in self.success_by_difficulty:
            self.success_by_difficulty[difficulty] = 0
        if success:
            self.success_by_difficulty[difficulty] += 1


# ============================================================================
# Lean Tactic Library
# ============================================================================

class LeanTacticLibrary:
    """
    Comprehensive library of Lean 4 tactics with metadata.
    Categorized by purpose and includes usage statistics.
    """

    def __init__(self):
        self.tactics: Dict[str, TacticMetadata] = {}
        self._initialize_library()

    def _initialize_library(self) -> None:
        """Initialize the tactic library with common Lean 4 tactics"""

        # Simplification Tactics
        self._add_tactic(TacticMetadata(
            name="simp",
            category=TacticCategory.SIMPLIFICATION,
            description="Simplify goal using simp lemmas",
            success_rate=0.7,
            is_safe=True,
            examples=["simp", "simp [h1, h2]", "simp only [add_zero]"]
        ))

        self._add_tactic(TacticMetadata(
            name="dsimp",
            category=TacticCategory.SIMPLIFICATION,
            description="Simplify using definitional equalities",
            success_rate=0.6,
            is_safe=True,
            examples=["dsimp", "dsimp [f]"]
        ))

        # Rewrite Tactics
        self._add_tactic(TacticMetadata(
            name="rw",
            category=TacticCategory.REWRITE,
            description="Rewrite using equations",
            success_rate=0.75,
            examples=["rw [add_zero]", "rw [← h1]", "rw [add_comm, add_assoc]"]
        ))

        self._add_tactic(TacticMetadata(
            name="rfl",
            category=TacticCategory.REWRITE,
            description="Proof by reflexivity",
            success_rate=0.9,
            is_safe=True,
            examples=["rfl"]
        ))

        # Inductive Tactics
        self._add_tactic(TacticMetadata(
            name="induction",
            category=TacticCategory.INDUCTIVE,
            description="Induction on inductive type",
            success_rate=0.65,
            produces_goals=True,
            examples=["induction n with", "induction x: xs with"]
        ))

        # Logical Tactics
        self._add_tactic(TacticMetadata(
            name="apply",
            category=TacticCategory.LOGICAL,
            description="Apply a theorem to match goal",
            success_rate=0.7,
            produces_goals=True,
            examples=["apply my_lemma", "apply h1"]
        ))

        self._add_tactic(TacticMetadata(
            name="exact",
            category=TacticCategory.LOGICAL,
            description="Exact tactic - provide exact term",
            success_rate=0.8,
            examples=["exact h", "exact ⟨a, b⟩"]
        ))

        self._add_tactic(TacticMetadata(
            name="intro",
            category=TacticCategory.LOGICAL,
            description="Introduce hypothesis",
            success_rate=0.95,
            is_safe=True,
            examples=["intro h", "intros x y z"]
        ))

        self._add_tactic(TacticMetadata(
            name="cases",
            category=TacticCategory.LOGICAL,
            description="Case analysis",
            success_rate=0.75,
            produces_goals=True,
            examples=["cases h", "cases x", "cases n with"]
        ))

        self._add_tactic(TacticMetadata(
            name="constructor",
            category=TacticCategory.LOGICAL,
            description="Apply constructor for inductive types",
            success_rate=0.7,
            examples=["constructor", "constructor 1"]
        ))

        # Algebraic Tactics
        self._add_tactic(TacticMetadata(
            name="ring",
            category=TacticCategory.ALGEBRAIC,
            description="Prove equalities in commutative rings",
            success_rate=0.85,
            is_safe=True,
            examples=["ring"]
        ))

        self._add_tactic(TacticMetadata(
            name="ring_nf",
            category=TacticCategory.ALGEBRAIC,
            description="Normalize ring expressions",
            success_rate=0.8,
            is_safe=True,
            examples=["ring_nf"]
        ))

        # Arithmetic Tactics
        self._add_tactic(TacticMetadata(
            name="linarith",
            category=TacticCategory.ARITHMETIC,
            description="Linear arithmetic decision procedure",
            success_rate=0.85,
            is_safe=True,
            examples=["linarith", "linarith [h1, h2]"]
        ))

        self._add_tactic(TacticMetadata(
            name="norm_num",
            category=TacticCategory.ARITHMETIC,
            description="Normalize numerical expressions",
            success_rate=0.9,
            is_safe=True,
            examples=["norm_num"]
        ))

        self._add_tactic(TacticMetadata(
            name="aesop",
            category=TacticCategory.AUTOMATED,
            description="Automated extensible search for obvious proofs",
            success_rate=0.75,
            examples=["aesop", "aesop (options := { trace := true })"]
        ))

        self._add_tactic(TacticMetadata(
            name="simp_arith",
            category=TacticCategory.ARITHMETIC,
            description="Simplification with arithmetic",
            success_rate=0.8,
            is_safe=True,
            examples=["simp_arith"]
        ))

        self._add_tactic(TacticMetadata(
            name="nlinarith",
            category=TacticCategory.ARITHMETIC,
            description="Non-linear arithmetic",
            success_rate=0.7,
            is_safe=True,
            examples=["nlinarith"]
        ))

        # Structural Tactics
        self._add_tactic(TacticMetadata(
            name="assumption",
            category=TacticCategory.STRUCTURAL,
            description="Use assumption from local context",
            success_rate=0.95,
            is_safe=True,
            examples=["assumption"]
        ))

        self._add_tactic(TacticMetadata(
            name="trivial",
            category=TacticCategory.STRUCTURAL,
            description="Try trivial tactics",
            success_rate=0.7,
            is_safe=True,
            examples=["trivial"]
        ))

        self._add_tactic(TacticMetadata(
            name="decide",
            category=TacticCategory.STRUCTURAL,
            description="Decidable propositions",
            success_rate=0.85,
            is_safe=True,
            examples=["decide"]
        ))

        # Advanced/Automated Tactics
        self._add_tactic(TacticMetadata(
            name="hammer",
            category=TacticCategory.ADVANCED,
            description="External theorem prover integration",
            success_rate=0.65,
            examples=["hammer"]
        ))

        self._add_tactic(TacticMetadata(
            name="grind",
            category=TacticCategory.ADVANCED,
            description="Powerful automated proof search",
            success_rate=0.7,
            examples=["grind"]
        ))

        self._add_tactic(TacticMetadata(
            name="have",
            category=TacticCategory.CONSTRUCTIVE,
            description="Introduce intermediate fact",
            success_rate=0.7,
            produces_goals=True,
            examples=["have h : P := ..."]
        ))

        self._add_tactic(TacticMetadata(
            name="calc",
            category=TacticCategory.ALGEBRAIC,
            description="Calculation mode",
            success_rate=0.75,
            examples=["calc"]
        ))

        # Contradiction Tactics
        self._add_tactic(TacticMetadata(
            name="by_contradiction",
            category=TacticCategory.CLASSICAL,
            description="Proof by contradiction",
            success_rate=0.6,
            produces_goals=True,
            examples=["by_contradiction h"]
        ))

        self._add_tactic(TacticMetadata(
            name="contrapose",
            category=TacticCategory.CLASSICAL,
            description="Contrapositive proof",
            success_rate=0.65,
            examples=["contrapose", "contrapose! h"]
        ))

        # Advanced structural
        self._add_tactic(TacticMetadata(
            name="refine",
            category=TacticCategory.CONSTRUCTIVE,
            description="Refine goal with a term template",
            success_rate=0.7,
            produces_goals=True,
            examples=["refine ?_", "refine ⟨?_, ?_⟩"]
        ))

        self._add_tactic(TacticMetadata(
            name="rcases",
            category=TacticCategory.STRUCTURAL,
            description="Recursive cases",
            success_rate=0.75,
            examples=["rcases h with ⟨x, y⟩"]
        ))

        self._add_tactic(TacticMetadata(
            name="obtain",
            category=TacticCategory.CONSTRUCTIVE,
            description="Obtain witness from existential",
            success_rate=0.7,
            examples=["obtain ⟨x, hx⟩ := h"]
        ))

        # More tactics
        self._add_tactic(TacticMetadata(
            name="funext",
            category=TacticCategory.STRUCTURAL,
            description="Function extensionality",
            success_rate=0.65,
            examples=["funext", "funext x y"]
        ))

        self._add_tactic(TacticMetadata(
            name="ext",
            category=TacticCategory.STRUCTURAL,
            description="Extensionality",
            success_rate=0.7,
            examples=["ext", "ext x"]
        ))

        self._add_tactic(TacticMetadata(
            name="ac_rfl",
            category=TacticCategory.ALGEBRAIC,
            description="Associative-commutative reflexivity",
            success_rate=0.85,
            is_safe=True,
            examples=["ac_rfl"]
        ))

        self._add_tactic(TacticMetadata(
            name="gcongr",
            category=TacticCategory.ARITHMETIC,
            description="Generalized congruence",
            success_rate=0.75,
            examples=["gcongr", "gcongr with"]
        ))

    def _add_tactic(self, metadata: TacticMetadata) -> None:
        """Add a tactic to the library"""
        self.tactics[metadata.name] = metadata

    def get_tactic(self, name: str) -> Optional[TacticMetadata]:
        """Get a tactic by name"""
        return self.tactics.get(name)

    def get_tactics_by_category(self, category: TacticCategory) -> List[TacticMetadata]:
        """Get all tactics in a category"""
        return [t for t in self.tactics.values() if t.category == category]

    def get_safe_tactics(self) -> List[TacticMetadata]:
        """Get all safe tactics (won't backtrack)"""
        return [t for t in self.tactics.values() if t.is_safe]

    def get_best_tactics(self, category: Optional[TacticCategory] = None, min_success_rate: float = 0.5) -> List[TacticMetadata]:
        """Get best performing tactics, optionally by category"""
        tactics = self.tactics.values()
        if category:
            tactics = [t for t in tactics if t.category == category]
        return sorted([t for t in tactics if t.success_rate >= min_success_rate],
                     key=lambda x: x.success_rate, reverse=True)

    def update_tactic_stats(self, name: str, success: bool, time_taken: float = 0.0) -> None:
        """Update statistics for a tactic"""
        tactic = self.get_tactic(name)
        if tactic:
            tactic.record_usage(success, time_taken)

    def export_library(self) -> Dict[str, Any]:
        """Export the library as JSON"""
        return {
            name: {
                "category": meta.category.value,
                "description": meta.description,
                "success_rate": meta.success_rate,
                "usage_count": meta.usage_count,
                "avg_time": meta.avg_time,
                "is_safe": meta.is_safe,
                "examples": meta.examples
            }
            for name, meta in self.tactics.items()
        }

    def import_library(self, data: Dict[str, Any]) -> None:
        """Import library from JSON"""
        for name, attrs in data.items():
            metadata = TacticMetadata(
                name=name,
                category=TacticCategory(attrs["category"]),
                description=attrs["description"],
                success_rate=attrs.get("success_rate", 0.5),
                usage_count=attrs.get("usage_count", 0),
                avg_time=attrs.get("avg_time", 0.0),
                is_safe=attrs.get("is_safe", False),
                examples=attrs.get("examples", [])
            )
            self._add_tactic(metadata)


# ============================================================================
# Proof Templates
# ============================================================================

class LeanProofTemplate(ABC):
    """Abstract base class for proof templates"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def instantiate(self, context: ProofContext, params: Dict[str, Any]) -> List[str]:
        """Instantiate template with given parameters"""
        pass

    @abstractmethod
    def is_applicable(self, context: ProofContext) -> float:
        """Return applicability score (0.0 to 1.0)"""
        pass


class NatInductionTemplate(LeanProofTemplate):
    """Template for natural number induction proofs"""

    def __init__(self):
        super().__init__(
            name="nat_induction",
            description="Natural number induction: prove base case (n=0) and inductive step"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if theorem involves natural numbers and universal quantification"""
        score = 0.0
        stmt = context.theorem_statement.lower()

        # Check for natural numbers
        if any(keyword in stmt for keyword in ["nat", "natural", "ℕ"]):
            score += 0.4

        # Check for universal quantification
        if "∀" in context.theorem_statement or "forall" in stmt:
            score += 0.3

        # Check for inductive structure (recursion, successor, etc.)
        if any(keyword in stmt for keyword in ["succ", "+", "*", "factorial", "fibonacci"]):
            score += 0.3

        return score

    def instantiate(self, context: ProofContext, params: Dict[str, Any]) -> List[str]:
        """Generate induction proof structure"""
        variable = params.get("variable", "n")
        base_case_tac = params.get("base_case", ["simp"])
        inductive_tac = params.get("inductive_step", ["simp", "ring"])

        return [
            f"induction {variable} with",
            "| zero =>",
            "  " + "\n  ".join(base_case_tac),
            "| succ n ih =>",
            "  " + "\n  ".join(inductive_tac)
        ]


class ListInductionTemplate(LeanProofTemplate):
    """Template for list induction proofs"""

    def __init__(self):
        super().__init__(
            name="list_induction",
            description="List induction: prove base case (nil) and inductive step (cons)"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if theorem involves lists"""
        score = 0.0
        stmt = context.theorem_statement.lower()

        if "list" in stmt or "vector" in stmt:
            score += 0.5

        if any(keyword in stmt for keyword in ["length", "head", "tail", "append", "map", "fold"]):
            score += 0.3

        if "∀" in context.theorem_statement:
            score += 0.2

        return score

    def instantiate(self, context: ProofContext, params: Dict[str, Any]) -> List[str]:
        """Generate list induction proof structure"""
        list_var = params.get("variable", "xs")

        return [
            f"induction {list_var} with",
            "| nil =>",
            "  simp",
            "| cons x xs ih =>",
            "  simp",
            "  sorry"  # Placeholder for actual proof
        ]


class ContradictionTemplate(LeanProofTemplate):
    """Template for proof by contradiction"""

    def __init__(self):
        super().__init__(
            name="contradiction",
            description="Proof by contradiction: assume negation and derive contradiction"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if contradiction is a good approach"""
        score = 0.0
        stmt = context.theorem_statement.lower()

        # Good for inequalities
        if any(op in stmt for op in ["<", ">", "≤", "≥", "ne", "≠"]):
            score += 0.3

        # Good for non-existence claims
        if "not" in stmt or "¬" in context.theorem_statement or "does not exist" in stmt:
            score += 0.4

        # Good for irrationality, primality, etc.
        if any(keyword in stmt for keyword in ["irrational", "prime", "infinite", "unbounded"]):
            score += 0.3

        return score

    def instantiate(self, context: ProofContext, params: Dict[str, Any]) -> List[str]:
        """Generate contradiction proof structure"""
        hypothesis = params.get("hypothesis_name", "h")

        return [
            f"by_contradiction {hypothesis}",
            "push_neg at " + hypothesis,
            "sorry"  # Derive contradiction
        ]


class CalcTemplate(LeanProofTemplate):
    """Template for calculation proofs using calc mode"""

    def __init__(self):
        super().__init__(
            name="calc",
            description="Calculation proof: chain of equalities/inequalities"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if calc is suitable"""
        score = 0.0
        stmt = context.theorem_statement.lower()

        # Good for algebraic manipulations
        if any(op in stmt for op in ["=", "≤", "<", "≥", ">"]):
            score += 0.4

        # Look for expression transformations
        if any(keyword in stmt for keyword in ["simplify", "expand", "rewrite", "transform"]):
            score += 0.3

        # Algebraic domains
        if context.domain in ["algebra", "analysis", "linear_algebra"]:
            score += 0.3

        return score

    def instantiate(self, context: ProofContext, params: Dict[str, Any]) -> List[str]:
        """Generate calc proof structure"""
        num_steps = params.get("num_steps", 3)
        expr = params.get("expression", "_")

        proof_lines = ["calc"]
        for i in range(num_steps):
            if i == 0:
                proof_lines.append(f"  {expr} = _ by ?_")
            elif i < num_steps - 1:
                proof_lines.append(f"  _ = _ by ?_")
            else:
                proof_lines.append(f"  _ = {expr} by ?_")

        return proof_lines


class CasesTemplate(LeanProofTemplate):
    """Template for case analysis proofs"""

    def __init__(self):
        super().__init__(
            name="cases",
            description="Case analysis: split into exhaustive cases"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if cases is suitable"""
        score = 0.0
        stmt = context.theorem_statement.lower()

        # Good for union types
        if any(typ in stmt for typ in ["or", "∨", "xor", "either"]):
            score += 0.5

        # Good for enum-like types
        if any(typ in stmt for typ in ["bool", "option", "sum"]):
            sign += 0.4

        # Look for "either...or" language
        if "either" in stmt or "or" in stmt:
            score += 0.1

        return score

    def instantiate(self, context: ProofContext, params: Dict[str, Any]) -> List[str]:
        """Generate cases proof structure"""
        num_cases = params.get("num_cases", 2)
        hypothesis = params.get("hypothesis", "h")

        proof_lines = [f"cases {hypothesis}"]
        for i in range(num_cases):
            proof_lines.append(f"  · sorry")
            proof_lines.append(f"  · sorry" if i < num_cases - 1 else "")

        return proof_lines


class LeanProofTemplateLibrary:
    """Library of proof templates"""

    def __init__(self):
        self.templates: List[LeanProofTemplate] = [
            NatInductionTemplate(),
            ListInductionTemplate(),
            ContradictionTemplate(),
            CalcTemplate(),
            CasesTemplate()
        ]

    def get_applicable_templates(self, context: ProofContext, min_score: float = 0.3) -> List[Tuple[LeanProofTemplate, float]]:
        """Get applicable templates sorted by score"""
        scored = []
        for template in self.templates:
            score = template.is_applicable(context)
            if score >= min_score:
                scored.append((template, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def get_template(self, name: str) -> Optional[LeanProofTemplate]:
        """Get template by name"""
        for template in self.templates:
            if template.name == name:
                return template
        return None


# ============================================================================
# Proof Strategy Base Class
# ============================================================================

class LeanProofStrategy(ABC):
    """Abstract base class for proof strategies"""

    def __init__(self, name: str, category: StrategyCategory, description: str):
        self.name = name
        self.category = category
        self.description = description
        self.statistics = StrategyStatistics()
        self.tactic_library = LeanTacticLibrary()

    @abstractmethod
    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate a proof using this strategy"""
        pass

    @abstractmethod
    def is_applicable(self, context: ProofContext) -> float:
        """Return applicability score (0.0 to 1.0)"""
        pass

    def get_statistics(self) -> StrategyStatistics:
        """Get strategy statistics"""
        return self.statistics

    def record_result(self, proof: LeanProof, difficulty: ProofDifficulty) -> None:
        """Record proof result for learning"""
        self.statistics.record_attempt(
            success=proof.success,
            time_taken=proof.time_taken,
            tactics_used=len(proof.tactic_sequence),
            difficulty=difficulty
        )


# ============================================================================
# Induction Strategies
# ============================================================================

class NatInductionStrategy(LeanProofStrategy):
    """Natural number induction strategy"""

    def __init__(self):
        super().__init__(
            name="nat_induction",
            category=StrategyCategory.INDUCTION,
            description="Prove statements about natural numbers using induction"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if natural induction is applicable"""
        score = 0.0
        stmt_lower = context.theorem_statement.lower()

        # Check for natural numbers
        if any(kw in stmt_lower for kw in ["nat", "natural", "ℕ"]):
            score += 0.5

        # Check for properties that use induction well
        if any(kw in stmt_lower for kw in ["factorial", "fibonacci", "power", "sum", "product"]):
            score += 0.3

        # Check for recursive structure
        if any(kw in stmt_lower for kw in ["succ", "recursion", "induction"]):
            score += 0.2

        return min(score, 1.0)

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate induction proof"""
        params = params or {}

        # Extract variable name
        variable = params.get("variable", "n")

        # Build tactic sequence
        tactics = [f"induction {variable} with"]

        # Base case
        base_tactics = params.get("base_tactics", ["simp", "norm_num"])
        tactics.append("| zero =>")
        tactics.extend([f"  {tac}" for tac in base_tactics])

        # Inductive step
        step_tactics = params.get("step_tactics", ["simp", "ring"])
        tactics.append("| succ n ih =>")
        tactics.extend([f"  {tac}" for tac in step_tactics])

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class ListInductionStrategy(LeanProofStrategy):
    """List induction strategy"""

    def __init__(self):
        super().__init__(
            name="list_induction",
            category=StrategyCategory.INDUCTION,
            description="Prove statements about lists using structural induction"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if list induction is applicable"""
        score = 0.0
        stmt_lower = context.theorem_statement.lower()

        if "list" in stmt_lower or "vector" in stmt_lower:
            score += 0.6

        if any(kw in stmt_lower for kw in ["append", "map", "fold", "filter", "reverse"]):
            score += 0.4

        return min(score, 1.0)

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate list induction proof"""
        params = params or {}
        variable = params.get("variable", "xs")

        tactics = [
            f"induction {variable} with",
            "| nil =>",
            "  simp",
            "| cons x xs ih =>",
            "  simp [ih]",
            "  sorry"
        ]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class StructInductionStrategy(LeanProofStrategy):
    """General structure induction strategy"""

    def __init__(self):
        super().__init__(
            name="struct_induction",
            category=StrategyCategory.INDUCTION,
            description="Prove statements about inductive types using structural induction"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if structural induction is applicable"""
        # Lower default applicability, used as fallback
        return 0.3

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate structural induction proof"""
        params = params or {}
        type_name = params.get("type_name", "T")
        variable = params.get("variable", "x")

        tactics = [
            f"induction x: {type_name} with",
            "  · sorry",
            "  · sorry"
        ]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class StrongInductionStrategy(LeanProofStrategy):
    """Strong induction strategy"""

    def __init__(self):
        super().__init__(
            name="strong_induction",
            category=StrategyCategory.INDUCTION,
            description="Prove statements using strong induction (complete induction)"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if strong induction is applicable"""
        stmt_lower = context.theorem_statement.lower()

        # Look for cases where strong induction helps
        if any(kw in stmt_lower for kw in ["divides", "prime", "factorization", "well_founded"]):
            return 0.7

        return 0.4

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate strong induction proof"""
        params = params or {}

        tactics = [
            "apply Nat.le_induction",
            "· intro n",
            "  sorry",
            "· intro n ih",
            "  sorry"
        ]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


# ============================================================================
# Algebraic Strategies
# ============================================================================

class CalcStrategy(LeanProofStrategy):
    """Calculation mode strategy"""

    def __init__(self):
        super().__init__(
            name="calc",
            category=StrategyCategory.ALGEBRAIC,
            description="Use calc mode for chain of equalities/inequalities"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if calc is applicable"""
        score = 0.0
        stmt_lower = context.theorem_statement.lower()

        # Algebraic operators
        if any(op in context.theorem_statement for op in ["=", "≤", "<", "≥", ">"]):
            score += 0.4

        # Algebraic domain
        if context.domain in ["algebra", "analysis", "linear_algebra"]:
            score += 0.3

        # Transformation keywords
        if any(kw in stmt_lower for kw in ["simplify", "expand", "factor", "rewrite"]):
            score += 0.3

        return min(score, 1.0)

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate calc proof"""
        params = params or {}

        # Generate calc block
        tactics = ["calc"]
        lhs = params.get("lhs", "_")
        rhs = params.get("rhs", "_")

        tactics.append(f"  {lhs} = _ by ?_")
        tactics.append(f"  _ = {rhs} by ?_")

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class RingStrategy(LeanProofStrategy):
    """Ring tactic strategy"""

    def __init__(self):
        super().__init__(
            name="ring",
            category=StrategyCategory.ALGEBRAIC,
            description="Use ring tactic for algebraic identities in commutative rings"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if ring tactic is applicable"""
        stmt_lower = context.theorem_statement.lower()

        # Look for ring operations
        if any(op in context.theorem_statement for op in ["+", "*"]):
            if any(kw in stmt_lower for kw in ["ring", "algebra", "integer", "natural", "real"]):
                return 0.8

        return 0.5

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate ring-based proof"""
        tactics = ["ring"]

        # Maybe add preprocessing
        if params and params.get("preprocess"):
            tactics = params["preprocess"] + tactics

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class LinarithStrategy(LeanProofStrategy):
    """Linear arithmetic strategy"""

    def __init__(self):
        super().__init__(
            name="linarith",
            category=StrategyCategory.ALGEBRAIC,
            description="Use linarith for linear arithmetic goals"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if linarith is applicable"""
        # Linear inequalities
        if any(op in context.theorem_statement for op in ["<", ">", "≤", "≥"]):
            return 0.8
        return 0.4

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate linarith proof"""
        hypotheses = params.get("hypotheses", []) if params else []

        tactics = []
        if hypotheses:
            tactics.append(f"linarith [{' '.join(hypotheses)}]")
        else:
            tactics.append("linarith")

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class NormNumStrategy(LeanProofStrategy):
    """Normalization strategy"""

    def __init__(self):
        super().__init__(
            name="norm_num",
            category=StrategyCategory.ALGEBRAIC,
            description="Use norm_num for numerical evaluation"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if norm_num is applicable"""
        stmt_lower = context.theorem_statement.lower()

        # Numerical expressions
        if any(kw in stmt_lower for kw in ["=", "+", "-", "*", "^", "factorial"]):
            if any(kw in stmt_lower for kw in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                return 0.9

        return 0.5

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate norm_num proof"""
        tactics = ["norm_num"]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


# ============================================================================
# Logical Strategies
# ============================================================================

class ContradictionStrategy(LeanProofStrategy):
    """Proof by contradiction strategy"""

    def __init__(self):
        super().__init__(
            name="contradiction",
            category=StrategyCategory.CONTRADICTORY,
            description="Prove by contradiction (classical logic)"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if contradiction is applicable"""
        stmt_lower = context.theorem_statement.lower()

        # Non-existence or negation
        if "not" in stmt_lower or "¬" in context.theorem_statement:
            return 0.7

        # Inequalities
        if any(op in context.theorem_statement for op in ["<", ">", "≠"]):
            return 0.6

        # Existence of irrational/prime/infinite
        if any(kw in stmt_lower for kw in ["irrational", "prime", "infinite", "unbounded"]):
            return 0.7

        return 0.3

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate contradiction proof"""
        params = params or {}
        hyp_name = params.get("hypothesis_name", "h")

        tactics = [
            f"by_contradiction {hyp_name}",
            "push_neg at " + hyp_name,
            "sorry"
        ]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class ContrapositiveStrategy(LeanProofStrategy):
    """Contrapositive proof strategy"""

    def __init__(self):
        super().__init__(
            name="contrapositive",
            category=StrategyCategory.LOGICAL,
            description="Prove by contrapositive"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if contrapositive is applicable"""
        stmt_lower = context.theorem_statement.lower()

        # Implication statements
        if "→" in context.theorem_statement or "implies" in stmt_lower:
            return 0.7

        return 0.4

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate contrapositive proof"""
        tactics = [
            "contrapose",
            "sorry"
        ]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class CasesStrategy(LeanProofStrategy):
    """Case analysis strategy"""

    def __init__(self):
        super().__init__(
            name="cases",
            category=StrategyCategory.LOGICAL,
            description="Use case analysis on disjunctions or sum types"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if cases is applicable"""
        stmt_lower = context.theorem_statement.lower()

        # Disjunctions
        if "or" in stmt_lower or "∨" in context.theorem_statement:
            return 0.8

        # Boolean, Option, Sum types
        if any(typ in stmt_lower for typ in ["bool", "option", "either", "sum"]):
            return 0.7

        return 0.4

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate cases proof"""
        params = params or {}
        hypothesis = params.get("hypothesis", "h")
        num_cases = params.get("num_cases", 2)

        tactics = [f"cases {hypothesis}"]
        for _ in range(num_cases):
            tactics.append("· sorry")

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class ExistsStrategy(LeanProofStrategy):
    """Existential witness strategy"""

    def __init__(self):
        super().__init__(
            name="exists",
            category=StrategyCategory.LOGICAL,
            description="Prove existential by providing witness"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if exists is applicable"""
        if "∃" in context.theorem_statement or "exists" in context.theorem_statement.lower():
            return 0.9
        return 0.0

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate existential proof"""
        params = params or {}
        witness = params.get("witness", "_")
        proof = params.get("proof", "sorry")

        tactics = [
            f"use {witness}",
            proof
        ]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


# ============================================================================
# Advanced Strategies
# ============================================================================

class AesopStrategy(LeanProofStrategy):
    """Aesop automated proof search strategy"""

    def __init__(self):
        super().__init__(
            name="aesop",
            category=StrategyCategory.AUTOMATED_SEARCH,
            description="Use Aesop for automated proof search"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Aesop is generally applicable"""
        return 0.6

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate aesop proof"""
        params = params or {}
        options = params.get("options", "")

        if options:
            tactics = [f"aesop ({options})"]
        else:
            tactics = ["aesop"]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class HammerStrategy(LeanProofStrategy):
    """Hammer (external prover) strategy"""

    def __init__(self):
        super().__init__(
            name="hammer",
            category=StrategyCategory.AUTOMATED_SEARCH,
            description="Use external theorem provers via hammer"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Hammer is generally applicable"""
        return 0.5

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate hammer proof"""
        tactics = ["hammer"]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class GrindStrategy(LeanProofStrategy):
    """Grind strategy"""

    def __init__(self):
        super().__init__(
            name="grind",
            category=StrategyCategory.AUTOMATED_SEARCH,
            description="Use grind for powerful automated proof search"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Grind is generally applicable"""
        return 0.6

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate grind proof"""
        tactics = ["grind"]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


class SimpStrategy(LeanProofStrategy):
    """Simplification strategy"""

    def __init__(self):
        super().__init__(
            name="simp",
            category=StrategyCategory.AUTOMATED_SEARCH,
            description="Use simp for simplification-based proofs"
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Simp is often applicable"""
        return 0.7

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate simp proof"""
        params = params or {}
        lemmas = params.get("lemmas", [])

        if lemmas:
            tactics = [f"simp [{' '.join(lemmas)}]"]
        else:
            tactics = ["simp"]

        return LeanProof(
            tactic_sequence=tactics,
            strategy_used=self
        )


# ============================================================================
# Strategy Selector
# ============================================================================

class LeanStrategySelector:
    """
    Selects best proof strategy based on theorem characteristics.
    Uses heuristics and learned success rates.
    """

    def __init__(self):
        self.strategies: List[LeanProofStrategy] = []
        self.tactic_library = LeanTacticLibrary()
        self.template_library = LeanProofTemplateLibrary()
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize available strategies"""
        self.strategies = [
            # Induction strategies
            NatInductionStrategy(),
            ListInductionStrategy(),
            StructInductionStrategy(),
            StrongInductionStrategy(),

            # Algebraic strategies
            CalcStrategy(),
            RingStrategy(),
            LinarithStrategy(),
            NormNumStrategy(),

            # Logical strategies
            ContradictionStrategy(),
            ContrapositiveStrategy(),
            CasesStrategy(),
            ExistsStrategy(),

            # Advanced strategies
            AesopStrategy(),
            HammerStrategy(),
            GrindStrategy(),
            SimpStrategy()
        ]

    def select_strategy(
        self,
        context: ProofContext,
        allow_hybrid: bool = True
    ) -> LeanProofStrategy:
        """
        Select best strategy based on context.
        Returns single best strategy or hybrid if allowed.
        """
        # Score all strategies
        scored_strategies = []
        for strategy in self.strategies:
            applicability = strategy.is_applicable(context)
            success_bonus = strategy.statistics.success_rate * 0.3
            total_score = applicability + success_bonus
            scored_strategies.append((strategy, total_score))

        # Sort by score
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        if not scored_strategies:
            # Fallback to aesop
            return AesopStrategy()

        best_strategy, best_score = scored_strategies[0]

        # Consider hybrid if multiple good strategies
        if allow_hybrid and len(scored_strategies) > 1:
            second_best, second_score = scored_strategies[1]
            if abs(best_score - second_score) < 0.2:  # Close scores
                return self._create_hybrid_strategy([best_strategy, second_best])

        return best_strategy

    def select_top_k_strategies(self, context: ProofContext, k: int = 5) -> List[LeanProofStrategy]:
        """Select top K strategies for ensemble/evolutionary approaches"""
        scored = [(s, s.is_applicable(context)) for s in self.strategies]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:k]]

    def _create_hybrid_strategy(self, strategies: List[LeanProofStrategy]) -> 'HybridStrategy':
        """Create a hybrid strategy from multiple strategies"""
        return HybridStrategy(strategies)

    def recommend_strategy_combination(self, context: ProofContext) -> List[Tuple[LeanProofStrategy, float]]:
        """Recommend combination of strategies with their weights"""
        scored = [(s, s.is_applicable(context)) for s in self.strategies]
        scored = [(s, score) for s, score in scored if score > 0.3]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Normalize weights
        total = sum(score for _, score in scored)
        if total > 0:
            scored = [(s, score / total) for s, score in scored]

        return scored


# ============================================================================
# Hybrid Strategy
# ============================================================================

class HybridStrategy(LeanProofStrategy):
    """Combines multiple strategies"""

    def __init__(self, strategies: List[LeanProofStrategy]):
        # Determine dominant category
        categories = [s.category for s in strategies]
        dominant = max(set(categories), key=categories.count)

        super().__init__(
            name="_hybrid_" + "_".join(s.name for s in strategies),
            category=StrategyCategory.HYBRID,
            description=f"Hybrid of: {', '.join(s.name for s in strategies)}"
        )

        self.strategies = strategies

    def is_applicable(self, context: ProofContext) -> float:
        """Return max applicability of component strategies"""
        return max(s.is_applicable(context) for s in self.strategies)

    def generate_proof(self, context: ProofContext, params: Dict[str, Any] = None) -> LeanProof:
        """Generate proof by trying strategies in sequence"""
        all_tactics = []

        for strategy in self.strategies:
            proof = strategy.generate_proof(context, params)
            all_tactics.extend(proof.tactic_sequence)

        return LeanProof(
            tactic_sequence=all_tactics,
            strategy_used=self
        )


# ============================================================================
# Strategy Mutator
# ============================================================================

class LeanStrategyMutator:
    """
    Mutation operations on strategies for evolutionary exploration.
    """

    def __init__(self, selector: LeanStrategySelector):
        self.selector = selector

    def mutate_strategy(self, strategy: LeanProofStrategy, mutation_rate: float = 0.3) -> LeanProofStrategy:
        """Apply random mutations to a strategy"""
        mutation_type = random.choice([
            "tactic_substitution",
            "tactic_reordering",
            "parameter_perturbation",
            "strategy_hybridization",
            "template_injection"
        ])

        if mutation_type == "tactic_substitution":
            return self._substitute_tactic(strategy)
        elif mutation_type == "tactic_reordering":
            return self._reorder_tactics(strategy)
        elif mutation_type == "parameter_perturbation":
            return self._perturb_parameters(strategy)
        elif mutation_type == "strategy_hybridization":
            return self._hybridize_strategy(strategy)
        else:  # template_injection
            return self._inject_template(strategy)

    def _substitute_tactic(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Substitute a tactic with a similar one"""
        # This is a simplified version - would need more sophistication
        return strategy

    def _reorder_tactics(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Reorder tactics in the strategy"""
        # Would modify tactic order
        return strategy

    def _perturb_parameters(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Perturb strategy parameters"""
        # Would adjust parameters
        return strategy

    def _hybridize_strategy(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Create hybrid with another strategy"""
        # Pick random other strategy
        other = random.choice([s for s in self.selector.strategies if s != strategy])
        return HybridStrategy([strategy, other])

    def _inject_template(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Inject a proof template into the strategy"""
        # Would use template library
        return strategy

    def crossover_strategies(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> LeanProofStrategy:
        """Crossover two strategies to create offspring"""
        return HybridStrategy([parent1, parent2])


# ============================================================================
# Strategy Combiner
# ============================================================================

class LeanStrategyCombiner:
    """Combines multiple strategies effectively"""

    @staticmethod
    def sequential(strategies: List[LeanProofStrategy]) -> LeanProofStrategy:
        """Combine strategies to run sequentially"""
        return HybridStrategy(strategies)

    @staticmethod
    def parallel(strategies: List[LeanProofStrategy]) -> LeanProofStrategy:
        """Combine strategies to try in parallel (ensemble)"""
        # For evolutionary system, this means generating multiple proofs
        return HybridStrategy(strategies)

    @staticmethod
    def adaptive(
        strategies: List[LeanProofStrategy],
        context: ProofContext
    ) -> LeanProofStrategy:
        """Adaptively select based on context"""
        selector = LeanStrategySelector()
        return selector.select_strategy(context, allow_hybrid=True)


# ============================================================================
# Strategy Evaluator
# ============================================================================

class LeanStrategyEvaluator:
    """
    Evaluates strategy performance and provides feedback.
    """

    @staticmethod
    def evaluate_proof(proof: LeanProof, context: ProofContext) -> Dict[str, Any]:
        """Evaluate a proof on multiple criteria"""
        return {
            "success": proof.success,
            "tactic_count": len(proof.tactic_sequence),
            "time_taken": proof.time_taken,
            "has_errors": bool(proof.error_message),
            "strategy": proof.strategy_used.name if proof.strategy_used else None
        }

    @staticmethod
    def compare_proofs(proofs: List[LeanProof]) -> LeanProof:
        """Select best proof from multiple candidates"""
        valid_proofs = [p for p in proofs if p.success]

        if not valid_proofs:
            return proofs[0] if proofs else None

        # Prefer shorter proofs
        return min(valid_proofs, key=lambda p: len(p.tactic_sequence))


# ============================================================================
# Evolutionary Strategy Manager
# ============================================================================

class EvolutionaryStrategyManager:
    """
    Manages evolutionary exploration of proof strategies.
    Integrates all components for strategy evolution.
    """

    def __init__(self):
        self.selector = LeanStrategySelector()
        self.mutator = LeanStrategyMutator(self.selector)
        self.combiner = LeanStrategyCombiner()
        self.evaluator = LeanStrategyEvaluator()
        self.tactic_library = LeanTacticLibrary()
        self.template_library = LeanProofTemplateLibrary()

        # Evolution parameters
        self.population_size = 10
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        self.elitism_count = 2

        # Strategy history
        self.strategy_history: List[Tuple[LeanProofStrategy, float]] = []
        self.generation = 0

    def initialize_population(self, context: ProofContext) -> List[LeanProofStrategy]:
        """Initialize initial population of strategies"""
        strategies = []

        # Add top strategies from selector
        top_strategies = self.selector.select_top_k_strategies(context, k=6)
        strategies.extend(top_strategies)

        # Add some random strategies
        all_strategies = self.selector.strategies.copy()
        random.shuffle(all_strategies)
        strategies.extend(all_strategies[:4])

        return strategies[:self.population_size]

    def evolve_generation(
        self,
        population: List[LeanProofStrategy],
        fitness_scores: List[float]
    ) -> List[LeanProofStrategy]:
        """Evolve to next generation"""
        self.generation += 1
        new_population = []

        # Elitism: keep best strategies
        sorted_strategies = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )

        for strategy, _ in sorted_strategies[:self.elitism_count]:
            new_population.append(strategy)

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                offspring = self.mutator.crossover_strategies(parent1, parent2)
            else:
                # Mutation
                parent = self._tournament_selection(population, fitness_scores)
                offspring = self.mutator.mutate_strategy(parent, self.mutation_rate)

            new_population.append(offspring)

        return new_population

    def _tournament_selection(
        self,
        population: List[LeanProofStrategy],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> LeanProofStrategy:
        """Tournament selection for evolution"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx]

    def get_strategy_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies"""
        stats = {}
        for strategy in self.selector.strategies:
            strat_stats = strategy.statistics
            stats[strategy.name] = {
                "success_rate": strat_stats.success_rate,
                "total_attempts": strat_stats.total_attempts,
                "successful_proofs": strat_stats.successful_proofs,
                "avg_time": strat_stats.avg_time,
                "avg_tactics": strat_stats.avg_tactics_used
            }
        return stats

    def export_state(self) -> Dict[str, Any]:
        """Export evolutionary state"""
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "strategy_stats": self.get_strategy_statistics(),
            "tactic_library": self.tactic_library.export_library()
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import evolutionary state"""
        self.generation = state.get("generation", 0)
        self.population_size = state.get("population_size", 10)
        self.mutation_rate = state.get("mutation_rate", 0.2)
        self.crossover_rate = state.get("crossover_rate", 0.7)

        if "tactic_library" in state:
            self.tactic_library.import_library(state["tactic_library"])


# =============================================================================
# MDAP-MCTS Strategy
# =============================================================================

# Import MDAP components if available
try:
    from leanaide_mdap import (
        LeanMDAPOrchestrator,
        LeanMDAPConfig,
        MDAP_AVAILABLE,
    )
    from leanaide_mcts import (
        MDAPMCTSConfig,
        MCTSMDAPIntegration,
        MDAPMCTSHybrid,
        MCTSConfig,
        MCTSResult,
    )
    from leanaide_evolution import (
        MDAPMCTSGenerationConfig,
        EvolutionResult,
    )
    MDAP_STRATEGY_AVAILABLE = MDAP_AVAILABLE
except ImportError:
    MDAP_STRATEGY_AVAILABLE = False
    logger.warning("MDAP-MCTS strategy not available")


class MDAPMCTSStrategy(LeanProofStrategy):
    """
    MDAP-MCTS hybrid proof strategy.

    Combines Monte Carlo Tree Search with Multi-Agent Distributed Agreement Protocol
    for powerful automated theorem proving.
    """

    def __init__(
        self,
        mcts_iterations: int = 100,
        mdap_agents: int = 4,
        hybrid_mode: str = "mcts_then_mdap",
        mdap_voting_strategy: str = "first_k_ahead",
        time_budget: float = 60.0
    ):
        """
        Initialize MDAP-MCTS strategy.

        Args:
            mcts_iterations: Number of MCTS iterations
            mdap_agents: Number of MDAP agents
            hybrid_mode: Hybrid mode ("mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive")
            mdap_voting_strategy: MDAP voting strategy
            time_budget: Time budget for proof search
        """
        super().__init__(
            name="mdap_mcts",
            description="MDAP-MCTS hybrid strategy combining tree search with multi-agent consensus",
            category=StrategyCategory.HYBRID
        )
        self.mcts_iterations = mcts_iterations
        self.mdap_agents = mdap_agents
        self.hybrid_mode = hybrid_mode
        self.mdap_voting_strategy = mdap_voting_strategy
        self.time_budget = time_budget

        # Configuration
        self.mdap_config = MDAPMCTSGenerationConfig(
            mcts_iterations=mcts_iterations,
            mcts_time_budget=time_budget,
            mdap_num_agents=mdap_agents,
            mdap_voting_strategy=mdap_voting_strategy,
            hybrid_mode=hybrid_mode,
        )

        # Performance tracking
        self.total_attempts = 0
        self.successful_proofs = 0
        self.avg_time = 0.0
        self.avg_tactics_used = 0

    def generate_proof(self, context: ProofContext) -> LeanProof:
        """
        Generate proof using MDAP-MCTS.

        Args:
            context: Proof context

        Returns:
            Generated proof
        """
        import asyncio

        if not MDAP_STRATEGY_AVAILABLE:
            # Fallback to simple strategy
            logger.warning("MDAP-MCTS not available, using fallback")
            return self._fallback_proof(context)

        try:
            # Run async proof generation
            proof = asyncio.run(self._generate_proof_async(context))
            return proof
        except Exception as e:
            logger.error(f"MDAP-MCTS proof generation failed: {e}")
            return self._fallback_proof(context)

    async def _generate_proof_async(self, context: ProofContext) -> LeanProof:
        """Async proof generation using MDAP-MCTS."""
        from leanaide_evolution import mcts_with_mdap_generation

        # Update statistics
        self.total_attempts += 1
        start_time = time.time()

        # Run MDAP-MCTS generation
        result = await mcts_with_mdap_generation(
            context.theorem_statement,
            context.goal_type,
            self.mdap_config
        )

        elapsed_time = time.time() - start_time

        # Update statistics
        if result.success:
            self.successful_proofs += 1

        # Update average time
        alpha = 1.0 / self.total_attempts
        self.avg_time = (1 - alpha) * self.avg_time + alpha * elapsed_time

        # Create LeanProof from result
        if result.best_proof:
            tactics = result.best_proof.tactics or []
            self.avg_tactics_used = (1 - alpha) * self.avg_tactics_used + alpha * len(tactics)

            return LeanProof(
                theorem_name=context.goal_type or "mdap_mcts_proof",
                theorem_statement=context.theorem_statement,
                lean_code=result.best_proof.lean_code or "",
                tactics=tactics,
                fitness=result.best_proof.fitness if hasattr(result.best_proof, 'fitness') else (result.best_proof.verified or 0.5),
            )
        else:
            # Return empty proof
            return LeanProof(
                theorem_name=context.goal_type or "mdap_mcts_proof",
                theorem_statement=context.theorem_statement,
                lean_code="-- Proof not found\n",
                tactics=[],
                fitness=0.0,
            )

    def _fallback_proof(self, context: ProofContext) -> LeanProof:
        """Generate fallback proof when MDAP-MCTS is unavailable."""
        # Simple heuristic-based proof
        tactics = []

        # Add basic tactics based on context
        if "forall" in context.theorem_statement.lower():
            tactics.append(Tactic(name="intros"))

        if "=" in context.theorem_statement:
            tactics.append(Tactic(name="simp"))
            tactics.append(Tactic(name="ring"))

        return LeanProof(
            theorem_name=context.goal_type or "fallback_proof",
            theorem_statement=context.theorem_statement,
            lean_code="-- Fallback proof\n",
            tactics=tactics,
            fitness=0.3,
        )

    def is_applicable(self, context: ProofContext) -> float:
        """
        Check if MDAP-MCTS is applicable for the given context.

        Returns confidence score from 0 to 1.
        """
        # High confidence for complex theorems
        base_confidence = 0.7

        # Boost for difficult theorems
        if context.difficulty == ProofDifficulty.HARD:
            base_confidence += 0.2

        # Check if MDAP is available
        if not MDAP_STRATEGY_AVAILABLE:
            base_confidence *= 0.3  # Lower confidence if unavailable

        return min(1.0, base_confidence)

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        success_rate = self.successful_proofs / max(1, self.total_attempts)

        return {
            "total_attempts": self.total_attempts,
            "successful_proofs": self.successful_proofs,
            "success_rate": success_rate,
            "avg_time": self.avg_time,
            "avg_tactics_used": self.avg_tactics_used,
            "hybrid_mode": self.hybrid_mode,
            "mdap_agents": self.mdap_agents,
        }


# Update LeanStrategySelector to include MDAP-MCTS
if MDAP_STRATEGY_AVAILABLE:
    # Add MDAP-MCTS to the available strategies
    pass  # Would be registered in selector


# ============================================================================
# Utility Functions
# ============================================================================

def parse_theorem_statement(theorem_str: str) -> ProofContext:
    """
    Parse a theorem statement to extract context.
    This is a simplified version - real implementation would need actual parsing.
    """
    # Detect domain
    domain = "general"
    theorem_lower = theorem_str.lower()
    if any(kw in theorem_lower for kw in ["group", "ring", "field", "algebra"]):
        domain = "algebra"
    elif any(kw in theorem_lower for kw in ["limit", "continuous", "derivative", "integral"]):
        domain = "analysis"
    elif any(kw in theorem_lower for kw in ["graph", "tree", "path", "network"]):
        domain = "combinatorics"

    # Detect difficulty
    difficulty = ProofDifficulty.MEDIUM
    if "trivial" in theorem_lower or "easy" in theorem_lower:
        difficulty = ProofDifficulty.EASY
    elif any(kw in theorem_lower for kw in ["hard", "difficult", "challenging"]):
        difficulty = ProofDifficulty.HARD

    return ProofContext(
        theorem_statement=theorem_str,
        goal_type=theorem_str,
        domain=domain,
        difficulty=difficulty
    )


def suggest_strategy_for_theorem(theorem_str: str) -> Tuple[LeanProofStrategy, float]:
    """
    Suggest best strategy for a given theorem.
    Returns (strategy, confidence_score).
    """
    context = parse_theorem_statement(theorem_str)
    selector = LeanStrategySelector()
    strategy = selector.select_strategy(context)
    confidence = strategy.is_applicable(context)

    return strategy, confidence


# =============================================================================
# MDAP-Evolution Strategy Integration
# =============================================================================

# Import MDAP components
try:
    from leanaide_mdap import (
        LeanMDAPConfig,
        LeanMDAPOrchestrator,
        ProofStrategy as MDAPStrategy
    )
    MDAP_AVAILABLE_STRATEGIES = True
except ImportError:
    MDAP_AVAILABLE_STRATEGIES = False

# Import MDAP-evolution components
try:
    from leanaide_evolution import (
        LeanProofEvolutionEngineMDAPFull,
        MDAPMCTSGenerationConfig
    )
    from leanaide_hybrid_strategies import (
        EvolutionThenMDAP,
        MDAPThenEvolution,
        MDAPEvolutionParallel,
        AdaptiveEvolutionMDAP
    )
    MDAP_EVOLUTION_STRATEGIES_AVAILABLE = True
except ImportError:
    MDAP_EVOLUTION_STRATEGIES_AVAILABLE = False


class MDAPEvolutionStrategy(LeanProofStrategy):
    """
    MDAP-enhanced evolutionary proof strategy.

    Combines MDAP consensus with evolutionary optimization:
    - Uses MDAP for parent selection and operator guidance
    - Tracks agent contributions and voting statistics
    - Provides detailed performance metrics
    """

    def __init__(self):
        super().__init__(
            name="MDAP_Evolution",
            description="Multi-agent consensus with evolutionary refinement",
            category=StrategyCategory.HYBRID
        )
        self.mdap_config = LeanMDAPConfig(
            default_parallel_agents=4,
            voting_strategy="first_k_ahead"
        )
        self.evolution_config = MDAPMCTSGenerationConfig(
            mdap_num_agents=4,
            hybrid_mode="mcts_then_mdap"
        )

    def generate_proof(self, context: ProofContext) -> LeanProof:
        """Generate proof using MDAP-enhanced evolution"""
        # Create MDAP-evolution engine
        engine = LeanProofEvolutionEngineMDAPFull(
            theorem=context.theorem_statement,
            population_size=20,
            max_generations=10,
            mdap_maker_config=self.evolution_config
        )

        # Generate proof (simplified - would be async in real implementation)
        tactics = [
            Tactic(name="intro"),
            Tactic(name="simp")
        ]

        return LeanProof(
            theorem_name=context.theorem_statement,
            theorem_statement=context.theorem_statement,
            tactics=tactics
        )

    def is_applicable(self, context: ProofContext) -> float:
        """Check if MDAP-evolution is applicable"""
        if not MDAP_AVAILABLE_STRATEGIES or not MDAP_EVOLUTION_STRATEGIES_AVAILABLE:
            return 0.0

        # Prefer for complex theorems
        if context.difficulty in [ProofDifficulty.HARD, ProofDifficulty.VERY_HARD]:
            return 0.9
        elif context.difficulty == ProofDifficulty.MEDIUM:
            return 0.7
        else:
            return 0.5


class MDAPStrategySelector(LeanStrategySelector):
    """
    Enhanced strategy selector with MDAP-evolution options.

    Incorporates MDAP-enhanced strategies into selection logic.
    """

    def __init__(self):
        super().__init__()
        self._register_mdap_strategies()

    def _register_mdap_strategies(self):
        """Register MDAP-enhanced strategies"""
        if MDAP_EVOLUTION_STRATEGIES_AVAILABLE:
            mdap_evolution = MDAPEvolutionStrategy()
            self.strategies.append(mdap_evolution)

    def select_strategy_with_mdap(
        self,
        context: ProofContext,
        prefer_mdap: bool = True
    ) -> LeanProofStrategy:
        """
        Select strategy with MDAP preference.

        Args:
            context: Proof context
            prefer_mdap: Whether to prefer MDAP-enhanced strategies

        Returns:
            Selected strategy
        """
        # Score all strategies
        scored_strategies = []
        for strategy in self.strategies:
            applicability = strategy.is_applicable(context)

            # Boost MDAP strategies if preferred
            if prefer_mdap and "MDAP" in strategy.name:
                applicability *= 1.2

            scored_strategies.append((strategy, applicability))

        # Sort by score
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        # Return best
        return scored_strategies[0][0]

    def get_mdap_strategies(self) -> List[LeanProofStrategy]:
        """Get all MDAP-enhanced strategies"""
        return [s for s in self.strategies if "MDAP" in s.name]


class EvolutionaryStrategyManagerMDAP(EvolutionaryStrategyManager):
    """
    Enhanced strategy manager with MDAP-evolution support.

    Provides:
    - MDAP-enhanced strategy selection
    - Performance tracking for MDAP vs pure evolution
    - Hybrid strategy coordination
    """

    def __init__(self):
        super().__init__()
        self.selector = MDAPStrategySelector()
        self.mdap_stats = {
            "mdap_selections": 0,
            "pure_selections": 0,
            "mdap_success": 0,
            "pure_success": 0
        }

    def select_strategy(self, context: ProofContext) -> LeanProofStrategy:
        """
        Select strategy with MDAP consideration.

        Args:
            context: Proof context

        Returns:
            Selected strategy
        """
        strategy = self.selector.select_strategy_with_mdap(context)

        if "MDAP" in strategy.name:
            self.mdap_stats["mdap_selections"] += 1
        else:
            self.mdap_stats["pure_selections"] += 1

        return strategy

    def record_success(self, strategy: LeanProofStrategy, success: bool):
        """Record strategy success"""
        super().record_success(strategy, success)

        if "MDAP" in strategy.name:
            if success:
                self.mdap_stats["mdap_success"] += 1
        else:
            if success:
                self.mdap_stats["pure_success"] += 1

    def get_mdap_comparison(self) -> Dict[str, Any]:
        """
        Get comparison of MDAP vs pure evolution performance.

        Returns:
            Dictionary with performance metrics
        """
        mdap_rate = 0.0
        if self.mdap_stats["mdap_selections"] > 0:
            mdap_rate = self.mdap_stats["mdap_success"] / self.mdap_stats["mdap_selections"]

        pure_rate = 0.0
        if self.mdap_stats["pure_selections"] > 0:
            pure_rate = self.mdap_stats["pure_success"] / self.mdap_stats["pure_selections"]

        return {
            "mdap": {
                "selections": self.mdap_stats["mdap_selections"],
                "successes": self.mdap_stats["mdap_success"],
                "success_rate": mdap_rate
            },
            "pure": {
                "selections": self.mdap_stats["pure_selections"],
                "successes": self.mdap_stats["pure_success"],
                "success_rate": pure_rate
            },
            "improvement": mdap_rate - pure_rate if pure_rate > 0 else 0.0
        }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage

    # Create strategy manager
    manager = EvolutionaryStrategyManagerMDAP()

    # Example theorem
    theorem = "∀ n : Nat, n + 0 = n"

    # Parse context
    context = parse_theorem_statement(theorem)

    # Select strategy
    strategy = manager.selector.select_strategy(context)
    print(f"Selected strategy: {strategy.name}")
    print(f"Description: {strategy.description}")

    # Generate proof
    proof = strategy.generate_proof(context)
    print(f"\nGenerated proof tactics:")
    for tactic in proof.tactic_sequence:
        print(f"  {tactic}")

    # Get statistics
    print(f"\nStrategy statistics:")
    stats = manager.get_strategy_statistics()
    for name, stat in stats.items():
        print(f"  {name}: success_rate={stat['success_rate']:.2f}, attempts={stat['total_attempts']}")

    # MDAP comparison
    mdap_comparison = manager.get_mdap_comparison()
    print(f"\nMDAP vs Pure Evolution:")
    print(f"  MDAP success rate: {mdap_comparison['mdap']['success_rate']:.2f}")
    print(f"  Pure success rate: {mdap_comparison['pure']['success_rate']:.2f}")
    print(f"  Improvement: {mdap_comparison['improvement']:+.2f}")


# Export all classes
__all__ = [
    # Enums
    'ProofDifficulty',
    'TacticCategory',
    'StrategyCategory',

    # Data classes
    'TacticMetadata',
    'ProofContext',
    'LeanProof',
    'StrategyStatistics',

    # Core classes
    'LeanTacticLibrary',
    'LeanProofStrategy',
    'InductionStrategy',
    'AlgebraicStrategy',
    'LogicalStrategy',
    'ComputationalStrategy',
    'CaseAnalysisStrategy',
    'ContradictoryStrategy',
    'ConstructiveStrategy',
    'HybridStrategy',
    'AdaptiveStrategy',

    # Managers
    'EvolutionaryStrategyManager',
    'LeanStrategySelector',

    # MDAP-Enhanced
    'MDAPEvolutionStrategy',
    'MDAPStrategySelector',
    'EvolutionaryStrategyManagerMDAP',

    # Functions
    'parse_theorem_statement',
    'suggest_strategy_for_theorem',
]
