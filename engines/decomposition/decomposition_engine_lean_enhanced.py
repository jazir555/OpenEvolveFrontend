"""
LeanAide-Enhanced Decomposition Engine

Integrates LeanAide evolutionary proof generation with the existing decomposition engine
to support mathematical problem decomposition with evolutionary proof strategies.

Key Enhancements:
1. Mathematical problem detection and classification
2. LeanAide decomposition routing for mathematical problems
3. Evolutionary proof strategy generation
4. Lean-friendly sub-problem creation
5. Proof complexity estimation
6. Evolutionary strategy suggestion
7. ROMA/CREWAI integration for Lean tickets

Author: OpenEvolve
Created: 2025-12-30
"""
from __future__ import annotations


import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import existing decomposition engine
try:
    from decomposition_engine import (
        DecompositionEngine,
        DecompositionStrategyBase,
        SemanticDecomposition,
        DependencyDecomposition,
        ComplexityDecomposition,
        HybridDecomposition,
        DecompositionPlan
    )
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        ComplexityScore,
        DecompositionStrategy as DataDecompositionStrategy,
        generate_id
    )
    from problem_analyzer import ProblemAnalyzer
    DECOMPOSITION_ENGINE_AVAILABLE = True
except ImportError as e:
    DECOMPOSITION_ENGINE_AVAILABLE = False
    logging.warning(f"Decomposition engine not available: {e}")

# Import LeanAide components
try:
    from leanaide_evolution import (
        LeanProofEvolutionEngine,
        LeanProofStrategy,
        EvolutionResult,
        MutationType,
        SelectionMethod,
        CrossoverMethod
    )
    LEANAIDE_EVOLUTION_AVAILABLE = True
except ImportError:
    LEANAIDE_EVOLUTION_AVAILABLE = False
    logging.warning("LeanAide evolution not available")

try:
    from leanaide_decomposition_integration import (
        LeanDecomposer,
        LeanDecompositionPlan,
        LeanSubProblem,
        MathematicalComponent,
        ComponentType,
        DecompositionStrategy as LeanDecompositionStrategy,
    )
    LEANAIDE_DECOMPOSITION_AVAILABLE = True
except ImportError:
    LEANAIDE_DECOMPOSITION_AVAILABLE = False
    logging.warning("LeanAide decomposition not available")

# MathematicalDomain shared symbol (prefer engines/other/math_domain)
try:
    from math_domain import MathematicalDomain
except ImportError:
    try:
        from leanaide_decomposition_integration import MathematicalDomain
    except ImportError:
        MathematicalDomain = None

# Import workflow structures
try:
    from workflow_structures import MathematicalDomain as WorkflowMathDomain
except ImportError:
    WorkflowMathDomain = None

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class MathematicalProblemType(Enum):
    """Types of mathematical problems"""
    THEOREM_PROOF = "theorem_proof"
    LEMMA_PROOF = "lemma_proof"
    DEFINITION_FORMALIZATION = "definition_formalization"
    CONJECTURE_INVESTIGATION = "conjecture_investigation"
    EXERCISE_SOLUTION = "exercise_solution"
    CONSTRUCTION_PROBLEM = "construction_problem"
    COMPUTATION_PROBLEM = "computation_problem"
    GENERAL_MATHEMATICS = "general_mathematics"


class EvolutionaryStrategyType(Enum):
    """Types of evolutionary strategies for proof generation"""
    STANDARD_EVOLUTION = "standard_evolution"
    ADVERSARIAL_EVOLUTION = "adversarial_evolution"
    SELF_PLAY = "self_play"
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"
    HYBRID_EVOLUTIONARY = "hybrid_evolutionary"


@dataclass
class MathematicalProblemMetadata:
    """
    Metadata for mathematical problems detected by the analyzer.

    Attributes:
        is_mathematical: Whether the problem is mathematical in nature
        problem_type: Type of mathematical problem
        domain: Mathematical domain (algebra, analysis, topology, etc.)
        proof_difficulty: Estimated proof difficulty (1-10)
        formalization_complexity: Estimated formalization complexity (1-10)
        recommended_evolutionary_strategy: Suggested evolutionary approach
        lean_components: Extracted Lean-formalizable components
        dependencies: Logical dependencies between components
        suggested_tactics: Suggested Lean 4 tactics
        requires_evolution: Whether evolutionary approach is recommended
    """
    is_mathematical: bool = False
    problem_type: Optional[MathematicalProblemType] = None
    domain: Optional[MathematicalDomain] = None
    proof_difficulty: int = 5
    formalization_complexity: int = 5
    recommended_evolutionary_strategy: Optional[EvolutionaryStrategyType] = None
    lean_components: List[MathematicalComponent] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    suggested_tactics: List[str] = field(default_factory=list)
    requires_evolution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_mathematical": self.is_mathematical,
            "problem_type": self.problem_type.value if self.problem_type else None,
            "domain": self.domain.value if self.domain else None,
            "proof_difficulty": self.proof_difficulty,
            "formalization_complexity": self.formalization_complexity,
            "recommended_evolutionary_strategy": self.recommended_evolutionary_strategy.value if self.recommended_evolutionary_strategy else None,
            "lean_components": [c.to_dict() for c in self.lean_components],
            "dependencies": self.dependencies,
            "suggested_tactics": self.suggested_tactics,
            "requires_evolution": self.requires_evolution
        }


@dataclass
class LeanEnhancedSubProblem:
    """
    Enhanced sub-problem with Lean-specific metadata and evolutionary guidance.

    Extends the base SubProblem with Lean-specific fields for mathematical components.
    """
    base_subproblem: SubProblem
    mathematical_metadata: MathematicalProblemMetadata
    lean_code_stub: Optional[str] = None
    evolutionary_config: Optional[Dict[str, Any]] = None
    verification_ticket: Optional[str] = None
    formalization_status: str = "pending"

    def to_subproblem(self) -> SubProblem:
        """Convert to base SubProblem for workflow integration."""
        # Enhance metadata with Lean information
        enhanced_metadata = self.base_subproblem.metadata or {}
        enhanced_metadata.update({
            "mathematical_metadata": self.mathematical_metadata.to_dict(),
            "lean_code_stub": self.lean_code_stub,
            "evolutionary_config": self.evolutionary_config,
            "verification_ticket": self.verification_ticket,
            "formalization_status": self.formalization_status
        })

        return SubProblem(
            id=self.base_subproblem.id,
            parent_id=self.base_subproblem.parent_id,
            title=self.base_subproblem.title,
            description=self.base_subproblem.description,
            type=self.base_subproblem.type,
            complexity_score=self.base_subproblem.complexity_score,
            dependencies=self.base_subproblem.dependencies,
            success_criteria=self.base_subproblem.success_criteria,
            validation_gauntlet=self.base_subproblem.validation_gauntlet,
            priority=self.base_subproblem.priority,
            estimated_effort=self.base_subproblem.estimated_effort,
            solution_requirements=self.base_subproblem.solution_requirements,
            acceptance_criteria=self.base_subproblem.acceptance_criteria,
            mathematical_components=self.mathematical_metadata.lean_components,
            mathematical_domain=self.mathematical_metadata.domain,
            requires_formal_verification=True,
            formal_verification_enabled=True,
            metadata=enhanced_metadata,
            **{k: v for k, v in self.base_subproblem.__dict__.items()
               if k not in ['id', 'parent_id', 'title', 'description', 'type',
                           'complexity_score', 'dependencies', 'success_criteria',
                           'validation_gauntlet', 'priority', 'estimated_effort',
                           'solution_requirements', 'acceptance_criteria', 'metadata',
                           'mathematical_components', 'mathematical_domain',
                           'requires_formal_verification', 'formal_verification_enabled']}
        )


# =============================================================================
# LEAN MATHEMATICAL PROBLEM DETECTOR
# =============================================================================

class LeanMathematicalDetector:
    """
    Detects and classifies mathematical problems suitable for LeanAide.

    Analyzes problem statements to identify:
    - Mathematical content and domain
    - Proof-theoretic structure
    - Formalization feasibility
    - Appropriate evolutionary strategies
    """

    # Mathematical keywords by domain
    MATH_KEYWORDS = {
        MathematicalDomain.ALGEBRA: [
            "group", "ring", "field", "vector space", "matrix", "linear",
            "polynomial", "algebraic", "homomorphism", "isomorphism"
        ],
        MathematicalDomain.ANALYSIS: [
            "limit", "continuous", "derivative", "integral", "converge",
            "series", "function", "sequence", "calculus", "real number"
        ],
        MathematicalDomain.TOPOLOGY: [
            "topology", "compact", "connected", "open set", "closed set",
            "continuous function", "topological space", "neighborhood"
        ],
        MathematicalDomain.NUMBER_THEORY: [
            "prime", "divisible", "integer", "natural number", "modular",
            "congruence", "divisor", "factorization", "arithmetic"
        ],
        MathematicalDomain.COMBINATORICS: [
            "graph", "tree", "permutation", "combination", "count",
            "binomial", "set", "subset", "cardinality", "bijection"
        ],
        MathematicalDomain.GEOMETRY: [
            "angle", "triangle", "circle", "line", "plane", "distance",
            "congruent", "similar", "parallel", "perpendicular", "polygon"
        ],
        MathematicalDomain.LOGIC: [
            "proof", "proposition", "theorem", "lemma", "implies",
            "quantifier", "predicate", "formal", "logic", "induction"
        ],
        MathematicalDomain.SET_THEORY: [
            "set", "subset", "union", "intersection", "function",
            "relation", "cardinality", "infinite", "axiom"
        ]
    }

    # Proof-related keywords
    PROOF_KEYWORDS = [
        "prove", "proof", "show", "demonstrate", "theorem", "lemma",
        "proposition", "corollary", "verify", "formalize"
    ]

    def __init__(self, enable_llm: bool = True):
        """
        Initialize mathematical detector.

        Args:
            enable_llm: Whether to use LLM-based detection (when available)
        """
        self.enable_llm = enable_llm
        self.logger = logging.getLogger(__name__)

    def detect_mathematical_problem(
        self,
        problem_text: str,
        problem_title: str = ""
    ) -> MathematicalProblemMetadata:
        """
        Detect if a problem is mathematical and extract metadata.

        Args:
            problem_text: Problem description
            problem_title: Optional problem title

        Returns:
            MathematicalProblemMetadata with analysis results
        """
        combined_text = f"{problem_title}\n\n{problem_text}".lower()

        # Check for mathematical content
        is_mathematical = self._is_mathematical(combined_text)

        if not is_mathematical:
            return MathematicalProblemMetadata(is_mathematical=False)

        # Classify problem type
        problem_type = self._classify_problem_type(combined_text)

        # Identify mathematical domain
        domain = self._identify_domain(combined_text)

        # Estimate proof difficulty
        proof_difficulty = self._estimate_proof_difficulty(combined_text, problem_type)

        # Estimate formalization complexity
        formalization_complexity = self._estimate_formalization_complexity(
            combined_text, problem_type, domain
        )

        # Suggest evolutionary strategy
        evolutionary_strategy = self._suggest_evolutionary_strategy(
            problem_type, domain, proof_difficulty
        )

        # Determine if evolution is required
        requires_evolution = proof_difficulty >= 7 or formalization_complexity >= 7

        return MathematicalProblemMetadata(
            is_mathematical=True,
            problem_type=problem_type,
            domain=domain,
            proof_difficulty=proof_difficulty,
            formalization_complexity=formalization_complexity,
            recommended_evolutionary_strategy=evolutionary_strategy,
            requires_evolution=requires_evolution
        )

    def _is_mathematical(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        # Check for proof keywords
        has_proof_keywords = any(kw in text for kw in self.PROOF_KEYWORDS)

        # Check for mathematical symbols
        has_math_symbols = bool(re.search(r'∀|∃|->|⇒|∈|⊂|∪|∩|≤|≥|≠|∑|∏|∫', text))

        # Check for mathematical keywords
        math_keyword_count = sum(
            sum(1 for kw in keywords if kw in text)
            for keywords in self.MATH_KEYWORDS.values()
        )
        has_math_keywords = math_keyword_count >= 2

        # Check for mathematical notation patterns
        has_math_notation = bool(re.search(r'\$[^$]+\$|\\[a-zA-Z]+\{', text))

        return has_proof_keywords or has_math_symbols or has_math_keywords or has_math_notation

    def _classify_problem_type(self, text: str) -> MathematicalProblemType:
        """Classify the type of mathematical problem."""
        if "theorem" in text and "prove" in text:
            return MathematicalProblemType.THEOREM_PROOF
        elif "lemma" in text:
            return MathematicalProblemType.LEMMA_PROOF
        elif "definition" in text or "define" in text:
            return MathematicalProblemType.DEFINITION_FORMALIZATION
        elif "conjecture" in text or "hypothesis" in text:
            return MathematicalProblemType.CONJECTURE_INVESTIGATION
        elif "exercise" in text or "show that" in text:
            return MathematicalProblemType.EXERCISE_SOLUTION
        elif "construct" in text or "find" in text:
            return MathematicalProblemType.CONSTRUCTION_PROBLEM
        elif "compute" in text or "calculate" in text or "evaluate" in text:
            return MathematicalProblemType.COMPUTATION_PROBLEM
        else:
            return MathematicalProblemType.GENERAL_MATHEMATICS

    def _identify_domain(self, text: str) -> MathematicalDomain:
        """Identify the mathematical domain."""
        domain_scores = {}

        for domain, keywords in self.MATH_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return MathematicalDomain.GENERAL

    def _estimate_proof_difficulty(
        self,
        text: str,
        problem_type: MathematicalProblemType
    ) -> int:
        """Estimate proof difficulty on a scale of 1-10."""
        base_difficulty = 5

        # Problem type adjustments
        type_difficulty = {
            MathematicalProblemType.DEFINITION_FORMALIZATION: 3,
            MathematicalProblemType.EXERCISE_SOLUTION: 4,
            MathematicalProblemType.LEMMA_PROOF: 5,
            MathematicalProblemType.THEOREM_PROOF: 6,
            MathematicalProblemType.CONSTRUCTION_PROBLEM: 7,
            MathematicalProblemType.CONJECTURE_INVESTIGATION: 9,
            MathematicalProblemType.COMPUTATION_PROBLEM: 4,
            MathematicalProblemType.GENERAL_MATHEMATICS: 5,
        }
        base_difficulty = type_difficulty.get(problem_type, 5)

        # Length factors
        if len(text) > 1000:
            base_difficulty += 2
        elif len(text) > 500:
            base_difficulty += 1

        # Complexity indicators
        complexity_keywords = [
            "infinite", "uncountable", "transfinite", "non-constructive",
            "axiom of choice", "transcendental", "non-elementary"
        ]
        if any(kw in text for kw in complexity_keywords):
            base_difficulty += 2

        # Proof technique indicators
        advanced_techniques = [
            "induction", "recursion", "diagonal argument", "compactness",
            "contradiction", "contrapositive", "invariant"
        ]
        technique_count = sum(1 for kw in advanced_techniques if kw in text)
        base_difficulty += min(technique_count, 2)

        return min(10, max(1, base_difficulty))

    def _estimate_formalization_complexity(
        self,
        text: str,
        problem_type: MathematicalProblemType,
        domain: MathematicalDomain
    ) -> int:
        """Estimate Lean 4 formalization complexity."""
        base_complexity = 5

        # Domain complexity
        domain_complexity = {
            MathematicalDomain.LOGIC: 7,
            MathematicalDomain.SET_THEORY: 8,
            MathematicalDomain.TOPOLOGY: 8,
            MathematicalDomain.ANALYSIS: 9,
            MathematicalDomain.ALGEBRA: 7,
            MathematicalDomain.NUMBER_THEORY: 6,
            MathematicalDomain.COMBINATORICS: 6,
            MathematicalDomain.GEOMETRY: 7,
            MathematicalDomain.GENERAL: 5,
        }
        base_complexity = domain_complexity.get(domain, 5)

        # Problem type adjustment
        if problem_type == MathematicalProblemType.DEFINITION_FORMALIZATION:
            base_complexity -= 2
        elif problem_type == MathematicalProblemType.THEOREM_PROOF:
            base_complexity += 1

        return min(10, max(1, base_complexity))

    def _suggest_evolutionary_strategy(
        self,
        problem_type: MathematicalProblemType,
        domain: MathematicalDomain,
        proof_difficulty: int
    ) -> Optional[EvolutionaryStrategyType]:
        """Suggest appropriate evolutionary strategy."""
        if proof_difficulty < 5:
            # Simple problems don't need evolution
            return None
        elif proof_difficulty < 7:
            return EvolutionaryStrategyType.STANDARD_EVOLUTION
        elif proof_difficulty < 9:
            if domain in [MathematicalDomain.LOGIC, MathematicalDomain.SET_THEORY]:
                return EvolutionaryStrategyType.SELF_PLAY
            else:
                return EvolutionaryStrategyType.HYBRID_EVOLUTIONARY
        else:
            # Very difficult problems benefit from adversarial approaches
            return EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION


# =============================================================================
# LEAN SUB-PROBLEM DECOMPOSER
# =============================================================================

class LeanSubProblemDecomposer:
    """
    Specialized decomposer for mathematical sub-problems.

    Creates Lean-friendly sub-problems with:
    - Clear formalization goals
    - Identified dependencies
    - Proof complexity estimates
    - Evolutionary configuration
    """

    def __init__(
        self,
        leanaide_decomposer: Optional[LeanDecomposer] = None,
        enable_evolution: bool = True
    ):
        """
        Initialize Lean sub-problem decomposer.

        Args:
            leanaide_decomposer: Optional LeanDecomposer for detailed analysis
            enable_evolution: Whether to enable evolutionary proof generation
        """
        self.leanaide_decomposer = leanaide_decomposer
        self.enable_evolution = enable_evolution and LEANAIDE_EVOLUTION_AVAILABLE
        self.logger = logging.getLogger(__name__)

    async def decompose_mathematical_subproblem(
        self,
        subproblem: SubProblem,
        math_metadata: MathematicalProblemMetadata
    ) -> List[LeanEnhancedSubProblem]:
        """
        Decompose a mathematical sub-problem into Lean-formalizable components.

        Args:
            subproblem: The sub-problem to decompose
            math_metadata: Mathematical metadata for the sub-problem

        Returns:
            List of LeanEnhancedSubProblem objects
        """
        # If LeanAide decomposer is available, use it
        if self.leanaide_decomposer:
            return await self._decompose_with_leanaide(subproblem, math_metadata)
        else:
            return await self._decompose_heuristic(subproblem, math_metadata)

    async def _decompose_with_leanaide(
        self,
        subproblem: SubProblem,
        math_metadata: MathematicalProblemMetadata
    ) -> List[LeanEnhancedSubProblem]:
        """Decompose using LeanAide decomposition engine."""
        try:
            # Create Lean decomposition plan
            plan = await self.leanaide_decomposer.decompose_mathematical_problem(
                subproblem.description
            )

            # Convert to enhanced sub-problems
            enhanced_subproblems = []
            for component in plan.components:
                # Create base sub-problem
                base_sp = SubProblem(
                    id=component.component_id,
                    parent_id=subproblem.id,
                    title=f"{component.type.value.title()}: {component.name}",
                    description=component.statement,
                    type=subproblem.type,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=component.complexity,
                        computational_complexity=component.complexity * 0.8,
                        domain_complexity=component.complexity * 0.9,
                        integration_complexity=component.complexity * 0.7,
                        overall_complexity=component.complexity,
                        explanation=f"Lean formalization complexity for {component.name}"
                    ),
                    dependencies=component.dependencies,
                    success_criteria=subproblem.success_criteria,
                    validation_gauntlet="lean_verification",
                    priority=subproblem.priority,
                    estimated_effort=component.complexity * 2,
                    mathematical_components=[component],
                    mathematical_domain=component.domain,
                    requires_formal_verification=True,
                    formal_verification_enabled=True
                )

                # Create mathematical metadata
                component_metadata = MathematicalProblemMetadata(
                    is_mathematical=True,
                    problem_type=self._component_type_to_problem_type(component.type),
                    domain=component.domain,
                    proof_difficulty=component.complexity,
                    formalization_complexity=component.complexity,
                    lean_components=[component],
                    dependencies={component.component_id: component.dependencies}
                )

                # Create enhanced sub-problem
                enhanced_sp = LeanEnhancedSubProblem(
                    base_subproblem=base_sp,
                    mathematical_metadata=component_metadata
                )

                enhanced_subproblems.append(enhanced_sp)

            return enhanced_subproblems

        except Exception as e:
            self.logger.error(f"LeanAide decomposition failed: {e}")
            return await self._decompose_heuristic(subproblem, math_metadata)

    async def _decompose_heuristic(
        self,
        subproblem: SubProblem,
        math_metadata: MathematicalProblemMetadata
    ) -> List[LeanEnhancedSubProblem]:
        """Decompose using heuristic analysis."""
        # Create single enhanced sub-problem
        enhanced_sp = LeanEnhancedSubProblem(
            base_subproblem=subproblem,
            mathematical_metadata=math_metadata
        )

        return [enhanced_sp]

    def _component_type_to_problem_type(
        self,
        component_type: ComponentType
    ) -> MathematicalProblemType:
        """Map component type to problem type."""
        mapping = {
            ComponentType.THEOREM: MathematicalProblemType.THEOREM_PROOF,
            ComponentType.LEMMA: MathematicalProblemType.LEMMA_PROOF,
            ComponentType.DEFINITION: MathematicalProblemType.DEFINITION_FORMALIZATION,
            ComponentType.PROPOSITION: MathematicalProblemType.THEOREM_PROOF,
            ComponentType.COROLLARY: MathematicalProblemType.THEOREM_PROOF,
            ComponentType.CONJECTURE: MathematicalProblemType.CONJECTURE_INVESTIGATION,
        }
        return mapping.get(component_type, MathematicalProblemType.GENERAL_MATHEMATICS)


# =============================================================================
# LEAN-ENHANCED DECOMPOSITION ENGINE
# =============================================================================

class LeanEnhancedDecompositionEngine(DecompositionEngine if DECOMPOSITION_ENGINE_AVAILABLE else object):
    """
    Enhanced decomposition engine with LeanAide integration.

    Extends the existing DecompositionEngine to:
    - Detect mathematical problems
    - Route to LeanAide decomposition when appropriate
    - Generate Lean-friendly sub-problems
    - Support evolutionary proof generation
    """

    def __init__(
        self,
        problem_analyzer: Optional['ProblemAnalyzer'] = None,
        knowledge_manager: Optional['KnowledgeManager'] = None,
        enable_lean_detection: bool = True,
        enable_evolution: bool = True,
        leanaide_decomposer: Optional[LeanDecomposer] = None
    ):
        """
        Initialize Lean-enhanced decomposition engine.

        Args:
            problem_analyzer: Optional ProblemAnalyzer instance
            knowledge_manager: Optional KnowledgeManager instance
            enable_lean_detection: Whether to enable Lean mathematical detection
            enable_evolution: Whether to enable evolutionary proof generation
            leanaide_decomposer: Optional LeanDecomposer for Lean-specific decomposition
        """
        # Initialize parent class
        if DECOMPOSITION_ENGINE_AVAILABLE:
            super().__init__(problem_analyzer, knowledge_manager)
        else:
            self.strategies = {}
            self.problem_analyzer = problem_analyzer
            self.knowledge_manager = knowledge_manager

        # Lean-specific components
        self.enable_lean_detection = enable_lean_detection
        self.enable_evolution = enable_evolution
        self.leanaide_decomposer = leanaide_decomposer

        # Initialize detectors and decomposers
        self.math_detector = LeanMathematicalDetector(enable_llm=True) if enable_lean_detection else None
        self.lean_subproblem_decomposer = LeanSubProblemDecomposer(
            leanaide_decomposer=leanaide_decomposer,
            enable_evolution=enable_evolution
        ) if enable_lean_detection else None

        self.logger = logging.getLogger(__name__)

    async def decompose_with_leanaide(
        self,
        problem: ProblemDefinition,
        strategy: Optional[str] = None
    ) -> DecompositionPlan:
        """
        Decompose problem with LeanAide integration for mathematical problems.

        Args:
            problem: The problem to decompose
            strategy: Optional strategy name (auto-selected if not provided)

        Returns:
            DecompositionPlan with potentially Lean-enhanced sub-problems
        """
        self.logger.info(f"Decomposing problem with LeanAide integration: {problem.id}")

        # Step 1: Detect if problem is mathematical
        math_metadata = None
        if self.math_detector:
            math_metadata = self.math_detector.detect_mathematical_problem(
                problem.description,
                problem.title
            )
            self.logger.info(f"Mathematical detection: is_mathematical={math_metadata.is_mathematical}")

        # Step 2: If mathematical and LeanAide is available, use Lean decomposition
        if math_metadata and math_metadata.is_mathematical and self.leanaide_decomposer:
            self.logger.info("Using LeanAide-specific decomposition for mathematical problem")
            return await self._decompose_mathematical_problem(problem, math_metadata, strategy)

        # Step 3: Otherwise, use standard decomposition
        self.logger.info("Using standard decomposition")
        if DECOMPOSITION_ENGINE_AVAILABLE:
            return self.decompose(problem, strategy)
        else:
            # Fallback: create basic decomposition
            return await self._fallback_decomposition(problem)

    async def _decompose_mathematical_problem(
        self,
        problem: ProblemDefinition,
        math_metadata: MathematicalProblemMetadata,
        strategy: Optional[str]
    ) -> DecompositionPlan:
        """
        Decompose a mathematical problem using LeanAide integration.

        Args:
            problem: The mathematical problem to decompose
            math_metadata: Mathematical problem metadata
            strategy: Optional strategy preference

        Returns:
            DecompositionPlan with Lean-enhanced sub-problems
        """
        start_time = time.time()

        # Use LeanAide decomposer to get mathematical components
        lean_plan = await self.leanaide_decomposer.decompose_mathematical_problem(
            problem.description,
            LeanDecompositionStrategy.HYBRID
        )

        # Convert Lean components to sub-problems
        sub_problems = []
        for component in lean_plan.components:
            # Create sub-problem from component
            sub_problem = SubProblem(
                id=component.component_id,
                parent_id=problem.id,
                title=f"{component.type.value.title()}: {component.name}",
                description=component.statement,
                type=self._map_component_type_to_subproblem_type(component.type),
                complexity_score=ComplexityScore(
                    cognitive_complexity=component.complexity,
                    computational_complexity=component.complexity * 0.8,
                    domain_complexity=component.complexity * 0.9,
                    integration_complexity=component.complexity * 0.7,
                    overall_complexity=component.complexity,
                    explanation=f"Mathematical formalization complexity for {component.name}"
                ),
                dependencies=component.dependencies,
                success_criteria=[
                    SuccessCriterion(
                        id=generate_id("criterion"),
                        description=f"Formalize and verify {component.name} in Lean 4",
                        metric="verification_success",
                        threshold=0.95,
                        validation_method="lean_verification"
                    )
                ],
                validation_gauntlet="lean_verification",
                priority=self._calculate_priority_from_complexity(component.complexity),
                estimated_effort=component.complexity * 4,  # Hours
                mathematical_components=[component],
                mathematical_domain=component.domain,
                requires_formal_verification=True,
                formal_verification_enabled=True,
                metadata={
                    "lean_formalization": True,
                    "mathematical_type": component.type.value,
                    "proof_difficulty": math_metadata.proof_difficulty,
                    "formalization_complexity": math_metadata.formalization_complexity,
                    "evolutionary_strategy": math_metadata.recommended_evolutionary_strategy.value if math_metadata.recommended_evolutionary_strategy else None
                }
            )

            sub_problems.append(sub_problem)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(sub_problems)

        # Assess quality
        quality_scores = self._assess_quality(problem, sub_problems)

        # Create decomposition plan
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id=problem.id,
            strategy=DataDecompositionStrategy.SEMANTIC,
            sub_problems=sub_problems,
            dependency_graph=dependency_graph,
            validation_checkpoints=[],
            quality_scores=quality_scores,
            confidence_level=0.85,
            created_by="lean_enhanced_decomposition_engine",
            metadata={
                "lean_decomposition": True,
                "mathematical_domain": math_metadata.domain.value if math_metadata.domain else None,
                "proof_difficulty": math_metadata.proof_difficulty,
                "decomposition_time": time.time() - start_time
            }
        )

        self.logger.info(
            f"LeanAide decomposition complete: {len(sub_problems)} sub-problems, "
            f"domain={math_metadata.domain.value if math_metadata.domain else 'unknown'}"
        )

        return plan

    def _map_component_type_to_subproblem_type(
        self,
        component_type: ComponentType
    ) -> str:
        """Map Lean component type to sub-problem type."""
        mapping = {
            ComponentType.THEOREM: "implementation",
            ComponentType.LEMMA: "implementation",
            ComponentType.DEFINITION: "analysis",
            ComponentType.PROPOSITION: "implementation",
            ComponentType.COROLLARY: "validation",
            ComponentType.EXAMPLE: "validation",
        }
        return mapping.get(component_type, "implementation")

    def _calculate_priority_from_complexity(self, complexity: int) -> int:
        """Calculate priority from complexity (higher complexity = higher priority)."""
        return min(10, max(1, complexity))

    async def _fallback_decomposition(self, problem: ProblemDefinition) -> DecompositionPlan:
        """Fallback decomposition when engine is not available."""
        # Create simple decomposition
        sub_problem = SubProblem(
            id=generate_id("subproblem"),
            parent_id=problem.id,
            title=problem.title,
            description=problem.description,
            type="implementation",
            complexity_score=problem.complexity_score,
            dependencies=[],
            success_criteria=problem.success_criteria,
            validation_gauntlet="standard",
            priority=5,
            estimated_effort=8,
            metadata={"fallback_decomposition": True}
        )

        return DecompositionPlan(
            id=generate_id("plan"),
            problem_id=problem.id,
            strategy=DataDecompositionStrategy.SEMANTIC,
            sub_problems=[sub_problem],
            dependency_graph={sub_problem.id: []},
            validation_checkpoints=[],
            quality_scores=None,  # type: ignore
            confidence_level=0.5,
            created_by="lean_enhanced_fallback",
            metadata={"fallback": True}
        )


# =============================================================================
# EVOLUTIONARY STRATEGY SUGGESTOR
# =============================================================================

class EvolutionaryStrategySuggestor:
    """
    Suggests evolutionary proof generation strategies based on problem characteristics.

    Analyzes mathematical problems and recommends appropriate evolutionary approaches:
    - Standard evolution (genetic algorithm)
    - Adversarial evolution (red team vs blue team)
    - Self-play (reinforcement learning style)
    - Hill climbing (iterative refinement)
    - Simulated annealing (temperature-based search)
    - Hybrid approaches
    """

    def __init__(self):
        """Initialize evolutionary strategy suggestor."""
        self.logger = logging.getLogger(__name__)

    def suggest_strategy(
        self,
        math_metadata: MathematicalProblemMetadata
    ) -> Dict[str, Any]:
        """
        Suggest evolutionary strategy configuration.

        Args:
            math_metadata: Mathematical problem metadata

        Returns:
            Dictionary with evolutionary configuration
        """
        if not math_metadata.recommended_evolutionary_strategy:
            return {"enable_evolution": False}

        strategy_type = math_metadata.recommended_evolutionary_strategy

        # Base configuration
        config = {
            "enable_evolution": True,
            "strategy_type": strategy_type.value,
            "population_size": self._suggest_population_size(math_metadata),
            "max_generations": self._suggest_max_generations(math_metadata),
            "mutation_rate": self._suggest_mutation_rate(math_metadata),
            "crossover_rate": 0.8,
            "selection_method": SelectionMethod.TOURNAMENT.value,
            "crossover_method": CrossoverMethod.UNIFORM.value,
            "elitism_ratio": 0.1
        }

        # Strategy-specific adjustments
        if strategy_type == EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION:
            config.update({
                "adversarial_epochs": 5,
                "red_team_size": 10,
                "blue_team_size": 10,
                "adversarial_mutation_rate": 0.15
            })
        elif strategy_type == EvolutionaryStrategyType.SELF_PLAY:
            config.update({
                "self_play_episodes": 100,
                "opponent_pool_size": 20,
                "win_threshold": 0.7
            })
        elif strategy_type == EvolutionaryStrategyType.SIMULATED_ANNEALING:
            config.update({
                "initial_temperature": 100.0,
                "cooling_rate": 0.95,
                "min_temperature": 0.1
            })

        return config

    def _suggest_population_size(self, math_metadata: MathematicalProblemMetadata) -> int:
        """Suggest population size based on problem complexity."""
        if math_metadata.formalization_complexity >= 8:
            return 50
        elif math_metadata.formalization_complexity >= 6:
            return 30
        else:
            return 20

    def _suggest_max_generations(self, math_metadata: MathematicalProblemMetadata) -> int:
        """Suggest maximum generations based on problem difficulty."""
        if math_metadata.proof_difficulty >= 8:
            return 100
        elif math_metadata.proof_difficulty >= 6:
            return 50
        else:
            return 30

    def _suggest_mutation_rate(self, math_metadata: MathematicalProblemMetadata) -> float:
        """Suggest mutation rate based on problem characteristics."""
        # Higher mutation for more complex problems to maintain diversity
        base_rate = 0.1
        complexity_bonus = math_metadata.formalization_complexity * 0.01
        return min(0.3, base_rate + complexity_bonus)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def detect_and_route_mathematical_problem(
    problem: ProblemDefinition,
    enable_lean: bool = True,
    enable_evolution: bool = True
) -> Tuple[Optional[DecompositionPlan], Optional[MathematicalProblemMetadata]]:
    """
    High-level function to detect mathematical problems and route appropriately.

    Args:
        problem: Problem to analyze
        enable_lean: Whether to enable LeanAide integration
        enable_evolution: Whether to enable evolutionary proof generation

    Returns:
        Tuple of (decomposition_plan, mathematical_metadata)
    """
    detector = LeanMathematicalDetector(enable_llm=True)
    math_metadata = detector.detect_mathematical_problem(
        problem.description,
        problem.title
    )

    if not math_metadata.is_mathematical or not enable_lean:
        return None, math_metadata

    # Create Lean-enhanced engine
    engine = LeanEnhancedDecompositionEngine(
        enable_lean_detection=True,
        enable_evolution=enable_evolution
    )

    # Decompose with LeanAide integration
    plan = await engine.decompose_with_leanaide(problem)

    return plan, math_metadata


async def generate_evolutionary_config(
    math_metadata: MathematicalProblemMetadata
) -> Dict[str, Any]:
    """
    Generate evolutionary configuration for mathematical problem.

    Args:
        math_metadata: Mathematical problem metadata

    Returns:
        Evolutionary configuration dictionary
    """
    suggestor = EvolutionaryStrategySuggestor()
    return suggestor.suggest_strategy(math_metadata)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of Lean-enhanced decomposition engine."""
    from sovereign_data_models import ProblemDefinition, DomainContext, ComplexityScore

    # Example mathematical problem
    problem = ProblemDefinition(
        id="test_problem_001",
        title="Infinite Primes Theorem",
        description="""
        Prove that there are infinitely many prime numbers.

        Hint: Assume there are finitely many primes p1, p2, ..., pn.
        Consider the number N = p1 * p2 * ... * pn + 1.
        Show that N must have a prime divisor not in the list.
        """,
        problem_type="theorem_proof",
        domain_context=DomainContext(
            domain="number_theory",
            subdomain="elementary_number_theory",
            related_domains=["algebra", "set_theory"],
            domain_knowledge={}
        ),
        complexity_score=ComplexityScore(
            cognitive_complexity=6.0,
            computational_complexity=2.0,
            domain_complexity=5.0,
            integration_complexity=3.0,
            overall_complexity=4.0,
            explanation="Classic proof with moderate complexity"
        ),
        constraints=[],
        success_criteria=[],
        stakeholders=[],
        resources_available={}
    )

    # Detect and route
    plan, math_metadata = await detect_and_route_mathematical_problem(
        problem,
        enable_lean=True,
        enable_evolution=True
    )

    print(f"Mathematical Problem Detected: {math_metadata.is_mathematical}")
    print(f"Domain: {math_metadata.domain.value if math_metadata.domain else 'N/A'}")
    print(f"Problem Type: {math_metadata.problem_type.value if math_metadata.problem_type else 'N/A'}")
    print(f"Proof Difficulty: {math_metadata.proof_difficulty}/10")
    print(f"Formalization Complexity: {math_metadata.formalization_complexity}/10")
    print(f"Evolutionary Strategy: {math_metadata.recommended_evolutionary_strategy.value if math_metadata.recommended_evolutionary_strategy else 'N/A'}")

    if plan:
        print(f"\nDecomposition Plan:")
        print(f"  Sub-problems: {len(plan.sub_problems)}")
        for sp in plan.sub_problems:
            print(f"  - {sp.title}")
            print(f"    Complexity: {sp.complexity_score.overall_complexity}/10")
            print(f"    Dependencies: {len(sp.dependencies)}")

    # Generate evolutionary config
    if math_metadata.requires_evolution:
        config = await generate_evolutionary_config(math_metadata)
        print(f"\nEvolutionary Configuration:")
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
