"""
LeanAide Decomposition Integration for Mathematical Problems

This module provides intelligent decomposition of mathematical problems into Lean 4
formalization tasks. It bridges the gap between natural language mathematical problems
and formal verification workflows.

Architecture:
    Mathematical Problem -> LeanDecomposer -> LeanDecompositionPlan -> SubProblems
                                  |
                                  v
                          LeanComponentExtractor
                                  |
                                  v
                          LeanSubProblemGenerator
                                  |
                                  v
                          ROMA/CREWAI Tickets

Key Features:
1. Mathematical structure identification (theorems, lemmas, definitions)
2. Dependency graph construction for components
3. Complexity estimation for formalization
4. Lean 4 code generation via LeanAide client
5. Integration with ROMA for recursive decomposition
6. CREWAI ticket creation for tracking
7. Parallel formalization support
8. Context-aware decomposition

Author: OpenEvolve
Created: 2025-12-30
"""

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

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult, TaskType
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    LeanAideClient = None
    LeanAideConfig = None
    LeanAideResult = None
    TaskType = None

# Import workflow structures
try:
    from workflow_structures import (
        SubProblem,
        DecompositionPlan,
        MathematicalComponent as WorkflowMathematicalComponent,
        MathematicalDomain,
        LeanProofStatus
    )
except ImportError:
    # Fallback implementations
    class MathematicalDomain(Enum):
        ALGEBRA = "algebra"
        ANALYSIS = "analysis"
        TOPOLOGY = "topology"
        NUMBER_THEORY = "number_theory"
        COMBINATORICS = "combinatorics"
        GEOMETRY = "geometry"
        LOGIC = "logic"
        SET_THEORY = "set_theory"
        GENERAL = "general"

    class LeanProofStatus(Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        VERIFIED = "verified"
        FAILED = "failed"
        PARTIAL = "partial"
        TIMEOUT = "timeout"
        ERROR = "error"

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class ComponentType(Enum):
    """Types of mathematical components"""
    THEOREM = "theorem"
    LEMMA = "lemma"
    DEFINITION = "definition"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    EXAMPLE = "example"
    AXIOM = "axiom"
    CONJECTURE = "conjecture"
    EXERCISE = "exercise"
    REMARK = "remark"


class DecompositionStrategy(Enum):
    """Strategies for mathematical problem decomposition"""
    STRUCTURAL = "structural"  # Decompose by mathematical structure
    DEPENDENCY = "dependency"  # Decompose by logical dependencies
    COMPLEXITY = "complexity"  # Decompose by formalization complexity
    DOMAIN = "domain"  # Decompose by mathematical domain
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class MathematicalComponent:
    """
    A mathematical component extracted from a problem statement.

    Attributes:
        component_id: Unique identifier for this component
        type: Type of mathematical component
        name: Name of the component
        statement: Mathematical statement in natural language
        domain: Mathematical domain classification
        complexity: Estimated complexity for formalization (1-10)
        dependencies: List of component IDs this depends on
        formalized: Whether this has been formalized in Lean
        lean_code: Lean 4 code if formalized
        verification_status: Verification status
        metadata: Additional metadata
    """
    component_id: str
    type: ComponentType
    name: str
    statement: str
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    complexity: int = 5
    dependencies: List[str] = field(default_factory=list)
    formalized: bool = False
    lean_code: str = ""
    verification_status: Optional[LeanProofStatus] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_id": self.component_id,
            "type": self.type.value,
            "name": self.name,
            "statement": self.statement,
            "domain": self.domain.value,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "formalized": self.formalized,
            "lean_code": self.lean_code,
            "verification_status": self.verification_status.value if self.verification_status else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathematicalComponent":
        """Create MathematicalComponent from dictionary."""
        data = data.copy()
        if isinstance(data.get("type"), str):
            data["type"] = ComponentType(data["type"])
        if isinstance(data.get("domain"), str):
            data["domain"] = MathematicalDomain(data["domain"])
        if isinstance(data.get("verification_status"), str):
            data["verification_status"] = LeanProofStatus(data["verification_status"])
        return cls(**data)


@dataclass
class LeanDecompositionPlan:
    """
    Represents a decomposition plan for Lean 4 formalization.

    Attributes:
        plan_id: Unique identifier for this plan
        problem_statement: Original problem statement
        components: List of mathematical components
        component_order: Optimal order for formalization (topological)
        dependencies: Dependency graph {component_id: [dependent_ids]}
        parallel_groups: Groups that can be formalized in parallel
        formalization_strategy: Suggested approach for formalization
        complexity_estimate: Overall complexity estimate (1-10)
        metadata: Additional metadata
    """
    plan_id: str
    problem_statement: str
    components: List[MathematicalComponent]
    component_order: List[str]
    dependencies: Dict[str, List[str]]
    parallel_groups: List[List[str]]
    formalization_strategy: str
    complexity_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "plan_id": self.plan_id,
            "problem_statement": self.problem_statement,
            "components": [c.to_dict() for c in self.components],
            "component_order": self.component_order,
            "dependencies": self.dependencies,
            "parallel_groups": self.parallel_groups,
            "formalization_strategy": self.formalization_strategy,
            "complexity_estimate": self.complexity_estimate,
            "metadata": self.metadata
        }


@dataclass
class LeanSubProblem:
    """
    A sub-problem for Lean 4 formalization.

    Extends the base SubProblem with Lean-specific fields.
    """
    component: MathematicalComponent
    context: str  # Previous definitions and theorems
    imports: List[str]  # Required Lean imports
    lean_code: Optional[str] = None  # Generated Lean code
    verification_ticket: Optional[str] = None  # CREWAI ticket ID
    status: LeanProofStatus = LeanProofStatus.PENDING

    def to_subproblem(self) -> "SubProblem":
        """Convert to base SubProblem for workflow integration."""
        return SubProblem(
            id=self.component.component_id,
            description=f"{self.component.type.value.title()}: {self.component.name}\n\n{self.component.statement}",
            dependencies=self.component.dependencies,
            ai_suggested_evolution_mode="standard",
            ai_suggested_complexity_score=self.component.complexity,
            content_type="lean4_formalization",
            status="pending" if self.status == LeanProofStatus.PENDING else "in_progress",
            mathematical_components=[self.component],
            requires_formal_verification=True,
            mathematical_domain=self.component.domain,
            formal_verification_enabled=True,
            metadata={
                "lean_code": self.lean_code,
                "context": self.context,
                "imports": self.imports,
                "verification_ticket": self.verification_ticket
            }
        )


# =============================================================================
# MAIN DECOMPOSITION CLASS
# =============================================================================

class LeanDecomposer:
    """
    Decomposes mathematical problems into Lean 4 components.

    This class analyzes mathematical problems and creates structured decomposition
    plans for formalization in Lean 4.

    Features:
    - Mathematical structure identification
    - Dependency graph construction
    - Complexity estimation
    - Formalization strategy suggestion
    - Lean 4 code generation via LeanAide
    """

    def __init__(
        self,
        leanaide_client: Optional["LeanAideClient"] = None,
        enable_llm: bool = True,
        default_strategy: DecompositionStrategy = DecompositionStrategy.HYBRID
    ):
        """
        Initialize LeanDecomposer.

        Args:
            leanaide_client: Optional LeanAide client for code generation
            enable_llm: Whether to use LLM-based analysis
            default_strategy: Default decomposition strategy
        """
        self.leanaide_client = leanaide_client
        self.enable_llm = enable_llm and LEANAIDE_AVAILABLE
        self.default_strategy = default_strategy
        self.logger = logging.getLogger(__name__)

        # Initialize LeanAide client if needed and available
        if not self.leanaide_client and LEANAIDE_AVAILABLE:
            config = LeanAideConfig()
            self.leanaide_client = LeanAideClient(config)
            self._own_client = True
        else:
            self._own_client = False

    async def close(self):
        """Close resources."""
        if self._own_client and self.leanaide_client:
            await self.leanaide_client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def decompose_mathematical_problem(
        self,
        problem: str,
        strategy: Optional[DecompositionStrategy] = None
    ) -> LeanDecompositionPlan:
        """
        Decompose a mathematical problem into Lean 4 components.

        Args:
            problem: Mathematical problem statement in natural language
            strategy: Decomposition strategy (uses default if not specified)

        Returns:
            LeanDecompositionPlan with components, dependencies, and strategy
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        self.logger.info(f"Decomposing mathematical problem using {strategy.value} strategy")

        # Step 1: Extract mathematical components
        components = await self._extract_components(problem)

        if not components:
            self.logger.warning("No components extracted, creating single component")
            components = [
                MathematicalComponent(
                    component_id=str(uuid.uuid4()),
                    type=ComponentType.THEOREM,
                    name="main_theorem",
                    statement=problem,
                    domain=MathematicalDomain.GENERAL,
                    complexity=5
                )
            ]

        # Step 2: Identify dependencies between components
        dependencies = await self._identify_dependencies(components)

        # Step 3: Estimate complexity for each component
        for component in components:
            if not component.lean_code:
                component.complexity = await self._estimate_complexity(component)

        # Step 4: Determine optimal order for formalization
        component_order = self._suggest_order(components, dependencies)

        # Step 5: Identify parallelization opportunities
        parallel_groups = self._identify_parallel_groups(component_order, dependencies)

        # Step 6: Suggest formalization strategy
        formalization_strategy = self._suggest_strategy(components, dependencies)

        # Step 7: Estimate overall complexity
        complexity_estimate = self._estimate_overall_complexity(components)

        # Create plan
        plan = LeanDecompositionPlan(
            plan_id=str(uuid.uuid4()),
            problem_statement=problem,
            components=components,
            component_order=component_order,
            dependencies=dependencies,
            parallel_groups=parallel_groups,
            formalization_strategy=formalization_strategy,
            complexity_estimate=complexity_estimate,
            metadata={
                "strategy": strategy.value,
                "decomposition_time": time.time() - start_time,
                "component_count": len(components),
                "llm_enabled": self.enable_llm
            }
        )

        self.logger.info(f"Decomposition complete: {len(components)} components, complexity {complexity_estimate}/10")
        return plan

    async def _extract_components(
        self,
        problem: str
    ) -> List[MathematicalComponent]:
        """
        Extract mathematical components from problem statement.

        Uses LLM-based analysis when available, falls back to pattern matching.
        """
        if self.enable_llm and self.leanaide_client:
            return await self._extract_components_llm(problem)
        else:
            return self._extract_components_heuristic(problem)

    async def _extract_components_llm(
        self,
        problem: str
    ) -> List[MathematicalComponent]:
        """
        Extract components using LLM analysis via LeanAide.

        Uses LeanAide's json_structured task to parse mathematical structure.
        """
        try:
            # Use LeanAide to get structured JSON
            result = await self.leanaide_client.json_structured(problem)

            if result.success and result.data:
                return self._parse_structured_components(result.data)
            else:
                self.logger.warning(f"LeanAide json_structured failed: {result.error}")
                return self._extract_components_heuristic(problem)

        except (IOError, ConnectionError, TimeoutError, ValueError) as e:
            self.logger.error(f"LLM component extraction failed: {e}")
            return self._extract_components_heuristic(problem)

    def _extract_components_heuristic(
        self,
        problem: str
    ) -> List[MathematicalComponent]:
        """
        Extract components using heuristic pattern matching.

        Identifies:
        - Theorems, lemmas, propositions
        - Definitions
        - Examples
        - Dependencies based on references
        """
        components = []

        # Split problem into sections
        sections = self._split_into_sections(problem)

        for i, section in enumerate(sections):
            # Classify section type
            component_type, name = self._classify_section(section)

            # Extract statement
            statement = self._extract_statement(section)

            # Classify domain
            domain = self._classify_domain(statement)

            # Create component
            component = MathematicalComponent(
                component_id=str(uuid.uuid4()),
                type=component_type,
                name=name or f"component_{i+1}",
                statement=statement,
                domain=domain,
                complexity=5  # Will be refined later
            )

            components.append(component)

        # Post-process: extract dependencies
        self._extract_dependencies_heuristic(components)

        return components

    def _split_into_sections(self, problem: str) -> List[str]:
        """Split problem into logical sections."""
        # Split by common delimiters
        delimiters = [
            r"\n\s*Theorem\s*",
            r"\n\s*Lemma\s*",
            r"\n\s*Definition\s*",
            r"\n\s*Proposition\s*",
            r"\n\s*Example\s*",
            r"\n\s*\d+\.\s*",  # Numbered sections
            r"\n\s*-\s+",  # Bullet points
        ]

        sections = [problem]

        for delimiter in delimiters:
            new_sections = []
            for section in sections:
                new_sections.extend(re.split(delimiter, section, flags=re.IGNORECASE))
            sections = new_sections

        # Filter empty sections
        return [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]

    def _classify_section(self, section: str) -> Tuple[ComponentType, str]:
        """Classify section type and extract name."""
        section_lower = section.lower()

        # Check for type keywords
        type_patterns = {
            ComponentType.THEOREM: r"theorem\s+(.*?):",
            ComponentType.LEMMA: r"lemma\s+(.*?):",
            ComponentType.DEFINITION: r"definition\s+(.*?):",
            ComponentType.PROPOSITION: r"proposition\s+(.*?):",
            ComponentType.EXAMPLE: r"example\s*(.*?):",
            ComponentType.COROLLARY: r"corollary\s+(.*?):",
        }

        for component_type, pattern in type_patterns.items():
            match = re.search(pattern, section_lower, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                return component_type, name

        # Default to theorem
        return ComponentType.THEOREM, ""

    def _extract_statement(self, section: str) -> str:
        """Extract the mathematical statement from a section."""
        # Remove common prefixes
        lines = section.split("\n")
        statement_lines = []

        for line in lines[1:]:  # Skip first line (type/name)
            line = line.strip()
            if line and not line.startswith("#"):
                statement_lines.append(line)

        return "\n".join(statement_lines).strip()

    def _classify_domain(self, statement: str) -> MathematicalDomain:
        """Classify mathematical domain from statement."""
        statement_lower = statement.lower()

        # Domain keywords
        domain_keywords = {
            MathematicalDomain.ALGEBRA: ["group", "ring", "field", "algebra", "vector", "matrix"],
            MathematicalDomain.ANALYSIS: ["limit", "continuous", "derivative", "integral", "converge", "series"],
            MathematicalDomain.TOPOLOGY: ["topology", "compact", "connected", "continuous", "open", "closed"],
            MathematicalDomain.NUMBER_THEORY: ["prime", "divisible", "integer", "natural", "modular", "congruence"],
            MathematicalDomain.COMBINATORICS: ["graph", "tree", "permutation", "combination", "count"],
            MathematicalDomain.GEOMETRY: ["angle", "triangle", "circle", "line", "plane", "distance"],
            MathematicalDomain.LOGIC: ["proof", "proposition", "implies", "quantifier", "predicate"],
            MathematicalDomain.SET_THEORY: ["set", "subset", "union", "intersection", "function"],
        }

        # Score each domain
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in statement_lower)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        else:
            return MathematicalDomain.GENERAL

    def _extract_dependencies_heuristic(
        self,
        components: List[MathematicalComponent]
    ):
        """Extract dependencies between components using heuristic analysis."""
        for i, component in enumerate(components):
            dependencies = []

            # Check for references to other components
            for j, other in enumerate(components):
                if i == j:
                    continue

                # Check if component references other by name
                if other.name.lower() in component.statement.lower():
                    dependencies.append(other.component_id)

                # Check for common reference patterns
                if re.search(rf"\b{re.escape(other.name)}\b", component.statement, re.IGNORECASE):
                    dependencies.append(other.component_id)

            component.dependencies = list(set(dependencies))

    def _parse_structured_components(
        self,
        structured_data: Dict[str, Any]
    ) -> List[MathematicalComponent]:
        """Parse structured JSON from LeanAide into components."""
        components = []

        # Extract theorems, lemmas, definitions from structured data
        for category in ["theorems", "lemmas", "definitions", "examples"]:
            items = structured_data.get(category, [])

            if not isinstance(items, list):
                items = [items] if items else []

            for item in items:
                if not isinstance(item, dict):
                    continue

                # Map category to ComponentType
                type_mapping = {
                    "theorems": ComponentType.THEOREM,
                    "lemmas": ComponentType.LEMMA,
                    "definitions": ComponentType.DEFINITION,
                    "examples": ComponentType.EXAMPLE
                }

                component_type = type_mapping.get(category, ComponentType.THEOREM)

                component = MathematicalComponent(
                    component_id=str(uuid.uuid4()),
                    type=component_type,
                    name=item.get("name", f"{category[:-1]}_{len(components)+1}"),
                    statement=item.get("statement", ""),
                    domain=self._classify_domain(item.get("statement", "")),
                    complexity=item.get("complexity", 5)
                )

                components.append(component)

        return components

    async def _identify_dependencies(
        self,
        components: List[MathematicalComponent]
    ) -> Dict[str, List[str]]:
        """
        Identify dependencies between components.

        Uses LLM analysis when available for sophisticated dependency detection.
        """
        if not components:
            return {}

        if self.enable_llm and self.leanaide_client and len(components) > 2:
            return await self._identify_dependencies_llm(components)
        else:
            return self._identify_dependencies_heuristic(components)

    async def _identify_dependencies_llm(
        self,
        components: List[MathematicalComponent]
    ) -> Dict[str, List[str]]:
        """
        Identify dependencies using LLM analysis.
        """
        # Build prompt for dependency analysis
        component_descriptions = []
        for i, comp in enumerate(components, 1):
            component_descriptions.append(
                f"{i}. {comp.type.value.title()}: {comp.name}\n"
                f"   Statement: {comp.statement[:200]}..."
            )

        prompt = f"""Analyze these mathematical components and identify TRUE prerequisite dependencies.

COMPONENTS:
{chr(10).join(component_descriptions)}

TASK:
For each component, identify which OTHER components must be formalized FIRST (true prerequisites).
Only specify dependencies that are NECESSARY for the proof or definition.

OUTPUT FORMAT:
For each component, list its dependencies as comma-separated numbers, or "none" if independent.

1: [dependencies or "none"]
2: [dependencies or "none"]
...

Provide dependencies for all {len(components)} components:"""

        try:
            # Use math_query for dependency analysis
            result = await self.leanaide_client.math_query(
                query=prompt,
                n=1
            )

            if result.success and result.data:
                # Parse LLM response
                return self._parse_dependency_response(result.data, components)
            else:
                return self._identify_dependencies_heuristic(components)

        except (IOError, ConnectionError, TimeoutError, ValueError) as e:
            self.logger.error(f"LLM dependency analysis failed: {e}")
            return self._identify_dependencies_heuristic(components)

    def _parse_dependency_response(
        self,
        llm_data: Dict[str, Any],
        components: List[MathematicalComponent]
    ) -> Dict[str, List[str]]:
        """Parse LLM dependency response."""
        dependencies = {}

        # Extract answers from LLM response
        answers = llm_data.get("result", llm_data.get("answers", []))
        if not isinstance(answers, list):
            answers = [answers]

        if not answers:
            return {c.component_id: [] for c in components}

        # Parse the first answer
        response_text = str(answers[0])

        # Create ID mapping
        id_map = {i+1: c.component_id for i, c in enumerate(components)}

        # Parse dependencies from response
        lines = response_text.split("\n")
        for line in lines:
            match = re.match(r"(\d+)\s*:\s*(.+)", line.strip())
            if match:
                num_str = int(match.group(1))
                deps_str = match.group(2).strip().lower()

                component_id = id_map.get(num_str)
                if not component_id:
                    continue

                if deps_str in ["none", "n/a", ""]:
                    dependencies[component_id] = []
                else:
                    # Extract dependency numbers
                    dep_nums = [int(d) for d in re.findall(r"\d+", deps_str)]
                    dep_ids = [id_map[d] for d in dep_nums if d in id_map and d != num_str]
                    dependencies[component_id] = dep_ids

        # Fill in missing components
        for component in components:
            if component.component_id not in dependencies:
                dependencies[component.component_id] = []

        return dependencies

    def _identify_dependencies_heuristic(
        self,
        components: List[MathematicalComponent]
    ) -> Dict[str, List[str]]:
        """Identify dependencies using heuristic analysis."""
        dependencies = {}

        for i, component in enumerate(components):
            deps = []

            # Check for references to other components
            for j, other in enumerate(components):
                if i == j:
                    continue

                # Check name references
                if other.name.lower() in component.statement.lower():
                    deps.append(other.component_id)

                # Check type-based dependencies (definitions before theorems)
                if other.type == ComponentType.DEFINITION and component.type in [ComponentType.THEOREM, ComponentType.LEMMA]:
                    # Check if component uses concepts from definition
                    if self._shares_concepts(component, other):
                        deps.append(other.component_id)

            dependencies[component.component_id] = list(set(deps))

        return dependencies

    def _shares_concepts(
        self,
        component1: MathematicalComponent,
        component2: MathematicalComponent
    ) -> bool:
        """Check if two components share mathematical concepts."""
        # Extract keywords from both statements
        words1 = set(re.findall(r"\b\w+\b", component1.statement.lower()))
        words2 = set(re.findall(r"\b\w+\b", component2.statement.lower()))

        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "of", "in", "for", "to", "is", "are", "be"}
        words1 -= common_words
        words2 -= common_words

        # Check for significant overlap
        overlap = words1 & words2
        overlap_ratio = len(overlap) / max(len(words1), len(words2), 1)

        return overlap_ratio > 0.2

    async def _estimate_complexity(
        self,
        component: MathematicalComponent
    ) -> int:
        """
        Estimate formalization complexity for a component.

        Returns complexity score from 1 (trivial) to 10 (extremely complex).
        """
        base_complexity = 5

        # Length factor
        length = len(component.statement)
        if length > 500:
            base_complexity += 2
        elif length > 200:
            base_complexity += 1

        # Type factor
        type_complexity = {
            ComponentType.DEFINITION: 3,
            ComponentType.THEOREM: 6,
            ComponentType.LEMMA: 4,
            ComponentType.PROPOSITION: 5,
            ComponentType.COROLLARY: 3,
            ComponentType.EXAMPLE: 2,
            ComponentType.AXIOM: 1,
            ComponentType.CONJECTURE: 8,
        }
        base_complexity = (base_complexity + type_complexity.get(component.type, 5)) // 2

        # Dependency factor
        if component.dependencies:
            base_complexity += min(len(component.dependencies), 2)

        # Domain factor
        domain_complexity = {
            MathematicalDomain.LOGIC: 7,
            MathematicalDomain.SET_THEORY: 6,
            MathematicalDomain.TOPOLOGY: 7,
            MathematicalDomain.ANALYSIS: 8,
            MathematicalDomain.ALGEBRA: 6,
            MathematicalDomain.NUMBER_THEORY: 5,
            MathematicalDomain.COMBINATORICS: 5,
            MathematicalDomain.GEOMETRY: 6,
            MathematicalDomain.GENERAL: 4,
        }
        base_complexity = (base_complexity + domain_complexity.get(component.domain, 5)) // 2

        return min(10, max(1, base_complexity))

    def _suggest_order(
        self,
        components: List[MathematicalComponent],
        dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """
        Suggest optimal order for formalization using topological sort.

        Returns list of component IDs in dependency order.
        """
        # Topological sort (Kahn's algorithm)
        in_degree = {c.component_id: len(c.dependencies) for c in components}
        queue = [c_id for c_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by complexity (do simpler ones first within same level)
            queue.sort(key=lambda x: next(c.complexity for c in components if c.component_id == x))

            node = queue.pop(0)
            result.append(node)

            # Find dependents
            for component_id, deps in dependencies.items():
                if node in deps:
                    in_degree[component_id] -= 1
                    if in_degree[component_id] == 0:
                        queue.append(component_id)

        # Check for cycles
        if len(result) != len(components):
            self.logger.warning("Cycle detected in dependencies, returning partial order")
            # Add remaining components
            for component in components:
                if component.component_id not in result:
                    result.append(component.component_id)

        return result

    def _identify_parallel_groups(
        self,
        component_order: List[str],
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        Identify groups of components that can be formalized in parallel.

        Components at the same dependency level can be done in parallel.
        """
        levels = []
        current_level = []
        current_level_deps = set()

        for component_id in component_order:
            component_deps = set(dependencies.get(component_id, []))

            # Check if dependencies are satisfied by previous levels
            if not component_deps or component_deps.issubset(current_level_deps):
                current_level.append(component_id)
            else:
                # Start new level
                if current_level:
                    levels.append(current_level)
                    current_level_deps.update(current_level)
                current_level = [component_id]

        # Add final level
        if current_level:
            levels.append(current_level)

        # Only return levels with multiple components (actual parallelization)
        return [level for level in levels if len(level) > 1]

    def _suggest_strategy(
        self,
        components: List[MathematicalComponent],
        dependencies: Dict[str, List[str]]
    ) -> str:
        """
        Suggest formalization strategy based on decomposition.

        Returns human-readable strategy description.
        """
        # Analyze characteristics
        has_definitions = any(c.type == ComponentType.DEFINITION for c in components)
        has_multiple_theorems = sum(1 for c in components if c.type == ComponentType.THEOREM) > 1
        avg_complexity = sum(c.complexity for c in components) / len(components)
        dependency_depth = max(len(deps) for deps in dependencies.values())

        strategy_parts = []

        # Base approach
        if has_definitions:
            strategy_parts.append("Start with definitions to establish the formal foundation")

        if dependency_depth > 2:
            strategy_parts.append("Build up complex results through multiple layers of lemmas")
        elif has_multiple_theorems:
            strategy_parts.append("Prove main theorems after establishing necessary lemmas")

        # Complexity approach
        if avg_complexity > 7:
            strategy_parts.append("Use proof automation tactics for complex steps")
        elif avg_complexity < 4:
            strategy_parts.append("Direct proofs should suffice for most components")

        # Parallelization
        parallelizable = sum(1 for deps in dependencies.values() if not deps)
        if parallelizable > 1:
            strategy_parts.append(f"Can parallelize {parallelizable} independent components")

        return ". ".join(strategy_parts) + "."

    def _estimate_overall_complexity(
        self,
        components: List[MathematicalComponent]
    ) -> int:
        """Estimate overall complexity of the formalization."""
        if not components:
            return 0

        # Weighted average (main theorems count more)
        weights = {ComponentType.THEOREM: 1.5, ComponentType.LEMMA: 1.2, ComponentType.DEFINITION: 1.0}
        total_weight = 0.0
        weighted_sum = 0.0

        for component in components:
            weight = weights.get(component.type, 1.0)
            weighted_sum += component.complexity * weight
            total_weight += weight

        avg_complexity = weighted_sum / total_weight if total_weight > 0 else 5

        # Add overhead for integration
        overhead = min(2, len(components) * 0.2)

        return min(10, max(1, int(avg_complexity + overhead)))


# =============================================================================
# COMPONENT EXTRACTOR
# =============================================================================

class LeanComponentExtractor:
    """
    Extracts mathematical components from natural language text.

    This class is responsible for identifying and classifying mathematical
    concepts, theorems, definitions, and their relationships.
    """

    def __init__(
        self,
        leanaide_client: Optional["LeanAideClient"] = None,
        enable_llm: bool = True
    ):
        """
        Initialize component extractor.

        Args:
            leanaide_client: Optional LeanAide client
            enable_llm: Whether to use LLM-based extraction
        """
        self.leanaide_client = leanaide_client
        self.enable_llm = enable_llm and LEANAIDE_AVAILABLE
        self.logger = logging.getLogger(__name__)

    async def extract_components(
        self,
        problem: str
    ) -> List[MathematicalComponent]:
        """
        Extract mathematical components from problem text.

        Args:
            problem: Mathematical problem in natural language

        Returns:
            List of extracted components
        """
        if self.enable_llm and self.leanaide_client:
            return await self._extract_with_llm(problem)
        else:
            return self._extract_heuristic(problem)

    async def _extract_with_llm(
        self,
        problem: str
    ) -> List[MathematicalComponent]:
        """Extract components using LLM via LeanAide."""
        try:
            result = await self.leanaide_client.json_structured(problem)

            if result.success and result.data:
                decomposer = LeanDecomposer(
                    leanaide_client=self.leanaide_client,
                    enable_llm=False
                )
                return decomposer._parse_structured_components(result.data)
            else:
                return self._extract_heuristic(problem)

        except (IOError, ConnectionError, TimeoutError, ValueError) as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return self._extract_heuristic(problem)

    def _extract_heuristic(
        self,
        problem: str
    ) -> List[MathematicalComponent]:
        """Extract components using heuristic analysis."""
        decomposer = LeanDecomposer(enable_llm=False)
        return decomposer._extract_components_heuristic(problem)


# =============================================================================
# SUB-PROBLEM GENERATOR
# =============================================================================

class LeanSubProblemGenerator:
    """
    Generates SubProblems for Lean 4 formalization workflow.

    Takes decomposition plans and converts them into SubProblem objects
    that can be processed by the workflow engine, ROMA, and CREWAI.
    """

    def __init__(
        self,
        leanaide_client: Optional["LeanAideClient"] = None,
        enable_CREWAI: bool = True
    ):
        """
        Initialize sub-problem generator.

        Args:
            leanaide_client: Optional LeanAide client for code generation
            enable_CREWAI: Whether to create CREWAI tickets
        """
        self.leanaide_client = leanaide_client
        self.enable_CREWAI = enable_CREWAI
        self.logger = logging.getLogger(__name__)

    async def generate_lean_subproblems(
        self,
        plan: LeanDecompositionPlan,
        context: Optional[str] = None
    ) -> List[LeanSubProblem]:
        """
        Generate LeanSubProblems from decomposition plan.

        Args:
            plan: Lean decomposition plan
            context: Optional additional context (imports, previous definitions)

        Returns:
            List of LeanSubProblem objects
        """
        subproblems = []
        context = context or ""

        # Generate Lean code for each component
        for component_id in plan.component_order:
            component = next((c for c in plan.components if c.component_id == component_id), None)
            if not component:
                continue

            # Build context for this component
            component_context = self._build_context(component, plan.components, context)

            # Generate Lean code if available
            lean_code = ""
            if self.leanaide_client and not component.lean_code:
                lean_code = await self._generate_lean_code(component, component_context)
            else:
                lean_code = component.lean_code

            # Extract imports
            imports = self._extract_imports(component, component_context)

            # Create sub-problem
            subproblem = LeanSubProblem(
                component=component,
                context=component_context,
                imports=imports,
                lean_code=lean_code,
                status=LeanProofStatus.PENDING
            )

            subproblems.append(subproblem)

        return subproblems

    def _build_context(
        self,
        component: MathematicalComponent,
        all_components: List[MathematicalComponent],
        base_context: str
    ) -> str:
        """Build formalization context for a component."""
        context_parts = [base_context]

        # Add dependencies
        for dep_id in component.dependencies:
            dep_component = next((c for c in all_components if c.component_id == dep_id), None)
            if dep_component and dep_component.lean_code:
                context_parts.append(f"-- {dep_component.type.value.title()}: {dep_component.name}")
                context_parts.append(dep_component.lean_code)

        return "\n\n".join(context_parts)

    async def _generate_lean_code(
        self,
        component: MathematicalComponent,
        context: str
    ) -> str:
        """Generate Lean code for a component using LeanAide."""
        if not self.leanaide_client:
            return ""

        try:
            # Choose appropriate task
            if component.type == ComponentType.DEFINITION:
                result = await self.leanaide_client.translate_def(component.statement)
            else:
                result = await self.leanaide_client.translate_thm_detailed(
                    component.statement,
                    theorem_name=component.name
                )

            if result.success and result.data:
                # Extract Lean code from response
                return self._extract_lean_code_from_result(result.data)
            else:
                self.logger.warning(f"Lean code generation failed for {component.name}: {result.error}")
                return ""

        except (IOError, ConnectionError, TimeoutError, ValueError) as e:
            self.logger.error(f"Lean code generation error for {component.name}: {e}")
            return ""

    def _extract_lean_code_from_result(self, data: Dict[str, Any]) -> str:
        """Extract Lean code from LeanAide result."""
        # Try various possible fields
        for field in ["lean_code", "code", "result", "output", "theorem"]:
            if field in data:
                code = data[field]
                if isinstance(code, str) and code.strip():
                    return code.strip()

        return ""

    def _extract_imports(
        self,
        component: MathematicalComponent,
        context: str
    ) -> List[str]:
        """Extract required Lean imports."""
        imports = []

        # Domain-specific imports
        domain_imports = {
            MathematicalDomain.ALGEBRA: ["Mathlib.Algebra.*", "Mathlib.Data.*"],
            MathematicalDomain.ANALYSIS: ["Mathlib.Analysis.*", "Mathlib.Topology.*"],
            MathematicalDomain.TOPOLOGY: ["Mathlib.Topology.*", "Mathlib.Order.*"],
            MathematicalDomain.NUMBER_THEORY: ["Mathlib.Data.Nat.*", "Mathlib.Data.Int.*", "Mathlib.NumberTheory.*"],
            MathematicalDomain.COMBINATORICS: ["Mathlib.Combinatorics.*"],
            MathematicalDomain.GEOMETRY: ["Mathlib.Geometry.*"],
            MathematicalDomain.LOGIC: ["Mathlib.Logic.*"],
            MathematicalDomain.SET_THEORY: ["Mathlib.SetTheory.*", "Mathlib.Data.Set.*"],
        }

        domain_imports_list = domain_imports.get(component.domain, [])
        imports.extend(domain_imports_list)

        # Extract imports from context
        import_pattern = r"import\s+(\S+)"
        imports_in_context = re.findall(import_pattern, context)
        imports.extend(imports_in_context)

        # Deduplicate
        return list(set(imports))

    async def convert_to_subproblems(
        self,
        lean_subproblems: List[LeanSubProblem]
    ) -> List["SubProblem"]:
        """
        Convert LeanSubProblems to base SubProblem objects.

        Args:
            lean_subproblems: List of LeanSubProblem objects

        Returns:
            List of SubProblem objects for workflow integration
        """
        return [lsp.to_subproblem() for lsp in lean_subproblems]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def decompose_and_generate_subproblems(
    problem: str,
    leanaide_client: Optional["LeanAideClient"] = None,
    strategy: DecompositionStrategy = DecompositionStrategy.HYBRID
) -> Tuple[LeanDecompositionPlan, List["SubProblem"]]:
    """
    High-level function to decompose problem and generate subproblems.

    Args:
        problem: Mathematical problem statement
        leanaide_client: Optional LeanAide client
        strategy: Decomposition strategy

    Returns:
        Tuple of (decomposition plan, subproblems for workflow)
    """
    async with LeanDecomposer(leanaide_client=leanaide_client) as decomposer:
        # Decompose problem
        plan = await decomposer.decompose_mathematical_problem(problem, strategy)

        # Generate subproblems
        generator = LeanSubProblemGenerator(leanaide_client=leanaide_client)
        lean_subproblems = await generator.generate_lean_subproblems(plan)

        # Convert to workflow subproblems
        subproblems = await generator.convert_to_subproblems(lean_subproblems)

        return plan, subproblems


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of LeanAide decomposition integration."""
    # Example problem
    problem = """
    Theorem: There are infinitely many prime numbers.

    Proof:
    Assume there are finitely many primes p1, p2, ..., pn.
    Consider N = p1 * p2 * ... * pn + 1.
    Then N is not divisible by any of the primes p1, ..., pn.
    Therefore N must be prime or divisible by a prime not in our list.
    This contradicts the assumption that p1, ..., pn are all the primes.
    Hence there are infinitely many primes.
    """

    # Create decomposition plan
    async with LeanDecomposer() as decomposer:
        plan = await decomposer.decompose_mathematical_problem(problem)

        print(f"Decomposition Plan:")
        print(f"  Components: {len(plan.components)}")
        print(f"  Complexity: {plan.complexity_estimate}/10")
        print(f"  Strategy: {plan.formalization_strategy}")
        print()

        for component_id in plan.component_order:
            component = next(c for c in plan.components if c.component_id == component_id)
            print(f"  - {component.type.value.title()}: {component.name}")
            print(f"    Complexity: {component.complexity}/10")
            print(f"    Dependencies: {len(component.dependencies)}")
            print()

        # Generate subproblems
        generator = LeanSubProblemGenerator()
        lean_subproblems = await generator.generate_lean_subproblems(plan)

        print(f"Generated {len(lean_subproblems)} subproblems for formalization")


if __name__ == "__main__":
    asyncio.run(main())
