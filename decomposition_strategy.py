"""
decomposition_strategy.py - Sovereign Decomposition Strategy Implementation

Production-ready implementation of three decomposition strategies:
1. HYBRID: Combined multi-technique approach
2. ROMA: Hierarchical recursive decomposition
3. SEMANTIC: Meaning-based grouping and clustering

This module provides intelligent strategy selection and execution with
comprehensive error handling, type hints, and production safeguards.

Author: Sovereign System
Date: 2026-01-21
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class SovereignDecompositionStrategy(Enum):
    """Decomposition strategy types for Sovereign system."""
    HYBRID = "HYBRID"
    ROMA = "ROMA"
    SEMANTIC = "SEMANTIC"


class ProblemComplexity(Enum):
    """Problem complexity levels."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class DomainType(Enum):
    """Domain types for problems."""
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_SCIENCE = "data_science"
    RESEARCH = "research"
    OPERATIONS = "operations"
    BUSINESS = "business"
    GENERAL = "general"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ComplexityScore:
    """Complexity score for problems and sub-problems."""
    explanation: str
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    overall_complexity: float

    def __post_init__(self):
        """Validate complexity scores are in valid range."""
        for score_field in ['cognitive_complexity', 'computational_complexity',
                           'domain_complexity', 'integration_complexity',
                           'overall_complexity']:
            score = getattr(self, score_field)
            if not 0.0 <= score <= 10.0:
                raise ValueError(f"{score_field} must be between 0.0 and 10.0, got {score}")


@dataclass
class DependencyGraph:
    """Dependency graph for sub-problems."""
    nodes: Dict[str, Any] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a dependency edge from one node to another."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order."""
        if self.execution_order:
            return self.execution_order

        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        all_nodes = set(self.nodes.keys()) | set(self.edges.keys())

        for node in all_nodes:
            in_degree[node] = 0

        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                in_degree[to_node] += 1

        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            if node in self.edges:
                for neighbor in self.edges[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        if len(result) != len(all_nodes):
            logger.warning("Cycle detected in dependency graph")
            return list(all_nodes)

        self.execution_order = result
        return result


# Import from sovereign_data_models
try:
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        ProblemStatus,
        generate_id
    )
except ImportError:
    # Fallback definitions
    logger.warning("sovereign_data_models not available, using fallback definitions")

    class ProblemStatus(Enum):
        """Status of a problem in the sovereign system."""
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        BLOCKED = "blocked"
        FAILED = "failed"

    @dataclass
    class SubProblem:
        """A sub-problem in the decomposition."""
        sub_problem_id: str
        parent_id: Optional[str]
        title: str
        description: str
        status: ProblemStatus
        confidence: float
        assigned_agent: Optional[str]
        created_at: datetime
        completed_at: Optional[datetime]
        result: Optional[Any] = None

    @dataclass
    class ProblemDefinition:
        """Definition of a problem to be solved."""
        problem_id: str
        title: str
        description: str
        domain: str
        complexity: str
        priority: str
        estimated_effort: str
        requirements: List[str]
        constraints: List[str]
        created_at: datetime

    @dataclass
    class DecompositionPlan:
        """A complete decomposition plan."""
        plan_id: str
        problem: ProblemDefinition
        sub_problems: List[SubProblem]
        dependencies: Dict[str, List[str]]
        execution_order: List[str]
        created_at: datetime
        modified_at: datetime
        status: ProblemStatus

    def generate_id(prefix: str = "") -> str:
        """Generate a unique ID with optional prefix."""
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{unique_id}" if prefix else unique_id


# ============================================================================
# ABSTRACT STRATEGY BASE
# ============================================================================

class DecompositionStrategyBase(ABC):
    """Abstract base class for decomposition strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy with optional configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def decompose(
        self,
        problem: ProblemDefinition,
        **kwargs
    ) -> DecompositionPlan:
        """
        Decompose a problem into sub-problems.

        Args:
            problem: The problem definition to decompose
            **kwargs: Additional strategy-specific parameters

        Returns:
            DecompositionPlan with sub-problems and dependencies
        """
        raise NotImplementedError("Subclasses must implement decompose()")

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        raise NotImplementedError("Subclasses must implement get_strategy_name()")

    def validate_problem(self, problem: ProblemDefinition) -> bool:
        """
        Validate that problem definition is complete.

        Args:
            problem: Problem definition to validate

        Returns:
            True if valid, False otherwise
        """
        if not problem.description or not problem.title:
            self.logger.error("Problem must have title and description")
            return False
        if not problem.domain:
            self.logger.warning("Problem has no domain specified")
        return True

    def calculate_complexity_score(
        self,
        problem: ProblemDefinition,
        description_length: int,
        num_requirements: int
    ) -> ComplexityScore:
        """
        Calculate complexity score for a problem.

        Args:
            problem: Problem definition
            description_length: Length of description text
            num_requirements: Number of requirements

        Returns:
            ComplexityScore object
        """
        # Cognitive complexity based on text analysis
        cognitive = min(10.0, description_length / 200.0 + num_requirements * 0.5)

        # Computational complexity based on domain
        domain_multipliers = {
            "software_engineering": 1.2,
            "data_science": 1.3,
            "research": 1.1,
            "operations": 0.8,
            "business": 0.7,
        }
        multiplier = domain_multipliers.get(problem.domain.lower(), 1.0)
        computational = min(10.0, cognitive * multiplier)

        # Domain complexity
        domain_complexity = min(10.0, computational * 0.9)

        # Integration complexity based on constraints
        integration = min(10.0, len(problem.constraints) * 0.5 + 2.0)

        # Overall complexity
        overall = (cognitive + computational + domain_complexity + integration) / 4.0

        return ComplexityScore(
            explanation=f"Complexity calculated from text analysis and domain factors",
            cognitive_complexity=round(cognitive, 2),
            computational_complexity=round(computational, 2),
            domain_complexity=round(domain_complexity, 2),
            integration_complexity=round(integration, 2),
            overall_complexity=round(overall, 2)
        )


# ============================================================================
# HYBRID STRATEGY IMPLEMENTATION
# ============================================================================

class HybridDecompositionStrategy(DecompositionStrategyBase):
    """
    HYBRID Strategy: Combined multi-technique approach

    Combines hierarchical, semantic, and functional decomposition techniques
    for comprehensive problem breakdown. Best for complex, multi-faceted problems.

    Features:
    - Multi-technique decomposition (hierarchical + semantic + functional)
    - Intelligent strategy blending based on problem characteristics
    - Redundancy checking across techniques
    - Quality-weighted sub-problem selection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_depth = self.config.get('max_depth', 3)
        self.min_subproblems = self.config.get('min_subproblems', 3)
        self.max_subproblems = self.config.get('max_subproblems', 10)

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return SovereignDecompositionStrategy.HYBRID.value

    def decompose(
        self,
        problem: ProblemDefinition,
        depth: int = 3,
        **kwargs
    ) -> DecompositionPlan:
        """
        Decompose problem using hybrid approach.

        Args:
            problem: Problem to decompose
            depth: Maximum decomposition depth
            **kwargs: Additional parameters

        Returns:
            DecompositionPlan with hybrid approach
        """
        if not self.validate_problem(problem):
            raise ValueError("Invalid problem definition")

        self.logger.info(f"Starting HYBRID decomposition for problem: {problem.title}")

        depth = min(depth, self.max_depth)
        sub_problems = []
        dependencies = defaultdict(list)

        # Step 1: Analyze problem structure
        phases = self._identify_phases(problem)
        components = self._identify_components(problem)
        aspects = self._identify_aspects(problem)

        # Step 2: Generate sub-problems from multiple perspectives
        phase_problems = self._create_phase_problems(problem, phases)
        component_problems = self._create_component_problems(problem, components)
        aspect_problems = self._create_aspect_problems(problem, aspects)

        # Step 3: Merge and deduplicate sub-problems
        merged_problems = self._merge_sub_problems(
            phase_problems + component_problems + aspect_problems
        )

        # Step 4: Apply depth-based recursive decomposition if needed
        if depth > 1:
            for sub_prob in merged_problems:
                if self._should_decompose_further(sub_prob, depth):
                    # For shallow recursion, just add metadata
                    sub_prob.description += f"\n\nNote: Further decomposition possible at depth {depth - 1}"

        # Step 5: Identify dependencies
        dependencies = self._identify_dependencies(merged_problems, problem)

        # Step 6: Create execution order
        dep_graph = DependencyGraph(
            nodes={sp.sub_problem_id: sp for sp in merged_problems},
            edges=dependencies
        )
        execution_order = dep_graph.get_execution_order()

        # Step 7: Create decomposition plan
        plan = DecompositionPlan(
            plan_id=generate_id("hybrid_plan"),
            problem=problem,
            sub_problems=merged_problems,
            dependencies=dict(dependencies),
            execution_order=execution_order,
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            status=ProblemStatus.PENDING
        )

        self.logger.info(f"Created HYBRID plan with {len(merged_problems)} sub-problems")
        return plan

    def _identify_phases(self, problem: ProblemDefinition) -> List[str]:
        """Identify temporal phases in the problem."""
        phases = []
        desc_lower = problem.description.lower()

        # Common phase indicators
        phase_keywords = {
            'planning': ['plan', 'design', 'architecture', 'specification'],
            'implementation': ['implement', 'develop', 'code', 'build', 'create'],
            'testing': ['test', 'verify', 'validate', 'check'],
            'deployment': ['deploy', 'release', 'ship', 'deliver'],
            'maintenance': ['maintain', 'update', 'monitor', 'support']
        }

        for phase, keywords in phase_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                phases.append(phase)

        return phases if phases else ['execution']

    def _identify_components(self, problem: ProblemDefinition) -> List[str]:
        """Identify functional components in the problem."""
        components = []
        desc_lower = problem.description.lower()

        # Common component indicators
        component_patterns = [
            r'(\w+)\s+module',
            r'(\w+)\s+component',
            r'(\w+)\s+service',
            r'(\w+)\s+system',
            r'(\w+)\s+interface'
        ]

        for pattern in component_patterns:
            matches = re.findall(pattern, desc_lower)
            components.extend(matches)

        # If no explicit components, extract from requirements
        if not components:
            for req in problem.requirements[:5]:
                # Extract nouns/components from requirements
                words = req.split()
                if len(words) > 2:
                    components.append(words[0].lower())

        return list(set(components)) if components else ['core']

    def _identify_aspects(self, problem: ProblemDefinition) -> List[str]:
        """Identify cross-cutting aspects."""
        aspects = []
        desc_lower = problem.description.lower()

        aspect_keywords = {
            'security': ['security', 'authentication', 'authorization', 'encryption'],
            'performance': ['performance', 'optimization', 'scalability', 'efficiency'],
            'usability': ['user experience', 'interface', 'usability', 'accessible'],
            'data': ['data', 'database', 'storage', 'persistence'],
            'integration': ['api', 'integration', 'interface', 'communication']
        }

        for aspect, keywords in aspect_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                aspects.append(aspect)

        return aspects if aspects else ['functionality']

    def _create_phase_problems(
        self,
        problem: ProblemDefinition,
        phases: List[str]
    ) -> List[SubProblem]:
        """Create sub-problems based on temporal phases."""
        sub_problems = []

        for i, phase in enumerate(phases):
            sub_problem = SubProblem(
                sub_problem_id=generate_id(f"phase_{phase}"),
                parent_id=problem.problem_id,
                title=f"{phase.title()} Phase",
                description=f"Execute {phase} activities for: {problem.title}",
                status=ProblemStatus.PENDING,
                confidence=0.8,
                assigned_agent=None,
                created_at=datetime.utcnow(),
                completed_at=None
            )
            sub_problems.append(sub_problem)

        return sub_problems

    def _create_component_problems(
        self,
        problem: ProblemDefinition,
        components: List[str]
    ) -> List[SubProblem]:
        """Create sub-problems based on functional components."""
        sub_problems = []

        for component in components:
            sub_problem = SubProblem(
                sub_problem_id=generate_id(f"comp_{component}"),
                parent_id=problem.problem_id,
                title=f"{component.title()} Component",
                description=f"Implement {component} component for: {problem.title}",
                status=ProblemStatus.PENDING,
                confidence=0.85,
                assigned_agent=None,
                created_at=datetime.utcnow(),
                completed_at=None
            )
            sub_problems.append(sub_problem)

        return sub_problems

    def _create_aspect_problems(
        self,
        problem: ProblemDefinition,
        aspects: List[str]
    ) -> List[SubProblem]:
        """Create sub-problems based on cross-cutting aspects."""
        sub_problems = []

        for aspect in aspects:
            sub_problem = SubProblem(
                sub_problem_id=generate_id(f"aspect_{aspect}"),
                parent_id=problem.problem_id,
                title=f"{aspect.title()} Aspect",
                description=f"Address {aspect} concerns for: {problem.title}",
                status=ProblemStatus.PENDING,
                confidence=0.75,
                assigned_agent=None,
                created_at=datetime.utcnow(),
                completed_at=None
            )
            sub_problems.append(sub_problem)

        return sub_problems

    def _merge_sub_problems(
        self,
        all_problems: List[SubProblem]
    ) -> List[SubProblem]:
        """Merge and deduplicate sub-problems."""
        # Use title similarity to detect duplicates
        seen = set()
        merged = []

        for sp in all_problems:
            # Normalize title for comparison
            title_key = sp.title.lower().replace(' ', '')
            if title_key not in seen:
                seen.add(title_key)
                merged.append(sp)

        # Limit to max_subproblems
        if len(merged) > self.max_subproblems:
            # Keep high confidence problems
            merged.sort(key=lambda x: x.confidence, reverse=True)
            merged = merged[:self.max_subproblems]

        # Ensure minimum subproblems
        if len(merged) < self.min_subproblems:
            self.logger.warning(
                f"Only {len(merged)} sub-problems generated, "
                f"below recommended minimum of {self.min_subproblems}"
            )

        return merged

    def _should_decompose_further(
        self,
        sub_problem: SubProblem,
        current_depth: int
    ) -> bool:
        """Determine if sub-problem should be decomposed further."""
        # Don't decompose if at max depth
        if current_depth >= self.max_depth:
            return False

        # Decompose if description is long (indicates complexity)
        if len(sub_problem.description) > 500:
            return True

        return False

    def _identify_dependencies(
        self,
        sub_problems: List[SubProblem],
        problem: ProblemDefinition
    ) -> Dict[str, List[str]]:
        """Identify dependencies between sub-problems."""
        dependencies = defaultdict(list)

        # Phase-based dependencies
        phase_order = ['planning', 'implementation', 'testing', 'deployment', 'maintenance']
        phase_problems = {sp.title.split()[0].lower(): sp.sub_problem_id
                         for sp in sub_problems if 'phase' in sp.title.lower()}

        for i, phase in enumerate(phase_order):
            if phase in phase_problems and i > 0:
                prev_phase = phase_order[i - 1]
                if prev_phase in phase_problems:
                    dependencies[phase_problems[prev_phase]].append(phase_problems[phase])

        # Component dependencies (components might depend on each other)
        comp_problems = [sp for sp in sub_problems if 'component' in sp.title.lower()]
        for i, comp in enumerate(comp_problems):
            if i > 0:
                dependencies[comp_problems[i-1].sub_problem_id].append(comp.sub_problem_id)

        return dict(dependencies)


# ============================================================================
# ROMA STRATEGY IMPLEMENTATION
# ============================================================================

class RomadecompositionStrategy(DecompositionStrategyBase):
    """
    ROMA Strategy: Hierarchical Recursive Decomposition

    Performs recursive, hierarchical breakdown of problems until
    atomic units are reached. Named after ROMA (Recursive Object-based
    Multi-level Abstraction).

    Features:
    - Recursive hierarchical breakdown
    - Atomic unit detection
    - Depth-limited decomposition
    - Parent-child relationship tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_depth = self.config.get('max_depth', 5)
        self.atomic_threshold = self.config.get('atomic_threshold', 100)

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return SovereignDecompositionStrategy.ROMA.value

    def decompose(
        self,
        problem: ProblemDefinition,
        max_depth: int = 5,
        **kwargs
    ) -> DecompositionPlan:
        """
        Decompose problem using ROMA hierarchical approach.

        Args:
            problem: Problem to decompose
            max_depth: Maximum recursion depth
            **kwargs: Additional parameters

        Returns:
            DecompositionPlan with ROMA hierarchical structure
        """
        if not self.validate_problem(problem):
            raise ValueError("Invalid problem definition")

        self.logger.info(f"Starting ROMA decomposition for problem: {problem.title}")

        max_depth = min(max_depth, self.max_depth)
        sub_problems = []

        # Recursive decomposition
        self._recursive_decompose(
            problem=problem,
            parent_id=None,
            current_depth=0,
            max_depth=max_depth,
            sub_problems=sub_problems
        )

        # Identify dependencies
        dependencies = self._build_hierarchical_dependencies(sub_problems)

        # Create execution order (breadth-first for hierarchy)
        execution_order = self._breadth_first_order(sub_problems)

        # Create decomposition plan
        plan = DecompositionPlan(
            plan_id=generate_id("roma_plan"),
            problem=problem,
            sub_problems=sub_problems,
            dependencies=dependencies,
            execution_order=execution_order,
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            status=ProblemStatus.PENDING
        )

        self.logger.info(
            f"Created ROMA plan with {len(sub_problems)} sub-problems "
            f"at max depth {max_depth}"
        )
        return plan

    def _recursive_decompose(
        self,
        problem: ProblemDefinition,
        parent_id: Optional[str],
        current_depth: int,
        max_depth: int,
        sub_problems: List[SubProblem]
    ) -> None:
        """
        Recursively decompose problem into hierarchy.

        Args:
            problem: Problem to decompose
            parent_id: Parent sub-problem ID
            current_depth: Current recursion depth
            max_depth: Maximum allowed depth
            sub_problems: Accumulator for sub-problems
        """
        # Check if we should stop decomposition
        if current_depth >= max_depth or self._is_atomic(problem):
            # Create atomic sub-problem
            atomic_sub = SubProblem(
                sub_problem_id=generate_id("atomic"),
                parent_id=parent_id,
                title=problem.title,
                description=problem.description,
                status=ProblemStatus.PENDING,
                confidence=0.95,
                assigned_agent=None,
                created_at=datetime.utcnow(),
                completed_at=None
            )
            sub_problems.append(atomic_sub)
            return

        # Decompose into children
        children = self._decompose_into_children(problem, current_depth)

        for child_def in children:
            # Create sub-problem for child
            sub_problem = SubProblem(
                sub_problem_id=child_def['id'],
                parent_id=parent_id,
                title=child_def['title'],
                description=child_def['description'],
                status=ProblemStatus.PENDING,
                confidence=child_def.get('confidence', 0.8),
                assigned_agent=None,
                created_at=datetime.utcnow(),
                completed_at=None
            )
            sub_problems.append(sub_problem)

            # Recurse into child
            child_problem = ProblemDefinition(
                problem_id=child_def['id'],
                title=child_def['title'],
                description=child_def['description'],
                domain=problem.domain,
                complexity=problem.complexity,
                priority=problem.priority,
                estimated_effort=problem.estimated_effort,
                requirements=child_def.get('requirements', []),
                constraints=problem.constraints,
                created_at=datetime.utcnow()
            )

            self._recursive_decompose(
                problem=child_problem,
                parent_id=child_def['id'],
                current_depth=current_depth + 1,
                max_depth=max_depth,
                sub_problems=sub_problems
            )

    def _is_atomic(self, problem: ProblemDefinition) -> bool:
        """
        Check if problem is atomic (cannot be decomposed further).

        Args:
            problem: Problem to check

        Returns:
            True if atomic, False otherwise
        """
        # Check description length
        if len(problem.description) <= self.atomic_threshold:
            return True

        # Check for decomposition keywords
        atomic_indicators = ['implement', 'create', 'build', 'write']
        has_atomic_indicator = any(
            indicator in problem.description.lower()
            for indicator in atomic_indicators
        )

        # Check for structural indicators
        structural_indicators = ['component', 'module', 'system', 'part']
        has_structure = any(
            indicator in problem.description.lower()
            for indicator in structural_indicators
        )

        return has_atomic_indicator and not has_structure

    def _decompose_into_children(
        self,
        problem: ProblemDefinition,
        current_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Decompose problem into child sub-problems.

        Args:
            problem: Problem to decompose
            current_depth: Current depth level

        Returns:
            List of child problem definitions
        """
        children = []
        description = problem.description

        # Strategy 1: Sentence-based decomposition
        sentences = re.split(r'[.!?]+', description)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group sentences into logical chunks
        chunk_size = max(1, len(sentences) // (3 - current_depth // 2))

        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = '. '.join(chunk_sentences)

            # Extract title from first sentence
            first_words = chunk_sentences[0].split()[:4]
            title = ' '.join(first_words).title()

            child = {
                'id': generate_id(f"child_{i}"),
                'title': title,
                'description': chunk_text,
                'confidence': 0.8,
                'requirements': problem.requirements[:2]  # Distribute requirements
            }
            children.append(child)

        # Ensure we have at least 2 children
        if len(children) < 2:
            # Fallback: split by requirements
            for i, req in enumerate(problem.requirements):
                child = {
                    'id': generate_id(f"req_{i}"),
                    'title': f"Requirement {i + 1}",
                    'description': req,
                    'confidence': 0.85,
                    'requirements': [req]
                }
                children.append(child)

        return children

    def _build_hierarchical_dependencies(
        self,
        sub_problems: List[SubProblem]
    ) -> Dict[str, List[str]]:
        """
        Build dependencies based on hierarchical structure.

        Args:
            sub_problems: List of all sub-problems

        Returns:
            Dependency mapping
        """
        dependencies = defaultdict(list)

        # Build parent-child map
        children_by_parent = defaultdict(list)
        for sp in sub_problems:
            if sp.parent_id:
                children_by_parent[sp.parent_id].append(sp.sub_problem_id)

        # Parent must complete before children can start
        for parent_id, child_ids in children_by_parent.items():
            for child_id in child_ids:
                dependencies[parent_id].append(child_id)

        return dict(dependencies)

    def _breadth_first_order(
        self,
        sub_problems: List[SubProblem]
    ) -> List[str]:
        """
        Generate breadth-first execution order.

        Args:
            sub_problems: List of sub-problems

        Returns:
            Ordered list of sub-problem IDs
        """
        # Group by depth (level in hierarchy)
        levels = defaultdict(list)
        depth_map = {}

        # Calculate depth for each sub-problem
        for sp in sub_problems:
            if sp.parent_id is None:
                depth = 0
            else:
                depth = depth_map.get(sp.parent_id, 0) + 1

            depth_map[sp.sub_problem_id] = depth
            levels[depth].append(sp.sub_problem_id)

        # Order by depth (breadth-first)
        execution_order = []
        for depth in sorted(levels.keys()):
            execution_order.extend(levels[depth])

        return execution_order


# ============================================================================
# SEMANTIC STRATEGY IMPLEMENTATION
# ============================================================================

class SemanticDecompositionStrategy(DecompositionStrategyBase):
    """
    SEMANTIC Strategy: Meaning-based Grouping and Clustering

    Decomposes problems based on semantic meaning, concepts, and themes.
    Uses keyword analysis and conceptual clustering for intelligent grouping.

    Features:
    - Concept-based clustering
    - Theme extraction
    - Semantic similarity analysis
    - Natural language processing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_clusters = self.config.get('num_clusters', 5)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return SovereignDecompositionStrategy.SEMANTIC.value

    def decompose(
        self,
        problem: ProblemDefinition,
        clusters: int = 5,
        **kwargs
    ) -> DecompositionPlan:
        """
        Decompose problem using semantic clustering.

        Args:
            problem: Problem to decompose
            clusters: Number of semantic clusters to create
            **kwargs: Additional parameters

        Returns:
            DecompositionPlan with semantic clustering
        """
        if not self.validate_problem(problem):
            raise ValueError("Invalid problem definition")

        self.logger.info(f"Starting SEMANTIC decomposition for problem: {problem.title}")

        clusters = min(clusters, self.num_clusters)

        # Step 1: Extract concepts from problem
        concepts = self._extract_concepts(problem)

        # Step 2: Group concepts into semantic clusters
        semantic_groups = self._cluster_concepts(concepts, clusters)

        # Step 3: Create sub-problems from clusters
        sub_problems = self._create_clustered_sub_problems(
            problem,
            semantic_groups
        )

        # Step 4: Identify semantic dependencies
        dependencies = self._identify_semantic_dependencies(
            sub_problems,
            semantic_groups
        )

        # Step 5: Create execution order
        execution_order = self._semantic_execution_order(sub_problems, dependencies)

        # Create decomposition plan
        plan = DecompositionPlan(
            plan_id=generate_id("semantic_plan"),
            problem=problem,
            sub_problems=sub_problems,
            dependencies=dependencies,
            execution_order=execution_order,
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            status=ProblemStatus.PENDING
        )

        self.logger.info(
            f"Created SEMANTIC plan with {len(sub_problems)} sub-problems "
            f"in {clusters} clusters"
        )
        return plan

    def _extract_concepts(self, problem: ProblemDefinition) -> List[Dict[str, Any]]:
        """
        Extract key concepts from problem definition.

        Args:
            problem: Problem to analyze

        Returns:
            List of concepts with metadata
        """
        concepts = []

        # Extract from description
        desc_words = re.findall(r'\b\w+\b', problem.description.lower())
        desc_freq = defaultdict(int)
        for word in desc_words:
            if len(word) > 3:  # Ignore short words
                desc_freq[word] += 1

        # Extract from requirements
        req_concepts = []
        for req in problem.requirements:
            req_words = re.findall(r'\b\w+\b', req.lower())
            req_concepts.extend([w for w in req_words if len(w) > 3])

        # Extract from constraints
        constraint_concepts = []
        for constraint in problem.constraints:
            constraint_words = re.findall(r'\b\w+\b', constraint.lower())
            constraint_concepts.extend([w for w in constraint_words if len(w) > 3])

        # Build concept list
        all_words = list(set(desc_words + req_concepts + constraint_concepts))

        # Domain-specific concept categories
        domain_themes = {
            'software_engineering': ['architecture', 'design', 'implementation', 'testing', 'deployment'],
            'data_science': ['analysis', 'modeling', 'training', 'evaluation', 'deployment'],
            'research': ['investigation', 'analysis', 'experimentation', 'validation', 'documentation'],
            'operations': ['planning', 'execution', 'monitoring', 'optimization', 'maintenance']
        }

        themes = domain_themes.get(problem.domain.lower(), [])
        concept_id = 0

        for word in all_words:
            # Calculate relevance score
            freq_score = desc_freq.get(word, 0) / len(desc_words) if desc_words else 0
            theme_score = 1.0 if word in themes else 0.5

            concept = {
                'id': f"concept_{concept_id}",
                'word': word,
                'frequency': desc_freq.get(word, 0),
                'relevance': freq_score + theme_score,
                'category': self._categorize_concept(word, themes)
            }
            concepts.append(concept)
            concept_id += 1

        # Sort by relevance and take top concepts
        concepts.sort(key=lambda x: x['relevance'], reverse=True)
        return concepts[:20]  # Top 20 concepts

    def _categorize_concept(self, word: str, themes: List[str]) -> str:
        """Categorize a concept into a theme."""
        word_lower = word.lower()

        categories = {
            'action': ['create', 'build', 'implement', 'develop', 'design'],
            'object': ['system', 'module', 'component', 'service', 'interface'],
            'quality': ['performance', 'security', 'reliability', 'scalability', 'usability'],
            'process': ['analyze', 'test', 'validate', 'deploy', 'maintain']
        }

        for category, keywords in categories.items():
            if any(keyword in word_lower for keyword in keywords):
                return category

        return 'general'

    def _cluster_concepts(
        self,
        concepts: List[Dict[str, Any]],
        num_clusters: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group concepts into semantic clusters.

        Args:
            concepts: List of concepts
            num_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster names to concept lists
        """
        # Simple clustering based on categories
        clusters = defaultdict(list)

        # Group by category first
        category_groups = defaultdict(list)
        for concept in concepts:
            category_groups[concept['category']].append(concept)

        # Split categories into clusters
        categories = list(category_groups.keys())
        concepts_per_cluster = max(1, len(concepts) // max(1, num_clusters))

        cluster_id = 0
        for category in categories[:num_clusters]:  # Limit to num_clusters
            category_concepts = category_groups[category]

            # Split large categories across multiple clusters
            for i in range(0, len(category_concepts), concepts_per_cluster):
                if len(clusters) >= num_clusters:  # Don't exceed requested clusters
                    break
                cluster_name = f"cluster_{category}_{cluster_id}"
                cluster_concepts = category_concepts[i:i + concepts_per_cluster]
                if cluster_concepts:  # Only add non-empty clusters
                    clusters[cluster_name] = cluster_concepts
                    cluster_id += 1

            if len(clusters) >= num_clusters:
                break

        # Ensure we have exactly the requested number of clusters
        while len(clusters) < num_clusters and concepts:
            # Create additional clusters from remaining concepts
            used_concepts = set()
            for cluster_list in clusters.values():
                for c in cluster_list:
                    used_concepts.add(id(c))

            remaining = [c for c in concepts if id(c) not in used_concepts]
            if remaining and len(clusters) < num_clusters:
                cluster_name = f"cluster_additional_{len(clusters)}"
                clusters[cluster_name] = remaining[:5]
            else:
                break

        return dict(clusters)

    def _create_clustered_sub_problems(
        self,
        problem: ProblemDefinition,
        semantic_groups: Dict[str, List[Dict[str, Any]]]
    ) -> List[SubProblem]:
        """
        Create sub-problems from semantic clusters.

        Args:
            problem: Original problem
            semantic_groups: Clustered concepts

        Returns:
            List of sub-problems
        """
        sub_problems = []

        for cluster_name, concepts in semantic_groups.items():
            # Extract main theme from cluster
            main_concept = concepts[0]['word'] if concepts else 'general'

            # Create description from concepts
            concept_words = [c['word'] for c in concepts[:5]]
            description = (
                f"Address {main_concept}-related aspects: "
                f"{', '.join(concept_words)}. "
                f"Context: {problem.title}"
            )

            sub_problem = SubProblem(
                sub_problem_id=generate_id(f"semantic_{main_concept}"),
                parent_id=problem.problem_id,
                title=f"{main_concept.title()} Aspects",
                description=description,
                status=ProblemStatus.PENDING,
                confidence=sum(c['relevance'] for c in concepts) / len(concepts),
                assigned_agent=None,
                created_at=datetime.utcnow(),
                completed_at=None
            )
            sub_problems.append(sub_problem)

        return sub_problems

    def _identify_semantic_dependencies(
        self,
        sub_problems: List[SubProblem],
        semantic_groups: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[str]]:
        """
        Identify dependencies based on semantic relationships.

        Args:
            sub_problems: List of sub-problems
            semantic_groups: Original semantic groupings

        Returns:
            Dependency mapping
        """
        dependencies = defaultdict(list)

        # Action-oriented clusters typically depend on object-oriented clusters
        action_problems = [sp for sp in sub_problems
                          if any(c in sp.title.lower()
                                for c in ['create', 'build', 'implement'])]
        object_problems = [sp for sp in sub_problems
                          if any(c in sp.title.lower()
                                for c in ['system', 'module', 'component'])]

        for obj_prob in object_problems:
            for act_prob in action_problems:
                dependencies[obj_prob.sub_problem_id].append(act_prob.sub_problem_id)

        return dict(dependencies)

    def _semantic_execution_order(
        self,
        sub_problems: List[SubProblem],
        dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """
        Generate execution order based on semantic dependencies.

        Args:
            sub_problems: List of sub-problems
            dependencies: Dependency mapping

        Returns:
            Ordered list of sub-problem IDs
        """
        # Use topological sort
        dep_graph = DependencyGraph(
            nodes={sp.sub_problem_id: sp for sp in sub_problems},
            edges=dependencies
        )
        return dep_graph.get_execution_order()


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================

class StrategySelector:
    """
    Intelligent strategy selection based on problem characteristics.

    Analyzes problem features and selects the most appropriate decomposition
    strategy for optimal results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy selector.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.strategies = {
            SovereignDecompositionStrategy.HYBRID: HybridDecompositionStrategy(config),
            SovereignDecompositionStrategy.ROMA: RomadecompositionStrategy(config),
            SovereignDecompositionStrategy.SEMANTIC: SemanticDecompositionStrategy(config)
        }
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def select_strategy(self, problem: ProblemDefinition) -> SovereignDecompositionStrategy:
        """
        Select the best strategy for a given problem.

        Args:
            problem: Problem definition to analyze

        Returns:
            Selected strategy enum
        """
        self.logger.info(f"Selecting strategy for problem: {problem.title}")

        # Score each strategy
        scores = {}

        for strategy in SovereignDecompositionStrategy:
            scores[strategy] = self._score_strategy(problem, strategy)

        # Select highest scoring strategy
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]

        self.logger.info(
            f"Selected {best_strategy.value} strategy (score: {best_score:.2f})"
        )

        return best_strategy

    def _score_strategy(
        self,
        problem: ProblemDefinition,
        strategy: SovereignDecompositionStrategy
    ) -> float:
        """
        Score a strategy for a given problem.

        Args:
            problem: Problem definition
            strategy: Strategy to score

        Returns:
            Score between 0 and 1
        """
        score = 0.0

        if strategy == SovereignDecompositionStrategy.HYBRID:
            # HYBRID is good for complex, multi-faceted problems
            score += self._complexity_score(problem) * 0.4
            score += self._requirement_count_score(problem) * 0.3
            score += self._constraint_count_score(problem) * 0.3

        elif strategy == SovereignDecompositionStrategy.ROMA:
            # ROMA is good for structured, hierarchical problems
            score += self._structure_score(problem) * 0.5
            score += self._depth_score(problem) * 0.3
            score += self._clarity_score(problem) * 0.2

        elif strategy == SovereignDecompositionStrategy.SEMANTIC:
            # SEMANTIC is good for conceptually rich problems
            score += self._concept_density_score(problem) * 0.5
            score += self._domain_specificity_score(problem) * 0.3
            score += self._ambiguity_score(problem) * 0.2

        return min(1.0, max(0.0, score))

    def _complexity_score(self, problem: ProblemDefinition) -> float:
        """Calculate complexity score for strategy selection."""
        desc_length = len(problem.description)
        num_requirements = len(problem.requirements)

        complexity = (desc_length / 1000.0) + (num_requirements / 20.0)
        return min(1.0, complexity)

    def _requirement_count_score(self, problem: ProblemDefinition) -> float:
        """Score based on number of requirements."""
        return min(1.0, len(problem.requirements) / 10.0)

    def _constraint_count_score(self, problem: ProblemDefinition) -> float:
        """Score based on number of constraints."""
        return min(1.0, len(problem.constraints) / 5.0)

    def _structure_score(self, problem: ProblemDefinition) -> float:
        """Score based on problem structure."""
        desc_lower = problem.description.lower()
        structure_words = ['first', 'then', 'next', 'finally', 'step', 'phase']
        structure_count = sum(1 for word in structure_words if word in desc_lower)
        return min(1.0, structure_count / 5.0)

    def _depth_score(self, problem: ProblemDefinition) -> float:
        """Score based on decomposition depth potential."""
        desc_length = len(problem.description)
        # Longer descriptions can support deeper decomposition
        return min(1.0, desc_length / 500.0)

    def _clarity_score(self, problem: ProblemDefinition) -> float:
        """Score based on problem clarity."""
        # More structured text = higher clarity
        sentences = re.split(r'[.!?]+', problem.description)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Ideal sentence length is 15-25 words
        if 15 <= avg_sentence_length <= 25:
            return 1.0
        elif 10 <= avg_sentence_length <= 30:
            return 0.7
        else:
            return 0.4

    def _concept_density_score(self, problem: ProblemDefinition) -> float:
        """Score based on concept density."""
        words = re.findall(r'\b\w+\b', problem.description.lower())
        unique_words = set(words)
        density = len(unique_words) / len(words) if words else 0
        return min(1.0, density * 2)  # Normalize

    def _domain_specificity_score(self, problem: ProblemDefinition) -> float:
        """Score based on domain-specific terminology."""
        desc_lower = problem.description.lower()

        # Domain-specific terms
        domain_terms = {
            'software_engineering': ['api', 'database', 'interface', 'algorithm'],
            'data_science': ['model', 'training', 'dataset', 'feature'],
            'research': ['hypothesis', 'experiment', 'analysis', 'methodology']
        }

        terms = domain_terms.get(problem.domain.lower(), [])
        term_count = sum(1 for term in terms if term in desc_lower)

        return min(1.0, term_count / 5.0)

    def _ambiguity_score(self, problem: ProblemDefinition) -> float:
        """Score based on ambiguity (higher is better for semantic)."""
        # Ambiguity indicators: vague terms, multiple interpretations
        ambiguous_terms = ['approximately', 'roughly', 'around', 'maybe', 'possibly']

        desc_lower = problem.description.lower()
        ambiguity_count = sum(1 for term in ambiguous_terms if term in desc_lower)

        return min(1.0, ambiguity_count / 3.0)


# ============================================================================
# MAIN STRATEGY EXECUTOR
# ============================================================================

class DecompositionStrategyExecutor:
    """
    Main executor for decomposition strategies.

    Provides unified interface for strategy selection and execution
    with comprehensive error handling and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize executor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.selector = StrategySelector(config)
        self.strategies = self.selector.strategies
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def select_strategy(self, problem: ProblemDefinition) -> SovereignDecompositionStrategy:
        """
        Select the best strategy for a problem.

        Args:
            problem: Problem definition

        Returns:
            Selected strategy enum
        """
        return self.selector.select_strategy(problem)

    def execute_strategy(
        self,
        strategy: str,
        problem: ProblemDefinition,
        **kwargs
    ) -> DecompositionPlan:
        """
        Execute a specific decomposition strategy.

        Args:
            strategy: Strategy name (HYBRID, ROMA, SEMANTIC)
            problem: Problem definition to decompose
            **kwargs: Strategy-specific parameters

        Returns:
            DecompositionPlan

        Raises:
            ValueError: If strategy name is invalid
            RuntimeError: If decomposition fails
        """
        try:
            # Convert string to enum
            if isinstance(strategy, str):
                strategy_enum = SovereignDecompositionStrategy[strategy.upper()]
            else:
                strategy_enum = strategy

            # Get strategy instance
            strategy_instance = self.strategies.get(strategy_enum)

            if not strategy_instance:
                raise ValueError(f"Unknown strategy: {strategy}")

            self.logger.info(
                f"Executing {strategy_enum.value} strategy for problem: {problem.title}"
            )

            # Execute decomposition
            plan = strategy_instance.decompose(problem, **kwargs)

            # Validate plan
            if not self._validate_plan(plan):
                raise RuntimeError("Generated decomposition plan failed validation")

            return plan

        except KeyError:
            raise ValueError(
                f"Invalid strategy name: {strategy}. "
                f"Must be one of: {[s.value for s in SovereignDecompositionStrategy]}"
            )
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {e}")
            raise RuntimeError(f"Decomposition failed: {str(e)}")

    def execute_with_auto_selection(
        self,
        problem: ProblemDefinition,
        **kwargs
    ) -> DecompositionPlan:
        """
        Execute decomposition with automatic strategy selection.

        Args:
            problem: Problem definition to decompose
            **kwargs: Strategy-specific parameters

        Returns:
            DecompositionPlan
        """
        strategy = self.select_strategy(problem)
        return self.execute_strategy(strategy, problem, **kwargs)

    def _validate_plan(self, plan: DecompositionPlan) -> bool:
        """
        Validate a decomposition plan.

        Args:
            plan: Plan to validate

        Returns:
            True if valid, False otherwise
        """
        # Check sub-problems
        if not plan.sub_problems:
            self.logger.error("Plan has no sub-problems")
            return False

        # Check execution order
        if not plan.execution_order:
            self.logger.error("Plan has no execution order")
            return False

        # Check all sub-problems in execution order exist
        sub_problem_ids = {sp.sub_problem_id for sp in plan.sub_problems}
        for sp_id in plan.execution_order:
            if sp_id not in sub_problem_ids:
                self.logger.error(f"Execution order references unknown sub-problem: {sp_id}")
                return False

        # Check dependencies reference valid sub-problems
        for from_id, to_ids in plan.dependencies.items():
            if from_id not in sub_problem_ids:
                self.logger.error(f"Dependency references unknown source: {from_id}")
                return False
            for to_id in to_ids:
                if to_id not in sub_problem_ids:
                    self.logger.error(f"Dependency references unknown target: {to_id}")
                    return False

        return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def decompose_hybrid(
    problem: ProblemDefinition,
    depth: int = 3
) -> DecompositionPlan:
    """
    Decompose problem using HYBRID strategy.

    Args:
        problem: Problem to decompose
        depth: Decomposition depth

    Returns:
        DecompositionPlan
    """
    executor = DecompositionStrategyExecutor()
    return executor.execute_strategy(
        SovereignDecompositionStrategy.HYBRID.value,
        problem,
        depth=depth
    )


def decompose_roma(
    problem: ProblemDefinition,
    max_depth: int = 5
) -> DecompositionPlan:
    """
    Decompose problem using ROMA strategy.

    Args:
        problem: Problem to decompose
        max_depth: Maximum recursion depth

    Returns:
        DecompositionPlan
    """
    executor = DecompositionStrategyExecutor()
    return executor.execute_strategy(
        SovereignDecompositionStrategy.ROMA.value,
        problem,
        max_depth=max_depth
    )


def decompose_semantic(
    problem: ProblemDefinition,
    clusters: int = 5
) -> DecompositionPlan:
    """
    Decompose problem using SEMANTIC strategy.

    Args:
        problem: Problem to decompose
        clusters: Number of semantic clusters

    Returns:
        DecompositionPlan
    """
    executor = DecompositionStrategyExecutor()
    return executor.execute_strategy(
        SovereignDecompositionStrategy.SEMANTIC.value,
        problem,
        clusters=clusters
    )


def select_strategy(problem: ProblemDefinition) -> SovereignDecompositionStrategy:
    """
    Select the best decomposition strategy for a problem.

    Args:
        problem: Problem definition

    Returns:
        Selected strategy enum
    """
    executor = DecompositionStrategyExecutor()
    return executor.select_strategy(problem)


def execute_strategy(
    strategy: str,
    problem: ProblemDefinition,
    **kwargs
) -> DecompositionPlan:
    """
    Execute a decomposition strategy.

    Args:
        strategy: Strategy name (HYBRID, ROMA, SEMANTIC)
        problem: Problem to decompose
        **kwargs: Strategy-specific parameters

    Returns:
        DecompositionPlan
    """
    executor = DecompositionStrategyExecutor()
    return executor.execute_strategy(strategy, problem, **kwargs)


# ============================================================================
# EXAMPLE USAGE AND UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("DECOMPOSITION STRATEGY EXAMPLE USAGE")
    print("=" * 80)

    # Create a sample problem
    sample_problem = ProblemDefinition(
        problem_id="test_problem_001",
        title="Build a Microservices E-commerce Platform",
        description=(
            "Design and implement a scalable e-commerce platform using microservices architecture. "
            "The system should include user authentication, product catalog, shopping cart, "
            "order processing, payment integration, and inventory management. "
            "The platform must handle high traffic, ensure data consistency, "
            "and provide excellent user experience."
        ),
        domain="software_engineering",
        complexity="complex",
        priority="high",
        estimated_effort="large",
        requirements=[
            "Support for 10,000 concurrent users",
            "99.9% uptime requirement",
            "Secure payment processing",
            "Real-time inventory updates",
            "Responsive web interface"
        ],
        constraints=[
            "Must use microservices architecture",
            "Budget constraints: cloud costs under $5000/month",
            "Timeline: 6 months",
            "Team size: 8 developers"
        ],
        created_at=datetime.utcnow()
    )

    print(f"\nProblem: {sample_problem.title}")
    print(f"Domain: {sample_problem.domain}")
    print(f"Complexity: {sample_problem.complexity}")
    print(f"Requirements: {len(sample_problem.requirements)}")
    print(f"Constraints: {len(sample_problem.constraints)}")

    # Test 1: Strategy Selection
    print("\n" + "-" * 80)
    print("TEST 1: Strategy Selection")
    print("-" * 80)
    selected_strategy = select_strategy(sample_problem)
    print(f"Selected Strategy: {selected_strategy.value}")

    # Test 2: HYBRID Decomposition
    print("\n" + "-" * 80)
    print("TEST 2: HYBRID Decomposition")
    print("-" * 80)
    try:
        hybrid_plan = decompose_hybrid(sample_problem, depth=2)
        print(f"Plan ID: {hybrid_plan.plan_id}")
        print(f"Sub-problems: {len(hybrid_plan.sub_problems)}")
        print(f"Execution Order: {len(hybrid_plan.execution_order)} steps")
        print("\nSub-problems:")
        for i, sp in enumerate(hybrid_plan.sub_problems, 1):
            print(f"  {i}. {sp.title} (confidence: {sp.confidence:.2f})")
    except Exception as e:
        print(f"HYBRID decomposition failed: {e}")

    # Test 3: ROMA Decomposition
    print("\n" + "-" * 80)
    print("TEST 3: ROMA Decomposition")
    print("-" * 80)
    try:
        roma_plan = decompose_roma(sample_problem, max_depth=3)
        print(f"Plan ID: {roma_plan.plan_id}")
        print(f"Sub-problems: {len(roma_plan.sub_problems)}")
        print(f"Execution Order: {len(roma_plan.execution_order)} steps")
        print("\nSub-problems:")
        for i, sp in enumerate(roma_plan.sub_problems[:5], 1):
            print(f"  {i}. {sp.title} (confidence: {sp.confidence:.2f})")
        if len(roma_plan.sub_problems) > 5:
            print(f"  ... and {len(roma_plan.sub_problems) - 5} more")
    except Exception as e:
        print(f"ROMA decomposition failed: {e}")

    # Test 4: SEMANTIC Decomposition
    print("\n" + "-" * 80)
    print("TEST 4: SEMANTIC Decomposition")
    print("-" * 80)
    try:
        semantic_plan = decompose_semantic(sample_problem, clusters=4)
        print(f"Plan ID: {semantic_plan.plan_id}")
        print(f"Sub-problems: {len(semantic_plan.sub_problems)}")
        print(f"Execution Order: {len(semantic_plan.execution_order)} steps")
        print("\nSub-problems:")
        for i, sp in enumerate(semantic_plan.sub_problems, 1):
            print(f"  {i}. {sp.title} (confidence: {sp.confidence:.2f})")
    except Exception as e:
        print(f"SEMANTIC decomposition failed: {e}")

    # Test 5: Auto-Selection
    print("\n" + "-" * 80)
    print("TEST 5: Auto-Selection Execution")
    print("-" * 80)
    try:
        executor = DecompositionStrategyExecutor()
        auto_plan = executor.execute_with_auto_selection(sample_problem)
        print(f"Auto-selected Plan ID: {auto_plan.plan_id}")
        print(f"Sub-problems: {len(auto_plan.sub_problems)}")
        print(f"Execution Order: {len(auto_plan.execution_order)} steps")
    except Exception as e:
        print(f"Auto-selection execution failed: {e}")

    print("\n" + "=" * 80)
    print("EXAMPLE USAGE COMPLETE")
    print("=" * 80)
