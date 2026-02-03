"""
Universal Recomposition Engine - Industry-Agnostic Solution Assembly System

This module implements comprehensive solution recomposition for any domain.
It takes solved sub-problems and intelligently combines them into a coherent,
integrated solution.

Core Capabilities:
    - Multiple assembly strategies (hierarchical, linear, parallel, adaptive)
    - Conflict detection and resolution
    - Quality assessment and metrics
    - Domain-aware reassembly
    - Solution validation

Usage:
    >>> from universal_recomposition_engine import UniversalRecompositionEngine
    >>> from universal_decomposition_engine import DecompositionPlan
    >>> 
    >>> engine = UniversalRecompositionEngine()
    >>> 
    >>> # After solving sub-problems
    >>> solutions = {
    ...     "sub_1": SubProblemSolution(...),
    ...     "sub_2": SubProblemSolution(...),
    ... }
    >>> 
    >>> # Reassemble into integrated solution
    >>> integrated = engine.assemble(
    ...     plan=decomposition_plan,
    ...     sub_solutions=solutions,
    ...     strategy=AssemblyStrategy.HIERARCHICAL
    ... )
"""

import logging
import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib
from abc import ABC, abstractmethod
from utils.entanglement_utils import normalize_entanglement_matrix
from utils.symbolic_analyzer import SymbolicAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Universal Recomposition Engine
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# ============================================================================
# DATA CLASSES (Mirroring/Syncing with decomposition engine)
# ============================================================================

class AssemblyStrategy(Enum):
    """Strategies for reassembling solutions"""
    HIERARCHICAL = "hierarchical"
    LINEAR = "linear"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    ROMA_DETERMINISTIC = "roma_deterministic"
    ROMA_CREATIVE = "roma_creative"


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity assessment"""
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    overall_complexity: float
    explanation: str = ""


@dataclass
class SuccessCriterion:
    """Success criterion"""
    id: str
    description: str
    metric: str
    threshold: float


@dataclass
class Constraint:
    """Problem constraint"""
    id: str
    description: str
    type: str
    severity: str


@dataclass
class ProblemDefinition:
    """Problem definition"""
    id: str
    title: str
    description: str
    domain: str
    complexity_score: ComplexityScore
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)


@dataclass
class SubProblem:
    """Sub-problem"""
    id: str
    parent_id: str
    title: str
    description: str
    type: str
    complexity_score: ComplexityScore
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)


@dataclass
class DecompositionPlan:
    """Decomposition plan"""
    id: str
    original_problem: ProblemDefinition
    sub_problems: List[SubProblem]
    strategy_used: str
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubProblemSolution:
    """Solution for a sub-problem"""
    sub_problem_id: str
    solution_content: str
    quality_score: float
    verification_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    """Detected conflict between sub-solutions"""
    conflict_id: str
    conflict_type: str  # contradiction, overlap, dependency, inconsistency
    severity: str  # critical, high, medium, low
    involved_solutions: List[str]
    description: str
    suggested_resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for integrated solution"""
    completeness: float  # 0-1
    consistency: float  # 0-1
    coherence: float  # 0-1
    integration_quality: float  # 0-1
    overall_score: float  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedSolution:
    """Final integrated solution"""
    solution_id: str
    problem_id: str
    decomposition_plan_id: str
    assembled_content: str
    assembly_strategy: str
    sub_solutions: Dict[str, SubProblemSolution]
    quality_metrics: QualityMetrics
    conflicts_detected: List[Conflict]
    conflicts_resolved: List[Conflict]
    assembly_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'solution_id': self.solution_id,
            'problem_id': self.problem_id,
            'decomposition_plan_id': self.decomposition_plan_id,
            'assembled_content': self.assembled_content,
            'assembly_strategy': self.assembly_strategy,
            'quality_metrics': {
                'completeness': self.quality_metrics.completeness,
                'consistency': self.quality_metrics.consistency,
                'coherence': self.quality_metrics.coherence,
                'integration_quality': self.quality_metrics.integration_quality,
                'overall_score': self.quality_metrics.overall_score
            },
            'conflicts_detected': len(self.conflicts_detected),
            'conflicts_resolved': len(self.conflicts_resolved),
            'created_at': self.created_at.isoformat()
        }


# ============================================================================
# CONFLICT DETECTION
# ============================================================================

class ConflictDetector:
    """
    Advanced conflict detection between sub-solutions.
    
    Detects:
        - Contradictions: Solutions that contradict each other
        - Overlaps: Unnecessary duplication
        - Dependency violations: Missing prerequisites
        - Inconsistencies: Style/format mismatches
    """
    
    def __init__(self, semantic_threshold: float = 0.75, overlap_threshold: float = 0.7):
        self.semantic_threshold = semantic_threshold
        self.overlap_threshold = overlap_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_conflicts(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependency_graph: Dict[str, List[str]],
        entanglement_matrix: Optional[Dict[str, Set[str]]] = None
    ) -> List[Conflict]:
        """Detect all conflicts between sub-solutions"""
        conflicts = []
        
        # Detect contradictions
        conflicts.extend(self._detect_contradictions(sub_solutions, entanglement_matrix))
        
        # Detect overlaps
        conflicts.extend(self._detect_overlaps(sub_solutions, entanglement_matrix))
        
        # Detect dependency violations
        conflicts.extend(self._detect_dependency_violations(sub_solutions, dependency_graph))
        
        # Detect inconsistencies
        conflicts.extend(self._detect_inconsistencies(sub_solutions, entanglement_matrix))

        # Detect entanglement alignment issues
        conflicts.extend(self._detect_entanglement_alignment(sub_solutions, entanglement_matrix))
        
        return conflicts
    
    def _detect_contradictions(
        self, 
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, Set[str]]] = None
    ) -> List[Conflict]:
        """Detect contradictory solutions"""
        conflicts = []
        solution_ids = list(sub_solutions.keys())
        entangled_pairs = self._build_entangled_pairs(entanglement_matrix)
        
        # Look for explicit contradiction markers
        contradiction_markers = [
            (r'\b(must|should|will)\s+\w+', r'\b(must not|should not|will not)\s+\w+'),
            (r'\benable\b', r'\bdisable\b'),
            (r'\bincrease\b', r'\bdecrease\b'),
            (r'\badd\b', r'\bremove\b'),
        ]
        
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i+1:]:
                content1 = sub_solutions[id1].solution_content.lower()
                content2 = sub_solutions[id2].solution_content.lower()
                
                for pattern1, pattern2 in contradiction_markers:
                    matches1 = set(re.findall(pattern1, content1))
                    matches2 = set(re.findall(pattern2, content2))
                    
                    if matches1 and matches2:
                        conflict = Conflict(
                            conflict_id=self._generate_id("conf"),
                            conflict_type="contradiction",
                            severity="critical",
                            involved_solutions=[id1, id2],
                            description=f"Potential contradiction: {matches1} vs {matches2}",
                            suggested_resolution="Review and reconcile conflicting requirements",
                            metadata={"entangled_pair": frozenset([id1, id2]) in entangled_pairs}
                        )
                        conflicts.append(conflict)
                        break
        
        return conflicts
    
    def _detect_overlaps(
        self, 
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, Set[str]]] = None
    ) -> List[Conflict]:
        """Detect overlapping/duplicate content"""
        conflicts = []
        solution_ids = list(sub_solutions.keys())
        entangled_pairs = self._build_entangled_pairs(entanglement_matrix)
        
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i+1:]:
                content1 = sub_solutions[id1].solution_content
                content2 = sub_solutions[id2].solution_content
                
                # Calculate Jaccard similarity
                similarity = self._jaccard_similarity(content1, content2)
                
                entangled = frozenset([id1, id2]) in entangled_pairs
                threshold = 0.55 if entangled else self.overlap_threshold
                if similarity > threshold:
                    conflict = Conflict(
                        conflict_id=self._generate_id("ovlp"),
                        conflict_type="overlap",
                        severity="medium",
                        involved_solutions=[id1, id2],
                        description=f"Significant overlap detected (similarity: {similarity:.2f})",
                        suggested_resolution="Consolidate overlapping sections or refine sub-problem boundaries",
                        metadata={"entangled_pair": entangled}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_dependency_violations(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependency_graph: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect missing dependencies or ordering issues"""
        conflicts = []
        
        for solution_id, solution in sub_solutions.items():
            # Check if solution references undefined components
            content = solution.solution_content
            
            # Look for references to other sub-solutions
            for other_id in sub_solutions:
                if other_id == solution_id:
                    continue
                
                # Check if references other solution without declaring dependency
                other_title = sub_solutions[other_id].metadata.get('title', other_id)
                if other_title.lower() in content.lower():
                    if other_id not in dependency_graph.get(solution_id, []):
                        conflict = Conflict(
                            conflict_id=self._generate_id("dep"),
                            conflict_type="dependency",
                            severity="high",
                            involved_solutions=[solution_id, other_id],
                            description=f"Solution references '{other_title}' without declared dependency",
                            suggested_resolution=f"Add {other_id} as a dependency"
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_inconsistencies(
        self, 
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, Set[str]]] = None
    ) -> List[Conflict]:
        """Detect style/format inconsistencies"""
        conflicts = []
        
        # Check for inconsistent formatting
        formats_detected = defaultdict(list)
        
        for solution_id, solution in sub_solutions.items():
            content = solution.solution_content
            
            # Detect format type
            if content.startswith('#') or content.startswith('##'):
                formats_detected['markdown'].append(solution_id)
            elif '<' in content and '>' in content:
                formats_detected['html'].append(solution_id)
            elif 'def ' in content or 'class ' in content:
                formats_detected['code'].append(solution_id)
            else:
                formats_detected['plain'].append(solution_id)
        
        # Flag if multiple formats detected
        if len(formats_detected) > 1:
            conflict = Conflict(
                conflict_id=self._generate_id("fmt"),
                conflict_type="inconsistency",
                severity="low",
                involved_solutions=list(sub_solutions.keys()),
                description=f"Inconsistent formats detected: {list(formats_detected.keys())}",
                suggested_resolution="Standardize on a single format for all sub-solutions",
                metadata={"entanglement_context": bool(entanglement_matrix)}
            )
            conflicts.append(conflict)
        
        return conflicts

    def _detect_entanglement_alignment(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, Set[str]]] = None,
    ) -> List[Conflict]:
        """Detect entanglement alignment issues across coupled sub-solutions."""
        if not entanglement_matrix:
            return []

        conflicts = []
        analyzer = SymbolicAnalyzer()
        token_cache: Dict[str, Set[str]] = {}

        for sp_id, solution in sub_solutions.items():
            token_cache[sp_id] = analyzer.analyze(solution.solution_content or "").symbols

        seen_pairs = set()
        for source, targets in entanglement_matrix.items():
            for target in targets:
                pair = frozenset([source, target])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if source not in token_cache or target not in token_cache:
                    continue
                tokens_a = token_cache[source]
                tokens_b = token_cache[target]
                if not tokens_a or not tokens_b:
                    continue
                overlap = tokens_a & tokens_b
                union = tokens_a | tokens_b
                similarity = len(overlap) / max(1, len(union))
                if similarity < 0.05:
                    conflict = Conflict(
                        conflict_id=self._generate_id("ent"),
                        conflict_type="entanglement_alignment",
                        severity="low",
                        involved_solutions=[source, target],
                        description=(
                            f"Entangled components show low interface overlap "
                            f"(similarity: {similarity:.2f})"
                        ),
                        suggested_resolution=(
                            "Review shared interfaces and synchronize entangled outputs"
                        ),
                        metadata={"entangled_pair": True},
                    )
                    conflicts.append(conflict)

        return conflicts

    @staticmethod
    def _build_entangled_pairs(
        entanglement_matrix: Optional[Dict[str, Set[str]]]
    ) -> Set[frozenset]:
        pairs: Set[frozenset] = set()
        if not entanglement_matrix:
            return pairs
        for source, targets in entanglement_matrix.items():
            for target in targets:
                pairs.add(frozenset([source, target]))
        return pairs
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        # Normalize and tokenize
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# CONFLICT RESOLUTION
# ============================================================================

class ConflictResolver:
    """
    Conflict resolution strategies.
    
    Strategies:
        - Priority-based: Select based on quality/confidence
        - Merge-based: Combine overlapping content
        - Manual: Flag for human review
        - LLM-mediated: Use AI to resolve
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def resolve_conflicts(
        self,
        conflicts: List[Conflict],
        sub_solutions: Dict[str, SubProblemSolution],
        strategy: str = "priority"
    ) -> Tuple[Dict[str, SubProblemSolution], List[Conflict]]:
        """
        Resolve conflicts using specified strategy.
        
        Returns:
            Tuple of (updated_solutions, unresolved_conflicts)
        """
        resolved = []
        unresolved = []
        updated_solutions = dict(sub_solutions)
        
        for conflict in conflicts:
            if strategy == "priority":
                result = self._resolve_by_priority(conflict, updated_solutions)
            elif strategy == "merge":
                result = self._resolve_by_merge(conflict, updated_solutions)
            elif strategy == "llm" and self.llm_client:
                result = self._resolve_by_llm(conflict, updated_solutions)
            else:
                result = None
            
            if result:
                updated_solutions.update(result)
                resolved.append(conflict)
            else:
                unresolved.append(conflict)
        
        return updated_solutions, unresolved
    
    def _resolve_by_priority(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Optional[Dict[str, SubProblemSolution]]:
        """Resolve by selecting higher quality solution"""
        if conflict.conflict_type != "contradiction":
            return None
        
        # Get quality scores
        scores = {
            sid: sub_solutions[sid].quality_score 
            for sid in conflict.involved_solutions
            if sid in sub_solutions
        }
        
        if not scores:
            return None
        
        # Select highest quality
        best_id = max(scores, key=scores.get)
        
        # Mark others as superseded
        for sid in conflict.involved_solutions:
            if sid != best_id and sid in sub_solutions:
                sub_solutions[sid].metadata['superseded_by'] = best_id
        
        return sub_solutions
    
    def _resolve_by_merge(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Optional[Dict[str, SubProblemSolution]]:
        """Resolve by merging overlapping content"""
        if conflict.conflict_type != "overlap":
            return None
        
        # Get solutions to merge
        solutions_to_merge = [
            sub_solutions[sid] 
            for sid in conflict.involved_solutions 
            if sid in sub_solutions
        ]
        
        if len(solutions_to_merge) < 2:
            return None
        
        # Simple merge: concatenate with separator
        merged_content = "\n\n".join(
            f"## Section from {s.sub_problem_id}\n{s.solution_content}"
            for s in solutions_to_merge
        )
        
        # Create merged solution (using first ID)
        merged_id = conflict.involved_solutions[0]
        sub_solutions[merged_id].solution_content = merged_content
        sub_solutions[merged_id].metadata['merged_from'] = conflict.involved_solutions
        
        # Mark others as merged
        for sid in conflict.involved_solutions[1:]:
            if sid in sub_solutions:
                sub_solutions[sid].metadata['merged_into'] = merged_id
        
        return sub_solutions
    
    def _resolve_by_llm(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Optional[Dict[str, SubProblemSolution]]:
        """Resolve using LLM mediation"""
        if not self.llm_client:
            return None
        
        # This is a placeholder for actual LLM-based resolution
        # In practice, would construct a prompt and call the LLM
        self.logger.info(f"LLM resolution requested for conflict {conflict.conflict_id}")
        return None


# ============================================================================
# ASSEMBLY STRATEGIES
# ============================================================================

class AssemblyStrategyBase(ABC):
    """Abstract base for assembly strategies"""
    
    @abstractmethod
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Assemble sub-solutions into integrated solution"""
        pass


class HierarchicalAssembly(AssemblyStrategyBase):
    """Bottom-up hierarchical assembly"""
    
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Assemble following dependency tree bottom-up"""
        # Build reverse dependency graph (who depends on whom)
        reverse_deps = defaultdict(list)
        for sp_id, deps in plan.dependency_graph.items():
            for dep in deps:
                reverse_deps[dep].append(sp_id)
        
        # Start with leaf nodes (no dependents)
        all_ids = set(plan.dependency_graph.keys())
        dependent_ids = set(reverse_deps.keys())
        leaf_ids = all_ids - dependent_ids
        
        assembled = []
        assembled.append(f"# {plan.original_problem.title}\n")
        assembled.append(f"\n## Overview\n{plan.original_problem.description}\n")
        
        # Add leaf solutions first
        for leaf_id in sorted(leaf_ids):
            if leaf_id in sub_solutions:
                sol = sub_solutions[leaf_id]
                assembled.append(f"\n## Component: {leaf_id}\n")
                assembled.append(sol.solution_content)
        
        # Add dependent solutions
        for sp_id in plan.execution_order:
            if sp_id in leaf_ids:
                continue
            if sp_id in sub_solutions:
                sol = sub_solutions[sp_id]
                deps = plan.dependency_graph.get(sp_id, [])
                dep_text = f" (builds on: {', '.join(deps)})" if deps else ""
                assembled.append(f"\n## Component: {sp_id}{dep_text}\n")
                assembled.append(sol.solution_content)
        
        return "\n".join(assembled)


class LinearAssembly(AssemblyStrategyBase):
    """Sequential linear assembly"""
    
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Assemble in linear execution order"""
        assembled = []
        assembled.append(f"# {plan.original_problem.title}\n")
        assembled.append(f"\n## Problem Statement\n{plan.original_problem.description}\n")
        
        for i, sp_id in enumerate(plan.execution_order, 1):
            if sp_id in sub_solutions:
                sol = sub_solutions[sp_id]
                assembled.append(f"\n## Step {i}: {sp_id}\n")
                assembled.append(sol.solution_content)
        
        return "\n".join(assembled)


class ParallelAssembly(AssemblyStrategyBase):
    """Parallel group assembly"""
    
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Assemble grouping parallelizable components"""
        assembled = []
        assembled.append(f"# {plan.original_problem.title}\n")
        assembled.append(f"\n## Problem Statement\n{plan.original_problem.description}\n")
        
        # Group by parallel groups
        grouped = defaultdict(list)
        for sp in plan.sub_problems:
            # Find which parallel group this belongs to
            group_idx = self._find_parallel_group(sp.id, plan)
            grouped[group_idx].append(sp.id)
        
        # Assemble by groups
        for group_idx, sp_ids in sorted(grouped.items()):
            assembled.append(f"\n## Phase {group_idx + 1}\n")
            for sp_id in sp_ids:
                if sp_id in sub_solutions:
                    sol = sub_solutions[sp_id]
                    assembled.append(f"\n### {sp_id}\n")
                    assembled.append(sol.solution_content)
        
        return "\n".join(assembled)
    
    def _find_parallel_group(self, sp_id: str, plan: DecompositionPlan) -> int:
        """Find which parallel group a sub-problem belongs to"""
        for i, group in enumerate(plan.parallel_groups if hasattr(plan, 'parallel_groups') else []):
            if sp_id in group:
                return i
        return 0


class AdaptiveAssembly(AssemblyStrategyBase):
    """Adaptive assembly based on problem characteristics"""
    
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Select best assembly strategy based on problem"""
        complexity = plan.original_problem.complexity_score.overall_complexity
        num_subproblems = len(plan.sub_problems)
        
        # Select strategy based on characteristics
        if complexity > 7.5 or num_subproblems > 10:
            # Complex problems: use hierarchical
            strategy = HierarchicalAssembly()
        elif num_subproblems <= 3:
            # Simple problems: linear
            strategy = LinearAssembly()
        else:
            # Medium complexity: parallel
            strategy = ParallelAssembly()
        
        return strategy.assemble(plan, sub_solutions)


class ROMADeterministicAssembly(AssemblyStrategyBase):
    """
    ROMA deterministic assembly - verbatim sub-solution insertion.
    Preserves exact content of sub-solutions without modification.
    """
    
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Assemble by inserting sub-solutions verbatim"""
        sections = []
        
        # Header
        sections.append(f"# {plan.original_problem.title}\n")
        sections.append(f"\n## Problem Statement\n{plan.original_problem.description}\n")
        sections.append("\n## Solution\n")
        
        # Insert each sub-solution verbatim
        for sp_id in plan.execution_order:
            if sp_id in sub_solutions:
                sol = sub_solutions[sp_id]
                # Insert without modification
                sections.append(f"\n<!-- BEGIN {sp_id} -->\n")
                sections.append(sol.solution_content)
                sections.append(f"\n<!-- END {sp_id} -->\n")
        
        return "\n".join(sections)


# ============================================================================
# MAIN RECOMPOSITION ENGINE
# ============================================================================

class UniversalRecompositionEngine:
    """
    Universal recomposition engine for assembling sub-solutions.
    
    Provides:
        - Multiple assembly strategies
        - Conflict detection and resolution
        - Quality assessment
        - Domain-aware reassembly
    
    Example:
        >>> engine = UniversalRecompositionEngine()
        >>> 
        >>> # After solving sub-problems
        >>> integrated = engine.assemble(
        ...     plan=decomposition_plan,
        ...     sub_solutions=solutions,
        ...     strategy=AssemblyStrategy.HIERARCHICAL
        ... )
        >>> 
        >>> print(f"Quality: {integrated.quality_metrics.overall_score}")
        >>> print(f"Conflicts: {len(integrated.conflicts_detected)}")
    """
    
    # Strategy registry
    STRATEGIES = {
        AssemblyStrategy.HIERARCHICAL: HierarchicalAssembly,
        AssemblyStrategy.LINEAR: LinearAssembly,
        AssemblyStrategy.PARALLEL: ParallelAssembly,
        AssemblyStrategy.ADAPTIVE: AdaptiveAssembly,
        AssemblyStrategy.ROMA_DETERMINISTIC: ROMADeterministicAssembly,
    }
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver(llm_client)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.assembly_history: List[IntegratedSolution] = []

    def _extract_entanglement_matrix(self, plan: DecompositionPlan) -> Dict[str, Set[str]]:
        """Extract and normalize entanglement matrix from the plan."""
        raw_matrix = {}
        if hasattr(plan, "entanglement_matrix"):
            raw_matrix = getattr(plan, "entanglement_matrix") or {}
        if not raw_matrix and hasattr(plan, "metadata") and isinstance(plan.metadata, dict):
            raw_matrix = plan.metadata.get("entanglement_matrix", {}) or {}
        if not raw_matrix and hasattr(plan, "analyzed_context") and isinstance(plan.analyzed_context, dict):
            raw_matrix = plan.analyzed_context.get("entanglement_matrix", {}) or {}
        try:
            return normalize_entanglement_matrix(
                raw_matrix,
                allowed_ids=[sp.id for sp in plan.sub_problems],
                enforce_symmetry=True,
                strict=False,
            )
        except Exception as exc:
            self.logger.warning(f"Failed to normalize entanglement matrix: {exc}")
            return {}

    @staticmethod
    def _apply_entanglement_invalidation(
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Dict[str, Set[str]],
        conflicts: List[Conflict],
    ) -> None:
        """Propagate entanglement invalidation across coupled sub-solutions."""
        if not conflicts:
            return
        conflict_pairs = set()
        for conflict in conflicts:
            involved = conflict.involved_solutions or []
            for i in range(len(involved)):
                for j in range(i + 1, len(involved)):
                    conflict_pairs.add(frozenset([involved[i], involved[j]]))

        for pair in conflict_pairs:
            if len(pair) != 2:
                continue
            a, b = tuple(pair)
            if b not in entanglement_matrix.get(a, set()) and a not in entanglement_matrix.get(b, set()):
                continue
            for source, target in [(a, b), (b, a)]:
                if target not in sub_solutions:
                    continue
                meta = sub_solutions[target].metadata
                meta.setdefault("entanglement_invalidation", []).append(source)
                meta["needs_consistency_refinement"] = True
    
    def assemble(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution],
        strategy: AssemblyStrategy = AssemblyStrategy.ADAPTIVE,
        detect_conflicts: bool = True,
        resolve_conflicts: bool = True,
        min_quality_threshold: float = 0.5
    ) -> IntegratedSolution:
        """
        Assemble sub-solutions into integrated solution.

        Args:
            plan: Decomposition plan
            sub_solutions: Dictionary mapping sub-problem IDs to solutions
            strategy: Assembly strategy to use
            detect_conflicts: Whether to detect conflicts
            resolve_conflicts: Whether to attempt conflict resolution
            min_quality_threshold: Minimum quality score for acceptance

        Returns:
            IntegratedSolution with assembled content and metadata
        """
        import time
        start_time_total = time.time()
        success = False
        plan_id = plan.id

        try:
            start_time = datetime.now()
            self.logger.info(f"Starting assembly of {len(sub_solutions)} sub-solutions")

            # Filter to only include solutions for sub-problems in plan
            valid_ids = {sp.id for sp in plan.sub_problems}
            filtered_solutions = {
                k: v for k, v in sub_solutions.items()
                if k in valid_ids
            }

            entanglement_matrix = self._extract_entanglement_matrix(plan)

            # Detect conflicts
            conflicts_detected = []
            if detect_conflicts:
                conflicts_detected = self.conflict_detector.detect_conflicts(
                    filtered_solutions,
                    plan.dependency_graph,
                    entanglement_matrix=entanglement_matrix
                )
                self.logger.info(f"Detected {len(conflicts_detected)} conflicts")

            # Resolve conflicts
            conflicts_resolved = []
            if resolve_conflicts and conflicts_detected:
                filtered_solutions, unresolved = self.conflict_resolver.resolve_conflicts(
                    conflicts_detected,
                    filtered_solutions,
                    strategy="priority"
                )
                conflicts_resolved = [c for c in conflicts_detected if c not in unresolved]
                conflicts_detected = unresolved
                self.logger.info(f"Resolved {len(conflicts_resolved)} conflicts, {len(conflicts_detected)} remain")

            if entanglement_matrix and conflicts_detected:
                self._apply_entanglement_invalidation(filtered_solutions, entanglement_matrix, conflicts_detected)

            # Select and execute assembly strategy
            strategy_class = self.STRATEGIES.get(strategy, AdaptiveAssembly)
            strategy_instance = strategy_class()

            assembled_content = strategy_instance.assemble(plan, filtered_solutions)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                plan,
                filtered_solutions,
                assembled_content,
                conflicts_detected
            )

            # Create integrated solution
            solution = IntegratedSolution(
                solution_id=self._generate_id("sol"),
                problem_id=plan.original_problem.id,
                decomposition_plan_id=plan.id,
                assembled_content=assembled_content,
                assembly_strategy=strategy.value,
                sub_solutions=filtered_solutions,
                quality_metrics=quality_metrics,
                conflicts_detected=conflicts_detected,
                conflicts_resolved=conflicts_resolved,
                assembly_log=[
                    f"Strategy: {strategy.value}",
                    f"Sub-solutions: {len(filtered_solutions)}",
                    f"Conflicts detected: {len(conflicts_detected) + len(conflicts_resolved)}",
                    f"Conflicts resolved: {len(conflicts_resolved)}",
                    f"Quality score: {quality_metrics.overall_score:.2f}"
                ],
                metadata={
                    "entanglement_matrix": {
                        key: sorted(list(value)) for key, value in entanglement_matrix.items()
                    },
                    "entanglement_conflicts": [
                        c.conflict_id for c in conflicts_detected if c.conflict_type.startswith("entanglement")
                    ],
                }
            )

            self.assembly_history.append(solution)

            self.logger.info(f"Assembly complete: quality={quality_metrics.overall_score:.2f}")

            success = True
            duration = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful assembly
            self._extract_universal_recomp_knowledge("assemble", plan_id, strategy, solution)
            self._track_universal_recomp_performance("assemble", True, duration, len(filtered_solutions))

            return solution

        except Exception as e:
            duration = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_universal_recomp_alerts("assemble", False, plan_id, str(e))
            self._track_universal_recomp_performance("assemble", False, duration, 0)

            self.logger.error(f"Assembly failed: {e}")
            raise
    
    def _calculate_quality_metrics(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution],
        assembled_content: str,
        conflicts: List[Conflict]
    ) -> QualityMetrics:
        """Calculate quality metrics for assembled solution"""
        
        # Completeness: did we include all sub-solutions?
        expected_count = len(plan.sub_problems)
        actual_count = len(sub_solutions)
        completeness = actual_count / expected_count if expected_count > 0 else 1.0
        
        # Consistency: based on conflicts
        critical_conflicts = sum(1 for c in conflicts if c.severity == 'critical')
        high_conflicts = sum(1 for c in conflicts if c.severity == 'high')
        consistency = max(0.0, 1.0 - (critical_conflicts * 0.3 + high_conflicts * 0.1))
        
        # Coherence: based on content flow (simplified)
        # Check if content has proper structure
        has_structure = bool(re.search(r'^#{1,3}\s+', assembled_content, re.MULTILINE))
        coherence = 0.8 if has_structure else 0.5
        
        # Integration quality: based on average solution quality
        avg_quality = sum(s.quality_score for s in sub_solutions.values()) / len(sub_solutions) if sub_solutions else 0.5
        
        # Overall score
        overall = (completeness * 0.3 + consistency * 0.3 + coherence * 0.2 + avg_quality * 0.2)
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            coherence=coherence,
            integration_quality=avg_quality,
            overall_score=overall,
            details={
                'sub_solutions_included': actual_count,
                'sub_solutions_expected': expected_count,
                'conflict_count': len(conflicts)
            }
        )
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
    
    def get_assembly_history(self) -> List[IntegratedSolution]:
        """Get history of all assemblies"""
        return self.assembly_history.copy()

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Universal Recomposition
    # =========================================================================

    def _trigger_universal_recomp_alerts(
        self,
        operation: str,
        success: bool,
        plan_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for universal recomposition failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                alert_manager.create_alert(
                    title=f"Universal Recomposition Alert: {operation}",
                    description=f"Universal Recomposition operation '{operation}' failed" +
                                 (f" for plan '{plan_id}'" if plan_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="universal_recomposition_engine",
                    component="universal_recomposition",
                    metadata=metadata or {}
                )

        except Exception as e:
            self.logger.error(f"Failed to trigger Universal Recomposition alert: {e}")

    def _extract_universal_recomp_knowledge(
        self,
        operation: str,
        plan_id: str,
        strategy: 'AssemblyStrategy',
        solution: 'IntegratedSolution'
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract universal recomposition knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"univ_recomp_{operation}_{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="universal_recomposition_execution",
                source_component="universal_recomposition_engine",
                title=f"Universal Recomposition: {operation} - {plan_id}",
                content={
                    "operation": operation,
                    "plan_id": plan_id,
                    "assembly_strategy": strategy.value if strategy else "unknown",
                    "num_sub_solutions": len(solution.sub_solutions),
                    "quality_score": solution.quality_metrics.overall_score,
                    "conflicts_detected": len(solution.conflicts_detected),
                    "conflicts_resolved": len(solution.conflicts_resolved),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "solution_id": solution.solution_id,
                    "completeness": solution.quality_metrics.completeness,
                    "consistency": solution.quality_metrics.consistency,
                    "coherence": solution.quality_metrics.coherence
                },
                tags=["universal_recomposition", operation, strategy.value if strategy else "unknown"]
            )

            knowledge_engine.store_artifact(artifact)
            self.logger.debug(f"Extracted Universal Recomposition knowledge for {plan_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to extract Universal Recomposition knowledge: {e}")
            return False

    def _track_universal_recomp_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        num_sub_solutions: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track universal recomposition performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"univ_recomp_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "num_sub_solutions": num_sub_solutions
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                self.logger.debug(f"Tracked Universal Recomposition performance for {operation}")

        except Exception as e:
            self.logger.error(f"Failed to track Universal Recomposition performance: {e}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'AssemblyStrategy',
    
    # Data classes
    'SubProblemSolution',
    'IntegratedSolution',
    'Conflict',
    'QualityMetrics',
    'DecompositionPlan',
    'ProblemDefinition',
    'SubProblem',
    
    # Components
    'ConflictDetector',
    'ConflictResolver',
    
    # Assembly strategies
    'AssemblyStrategyBase',
    'HierarchicalAssembly',
    'LinearAssembly',
    'ParallelAssembly',
    'AdaptiveAssembly',
    'ROMADeterministicAssembly',
    
    # Main engine
    'UniversalRecompositionEngine',
]


# ============================================================================
# MAIN EXECUTION (EXAMPLES)
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 70)
    print("Universal Recomposition Engine - Examples")
    print("=" * 70)
    
    # Create mock decomposition plan
    plan = DecompositionPlan(
        id="plan_test",
        original_problem=ProblemDefinition(
            id="prob_1",
            title="Build Authentication System",
            description="Create a secure authentication microservice with JWT and OAuth2",
            domain="software",
            complexity_score=ComplexityScore(
                cognitive_complexity=7.0,
                computational_complexity=6.0,
                domain_complexity=6.0,
                integration_complexity=7.0,
                overall_complexity=6.5
            )
        ),
        sub_problems=[
            SubProblem(
                id="sub_1",
                parent_id="prob_1",
                title="Database Schema",
                description="Design user table and authentication tables",
                type="implementation",
                complexity_score=ComplexityScore(5, 5, 5, 5, 5),
                dependencies=[]
            ),
            SubProblem(
                id="sub_2",
                parent_id="prob_1",
                title="JWT Implementation",
                description="Implement JWT token generation and validation",
                type="implementation",
                complexity_score=ComplexityScore(7, 7, 7, 6, 7),
                dependencies=["sub_1"]
            ),
            SubProblem(
                id="sub_3",
                parent_id="prob_1",
                title="OAuth2 Integration",
                description="Integrate OAuth2 providers",
                type="integration",
                complexity_score=ComplexityScore(6, 6, 6, 8, 6),
                dependencies=["sub_2"]
            )
        ],
        strategy_used="hybrid",
        dependency_graph={
            "sub_1": [],
            "sub_2": ["sub_1"],
            "sub_3": ["sub_2"]
        },
        execution_order=["sub_1", "sub_2", "sub_3"]
    )
    
    # Create mock solutions
    solutions = {
        "sub_1": SubProblemSolution(
            sub_problem_id="sub_1",
            solution_content="""
## Database Schema

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```
""",
            quality_score=0.9,
            metadata={'title': 'Database Schema'}
        ),
        "sub_2": SubProblemSolution(
            sub_problem_id="sub_2",
            solution_content="""
## JWT Implementation

```python
import jwt

def generate_token(user_id):
    return jwt.encode({'user_id': str(user_id)}, SECRET_KEY, algorithm='HS256')
```
""",
            quality_score=0.85,
            metadata={'title': 'JWT Implementation'}
        ),
        "sub_3": SubProblemSolution(
            sub_problem_id="sub_3",
            solution_content="""
## OAuth2 Integration

```python
from authlib.integrations.flask_client import OAuth

oauth = OAuth()
google = oauth.register('google', ...)
```
""",
            quality_score=0.8,
            metadata={'title': 'OAuth2 Integration'}
        )
    }
    
    # Initialize engine
    engine = UniversalRecompositionEngine()
    
    # Example 1: Hierarchical Assembly
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Hierarchical Assembly")
    print("=" * 70)
    
    result = engine.assemble(
        plan=plan,
        sub_solutions=solutions,
        strategy=AssemblyStrategy.HIERARCHICAL
    )
    
    print(f"\nQuality Score: {result.quality_metrics.overall_score:.2f}")
    print(f"Conflicts Detected: {len(result.conflicts_detected)}")
    print(f"Content Length: {len(result.assembled_content)} chars")
    print("\nAssembly Log:")
    for log in result.assembly_log:
        print(f"  - {log}")
    
    # Example 2: Linear Assembly
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Linear Assembly")
    print("=" * 70)
    
    result = engine.assemble(
        plan=plan,
        sub_solutions=solutions,
        strategy=AssemblyStrategy.LINEAR
    )
    
    print(f"\nQuality Score: {result.quality_metrics.overall_score:.2f}")
    print(f"Strategy Used: {result.assembly_strategy}")
    
    # Example 3: ROMA Deterministic Assembly
    print("\n" + "=" * 70)
    print("EXAMPLE 3: ROMA Deterministic Assembly")
    print("=" * 70)
    
    result = engine.assemble(
        plan=plan,
        sub_solutions=solutions,
        strategy=AssemblyStrategy.ROMA_DETERMINISTIC
    )
    
    print(f"\nQuality Score: {result.quality_metrics.overall_score:.2f}")
    print(f"Strategy Used: {result.assembly_strategy}")
    
    # Show sample of assembled content
    print("\nSample Assembled Content:")
    print("-" * 50)
    print(result.assembled_content[:500])
    print("...")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
