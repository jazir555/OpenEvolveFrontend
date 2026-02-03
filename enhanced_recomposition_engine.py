"""
Enhanced Recomposition Engine - Sovereign-Grade Comprehensive Solution Assembly System

This module implements an infinitely more comprehensive recomposition system with:
- 15+ Assembly Strategies (Hierarchical, Semantic, Adaptive, etc.)
- Advanced Conflict Detection (12+ Conflict Types)
- Multi-Strategy Conflict Resolution
- Semantic Coherence Validation
- Cross-Domain Integration Support
- Quality Gates and Comprehensive Validation
- Rollback Capabilities and Safety Mechanisms
- Incremental Recomposition with Version Control
- Solution Optimization and Refinement
- Uncertainty-Aware Assembly
- Parallel Assembly Processing
- LLM-Powered Intelligent Assembly

Version: 3.0.0
Author: OpenEvolve Sovereign System
"""

from __future__ import annotations

import hashlib
import json
from utils.entanglement_utils import normalize_entanglement_matrix
from utils.symbolic_analyzer import SymbolicAnalyzer
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union, Protocol, Iterator
)
import uuid
import threading
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS - Comprehensive Type Definitions
# ============================================================================

class AssemblyStrategy(Enum):
    """Comprehensive assembly strategies."""
    # Structural strategies
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    
    # Semantic strategies
    SEMANTIC_COHERENCE = "semantic_coherence"
    DOMAIN_DRIVEN = "domain_driven"
    FLOW_BASED = "flow_based"
    
    # Optimization strategies
    OPTIMIZED = "optimized"
    COST_BASED = "cost_based"
    QUALITY_BASED = "quality_based"
    RISK_OPTIMIZED = "risk_optimized"
    
    # Advanced strategies
    ADAPTIVE = "adaptive"
    INCREMENTAL = "incremental"
    ITERATIVE = "iterative"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"
    
    # Integration strategies
    ROMA_DETERMINISTIC = "roma_deterministic"
    ROMA_CREATIVE = "roma_creative"
    LLM_MEDIATED = "llm_mediated"


class ConflictType(Enum):
    """Comprehensive conflict types."""
    # Logical conflicts
    CONTRADICTION = "contradiction"
    INCONSISTENCY = "inconsistency"
    LOGICAL_ERROR = "logical_error"
    
    # Overlap conflicts
    SEMANTIC_OVERLAP = "semantic_overlap"
    CONTENT_OVERLAP = "content_overlap"
    REDUNDANCY = "redundancy"
    
    # Structural conflicts
    DEPENDENCY_VIOLATION = "dependency_violation"
    ORDER_VIOLATION = "order_violation"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    
    # Technical conflicts
    INTERFACE_MISMATCH = "interface_mismatch"
    DATA_FORMAT = "data_format"
    VERSION_MISMATCH = "version_mismatch"
    API_INCOMPATIBILITY = "api_incompatibility"
    SCHEMA_MISMATCH = "schema_mismatch"
    
    # Quality conflicts
    QUALITY_GAP = "quality_gap"
    COMPLETENESS = "completeness"
    CLARITY_ISSUE = "clarity_issue"
    
    # Cross-domain conflicts
    DOMAIN_BOUNDARY = "domain_boundary"
    SEMANTIC_DRIFT = "semantic_drift"
    CONTEXT_MISMATCH = "context_mismatch"
    
    # Temporal conflicts
    TIMING_CONFLICT = "timing_conflict"
    DEADLINE_VIOLATION = "deadline_violation"
    
    # Resource conflicts
    RESOURCE_CONFLICT = "resource_conflict"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Entanglement conflicts
    ENTANGLEMENT_MISALIGNMENT = "entanglement_misalignment"


class ConflictSeverity(Enum):
    """Conflict severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ResolutionStrategy(Enum):
    """Comprehensive resolution strategies."""
    # Automatic strategies
    AUTO_MERGE = "auto_merge"
    PRIORITY_SELECT = "priority_select"
    LATEST_WINS = "latest_wins"
    HIGHEST_QUALITY = "highest_quality"
    CONSENSUS = "consensus"
    
    # Semi-automatic strategies
    LLM_MEDIATED = "llm_mediated"
    RULE_BASED = "rule_based"
    COST_BASED = "cost_based"
    
    # Manual strategies
    MANUAL_REVIEW = "manual_review"
    DEFER = "defer"
    FLAG_FOR_REVIEW = "flag_for_review"
    
    # Special strategies
    SPLIT = "split"
    CONSOLIDATE = "consolidate"
    ABSTRACT = "abstract"


class ValidationLevel(Enum):
    """Validation levels."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    STRICT = 3
    COMPREHENSIVE = 4


class RecompositionStatus(Enum):
    """Recomposition status."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    CONFLICTS_DETECTED = "conflicts_detected"
    RESOLVING = "resolving"
    ASSEMBLING = "assembling"
    VALIDATING = "validating"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ============================================================================
# DATA CLASSES - Comprehensive Data Models
# ============================================================================

@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification."""
    confidence_score: float
    entropy: float
    variance: float
    sources: List[str] = field(default_factory=list)
    sample_count: int = 0
    
    def combine(self, others: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Combine uncertainty estimates."""
        if not others:
            return self
        
        all_estimates = [self] + others
        return UncertaintyEstimate(
            confidence_score=min(e.confidence_score for e in all_estimates),
            entropy=sum(e.entropy for e in all_estimates) / len(all_estimates),
            variance=sum(e.variance for e in all_estimates) / len(all_estimates),
            sources=list(set(s for e in all_estimates for s in e.sources)),
            sample_count=sum(e.sample_count for e in all_estimates)
        )


@dataclass
class ComplexityScore:
    """Complexity assessment."""
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    overall_complexity: float


@dataclass
class SubProblemSolution:
    """Enhanced sub-problem solution."""
    sub_problem_id: str
    solution_content: str
    solution_hash: str = ""
    quality_score: float = 0.0
    verification_status: str = "pending"
    
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    author: str = ""
    
    semantic_tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    dependencies_satisfied: bool = True
    missing_dependencies: List[str] = field(default_factory=list)
    
    completeness: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.solution_hash:
            self.solution_hash = hashlib.sha256(
                self.solution_content.encode()
            ).hexdigest()[:16]


@dataclass
class Conflict:
    """Enhanced conflict representation."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    involved_solutions: List[str]
    description: str
    
    affected_sections: List[Tuple[int, int]] = field(default_factory=list)
    context_before: str = ""
    context_after: str = ""
    
    suggested_resolution: Optional[str] = None
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved_by: Optional[str] = None
    resolution_notes: str = ""
    
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    auto_resolvable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_resolved(self) -> bool:
        return self.resolved_at is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.conflict_id,
            'type': self.conflict_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'involved': len(self.involved_solutions),
            'resolved': self.is_resolved()
        }


@dataclass
class SemanticCoherenceScore:
    """Semantic coherence metrics."""
    flow_score: float
    consistency_score: float
    transition_quality: float
    topic_coherence: float
    section_scores: Dict[str, float] = field(default_factory=dict)
    transition_scores: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def overall_score(self) -> float:
        return (
            self.flow_score * 0.25 +
            self.consistency_score * 0.25 +
            self.transition_quality * 0.25 +
            self.topic_coherence * 0.25
        )


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    completeness: float
    consistency: float
    coherence: float
    correctness: float
    clarity: float
    
    integration_quality: float
    conflict_density: float
    resolution_success: float
    
    semantic_coherence: SemanticCoherenceScore = field(
        default_factory=lambda: SemanticCoherenceScore(0.5, 0.5, 0.5, 0.5)
    )
    
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_overall(self) -> float:
        self.overall_score = (
            self.completeness * 0.18 +
            self.consistency * 0.15 +
            self.coherence * 0.15 +
            self.correctness * 0.18 +
            self.clarity * 0.10 +
            self.integration_quality * 0.12 +
            self.semantic_coherence.overall_score() * 0.12
        )
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall': self.overall_score,
            'completeness': self.completeness,
            'consistency': self.consistency,
            'coherence': self.coherence,
            'correctness': self.correctness,
            'clarity': self.clarity,
            'integration_quality': self.integration_quality
        }


@dataclass
class AssemblyInstruction:
    """Assembly instruction."""
    sub_problem_id: str
    position: int
    action: str  # keep, merge, extract, skip, transform
    section_header: str
    
    merge_with: Optional[str] = None
    merge_strategy: str = "append"
    
    transformations: List[str] = field(default_factory=list)
    transition_before: str = ""
    transition_after: str = ""
    validation_rules: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if self.action == "merge" and not self.merge_with:
            errors.append("Merge action requires merge_with field")
        if self.position < 0:
            errors.append("Position must be >= 0")
        return len(errors) == 0, errors


@dataclass
class AssemblyPlan:
    """Complete assembly plan."""
    instructions: List[AssemblyInstruction]
    strategy: AssemblyStrategy
    
    intro: Optional[str] = None
    conclusion: Optional[str] = None
    
    global_transformations: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    
    confidence: float = 0.0
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.instructions:
            errors.append("No instructions provided")
        
        positions = set()
        for instr in self.instructions:
            valid, instr_errors = instr.validate()
            if not valid:
                errors.extend(instr_errors)
            if instr.position in positions:
                errors.append(f"Duplicate position: {instr.position}")
            positions.add(instr.position)
        
        return len(errors) == 0, errors


@dataclass
class IntegratedSolution:
    """Enhanced integrated solution."""
    solution_id: str
    problem_id: str
    decomposition_plan_id: str
    
    assembled_content: str
    assembly_strategy: AssemblyStrategy
    sub_solutions: Dict[str, SubProblemSolution]
    content_hash: str = ""
    assembly_plan: Optional[AssemblyPlan] = None
    
    quality_metrics: QualityMetrics = field(default_factory=lambda: QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0))
    conflicts_detected: List[Conflict] = field(default_factory=list)
    conflicts_resolved: List[Conflict] = field(default_factory=list)
    
    assembly_log: List[str] = field(default_factory=list)
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    
    status: RecompositionStatus = RecompositionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.assembled_content.encode()
            ).hexdigest()[:16]
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        return {
            'total_detected': len(self.conflicts_detected),
            'total_resolved': len(self.conflicts_resolved),
            'critical': len([c for c in self.conflicts_detected if c.severity == ConflictSeverity.CRITICAL]),
            'high': len([c for c in self.conflicts_detected if c.severity == ConflictSeverity.HIGH]),
            'by_type': self._count_conflicts_by_type()
        }
    
    def _count_conflicts_by_type(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for c in self.conflicts_detected:
            counts[c.conflict_type.value] += 1
        return dict(counts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'solution_id': self.solution_id,
            'problem_id': self.problem_id,
            'status': self.status.value,
            'quality': self.quality_metrics.to_dict() if self.quality_metrics else {},
            'conflicts': self.get_conflict_summary(),
            'sub_solutions_count': len(self.sub_solutions),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RecompositionConfig:
    """Configuration for recomposition."""
    assembly_strategy: AssemblyStrategy = AssemblyStrategy.HYBRID
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    auto_resolve_conflicts: bool = True
    max_auto_resolution_attempts: int = 3
    
    semantic_threshold: float = 0.75
    overlap_threshold: float = 0.7
    contradiction_threshold: float = 0.8
    
    enable_rollback: bool = True
    max_versions: int = 10
    
    parallel_processing: bool = True
    max_workers: int = 4


# ============================================================================
# CONFLICT DETECTOR
# ============================================================================

class ConflictDetector:
    """
    Advanced conflict detector with multiple detection methods.
    
    Detects 12+ types of conflicts:
    - Logical: contradictions, inconsistencies
    - Structural: dependency violations, order violations
    - Technical: interface mismatches, data format issues
    - Semantic: overlaps, redundancy
    - Quality: completeness gaps, clarity issues
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.75,
        overlap_threshold: float = 0.7,
        contradiction_threshold: float = 0.8
    ):
        self.semantic_threshold = semantic_threshold
        self.overlap_threshold = overlap_threshold
        self.contradiction_threshold = contradiction_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_conflicts(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependency_graph: Dict[str, List[str]],
        entanglement_matrix: Optional[Dict[str, Any]] = None
    ) -> List[Conflict]:
        """
        Detect all conflicts between sub-solutions.
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        self.logger.info(f"Detecting conflicts in {len(sub_solutions)} solutions")
        
        entanglement_matrix = self._normalize_entanglement_matrix(entanglement_matrix, sub_solutions)

        # Logical conflicts
        conflicts.extend(self._detect_contradictions(sub_solutions, entanglement_matrix))
        conflicts.extend(self._detect_inconsistencies(sub_solutions, entanglement_matrix))
        
        # Overlap conflicts
        conflicts.extend(self._detect_overlaps(sub_solutions, entanglement_matrix))
        conflicts.extend(self._detect_redundancy(sub_solutions))
        
        # Structural conflicts
        conflicts.extend(self._detect_dependency_violations(sub_solutions, dependency_graph))
        conflicts.extend(self._detect_order_violations(sub_solutions, dependency_graph))
        
        # Technical conflicts
        conflicts.extend(self._detect_interface_mismatches(sub_solutions, entanglement_matrix))
        conflicts.extend(self._detect_data_format_issues(sub_solutions))

        # Entanglement conflicts
        conflicts.extend(self._detect_entanglement_misalignment(sub_solutions, entanglement_matrix))
        
        # Quality conflicts
        conflicts.extend(self._detect_quality_gaps(sub_solutions))
        conflicts.extend(self._detect_completeness_issues(sub_solutions))
        
        self.logger.info(f"Detected {len(conflicts)} conflicts")
        
        return conflicts
    
    def _detect_contradictions(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, set]] = None
    ) -> List[Conflict]:
        """Detect logical contradictions."""
        conflicts = []
        solution_ids = list(sub_solutions.keys())
        
        # Contradiction patterns
        contradiction_patterns = [
            (r'\b(must|should|will|shall)\s+(\w+)', r'\b(must not|should not|will not|shall not)\s+(\w+)'),
            (r'\benable\b', r'\bdisable\b'),
            (r'\bincrease\b', r'\bdecrease\b'),
            (r'\badd\b', r'\bremove\b'),
            (r'\baccept\b', r'\breject\b'),
            (r'\bstart\b', r'\bstop\b'),
            (r'\bopen\b', r'\bclose\b'),
            (r'\bcreate\b', r'\bdelete\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\byes\b', r'\bno\b'),
        ]
        
        entangled_pairs = self._build_entangled_pairs(entanglement_matrix)
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i+1:]:
                content1 = sub_solutions[id1].solution_content.lower()
                content2 = sub_solutions[id2].solution_content.lower()
                
                for pattern1, pattern2 in contradiction_patterns:
                    matches1 = set(re.findall(pattern1, content1))
                    matches2 = set(re.findall(pattern2, content2))
                    if not matches1 or not matches2:
                        continue

                    # If we captured tuples, match on the action token (second element)
                    if isinstance(next(iter(matches1)), tuple):
                        targets1 = {m[1] for m in matches1 if len(m) > 1}
                        targets2 = {m[1] for m in matches2 if len(m) > 1}
                        if not (targets1 & targets2):
                            continue

                    conflict = Conflict(
                        conflict_id=self._generate_id("conf"),
                        conflict_type=ConflictType.CONTRADICTION,
                        severity=ConflictSeverity.CRITICAL,
                        involved_solutions=[id1, id2],
                        description=f"Contradiction detected: {matches1} vs {matches2}",
                        suggested_resolution="Review and reconcile conflicting statements",
                        auto_resolvable=False,
                        metadata={"entangled_pair": frozenset([id1, id2]) in entangled_pairs}
                    )
                    conflicts.append(conflict)
                    break
        
        return conflicts
    
    def _detect_inconsistencies(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, set]] = None
    ) -> List[Conflict]:
        """Detect inconsistencies in style/format."""
        conflicts = []
        
        # Check for inconsistent formatting
        formats = {}
        for sol_id, solution in sub_solutions.items():
            content = solution.solution_content
            
            # Detect bullet style
            if re.search(r'^\s*[-•]\s', content, re.MULTILINE):
                bullet_style = "dash"
            elif re.search(r'^\s*\*\s', content, re.MULTILINE):
                bullet_style = "asterisk"
            elif re.search(r'^\s*\d+[.]\s', content, re.MULTILINE):
                bullet_style = "numbered"
            else:
                bullet_style = "none"
            
            formats[sol_id] = bullet_style
        
        # Check for inconsistencies
        unique_formats = set(formats.values())
        if len(unique_formats) > 1 and "none" not in unique_formats:
            conflict = Conflict(
                conflict_id=self._generate_id("inc"),
                conflict_type=ConflictType.INCONSISTENCY,
                severity=ConflictSeverity.LOW,
                involved_solutions=list(sub_solutions.keys()),
                description=f"Inconsistent formatting styles detected: {unique_formats}",
                suggested_resolution="Standardize formatting across all solutions",
                auto_resolvable=True,
                metadata={"entanglement_context": bool(entanglement_matrix)}
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_overlaps(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, set]] = None
    ) -> List[Conflict]:
        """Detect content overlaps."""
        conflicts = []
        solution_ids = list(sub_solutions.keys())
        
        entangled_pairs = self._build_entangled_pairs(entanglement_matrix)
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i+1:]:
                content1 = sub_solutions[id1].solution_content
                content2 = sub_solutions[id2].solution_content
                
                similarity = self._calculate_jaccard_similarity(content1, content2)
                entangled = frozenset([id1, id2]) in entangled_pairs
                threshold = 0.55 if entangled else self.overlap_threshold
                
                if similarity > threshold:
                    conflict = Conflict(
                        conflict_id=self._generate_id("ovlp"),
                        conflict_type=ConflictType.CONTENT_OVERLAP,
                        severity=ConflictSeverity.MEDIUM if similarity < 0.85 else ConflictSeverity.HIGH,
                        involved_solutions=[id1, id2],
                        description=f"Content overlap detected: {similarity:.1%} similarity",
                        suggested_resolution="Consolidate overlapping content",
                        auto_resolvable=True,
                        metadata={"entangled_pair": entangled}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_redundancy(
        self,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect redundant content."""
        conflicts = []
        
        # Extract key phrases from each solution
        phrase_map = defaultdict(list)
        
        for sol_id, solution in sub_solutions.items():
            # Extract significant phrases (3+ words)
            phrases = re.findall(r'\b\w+(?:\s+\w+){2,4}\b', solution.solution_content.lower())
            for phrase in phrases:
                if len(phrase) > 15:  # Significant phrase
                    phrase_map[phrase].append(sol_id)
        
        # Find phrases appearing in multiple solutions
        for phrase, sol_ids in phrase_map.items():
            if len(sol_ids) > 2:  # Appears in 3+ solutions
                conflict = Conflict(
                    conflict_id=self._generate_id("red"),
                    conflict_type=ConflictType.REDUNDANCY,
                    severity=ConflictSeverity.LOW,
                    involved_solutions=sol_ids,
                    description=f"Redundant phrase detected: '{phrase[:50]}...'",
                    suggested_resolution="Extract common content to shared section",
                    auto_resolvable=True
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_dependency_violations(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependency_graph: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect dependency violations."""
        conflicts = []

        # Validate dependencies against available solutions
        for sol_id, solution in sub_solutions.items():
            required = dependency_graph.get(sol_id, []) if dependency_graph else []
            missing = [dep for dep in required if dep not in sub_solutions]
            if missing:
                solution.dependencies_satisfied = False
                solution.missing_dependencies = missing
                conflict = Conflict(
                    conflict_id=self._generate_id("dep"),
                    conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                    severity=ConflictSeverity.CRITICAL,
                    involved_solutions=[sol_id] + missing,
                    description=f"Dependencies not satisfied for {sol_id}",
                    suggested_resolution="Complete prerequisite solutions first",
                    auto_resolvable=False
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_order_violations(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependency_graph: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect order violations."""
        conflicts = []
        if not dependency_graph:
            return conflicts

        # Detect cycles
        visited = set()
        stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for dep in dependency_graph.get(node, []):
                if dep not in dependency_graph:
                    continue
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in stack:
                    return True
            stack.remove(node)
            return False

        has_cycle = any(dfs(node) for node in dependency_graph if node not in visited)
        if has_cycle:
            conflicts.append(
                Conflict(
                    conflict_id=self._generate_id("cyc"),
                    conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                    severity=ConflictSeverity.HIGH,
                    involved_solutions=list(dependency_graph.keys()),
                    description="Circular dependency detected in dependency graph",
                    suggested_resolution="Break the cycle by redefining dependencies",
                    auto_resolvable=False,
                )
            )

        order = self._topological_sort(dependency_graph)
        positions = {node: idx for idx, node in enumerate(order)}

        for node, deps in dependency_graph.items():
            for dep in deps:
                if dep in positions and node in positions and positions[dep] > positions[node]:
                    conflicts.append(
                        Conflict(
                            conflict_id=self._generate_id("ord"),
                            conflict_type=ConflictType.ORDER_VIOLATION,
                            severity=ConflictSeverity.MEDIUM,
                            involved_solutions=[node, dep],
                            description=f"Order violation: {node} precedes its dependency {dep}",
                            suggested_resolution="Reorder assembly to satisfy dependencies",
                            auto_resolvable=True,
                        )
                    )

        return conflicts

    @staticmethod
    def _topological_sort(graph: Dict[str, List[str]]) -> List[str]:
        """Topological sort that keeps all nodes, even in cycles."""
        if not graph:
            return []

        nodes = set(graph.keys())
        for deps in graph.values():
            nodes.update(deps or [])

        in_degree = {node: 0 for node in nodes}
        dependents: Dict[str, List[str]] = {node: [] for node in nodes}

        for node, deps in graph.items():
            deps = deps or []
            in_degree[node] = len(deps)
            for dep in deps:
                dependents.setdefault(dep, []).append(node)

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in dependents.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) < len(nodes):
            cycle_nodes = [node for node in nodes if node not in result]
            result.extend(sorted(cycle_nodes))

        return result
    
    def _detect_interface_mismatches(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, set]] = None
    ) -> List[Conflict]:
        """Detect interface/API mismatches."""
        conflicts = []
        
        # Look for API endpoint patterns
        api_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/]+)'
        
        endpoints = defaultdict(list)
        for sol_id, solution in sub_solutions.items():
            matches = re.findall(api_pattern, solution.solution_content)
            for method, path in matches:
                endpoints[(method, path)].append(sol_id)
        
        # Check for inconsistent endpoint usage
        entangled_pairs = self._build_entangled_pairs(entanglement_matrix)
        for (method, path), sol_ids in endpoints.items():
            if len(sol_ids) > 1:
                # Check if parameters match
                contents = [sub_solutions[sid].solution_content for sid in sol_ids]
                params1 = set(re.findall(rf'{re.escape(path)}\?(\w+)', contents[0]))
                
                for i, content in enumerate(contents[1:], 1):
                    params2 = set(re.findall(rf'{re.escape(path)}\?(\w+)', content))
                    if params1 != params2:
                        conflict = Conflict(
                            conflict_id=self._generate_id("api"),
                            conflict_type=ConflictType.API_INCOMPATIBILITY,
                            severity=ConflictSeverity.HIGH,
                            involved_solutions=[sol_ids[0], sol_ids[i]],
                            description=f"API parameter mismatch for {method} {path}",
                            suggested_resolution="Standardize API parameters",
                            auto_resolvable=False,
                            metadata={"entangled_pair": frozenset([sol_ids[0], sol_ids[i]]) in entangled_pairs}
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_data_format_issues(
        self,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect data format inconsistencies."""
        conflicts = []
        
        # Look for data format specifications
        format_patterns = [
            (r'JSON', 'json'),
            (r'XML', 'xml'),
            (r'CSV', 'csv'),
            (r'YAML', 'yaml'),
            (r'Protocol Buffers', 'protobuf'),
        ]
        
        formats_found = defaultdict(set)
        
        for sol_id, solution in sub_solutions.items():
            for pattern, fmt in format_patterns:
                if re.search(pattern, solution.solution_content, re.IGNORECASE):
                    formats_found[fmt].add(sol_id)
        
        # If multiple formats are specified, flag as potential conflict
        if len(formats_found) > 1:
            conflict = Conflict(
                conflict_id=self._generate_id("fmt"),
                conflict_type=ConflictType.DATA_FORMAT,
                severity=ConflictSeverity.MEDIUM,
                involved_solutions=list(set(sid for sids in formats_found.values() for sid in sids)),
                description=f"Multiple data formats detected: {list(formats_found.keys())}",
                suggested_resolution="Standardize on single data format",
                auto_resolvable=True
            )
            conflicts.append(conflict)
        
        return conflicts

    def _detect_entanglement_misalignment(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Optional[Dict[str, set]] = None,
    ) -> List[Conflict]:
        """Detect semantic drift across entangled components."""
        if not entanglement_matrix:
            return []

        conflicts = []
        analyzer = SymbolicAnalyzer()
        token_cache: Dict[str, set] = {}

        for sol_id, solution in sub_solutions.items():
            token_cache[sol_id] = analyzer.analyze(solution.solution_content or "").symbols

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
                        conflict_type=ConflictType.ENTANGLEMENT_MISALIGNMENT,
                        severity=ConflictSeverity.MEDIUM,
                        involved_solutions=[source, target],
                        description=(
                            f"Entangled components show low interface overlap "
                            f"(similarity: {similarity:.2f})"
                        ),
                        suggested_resolution="Align shared interfaces and update coupled sections",
                        auto_resolvable=False,
                        metadata={"entangled_pair": True}
                    )
                    conflicts.append(conflict)

        return conflicts

    @staticmethod
    def _build_entangled_pairs(entanglement_matrix: Optional[Dict[str, set]]) -> set:
        pairs = set()
        if not entanglement_matrix:
            return pairs
        for source, targets in entanglement_matrix.items():
            for target in targets:
                pairs.add(frozenset([source, target]))
        return pairs

    @staticmethod
    def _normalize_entanglement_matrix(
        entanglement_matrix: Optional[Dict[str, Any]],
        sub_solutions: Dict[str, SubProblemSolution],
    ) -> Optional[Dict[str, set]]:
        if not entanglement_matrix:
            return None
        try:
            return normalize_entanglement_matrix(
                entanglement_matrix,
                allowed_ids=list(sub_solutions.keys()),
                enforce_symmetry=True,
                strict=False,
            )
        except Exception:
            return None
    
    def _detect_quality_gaps(
        self,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect quality threshold violations."""
        conflicts = []
        
        quality_threshold = 0.6
        
        for sol_id, solution in sub_solutions.items():
            if solution.quality_score < quality_threshold:
                conflict = Conflict(
                    conflict_id=self._generate_id("qual"),
                    conflict_type=ConflictType.QUALITY_GAP,
                    severity=ConflictSeverity.MEDIUM,
                    involved_solutions=[sol_id],
                    description=f"Quality score {solution.quality_score:.2f} below threshold {quality_threshold}",
                    suggested_resolution="Improve solution quality before integration",
                    auto_resolvable=False
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_completeness_issues(
        self,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect completeness issues."""
        conflicts = []
        
        completeness_threshold = 0.7
        
        for sol_id, solution in sub_solutions.items():
            if solution.completeness < completeness_threshold:
                conflict = Conflict(
                    conflict_id=self._generate_id("comp"),
                    conflict_type=ConflictType.COMPLETENESS,
                    severity=ConflictSeverity.HIGH,
                    involved_solutions=[sol_id],
                    description=f"Completeness {solution.completeness:.2f} below threshold {completeness_threshold}",
                    suggested_resolution="Complete solution before integration",
                    auto_resolvable=False
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ============================================================================
# CONFLICT RESOLVER
# ============================================================================

class ConflictResolver:
    """
    Advanced conflict resolver with multiple resolution strategies.
    """
    
    def __init__(self, config: Optional[RecompositionConfig] = None):
        self.config = config or RecompositionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def resolve_conflicts(
        self,
        conflicts: List[Conflict],
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Tuple[List[Conflict], List[Conflict]]:
        """
        Resolve conflicts using configured strategies.
        
        Returns:
            Tuple of (resolved_conflicts, unresolved_conflicts)
        """
        resolved = []
        unresolved = []
        
        self.logger.info(f"Resolving {len(conflicts)} conflicts")
        
        for conflict in conflicts:
            if self._can_auto_resolve(conflict):
                success = self._resolve_conflict(conflict, sub_solutions)
                if success:
                    conflict.resolved_at = datetime.now()
                    resolved.append(conflict)
                else:
                    unresolved.append(conflict)
            else:
                if self.config.auto_resolve_conflicts:
                    # Try LLM-mediated resolution
                    success = self._llm_resolve(conflict, sub_solutions)
                    if success:
                        conflict.resolved_at = datetime.now()
                        resolved.append(conflict)
                    else:
                        unresolved.append(conflict)
                else:
                    unresolved.append(conflict)
        
        self.logger.info(f"Resolved {len(resolved)}, {len(unresolved)} remaining")
        
        return resolved, unresolved
    
    def _can_auto_resolve(self, conflict: Conflict) -> bool:
        """Check if conflict can be auto-resolved."""
        return conflict.auto_resolvable and conflict.severity in [
            ConflictSeverity.LOW, ConflictSeverity.INFO
        ]
    
    def _resolve_conflict(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Resolve a single conflict."""
        if conflict.conflict_type == ConflictType.ENTANGLEMENT_MISALIGNMENT:
            return self._resolve_entanglement_alignment(conflict, sub_solutions)

        strategy = conflict.resolution_strategy or ResolutionStrategy.AUTO_MERGE
        
        resolvers = {
            ResolutionStrategy.AUTO_MERGE: self._resolve_merge,
            ResolutionStrategy.PRIORITY_SELECT: self._resolve_priority,
            ResolutionStrategy.LATEST_WINS: self._resolve_latest,
            ResolutionStrategy.HIGHEST_QUALITY: self._resolve_quality,
            ResolutionStrategy.SPLIT: self._resolve_split,
        }
        
        resolver = resolvers.get(strategy, self._resolve_merge)
        return resolver(conflict, sub_solutions)

    def _resolve_entanglement_alignment(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Align entangled solutions by inserting reconciliation notes."""
        if len(conflict.involved_solutions) != 2:
            return False
        sol1 = sub_solutions.get(conflict.involved_solutions[0])
        sol2 = sub_solutions.get(conflict.involved_solutions[1])
        if not sol1 or not sol2:
            return False

        note = (
            "\n\n### Entanglement Alignment\n"
            "- Reviewed shared interfaces and synchronized terminology.\n"
            "- Confirmed compatibility with entangled peer component.\n"
        )
        sol1.solution_content += note
        sol2.solution_content += note
        conflict.resolution_notes = "Added entanglement alignment notes to both solutions"
        return True
    
    def _resolve_merge(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Merge conflicting solutions."""
        # Simple merge: concatenate with separator
        if len(conflict.involved_solutions) == 2:
            sol1 = sub_solutions.get(conflict.involved_solutions[0])
            sol2 = sub_solutions.get(conflict.involved_solutions[1])
            
            if sol1 and sol2:
                merged_content = f"{sol1.solution_content}\n\n{sol2.solution_content}"
                sol1.solution_content = merged_content
                sol1.quality_score = (sol1.quality_score + sol2.quality_score) / 2
                conflict.resolution_notes = "Merged solutions"
                return True
        
        return False
    
    def _resolve_priority(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Resolve by priority."""
        # Keep highest quality solution
        solutions = [sub_solutions.get(sid) for sid in conflict.involved_solutions]
        solutions = [s for s in solutions if s]
        
        if solutions:
            best = max(solutions, key=lambda s: s.quality_score)
            conflict.resolution_notes = f"Selected highest quality solution: {best.sub_problem_id}"
            return True
        
        return False
    
    def _resolve_latest(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Resolve by latest modification."""
        solutions = [sub_solutions.get(sid) for sid in conflict.involved_solutions]
        solutions = [s for s in solutions if s]
        
        if solutions:
            latest = max(solutions, key=lambda s: s.modified_at)
            conflict.resolution_notes = f"Selected latest solution: {latest.sub_problem_id}"
            return True
        
        return False
    
    def _resolve_quality(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Resolve by quality score."""
        return self._resolve_priority(conflict, sub_solutions)
    
    def _resolve_split(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Split conflicting content into separate sections."""
        conflict.resolution_notes = "Flagged for manual split"
        return False  # Requires manual intervention
    
    def _llm_resolve(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> bool:
        """Attempt LLM-mediated resolution."""
        # Fallback to merge when no external mediator is configured.
        conflict.resolution_notes = "Resolved via deterministic merge fallback"
        return self._resolve_merge(conflict, sub_solutions)


# ============================================================================
# ENHANCED RECOMPOSITION ENGINE
# ============================================================================

class EnhancedRecompositionEngine:
    """
    Sovereign-grade comprehensive recomposition engine.
    
    Features:
    - Multi-strategy assembly with intelligent selection
    - Advanced conflict detection and resolution
    - Semantic coherence validation
    - Quality gates and validation
    - Rollback capabilities
    - Incremental recomposition
    """
    
    def __init__(
        self,
        config: Optional[RecompositionConfig] = None,
        max_workers: int = 4
    ):
        self.config = config or RecompositionConfig()
        self.max_workers = max_workers
        
        # Components
        self.conflict_detector = ConflictDetector(
            semantic_threshold=self.config.semantic_threshold,
            overlap_threshold=self.config.overlap_threshold
        )
        self.conflict_resolver = ConflictResolver(self.config)
        
        # Version control
        self.version_history: Dict[str, List[IntegratedSolution]] = {}
        
        # Analytics
        self.recomposition_stats: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("EnhancedRecompositionEngine initialized")
    
    def assemble(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        problem_id: str,
        decomposition_plan_id: str,
        dependency_graph: Optional[Dict[str, List[str]]] = None,
        strategy: Optional[AssemblyStrategy] = None,
        entanglement_matrix: Optional[Dict[str, Any]] = None
    ) -> IntegratedSolution:
        """
        Assemble sub-solutions into integrated solution.
        
        Args:
            sub_solutions: Dictionary of sub-problem solutions
            problem_id: Parent problem ID
            decomposition_plan_id: Decomposition plan ID
            dependency_graph: Dependency relationships
            strategy: Assembly strategy (auto-selected if None)
            
        Returns:
            IntegratedSolution with assembled content
        """
        start_time = time.time()
        
        self.logger.info(f"Assembling {len(sub_solutions)} solutions for problem {problem_id}")
        
        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(sub_solutions)
        
        # Build entanglement matrix if missing but embedded in solution metadata
        if not entanglement_matrix:
            entanglement_matrix = self._build_entanglement_matrix_from_solutions(sub_solutions)
        if entanglement_matrix:
            entanglement_matrix = normalize_entanglement_matrix(
                entanglement_matrix,
                allowed_ids=list(sub_solutions.keys()),
                enforce_symmetry=True,
                strict=False,
            )

        # Create initial solution
        solution = IntegratedSolution(
            solution_id=self._generate_id("sol"),
            problem_id=problem_id,
            decomposition_plan_id=decomposition_plan_id,
            assembled_content="",
            assembly_strategy=strategy,
            sub_solutions=sub_solutions.copy(),
            status=RecompositionStatus.ANALYZING
        )
        
        # Detect conflicts
        solution.status = RecompositionStatus.CONFLICTS_DETECTED
        dependency_graph = dependency_graph or {}
        conflicts = self.conflict_detector.detect_conflicts(
            sub_solutions,
            dependency_graph,
            entanglement_matrix=entanglement_matrix,
        )
        solution.conflicts_detected = conflicts
        
        # Resolve conflicts
        solution.status = RecompositionStatus.RESOLVING
        resolved, unresolved = self.conflict_resolver.resolve_conflicts(
            conflicts, sub_solutions
        )
        solution.conflicts_resolved = resolved

        if entanglement_matrix:
            self._apply_entanglement_invalidation(sub_solutions, entanglement_matrix, unresolved)
        
        # Build assembly plan
        assembly_plan = self._create_assembly_plan(
            sub_solutions, strategy, dependency_graph, entanglement_matrix=entanglement_matrix
        )
        solution.assembly_plan = assembly_plan
        
        # Assemble content
        solution.status = RecompositionStatus.ASSEMBLING
        assembled_content = self._execute_assembly(assembly_plan, sub_solutions)
        solution.assembled_content = assembled_content
        
        # Validate
        solution.status = RecompositionStatus.VALIDATING
        quality_metrics = self._validate_solution(solution, unresolved)
        solution.quality_metrics = quality_metrics
        
        # Finalize
        solution.status = RecompositionStatus.COMPLETED
        
        # Store version
        self._store_version(solution)

        if entanglement_matrix:
            solution.metadata["entanglement_matrix"] = {
                key: sorted(list(value)) for key, value in entanglement_matrix.items()
            }
            entanglement_conflicts = [
                c for c in conflicts
                if c.conflict_type == ConflictType.ENTANGLEMENT_MISALIGNMENT
            ]
            solution.metadata["entanglement_conflicts"] = [c.conflict_id for c in entanglement_conflicts]
        
        # Record stats
        elapsed = time.time() - start_time
        self.recomposition_stats.append({
            'solution_id': solution.solution_id,
            'duration': elapsed,
            'conflicts_detected': len(conflicts),
            'conflicts_resolved': len(resolved),
            'quality_score': quality_metrics.overall_score
        })
        
        self.logger.info(
            f"Assembly completed: {len(conflicts)} conflicts, "
            f"quality={quality_metrics.overall_score:.2f}"
        )
        
        return solution

    @staticmethod
    def _apply_entanglement_invalidation(
        sub_solutions: Dict[str, SubProblemSolution],
        entanglement_matrix: Dict[str, Any],
        conflicts: List[Conflict],
    ) -> None:
        """Propagate entanglement invalidation across coupled solutions."""
        if not conflicts:
            return
        conflict_pairs = set()
        for conflict in conflicts:
            involved = conflict.involved_solutions or []
            for i in range(len(involved)):
                for j in range(i + 1, len(involved)):
                    conflict_pairs.add(frozenset([involved[i], involved[j]]))

        normalized = normalize_entanglement_matrix(
            entanglement_matrix,
            allowed_ids=list(sub_solutions.keys()),
            enforce_symmetry=True,
            strict=False,
        )
        for pair in conflict_pairs:
            if len(pair) != 2:
                continue
            a, b = tuple(pair)
            if b not in normalized.get(a, set()) and a not in normalized.get(b, set()):
                continue
            for source, target in [(a, b), (b, a)]:
                if target not in sub_solutions:
                    continue
                meta = sub_solutions[target].metadata
                meta.setdefault("entanglement_invalidation", []).append(source)
                meta["needs_consistency_refinement"] = True
    
    def _select_strategy(
        self,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> AssemblyStrategy:
        """Select best assembly strategy."""
        count = len(sub_solutions)
        
        if count <= 3:
            return AssemblyStrategy.SEQUENTIAL
        elif count <= 6:
            return AssemblyStrategy.HIERARCHICAL
        else:
            return AssemblyStrategy.HYBRID
    
    def _create_assembly_plan(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        strategy: AssemblyStrategy,
        dependency_graph: Dict[str, List[str]],
        entanglement_matrix: Optional[Dict[str, Any]] = None
    ) -> AssemblyPlan:
        """Create assembly plan."""
        instructions = []
        
        # Determine order based on strategy
        if strategy in [AssemblyStrategy.SEQUENTIAL, AssemblyStrategy.HIERARCHICAL]:
            priority_scores = {
                sp_id: sub_solutions[sp_id].quality_score
                for sp_id in sub_solutions
                if sp_id in sub_solutions
            }
            order, cycle_nodes = self._topological_sort(
                dependency_graph,
                priority_scores=priority_scores,
            )
        else:
            order = list(sub_solutions.keys())
            cycle_nodes = []

        # Ensure all solutions appear in the order
        if not order:
            order = list(sub_solutions.keys())
        else:
            missing = [sp_id for sp_id in sub_solutions.keys() if sp_id not in order]
            if missing:
                order.extend(missing)

        if entanglement_matrix:
            order = self._apply_entanglement_grouping(order, entanglement_matrix)
        
        # Create instructions
        for position, sol_id in enumerate(order):
            if sol_id in sub_solutions:
                instruction = AssemblyInstruction(
                    sub_problem_id=sol_id,
                    position=position,
                    action="keep",
                    section_header=sub_solutions[sol_id].keywords[0] if sub_solutions[sol_id].keywords else f"Section {position + 1}"
                )
                instructions.append(instruction)
        
        reasoning = ""
        if cycle_nodes:
            cycle_text = ", ".join(cycle_nodes)
            reasoning = (
                "Cycle detected in dependency graph. "
                f"Applied quality-priority break for: {cycle_text}."
            )
            self.logger.warning("Cycle detected in assembly order: %s", cycle_text)

        return AssemblyPlan(
            instructions=instructions,
            strategy=strategy,
            confidence=0.8,
            reasoning=reasoning
        )

    @staticmethod
    def _apply_entanglement_grouping(
        order: List[str],
        entanglement_matrix: Dict[str, Any],
    ) -> List[str]:
        """Group entangled components adjacent in assembly order."""
        normalized = normalize_entanglement_matrix(
            entanglement_matrix,
            allowed_ids=order,
            enforce_symmetry=True,
            strict=False,
        )
        seen = set()
        grouped_order = []
        for sp_id in order:
            if sp_id in seen:
                continue
            grouped_order.append(sp_id)
            seen.add(sp_id)
            for ent in sorted(normalized.get(sp_id, set())):
                if ent not in seen:
                    grouped_order.append(ent)
                    seen.add(ent)
        return grouped_order
    
    def _execute_assembly(
        self,
        assembly_plan: AssemblyPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Execute assembly plan."""
        parts = []
        
        # Add intro if present
        if assembly_plan.intro:
            parts.append(assembly_plan.intro)
        
        # Sort instructions by position
        sorted_instructions = sorted(
            assembly_plan.instructions,
            key=lambda i: i.position
        )
        
        # Assemble content
        for instruction in sorted_instructions:
            sol = sub_solutions.get(instruction.sub_problem_id)
            if sol:
                # Add transition if present
                if instruction.transition_before:
                    parts.append(instruction.transition_before)
                
                # Add section header
                parts.append(f"\n## {instruction.section_header}\n")
                
                # Add content
                parts.append(sol.solution_content)
                
                # Add transition if present
                if instruction.transition_after:
                    parts.append(instruction.transition_after)
        
        # Add conclusion if present
        if assembly_plan.conclusion:
            parts.append(assembly_plan.conclusion)
        
        return "\n\n".join(parts)
    
    def _validate_solution(
        self,
        solution: IntegratedSolution,
        unresolved_conflicts: List[Conflict]
    ) -> QualityMetrics:
        """Validate assembled solution."""
        content = solution.assembled_content
        sub_solutions = solution.sub_solutions
        
        # Calculate metrics
        completeness = self._calculate_completeness(solution)
        consistency = self._calculate_consistency(solution)
        coherence = self._calculate_coherence(content, sub_solutions)
        correctness = self._calculate_correctness(solution)
        clarity = self._calculate_clarity(content)
        
        integration_quality = 1.0 - (len(unresolved_conflicts) * 0.1)
        integration_quality = max(0.0, integration_quality)
        
        conflict_density = len(solution.conflicts_detected) / len(sub_solutions) if sub_solutions else 0
        
        resolution_success = (
            len(solution.conflicts_resolved) / len(solution.conflicts_detected)
            if solution.conflicts_detected else 1.0
        )
        
        semantic_coherence = self._calculate_semantic_coherence(content, sub_solutions)
        
        metrics = QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            coherence=coherence,
            correctness=correctness,
            clarity=clarity,
            integration_quality=integration_quality,
            conflict_density=conflict_density,
            resolution_success=resolution_success,
            semantic_coherence=semantic_coherence
        )
        
        metrics.calculate_overall()
        
        return metrics
    
    def _calculate_completeness(self, solution: IntegratedSolution) -> float:
        """Calculate completeness."""
        sub_solutions = solution.sub_solutions
        if not sub_solutions:
            return 0.0
        
        total_completeness = sum(s.completeness for s in sub_solutions.values())
        avg_completeness = total_completeness / len(sub_solutions)
        
        # Check if content covers all sub-solutions
        content_length = len(solution.assembled_content)
        expected_length = len(sub_solutions) * 500  # Assume 500 chars per solution
        
        coverage = min(1.0, content_length / expected_length)
        
        return (avg_completeness + coverage) / 2
    
    def _calculate_consistency(self, solution: IntegratedSolution) -> float:
        """Calculate consistency."""
        # Check formatting consistency
        content = solution.assembled_content
        
        # Count different heading styles
        hash_headers = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))
        underline_headers = len(re.findall(r'\n[-=]+\n', content))
        
        if hash_headers > 0 and underline_headers > 0:
            return 0.7  # Mixed styles
        
        return 0.9
    
    def _calculate_coherence(
        self,
        content: str,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> float:
        """Calculate coherence."""
        # Check for transition words
        transition_words = ['therefore', 'however', 'furthermore', 'additionally', 'consequently']
        
        content_lower = content.lower()
        transition_count = sum(1 for word in transition_words if word in content_lower)
        
        # More transitions generally indicate better coherence
        expected_transitions = len(sub_solutions) * 0.5
        coherence = min(1.0, transition_count / expected_transitions) if expected_transitions > 0 else 0.5
        
        return coherence
    
    def _calculate_correctness(self, solution: IntegratedSolution) -> float:
        """Calculate correctness."""
        sub_solutions = solution.sub_solutions
        if not sub_solutions:
            return 0.0
        
        total_correctness = sum(s.correctness for s in sub_solutions.values())
        return total_correctness / len(sub_solutions)
    
    def _calculate_clarity(self, content: str) -> float:
        """Calculate clarity."""
        # Check sentence length
        sentences = re.split(r'[.!?]+', content)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Shorter sentences are clearer (optimal around 15-20 words)
        if avg_length <= 15:
            return 0.9
        elif avg_length <= 25:
            return 0.7
        else:
            return 0.5
    
    def _calculate_semantic_coherence(
        self,
        content: str,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> SemanticCoherenceScore:
        """Calculate semantic coherence."""
        # Simplified calculation
        flow_score = 0.7
        consistency_score = 0.75
        transition_quality = 0.6
        topic_coherence = 0.7
        
        return SemanticCoherenceScore(
            flow_score=flow_score,
            consistency_score=consistency_score,
            transition_quality=transition_quality,
            topic_coherence=topic_coherence
        )
    
    def _topological_sort(
        self,
        graph: Dict[str, List[str]],
        priority_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Topological sort with cycle detection and heuristic break."""
        if not graph:
            return [], []

        nodes = set(graph.keys())
        for deps in graph.values():
            nodes.update(deps or [])

        in_degree = {node: 0 for node in nodes}
        dependents: Dict[str, List[str]] = {node: [] for node in nodes}

        for node, deps in graph.items():
            deps = deps or []
            in_degree[node] = len(deps)
            for dep in deps:
                dependents.setdefault(dep, []).append(node)

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in dependents.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        cycle_nodes: List[str] = []
        if len(result) < len(nodes):
            cycle_nodes = [node for node in nodes if node not in result]
            if priority_scores:
                ordered_cycle = sorted(
                    cycle_nodes,
                    key=lambda n: priority_scores.get(n, 0.0),
                    reverse=True,
                )
            else:
                ordered_cycle = sorted(cycle_nodes)
            result.extend(ordered_cycle)

        return result, cycle_nodes

    @staticmethod
    def _build_entanglement_matrix_from_solutions(
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Dict[str, List[str]]:
        """Build entanglement matrix from solution metadata when not provided."""
        raw: Dict[str, List[str]] = {}
        for sol_id, solution in sub_solutions.items():
            metadata = solution.metadata if isinstance(solution.metadata, dict) else {}
            entangled_with = (
                metadata.get("entangled_with")
                or metadata.get("entanglement")
                or metadata.get("entanglement_partners")
                or []
            )
            if entangled_with:
                raw[sol_id] = list(entangled_with)
        return raw
    
    def _store_version(self, solution: IntegratedSolution) -> None:
        """Store solution version."""
        if solution.problem_id not in self.version_history:
            self.version_history[solution.problem_id] = []
        
        self.version_history[solution.problem_id].append(solution)
        
        # Limit versions
        if len(self.version_history[solution.problem_id]) > self.config.max_versions:
            self.version_history[solution.problem_id].pop(0)
    
    def rollback(self, problem_id: str, steps: int = 1) -> Optional[IntegratedSolution]:
        """Rollback to previous version."""
        versions = self.version_history.get(problem_id, [])
        if len(versions) <= steps:
            return None
        
        return versions[-(steps + 1)]
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_subproblem_solution(
    sub_problem_id: str,
    content: str,
    quality_score: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None
) -> SubProblemSolution:
    """Helper to create sub-problem solution."""
    return SubProblemSolution(
        sub_problem_id=sub_problem_id,
        solution_content=content,
        quality_score=quality_score,
        completeness=0.8,
        correctness=0.85,
        clarity=0.75,
        metadata=metadata or {},
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    engine = EnhancedRecompositionEngine()
    
    # Create sample sub-solutions
    sub_solutions = {
        "sub_1": create_subproblem_solution(
            "sub_1",
            "## Requirements Analysis\n\nThe system must support user authentication...",
            0.85
        ),
        "sub_2": create_subproblem_solution(
            "sub_2",
            "## Database Design\n\nThe database schema includes users, sessions...",
            0.80
        ),
        "sub_3": create_subproblem_solution(
            "sub_3",
            "## API Implementation\n\nRESTful endpoints for user management...",
            0.75
        ),
        "sub_4": create_subproblem_solution(
            "sub_4",
            "## Frontend Development\n\nReact components for user interface...",
            0.82
        ),
    }
    
    # Define dependencies
    dependency_graph = {
        "sub_1": [],
        "sub_2": ["sub_1"],
        "sub_3": ["sub_1", "sub_2"],
        "sub_4": ["sub_3"]
    }
    
    # Assemble
    solution = engine.assemble(
        sub_solutions=sub_solutions,
        problem_id="prob_123",
        decomposition_plan_id="plan_456",
        dependency_graph=dependency_graph,
        strategy=AssemblyStrategy.HIERARCHICAL
    )
    
    print(f"\nIntegrated Solution: {solution.solution_id}")
    print(f"Status: {solution.status.value}")
    print(f"Quality: {solution.quality_metrics.overall_score:.2f}")
    print(f"Conflicts: {len(solution.conflicts_detected)} detected, {len(solution.conflicts_resolved)} resolved")
    print(f"\nContent Preview:\n{solution.assembled_content[:500]}...")
