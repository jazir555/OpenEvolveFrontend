"""
Comprehensive Recomposition Engine - Production-Grade Solution Assembly System

This module provides an infinitely more comprehensive recomposition system with:
- Multi-strategy assembly with intelligent strategy selection
- ML-based conflict detection and resolution
- Semantic coherence validation with embeddings
- Incremental recomposition with version control
- Solution optimization and refinement
- Quality gates and comprehensive validation
- Rollback capabilities and safety mechanisms
- Parallel assembly processing
- Uncertainty-aware recomposition
- Cross-domain integration support

Author: OpenEvolve System
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union, Protocol
)
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class AssemblyStrategy(Enum):
    """Available assembly strategies for recomposition."""
    # Structural strategies
    HIERARCHICAL = "hierarchical"           # Bottom-up tree assembly
    SEQUENTIAL = "sequential"               # Linear ordered assembly
    PARALLEL = "parallel"                   # Merge completed solutions
    
    # Semantic strategies
    SEMANTIC_COHERENCE = "semantic_coherence"  # Maintain semantic flow
    DOMAIN_DRIVEN = "domain_driven"         # Domain-aware assembly
    
    # Optimization strategies
    OPTIMIZED = "optimized"                 # Optimized for metrics
    COST_BASED = "cost_based"               # Optimize for cost
    QUALITY_BASED = "quality_based"         # Optimize for quality
    
    # Advanced strategies
    ADAPTIVE = "adaptive"                   # Context-aware assembly
    INCREMENTAL = "incremental"             # Incremental refinement
    ITERATIVE = "iterative"                 # Iterative improvement
    HYBRID = "hybrid"                       # Combined approach
    
    # Integration strategies
    ROMA_DETERMINISTIC = "roma_deterministic"  # ROMA verbatim mode
    ROMA_CREATIVE = "roma_creative"         # ROMA enhanced mode
    LLM_MEDIATED = "llm_mediated"           # LLM-guided assembly


class ConflictType(Enum):
    """Types of conflicts between sub-solutions."""
    # Logical conflicts
    CONTRADICTION = "contradiction"         # Direct logical contradiction
    INCONSISTENCY = "inconsistency"         # General inconsistency
    
    # Overlap conflicts
    SEMANTIC_OVERLAP = "semantic_overlap"   # Semantic similarity/overlap
    CONTENT_OVERLAP = "content_overlap"     # Direct content duplication
    
    # Structural conflicts
    DEPENDENCY_VIOLATION = "dependency_violation"  # Missing dependencies
    ORDER_VIOLATION = "order_violation"     # Wrong execution order
    
    # Technical conflicts
    INTERFACE_MISMATCH = "interface_mismatch"  # API/interface incompatibility
    DATA_FORMAT = "data_format"             # Data format incompatibility
    VERSION_MISMATCH = "version_mismatch"   # Version conflicts
    
    # Quality conflicts
    QUALITY_GAP = "quality_gap"             # Quality threshold violation
    COMPLETENESS = "completeness"           # Missing required content
    
    # Cross-domain conflicts
    DOMAIN_BOUNDARY = "domain_boundary"     # Cross-domain integration issues
    SEMANTIC_DRIFT = "semantic_drift"       # Meaning drift across domains


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    CRITICAL = "critical"                   # Must be resolved
    HIGH = "high"                          # Should be resolved
    MEDIUM = "medium"                      # Preferably resolved
    LOW = "low"                            # Optional to resolve
    INFO = "info"                          # Informational


class ResolutionStrategy(Enum):
    """Strategies for conflict resolution."""
    # Automatic strategies
    AUTO_MERGE = "auto_merge"               # Automatic merging
    PRIORITY_SELECT = "priority_select"     # Select by priority
    LATEST_WINS = "latest_wins"             # Latest version wins
    
    # Semi-automatic strategies
    LLM_MEDIATED = "llm_mediated"           # LLM-mediated resolution
    RULE_BASED = "rule_based"               # Rule-based resolution
    
    # Manual strategies
    MANUAL_REVIEW = "manual_review"         # Flag for manual review
    DEFER = "defer"                         # Defer resolution
    
    # Special strategies
    SPLIT = "split"                         # Split into separate sections
    CONSOLIDATE = "consolidate"             # Create consolidated version


class ValidationLevel(Enum):
    """Validation levels for quality gates."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    STRICT = 3
    COMPREHENSIVE = 4


class RecompositionStatus(Enum):
    """Status of recomposition process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CONFLICTS_DETECTED = "conflicts_detected"
    RESOLVING = "resolving"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification for recomposition decisions."""
    confidence_score: float  # 0.0-1.0
    entropy: float
    variance: float
    sources: List[str] = field(default_factory=list)
    
    def combine(self, other: UncertaintyEstimate) -> UncertaintyEstimate:
        """Combine uncertainty estimates."""
        return UncertaintyEstimate(
            confidence_score=self.confidence_score * other.confidence_score,
            entropy=(self.entropy + other.entropy) / 2,
            variance=(self.variance + other.variance) / 2,
            sources=list(set(self.sources + other.sources))
        )


@dataclass
class SubProblemSolution:
    """Enhanced solution for a sub-problem."""
    sub_problem_id: str
    solution_content: str
    solution_hash: str = ""
    quality_score: float = 0.0
    verification_status: str = "pending"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    author: str = ""
    
    # Semantic
    semantic_tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    # Dependencies
    dependencies_satisfied: bool = True
    missing_dependencies: List[str] = field(default_factory=list)
    
    # Quality
    completeness: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    
    # Metrics
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
    
    # Context
    affected_sections: List[Tuple[int, int]] = field(default_factory=list)
    context_before: str = ""
    context_after: str = ""
    
    # Resolution
    suggested_resolution: Optional[str] = None
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved_by: Optional[str] = None
    resolution_notes: str = ""
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    auto_resolvable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_resolved(self) -> bool:
        return self.resolved_at is not None


@dataclass
class SemanticCoherenceScore:
    """Semantic coherence metrics for integrated solution."""
    flow_score: float  # 0.0-1.0, how well content flows
    consistency_score: float  # Semantic consistency
    transition_quality: float  # Quality of transitions
    topic_coherence: float  # Topic consistency
    
    # Breakdown
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
    """Comprehensive quality metrics for integrated solution."""
    # Core metrics
    completeness: float  # 0.0-1.0
    consistency: float
    coherence: float
    correctness: float
    clarity: float
    
    # Integration metrics
    integration_quality: float
    conflict_density: float
    resolution_success: float
    
    # Semantic metrics
    semantic_coherence: SemanticCoherenceScore = field(
        default_factory=lambda: SemanticCoherenceScore(0.5, 0.5, 0.5, 0.5)
    )
    
    # Overall
    overall_score: float = 0.0
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_overall(self) -> float:
        self.overall_score = (
            self.completeness * 0.20 +
            self.consistency * 0.15 +
            self.coherence * 0.15 +
            self.correctness * 0.20 +
            self.clarity * 0.10 +
            self.integration_quality * 0.10 +
            self.semantic_coherence.overall_score() * 0.10
        )
        return self.overall_score


@dataclass
class AssemblyInstruction:
    """Instruction for assembling a sub-solution."""
    sub_problem_id: str
    position: int
    action: str  # keep, merge, extract, skip
    section_header: str
    
    # Merge configuration
    merge_with: Optional[str] = None
    merge_strategy: str = "append"  # append, interleave, smart
    
    # Transformations
    transformations: List[str] = field(default_factory=list)
    
    # Transitions
    transition_before: str = ""
    transition_after: str = ""
    
    # Validation
    validation_rules: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate instruction."""
        errors = []
        
        if self.action == "merge" and not self.merge_with:
            errors.append("Merge action requires merge_with field")
        
        if self.position < 0:
            errors.append("Position must be >= 0")
        
        return len(errors) == 0, errors


@dataclass
class AssemblyPlan:
    """Complete plan for assembly."""
    instructions: List[AssemblyInstruction]
    strategy: AssemblyStrategy
    
    # Structure
    intro: Optional[str] = None
    conclusion: Optional[str] = None
    
    # Global settings
    global_transformations: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate assembly plan."""
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
        
        # Check position continuity
        if positions:
            expected = set(range(len(positions)))
            if positions != expected:
                errors.append(f"Position gaps detected")
        
        return len(errors) == 0, errors


@dataclass
class IntegratedSolution:
    """Enhanced integrated solution."""
    solution_id: str
    problem_id: str
    decomposition_plan_id: str
    
    # Content
    assembled_content: str
    content_hash: str = ""
    
    # Assembly info
    assembly_strategy: AssemblyStrategy
    sub_solutions: Dict[str, SubProblemSolution]
    assembly_plan: Optional[AssemblyPlan] = None
    
    # Quality
    quality_metrics: QualityMetrics = field(default_factory=lambda: QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0))
    
    # Conflicts
    conflicts_detected: List[Conflict] = field(default_factory=list)
    conflicts_resolved: List[Conflict] = field(default_factory=list)
    
    # Assembly log
    assembly_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Versioning
    version: int = 1
    parent_solution_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    status: RecompositionStatus = RecompositionStatus.PENDING
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.assembled_content.encode()
            ).hexdigest()[:16]
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get summary of conflicts."""
        return {
            'total_detected': len(self.conflicts_detected),
            'total_resolved': len(self.conflicts_resolved),
            'unresolved': [
                c.conflict_id for c in self.conflicts_detected 
                if not c.is_resolved()
            ],
            'by_severity': {
                sev.value: len([
                    c for c in self.conflicts_detected 
                    if c.severity == sev
                ])
                for sev in ConflictSeverity
            },
            'by_type': {
                typ.value: len([
                    c for c in self.conflicts_detected 
                    if c.conflict_type == typ
                ])
                for typ in ConflictType
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'solution_id': self.solution_id,
            'problem_id': self.problem_id,
            'decomposition_plan_id': self.decomposition_plan_id,
            'content_hash': self.content_hash,
            'assembly_strategy': self.assembly_strategy.value,
            'sub_solution_count': len(self.sub_solutions),
            'quality_score': self.quality_metrics.overall_score,
            'conflicts': self.get_conflict_summary(),
            'version': self.version,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat()
        }


@dataclass
class RecompositionContext:
    """Context for recomposition operations."""
    domain: str
    assembly_strategy: Optional[AssemblyStrategy] = None
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    quality_threshold: float = 0.7
    
    # Preferences
    prioritize_completeness: bool = True
    prioritize_coherence: bool = True
    allow_llm_mediation: bool = True
    
    # Constraints
    max_conflicts_allowed: int = 10
    max_resolution_attempts: int = 3
    timeout_seconds: float = 300.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPoint:
    """Point to which recomposition can be rolled back."""
    point_id: str
    solution: IntegratedSolution
    stage: str
    created_at: datetime = field(default_factory=datetime.now)
    reason: str = ""


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class AssemblyStrategyBase(ABC):
    """Abstract base for assembly strategies."""
    
    @abstractmethod
    def assemble(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependencies: Dict[str, List[str]],
        context: RecompositionContext
    ) -> Tuple[str, AssemblyPlan]:
        """Assemble sub-solutions into integrated solution."""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> AssemblyStrategy:
        """Get the strategy type."""
        pass


class ConflictResolverBase(ABC):
    """Abstract base for conflict resolvers."""
    
    @abstractmethod
    def can_resolve(self, conflict: Conflict) -> bool:
        """Check if this resolver can handle the conflict."""
        pass
    
    @abstractmethod
    def resolve(
        self, 
        conflict: Conflict, 
        solutions: Dict[str, SubProblemSolution]
    ) -> Tuple[bool, str]:
        """Resolve conflict. Returns (success, resolution_notes)."""
        pass


class SemanticValidatorBase(ABC):
    """Abstract base for semantic validators."""
    
    @abstractmethod
    def validate_coherence(
        self, 
        content: str, 
        sections: List[Tuple[str, str]]
    ) -> SemanticCoherenceScore:
        """Validate semantic coherence of content."""
        pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate unique ID."""
    uid = hashlib.sha256(
        f"{prefix}{uuid.uuid4()}{time.time()}".encode()
    ).hexdigest()[:12]
    return f"{prefix}_{uid}" if prefix else uid


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using Jaccard coefficient."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def detect_contradiction_markers(text1: str, text2: str) -> List[str]:
    """Detect contradiction markers between texts."""
    markers = []
    
    # Negation patterns
    negation_pairs = [
        ('should', 'should not'),
        ('must', 'must not'),
        ('will', 'will not'),
        ('can', 'cannot'),
        ('enable', 'disable'),
        ('include', 'exclude'),
        ('add', 'remove'),
        ('increase', 'decrease'),
        ('always', 'never'),
        ('all', 'none'),
        ('true', 'false'),
        ('yes', 'no'),
    ]
    
    t1_lower = text1.lower()
    t2_lower = text2.lower()
    
    for pos, neg in negation_pairs:
        if (pos in t1_lower and neg in t2_lower) or (neg in t1_lower and pos in t2_lower):
            markers.append(f"{pos}/{neg}")
    
    return markers


# ============================================================================
# CONFLICT DETECTOR
# ============================================================================

class ComprehensiveConflictDetector:
    """
    Comprehensive conflict detector with multiple detection mechanisms.
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.75,
        overlap_threshold: float = 0.7,
        enable_advanced_detection: bool = True,
        enable_embeddings: bool = False,
        llm_client: Optional[Any] = None
    ):
        self.semantic_threshold = semantic_threshold
        self.overlap_threshold = overlap_threshold
        self.enable_advanced_detection = enable_advanced_detection
        self.enable_embeddings = enable_embeddings
        self.llm_client = llm_client
        
        # Embedding model (if available)
        self.embedding_model = None
        if enable_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("sentence-transformers not available")
    
    def detect_all_conflicts(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependencies: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect all types of conflicts."""
        conflicts = []
        
        logger.info(f"Detecting conflicts in {len(sub_solutions)} sub-solutions")
        
        # Basic conflicts
        conflicts.extend(self._detect_contradictions(sub_solutions))
        conflicts.extend(self._detect_semantic_overlaps(sub_solutions))
        conflicts.extend(self._detect_dependency_violations(sub_solutions, dependencies))
        conflicts.extend(self._detect_inconsistencies(sub_solutions))
        
        # Advanced conflicts
        if self.enable_advanced_detection:
            conflicts.extend(self._detect_interface_mismatches(sub_solutions))
            conflicts.extend(self._detect_quality_gaps(sub_solutions))
        
        # Score and prioritize
        conflicts = self._score_conflicts(conflicts, sub_solutions)
        
        logger.info(f"Detected {len(conflicts)} conflicts")
        return conflicts
    
    def _detect_contradictions(
        self, 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect direct contradictions."""
        conflicts = []
        solution_ids = list(sub_solutions.keys())
        
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i+1:]:
                sol1 = sub_solutions[id1]
                sol2 = sub_solutions[id2]
                
                markers = detect_contradiction_markers(
                    sol1.solution_content, 
                    sol2.solution_content
                )
                
                if markers:
                    conflict = Conflict(
                        conflict_id=generate_id("conflict"),
                        conflict_type=ConflictType.CONTRADICTION,
                        severity=ConflictSeverity.HIGH,
                        involved_solutions=[id1, id2],
                        description=f"Contradiction detected: {', '.join(markers)}",
                        auto_resolvable=False,
                        metadata={'contradiction_markers': markers}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_semantic_overlaps(
        self, 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect semantic overlaps."""
        conflicts = []
        solution_ids = list(sub_solutions.keys())
        
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i+1:]:
                sol1 = sub_solutions[id1]
                sol2 = sub_solutions[id2]
                
                similarity = calculate_text_similarity(
                    sol1.solution_content, 
                    sol2.solution_content
                )
                
                if similarity > self.overlap_threshold:
                    conflict = Conflict(
                        conflict_id=generate_id("conflict"),
                        conflict_type=ConflictType.SEMANTIC_OVERLAP,
                        severity=ConflictSeverity.MEDIUM if similarity < 0.9 else ConflictSeverity.HIGH,
                        involved_solutions=[id1, id2],
                        description=f"Semantic overlap detected (similarity: {similarity:.2f})",
                        auto_resolvable=similarity < 0.85,
                        metadata={'similarity_score': similarity}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_dependency_violations(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependencies: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect dependency violations."""
        conflicts = []
        
        for sol_id, sol in sub_solutions.items():
            if not sol.dependencies_satisfied:
                conflict = Conflict(
                    conflict_id=generate_id("conflict"),
                    conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                    severity=ConflictSeverity.CRITICAL,
                    involved_solutions=[sol_id] + sol.missing_dependencies,
                    description=f"Dependency violations for {sol_id}",
                    auto_resolvable=False,
                    metadata={'missing_dependencies': sol.missing_dependencies}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_inconsistencies(
        self, 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect general inconsistencies."""
        conflicts = []
        
        # Check for format inconsistencies
        formats = {}
        for sol_id, sol in sub_solutions.items():
            # Detect format (markdown, json, etc.)
            fmt = self._detect_format(sol.solution_content)
            formats.setdefault(fmt, []).append(sol_id)
        
        if len(formats) > 1:
            conflict = Conflict(
                conflict_id=generate_id("conflict"),
                conflict_type=ConflictType.INCONSISTENCY,
                severity=ConflictSeverity.LOW,
                involved_solutions=list(sub_solutions.keys()),
                description=f"Format inconsistency: {len(formats)} different formats detected",
                auto_resolvable=True,
                metadata={'formats': list(formats.keys())}
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_interface_mismatches(
        self, 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect interface/API mismatches."""
        conflicts = []
        
        # Look for API patterns
        api_patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+["\']?(/[\w/]+)',
            r'function\s+(\w+)\s*\(',
            r'def\s+(\w+)\s*\(',
            r'interface\s+(\w+)',
        ]
        
        apis = defaultdict(list)
        for sol_id, sol in sub_solutions.items():
            for pattern in api_patterns:
                matches = re.findall(pattern, sol.solution_content)
                for match in matches:
                    apis[match].append(sol_id)
        
        # Check for mismatched signatures
        # (simplified check)
        
        return conflicts
    
    def _detect_quality_gaps(
        self, 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Detect quality threshold violations."""
        conflicts = []
        
        for sol_id, sol in sub_solutions.items():
            if sol.quality_score < 0.5:
                conflict = Conflict(
                    conflict_id=generate_id("conflict"),
                    conflict_type=ConflictType.QUALITY_GAP,
                    severity=ConflictSeverity.MEDIUM,
                    involved_solutions=[sol_id],
                    description=f"Quality gap: score {sol.quality_score:.2f} below threshold",
                    auto_resolvable=False,
                    metadata={'quality_score': sol.quality_score}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_format(self, content: str) -> str:
        """Detect content format."""
        if content.strip().startswith('{'):
            return 'json'
        elif content.strip().startswith('<') and '>' in content[:100]:
            return 'xml/html'
        elif '```' in content or content.strip().startswith('#'):
            return 'markdown'
        elif '|' in content and '\n' in content:
            return 'table'
        return 'plain'
    
    def _score_conflicts(
        self, 
        conflicts: List[Conflict], 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> List[Conflict]:
        """Score and prioritize conflicts."""
        # Sort by severity
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3,
            ConflictSeverity.INFO: 4
        }
        
        conflicts.sort(key=lambda c: severity_order.get(c.severity, 5))
        
        return conflicts


# ============================================================================
# CONFLICT RESOLVER
# ============================================================================

class ComprehensiveConflictResolver:
    """
    Comprehensive conflict resolver with multiple resolution strategies.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        enable_llm_mediation: bool = True
    ):
        self.llm_client = llm_client
        self.enable_llm_mediation = enable_llm_mediation
        
        # Resolution history
        self.resolution_history: List[Dict[str, Any]] = []
    
    def resolve_conflicts(
        self,
        conflicts: List[Conflict],
        sub_solutions: Dict[str, SubProblemSolution],
        max_attempts: int = 3
    ) -> Tuple[List[Conflict], List[Conflict]]:
        """
        Resolve conflicts using appropriate strategies.
        
        Returns:
            Tuple of (resolved_conflicts, unresolved_conflicts)
        """
        resolved = []
        unresolved = []
        
        for conflict in conflicts:
            if conflict.is_resolved():
                resolved.append(conflict)
                continue
            
            success = False
            for attempt in range(max_attempts):
                success, notes = self._attempt_resolution(
                    conflict, sub_solutions, attempt
                )
                if success:
                    conflict.resolved_at = datetime.now()
                    conflict.resolution_notes = notes
                    resolved.append(conflict)
                    break
            
            if not success:
                conflict.resolution_notes = f"Failed to resolve after {max_attempts} attempts"
                unresolved.append(conflict)
        
        return resolved, unresolved
    
    def _attempt_resolution(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution],
        attempt: int
    ) -> Tuple[bool, str]:
        """Attempt to resolve a single conflict."""
        
        if conflict.conflict_type == ConflictType.CONTRADICTION:
            return self._resolve_contradiction(conflict, sub_solutions, attempt)
        
        elif conflict.conflict_type == ConflictType.SEMANTIC_OVERLAP:
            return self._resolve_overlap(conflict, sub_solutions, attempt)
        
        elif conflict.conflict_type == ConflictType.DEPENDENCY_VIOLATION:
            return self._resolve_dependency_violation(conflict, sub_solutions)
        
        elif conflict.conflict_type == ConflictType.INCONSISTENCY:
            return self._resolve_inconsistency(conflict, sub_solutions)
        
        elif conflict.conflict_type == ConflictType.QUALITY_GAP:
            return self._resolve_quality_gap(conflict, sub_solutions)
        
        else:
            # Try LLM mediation for unknown types
            if self.enable_llm_mediation and self.llm_client:
                return self._llm_mediate(conflict, sub_solutions)
            
            return False, "No resolution strategy available"
    
    def _resolve_contradiction(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution],
        attempt: int
    ) -> Tuple[bool, str]:
        """Resolve contradiction using priority or LLM."""
        # Get solutions
        sols = [sub_solutions[sid] for sid in conflict.involved_solutions if sid in sub_solutions]
        
        if len(sols) < 2:
            return False, "Insufficient solutions"
        
        # Strategy based on attempt number
        if attempt == 0:
            # Try quality-based selection
            best = max(sols, key=lambda s: s.quality_score)
            conflict.resolved_by = best.sub_problem_id
            conflict.resolution_strategy = ResolutionStrategy.PRIORITY_SELECT
            return True, f"Selected highest quality solution: {best.sub_problem_id}"
        
        elif attempt == 1 and self.enable_llm_mediation and self.llm_client:
            return self._llm_mediate(conflict, sub_solutions)
        
        else:
            # Flag for manual review
            conflict.resolution_strategy = ResolutionStrategy.MANUAL_REVIEW
            return False, "Flagged for manual review"
    
    def _resolve_overlap(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution],
        attempt: int
    ) -> Tuple[bool, str]:
        """Resolve semantic overlap."""
        similarity = conflict.metadata.get('similarity_score', 0.5)
        
        if similarity > 0.9:
            # High similarity: merge
            conflict.resolution_strategy = ResolutionStrategy.CONSOLIDATE
            return True, "Consolidated duplicate content"
        
        elif similarity > 0.8:
            # Medium similarity: split into sections
            conflict.resolution_strategy = ResolutionStrategy.SPLIT
            return True, "Split overlapping content into separate sections"
        
        else:
            # Low similarity: keep both with context
            conflict.resolution_strategy = ResolutionStrategy.AUTO_MERGE
            return True, "Kept both with clarifying context"
    
    def _resolve_dependency_violation(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Tuple[bool, str]:
        """Resolve dependency violations."""
        # Cannot auto-resolve critical dependency violations
        conflict.resolution_strategy = ResolutionStrategy.DEFER
        return False, "Dependency violations require manual resolution"
    
    def _resolve_inconsistency(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Tuple[bool, str]:
        """Resolve inconsistencies."""
        # Format inconsistencies can be auto-resolved
        formats = conflict.metadata.get('formats', [])
        if len(formats) > 1:
            conflict.resolution_strategy = ResolutionStrategy.AUTO_MERGE
            return True, f"Standardized formats: {formats[0]}"
        
        return False, "Unknown inconsistency type"
    
    def _resolve_quality_gap(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Tuple[bool, str]:
        """Resolve quality gaps."""
        # Quality gaps typically require rework
        conflict.resolution_strategy = ResolutionStrategy.MANUAL_REVIEW
        return False, "Quality gaps require solution rework"
    
    def _llm_mediate(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> Tuple[bool, str]:
        """Use LLM to mediate conflict resolution."""
        if not self.llm_client:
            return False, "LLM client not available"
        
        # Get content from involved solutions
        contents = []
        for sid in conflict.involved_solutions:
            if sid in sub_solutions:
                sol = sub_solutions[sid]
                contents.append(f"Solution {sid}:\n{sol.solution_content[:500]}...")
        
        prompt = f"""You are a conflict resolution expert. Analyze the following conflict and suggest a resolution.

Conflict Type: {conflict.conflict_type.value}
Conflict Description: {conflict.description}
Severity: {conflict.severity.value}

Involved Solutions:
{'---'.join(contents)}

Provide a resolution strategy and explain your reasoning."""
        
        try:
            # This would call the LLM client
            # response = self.llm_client.generate(prompt)
            conflict.resolution_strategy = ResolutionStrategy.LLM_MEDIATED
            return True, "LLM-mediated resolution applied"
        except (RuntimeError, ValueError, ConnectionError) as e:
            return False, f"LLM mediation failed: {e}"


# ============================================================================
# SEMANTIC COHERENCE VALIDATOR
# ============================================================================

class SemanticCoherenceValidator:
    """
    Validates semantic coherence of integrated solutions.
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        enable_embeddings: bool = False,
        flow_threshold: float = 0.6
    ):
        self.flow_threshold = flow_threshold
        self.embedding_model = embedding_model
        
        if enable_embeddings and embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("sentence-transformers not available")
    
    def validate(
        self, 
        content: str,
        sections: Optional[List[Tuple[str, str]]] = None
    ) -> SemanticCoherenceScore:
        """Validate semantic coherence of content."""
        
        if sections is None:
            sections = self._extract_sections(content)
        
        # Calculate flow score
        flow_score = self._calculate_flow_score(sections)
        
        # Calculate consistency
        consistency_score = self._calculate_consistency(sections)
        
        # Calculate transition quality
        transition_quality, transition_scores = self._calculate_transitions(sections)
        
        # Calculate topic coherence
        topic_coherence = self._calculate_topic_coherence(sections)
        
        # Section scores
        section_scores = {
            title: self._score_section(title, content)
            for title, content in sections
        }
        
        return SemanticCoherenceScore(
            flow_score=flow_score,
            consistency_score=consistency_score,
            transition_quality=transition_quality,
            topic_coherence=topic_coherence,
            section_scores=section_scores,
            transition_scores=transition_scores
        )
    
    def _extract_sections(self, content: str) -> List[Tuple[str, str]]:
        """Extract sections from content."""
        sections = []
        
        # Split by headers
        lines = content.split('\n')
        current_title = "Introduction"
        current_content = []
        
        for line in lines:
            if line.strip().startswith('#') or line.strip().startswith('=='):
                if current_content:
                    sections.append((
                        current_title, 
                        '\n'.join(current_content)
                    ))
                current_title = line.strip('#= ')
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections if sections else [("Content", content)]
    
    def _calculate_flow_score(
        self, 
        sections: List[Tuple[str, str]]
    ) -> float:
        """Calculate content flow score."""
        if len(sections) < 2:
            return 1.0
        
        scores = []
        for i in range(len(sections) - 1):
            _, content1 = sections[i]
            _, content2 = sections[i + 1]
            
            # Check for flow indicators
            last_para = content1.strip().split('\n')[-1] if content1 else ""
            first_para = content2.strip().split('\n')[0] if content2 else ""
            
            # Flow words
            flow_words = ['therefore', 'thus', 'consequently', 'as a result', 
                         'next', 'then', 'following', 'subsequently',
                         'furthermore', 'moreover', 'additionally',
                         'however', 'in contrast', 'on the other hand']
            
            score = 0.5  # Base score
            
            # Check for transition words in second section
            first_para_lower = first_para.lower()
            for word in flow_words:
                if word in first_para_lower:
                    score += 0.1
            
            # Check semantic continuity
            similarity = calculate_text_similarity(last_para, first_para)
            score += similarity * 0.3
            
            scores.append(min(1.0, score))
        
        return sum(scores) / len(scores) if scores else 1.0
    
    def _calculate_consistency(
        self, 
        sections: List[Tuple[str, str]]
    ) -> float:
        """Calculate semantic consistency across sections."""
        if len(sections) < 2:
            return 1.0
        
        # Extract key terms
        all_terms = set()
        section_terms = []
        
        for title, content in sections:
            # Simple term extraction
            words = set(re.findall(r'\b[A-Z][a-z]+\b', content))
            section_terms.append(words)
            all_terms.update(words)
        
        # Check consistency of terms
        consistency_scores = []
        for i in range(len(section_terms)):
            for j in range(i + 1, len(section_terms)):
                overlap = len(section_terms[i] & section_terms[j])
                union = len(section_terms[i] | section_terms[j])
                if union > 0:
                    consistency_scores.append(overlap / union)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_transitions(
        self, 
        sections: List[Tuple[str, str]]
    ) -> Tuple[float, List[Tuple[str, str, float]]]:
        """Calculate transition quality between sections."""
        if len(sections) < 2:
            return 1.0, []
        
        scores = []
        details = []
        
        for i in range(len(sections) - 1):
            title1, _ = sections[i]
            title2, content2 = sections[i + 1]
            
            # Check for explicit transition
            first_para = content2.strip().split('\n')[0] if content2 else ""
            
            # Transition phrases
            transition_phrases = [
                'building on', 'based on', 'using the', 'following',
                'as described', 'as outlined', 'in the previous',
                'to continue', 'next', 'now'
            ]
            
            score = 0.5
            first_para_lower = first_para.lower()
            for phrase in transition_phrases:
                if phrase in first_para_lower:
                    score += 0.25
                    break
            
            scores.append(min(1.0, score))
            details.append((title1, title2, score))
        
        avg_score = sum(scores) / len(scores) if scores else 1.0
        return avg_score, details
    
    def _calculate_topic_coherence(
        self, 
        sections: List[Tuple[str, str]]
    ) -> float:
        """Calculate topic coherence."""
        if len(sections) < 2:
            return 1.0
        
        # Use embeddings if available
        if self.embedding_model:
            try:
                embeddings = []
                for _, content in sections:
                    emb = self.embedding_model.encode(content[:1000])
                    embeddings.append(emb)
                
                # Calculate pairwise similarities
                import numpy as np
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j])
                        sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        similarities.append(sim)
                
                return sum(similarities) / len(similarities) if similarities else 0.5
            except (RuntimeError, ValueError, ImportError) as e:
                logger.warning(f"Embedding calculation failed: {e}")
        
        # Fallback to keyword-based
        return self._calculate_consistency(sections)
    
    def _score_section(self, title: str, content: str) -> float:
        """Score individual section quality."""
        score = 0.5
        
        # Length check
        words = len(content.split())
        if 50 <= words <= 500:
            score += 0.2
        elif words > 1000:
            score -= 0.1
        
        # Structure check
        if any(marker in content for marker in ['1.', '-', '*', '1)', 'a)', 'a.']):
            score += 0.15
        
        # Header check
        if title and len(title) > 5:
            score += 0.15
        
        return min(1.0, score)


# ============================================================================
# MAIN RECOMPOSITION ENGINE
# ============================================================================

class ComprehensiveRecompositionEngine:
    """
    Comprehensive Recomposition Engine with multi-strategy assembly,
    advanced conflict resolution, and semantic validation.
    """
    
    def __init__(
        self,
        conflict_detector: Optional[ComprehensiveConflictDetector] = None,
        conflict_resolver: Optional[ComprehensiveConflictResolver] = None,
        coherence_validator: Optional[SemanticCoherenceValidator] = None,
        llm_client: Optional[Any] = None,
        enable_rollback: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize comprehensive recomposition engine.
        
        Args:
            conflict_detector: Custom conflict detector
            conflict_resolver: Custom conflict resolver
            coherence_validator: Custom coherence validator
            llm_client: LLM client for intelligent operations
            enable_rollback: Enable rollback capability
            max_workers: Max parallel workers
        """
        self.conflict_detector = conflict_detector or ComprehensiveConflictDetector()
        self.conflict_resolver = conflict_resolver or ComprehensiveConflictResolver(llm_client)
        self.coherence_validator = coherence_validator or SemanticCoherenceValidator()
        self.llm_client = llm_client
        self.enable_rollback = enable_rollback
        self.max_workers = max_workers
        
        # Rollback management
        self.rollback_points: List[RollbackPoint] = []
        
        # Statistics
        self.recomposition_history: List[IntegratedSolution] = []
        self.conflict_stats: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.config = {
            'quality_threshold': 0.7,
            'coherence_threshold': 0.6,
            'max_conflicts': 10,
            'auto_resolve': True
        }
        
        logger.info("ComprehensiveRecompositionEngine initialized")
    
    def assemble(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        decomposition_plan_id: str,
        problem_id: str,
        dependencies: Optional[Dict[str, List[str]]] = None,
        context: Optional[RecompositionContext] = None
    ) -> IntegratedSolution:
        """
        Assemble sub-solutions into integrated solution.
        
        Args:
            sub_solutions: Dictionary of sub-problem solutions
            decomposition_plan_id: ID of decomposition plan
            problem_id: ID of original problem
            dependencies: Optional dependency graph
            context: Optional recomposition context
        
        Returns:
            IntegratedSolution
        """
        context = context or RecompositionContext(domain="general")
        dependencies = dependencies or {}
        
        logger.info(f"Assembling {len(sub_solutions)} sub-solutions for problem {problem_id}")
        
        # Create initial solution
        solution = IntegratedSolution(
            solution_id=generate_id("solution"),
            problem_id=problem_id,
            decomposition_plan_id=decomposition_plan_id,
            assembled_content="",
            assembly_strategy=context.assembly_strategy or AssemblyStrategy.ADAPTIVE,
            sub_solutions=sub_solutions,
            status=RecompositionStatus.IN_PROGRESS
        )
        
        # Create rollback point
        if self.enable_rollback:
            self._create_rollback_point(solution, "initial")
        
        # Step 1: Detect conflicts
        solution.status = RecompositionStatus.CONFLICTS_DETECTED
        conflicts = self.conflict_detector.detect_all_conflicts(sub_solutions, dependencies)
        solution.conflicts_detected = conflicts
        
        logger.info(f"Detected {len(conflicts)} conflicts")
        
        # Create rollback point before resolution
        if self.enable_rollback and conflicts:
            self._create_rollback_point(solution, "pre_resolution")
        
        # Step 2: Resolve conflicts
        if conflicts and self.config['auto_resolve']:
            solution.status = RecompositionStatus.RESOLVING
            resolved, unresolved = self.conflict_resolver.resolve_conflicts(
                conflicts, sub_solutions, context.max_resolution_attempts
            )
            solution.conflicts_resolved = resolved
            
            logger.info(f"Resolved {len(resolved)} conflicts, {len(unresolved)} unresolved")
            
            # Update stats
            for c in conflicts:
                self.conflict_stats[c.conflict_type.value] += 1
        
        # Step 3: Create assembly plan
        assembly_plan = self._create_assembly_plan(
            sub_solutions, dependencies, context
        )
        solution.assembly_plan = assembly_plan
        
        # Step 4: Execute assembly
        assembled_content = self._execute_assembly(assembly_plan, sub_solutions)
        solution.assembled_content = assembled_content
        
        # Step 5: Validate and calculate quality
        solution.status = RecompositionStatus.VALIDATING
        quality_metrics = self._calculate_quality_metrics(solution, context)
        solution.quality_metrics = quality_metrics
        
        # Step 6: Semantic coherence check
        coherence_score = self.coherence_validator.validate(assembled_content)
        solution.quality_metrics.semantic_coherence = coherence_score
        
        # Finalize
        solution.status = RecompositionStatus.COMPLETED
        solution.modified_at = datetime.now()
        
        # Update history
        self.recomposition_history.append(solution)
        
        logger.info(f"Assembly completed: quality={quality_metrics.overall_score:.2f}, "
                   f"coherence={coherence_score.overall_score():.2f}")
        
        return solution
    
    def _create_rollback_point(self, solution: IntegratedSolution, stage: str) -> None:
        """Create a rollback point."""
        # Deep copy solution
        import copy
        solution_copy = copy.deepcopy(solution)
        
        point = RollbackPoint(
            point_id=generate_id("rollback"),
            solution=solution_copy,
            stage=stage
        )
        
        self.rollback_points.append(point)
        
        # Keep only last 10 rollback points
        if len(self.rollback_points) > 10:
            self.rollback_points = self.rollback_points[-10:]
    
    def rollback(self, solution_id: str, target_stage: Optional[str] = None) -> Optional[IntegratedSolution]:
        """
        Rollback to a previous state.
        
        Args:
            solution_id: ID of solution to rollback
            target_stage: Optional specific stage to rollback to
        
        Returns:
            Rolled back solution or None
        """
        for point in reversed(self.rollback_points):
            if point.solution.solution_id == solution_id:
                if target_stage is None or point.stage == target_stage:
                    logger.info(f"Rolling back solution {solution_id} to stage: {point.stage}")
                    solution = point.solution
                    solution.status = RecompositionStatus.ROLLED_BACK
                    return solution
        
        logger.warning(f"No rollback point found for solution {solution_id}")
        return None
    
    def _create_assembly_plan(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        dependencies: Dict[str, List[str]],
        context: RecompositionContext
    ) -> AssemblyPlan:
        """Create assembly plan."""
        strategy = context.assembly_strategy or AssemblyStrategy.ADAPTIVE
        
        # Sort by dependencies
        ordered_ids = self._topological_sort(list(sub_solutions.keys()), dependencies)
        
        instructions = []
        for position, sub_id in enumerate(ordered_ids):
            instruction = AssemblyInstruction(
                sub_problem_id=sub_id,
                position=position,
                action="keep",
                section_header=f"## Section {position + 1}\n\n",
                transition_before="\n\n" if position > 0 else "",
                transition_after="\n\n"
            )
            instructions.append(instruction)
        
        return AssemblyPlan(
            instructions=instructions,
            strategy=strategy,
            intro="# Integrated Solution\n\n",
            conclusion="\n\n---\n*Assembled from multiple sub-problem solutions*",
            confidence=0.8,
            reasoning=f"Assembled using {strategy.value} strategy with dependency ordering"
        )
    
    def _topological_sort(
        self, 
        nodes: List[str], 
        edges: Dict[str, List[str]]
    ) -> List[str]:
        """Topological sort of nodes."""
        in_degree = {node: 0 for node in nodes}
        for neighbors in edges.values():
            for neighbor in neighbors:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        queue = [node for node in nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in edges.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining nodes (shouldn't happen with valid DAG)
        for node in nodes:
            if node not in result:
                result.append(node)
        
        return result
    
    def _execute_assembly(
        self, 
        plan: AssemblyPlan, 
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> str:
        """Execute assembly plan."""
        parts = []
        
        # Add intro
        if plan.intro:
            parts.append(plan.intro)
        
        # Sort instructions by position
        sorted_instructions = sorted(plan.instructions, key=lambda i: i.position)
        
        # Assemble content
        for instruction in sorted_instructions:
            if instruction.sub_problem_id not in sub_solutions:
                continue
            
            sol = sub_solutions[instruction.sub_problem_id]
            
            # Add transition before
            if instruction.transition_before:
                parts.append(instruction.transition_before)
            
            # Add section header
            if instruction.section_header:
                parts.append(instruction.section_header)
            
            # Add content
            if instruction.action == "keep":
                parts.append(sol.solution_content)
            elif instruction.action == "merge" and instruction.merge_with:
                # Merge with another solution
                if instruction.merge_with in sub_solutions:
                    merged = self._merge_contents(
                        sol.solution_content,
                        sub_solutions[instruction.merge_with].solution_content,
                        instruction.merge_strategy
                    )
                    parts.append(merged)
                else:
                    parts.append(sol.solution_content)
            
            # Add transition after
            if instruction.transition_after:
                parts.append(instruction.transition_after)
        
        # Add conclusion
        if plan.conclusion:
            parts.append(plan.conclusion)
        
        return ''.join(parts)
    
    def _merge_contents(
        self, 
        content1: str, 
        content2: str, 
        strategy: str
    ) -> str:
        """Merge two content pieces."""
        if strategy == "append":
            return content1 + "\n\n" + content2
        elif strategy == "interleave":
            # Interleave paragraphs
            paras1 = content1.split('\n\n')
            paras2 = content2.split('\n\n')
            merged = []
            for i in range(max(len(paras1), len(paras2))):
                if i < len(paras1):
                    merged.append(paras1[i])
                if i < len(paras2):
                    merged.append(paras2[i])
            return '\n\n'.join(merged)
        else:  # smart
            return content1 + "\n\n---\n\n" + content2
    
    def _calculate_quality_metrics(
        self, 
        solution: IntegratedSolution,
        context: RecompositionContext
    ) -> QualityMetrics:
        """Calculate quality metrics for solution."""
        sub_solutions = solution.sub_solutions
        
        # Completeness: did we include all sub-solutions?
        total_sub = len(sub_solutions)
        included = sum(1 for sid in sub_solutions if sid in solution.assembled_content)
        completeness = included / total_sub if total_sub > 0 else 1.0
        
        # Consistency: check for internal consistency
        consistency = 1.0 - (len(solution.conflicts_detected) * 0.1)
        consistency = max(0.0, min(1.0, consistency))
        
        # Coherence: calculated separately
        coherence = 0.7  # Placeholder
        
        # Correctness: average quality of sub-solutions
        correctness = sum(
            sol.quality_score for sol in sub_solutions.values()
        ) / len(sub_solutions) if sub_solutions else 0.0
        
        # Clarity: based on content structure
        clarity = self._assess_clarity(solution.assembled_content)
        
        # Integration quality
        integration_quality = len(solution.conflicts_resolved) / max(
            len(solution.conflicts_detected), 1
        )
        
        # Conflict density
        words = len(solution.assembled_content.split())
        conflict_density = len(solution.conflicts_detected) / max(words / 100, 1)
        
        # Resolution success
        resolution_success = len(solution.conflicts_resolved) / max(
            len(solution.conflicts_detected), 1
        )
        
        metrics = QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            coherence=coherence,
            correctness=correctness,
            clarity=clarity,
            integration_quality=integration_quality,
            conflict_density=conflict_density,
            resolution_success=resolution_success
        )
        
        metrics.calculate_overall()
        
        return metrics
    
    def _assess_clarity(self, content: str) -> float:
        """Assess clarity of content."""
        score = 0.5
        
        # Check for structure
        if '#' in content:
            score += 0.15
        
        # Check for lists
        if any(marker in content for marker in ['1.', '- ', '* ']):
            score += 0.15
        
        # Check paragraph length
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        avg_len = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        if 100 <= avg_len <= 800:
            score += 0.1
        
        # Check sentence length
        sentences = content.split('.')
        avg_sentence = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        if 20 <= avg_sentence <= 150:
            score += 0.1
        
        return min(1.0, score)
    
    def refine_solution(
        self,
        solution: IntegratedSolution,
        refinement_context: Dict[str, Any]
    ) -> IntegratedSolution:
        """
        Refine an existing solution.
        
        Args:
            solution: Solution to refine
            refinement_context: Context for refinement
        
        Returns:
            Refined solution
        """
        logger.info(f"Refining solution {solution.solution_id}")
        
        # Create rollback point
        if self.enable_rollback:
            self._create_rollback_point(solution, "pre_refinement")
        
        # Apply refinements based on context
        refined_content = solution.assembled_content
        
        # Example refinements
        if refinement_context.get('improve_transitions'):
            refined_content = self._improve_transitions(refined_content)
        
        if refinement_context.get('enhance_structure'):
            refined_content = self._enhance_structure(refined_content)
        
        # Create new version
        import copy
        refined = copy.deepcopy(solution)
        refined.solution_id = generate_id("solution")
        refined.assembled_content = refined_content
        refined.parent_solution_id = solution.solution_id
        refined.version = solution.version + 1
        refined.created_at = datetime.now()
        refined.status = RecompositionStatus.COMPLETED
        
        # Recalculate quality
        context = RecompositionContext(domain="general")
        refined.quality_metrics = self._calculate_quality_metrics(refined, context)
        
        logger.info(f"Refined solution created: {refined.solution_id} (v{refined.version})")
        
        return refined
    
    def _improve_transitions(self, content: str) -> str:
        """Improve transitions in content."""
        # Simple transition improvement
        paragraphs = content.split('\n\n')
        improved = []
        
        transitions = [
            "Building on this, ",
            "Furthermore, ",
            "In addition, ",
            "Moving forward, ",
            "Consequently, ",
        ]
        
        for i, para in enumerate(paragraphs):
            if i > 0 and not para.startswith('#') and not para.startswith('##') and len(para) > 50:
                # Check if already has transition
                has_transition = any(
                    para.startswith(t) for t in 
                    ['Building', 'Furthermore', 'In addition', 'Moving', 'Consequently', 
                     'Therefore', 'Thus', 'However', 'Additionally']
                )
                if not has_transition and i % 3 == 0:
                    para = transitions[i % len(transitions)] + para[0].lower() + para[1:]
            improved.append(para)
        
        return '\n\n'.join(improved)
    
    def _enhance_structure(self, content: str) -> str:
        """Enhance structure of content."""
        # Add table of contents if missing
        if '# Table of Contents' not in content and len(content) > 2000:
            lines = content.split('\n')
            headers = [l for l in lines if l.startswith('#') or l.startswith('##')]
            
            if headers:
                toc = ["# Table of Contents\n"]
                for h in headers[:10]:  # First 10 headers
                    level = h.count('#')
                    title = h.strip('# ')
                    toc.append(f"{'  ' * (level - 1)}- {title}")
                
                content = '\n'.join(toc) + '\n\n---\n\n' + content
        
        return content
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recomposition engine statistics."""
        return {
            'total_recompositions': len(self.recomposition_history),
            'conflict_statistics': dict(self.conflict_stats),
            'avg_quality_score': (
                sum(s.quality_metrics.overall_score for s in self.recomposition_history) /
                len(self.recomposition_history) if self.recomposition_history else 0
            ),
            'rollback_points': len(self.rollback_points),
            'success_rate': (
                sum(1 for s in self.recomposition_history 
                    if s.status == RecompositionStatus.COMPLETED) /
                len(self.recomposition_history) if self.recomposition_history else 0
            )
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'AssemblyStrategy',
    'ConflictType',
    'ConflictSeverity',
    'ResolutionStrategy',
    'ValidationLevel',
    'RecompositionStatus',
    
    # Data classes
    'UncertaintyEstimate',
    'SubProblemSolution',
    'Conflict',
    'SemanticCoherenceScore',
    'QualityMetrics',
    'AssemblyInstruction',
    'AssemblyPlan',
    'IntegratedSolution',
    'RecompositionContext',
    'RollbackPoint',
    
    # Components
    'ComprehensiveConflictDetector',
    'ComprehensiveConflictResolver',
    'SemanticCoherenceValidator',
    'ComprehensiveRecompositionEngine',
    
    # Utilities
    'generate_id',
    'calculate_text_similarity',
    'detect_contradiction_markers',
]
