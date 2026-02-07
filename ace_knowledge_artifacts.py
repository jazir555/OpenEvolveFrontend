"""
ACE Knowledge Artifacts Module

This module defines the knowledge artifact schemas and structures for Stage 6
Knowledge Extraction in the Sovereign-Grade Decomposition Workflow.

Knowledge Artifacts are structured learning outputs extracted from workflow
executions that capture reusable patterns, solutions, and insights.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
from enum import Enum
import json
import hashlib
import uuid
import logging
import copy
import os

# SECURITY FIX: Phase 1 - Import security utilities
from ace_security_utils import (
    validate_and_resolve_path,
    validate_file_path_safe,
    safe_load_json_file,
    atomic_save_json_file,
    validate_numeric_range,
    validate_list_size,
    validate_string_length,
    validate_model_name,
    validate_dict_structure,
    create_safe_error,
    sanitize_for_logging,
)

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of knowledge artifacts."""

    SOLUTION_PATTERN = "solution_pattern"           # Reusable solution patterns
    ANTI_PATTERN = "anti_pattern"                    # Common mistakes to avoid
    DECOMPOSITION_STRATEGY = "decomposition_strategy"  # Problem decomposition approaches
    TEAM_PERFORMANCE = "team_performance"            # Team effectiveness metrics
    GAUNTLET_EFFECTIVENESS = "gauntlet_effectiveness"  # Gauntlet validation patterns
    CODE_PATTERN = "code_pattern"                   # Reusable code patterns
    ARCHITECTURE_PATTERN = "architecture_pattern"    # Architecture patterns
    DEBUG_STRATEGY = "debug_strategy"                # Debugging approaches
    OPTIMIZATION = "optimization"                    # Performance optimizations
    DOMAIN_KNOWLEDGE = "domain_knowledge"            # Domain-specific insights
    REFINEMENT_TEMPLATE = "refinement_template"      # Core reasoning path templates


class ArtifactSource(Enum):
    """Where the artifact was extracted from."""

    AGENT_EXECUTION = "agent_execution"             # From agent task execution
    REFACTOR_LEARNING = "reflector_learning"         # From ACE reflector analysis
    SKILL_MANAGER = "skill_manager"                 # From ACE skill manager
    WORKFLOW_PHASE = "workflow_phase"               # From workflow stage
    GAUNTLET_RUN = "gauntlet_run"                   # From gauntlet validation
    TEAM_COLLABORATION = "team_collaboration"       # From team interactions
    MANUAL_ANNOTATION = "manual_annotation"         # Manually curated


class ArtifactStatus(Enum):
    """Artifact lifecycle status."""

    DRAFT = "draft"                                 # Initial extraction
    REVIEWED = "reviewed"                           # Human-reviewed
    APPROVED = "approved"                           # Approved for reuse
    DEPRECATED = "deprecated"                       # No longer recommended
    ARCHIVED = "archived"                           # Historical reference


@dataclass
class ArtifactMetadata:
    """
    Metadata for a knowledge artifact.

    Memory Management:
        - Tags list: Default factory creates new list per instance
        - Dependencies list: Default factory creates new list per instance
        - All string fields are immutable (thread-safe)
        - Datetime fields are immutable (thread-safe)
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType = ArtifactType.SOLUTION_PATTERN
    source: ArtifactSource = ArtifactSource.AGENT_EXECUTION
    status: ArtifactStatus = ArtifactStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"  # Agent ID or "system"
    version: int = 1
    hash: str = ""  # Content hash for deduplication
    tags: List[str] = field(default_factory=list)
    domain: str = ""  # Problem domain (e.g., "backend", "frontend")
    complexity: str = ""  # "low", "medium", "high"
    dependencies: List[str] = field(default_factory=list)  # IDs of related artifacts

    def __post_init__(self):
        """Generate hash after initialization."""
        if not self.hash:
            self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate content hash for deduplication."""
        # SECURITY FIX: Phase 1 - CVE-4 Weak Hashing - Replace MD5 with SHA-256
        # HASH FIX: Hash should only be calculated once in __post_init__ since tags is mutable
        # Converting tags to tuple for hash to ensure immutability
        content_str = f"{self.artifact_type.value}_{self.domain}_{self.version}_{tuple(sorted(self.tags))}"
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:32]


@dataclass
class UsageMetrics:
    """
    Usage metrics for an artifact (thread-safe).

    Memory Management:
        - Lock: Instance-level lock created via default_factory for thread safety
        - Counter fields: Immutable integers, protected by lock for updates
        - Datetime field: Immutable, replaced on update (thread-safe)
        - Float field: Immutable, updated with lock protection

    Serialization Support:
        - __getstate__ and __setstate__ for pickle support (locks are not serializable)
    """

    times_used: int = 0
    times_helpful: int = 0
    times_harmful: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 0.0  # 0.0 to 1.0
    # THREAD SAFETY FIX: TS-3 - Add lock for counter updates
    # LOCK FIX: TS-3 - Use RLock for re-entrancy (same thread may call record_usage multiple times)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def record_usage(self, helpful: bool = True):
        """Record a usage event (thread-safe)."""
        # THREAD SAFETY FIX: TS-3 - Synchronize counter updates
        with self._lock:
            self.times_used += 1
            if helpful:
                self.times_helpful += 1
            else:
                self.times_harmful += 1
            self.last_used = datetime.utcnow()
            if self.times_used > 0:
                self.success_rate = self.times_helpful / self.times_used

    # SERIALIZATION FIX: Add pickle support for locks
    def __getstate__(self):
        """Get state for pickling, excluding the non-serializable lock."""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        """Restore state from pickle, recreating the lock."""
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = threading.RLock()


@dataclass
class KnowledgeArtifact:
    """
    A knowledge artifact extracted from workflow execution.

    This is the core data structure for Stage 6 Knowledge Extraction.

    Memory Management:
        - Metadata: Nested dataclass with proper default factories
        - Lists (examples, counter_examples, related_artifacts): Default factories prevent sharing
        - Metrics: Instance-level UsageMetrics with own lock
        - Strings: Immutable (thread-safe)
    """

    metadata: ArtifactMetadata
    title: str
    description: str
    content: str  # The actual knowledge (in TOON format for tokens)
    context: str = ""  # When to apply this knowledge
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)  # Anti-patterns
    related_artifacts: List[str] = field(default_factory=list)  # IDs
    metrics: UsageMetrics = field(default_factory=UsageMetrics)

    def __post_init__(self):
        """
        Validate list sizes after initialization.

        VALIDATION FIX: Add bounds checking to prevent unbounded list growth
        which could cause memory exhaustion.
        """
        # Validate examples list size
        if len(self.examples) > 100:
            logger.warning(f"examples list too large ({len(self.examples)}), truncating to 100")
            object.__setattr__(self, 'examples', self.examples[:100])

        # Validate counter_examples list size
        if len(self.counter_examples) > 100:
            logger.warning(f"counter_examples list too large ({len(self.counter_examples)}), truncating to 100")
            object.__setattr__(self, 'counter_examples', self.counter_examples[:100])

        # Validate related_artifacts list size
        if len(self.related_artifacts) > 100:
            logger.warning(f"related_artifacts list too large ({len(self.related_artifacts)}), truncating to 100")
            object.__setattr__(self, 'related_artifacts', self.related_artifacts[:100])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": {
                "artifact_id": self.metadata.artifact_id,
                "artifact_type": self.metadata.artifact_type.value,
                "source": self.metadata.source.value,
                "status": self.metadata.status.value,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "created_by": self.metadata.created_by,
                "version": self.metadata.version,
                "hash": self.metadata.hash,
                "tags": self.metadata.tags,
                "domain": self.metadata.domain,
                "complexity": self.metadata.complexity,
                "dependencies": self.metadata.dependencies,
            },
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "context": self.context,
            "examples": self.examples,
            "counter_examples": self.counter_examples,
            "related_artifacts": self.related_artifacts,
            "metrics": {
                "times_used": self.metrics.times_used,
                "times_helpful": self.metrics.times_helpful,
                "times_harmful": self.metrics.times_harmful,
                "last_used": self.metrics.last_used.isoformat() if self.metrics.last_used else None,
                "success_rate": self.metrics.success_rate,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """
        Create from dictionary with comprehensive validation.

        SECURITY FIX: Add dictionary structure validation to prevent
        injection attacks and malformed data.

        Args:
            data: Dictionary containing artifact data

        Returns:
            KnowledgeArtifact instance

        Raises:
            ValueError: If data validation fails
        """
        # Validate metadata structure
        expected_metadata_fields = {
            "artifact_id": str,
            "artifact_type": str,
            "source": str,
            "status": str,
            "created_at": str,
            "updated_at": str,
            "created_by": str,
            "version": int,
            "hash": str,
            "tags": list,
            "domain": str,
            "complexity": str,
            "dependencies": list,
        }

        try:
            metadata_data = validate_dict_structure(
                data.get("metadata", {}),
                expected_metadata_fields,
                require_all=True
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid metadata structure: {e}")

        # Validate metrics structure
        expected_metrics_fields = {
            "times_used": int,
            "times_helpful": int,
            "times_harmful": int,
            "last_used": (str, type(None)),
            "success_rate": float,
        }

        try:
            metrics_data = validate_dict_structure(
                data.get("metrics", {}),
                expected_metrics_fields,
                require_all=True
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid metrics structure: {e}")

        # Safely parse datetime strings
        # DATETIME PARSING FIX: Add comprehensive error handling
        try:
            created_at = datetime.fromisoformat(metadata_data["created_at"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid created_at datetime, using now: {e}")
            created_at = datetime.utcnow()
        except (TypeError, AttributeError) as e:
            logger.warning(f"Invalid created_at type, using now: {e}")
            created_at = datetime.utcnow()

        try:
            updated_at = datetime.fromisoformat(metadata_data["updated_at"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid updated_at datetime, using now: {e}")
            updated_at = datetime.utcnow()
        except (TypeError, AttributeError) as e:
            logger.warning(f"Invalid updated_at type, using now: {e}")
            updated_at = datetime.utcnow()

        # Create metadata with safe defaults
        try:
            metadata = ArtifactMetadata(
                artifact_id=metadata_data["artifact_id"],
                artifact_type=ArtifactType(metadata_data["artifact_type"]),
                source=ArtifactSource(metadata_data["source"]),
                status=ArtifactStatus(metadata_data["status"]),
                created_at=created_at,
                updated_at=updated_at,
                created_by=metadata_data["created_by"],
                version=metadata_data["version"],
                hash=metadata_data["hash"],
                tags=metadata_data.get("tags", []),
                domain=metadata_data.get("domain", ""),
                complexity=metadata_data.get("complexity", ""),
                dependencies=metadata_data.get("dependencies", []),
            )
        except (ValueError, KeyError) as e:
            error_response = create_safe_error(
                "Failed to create artifact metadata",
                e,
                include_details=False
            )
            raise ValueError(error_response["error"])

        # Safely parse last_used datetime
        # DATETIME PARSING FIX: Add comprehensive error handling
        last_used = None
        if metrics_data.get("last_used"):
            try:
                last_used = datetime.fromisoformat(metrics_data["last_used"])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid last_used datetime, using None: {e}")
                last_used = None
            except (AttributeError, KeyError) as e:
                logger.warning(f"Invalid last_used type, using None: {e}")
                last_used = None

        # Create metrics with validation
        try:
            metrics = UsageMetrics(
                times_used=metrics_data.get("times_used", 0),
                times_helpful=metrics_data.get("times_helpful", 0),
                times_harmful=metrics_data.get("times_harmful", 0),
                last_used=last_used,
                success_rate=metrics_data.get("success_rate", 0.0),
            )
        except (ValueError, TypeError) as e:
            error_response = create_safe_error(
                "Failed to create usage metrics",
                e,
                include_details=False
            )
            raise ValueError(error_response["error"])

        # Create and return artifact
        # DEEP COPY FIX: Deep copy lists to prevent external modification
        return cls(
            metadata=metadata,
            title=copy.deepcopy(data.get("title", "")),
            description=copy.deepcopy(data.get("description", "")),
            content=copy.deepcopy(data.get("content", "")),
            context=copy.deepcopy(data.get("context", "")),
            examples=copy.deepcopy(data.get("examples", [])),  # Deep copy list
            counter_examples=copy.deepcopy(data.get("counter_examples", [])),  # Deep copy list
            related_artifacts=copy.deepcopy(data.get("related_artifacts", [])),  # Deep copy list
            metrics=metrics,
        )

    def save_to_file(self, filepath: str):
        """Save artifact to JSON file."""
        # SECURITY FIX: Phase 1 - CVE-1 Path Traversal - Validate filepath
        try:
            filepath = validate_file_path_safe(filepath, base_dir=".")
            atomic_save_json_file(filepath, self.to_dict())
        except (ValueError, IOError) as e:
            logger.error(f"Failed to save artifact: {e}")
            raise

    @classmethod
    def load_from_file(cls, filepath: str) -> "KnowledgeArtifact":
        """Load artifact from JSON file."""
        # SECURITY FIX: Phase 1 - CVE-1 Path Traversal - Validate filepath
        try:
            filepath = validate_file_path_safe(filepath, base_dir=".")
            data = safe_load_json_file(filepath)
            return cls.from_dict(data)
        except (ValueError, IOError) as e:
            logger.error(f"Failed to load artifact: {e}")
            raise


@dataclass
class SolutionPattern(KnowledgeArtifact):
    """
    A reusable solution pattern artifact.

    Memory Management:
        - Inherits from KnowledgeArtifact (see parent class docs)
        - Additional fields: Immutable strings (thread-safe)
    """

    problem_category: str = ""  # e.g., "authentication", "database"
    pattern_category: str = ""  # e.g., "creational", "structural", "behavioral"
    implementation_complexity: str = "medium"  # "low", "medium", "high"
    performance_impact: str = ""  # "positive", "neutral", "negative"

    def __post_init__(self):
        """Ensure correct artifact type."""
        # ASSIGNMENT FIX: Check before setting to avoid overwriting
        if self.metadata.artifact_type != ArtifactType.SOLUTION_PATTERN:
            self.metadata.artifact_type = ArtifactType.SOLUTION_PATTERN


@dataclass
class AntiPattern(KnowledgeArtifact):
    """
    A common mistake to avoid (anti-pattern).

    Memory Management:
        - Inherits from KnowledgeArtifact (see parent class docs)
        - Additional fields: Immutable strings (thread-safe)
    """

    problem_category: str = ""
    severity: str = "medium"  # "low", "medium", "high", "critical"
    common_mistake: str = ""
    correct_approach: str = ""

    def __post_init__(self):
        """Ensure correct artifact type."""
        self.metadata.artifact_type = ArtifactType.ANTI_PATTERN


@dataclass
class DecompositionStrategy(KnowledgeArtifact):
    """
    A problem decomposition strategy artifact.

    Memory Management:
        - Inherits from KnowledgeArtifact (see parent class docs)
        - Additional fields: Immutable primitives (thread-safe)
    """

    decomposition_depth: int = 1  # How deep to decompose
    granularity: str = "medium"  # "coarse", "medium", "fine"
    dependency_handling: str = ""  # How to handle dependencies

    def __post_init__(self):
        """Ensure correct artifact type."""
        self.metadata.artifact_type = ArtifactType.DECOMPOSITION_STRATEGY


@dataclass
class TeamPerformanceData:
    """
    Team performance metrics for knowledge extraction.

    Memory Management:
        - Lists and dicts: Default factories prevent sharing between instances
        - Datetime: Immutable field, replaced on update
        - Numeric fields: Immutable primitives
        - Validation: __post_init__ ensures data integrity

    Thread Safety:
        - This class is NOT thread-safe for mutations
        - External synchronization required for concurrent access
    """

    team_id: str
    team_name: str
    team_type: str  # "blue_team", "red_team", "gold_team"
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    avg_quality_score: float = 0.0
    preferred_problem_types: List[str] = field(default_factory=list)
    skill_affinities: Dict[str, float] = field(default_factory=dict)  # skill -> affinity score
    collaboration_effectiveness: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """
        Validate numeric ranges after initialization.

        VALIDATION FIX: EC-5 - Add parameter validation to prevent
        invalid values and edge cases.
        """
        # UNINITIALIZED VARIABLE FIX: Check for None before validation
        if self.total_tasks is None:
            self.total_tasks = 0

        # Validate numeric ranges
        validate_numeric_range(
            self.total_tasks,
            "total_tasks",
            min_val=0,
            max_val=1000000,
            value_type=int
        )

        validate_numeric_range(
            self.successful_tasks,
            "successful_tasks",
            min_val=0,
            max_val=self.total_tasks,  # Cannot exceed total
            value_type=int
        )

        validate_numeric_range(
            self.failed_tasks,
            "failed_tasks",
            min_val=0,
            max_val=self.total_tasks,  # Cannot exceed total
            value_type=int
        )

        validate_numeric_range(
            self.avg_execution_time,
            "avg_execution_time",
            min_val=0.0,
            max_val=86400.0,  # Max 24 hours in seconds
            value_type=float
        )

        validate_numeric_range(
            self.avg_quality_score,
            "avg_quality_score",
            min_val=0.0,
            max_val=1.0,  # Score is 0.0 to 1.0
            value_type=float
        )

        validate_numeric_range(
            self.collaboration_effectiveness,
            "collaboration_effectiveness",
            min_val=0.0,
            max_val=1.0,  # Score is 0.0 to 1.0
            value_type=float
        )

        # Validate lists and dicts
        validate_list_size(
            self.preferred_problem_types,
            "preferred_problem_types",
            max_size=1000
        )

        validate_list_size(
            list(self.skill_affinities.keys()),
            "skill_affinities",
            max_size=1000
        )

    def calculate_success_rate(self) -> float:
        """
        Calculate team success rate.

        VALIDATION FIX: EC-5 - Prevent division by zero
        """
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "team_type": self.team_type,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "avg_execution_time": self.avg_execution_time,
            "avg_quality_score": self.avg_quality_score,
            "preferred_problem_types": self.preferred_problem_types,
            "skill_affinities": self.skill_affinities,
            "collaboration_effectiveness": self.collaboration_effectiveness,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class GauntletEffectivenessData:
    """
    Gauntlet effectiveness metrics for knowledge extraction.

    Memory Management:
        - Lists and dicts: Default factories prevent sharing between instances
        - Datetime: Immutable field, replaced on update
        - Numeric fields: Immutable primitives
        - Validation: __post_init__ ensures data integrity

    Thread Safety:
        - This class is NOT thread-safe for mutations
        - External synchronization required for concurrent access
    """

    gauntlet_id: str
    gauntlet_name: str
    gauntlet_type: str  # "red_team", "gold_team"
    total_runs: int = 0
    issues_found: int = 0
    false_positives: int = 0
    true_positives: int = 0
    detection_rate: float = 0.0  # 0.0 to 1.0
    avg_execution_time: float = 0.0
    effective_problem_types: List[str] = field(default_factory=list)
    common_violations: Dict[str, int] = field(default_factory=dict)  # violation -> count
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """
        Validate numeric ranges after initialization.

        VALIDATION FIX: EC-5 - Add parameter validation to prevent
        invalid values and edge cases.
        """
        # Validate numeric ranges
        validate_numeric_range(
            self.total_runs,
            "total_runs",
            min_val=0,
            max_val=10000000,
            value_type=int
        )

        validate_numeric_range(
            self.issues_found,
            "issues_found",
            min_val=0,
            max_val=self.total_runs,  # Cannot exceed total
            value_type=int
        )

        validate_numeric_range(
            self.false_positives,
            "false_positives",
            min_val=0,
            max_val=self.total_runs,  # Cannot exceed total
            value_type=int
        )

        validate_numeric_range(
            self.true_positives,
            "true_positives",
            min_val=0,
            max_val=self.total_runs,  # Cannot exceed total
            value_type=int
        )

        validate_numeric_range(
            self.detection_rate,
            "detection_rate",
            min_val=0.0,
            max_val=1.0,
            value_type=float
        )

        validate_numeric_range(
            self.avg_execution_time,
            "avg_execution_time",
            min_val=0.0,
            max_val=86400.0,  # Max 24 hours
            value_type=float
        )

        # Validate lists and dicts
        validate_list_size(
            self.effective_problem_types,
            "effective_problem_types",
            max_size=1000
        )

        validate_list_size(
            list(self.common_violations.keys()),
            "common_violations",
            max_size=1000
        )

    def calculate_detection_rate(self) -> float:
        """
        Calculate gauntlet detection rate.

        VALIDATION FIX: EC-5 - Prevent division by zero
        """
        if self.total_runs == 0:
            return 0.0
        return self.issues_found / self.total_runs

    def calculate_precision(self) -> float:
        """
        Calculate gauntlet precision (true positives / all positives).

        VALIDATION FIX: EC-5 - Prevent division by zero
        """
        total_positives = self.true_positives + self.false_positives
        if total_positives == 0:
            return 0.0
        return self.true_positives / total_positives

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gauntlet_id": self.gauntlet_id,
            "gauntlet_name": self.gauntlet_name,
            "gauntlet_type": self.gauntlet_type,
            "total_runs": self.total_runs,
            "issues_found": self.issues_found,
            "false_positives": self.false_positives,
            "true_positives": self.true_positives,
            "detection_rate": self.detection_rate,
            "avg_execution_time": self.avg_execution_time,
            "effective_problem_types": self.effective_problem_types,
            "common_violations": self.common_violations,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class WorkflowExtractionResult:
    """
    Result of extracting knowledge from a workflow execution (thread-safe).

    Memory Management:
        - Lists: Default factories prevent sharing between instances
        - Lock: Instance-level lock for thread-safe mutations
        - Context manager support for proper resource cleanup

    Thread Safety:
        - All mutations are protected by instance-level lock
        - Safe for concurrent reads and writes
        - Supports context manager protocol for cleanup
    """

    workflow_id: str
    problem_statement: str
    extracted_artifacts: List[KnowledgeArtifact] = field(default_factory=list)
    team_performances: List[TeamPerformanceData] = field(default_factory=list)
    gauntlet_effectiveness: List[GauntletEffectivenessData] = field(default_factory=list)
    total_artifacts: int = 0
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    # THREAD SAFETY FIX: TS-11 - Add lock for artifact list operations
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        """
        Initialize the extraction result.

        NOTE: The lock is already created by field(default_factory=threading.Lock)
        This ensures each instance gets its own lock for thread safety.
        """
        # Validate basic parameters
        validate_string_length(
            self.workflow_id,
            "workflow_id",
            max_length=500
        )

        validate_string_length(
            self.problem_statement,
            "problem_statement",
            max_length=10000
        )

        # Validate lists (prevent excessive size)
        validate_list_size(
            self.extracted_artifacts,
            "extracted_artifacts",
            max_size=10000
        )

        validate_list_size(
            self.team_performances,
            "team_performances",
            max_size=1000
        )

        validate_list_size(
            self.gauntlet_effectiveness,
            "gauntlet_effectiveness",
            max_size=1000
        )

    def add_artifact(self, artifact: KnowledgeArtifact):
        """
        Add an artifact to the extraction result (thread-safe).

        THREAD SAFETY FIX: TS-11 - Synchronize artifact list operations
        """
        # THREAD SAFETY FIX: TS-11 - Synchronize artifact list operations
        with self._lock:
            self.extracted_artifacts.append(artifact)
            self.total_artifacts += 1

    def cleanup(self):
        """
        Clean up resources.

        RESOURCE MANAGEMENT FIX: Ensure proper cleanup of resources
        to prevent memory leaks in long-running applications.

        This method:
        - Clears large lists to free memory
        - Is safe to call multiple times
        - Is automatically called by context manager
        """
        with self._lock:
            # Clear lists to free memory
            self.extracted_artifacts.clear()
            self.team_performances.clear()
            self.gauntlet_effectiveness.clear()
            self.total_artifacts = 0

    # SERIALIZATION FIX: Add pickle support for locks
    def __getstate__(self):
        """Get state for pickling, excluding the non-serializable lock."""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        """Restore state from pickle, recreating the lock."""
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = threading.Lock()

    def __del__(self):
        """
        Destructor - cleanup when object is garbage collected.

        RESOURCE MANAGEMENT FIX: Best-effort cleanup on deletion.
        Note: __del__ is not guaranteed to be called in all circumstances,
        so prefer using the context manager protocol.
        """
        try:
            # Best-effort cleanup without acquiring lock
            # (to avoid deadlocks during garbage collection)
            if hasattr(self, 'extracted_artifacts'):
                self.extracted_artifacts.clear()
            if hasattr(self, 'team_performances'):
                self.team_performances.clear()
            if hasattr(self, 'gauntlet_effectiveness'):
                self.gauntlet_effectiveness.clear()
        except (AttributeError, TypeError, RuntimeError):
            # Silently ignore errors during cleanup
            pass

    def __enter__(self):
        """
        Context manager entry.

        Returns:
            Self for use in with statements
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with automatic cleanup.

        RESOURCE MANAGEMENT FIX: Ensure cleanup even if exception occurs.
        """
        self.cleanup()
        return False  # Don't suppress exceptions

    def to_summary(self) -> Dict[str, Any]:
        """Generate summary of extraction results."""
        return {
            "workflow_id": self.workflow_id,
            "problem_statement": self.problem_statement,
            "total_artifacts": self.total_artifacts,
            "artifact_types": {
                artifact_type.value: len([a for a in self.extracted_artifacts if a.metadata.artifact_type == artifact_type])
                for artifact_type in ArtifactType
            },
            "teams_analyzed": len(self.team_performances),
            "gauntlets_analyzed": len(self.gauntlet_effectiveness),
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
        }


# Factory functions for common artifacts

def create_solution_pattern(
    title: str,
    description: str,
    content: str,
    problem_category: str,
    domain: str = "backend",
    complexity: str = "medium",
    tags: List[str] = None,
) -> SolutionPattern:
    """Create a solution pattern artifact."""
    metadata = ArtifactMetadata(
        artifact_type=ArtifactType.SOLUTION_PATTERN,
        source=ArtifactSource.AGENT_EXECUTION,
        domain=domain,
        complexity=complexity,
        tags=tags or [],
    )
    return SolutionPattern(
        metadata=metadata,
        title=title,
        description=description,
        content=content,
        problem_category=problem_category,
    )


def create_anti_pattern(
    title: str,
    description: str,
    common_mistake: str,
    correct_approach: str,
    severity: str = "medium",
    domain: str = "backend",
) -> AntiPattern:
    """Create an anti-pattern artifact."""
    metadata = ArtifactMetadata(
        artifact_type=ArtifactType.ANTI_PATTERN,
        source=ArtifactSource.REFACTOR_LEARNING,
        domain=domain,
    )
    return AntiPattern(
        metadata=metadata,
        title=title,
        description=description,
        content=f"MISTAKE: {common_mistake}\nCORRECT: {correct_approach}",
        common_mistake=common_mistake,
        correct_approach=correct_approach,
        severity=severity,
    )


def create_decomposition_strategy(
    title: str,
    description: str,
    strategy: str,
    decomposition_depth: int = 2,
    granularity: str = "medium",
) -> DecompositionStrategy:
    """Create a decomposition strategy artifact."""
    metadata = ArtifactMetadata(
        artifact_type=ArtifactType.DECOMPOSITION_STRATEGY,
        source=ArtifactSource.WORKFLOW_PHASE,
    )
    return DecompositionStrategy(
        metadata=metadata,
        title=title,
        description=description,
        content=strategy,
        decomposition_depth=decomposition_depth,
        granularity=granularity,
    )


def create_refinement_template(
    title: str,
    description: str,
    reasoning_path: List[str],
    context_signature: Dict[str, Any],
    domain: str = "general",
    tags: Optional[List[str]] = None,
) -> KnowledgeArtifact:
    """Create a refinement template artifact for the ACE Skillbook."""
    metadata = ArtifactMetadata(
        artifact_type=ArtifactType.REFINEMENT_TEMPLATE,
        source=ArtifactSource.REFACTOR_LEARNING,
        domain=domain,
        tags=tags or [],
    )
    content = json.dumps(
        {
            "reasoning_path": reasoning_path,
            "context_signature": context_signature,
        },
        indent=2
    )
    return KnowledgeArtifact(
        metadata=metadata,
        title=title,
        description=description,
        content=content,
        context="Reusable refinement template extracted from a converged workflow."
    )


class SkillbookStore:
    """Persistent store for refinement templates (Skillbook 2.0)."""

    def __init__(self, storage_path: str = "./ace_skillbook.json"):
        self.storage_path = storage_path
        self.templates: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            self.templates = []
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.templates = json.load(f)
        except (OSError, IOError, json.JSONDecodeError):
            self.templates = []

    def _save(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2)
        except (OSError, IOError) as e:
            logger.warning("Failed to persist Skillbook: %s", sanitize_for_logging(str(e)))

    def add_template(self, template: KnowledgeArtifact) -> None:
        entry = {
            "metadata": template.metadata.__dict__,
            "title": template.title,
            "description": template.description,
            "content": template.content,
            "context": template.context,
            "created_at": template.metadata.created_at.isoformat(),
        }
        self.templates.append(entry)
        self._save()

    def find_templates(self, context_signature: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve templates matching the context signature."""
        if not context_signature:
            return self.templates[-limit:]

        def score(entry: Dict[str, Any]) -> int:
            content = entry.get("content", "")
            hits = 0
            for k, v in context_signature.items():
                if k in content or str(v) in content:
                    hits += 1
            return hits

        ranked = sorted(self.templates, key=score, reverse=True)
        return ranked[:limit]


# Export all classes
__all__ = [
    # Enums
    "ArtifactType",
    "ArtifactSource",
    "ArtifactStatus",
    # Core Classes
    "ArtifactMetadata",
    "UsageMetrics",
    "KnowledgeArtifact",
    # Specialized Artifacts
    "SolutionPattern",
    "AntiPattern",
    "DecompositionStrategy",
    # Performance Data
    "TeamPerformanceData",
    "GauntletEffectivenessData",
    # Results
    "WorkflowExtractionResult",
    # Factory Functions
    "create_solution_pattern",
    "create_anti_pattern",
    "create_decomposition_strategy",
    "create_refinement_template",
    "SkillbookStore",
]

class ACEKnowledgeManager:
    """Stub class for ACEKnowledgeManager."""
    pass

class KnowledgeArtifactManager:
    """Stub class for KnowledgeArtifactManager."""
    pass
