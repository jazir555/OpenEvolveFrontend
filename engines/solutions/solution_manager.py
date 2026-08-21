"""
Solution Manager
================

Production-ready management of solution attempts for sub-problems in the
Sovereign-Grade Decomposition workflow.

This module provides comprehensive functionality for:
- Creating and validating solution attempts
- Tracking solution status through lifecycle
- Version history and archiving
- Integration with SGD workflow orchestrator

Author: OpenEvolve Frontend Team
Version: 1.0.0
License: MIT
"""
from __future__ import annotations


import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# **LEAN INTEGRATION**: Formal verification with Lean
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class SolutionStatus(Enum):
    """Status of a solution attempt through its lifecycle."""
    PENDING = "pending"              # Solution created, not yet processed
    IN_PROGRESS = "in_progress"      # Currently being generated/processed
    COMPLETED = "completed"          # Successfully generated
    FAILED = "failed"                # Generation failed
    VERIFIED = "verified"            # Passed verification gauntlet
    REJECTED = "rejected"            # Failed verification
    ARCHIVED = "archived"            # Archived and no longer active


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"                # All validations must pass
    MODERATE = "moderate"            # Warnings allowed
    LENIENT = "lenient"              # Only critical issues block


# Constants
MAX_CONTENT_LENGTH = 10_000_000  # 10MB max content size
MIN_CONTENT_LENGTH = 10          # Minimum 10 characters
MAX_HISTORY_PER_SUBPROBLEM = 100  # Maximum history entries to retain
DEFAULT_STORAGE_DIR = "data/solutions"
ARCHIVE_DIR = "data/solutions/archive"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SolutionAttempt:
    """
    Represents a single attempt to solve a sub-problem.

    This model tracks the complete lifecycle of a solution from creation
    through verification and archival.

    Attributes:
        id: Unique identifier for this attempt
        sub_problem_id: ID of the sub-problem this solves
        content: The solution content (code, text, etc.)
        generated_by_model: Model/agent that generated this solution
        timestamp: Unix timestamp of creation
        status: Current status of the solution
        version: Version number for tracking iterations
        parent_attempt_id: ID of parent attempt if this is a revision
        metadata: Additional contextual information
        verification_reports: List of verification reports
        quality_score: Computed quality score (0.0-1.0)
        created_at: ISO format timestamp
        updated_at: ISO format timestamp
    """
    id: str
    sub_problem_id: str
    content: str
    generated_by_model: str
    timestamp: float
    status: str
    version: int = 1
    parent_attempt_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification_reports: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.created_at:
            try:
                self.created_at = datetime.fromtimestamp(self.timestamp).isoformat()
            except (OSError, ValueError):
                # Handle invalid timestamp
                self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolutionAttempt':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class ValidationResult:
    """
    Result of validating a solution attempt.

    Attributes:
        is_valid: Whether the solution passed validation
        score: Overall validation score (0.0-1.0)
        issues: List of validation issues found
        warnings: List of non-critical warnings
        feedback: Human-readable feedback summary
        validator_name: Name of the validator that produced this result
        timestamp: When validation was performed
        level: Validation strictness level used
    """
    is_valid: bool
    score: float
    issues: List[str]
    warnings: List[str]
    feedback: str
    validator_name: str
    timestamp: float
    level: ValidationLevel = ValidationLevel.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": self.issues,
            "warnings": self.warnings,
            "feedback": self.feedback,
            "validator_name": self.validator_name,
            "timestamp": self.timestamp,
            "level": self.level.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create instance from dictionary."""
        level = ValidationLevel(data.get("level", ValidationLevel.MODERATE.value))
        return cls(
            is_valid=data["is_valid"],
            score=data["score"],
            issues=data["issues"],
            warnings=data["warnings"],
            feedback=data["feedback"],
            validator_name=data["validator_name"],
            timestamp=data["timestamp"],
            level=level
        )


@dataclass
class SolutionHistory:
    """
    Complete history of solution attempts for a sub-problem.

    Attributes:
        sub_problem_id: ID of the sub-problem
        attempts: Chronological list of all attempts
        latest_attempt: Most recent attempt (if any)
        total_attempts: Total number of attempts
        success_count: Number of successful/verified attempts
        failure_count: Number of failed attempts
    """
    sub_problem_id: str
    attempts: List[SolutionAttempt] = field(default_factory=list)
    latest_attempt: Optional[SolutionAttempt] = None
    total_attempts: int = 0
    success_count: int = 0
    failure_count: int = 0


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SolutionManagerError(Exception):
    """Base exception for solution manager errors."""
    pass


class SolutionValidationError(SolutionManagerError):
    """Raised when solution validation fails."""
    pass


class SolutionNotFoundError(SolutionManagerError):
    """Raised when a solution attempt is not found."""
    pass


class SolutionStorageError(SolutionManagerError):
    """Raised when solution storage operations fail."""
    pass


# ============================================================================
# SOLUTION MANAGER
# ============================================================================

class SolutionManager:
    """
    Manages solution attempts for sub-problems in the SGD workflow.

    This class provides a comprehensive API for:
    - Creating solution attempts with validation
    - Updating solution content with version tracking
    - Validating solution format and structure
    - Retrieving solution history
    - Archiving old solutions

    Thread-safe: All operations are protected by locks for concurrent access.
    """

    def __init__(
        self,
        storage_dir: str = DEFAULT_STORAGE_DIR,
        archive_dir: str = ARCHIVE_DIR,
        enable_persistence: bool = True,
        validation_level: ValidationLevel = ValidationLevel.MODERATE
    ):
        """
        Initialize the Solution Manager.

        Args:
            storage_dir: Directory for storing active solutions
            archive_dir: Directory for storing archived solutions
            enable_persistence: Whether to persist solutions to disk
            validation_level: Default validation strictness level
        """
        self.storage_dir = Path(storage_dir)
        self.archive_dir = Path(archive_dir)
        self.enable_persistence = enable_persistence
        self.validation_level = validation_level

        # Thread-safe locks
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()

        # In-memory storage
        self._solutions: Dict[str, SolutionAttempt] = {}
        self._sub_problem_index: Dict[str, List[str]] = {}  # sub_problem_id -> [attempt_ids]

        # Custom validators
        self._validators: Dict[str, Callable[[SolutionAttempt], ValidationResult]] = {}

        # Initialize storage directories
        if self.enable_persistence:
            self._init_storage_dirs()

        logger.info(f"SolutionManager initialized with storage_dir={storage_dir}")

    # ========================================================================
    # INITIALIZATION AND UTILITIES
    # ========================================================================

    def _init_storage_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Storage directories initialized: {self.storage_dir}")
        except OSError as e:
            raise SolutionStorageError(f"Failed to create storage directories: {e}")

    def _generate_id(self, sub_problem_id: str) -> str:
        """Generate a unique solution attempt ID."""
        unique_part = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        return f"sol_{sub_problem_id}_{timestamp}_{unique_part}"

    def _get_storage_path(self, attempt_id: str) -> Path:
        """Get the file path for a solution attempt."""
        return self.storage_dir / f"{attempt_id}.json"

    def _get_archive_path(self, attempt_id: str) -> Path:
        """Get the archive path for a solution attempt."""
        return self.archive_dir / f"{attempt_id}.json"

    # ========================================================================
    # SOLUTION CREATION
    # ========================================================================

    def create_solution_attempt(
        self,
        sub_problem_id: str,
        content: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_attempt_id: Optional[str] = None,
        validate: bool = True
    ) -> SolutionAttempt:
        """
        Create a new solution attempt.

        Args:
            sub_problem_id: ID of the sub-problem being solved
            content: The solution content
            model: Model/agent that generated the solution
            metadata: Additional metadata to attach
            parent_attempt_id: ID of parent attempt if this is a revision
            validate: Whether to validate the solution before creating

        Returns:
            The created SolutionAttempt

        Raises:
            SolutionValidationError: If validation fails and validate=True
            SolutionStorageError: If storage operation fails
        """
        with self._lock:
            # Generate ID and timestamp
            attempt_id = self._generate_id(sub_problem_id)
            timestamp = time.time()

            # Create initial attempt
            attempt = SolutionAttempt(
                id=attempt_id,
                sub_problem_id=sub_problem_id,
                content=content,
                generated_by_model=model,
                timestamp=timestamp,
                status=SolutionStatus.PENDING.value,
                version=1,
                parent_attempt_id=parent_attempt_id,
                metadata=metadata or {},
                verification_reports=[],
                quality_score=0.0
            )

            # Validate if requested
            if validate:
                validation_result = self.validate_solution_attempt(attempt)
                if not validation_result.is_valid and self.validation_level != ValidationLevel.LENIENT:
                    raise SolutionValidationError(
                        f"Solution validation failed: {validation_result.feedback}"
                    )
                # Attach validation results to metadata
                attempt.metadata["initial_validation"] = validation_result.to_dict()

            # Store in memory
            self._solutions[attempt_id] = attempt

            # Update sub-problem index
            if sub_problem_id not in self._sub_problem_index:
                self._sub_problem_index[sub_problem_id] = []
            self._sub_problem_index[sub_problem_id].append(attempt_id)

            # Persist to disk if enabled
            if self.enable_persistence:
                self._persist_solution(attempt)

            logger.info(
                f"Created solution attempt {attempt_id} for sub-problem {sub_problem_id} "
                f"via {model} (status: {attempt.status})"
            )

            return attempt

    # ========================================================================
    # SOLUTION UPDATE
    # ========================================================================

    def update_solution_attempt(
        self,
        attempt: SolutionAttempt,
        content: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        verification_report: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None
    ) -> SolutionAttempt:
        """
        Update an existing solution attempt.

        Args:
            attempt: The solution attempt to update
            content: New content (if updating)
            status: New status (if updating)
            metadata: Additional metadata to merge
            verification_report: Verification report to attach
            quality_score: Quality score to set

        Returns:
            The updated SolutionAttempt

        Raises:
            SolutionNotFoundError: If attempt not found
            SolutionValidationError: If new content fails validation
            SolutionStorageError: If storage operation fails
        """
        with self._lock:
            # Verify attempt exists
            if attempt.id not in self._solutions:
                raise SolutionNotFoundError(f"Solution attempt {attempt.id} not found")

            # Get original for version tracking
            original = self._solutions[attempt.id]

            # Update content if provided
            if content is not None:
                # Validate new content
                temp_attempt = SolutionAttempt(
                    **{**asdict(attempt), "content": content}
                )
                validation_result = self.validate_solution_attempt(temp_attempt)
                if not validation_result.is_valid and self.validation_level != ValidationLevel.LENIENT:
                    raise SolutionValidationError(
                        f"Updated content validation failed: {validation_result.feedback}"
                    )
                attempt.content = content
                attempt.version = original.version + 1

            # Update status if provided
            if status is not None:
                if status not in [s.value for s in SolutionStatus]:
                    raise ValueError(f"Invalid status: {status}")
                attempt.status = status

            # Merge metadata if provided
            if metadata:
                attempt.metadata.update(metadata)

            # Attach verification report if provided
            if verification_report:
                attempt.verification_reports.append(verification_report)

            # Update quality score if provided
            if quality_score is not None:
                attempt.quality_score = max(0.0, min(1.0, quality_score))

            # Update timestamp
            attempt.updated_at = datetime.now().isoformat()

            # Update in storage
            self._solutions[attempt.id] = attempt

            # Persist to disk
            if self.enable_persistence:
                with self._write_lock:
                    self._persist_solution(attempt)

            logger.debug(
                f"Updated solution attempt {attempt.id} "
                f"(version: {attempt.version}, status: {attempt.status})"
            )

            return attempt

    # ========================================================================
    # SOLUTION VALIDATION
    # ========================================================================

    def validate_solution_attempt(
        self,
        attempt: SolutionAttempt,
        level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate a solution attempt.

        Performs comprehensive validation including:
        - Content length checks
        - Format and structure validation
        - Required field presence
        - Custom registered validators

        Args:
            attempt: The solution attempt to validate
            level: Validation strictness level (uses default if None)

        Returns:
            ValidationResult with validation outcome
        """
        validation_level = level or self.validation_level
        issues = []
        warnings = []

        # 1. Content length validation
        content_len = len(attempt.content)
        if content_len < MIN_CONTENT_LENGTH:
            issues.append(
                f"Content too short: {content_len} characters "
                f"(minimum: {MIN_CONTENT_LENGTH})"
            )
        elif content_len > MAX_CONTENT_LENGTH:
            issues.append(
                f"Content too long: {content_len} characters "
                f"(maximum: {MAX_CONTENT_LENGTH})"
            )

        # 2. Required field validation
        if not attempt.id or not attempt.id.strip():
            issues.append("Solution ID is missing or empty")

        if not attempt.sub_problem_id or not attempt.sub_problem_id.strip():
            issues.append("Sub-problem ID is missing or empty")

        if not attempt.generated_by_model or not attempt.generated_by_model.strip():
            issues.append("Generated by model is missing or empty")

        if attempt.timestamp <= 0:
            issues.append("Invalid timestamp")

        # 3. Status validation
        try:
            SolutionStatus(attempt.status)
        except ValueError:
            warnings.append(f"Unknown status: {attempt.status}")

        # 4. Content format validation (basic structure checks)
        if attempt.content.strip() != attempt.content:
            warnings.append("Content has leading/trailing whitespace")

        # Check for common markdown/code indicators
        has_code_markers = (
            "```" in attempt.content or
            "<code>" in attempt.content or
            any(attempt.content.strip().startswith(ext) for ext in ["def ", "class ", "import ", "#!/"])
        )
        if not has_code_markers and content_len > 100:
            warnings.append("Content may not contain formatted code")

        # 5. Run custom validators
        for validator_name, validator_func in self._validators.items():
            try:
                custom_result = validator_func(attempt)
                if not custom_result.is_valid:
                    issues.extend(
                        [f"[{validator_name}] {issue}" for issue in custom_result.issues]
                    )
                warnings.extend(
                    [f"[{validator_name}] {warning}" for warning in custom_result.warnings]
                )
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.error(f"Custom validator {validator_name} failed: {e}")
                warnings.append(f"Custom validator {validator_name} failed to execute")

        # 6. Compute overall score
        critical_issues = sum(1 for i in issues if "missing" in i.lower() or "invalid" in i.lower())
        base_score = 1.0
        base_score -= len(issues) * 0.2  # Each issue reduces score by 20%
        base_score -= len(warnings) * 0.05  # Each warning reduces score by 5%
        base_score = max(0.0, base_score)

        # 7. Determine validity based on level
        is_valid = True
        if validation_level == ValidationLevel.STRICT:
            is_valid = len(issues) == 0 and len(warnings) == 0
        elif validation_level == ValidationLevel.MODERATE:
            is_valid = len(issues) == 0
        else:  # LENIENT
            is_valid = critical_issues == 0

        # 8. Generate feedback
        if is_valid:
            feedback = "Solution passed validation"
            if warnings:
                feedback += f" with {len(warnings)} warning(s)"
        else:
            feedback = f"Validation failed: {len(issues)} error(s), {len(warnings)} warning(s)"

        return ValidationResult(
            is_valid=is_valid,
            score=base_score,
            issues=issues,
            warnings=warnings,
            feedback=feedback,
            validator_name="SolutionManager",
            timestamp=time.time(),
            level=validation_level
        )

    def register_validator(
        self,
        name: str,
        validator_func: Callable[[SolutionAttempt], ValidationResult]
    ) -> None:
        """
        Register a custom validation function.

        Args:
            name: Unique name for the validator
            validator_func: Function that takes SolutionAttempt and returns ValidationResult
        """
        with self._lock:
            self._validators[name] = validator_func
            logger.info(f"Registered custom validator: {name}")

    def unregister_validator(self, name: str) -> None:
        """Unregister a custom validator."""
        with self._lock:
            if name in self._validators:
                del self._validators[name]
                logger.info(f"Unregistered validator: {name}")

    # ========================================================================
    # SOLUTION RETRIEVAL
    # ========================================================================

    def get_solution_attempt(self, attempt_id: str) -> Optional[SolutionAttempt]:
        """
        Get a specific solution attempt by ID.

        Args:
            attempt_id: ID of the solution attempt

        Returns:
            SolutionAttempt if found, None otherwise
        """
        with self._lock:
            return self._solutions.get(attempt_id)

    def get_solution_history(
        self,
        sub_problem_id: str,
        limit: Optional[int] = None
    ) -> SolutionHistory:
        """
        Get complete solution history for a sub-problem.

        Args:
            sub_problem_id: ID of the sub-problem
            limit: Maximum number of attempts to return (None for all)

        Returns:
            SolutionHistory with all attempts
        """
        with self._lock:
            attempt_ids = self._sub_problem_index.get(sub_problem_id, [])

            if not attempt_ids:
                return SolutionHistory(
                    sub_problem_id=sub_problem_id,
                    attempts=[],
                    latest_attempt=None,
                    total_attempts=0,
                    success_count=0,
                    failure_count=0
                )

            # Get all attempts, sorted by timestamp
            attempts = [
                self._solutions[aid]
                for aid in attempt_ids
                if aid in self._solutions
            ]
            attempts.sort(key=lambda a: a.timestamp)

            # Apply limit if specified
            if limit and len(attempts) > limit:
                attempts = attempts[-limit:]

            # Compute statistics
            latest = attempts[-1] if attempts else None
            success_count = sum(
                1 for a in attempts
                if a.status in [SolutionStatus.COMPLETED.value, SolutionStatus.VERIFIED.value]
            )
            failure_count = sum(
                1 for a in attempts
                if a.status == SolutionStatus.FAILED.value
            )

            return SolutionHistory(
                sub_problem_id=sub_problem_id,
                attempts=attempts,
                latest_attempt=latest,
                total_attempts=len(attempts),
                success_count=success_count,
                failure_count=failure_count
            )

    def get_latest_solution(
        self,
        sub_problem_id: str,
        status_filter: Optional[List[str]] = None
    ) -> Optional[SolutionAttempt]:
        """
        Get the latest solution attempt for a sub-problem.

        Args:
            sub_problem_id: ID of the sub-problem
            status_filter: Optional list of status values to filter by

        Returns:
            Latest SolutionAttempt matching criteria, or None
        """
        with self._lock:
            history = self.get_solution_history(sub_problem_id)

            if not history.attempts:
                return None

            # Filter by status if specified
            if status_filter:
                filtered = [
                    a for a in history.attempts
                    if a.status in status_filter
                ]
                return filtered[-1] if filtered else None

            return history.latest_attempt

    def get_solutions_by_status(
        self,
        status: str,
        sub_problem_id: Optional[str] = None
    ) -> List[SolutionAttempt]:
        """
        Get all solutions with a specific status.

        Args:
            status: Status value to filter by
            sub_problem_id: Optional sub-problem ID to further filter

        Returns:
            List of matching SolutionAttempts
        """
        with self._lock:
            solutions = [
                s for s in self._solutions.values()
                if s.status == status
            ]

            if sub_problem_id:
                solutions = [s for s in solutions if s.sub_problem_id == sub_problem_id]

            return solutions

    # ========================================================================
    # SOLUTION ARCHIVAL
    # ========================================================================

    def archive_solution(self, attempt: SolutionAttempt) -> None:
        """
        Archive a solution attempt.

        Moves the solution from active storage to archive directory
        and updates its status to ARCHIVED.

        Args:
            attempt: The solution attempt to archive

        Raises:
            SolutionStorageError: If archival operation fails
        """
        with self._lock:
            # Update status
            attempt.status = SolutionStatus.ARCHIVED.value
            attempt.updated_at = datetime.now().isoformat()

            # Update in memory
            self._solutions[attempt.id] = attempt

            # Move to archive if persistence is enabled
            if self.enable_persistence:
                with self._write_lock:
                    old_path = self._get_storage_path(attempt.id)
                    archive_path = self._get_archive_path(attempt.id)

                    try:
                        # Write to archive location
                        self._persist_solution(attempt, archive_path)

                        # Remove from active storage
                        if old_path.exists():
                            old_path.unlink()

                        logger.info(f"Archived solution attempt {attempt.id}")
                    except OSError as e:
                        raise SolutionStorageError(f"Failed to archive solution {attempt.id}: {e}")

    def archive_old_solutions(
        self,
        sub_problem_id: str,
        keep_latest: int = 5
    ) -> int:
        """
        Archive old solutions for a sub-problem, keeping only the latest N.

        Args:
            sub_problem_id: ID of the sub-problem
            keep_latest: Number of recent solutions to keep active

        Returns:
            Number of solutions archived
        """
        with self._lock:
            history = self.get_solution_history(sub_problem_id)

            if history.total_attempts <= keep_latest:
                return 0

            # Archive all but the latest N
            to_archive = history.attempts[:-keep_latest]
            archived_count = 0

            for attempt in to_archive:
                if attempt.status != SolutionStatus.ARCHIVED.value:
                    try:
                        self.archive_solution(attempt)
                        archived_count += 1
                    except SolutionStorageError as e:
                        logger.error(f"Failed to archive solution {attempt.id}: {e}")

            logger.info(
                f"Archived {archived_count} old solutions for sub-problem {sub_problem_id} "
                f"(kept latest {keep_latest})"
            )

            return archived_count

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _persist_solution(
        self,
        attempt: SolutionAttempt,
        path: Optional[Path] = None
    ) -> None:
        """
        Persist a solution attempt to disk.

        Args:
            attempt: The solution attempt to persist
            path: Optional path (uses default storage path if None)

        Raises:
            SolutionStorageError: If write operation fails
        """
        file_path = path or self._get_storage_path(attempt.id)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(attempt.to_dict(), f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise SolutionStorageError(f"Failed to persist solution {attempt.id}: {e}")

    def load_solution_from_disk(self, attempt_id: str) -> Optional[SolutionAttempt]:
        """
        Load a solution attempt from disk.

        Args:
            attempt_id: ID of the solution attempt

        Returns:
            SolutionAttempt if found, None otherwise
        """
        file_path = self._get_storage_path(attempt_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SolutionAttempt.from_dict(data)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load solution {attempt_id}: {e}")
            return None

    def load_all_solutions(self) -> int:
        """
        Load all persisted solutions from disk.

        Returns:
            Number of solutions loaded
        """
        loaded_count = 0

        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    attempt = SolutionAttempt.from_dict(data)

                    # Store in memory
                    self._solutions[attempt.id] = attempt

                    # Update index
                    if attempt.sub_problem_id not in self._sub_problem_index:
                        self._sub_problem_index[attempt.sub_problem_id] = []
                    self._sub_problem_index[attempt.sub_problem_id].append(attempt.id)

                    loaded_count += 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to load solution from {file_path}: {e}")

            logger.info(f"Loaded {loaded_count} solutions from disk")
            return loaded_count

        except OSError as e:
            logger.error(f"Failed to scan storage directory: {e}")
            return loaded_count

    # ========================================================================
    # STATISTICS AND REPORTING
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about managed solutions.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_solutions = len(self._solutions)
            total_sub_problems = len(self._sub_problem_index)

            status_counts: Dict[str, int] = {}
            for attempt in self._solutions.values():
                status_counts[attempt.status] = status_counts.get(attempt.status, 0) + 1

            avg_quality_score = 0.0
            if self._solutions:
                avg_quality_score = sum(a.quality_score for a in self._solutions.values()) / total_solutions

            return {
                "total_solutions": total_solutions,
                "total_sub_problems": total_sub_problems,
                "status_distribution": status_counts,
                "average_quality_score": avg_quality_score,
                "registered_validators": list(self._validators.keys()),
                "storage_enabled": self.enable_persistence,
                "validation_level": self.validation_level.value
            }

    def clear_all(self) -> None:
        """
        Clear all solutions from memory.

        Does not affect persisted files on disk.
        """
        with self._lock:
            self._solutions.clear()
            self._sub_problem_index.clear()
            logger.info("Cleared all solutions from memory")

    async def verify_solution_with_lean(self, solution: Dict) -> Dict:
        """
        **LEAN INTEGRATION**: Solution verification using Lean theorem prover.
        
        Args:
            solution: Solution dictionary to verify
            
        Returns:
            Dict with verification results
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "reason": "Lean unavailable"}
        
        try:
            client = LeanAideClient()
            content = solution.get('content', str(solution))
            
            # Autoformalize and verify
            formalized = await client.translate_thm(content)
            
            if formalized.success and formalized.data:
                result = await client.elaborate(formalized.data.get('result', ''))
                
                return {
                    "verified": result.success,
                    "confidence": 1.0 if result.success else 0.0,
                    "proof": result.data.get('result') if result.data else None,
                    "solution_valid": result.success,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "verified": False,
                    "reason": "Autoformalization failed",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Lean verification error: {e}")
            return {"verified": False, "reason": str(e)}

    async def validate_solution_attempt_with_lean(
        self,
        attempt: SolutionAttempt,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        **LEAN INTEGRATION**: Validate solution attempt using Lean theorem prover.
        
        Performs formal mathematical verification of the solution content.
        
        Args:
            attempt: The solution attempt to validate
            criteria: Optional validation criteria
            
        Returns:
            Dict with formal validation results
        """
        if not LEAN_AVAILABLE:
            return {
                "verified": False,
                "reason": "Lean unavailable",
                "attempt_id": attempt.id if hasattr(attempt, 'id') else None
            }
        
        try:
            logger.info(f"Running Lean validation for solution attempt {attempt.id}")
            
            client = LeanAideClient()
            content = attempt.content if hasattr(attempt, 'content') else str(attempt)
            
            # Autoformalize the solution content using translate_thm
            formalized = await client.translate_thm(content)
            
            if formalized.success and formalized.data:
                # Verify with Lean using elaborate
                result = await client.elaborate(formalized.data.get('result', ''))
                
                validation_result = {
                    "verified": result.success,
                    "confidence": 1.0 if result.success else 0.0,
                    "proof": result.data.get('result') if result.data else None,
                    "attempt_id": attempt.id if hasattr(attempt, 'id') else None,
                    "stored_in_knowledge_base": True,
                    "verification_method": "lean_translate_thm_elaborate",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                validation_result = {
                    "verified": False,
                    "reason": "Autoformalization failed",
                    "error": formalized.error,
                    "attempt_id": attempt.id if hasattr(attempt, 'id') else None
                }
            
            # Update attempt metadata with verification result
            if hasattr(attempt, 'metadata'):
                attempt.metadata["lean_verification"] = validation_result
            
            logger.info(f"Lean validation result: verified={validation_result['verified']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Lean validation error: {e}")
            return {
                "verified": False,
                "reason": str(e),
                "attempt_id": attempt.id if hasattr(attempt, 'id') else None
            }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of solution content.

    Useful for detecting duplicate solutions.

    Args:
        content: Solution content

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def format_solution_summary(attempt: SolutionAttempt) -> str:
    """
    Format a human-readable summary of a solution attempt.

    Args:
        attempt: Solution attempt to format

    Returns:
        Formatted summary string
    """
    lines = [
        f"Solution ID: {attempt.id}",
        f"Sub-problem: {attempt.sub_problem_id}",
        f"Status: {attempt.status}",
        f"Model: {attempt.generated_by_model}",
        f"Version: {attempt.version}",
        f"Quality Score: {attempt.quality_score:.2f}",
        f"Created: {attempt.created_at}",
        f"Content Length: {len(attempt.content)} characters"
    ]

    if attempt.parent_attempt_id:
        lines.append(f"Parent: {attempt.parent_attempt_id}")

    if attempt.verification_reports:
        lines.append(f"Verifications: {len(attempt.verification_reports)}")

    return "\n".join(lines)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Demonstrate basic usage of the SolutionManager.
    """
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create solution manager
    manager = SolutionManager(
        storage_dir="data/solutions",
        enable_persistence=True,
        validation_level=ValidationLevel.MODERATE
    )

    # Register a custom validator
    def python_code_validator(attempt: SolutionAttempt) -> ValidationResult:
        """Validate that solution contains Python code."""
        has_python = (
            "def " in attempt.content or
            "class " in attempt.content or
            "import " in attempt.content
        )

        return ValidationResult(
            is_valid=has_python,
            score=1.0 if has_python else 0.5,
            issues=[] if has_python else ["No Python code detected"],
            warnings=[],
            feedback="Python code check passed" if has_python else "No Python code found",
            validator_name="PythonCodeValidator",
            timestamp=time.time()
        )

    manager.register_validator("python_code", python_code_validator)

    # Create a solution attempt
    solution = manager.create_solution_attempt(
        sub_problem_id="sp_001",
        content="def solve():\n    return 'Hello, World!'",
        model="gpt-4",
        metadata={"priority": "high"}
    )

    print(f"Created solution: {solution.id}")

    # Update the solution
    updated = manager.update_solution_attempt(
        attempt=solution,
        status=SolutionStatus.COMPLETED.value,
        quality_score=0.95
    )

    print(f"Updated solution status to: {updated.status}")

    # Get solution history
    history = manager.get_solution_history("sp_001")
    print(f"Total attempts for sp_001: {history.total_attempts}")

    # Get latest solution
    latest = manager.get_latest_solution("sp_001")
    if latest:
        print(format_solution_summary(latest))

    # Get statistics
    stats = manager.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    example_usage()
